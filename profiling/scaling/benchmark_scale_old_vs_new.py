"""Benchmark old and new scaling implementations.

This script creates artificial particle SEDs with shape ``(npart, nlam)`` and
measures how the runtime of the old NumPy-based scaling path compares to the
current optimised ``Sed.scale`` implementation.

It writes a JSON file containing the raw timing measurements and a PNG plot
showing one scaling curve for the old path and one for the new path for each
requested scenario.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unyt import Hz, angstrom, erg, s

from synthesizer.emissions import Sed

SCENARIOS = {
    "row_broadcast": "Per-spectrum scaling with no masks.",
    "row_mask": "Per-spectrum scaling with a 1D row mask.",
    "lam_mask": "Per-spectrum scaling with a 1D wavelength mask.",
    "row_and_lam_mask": "Per-spectrum scaling with row and wavelength masks.",
}


def parse_counts(value):
    """Parse a comma-separated particle count list."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_scenarios(value):
    """Parse a comma-separated scenario list."""
    scenarios = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return scenarios


def old_scale_sed(lnu, scaling, mask=None, lam_mask=None):
    """Reproduce the pre-optimisation NumPy scaling path for Sed."""
    work = lnu.copy()

    if lam_mask is not None:
        work = work[..., lam_mask]

    if np.isscalar(scaling):
        if mask is not None:
            work[mask] *= scaling
        else:
            work *= scaling

    elif isinstance(scaling, np.ndarray) and len(scaling.shape) < len(
        work.shape
    ):
        expand_axes = tuple(range(len(scaling.shape), len(work.shape)))
        new_scaling = np.ones(work.shape) * np.expand_dims(
            scaling,
            axis=expand_axes,
        )

        if mask is not None:
            work[mask] *= new_scaling[mask]
        else:
            work *= new_scaling

    elif isinstance(scaling, np.ndarray) and scaling.shape == work.shape:
        if mask is not None:
            work[mask] *= scaling[mask]
        else:
            work *= scaling

    elif isinstance(scaling, np.ndarray):
        work = scaling[..., np.newaxis] * work

        if mask is not None:
            raise ValueError(
                "Masking is not supported for scaling arrays with "
                "different shapes"
            )

    else:
        raise TypeError(f"Unsupported scaling type: {type(scaling)}")

    if lam_mask is not None:
        new_lnu = lnu.copy()
        new_lnu[..., lam_mask] = work
        return new_lnu

    return work


def build_inputs(rng, npart, nlam, scenario):
    """Construct benchmark inputs for one scaling scenario."""
    lnu = rng.random((npart, nlam))
    scaling = rng.random(npart)
    mask = None
    lam_mask = None

    if scenario == "row_mask":
        mask = rng.random(npart) > 0.5
    elif scenario == "lam_mask":
        lam_mask = rng.random(nlam) > 0.5
    elif scenario == "row_and_lam_mask":
        mask = rng.random(npart) > 0.5
        lam_mask = rng.random(nlam) > 0.5

    return {
        "lnu": lnu,
        "scaling": scaling,
        "mask": mask,
        "lam_mask": lam_mask,
    }


def verify_outputs_match(lam, benchmark_input, nthreads):
    """Verify the old and new scaling paths produce matching results."""
    expected = old_scale_sed(
        benchmark_input["lnu"],
        benchmark_input["scaling"],
        mask=benchmark_input["mask"],
        lam_mask=benchmark_input["lam_mask"],
    )
    new = Sed(lam=lam, lnu=benchmark_input["lnu"] * erg / s / Hz).scale(
        benchmark_input["scaling"],
        mask=benchmark_input["mask"],
        lam_mask=benchmark_input["lam_mask"],
        nthreads=nthreads,
    )
    np.testing.assert_allclose(new.lnu.value, expected)


def benchmark_old_path(benchmark_input, repeats):
    """Benchmark the old NumPy scaling path."""
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        old_scale_sed(
            benchmark_input["lnu"],
            benchmark_input["scaling"],
            mask=benchmark_input["mask"],
            lam_mask=benchmark_input["lam_mask"],
        )
        times.append(time.perf_counter() - start)

    return summarise_times(times)


def benchmark_new_path(lam, benchmark_input, nthreads, repeats):
    """Benchmark the current Sed.scale implementation."""
    times = []

    # Warm up the Python and extension paths before recording timings.
    Sed(lam=lam, lnu=benchmark_input["lnu"] * erg / s / Hz).scale(
        benchmark_input["scaling"],
        mask=benchmark_input["mask"],
        lam_mask=benchmark_input["lam_mask"],
        nthreads=nthreads,
    )

    for _ in range(repeats):
        sed = Sed(lam=lam, lnu=benchmark_input["lnu"] * erg / s / Hz)

        start = time.perf_counter()
        sed.scale(
            benchmark_input["scaling"],
            mask=benchmark_input["mask"],
            lam_mask=benchmark_input["lam_mask"],
            nthreads=nthreads,
        )
        times.append(time.perf_counter() - start)

    return summarise_times(times)


def summarise_times(times):
    """Summarise repeated timing measurements."""
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "samples": times,
    }


def run_benchmark(counts, scenarios, nlam, nthreads, repeats, seed):
    """Run the scaling benchmark over particle count for each scenario."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(900.0, 9000.0, nlam) * angstrom

    results = {
        "nlam": nlam,
        "nthreads": nthreads,
        "repeats": repeats,
        "seed": seed,
        "counts": counts,
        "scenarios": {},
    }

    for scenario in scenarios:
        print(f"Scenario: {scenario} ({SCENARIOS[scenario]})")
        scenario_results = {"old": [], "new": []}

        for npart in counts:
            print(f"  Benchmarking npart={npart}...")
            benchmark_input = build_inputs(rng, npart, nlam, scenario)
            verify_outputs_match(lam, benchmark_input, nthreads)

            old_stats = benchmark_old_path(benchmark_input, repeats)
            new_stats = benchmark_new_path(
                lam,
                benchmark_input,
                nthreads,
                repeats,
            )

            scenario_results["old"].append(old_stats)
            scenario_results["new"].append(new_stats)

            print(
                "    "
                f"old={old_stats['mean']:.6f}s, "
                f"new={new_stats['mean']:.6f}s, "
                f"speedup={old_stats['mean'] / new_stats['mean']:.2f}x"
            )

        results["scenarios"][scenario] = scenario_results

    return results


def make_plot(results, output_path):
    """Plot runtime against particle count for each scaling scenario."""
    counts = np.asarray(results["counts"])
    scenario_names = list(results["scenarios"])

    fig, axes = plt.subplots(
        len(scenario_names),
        1,
        figsize=(8, 4 * len(scenario_names)),
        squeeze=False,
    )

    for ax, scenario in zip(axes[:, 0], scenario_names):
        old_means = [
            entry["mean"] for entry in results["scenarios"][scenario]["old"]
        ]
        new_means = [
            entry["mean"] for entry in results["scenarios"][scenario]["new"]
        ]

        ax.plot(counts, old_means, marker="o", label="old NumPy path")
        ax.plot(counts, new_means, marker="o", label="current Sed.scale")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Particle count")
        ax.set_ylabel("Runtime [s]")
        ax.set_title(
            f"{scenario}\n"
            f"nlam={results['nlam']}, nthreads={results['nthreads']}, "
            f"repeats={results['repeats']}"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    """Run the benchmark and write the outputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--counts",
        default="1000,3000,10000,30000,100000",
        help="Comma-separated particle counts to benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        default="row_broadcast,row_mask,lam_mask,row_and_lam_mask",
        help="Comma-separated scaling scenarios to benchmark.",
    )
    parser.add_argument("--nlam", type=int, default=2048)
    parser.add_argument("--nthreads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("profiling/plots"),
    )
    args = parser.parse_args()

    counts = parse_counts(args.counts)
    scenarios = parse_scenarios(args.scenarios)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = run_benchmark(
        counts=counts,
        scenarios=scenarios,
        nlam=args.nlam,
        nthreads=args.nthreads,
        repeats=args.repeats,
        seed=args.seed,
    )

    scenario_stem = "-".join(scenarios)
    stem = (
        f"scale_old_vs_new_{scenario_stem}"
        f"_nlam{args.nlam}_threads{args.nthreads}"
    )
    json_path = args.output_dir / f"{stem}.json"
    plot_path = args.output_dir / f"{stem}.png"

    json_path.write_text(json.dumps(results, indent=2))
    make_plot(results, plot_path)

    print(f"Wrote results to {json_path}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
