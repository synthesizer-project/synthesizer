"""Benchmark old and new line scaling implementations.

This script creates artificial line collections with shape ``(nobj, nlines)``
and measures how the runtime of the old NumPy-based scaling path compares to
the current optimised ``LineCollection.scale`` implementation.

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

from synthesizer.emissions.line import LineCollection

SCENARIOS = {
    "row_broadcast": "Per-row scaling with no masks.",
    "row_mask": "Per-row scaling with a 1D row mask.",
    "lam_mask": "Per-row scaling with a 1D wavelength mask.",
    "row_and_lam_mask": "Per-row scaling with row and wavelength masks.",
}


def parse_counts(value):
    """Parse a comma-separated object count list."""
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_scenarios(value):
    """Parse a comma-separated scenario list."""
    scenarios = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(scenarios) - set(SCENARIOS))
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return scenarios


def old_scale_lines(lum, cont, scaling, mask=None, lam_mask=None):
    """Reproduce the pre-optimisation NumPy scaling path for lines."""
    scaling_lum = scaling
    scaling_cont = scaling

    lum_work = lum.copy()
    cont_work = cont.copy()

    if (
        mask is not None
        and lam_mask is not None
        and mask.shape[-1] == lam_mask.shape[0]
    ):
        mask = np.logical_and(mask, lam_mask)
    elif mask is not None and lam_mask is not None:
        mask = np.logical_and(mask[:, None], lam_mask)
    elif lam_mask is not None and mask is None:
        mask = lam_mask
    elif mask is None and lam_mask is None:
        mask = None

    if mask is not None:
        if mask.shape == lum_work.shape:
            pass
        elif mask.ndim == 1 and mask.shape[0] == lum_work.shape[0]:
            pass
        elif mask.ndim == 1 and mask.shape[0] == lum_work.shape[-1]:
            mask = np.broadcast_to(mask[np.newaxis, :], lum_work.shape)
        else:
            raise ValueError(f"Incompatible mask shape: {mask.shape}")

    if np.isscalar(scaling_lum):
        if mask is None:
            lum_work *= scaling_lum
        else:
            lum_work[mask] *= scaling_lum
    elif scaling_lum.size == 1:
        scale = scaling_lum.item()
        if mask is None:
            lum_work *= scale
        else:
            lum_work[mask] *= scale
    elif scaling_lum.ndim == 1 and scaling_lum.size == lum_work.shape[-1]:
        if mask is None:
            lum_work *= scaling_lum
        else:
            lum_work[mask] *= scaling_lum[mask]
    elif isinstance(scaling_lum, np.ndarray) and len(scaling_lum.shape) < len(
        lum_work.shape
    ):
        expand_axes = tuple(range(len(scaling_lum.shape), len(lum_work.shape)))
        new_scaling_lum = np.ones(lum_work.shape) * np.expand_dims(
            scaling_lum,
            axis=expand_axes,
        )
        if mask is None:
            lum_work *= new_scaling_lum
        else:
            lum_work[mask] *= new_scaling_lum[mask]
    elif (
        isinstance(scaling_lum, np.ndarray)
        and scaling_lum.shape == lum_work.shape
    ):
        if mask is None:
            lum_work *= scaling_lum
        else:
            lum_work[mask] *= scaling_lum[mask]
    else:
        raise ValueError("Unsupported luminosity scaling")

    if np.isscalar(scaling_cont):
        if mask is None:
            cont_work *= scaling_cont
        else:
            cont_work[mask] *= scaling_cont
    elif scaling_cont.size == 1:
        scale = scaling_cont.item()
        if mask is None:
            cont_work *= scale
        else:
            cont_work[mask] *= scale
    elif scaling_cont.ndim == 1 and scaling_cont.size == cont_work.shape[-1]:
        if mask is None:
            cont_work *= scaling_cont
        else:
            cont_work[mask] *= scaling_cont[mask]
    elif isinstance(scaling_cont, np.ndarray) and len(
        scaling_cont.shape
    ) < len(cont_work.shape):
        expand_axes = tuple(
            range(len(scaling_cont.shape), len(cont_work.shape))
        )
        new_scaling_cont = np.ones(cont_work.shape) * np.expand_dims(
            scaling_cont,
            axis=expand_axes,
        )
        if mask is None:
            cont_work *= new_scaling_cont
        else:
            cont_work[mask] *= new_scaling_cont[mask]
    elif (
        isinstance(scaling_cont, np.ndarray)
        and scaling_cont.shape == cont_work.shape
    ):
        if mask is None:
            cont_work *= scaling_cont
        else:
            cont_work[mask] *= scaling_cont[mask]
    else:
        raise ValueError("Unsupported continuum scaling")

    return lum_work, cont_work


def build_inputs(rng, nobj, nlines, scenario):
    """Construct benchmark inputs for one line scaling scenario."""
    lum = rng.random((nobj, nlines))
    cont = rng.random((nobj, nlines))
    scaling = rng.random(nobj)
    mask = None
    lam_mask = None

    if scenario == "row_mask":
        mask = rng.random(nobj) > 0.5
    elif scenario == "lam_mask":
        lam_mask = rng.random(nlines) > 0.5
    elif scenario == "row_and_lam_mask":
        mask = rng.random(nobj) > 0.5
        lam_mask = rng.random(nlines) > 0.5

    return {
        "lum": lum,
        "cont": cont,
        "scaling": scaling,
        "mask": mask,
        "lam_mask": lam_mask,
    }


def make_lines(line_ids, lam, benchmark_input):
    """Construct a line collection for benchmarking."""
    return LineCollection(
        line_ids=line_ids,
        lam=lam,
        lum=benchmark_input["lum"] * erg / s,
        cont=benchmark_input["cont"] * erg / s / Hz,
    )


def verify_outputs_match(line_ids, lam, benchmark_input, nthreads):
    """Verify the old and new scaling paths produce matching results."""
    expected_lum, expected_cont = old_scale_lines(
        benchmark_input["lum"],
        benchmark_input["cont"],
        benchmark_input["scaling"],
        mask=benchmark_input["mask"],
        lam_mask=benchmark_input["lam_mask"],
    )
    lines = make_lines(line_ids, lam, benchmark_input)
    new = lines.scale(
        benchmark_input["scaling"],
        mask=benchmark_input["mask"],
        lam_mask=benchmark_input["lam_mask"],
        nthreads=nthreads,
    )
    np.testing.assert_allclose(new.luminosity.value, expected_lum)
    np.testing.assert_allclose(new.continuum.value, expected_cont)


def summarise_times(times):
    """Summarise repeated timing measurements."""
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "samples": times,
    }


def benchmark_old_path(benchmark_input, repeats):
    """Benchmark the old NumPy line scaling path."""
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        old_scale_lines(
            benchmark_input["lum"],
            benchmark_input["cont"],
            benchmark_input["scaling"],
            mask=benchmark_input["mask"],
            lam_mask=benchmark_input["lam_mask"],
        )
        times.append(time.perf_counter() - start)

    return summarise_times(times)


def benchmark_new_path(line_ids, lam, benchmark_input, nthreads, repeats):
    """Benchmark the current LineCollection.scale implementation."""
    times = []

    make_lines(line_ids, lam, benchmark_input).scale(
        benchmark_input["scaling"],
        mask=benchmark_input["mask"],
        lam_mask=benchmark_input["lam_mask"],
        nthreads=nthreads,
    )

    for _ in range(repeats):
        lines = make_lines(line_ids, lam, benchmark_input)
        start = time.perf_counter()
        lines.scale(
            benchmark_input["scaling"],
            mask=benchmark_input["mask"],
            lam_mask=benchmark_input["lam_mask"],
            nthreads=nthreads,
        )
        times.append(time.perf_counter() - start)

    return summarise_times(times)


def run_benchmark(counts, scenarios, nlines, nthreads, repeats, seed):
    """Run the line scaling benchmark over object count for each scenario."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(1000.0, 1000.0 + nlines - 1, nlines) * angstrom
    line_ids = [f"line_{index}" for index in range(nlines)]

    results = {
        "nlines": nlines,
        "nthreads": nthreads,
        "repeats": repeats,
        "seed": seed,
        "counts": counts,
        "scenarios": {},
    }

    for scenario in scenarios:
        print(f"Scenario: {scenario} ({SCENARIOS[scenario]})")
        scenario_results = {"old": [], "new": []}

        for nobj in counts:
            print(f"  Benchmarking nobj={nobj}...")
            benchmark_input = build_inputs(rng, nobj, nlines, scenario)
            verify_outputs_match(line_ids, lam, benchmark_input, nthreads)

            old_stats = benchmark_old_path(benchmark_input, repeats)
            new_stats = benchmark_new_path(
                line_ids,
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
    """Plot runtime against object count for each scaling scenario."""
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
        ax.plot(
            counts, new_means, marker="o", label="current LineCollection.scale"
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Object count")
        ax.set_ylabel("Runtime [s]")
        ax.set_title(
            f"{scenario}\n"
            f"nlines={results['nlines']}, nthreads={results['nthreads']}, "
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
        help="Comma-separated object counts to benchmark.",
    )
    parser.add_argument(
        "--scenarios",
        default="row_broadcast,row_mask,lam_mask,row_and_lam_mask",
        help="Comma-separated scaling scenarios to benchmark.",
    )
    parser.add_argument("--nlines", type=int, default=2048)
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
        nlines=args.nlines,
        nthreads=args.nthreads,
        repeats=args.repeats,
        seed=args.seed,
    )

    scenario_stem = "-".join(scenarios)
    stem = (
        f"line_scale_old_vs_new_{scenario_stem}_"
        f"nlines{args.nlines}_threads{args.nthreads}"
    )
    json_path = args.output_dir / f"{stem}.json"
    plot_path = args.output_dir / f"{stem}.png"

    json_path.write_text(json.dumps(results, indent=2))
    make_plot(results, plot_path)

    print(f"Wrote results to {json_path}")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
