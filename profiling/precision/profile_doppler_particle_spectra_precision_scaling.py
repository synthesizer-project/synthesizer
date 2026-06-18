"""Profile Doppler particle spectra scaling across input and output precisions.

This script benchmarks the Doppler particle spectra extension for a synthetic
1D grid workload. For each particle count, all four input/output precision
combinations are timed for each requested grid assignment method:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Usage:
    python profile_doppler_particle_spectra_precision_scaling.py \
        --basename test
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthesizer.extensions.doppler_particle_spectra import (
    compute_part_seds_with_vel_shift,
)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}


def make_synthetic_inputs(nparticles, ngrid, nlam, dtype, rng):
    """Create contiguous synthetic inputs for Doppler particle spectra."""
    axis = np.linspace(0.0, 1.0, ngrid, dtype=np.float64)
    wavelength = np.linspace(900.0, 9000.0, nlam, dtype=np.float64)
    wavelength_phase = np.linspace(-2.0, 2.0, nlam, dtype=np.float64)

    grid_amplitudes = rng.uniform(0.5, 2.0, size=(ngrid, 1))
    grid_centres = rng.uniform(-0.8, 0.8, size=(ngrid, 1))
    grid_widths = rng.uniform(0.2, 0.8, size=(ngrid, 1))
    grid_spectra = grid_amplitudes * np.exp(
        -0.5 * ((wavelength_phase[None, :] - grid_centres) / grid_widths) ** 2
    ) + 0.1 * (1.0 + wavelength_phase[None, :])

    part_prop = rng.uniform(axis[0], axis[-1], size=nparticles)
    weights = rng.uniform(1.0, 20.0, size=nparticles)
    velocities = rng.normal(0.0, 150.0e3, size=nparticles)
    lam_mask = np.ones(nlam, dtype=np.bool_)

    return {
        "grid_spectra": np.array(
            grid_spectra, dtype=dtype, order="C", copy=True
        ),
        "wavelength": np.array(wavelength, dtype=dtype, order="C", copy=True),
        "axes": (np.array(axis, dtype=dtype, order="C", copy=True),),
        "part_props": (
            np.array(part_prop, dtype=dtype, order="C", copy=True),
        ),
        "weights": np.array(weights, dtype=dtype, order="C", copy=True),
        "velocities": np.array(velocities, dtype=dtype, order="C", copy=True),
        "grid_dims": np.array([ngrid], dtype=np.int32),
        "lam_mask": lam_mask,
    }


def benchmark_doppler_particle_spectra(
    method,
    inputs,
    out_dtype,
    nthreads,
    repeats,
):
    """Time repeated Doppler particle spectra calls for prepared inputs."""
    compute_part_seds_with_vel_shift(
        inputs["grid_spectra"],
        inputs["wavelength"],
        inputs["axes"],
        inputs["part_props"],
        inputs["weights"],
        inputs["velocities"],
        inputs["grid_dims"],
        1,
        inputs["weights"].size,
        inputs["grid_spectra"].shape[-1],
        method,
        nthreads,
        299792458.0,
        None,
        inputs["lam_mask"],
        out_dtype,
        ("x",),
    )

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        compute_part_seds_with_vel_shift(
            inputs["grid_spectra"],
            inputs["wavelength"],
            inputs["axes"],
            inputs["part_props"],
            inputs["weights"],
            inputs["velocities"],
            inputs["grid_dims"],
            1,
            inputs["weights"].size,
            inputs["grid_spectra"].shape[-1],
            method,
            nthreads,
            299792458.0,
            None,
            inputs["lam_mask"],
            out_dtype,
            ("x",),
        )
        times.append(time.perf_counter() - start)

    times = np.array(times, dtype=np.float64)
    return {
        "mean_seconds": float(np.mean(times)),
        "min_seconds": float(np.min(times)),
        "std_seconds": float(np.std(times)),
    }


def write_results_csv(results, output_path):
    """Write benchmark results to a CSV file."""
    fieldnames = [
        "method",
        "nparticles",
        "ngrid",
        "nlam",
        "input_dtype",
        "output_dtype",
        "mean_seconds",
        "min_seconds",
        "std_seconds",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results, output_path):
    """Plot runtime against particle count for each precision pair."""
    methods = sorted({row["method"] for row in results})
    figure, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5))
    axes = np.atleast_1d(axes)

    for axis, method in zip(axes, methods, strict=True):
        for input_name in PRECISIONS:
            for output_name in PRECISIONS:
                rows = [
                    row
                    for row in results
                    if row["method"] == method
                    and row["input_dtype"] == input_name
                    and row["output_dtype"] == output_name
                ]
                rows.sort(key=lambda row: row["nparticles"])
                axis.plot(
                    [row["nparticles"] for row in rows],
                    [row["mean_seconds"] for row in rows],
                    marker="o",
                    label=f"{input_name} -> {output_name}",
                )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Number of particles")
        axis.set_title(method)
        axis.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Mean runtime (seconds)")
    axes[-1].legend(loc="best", fontsize=9)
    figure.suptitle("Doppler Particle Spectra Precision Scaling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def profile_doppler_particle_spectra_precision_scaling(
    basename,
    out_dir,
    nparticles,
    ngrid,
    nlam,
    methods,
    repeats,
    nthreads,
    seed,
):
    """Run the Doppler particle spectra precision scaling benchmarks."""
    rng = np.random.default_rng(seed)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        output_dir
        / f"{basename}_doppler_particle_spectra_precision_scaling.csv"
    )
    plot_path = (
        output_dir
        / f"{basename}_doppler_particle_spectra_precision_scaling.png"
    )

    results = []
    for particle_count in nparticles:
        print(
            "Profiling Doppler particle spectra precision for "
            f"nparticles={particle_count}"
        )

        inputs_by_dtype = {}
        for input_name, input_dtype in PRECISIONS.items():
            inputs_by_dtype[input_name] = make_synthetic_inputs(
                particle_count,
                ngrid,
                nlam,
                input_dtype,
                rng,
            )

        for method in methods:
            for input_name in PRECISIONS:
                for output_name, output_dtype in PRECISIONS.items():
                    benchmark_result = benchmark_doppler_particle_spectra(
                        method,
                        inputs_by_dtype[input_name],
                        output_dtype,
                        nthreads,
                        repeats,
                    )
                    results.append(
                        {
                            "method": method,
                            "nparticles": int(particle_count),
                            "ngrid": int(ngrid),
                            "nlam": int(nlam),
                            "input_dtype": input_name,
                            "output_dtype": output_name,
                            **benchmark_result,
                        }
                    )

                    print(
                        "  "
                        f"{method} {input_name} -> {output_name}: "
                        f"{benchmark_result['mean_seconds']:.6f}s"
                    )

    write_results_csv(results, csv_path)
    plot_results(results, plot_path)
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--basename",
        type=str,
        default="test",
        help="The basename of the output files.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="profiling/outputs",
        help="The output directory for the CSV file and plot.",
    )
    parser.add_argument(
        "--nparticles",
        type=int,
        nargs="+",
        default=[10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5],
        help="Particle counts to profile.",
    )
    parser.add_argument(
        "--ngrid",
        type=int,
        default=256,
        help="Number of grid cells along the synthetic property axis.",
    )
    parser.add_argument(
        "--nlam",
        type=int,
        default=2048,
        help="Number of wavelength bins in the synthetic grid spectra.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["cic", "ngp"],
        choices=["cic", "ngp"],
        help="Grid assignment methods to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="The number of timed repeats per precision combination.",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="The number of threads to use for each spectra call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for synthetic input generation.",
    )
    args = parser.parse_args()

    profile_doppler_particle_spectra_precision_scaling(
        basename=args.basename,
        out_dir=args.out_dir,
        nparticles=args.nparticles,
        ngrid=args.ngrid,
        nlam=args.nlam,
        methods=args.methods,
        repeats=args.repeats,
        nthreads=args.nthreads,
        seed=args.seed,
    )
