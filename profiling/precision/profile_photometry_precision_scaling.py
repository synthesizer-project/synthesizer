"""Profile photometry scaling across input and output precisions.

This script benchmarks the batched photometry path for one consistent workload:
a 2D spectra array with shape ``(nparticles, nlam)``. For each particle count,
all four input/output precision combinations are timed:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Usage:
    python profile_photometry_precision_scaling.py --basename test
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _helpers import make_synthetic_spectra, make_test_filters
from unyt import c

from synthesizer.grid import Grid

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}


def benchmark_photometry(filters, spectra, nu, out_dtype, nthreads, repeats):
    """Time repeated photometry calls for a prepared spectra array."""
    # Prime the extension and filter caches before timing.
    filters.apply_filters(
        spectra,
        nu=nu,
        nthreads=nthreads,
        out_dtype=out_dtype,
    )

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        filters.apply_filters(
            spectra,
            nu=nu,
            nthreads=nthreads,
            out_dtype=out_dtype,
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
        "benchmark",
        "nparticles",
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
    figure, axis = plt.subplots(1, 1, figsize=(8, 6))

    for input_name in PRECISIONS:
        for output_name in PRECISIONS:
            rows = [
                row
                for row in results
                if row["input_dtype"] == input_name
                and row["output_dtype"] == output_name
            ]
            rows.sort(key=lambda row: row["nparticles"])
            xvals = [row["nparticles"] for row in rows]
            yvals = [row["mean_seconds"] for row in rows]
            axis.plot(
                xvals,
                yvals,
                marker="o",
                label=f"{input_name} -> {output_name}",
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Number of particles")
    axis.set_ylabel("Mean runtime (seconds)")
    axis.set_title("Photometry Precision Scaling")
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(loc="best", fontsize=9)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def profile_photometry_precision_scaling(
    basename,
    out_dir,
    nparticles,
    nfilters,
    repeats,
    nthreads,
    seed,
):
    """Run the photometry precision scaling benchmarks."""
    rng = np.random.default_rng(seed)

    grid = Grid("test_grid")
    filters = make_test_filters(grid.lam, nfilters)
    nlam = grid.nlam

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_photometry_precision_scaling.csv"
    plot_path = output_dir / f"{basename}_photometry_precision_scaling.png"

    results = []
    for particle_count in nparticles:
        print(
            f"Profiling photometry precision for nparticles={particle_count}"
        )

        spectra_by_dtype = {}
        nu_by_dtype = {}

        base_spectra = make_synthetic_spectra(
            particle_count,
            nlam,
            np.float64,
            rng,
        )
        base_nu = np.array(
            (c / grid.lam).to("Hz").value,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        for input_name, input_dtype in PRECISIONS.items():
            spectra_by_dtype[input_name] = np.array(
                base_spectra, dtype=input_dtype, order="C", copy=True
            )
            nu_by_dtype[input_name] = np.array(
                base_nu, dtype=input_dtype, order="C", copy=True
            )

        for input_name, _input_dtype in PRECISIONS.items():
            for output_name, output_dtype in PRECISIONS.items():
                benchmark_result = benchmark_photometry(
                    filters,
                    spectra_by_dtype[input_name],
                    nu_by_dtype[input_name],
                    output_dtype,
                    nthreads,
                    repeats,
                )
                results.append(
                    {
                        "benchmark": "synthetic_2d",
                        "nparticles": int(particle_count),
                        "input_dtype": input_name,
                        "output_dtype": output_name,
                        **benchmark_result,
                    }
                )

                print(
                    "  "
                    f"{input_name} -> {output_name}: "
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
        "--nfilters",
        type=int,
        default=10,
        help="The number of filters to use for photometry.",
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
        help="The number of threads to use for each photometry call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for synthetic spectra generation.",
    )
    args = parser.parse_args()

    profile_photometry_precision_scaling(
        basename=args.basename,
        out_dir=args.out_dir,
        nparticles=args.nparticles,
        nfilters=args.nfilters,
        repeats=args.repeats,
        nthreads=args.nthreads,
        seed=args.seed,
    )
