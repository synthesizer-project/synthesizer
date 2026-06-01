"""Profile integration scaling across input and output precisions.

This script benchmarks the last-axis integration extension for one consistent
workload: a 2D input array with shape ``(nentries, nlam)``. For each entry
count, all four input/output precision combinations are timed for both
integration methods:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Usage:
    python profile_integration_precision_scaling.py --basename test
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synthesizer.extensions.integration import simps_last_axis, trapz_last_axis

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}

METHODS = {
    "trapz": trapz_last_axis,
    "simps": simps_last_axis,
}


def make_integration_inputs(nentries, nlam, dtype, rng):
    """Create contiguous synthetic inputs for last-axis integration."""
    xs = np.linspace(0.0, 10.0, nlam, dtype=np.float64)

    phases = rng.uniform(0.0, np.pi, size=(nentries, 1))
    amplitudes = rng.uniform(0.5, 2.0, size=(nentries, 1))
    frequencies = rng.uniform(0.5, 2.5, size=(nentries, 1))
    continuum = np.linspace(0.8, 1.2, nlam, dtype=np.float64)[None, :]

    ys = (
        amplitudes * np.sin(frequencies * xs[None, :] + phases)
        + 0.1 * continuum
        + 0.05 * np.cos(0.5 * xs[None, :])
    )

    return (
        np.array(xs, dtype=dtype, order="C", copy=True),
        np.array(ys, dtype=dtype, order="C", copy=True),
    )


def benchmark_integration(method, xs, ys, out_dtype, nthreads, repeats):
    """Time repeated integration calls for prepared inputs."""
    integration_function = METHODS[method]

    # Prime the extension before timing.
    integration_function(xs, ys, nthreads, out_dtype)

    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        integration_function(xs, ys, nthreads, out_dtype)
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
        "nentries",
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
    """Plot integration runtime against entry count for each precision pair."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    for axis, method in zip(axes, METHODS, strict=True):
        for input_name in PRECISIONS:
            for output_name in PRECISIONS:
                rows = [
                    row
                    for row in results
                    if row["method"] == method
                    and row["input_dtype"] == input_name
                    and row["output_dtype"] == output_name
                ]
                rows.sort(key=lambda row: row["nentries"])
                xvals = [row["nentries"] for row in rows]
                yvals = [row["mean_seconds"] for row in rows]
                axis.plot(
                    xvals,
                    yvals,
                    marker="o",
                    label=f"{input_name} -> {output_name}",
                )

        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Number of entries")
        axis.set_title(method)
        axis.grid(True, which="both", alpha=0.3)

    axes[0].set_ylabel("Mean runtime (seconds)")
    axes[1].legend(loc="best", fontsize=9)
    figure.suptitle("Integration Precision Scaling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def profile_integration_precision_scaling(
    basename,
    out_dir,
    nentries,
    nlam,
    repeats,
    nthreads,
    seed,
):
    """Run the integration precision scaling benchmarks."""
    rng = np.random.default_rng(seed)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_integration_precision_scaling.csv"
    plot_path = output_dir / f"{basename}_integration_precision_scaling.png"

    results = []
    for entry_count in nentries:
        print(f"Profiling integration precision for nentries={entry_count}")

        inputs_by_dtype = {}
        for input_name, input_dtype in PRECISIONS.items():
            inputs_by_dtype[input_name] = make_integration_inputs(
                entry_count,
                nlam,
                input_dtype,
                rng,
            )

        for method in METHODS:
            for input_name in PRECISIONS:
                xs, ys = inputs_by_dtype[input_name]
                for output_name, output_dtype in PRECISIONS.items():
                    benchmark_result = benchmark_integration(
                        method,
                        xs,
                        ys,
                        output_dtype,
                        nthreads,
                        repeats,
                    )
                    results.append(
                        {
                            "method": method,
                            "nentries": int(entry_count),
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
        "--nentries",
        type=int,
        nargs="+",
        default=[10**3, 3 * 10**3, 10**4, 3 * 10**4, 10**5],
        help="Entry counts to profile.",
    )
    parser.add_argument(
        "--nlam",
        type=int,
        default=2048,
        help="The number of samples along the integrated axis.",
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
        help="The number of threads to use for each integration call.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for synthetic input generation.",
    )
    args = parser.parse_args()

    profile_integration_precision_scaling(
        basename=args.basename,
        out_dir=args.out_dir,
        nentries=args.nentries,
        nlam=args.nlam,
        repeats=args.repeats,
        nthreads=args.nthreads,
        seed=args.seed,
    )
