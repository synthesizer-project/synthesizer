"""Profile integration memory across input and output precisions.

This script benchmarks the integration extension for a 1D workload.
For each array size, all four input/output precision combinations are profiled:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Memory (RSS) is sampled continuously at a configurable frequency by a
background thread while the extension runs.  The x-axis is normalised to
% progress so runtime does not affect the plot shape.

Usage:
    python profile_integration_memory_scaling.py --basename test
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil

from synthesizer.extensions.integration import (
    simps_last_axis,
    trapz_last_axis,
)

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


def make_synthetic_inputs(nentries, nlam, dtype, rng):
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

    return {
        "xs": np.array(xs, dtype=dtype, order="C", copy=True),
        "ys": np.array(ys, dtype=dtype, order="C", copy=True),
    }


def _sample_integration(
    method,
    inputs,
    out_dtype,
    nthreads,
    repeats,
    sample_freq_hz,
):
    """Run integration repeats while continuously sampling RSS.

    RSS is sampled on a background daemon thread at ``sample_freq_hz``
    using a busy-wait loop so the requested frequency is respected even
    for sub-ms calls.

    Returns (memory_trace, peak_mib).  Always includes at least a start
    and end sample so even fast operations produce a visible trace.
    """
    func = METHODS[method]
    rss_start = psutil.Process().memory_info().rss / 1e6
    samples = []
    stop_sampling = False
    interval = 1.0 / sample_freq_hz

    def sampler():
        next_sample = time.perf_counter()
        while not stop_sampling:
            now = time.perf_counter()
            if now >= next_sample:
                rss_mb = psutil.Process().memory_info().rss / 1e6
                samples.append(rss_mb)
                next_sample += interval

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    for _ in range(repeats):
        func(inputs["xs"], inputs["ys"], nthreads, out_dtype)

    stop_sampling = True
    sampler_thread.join()

    rss_end = psutil.Process().memory_info().rss / 1e6

    all_samples = [rss_start] + samples + [rss_end]
    n = len(all_samples)
    memory_trace = [
        (i / (n - 1) * 100.0, s) for i, s in enumerate(all_samples)
    ]
    return {"peak_mib": max(all_samples), "memory_trace": memory_trace}


def write_results_csv(results, output_path):
    """Write per-sample benchmark results to a CSV file."""
    fieldnames = [
        "method",
        "nentries",
        "nlam",
        "input_dtype",
        "output_dtype",
        "pct_complete",
        "rss_mib",
        "peak_mib",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results, output_path):
    """Plot RSS memory against % progress for each precision pair."""
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
                rows.sort(key=lambda row: row["pct_complete"])
                if not rows:
                    continue
                axis.plot(
                    [row["pct_complete"] for row in rows],
                    [row["rss_mib"] for row in rows],
                    linewidth=1,
                    label=f"{input_name} -> {output_name}",
                )

        axis.set_xlabel("Progress through benchmark (%)")
        axis.set_title(method)
        axis.grid(True, alpha=0.3)

    axes[0].set_ylabel("RSS memory (MiB)")
    axes[-1].legend(loc="best", fontsize=9)
    figure.suptitle("Integration Memory Scaling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def profile_integration_memory_scaling(
    basename,
    out_dir,
    nentries,
    nlam,
    methods,
    repeats,
    nthreads,
    sample_freq,
    seed,
):
    """Run integration memory scaling benchmarks."""
    rng = np.random.default_rng(seed)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_integration_memory_scaling.csv"
    plot_path = output_dir / f"{basename}_integration_memory_scaling.png"

    results = []
    for entry_count in nentries:
        print(f"Profiling integration memory for nentries={entry_count}")

        inputs_by_dtype = {}
        for input_name, input_dtype in PRECISIONS.items():
            inputs_by_dtype[input_name] = make_synthetic_inputs(
                entry_count, nlam, input_dtype, rng
            )

        for method in methods:
            for input_name in PRECISIONS:
                for output_name, output_dtype in PRECISIONS.items():
                    result = _sample_integration(
                        method,
                        inputs_by_dtype[input_name],
                        output_dtype,
                        nthreads,
                        repeats,
                        sample_freq,
                    )

                    peak = round(result["peak_mib"], 3)
                    for pct, rss_mib in result["memory_trace"]:
                        results.append(
                            {
                                "method": method,
                                "nentries": int(entry_count),
                                "nlam": int(nlam),
                                "input_dtype": input_name,
                                "output_dtype": output_name,
                                "pct_complete": round(pct, 2),
                                "rss_mib": round(rss_mib, 3),
                                "peak_mib": peak,
                            }
                        )

                    print(
                        f"  {method} {input_name} -> {output_name}: "
                        f"peak={peak:.3f}MiB, "
                        f"samples={len(result['memory_trace'])}"
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
        help="Number of integration entries to profile.",
    )
    parser.add_argument(
        "--nlam",
        type=int,
        default=2048,
        help="Number of wavelength bins.",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["trapz", "simps"],
        choices=["trapz", "simps"],
        help="Integration methods to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Calls per precision combo (lengthens sample window).",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="The number of threads to use.",
    )
    parser.add_argument(
        "--sample-freq",
        type=float,
        default=10000.0,
        help="RSS sampling frequency in Hz (busy-wait).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for synthetic input generation.",
    )
    args = parser.parse_args()

    profile_integration_memory_scaling(
        basename=args.basename,
        out_dir=args.out_dir,
        nentries=args.nentries,
        nlam=args.nlam,
        methods=args.methods,
        repeats=args.repeats,
        nthreads=args.nthreads,
        sample_freq=args.sample_freq,
        seed=args.seed,
    )
