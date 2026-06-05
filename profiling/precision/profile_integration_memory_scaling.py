"""Profile integration memory across input and output precisions.

This script benchmarks the integration extension for one isolated workload per
subprocess. For each array size, all four input/output precision combinations
are profiled:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Memory (RSS) is sampled continuously at a configurable frequency by a
background thread while each isolated worker process builds its inputs and runs
the extension. The x-axis is normalised to % progress so runtime does not
affect the plot shape.

Usage:
    python profile_integration_memory_scaling.py --basename test
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
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
    xs = np.linspace(dtype(0.0), dtype(10.0), nlam, dtype=dtype)

    phases = rng.uniform(0.0, np.pi, size=(nentries, 1)).astype(dtype)
    amplitudes = rng.uniform(0.5, 2.0, size=(nentries, 1)).astype(dtype)
    frequencies = rng.uniform(0.5, 2.5, size=(nentries, 1)).astype(dtype)
    continuum = np.linspace(dtype(0.8), dtype(1.2), nlam, dtype=dtype)[None, :]

    ys = (
        amplitudes * np.sin(frequencies * xs[None, :] + phases)
        + dtype(0.1) * continuum
        + dtype(0.05) * np.cos(dtype(0.5) * xs[None, :])
    )

    return {
        "xs": np.array(xs, dtype=dtype, order="C", copy=False),
        "ys": np.array(ys, dtype=dtype, order="C", copy=False),
    }


def _sample_worker_case(
    method,
    nentries,
    nlam,
    input_dtype_name,
    output_dtype_name,
    nthreads,
    repeats,
    sample_freq_hz,
    seed,
):
    """Run one isolated precision case while continuously sampling RSS.

    The worker process starts from a fresh interpreter so previous profiling
    cases cannot inflate the resident set of the current measurement.
    Sampling starts before inputs are created so input precision differences
    are reflected directly in the reported memory usage.
    """
    func = METHODS[method]
    input_dtype = PRECISIONS[input_dtype_name]
    output_dtype = PRECISIONS[output_dtype_name]
    rng = np.random.default_rng(seed)

    rss_start = psutil.Process().memory_info().rss / (1024**2)
    samples = []
    stop_sampling = False
    interval = 1.0 / sample_freq_hz

    def sampler():
        next_sample = time.perf_counter()
        while not stop_sampling:
            now = time.perf_counter()
            if now >= next_sample:
                rss_mib = psutil.Process().memory_info().rss / (1024**2)
                samples.append(rss_mib)
                next_sample += interval

    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    inputs = make_synthetic_inputs(nentries, nlam, input_dtype, rng)
    result = None
    for _ in range(repeats):
        result = func(inputs["xs"], inputs["ys"], nthreads, output_dtype)

    stop_sampling = True
    sampler_thread.join()

    rss_end = psutil.Process().memory_info().rss / (1024**2)

    # Keep the allocated arrays alive until after sampling ends so the trace
    # reflects the full memory footprint of the isolated case.
    _ = (inputs, result)

    all_samples = [rss_start] + samples + [rss_end]
    n = len(all_samples)
    peak_rss_mib = max(all_samples)
    memory_trace = []
    for i, rss_mib in enumerate(all_samples):
        memory_trace.append(
            {
                "pct_complete": i / (n - 1) * 100.0,
                "rss_mib": rss_mib,
                "delta_mib": rss_mib - rss_start,
            }
        )

    return {
        "baseline_rss_mib": rss_start,
        "peak_rss_mib": peak_rss_mib,
        "peak_delta_mib": peak_rss_mib - rss_start,
        "memory_trace": memory_trace,
    }


def _run_worker_subprocess(
    method,
    nentries,
    nlam,
    input_dtype_name,
    output_dtype_name,
    nthreads,
    repeats,
    sample_freq,
    seed,
):
    """Run one profiling case in a fresh subprocess.

    The worker serializes its result to JSON for the parent process to read.
    """
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        result_path = Path(handle.name)

    command = [
        sys.executable,
        __file__,
        "--worker-mode",
        "--worker-output",
        str(result_path),
        "--worker-method",
        method,
        "--worker-nentries",
        str(nentries),
        "--worker-nlam",
        str(nlam),
        "--worker-input-dtype",
        input_dtype_name,
        "--worker-output-dtype",
        output_dtype_name,
        "--worker-nthreads",
        str(nthreads),
        "--worker-repeats",
        str(repeats),
        "--worker-sample-freq",
        str(sample_freq),
        "--worker-seed",
        str(seed),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        with result_path.open() as handle:
            return json.load(handle)
    finally:
        result_path.unlink(missing_ok=True)


def _run_worker_mode(args):
    """Execute one isolated case and serialize the result to JSON."""
    result = _sample_worker_case(
        method=args.worker_method,
        nentries=args.worker_nentries,
        nlam=args.worker_nlam,
        input_dtype_name=args.worker_input_dtype,
        output_dtype_name=args.worker_output_dtype,
        nthreads=args.worker_nthreads,
        repeats=args.worker_repeats,
        sample_freq_hz=args.worker_sample_freq,
        seed=args.worker_seed,
    )
    with Path(args.worker_output).open("w") as handle:
        json.dump(result, handle)


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
        "delta_mib",
        "baseline_rss_mib",
        "peak_rss_mib",
        "peak_delta_mib",
    ]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def plot_results(results, output_path):
    """Plot RSS above baseline against % progress for each precision pair."""
    methods = sorted({row["method"] for row in results})
    nentries_values = sorted({row["nentries"] for row in results})
    all_deltas = [row["delta_mib"] for row in results]
    max_delta = max(all_deltas, default=0.0)

    figure, axes = plt.subplots(
        len(nentries_values),
        len(methods),
        figsize=(6 * len(methods), 3.5 * len(nentries_values)),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    for row_index, nentries in enumerate(nentries_values):
        for col_index, method in enumerate(methods):
            axis = axes[row_index][col_index]

            for input_name in PRECISIONS:
                for output_name in PRECISIONS:
                    rows = [
                        row
                        for row in results
                        if row["method"] == method
                        and row["nentries"] == nentries
                        and row["input_dtype"] == input_name
                        and row["output_dtype"] == output_name
                    ]
                    rows.sort(key=lambda row: row["pct_complete"])
                    if not rows:
                        continue

                    axis.step(
                        [row["pct_complete"] for row in rows],
                        [row["delta_mib"] for row in rows],
                        where="post",
                        linewidth=1.2,
                        label=f"{input_name} -> {output_name}",
                    )

            if row_index == 0:
                axis.set_title(method)
            if col_index == 0:
                axis.set_ylabel(
                    f"nentries={nentries}\nRSS above baseline (MiB)"
                )
            if row_index == len(nentries_values) - 1:
                axis.set_xlabel("Progress through benchmark (%)")

            axis.grid(True, alpha=0.3)
            upper_limit = max_delta * 1.05 if max_delta > 0.0 else 1.0
            axis.set_ylim(0.0, upper_limit)

    axes[0][-1].legend(loc="best", fontsize=9)
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
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_integration_memory_scaling.csv"
    plot_path = output_dir / f"{basename}_integration_memory_scaling.png"

    results = []
    for entry_count in nentries:
        print(f"Profiling integration memory for nentries={entry_count}")

        for method in methods:
            for input_name in PRECISIONS:
                for output_name, output_dtype in PRECISIONS.items():
                    result = _run_worker_subprocess(
                        method=method,
                        nentries=entry_count,
                        nlam=nlam,
                        input_dtype_name=input_name,
                        output_dtype_name=output_name,
                        nthreads=nthreads,
                        repeats=repeats,
                        sample_freq=sample_freq,
                        seed=seed,
                    )

                    peak_rss = round(result["peak_rss_mib"], 3)
                    peak_delta = round(result["peak_delta_mib"], 3)
                    baseline = round(result["baseline_rss_mib"], 3)
                    for sample in result["memory_trace"]:
                        results.append(
                            {
                                "method": method,
                                "nentries": int(entry_count),
                                "nlam": int(nlam),
                                "input_dtype": input_name,
                                "output_dtype": output_name,
                                "pct_complete": round(
                                    sample["pct_complete"], 2
                                ),
                                "rss_mib": round(sample["rss_mib"], 3),
                                "delta_mib": round(sample["delta_mib"], 3),
                                "baseline_rss_mib": baseline,
                                "peak_rss_mib": peak_rss,
                                "peak_delta_mib": peak_delta,
                            }
                        )

                    print(
                        f"  {method} {input_name} -> {output_name}: "
                        f"peak_delta={peak_delta:.3f}MiB, "
                        f"peak_rss={peak_rss:.3f}MiB, "
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
        default=[10**3, 10**5],
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
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", type=str, help=argparse.SUPPRESS)
    parser.add_argument("--worker-nentries", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-nlam", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-input-dtype", type=str, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-output-dtype", type=str, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-nthreads", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-repeats", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-sample-freq", type=float, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-seed", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker_mode:
        _run_worker_mode(args)
        sys.exit(0)

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
