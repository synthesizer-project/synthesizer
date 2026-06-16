"""Profile photometry memory across input and output precisions.

This script benchmarks the batched photometry path for one isolated workload
per subprocess: a 2D spectra array with shape ``(nparticles, nlam)``. For
each particle count, all four input/output precision combinations are
profiled:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Memory (RSS) is sampled continuously at a configurable frequency by a
background thread while each isolated worker process builds its inputs and runs
the extension. The x-axis is normalised to % progress so runtime does not
affect the plot shape.

Usage:
    python profile_photometry_memory_scaling.py --basename test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psutil
from unyt import c

from synthesizer.grid import Grid

pipeline_path = (
    Path(__file__).parent.parent / "pipeline" / "pipeline_test_data.py"
)
spec = importlib.util.spec_from_file_location(
    "pipeline_test_data", pipeline_path
)
pipeline_test_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline_test_data)
get_test_instrument = pipeline_test_data.get_test_instrument

LOGGER = logging.getLogger(__name__)

plt.rcParams["font.family"] = "DejaVu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}


def make_synthetic_spectra(nparticles, nlam, dtype, rng):
    """Create a contiguous synthetic 2D spectra array."""
    lam_axis = np.linspace(-3.0, 3.0, nlam, dtype=np.float64)
    base = np.exp(-0.5 * lam_axis**2) + 0.15 * np.sin(4.0 * lam_axis)
    base = base[None, :]

    amplitudes = rng.uniform(0.5, 2.0, size=(nparticles, 1))
    slopes = rng.uniform(0.0, 0.2, size=(nparticles, 1))
    continuum = np.linspace(0.8, 1.2, nlam, dtype=np.float64)[None, :]
    spectra = amplitudes * base + slopes * continuum

    return np.array(spectra, dtype=dtype, order="C", copy=True)


def _sample_worker_case(
    nparticles,
    nfilters,
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

    grid = Grid("test_grid")
    instrument = get_test_instrument(grid)
    filters = instrument.filters.select(
        *instrument.available_filters[:nfilters]
    )
    spectra = make_synthetic_spectra(nparticles, grid.nlam, input_dtype, rng)
    nu = np.array(
        (c / grid.lam).to("Hz").value,
        dtype=input_dtype,
        order="C",
        copy=True,
    )

    result = None
    for _ in range(repeats):
        result = filters.apply_filters(
            spectra,
            nu=nu,
            nthreads=nthreads,
            out_dtype=output_dtype,
        )

    stop_sampling = True
    sampler_thread.join()

    rss_end = psutil.Process().memory_info().rss / (1024**2)

    # Keep allocated objects alive until after sampling ends so the trace
    # reflects the full memory footprint of the isolated case.
    _ = (grid, instrument, filters, spectra, nu, result)

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
    nparticles,
    nfilters,
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
        "--worker-nparticles",
        str(nparticles),
        "--worker-nfilters",
        str(nfilters),
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
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as error:
            LOGGER.error(
                "Worker subprocess failed for %s -> %s at nparticles=%s\n"
                "stdout:\n%s\n"
                "stderr:\n%s",
                input_dtype_name,
                output_dtype_name,
                nparticles,
                error.stdout,
                error.stderr,
            )
            raise
        with result_path.open() as handle:
            return json.load(handle)
    finally:
        result_path.unlink(missing_ok=True)


def _run_worker_mode(args):
    """Execute one isolated case and serialize the result to JSON."""
    result = _sample_worker_case(
        nparticles=args.worker_nparticles,
        nfilters=args.worker_nfilters,
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
        "benchmark",
        "nparticles",
        "nfilters",
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
    nparticle_values = sorted({row["nparticles"] for row in results})
    figure, axes = plt.subplots(
        len(nparticle_values),
        1,
        figsize=(8, 3.5 * len(nparticle_values)),
        squeeze=False,
        sharex=True,
    )

    row_max_delta = {
        nparticles: max(
            (
                row["delta_mib"]
                for row in results
                if row["nparticles"] == nparticles
            ),
            default=0.0,
        )
        for nparticles in nparticle_values
    }

    for row_index, nparticles in enumerate(nparticle_values):
        axis = axes[row_index][0]

        for input_name in PRECISIONS:
            for output_name in PRECISIONS:
                rows = [
                    row
                    for row in results
                    if row["nparticles"] == nparticles
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

        axis.set_ylabel(f"nparticles={nparticles}\nRSS above baseline (MiB)")
        axis.grid(True, alpha=0.3)
        upper_limit = (
            row_max_delta[nparticles] * 1.05
            if row_max_delta[nparticles] > 0.0
            else 1.0
        )
        axis.set_ylim(0.0, upper_limit)

        if row_index == len(nparticle_values) - 1:
            axis.set_xlabel("Progress through benchmark (%)")

    axes[0][0].legend(loc="best", fontsize=9)
    figure.suptitle("Photometry Memory Scaling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def profile_photometry_memory_scaling(
    basename,
    out_dir,
    nparticles,
    nfilters,
    repeats,
    nthreads,
    sample_freq,
    seed,
):
    """Run photometry memory scaling benchmarks."""
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_photometry_memory_scaling.csv"
    plot_path = output_dir / f"{basename}_photometry_memory_scaling.png"

    results = []
    for particle_count in nparticles:
        print(f"Profiling photometry memory for nparticles={particle_count}")

        for input_name in PRECISIONS:
            for output_name, _output_dtype in PRECISIONS.items():
                result = _run_worker_subprocess(
                    nparticles=particle_count,
                    nfilters=nfilters,
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
                            "benchmark": "synthetic_2d",
                            "nparticles": int(particle_count),
                            "nfilters": int(nfilters),
                            "input_dtype": input_name,
                            "output_dtype": output_name,
                            "pct_complete": round(sample["pct_complete"], 2),
                            "rss_mib": round(sample["rss_mib"], 3),
                            "delta_mib": round(sample["delta_mib"], 3),
                            "baseline_rss_mib": baseline,
                            "peak_rss_mib": peak_rss,
                            "peak_delta_mib": peak_delta,
                        }
                    )

                print(
                    f"  {input_name} -> {output_name}: "
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
        help="Calls per precision combo (lengthens sample window).",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="The number of threads to use for each photometry call.",
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
        help="Random seed used for synthetic spectra generation.",
    )
    parser.add_argument(
        "--worker-mode",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--worker-output", type=str, help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-nparticles", type=int, help=argparse.SUPPRESS
    )
    parser.add_argument("--worker-nfilters", type=int, help=argparse.SUPPRESS)
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

    profile_photometry_memory_scaling(
        basename=args.basename,
        out_dir=args.out_dir,
        nparticles=args.nparticles,
        nfilters=args.nfilters,
        repeats=args.repeats,
        nthreads=args.nthreads,
        sample_freq=args.sample_freq,
        seed=args.seed,
    )
