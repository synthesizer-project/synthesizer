"""Profile photometry memory across input and output precisions.

This script benchmarks the batched photometry path for a 2D spectra array
with shape ``(nparticles, nlam)``.  For each particle count, all four
input/output precision combinations are profiled:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Memory (RSS) is sampled continuously at a configurable frequency by a
background thread while the extension runs.  The x-axis is normalised to
% progress so runtime does not affect the plot shape.

Usage:
    python profile_photometry_memory_scaling.py --basename test
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
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


def _sample_photometry(
    filters,
    spectra,
    nu,
    out_dtype,
    nthreads,
    repeats,
    sample_freq_hz,
):
    """Run photometry repeats while continuously sampling RSS.

    RSS is sampled on a background daemon thread at ``sample_freq_hz``
    using a busy-wait loop so the requested frequency is respected even
    for sub-ms calls.

    Returns (memory_trace, peak_mib).  Always includes at least a start
    and end sample so even fast operations produce a visible trace.
    """
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
        filters.apply_filters(
            spectra, nu=nu, nthreads=nthreads, out_dtype=out_dtype
        )

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
        "benchmark",
        "nparticles",
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
    figure, axis = plt.subplots(1, 1, figsize=(8, 6))

    for input_name in PRECISIONS:
        for output_name in PRECISIONS:
            rows = [
                row
                for row in results
                if row["input_dtype"] == input_name
                and row["output_dtype"] == output_name
            ]
            rows.sort(key=lambda row: row["pct_complete"])
            if not rows:
                continue
            xvals = [row["pct_complete"] for row in rows]
            yvals = [row["rss_mib"] for row in rows]
            axis.plot(
                xvals,
                yvals,
                linewidth=1,
                label=f"{input_name} -> {output_name}",
            )

    axis.set_xlabel("Progress through benchmark (%)")
    axis.set_ylabel("RSS memory (MiB)")
    axis.set_title("Photometry Memory Scaling")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best", fontsize=9)
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
    rng = np.random.default_rng(seed)

    grid = Grid("test_grid")
    instrument = get_test_instrument(grid)
    filters = instrument.filters.select(
        *instrument.available_filters[:nfilters]
    )
    nlam = grid.nlam

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{basename}_photometry_memory_scaling.csv"
    plot_path = output_dir / f"{basename}_photometry_memory_scaling.png"

    results = []
    for particle_count in nparticles:
        print(f"Profiling photometry memory for nparticles={particle_count}")

        spectra_by_dtype = {}
        nu_by_dtype = {}
        for input_name, input_dtype in PRECISIONS.items():
            spectra_by_dtype[input_name] = make_synthetic_spectra(
                particle_count, nlam, input_dtype, rng
            )
            nu_by_dtype[input_name] = np.array(
                (c / grid.lam).to("Hz").value,
                dtype=input_dtype,
                order="C",
                copy=True,
            )

        for input_name in PRECISIONS:
            for output_name, output_dtype in PRECISIONS.items():
                result = _sample_photometry(
                    filters,
                    spectra_by_dtype[input_name],
                    nu_by_dtype[input_name],
                    output_dtype,
                    nthreads,
                    repeats,
                    sample_freq,
                )

                peak = round(result["peak_mib"], 3)
                for pct, rss_mib in result["memory_trace"]:
                    results.append(
                        {
                            "benchmark": "synthetic_2d",
                            "nparticles": int(particle_count),
                            "input_dtype": input_name,
                            "output_dtype": output_name,
                            "pct_complete": round(pct, 2),
                            "rss_mib": round(rss_mib, 3),
                            "peak_mib": peak,
                        }
                    )

                print(
                    f"  {input_name} -> {output_name}: "
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
    args = parser.parse_args()

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
