"""Profile Doppler particle spectra memory across input and output precisions.

This script benchmarks the Doppler particle spectra extension for a synthetic
1D grid workload.  For each particle count, all four input/output precision
combinations are profiled for each grid assignment method:

- float32 -> float32
- float32 -> float64
- float64 -> float32
- float64 -> float64

Memory (RSS) is sampled continuously at a configurable frequency by a
background thread while the extension runs.  The x-axis is normalised to
% progress so runtime does not affect the plot shape.

Usage:
    python profile_doppler_particle_spectra_memory_scaling.py --basename test
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
    lam_mask = np.ones(nlam, dtype=bool)

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


def _sample_doppler_spectra(
    method,
    inputs,
    out_dtype,
    nthreads,
    repeats,
    sample_freq_hz,
):
    """Run Doppler particle spectra repeats while continuously sampling RSS.

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
        "nparticles",
        "ngrid",
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
    figure.suptitle("Doppler Particle Spectra Memory Scaling")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def profile_doppler_particle_spectra_memory_scaling(
    basename,
    out_dir,
    nparticles,
    ngrid,
    nlam,
    methods,
    repeats,
    nthreads,
    sample_freq,
    seed,
):
    """Run Doppler particle spectra memory scaling benchmarks."""
    rng = np.random.default_rng(seed)

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = (
        output_dir / f"{basename}_doppler_particle_spectra_memory_scaling.csv"
    )
    plot_path = (
        output_dir / f"{basename}_doppler_particle_spectra_memory_scaling.png"
    )

    results = []
    for particle_count in nparticles:
        print(
            "Profiling Doppler particle spectra memory for "
            f"nparticles={particle_count}"
        )

        inputs_by_dtype = {}
        for input_name, input_dtype in PRECISIONS.items():
            inputs_by_dtype[input_name] = make_synthetic_inputs(
                particle_count, ngrid, nlam, input_dtype, rng
            )

        for method in methods:
            for input_name in PRECISIONS:
                for output_name, output_dtype in PRECISIONS.items():
                    result = _sample_doppler_spectra(
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
                                "nparticles": int(particle_count),
                                "ngrid": int(ngrid),
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
        help="Calls per precision combo (lengthens sample window).",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="The number of threads to use for each spectra call.",
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

    profile_doppler_particle_spectra_memory_scaling(
        basename=args.basename,
        out_dir=args.out_dir,
        nparticles=args.nparticles,
        ngrid=args.ngrid,
        nlam=args.nlam,
        methods=args.methods,
        repeats=args.repeats,
        nthreads=args.nthreads,
        sample_freq=args.sample_freq,
        seed=args.seed,
    )
