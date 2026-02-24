"""Profile Pipeline execution timing using real Pipeline workflow.

This script builds galaxies, instruments, and an emission model, then uses
the Pipeline to run all requested operations and record timing data from
atomic timing output.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

from astropy.cosmology import Planck18
from unyt import kpc

from synthesizer import check_atomic_timing
from synthesizer.grid import Grid
from synthesizer.pipeline import Pipeline

# Add profiling/pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_test_data import (
    build_test_galaxies,
    get_test_emission_model,
    get_test_instrument,
    get_test_kernel,
)


def plot_time_vs_count_pipeline(
    timings: dict, outpath: str, threshold: float = 0.001
) -> None:
    """Create time vs count scatter plot for pipeline operations.

    Only includes operations that contribute more than threshold fraction
    of the total time (default 0.1% = 0.001).

    Args:
        timings (dict): Dictionary with operation timings.
        outpath (str): Path to save the plot.
        threshold (float): Minimum fraction of total time to include
            (default 0.001 = 0.1%).
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Calculate total time
    total_time = sum(op_data.get("time", 0.0) for op_data in timings.values())
    min_time = total_time * threshold

    # Collect data for scatter plot
    counts = []
    times = []
    labels = []
    colors = []

    for op_name, op_data in timings.items():
        count = op_data.get("count", 0)
        time_val = op_data.get("time", 0.0)
        source = op_data.get("source", "Python")

        # Only include if time exceeds threshold
        if count > 0 and time_val > 0 and time_val >= min_time:
            counts.append(count)
            times.append(time_val)
            labels.append(op_name)
            # Color by source
            if source == "C":
                colors.append("blue")
            else:
                colors.append("orange")

    if not counts:
        print("No operations with counts > 0 to plot")
        return

    # Create scatter plot with higher alpha to see overlaps
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(counts, times, c=colors, s=150, alpha=0.8, edgecolors="black")

    # Add labels for each point
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (counts[i], times[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            alpha=0.8,
        )

    ax.set_xlabel("Number of Calls", fontsize=12)
    ax.set_ylabel("Cumulative Time (s)", fontsize=12)
    ax.set_title("Pipeline Operation Time vs Call Count", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    # Add padding to axis limits so labels don't get cut off
    if counts and times:
        ax.set_xlim(min(counts) * 0.5, max(counts) * 2.0)
        ax.set_ylim(min(times) * 0.5, max(times) * 2.0)

    # Add legend for colors
    legend_elements = [
        Patch(facecolor="blue", edgecolor="black", label="C Extension"),
        Patch(facecolor="orange", edgecolor="black", label="Python"),
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=10)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    print(f"Saved time vs count plot to: {outpath}")
    plt.close(fig)


def run_pipeline_profiling(
    nparticles: int,
    ngalaxies: int,
    seed: int = 42,
    fov_kpc: float = 60.0,
    include_observer_frame: bool = False,
    nthreads: int = 8,
) -> tuple:
    """Run full Pipeline profiling and return stage timings.

    This function uses OperationTimers to collect timing data from all
    Pipeline operations without parsing stdout.

    Args:
        nparticles (int): Number of stellar particles per galaxy.
        ngalaxies (int): Number of galaxies.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.
        fov_kpc (float, optional): Field of view for imaging in kpc.
            Defaults to 60.0.
        include_observer_frame (bool, optional): If True, include
            observer-frame/flux operations in addition to rest-frame/luminosity
            operations. Defaults to False.
        nthreads (int, optional): Number of threads for Pipeline.
            Defaults to 8.

    Returns:
        tuple: A tuple containing:
            - dict: A dictionary with operation names and timing data.
                Each entry contains 'time', 'count', and 'source'.
            - Pipeline: The pipeline object with all computed results.
    """
    from synthesizer.utils.operation_timers import OperationTimers

    # Setup - load grid
    grid = Grid("test_grid")

    # Build test data
    galaxies = build_test_galaxies(grid, nparticles, ngalaxies, seed)
    instrument = get_test_instrument(grid)
    kernel = get_test_kernel()
    model = get_test_emission_model(grid)

    # Create Pipeline
    pipeline = Pipeline(
        emission_model=model,
        nthreads=nthreads,
        verbose=0,
    )
    pipeline.add_galaxies(galaxies)

    # Create and reset timers
    timers = OperationTimers()
    timers.reset()

    # Redirect stdout to capture print output (for logging)
    original_stdout_fd = os.dup(sys.stdout.fileno())
    temp_stdout = os.dup(original_stdout_fd)

    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            # Redirect stdout to temp file
            os.dup2(temp_file.fileno(), sys.stdout.fileno())

            # Signal operations
            # LOS optical depths (if we have gas and stars)
            pipeline.get_los_optical_depths(kernel=kernel)

            # SFZH and SFH
            pipeline.get_sfzh(grid.log10ages, grid.metallicities)
            pipeline.get_sfh(grid.log10ages)

            # Spectra (rest frame)
            pipeline.get_spectra()

            # Photometry (rest frame)
            pipeline.get_photometry_luminosities(instrument)

            # Lines (rest frame)
            pipeline.get_lines(line_ids=grid.available_lines)

            # Imaging (rest frame)
            fov = fov_kpc * kpc
            cosmo = Planck18
            pipeline.get_images_luminosity(
                instrument,
                fov=fov,
                kernel=kernel,
                cosmo=cosmo,
                labels="intrinsic",
            )

            # Observer frame operations if requested
            if include_observer_frame:
                cosmo = Planck18

                # Observed spectra
                pipeline.get_observed_spectra(cosmo=cosmo)

                # Photometric fluxes
                pipeline.get_photometry_fluxes(instrument, cosmo=cosmo)

                # Observed lines
                pipeline.get_observed_lines(cosmo=cosmo)

                # Flux images
                pipeline.get_images_flux(
                    instrument,
                    fov=fov,
                    kernel=kernel,
                    cosmo=cosmo,
                    labels="intrinsic",
                )

            # Run the Pipeline
            pipeline.run()

        finally:
            # Restore stdout
            os.dup2(temp_stdout, sys.stdout.fileno())
            os.close(temp_stdout)

        # Clean up temp file
        os.unlink(temp_file.name)

    # Extract timings from OperationTimers
    timings = {}
    for operation in timers.keys():
        cumulative_time, call_count, source = timers[operation]
        timings[operation] = {
            "time": cumulative_time,
            "count": call_count,
            "source": source,
        }

    return timings, pipeline


def main() -> None:
    """Main entry point for the timing profiling script."""
    # Check if atomic timing is available
    if not check_atomic_timing():
        raise RuntimeError(
            "Atomic timing not available. Recompile with: "
            "ATOMIC_TIMING=1 pip install -e ."
        )

    parser = argparse.ArgumentParser(
        description="Profile Pipeline execution timing"
    )
    parser.add_argument(
        "--basename", type=str, required=True, help="Basename for output"
    )
    parser.add_argument(
        "--nparticles",
        type=int,
        default=1000,
        help="Number of stellar particles per galaxy",
    )
    parser.add_argument(
        "--ngalaxies", type=int, default=10, help="Number of galaxies"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--fov-kpc",
        type=float,
        default=60.0,
        help="Field of view for imaging in kpc (default 60)",
    )
    parser.add_argument(
        "--include-observer-frame",
        action="store_true",
        help="Include observer-frame/flux operations",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=8,
        help="Number of threads for Pipeline (default 8)",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="Output directory for CSV and plots (default: current directory)",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.out_dir) / args.basename
    output_dir.mkdir(parents=True, exist_ok=True)

    particles_str = (
        f"particles={args.nparticles}, "
        f"galaxies={args.ngalaxies}, "
        f"fov={args.fov_kpc} kpc, "
        f"threads={args.nthreads}"
    )
    if args.include_observer_frame:
        particles_str += ", observer-frame=True"
    print(f"Profiling Pipeline timing ({particles_str})...")

    # Run pipeline profiling
    timings, pipeline = run_pipeline_profiling(
        args.nparticles,
        args.ngalaxies,
        args.seed,
        args.fov_kpc,
        args.include_observer_frame,
        args.nthreads,
    )

    # Write CSV with source and count columns
    csv_file = output_dir / "timing.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "seconds", "count", "source"])
        for operation, data in timings.items():
            writer.writerow(
                [
                    operation,
                    f"{data['time']:.6f}",
                    data["count"],
                    data["source"],
                ]
            )

    print(f"✓ Timing profile saved: {csv_file}")
    for op, data in timings.items():
        count_str = f"count={data['count']}"
        source_str = data["source"]
        print(f"  {op}: {data['time']:.3f}s ({count_str}, {source_str})")

    # Write pipeline output to HDF5
    h5_file = output_dir / "output.h5"
    pipeline.write(str(h5_file))
    print(f"✓ Pipeline output saved: {h5_file}")

    # Generate time vs count plot
    plot_file = output_dir / "time_vs_count.png"
    plot_time_vs_count_pipeline(timings, str(plot_file))
    print(f"✓ Time vs count plot saved: {plot_file}")


if __name__ == "__main__":
    main()
