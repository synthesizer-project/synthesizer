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


def parse_atomic_timing_output(output: str) -> dict:
    """Parse atomic timing output to extract operation timings.

    Args:
        output (str): The captured stdout containing atomic timing lines.

    Returns:
        dict: A dictionary with operation names as keys and dicts containing
            'time' and 'source' as values.
    """
    timings = {}
    for line in output.splitlines():
        if "took:" in line and ("[C]" in line or "[Python]" in line):
            try:
                # Split on "took:" to separate operation from time
                key, value = line.split("took:")
                # Determine source (C++ or Python)
                source = "C" if "[C]" in key else "Python"
                # Clean up operation name
                operation = (
                    key.replace("[Python]", "").replace("[C]", "").strip()
                )
                # Extract time value
                time_str = value.replace("seconds", "").strip()
                time_val = float(time_str)

                # Store with source
                timings[operation] = {"time": time_val, "source": source}
            except (ValueError, AttributeError):
                # Skip malformed lines
                continue
    return timings


def run_pipeline_profiling(
    nparticles: int,
    ngalaxies: int,
    seed: int = 42,
    fov_kpc: float = 60.0,
    include_observer_frame: bool = False,
    nthreads: int = 8,
) -> dict:
    """Run full Pipeline profiling and return stage timings.

    This function captures atomic timing output from the Pipeline operations
    and parses it to extract individual operation timings.

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
        dict: A dictionary with operation names and timings in seconds.
    """
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

    # Redirect stdout to capture atomic timing output
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

        # Read captured atomic timing output
        with open(temp_file.name, "r") as f:
            output = f.read()
        os.unlink(temp_file.name)

    # Parse atomic timing output
    timings = parse_atomic_timing_output(output)

    return timings


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

    args = parser.parse_args()

    # Create output directory
    output_dir = Path("profiling/outputs/timing") / args.basename
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
    timings = run_pipeline_profiling(
        args.nparticles,
        args.ngalaxies,
        args.seed,
        args.fov_kpc,
        args.include_observer_frame,
        args.nthreads,
    )

    # Write CSV with source column
    csv_file = output_dir / "timing.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "seconds", "source"])
        for operation, data in timings.items():
            writer.writerow([operation, f"{data['time']:.6f}", data["source"]])

    print(f"✓ Timing profile saved: {csv_file}")
    for op, data in timings.items():
        print(f"  {op}: {data['time']:.3f}s ({data['source']})")


if __name__ == "__main__":
    main()
