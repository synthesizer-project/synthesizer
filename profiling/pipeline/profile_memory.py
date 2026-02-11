"""Profile Pipeline memory usage with continuous sampling.

This script builds galaxies, instruments, and an emission model, then uses
the Pipeline to run all requested operations while continuously sampling
RSS memory at 1kHz to capture peak usage during setup and execution.
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from pathlib import Path

import psutil
from astropy.cosmology import Planck18
from unyt import kpc

from synthesizer.grid import Grid
from synthesizer.pipeline import Pipeline

from .pipeline_test_data import (
    build_test_galaxies,
    get_test_emission_model,
    get_test_instruments,
    get_test_kernel,
)


def run_pipeline_with_memory(
    nparticles: int,
    ngalaxies: int,
    seed: int = 42,
    fov_kpc: float = 60.0,
    include_observer_frame: bool = False,
) -> tuple[list, float]:
    """Run full Pipeline and collect memory samples at 1000 Hz.

    Parameters
    ----------
    nparticles : int
        Number of stellar particles per galaxy
    ngalaxies : int
        Number of galaxies
    seed : int
        Random seed for reproducibility
    fov_kpc : float
        Field of view for imaging in kpc (default 60)
    include_observer_frame : bool
        If True, include observer-frame/flux operations

    Returns:
    -------
    tuple[list, float]
        List of (timestamp_ms, rss_mb) tuples and total runtime in seconds
    """
    samples = []
    stop_sampling = False

    def sample_memory():
        """Background thread to sample memory."""
        timestamp_ms = 0
        while not stop_sampling:
            rss_mb = psutil.Process().memory_info().rss / 1024 / 1024
            samples.append((timestamp_ms, rss_mb))
            timestamp_ms += 1
            time.sleep(0.001)  # 1000 Hz

    # Start sampling thread
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()

    start_time = time.perf_counter()

    # Setup - load grid
    grid = Grid("test_grid")

    # Build test data
    galaxies = build_test_galaxies(grid, nparticles, ngalaxies, seed)
    instruments = get_test_instruments(grid)
    kernel = get_test_kernel()
    model = get_test_emission_model(grid)

    # Create Pipeline
    pipeline = Pipeline(
        emission_model=model,
        nthreads=1,
        verbose=0,
    )
    pipeline.add_galaxies(galaxies)

    # Signal operations
    fov = fov_kpc * kpc

    # LOS optical depths
    pipeline.get_los_optical_depths(kernel=kernel)

    # SFZH and SFH
    pipeline.get_sfzh(grid.log10ages, grid.metallicities)
    pipeline.get_sfh(grid.log10ages)

    # Spectra (rest frame)
    pipeline.get_spectra()

    # Photometry (rest frame)
    pipeline.get_photometry_luminosities(
        instruments["photometry"],
        labels="intrinsic",
    )

    # Lines (rest frame)
    pipeline.get_lines(line_ids=grid.available_lines)

    # Imaging (rest frame)
    pipeline.get_images_luminosity(
        instruments["photometry"],
        fov=fov,
        kernel=kernel,
        labels="intrinsic",
    )

    # Data cubes (rest frame)
    pipeline.get_data_cubes_lnu(
        instruments["ifu"],
        fov=fov,
        kernel=kernel,
        labels="intrinsic",
    )

    # Spectroscopy (rest frame)
    pipeline.get_spectroscopy_lnu(
        instruments["spectroscopy"],
        labels="intrinsic",
    )

    # Observer frame operations if requested
    if include_observer_frame:
        cosmo = Planck18

        pipeline.get_observed_spectra(cosmo=cosmo)
        pipeline.get_photometry_fluxes(
            instruments["photometry"],
            cosmo=cosmo,
            labels="intrinsic",
        )
        pipeline.get_observed_lines(cosmo=cosmo)
        pipeline.get_images_flux(
            instruments["photometry"],
            fov=fov,
            kernel=kernel,
            cosmo=cosmo,
            labels="intrinsic",
        )
        pipeline.get_data_cubes_fnu(
            instruments["ifu"],
            fov=fov,
            kernel=kernel,
            cosmo=cosmo,
            labels="intrinsic",
        )
        pipeline.get_spectroscopy_fnu(
            instruments["spectroscopy"],
            cosmo=cosmo,
            labels="intrinsic",
        )

    # Run the Pipeline
    pipeline.run()

    total_time = time.perf_counter() - start_time
    stop_sampling = True
    sampler.join()

    return samples, total_time


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Profile Pipeline memory usage (1000 Hz sampling)"
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

    args = parser.parse_args()

    # Create output directory
    output_dir = Path("profiling/outputs/memory") / args.basename
    output_dir.mkdir(parents=True, exist_ok=True)

    particles_str = (
        f"particles={args.nparticles}, "
        f"galaxies={args.ngalaxies}, "
        f"fov={args.fov_kpc} kpc"
    )
    if args.include_observer_frame:
        particles_str += ", observer-frame=True"
    print(f"Profiling Pipeline memory ({particles_str})...")

    # Run pipeline with memory sampling
    samples, total_time = run_pipeline_with_memory(
        args.nparticles,
        args.ngalaxies,
        args.seed,
        args.fov_kpc,
        args.include_observer_frame,
    )

    # Write CSV with all samples
    csv_file = output_dir / "memory.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp_ms", "rss_mb"])
        for timestamp_ms, rss_mb in samples:
            writer.writerow([timestamp_ms, f"{rss_mb:.2f}"])

    # Print summary
    peak_mb = max(s[1] for s in samples) if samples else 0
    mean_mb = sum(s[1] for s in samples) / len(samples) if samples else 0
    min_mb = min(s[1] for s in samples) if samples else 0

    print(f"✓ Memory profile saved: {csv_file}")
    print(f"  Samples: {len(samples)}")
    print(f"  Peak: {peak_mb:.2f} MB")
    print(f"  Mean: {mean_mb:.2f} MB")
    print(f"  Min: {min_mb:.2f} MB")
    print(f"  Duration: {total_time:.3f}s")


if __name__ == "__main__":
    main()
