"""Profile Pipeline execution timing using real Pipeline workflow.

This script builds galaxies, instruments, and an emission model, then uses
the Pipeline to run all requested operations and record timing data from
both setup stages and Pipeline._op_timing.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from astropy.cosmology import Planck18
from unyt import kpc

from synthesizer.grid import Grid
from synthesizer.pipeline import Pipeline

# Add profiling/pipeline to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_test_data import (
    build_test_galaxies,
    get_test_emission_model,
    get_test_instruments,
    get_test_kernel,
)


def run_pipeline_profiling(
    nparticles: int,
    ngalaxies: int,
    seed: int = 42,
    fov_kpc: float = 60.0,
    include_observer_frame: bool = False,
) -> dict:
    """Run full Pipeline profiling and return stage timings.

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
        If True, include observer-frame/flux operations in addition to
        rest-frame/luminosity operations

    Returns:
    -------
    dict
        Dictionary with stage names and timings in seconds
    """
    timings = {}

    # Setup - load grid
    t_start = time.perf_counter()
    grid = Grid("test_grid")
    timings["grid_load"] = time.perf_counter() - t_start

    # Build test data
    t_start = time.perf_counter()
    galaxies = build_test_galaxies(grid, nparticles, ngalaxies, seed)
    timings["build_galaxies"] = time.perf_counter() - t_start

    t_start = time.perf_counter()
    instruments = get_test_instruments(grid)
    kernel = get_test_kernel()
    timings["build_instruments"] = time.perf_counter() - t_start

    t_start = time.perf_counter()
    model = get_test_emission_model(grid)
    timings["model_setup"] = time.perf_counter() - t_start

    # Create Pipeline
    t_start = time.perf_counter()
    pipeline = Pipeline(
        emission_model=model,
        nthreads=1,
        verbose=0,
    )
    pipeline.add_galaxies(galaxies)
    timings["pipeline_setup"] = time.perf_counter() - t_start

    # Signal operations
    t_start = time.perf_counter()

    # LOS optical depths (if we have gas and stars)
    pipeline.get_los_optical_depths(kernel=kernel)

    # SFZH and SFH
    pipeline.get_sfzh(grid.log10ages, grid.metallicities)
    pipeline.get_sfh(grid.log10ages)

    # Spectra (rest frame)
    pipeline.get_spectra()

    # Photometry (rest frame)
    pipeline.get_photometry_luminosities(
        instruments["photometry"],
    )

    # Lines (rest frame)
    pipeline.get_lines(line_ids=grid.available_lines)

    # Imaging (rest frame)
    fov = fov_kpc * kpc
    cosmo = Planck18
    pipeline.get_images_luminosity(
        instruments["imaging"],
        fov=fov,
        kernel=kernel,
        cosmo=cosmo,
    )

    # Data cubes (rest frame)
    pipeline.get_data_cubes_lnu(
        instruments["ifu"],
        fov=fov,
        kernel=kernel,
    )

    # Spectroscopy (rest frame)
    pipeline.get_spectroscopy_lnu(
        instruments["spectroscopy"],
    )

    # Observer frame operations if requested
    if include_observer_frame:
        cosmo = Planck18

        # Observed spectra
        pipeline.get_observed_spectra(cosmo=cosmo)

        # Photometric fluxes
        pipeline.get_photometry_fluxes(
            instruments["photometry"],
            cosmo=cosmo,
        )

        # Observed lines
        pipeline.get_observed_lines(cosmo=cosmo)

        # Flux images
        pipeline.get_images_flux(
            instruments["imaging"],
            fov=fov,
            kernel=kernel,
            cosmo=cosmo,
        )

        # Data cubes (flux)
        pipeline.get_data_cubes_fnu(
            instruments["ifu"],
            fov=fov,
            kernel=kernel,
            cosmo=cosmo,
        )

        # Spectroscopy (flux)
        pipeline.get_spectroscopy_fnu(
            instruments["spectroscopy"],
            cosmo=cosmo,
        )

    timings["signal_operations"] = time.perf_counter() - t_start

    # Run the Pipeline
    t_start = time.perf_counter()
    pipeline.run()
    timings["pipeline_run"] = time.perf_counter() - t_start

    # Extract operation timings from Pipeline
    for op_name, op_time in pipeline._op_timing.items():
        if op_time > 0:
            timings[f"op_{op_name}"] = op_time

    # Compute total
    timings["total"] = sum(v for k, v in timings.items() if k != "total")

    return timings


def main() -> None:
    """Main entry point."""
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

    args = parser.parse_args()

    # Create output directory
    output_dir = Path("profiling/outputs/timing") / args.basename
    output_dir.mkdir(parents=True, exist_ok=True)

    particles_str = (
        f"particles={args.nparticles}, "
        f"galaxies={args.ngalaxies}, "
        f"fov={args.fov_kpc} kpc"
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
    )

    # Write CSV
    csv_file = output_dir / "timing.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["operation", "seconds"])
        for operation, seconds in timings.items():
            writer.writerow([operation, f"{seconds:.6f}"])

    print(f"✓ Timing profile saved: {csv_file}")
    for op, t in timings.items():
        print(f"  {op}: {t:.3f}s")


if __name__ == "__main__":
    main()
