"""A script to test the strong scaling of the full Pipeline execution.

This script follows the exact setup of all other scaling tests, but uses the
profile_timing pipeline setup where we set up N galaxies each with M particles
and then run the pipeline with every possible operation enabled. This allows us
to see what is scaling well as part of the pipeline and what isn't.

Usage:
    python strong_scaling_pipeline.py --basename test --max_threads 8
       --nparticles 1000 --ngalaxies 10
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from unyt import kpc

from synthesizer.grid import Grid
from synthesizer.pipeline import Pipeline
from synthesizer.utils.profiling_utils import run_scaling_test

# Add pipeline profiling to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from pipeline_test_data import (
    build_test_galaxies,
    get_test_emission_model,
    get_test_instrument,
    get_test_kernel,
)

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def run_full_pipeline(
    nparticles,
    ngalaxies,
    grid,
    model,
    instrument,
    kernel,
    fov_kpc,
    cosmo,
    include_observer_frame,
    nthreads,
):
    """Execute the full pipeline with all operations enabled.

    This function is designed to be called by run_scaling_test with
    different thread counts. It creates fresh galaxies and a new pipeline
    for each run.

    Args:
        nparticles (int): Number of particles per galaxy.
        ngalaxies (int): Number of galaxies.
        grid (Grid): The SPS grid.
        model: The emission model.
        instrument: The instrument for photometry and imaging.
        kernel: The SPH kernel for imaging.
        fov_kpc (float): Field of view in kpc.
        cosmo: The cosmology object.
        include_observer_frame (bool): Whether to include observer-frame ops.
        nthreads (int): Number of threads to use.
    """
    # Convert fov to unyt quantity
    fov = fov_kpc * kpc

    # Build fresh galaxies for this run (use a fixed seed for consistency)
    galaxies = build_test_galaxies(grid, nparticles, ngalaxies, seed=42)

    # Create a new Pipeline for this run with the specified nthreads
    pipeline = Pipeline(
        emission_model=model,
        nthreads=nthreads,
        verbose=0,
    )
    pipeline.add_galaxies(galaxies)

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
    pipeline.get_images_luminosity(
        instrument,
        fov=fov,
        kernel=kernel,
        cosmo=cosmo,
        labels="intrinsic",
    )

    # Observer frame operations if requested
    if include_observer_frame:
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


def pipeline_strong_scaling(
    basename,
    out_dir,
    max_threads,
    nparticles,
    ngalaxies,
    average_over,
    fov_kpc,
    low_thresh,
    paper_style,
    include_observer_frame,
):
    """Profile the cpu time usage of the full Pipeline execution.

    Args:
        basename (str): The basename of the output files.
        out_dir (str): The output directory for the log and plot files.
        max_threads (int): The maximum number of threads to use.
        nparticles (int): The number of stellar particles per galaxy.
        ngalaxies (int): The number of galaxies.
        average_over (int): The number of times to average over.
        fov_kpc (float): Field of view for imaging in kpc.
        low_thresh (float): The lower threshold on time for an operation to
            be included in the scaling test plot.
        paper_style (bool): Use the paper style for the plot.
        include_observer_frame (bool): Whether to include observer-frame ops.
    """
    # Define the grid
    grid_name = "test_grid"
    grid = Grid(grid_name)

    # Build test data (shared across runs)
    instrument = get_test_instrument(grid)
    kernel = get_test_kernel()
    model = get_test_emission_model(grid)
    cosmo = Planck18

    # Run the pipeline in serial first to get over any overhead
    print("Initial serial pipeline execution")
    run_full_pipeline(
        nparticles,
        ngalaxies,
        grid,
        model,
        instrument,
        kernel,
        fov_kpc,
        cosmo,
        include_observer_frame,
        nthreads=1,
    )
    print()

    # Define the log and plot output paths
    if include_observer_frame:
        log_outpath = (
            f"{out_dir}/{basename}_pipeline_"
            f"totThreads{max_threads}_"
            f"nparticles{nparticles}_"
            f"ngalaxies{ngalaxies}_"
            f"observer.log"
        )
        plot_outpath = (
            f"{out_dir}/{basename}_pipeline_"
            f"totThreads{max_threads}_"
            f"nparticles{nparticles}_"
            f"ngalaxies{ngalaxies}_"
            f"observer.png"
        )
    else:
        log_outpath = (
            f"{out_dir}/{basename}_pipeline_"
            f"totThreads{max_threads}_"
            f"nparticles{nparticles}_"
            f"ngalaxies{ngalaxies}.log"
        )
        plot_outpath = (
            f"{out_dir}/{basename}_pipeline_"
            f"totThreads{max_threads}_"
            f"nparticles{nparticles}_"
            f"ngalaxies{ngalaxies}.png"
        )

    # Run the scaling test
    run_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        plot_outpath,
        run_full_pipeline,
        {
            "nparticles": nparticles,
            "ngalaxies": ngalaxies,
            "grid": grid,
            "model": model,
            "instrument": instrument,
            "kernel": kernel,
            "fov_kpc": fov_kpc,
            "cosmo": cosmo,
            "include_observer_frame": include_observer_frame,
        },
        total_msg="Running full pipeline",
        low_thresh=low_thresh,
        paper_style=paper_style,
    )


if __name__ == "__main__":
    # Get the command line args
    args = argparse.ArgumentParser()

    args.add_argument(
        "--basename",
        type=str,
        default="test",
        help="The basename of the output files.",
    )

    args.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="The output directory for the log and plot files."
        " Defaults to the current directory.",
    )

    args.add_argument(
        "--max_threads",
        type=int,
        default=8,
        help="The maximum number of threads to use.",
    )

    args.add_argument(
        "--nparticles",
        type=int,
        default=1000,
        help="The number of stellar particles per galaxy.",
    )

    args.add_argument(
        "--ngalaxies",
        type=int,
        default=10,
        help="The number of galaxies.",
    )

    args.add_argument(
        "--average_over",
        type=int,
        default=10,
        help="The number of times to average over.",
    )

    args.add_argument(
        "--fov_kpc",
        type=float,
        default=60.0,
        help="Field of view for imaging in kpc.",
    )

    args.add_argument(
        "--low_thresh",
        type=float,
        default=0.1,
        help="the lower threshold on time for an operation to "
        "be included in the scaling test plot.",
    )

    args.add_argument(
        "--paper_style",
        action="store_true",
        help="Use the paper style for the plot (legend below the plot and "
        "smaller proportions).",
    )

    args.add_argument(
        "--include_observer_frame",
        action="store_true",
        help="Include observer-frame/flux operations in addition to "
        "rest-frame/luminosity operations.",
    )

    args = args.parse_args()

    # Check for atomic timing
    from synthesizer import check_atomic_timing

    if not check_atomic_timing():
        raise RuntimeError(
            "Atomic timing not available. Recompile with: "
            "ATOMIC_TIMING=1 pip install -e ."
        )

    pipeline_strong_scaling(
        args.basename,
        args.out_dir,
        args.max_threads,
        args.nparticles,
        args.ngalaxies,
        args.average_over,
        args.fov_kpc,
        args.low_thresh,
        args.paper_style,
        args.include_observer_frame,
    )
