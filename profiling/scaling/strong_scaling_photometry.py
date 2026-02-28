"""A script to test the strong scaling of the photometry calculation.

Usage:
    python strong_scaling_photometry.py --basename test --max_threads 8
       --nstars 100000 --nfilters 10
"""

import argparse
import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh
from synthesizer.utils.profiling_utils import run_scaling_test

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

# Set the seed
np.random.seed(42)


def photometry_strong_scaling(
    basename,
    out_dir,
    max_threads,
    nstars,
    nfilters,
    average_over,
    low_thresh,
    paper_style,
):
    """Profile the cpu time usage of the photometry calculation."""
    # Define the grid
    grid_name = "test_grid"

    grid = Grid(grid_name)

    # Get the emission model - use per_particle=True for particle photometry
    model = IncidentEmission(grid, per_particle=True)

    # Get the filters from cached instrument (no network access)
    webb_inst = get_test_instrument(grid)

    # Select the requested number of filters
    available_filters = webb_inst.available_filters[:nfilters]
    filters = webb_inst.filters.select(*available_filters)

    # Generate the star formation metallicity history
    mass = 10**10 * Msun
    param_stars = ParametricStars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=SFH.Constant(100 * Myr),
        metal_dist=ZDist.Normal(0.005, 0.01),
        initial_mass=mass,
    )

    # Sample the SFZH, producing a Stars object
    stars = sample_sfzh(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        nstars,
        redshift=1,
    )

    # Get spectra first so photometry has something to work with
    print("Generating initial per-particle spectra")
    stars.get_spectra(model)
    print()

    # Get photometry in serial first to get over any overhead due to linking
    # the first time the function is called
    print("Initial serial particle photometry calculation")
    stars.get_particle_photo_lnu(filters, nthreads=1)
    print()

    # Define the log and plot output paths
    log_outpath = (
        f"{out_dir}/{basename}_photometry_"
        f"totThreads{max_threads}_nstars{nstars}_nfilters{nfilters}.log"
    )
    plot_outpath = (
        f"{out_dir}/{basename}_photometry_"
        f"totThreads{max_threads}_nstars{nstars}_nfilters{nfilters}.png"
    )

    # Run the scaling test on particle photometry
    run_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        plot_outpath,
        stars.get_particle_photo_lnu,
        {
            "filters": filters,
        },
        total_msg="Getting Particle Photometry (lnu)",
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
        "--nstars",
        type=int,
        default=10**5,
        help="The number of stars to use in the simulation.",
    )

    args.add_argument(
        "--nfilters",
        type=int,
        default=10,
        help="The number of filters to use for photometry.",
    )

    args.add_argument(
        "--average_over",
        type=int,
        default=10,
        help="The number of times to average over.",
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

    args = args.parse_args()

    # Check for atomic timing
    from synthesizer import check_atomic_timing

    if not check_atomic_timing():
        raise RuntimeError(
            "Atomic timing not available. Recompile with: "
            "ATOMIC_TIMING=1 pip install -e ."
        )

    photometry_strong_scaling(
        args.basename,
        args.out_dir,
        args.max_threads,
        args.nstars,
        args.nfilters,
        args.average_over,
        args.low_thresh,
        args.paper_style,
    )
