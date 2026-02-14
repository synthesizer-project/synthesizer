"""A script to test the strong scaling of the particle spectra calculation.

Usage:
    python part_spectra_strong_scaling.py --basename test --max_threads 8
       --nstars 10**5
"""

import argparse
import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from unyt import Msun, Myr, kpc

from synthesizer import Grid
from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.stars import sample_sfzh
from synthesizer.particle.utils import calculate_smoothing_lengths
from synthesizer.utils.profiling_utils import run_scaling_test

# Add pipeline profiling to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "pipeline"))
from pipeline_test_data import get_test_instrument

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def images_strong_scaling(
    basename,
    out_dir,
    max_threads,
    nstars,
    average_over,
    low_thresh,
    paper_style,
):
    """Profile the cpu time usage of the particle spectra calculation."""
    # Define the grid
    grid_name = "test_grid"

    grid = Grid(grid_name)

    # Get the emission model
    model = IncidentEmission(grid)
    model.set_per_particle(True)

    # Get the filters from cached instrument (no network access)
    # The cached instrument has all JWST NIRCam filters available
    webb_inst = get_test_instrument(grid)

    # Generate the star formation metallicity history
    mass = 10**10 * Msun
    param_stars = ParametricStars(
        grid.log10ages,
        grid.metallicities,
        sf_hist=SFH.Constant(100 * Myr),
        metal_dist=ZDist.Normal(0.005, 0.01),
        initial_mass=mass,
    )

    # Generate some random coordinates
    coords = (
        CoordinateGenerator.generate_3D_gaussian(
            nstars,
            mean=np.array([50, 50, 50]),
            cov=np.array([[10, 0, 0], [0, 10, 0], [0, 0, 10]]),
        )
        * kpc
    )

    # Calculate the smoothing lengths
    smls = calculate_smoothing_lengths(coords, num_neighbours=56)

    # Sample the SFZH, producing a Stars object
    # we will also pass some keyword arguments for attributes
    # we will need for imaging
    stars = sample_sfzh(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        nstars,
        coordinates=coords,
        smoothing_lengths=smls,
        redshift=1,
        centre=np.array([50, 50, 50]) * kpc,
    )

    # Get the spectra
    stars.get_spectra(
        model,
        nthreads=max_threads,
    )

    # Get photometry - use only a single filter for faster imaging
    single_filter = webb_inst.filters.select(webb_inst.filters.filter_codes[0])
    stars.get_particle_photo_lnu(
        filters=single_filter,
        nthreads=max_threads,
    )

    # Get the kernel
    kernel = Kernel().get_kernel()

    # Get images in serial first to get over any overhead due to linking
    # the first time the function is called
    print("Initial imaging spectra calculation")
    stars.get_images_luminosity(
        "incident",
        fov=30 * kpc,
        instrument=webb_inst,
        kernel=kernel,
        cosmo=Planck18,
        nthreads=max_threads,
    )
    print()

    # Define the log and plot output paths
    log_outpath = (
        f"{out_dir}/{basename}_images_"
        f"totThreads{max_threads}_nstars{nstars}.log"
    )
    plot_outpath = (
        f"{out_dir}/{basename}_images_"
        f"totThreads{max_threads}_nstars{nstars}.png"
    )

    # Run the scaling test
    # Use partial to bind the label argument
    get_images = partial(
        stars.get_images_luminosity,
        "incident",
        fov=30 * kpc,
        instrument=webb_inst,
        kernel=kernel,
        cosmo=Planck18,
    )
    run_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        plot_outpath,
        get_images,
        {},
        total_msg="Generating images",
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

    images_strong_scaling(
        args.basename,
        args.out_dir,
        args.max_threads,
        args.nstars,
        args.average_over,
        args.low_thresh,
        args.paper_style,
    )
