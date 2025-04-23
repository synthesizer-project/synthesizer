"""A script to test the strong scaling of the particle spectra calculation.

Usage:
    python part_spectra_strong_scaling.py --basename test --max_threads 8
       --nstars 10**5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr, angstrom, kpc

from synthesizer import Grid
from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.stars import sample_sfzh
from synthesizer.particle.utils import calculate_smoothing_lengths
from synthesizer.utils.profiling_utils import run_scaling_test

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
):
    """Profile the cpu time usage of the particle spectra calculation."""
    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Get the emission model
    model = IncidentEmission(grid)
    model.set_per_particle(True)

    # Get the filters
    lam = np.linspace(10**3, 10**5, 1000) * angstrom
    webb_filters = FilterCollection(
        filter_codes=[
            f"JWST/NIRCam.{f}"
            for f in ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
        ],
        new_lam=lam,
    )

    # Instatiate the instruments
    webb_inst = Instrument("JWST", filters=webb_filters, resolution=0.1 * kpc)

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

    # Get photometry
    stars.get_particle_photo_lnu(
        filters=webb_inst.filters,
        nthreads=max_threads,
    )

    # Get the kernel
    kernel = Kernel().get_kernel()

    # Get images in serial first to get over any overhead due to linking
    # the first time the function is called
    print("Initial imaging spectra calculation")
    stars.get_images_luminosity(
        webb_inst.resolution,
        30 * kpc,
        model,
        kernel=kernel,
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
    run_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        plot_outpath,
        stars.get_images_luminosity,
        {
            "resolution": webb_inst.resolution,
            "fov": 30 * kpc,
            "emission_model": model,
            "kernel": kernel,
        },
        total_msg="Generating images",
        low_thresh=low_thresh,
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

    args = args.parse_args()

    images_strong_scaling(
        args.basename,
        args.out_dir,
        args.max_threads,
        args.nstars,
        args.average_over,
        args.low_thresh,
    )
