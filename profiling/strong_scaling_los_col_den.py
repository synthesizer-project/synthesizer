"""A script to test the strong scaling of the LOS surface density calculation.

Usage:
    python los_surf_strong_scaling.py --basename test --max_threads 8
       --nstars 10**5 --ngas 10**5 --average_over 10
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr, kpc

from synthesizer import Grid
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.gas import Gas
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.particle.stars import sample_sfzh
from synthesizer.particle.utils import calculate_smoothing_lengths
from synthesizer.utils.profiling_utils import run_scaling_test

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def los_surface_density_strong_scaling(
    basename,
    out_dir,
    max_threads,
    nstars,
    ngas,
    average_over,
    low_thresh,
):
    """Profile the cpu time usage of the LOS surface density calculation."""
    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

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
    )

    # Now make the gas

    # Generate some random coordinates
    coords = (
        CoordinateGenerator.generate_3D_gaussian(
            ngas,
            mean=np.array([50, 50, 50]),
        )
        * kpc
    )

    # Calculate the smoothing lengths
    smls = calculate_smoothing_lengths(coords, num_neighbours=56)

    gas = Gas(
        masses=np.random.uniform(10**6, 10**6.5, ngas) * Msun,
        metallicities=np.random.uniform(0.01, 0.05, ngas),
        coordinates=coords,
        smoothing_lengths=smls,
        dust_to_metal_ratio=0.2,
    )

    # Create galaxy object
    galaxy = Galaxy("Galaxy", stars=stars, gas=gas, redshift=1)

    # Get the kernel
    kernel = Kernel().get_kernel()

    # Run a single threaded test first to get overt any overhead due to linking
    # the first time the function is called
    print("Running single threaded test")
    galaxy.get_stellar_los_tau_v(
        kappa=0.075,
        kernel=kernel,
        nthreads=1,
    )
    print()

    # Define the log and plot output paths
    log_outpath = (
        f"{out_dir}/{basename}_los_column_density_"
        f"totThreads{max_threads}_nstars{nstars}_ngas{ngas}.log"
    )
    plot_outpath = (
        f"{out_dir}/{basename}_los_column_density_"
        f"totThreads{max_threads}_nstars{nstars}_ngas{ngas}.png"
    )

    # Run the scaling test
    run_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        plot_outpath,
        galaxy.get_stellar_los_tau_v,
        {
            "kappa": 0.075,
            "kernel": kernel,
        },
        total_msg="Calculating LOS surface density",
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
        default=10**4,
        help="The number of stars to use in the simulation.",
    )

    args.add_argument(
        "--ngas",
        type=int,
        default=10**4,
        help="The number of gas particles to use in the simulation",
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

    los_surface_density_strong_scaling(
        args.basename,
        args.out_dir,
        args.max_threads,
        args.nstars,
        args.ngas,
        args.average_over,
        args.low_thresh,
    )
