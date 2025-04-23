"""A script to test the strong scaling of the integrated spectra calculation.

Usage:
    python int_spectra_strong_scaling.py --basename test --max_threads 8
       --nstars 10**5
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from unyt import Msun, Myr

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh
from synthesizer.utils.profiling_utils import run_scaling_test

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def int_spectra_strong_scaling(
    basename,
    out_dir,
    max_threads,
    nstars,
    average_over,
    gam,
    low_thresh,
):
    """Profile the cpu time usage of the particle spectra calculation."""
    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Get the emission model
    model = IncidentEmission(grid)

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

    # Get spectra in serial first to get over any overhead due to linking
    # the first time the function is called
    print("Initial serial spectra calculation")
    stars.get_spectra(model, nthreads=1, grid_assignment_method=gam)
    print()

    # Define the log and plot output paths
    log_outpath = (
        f"{out_dir}/{basename}_int_spectra_{gam}_"
        f"totThreads{max_threads}_nstars{nstars}.log"
    )
    plot_outpath = (
        f"{out_dir}/{basename}_int_spectra_{gam}_"
        f"totThreads{max_threads}_nstars{nstars}.png"
    )

    # Run the scaling test
    run_scaling_test(
        max_threads,
        average_over,
        log_outpath,
        plot_outpath,
        stars.get_spectra,
        {
            "emission_model": model,
            "grid_assignment_method": gam,
        },
        total_msg="Generating spectra",
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
        "--grid_assign",
        type=str,
        default="cic",
        help="The grid assignment method (cic or ngp).",
    )

    args.add_argument(
        "--low_thresh",
        type=float,
        default=0.1,
        help="the lower threshold on time for an operation to "
        "be included in the scaling test plot.",
    )

    args = args.parse_args()

    int_spectra_strong_scaling(
        args.basename,
        args.out_dir,
        args.max_threads,
        args.nstars,
        args.average_over,
        args.grid_assign,
        args.low_thresh,
    )
