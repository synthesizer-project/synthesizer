"""A script to profile the memory usage of a particle spectra calculation.

Usage:
    python profile_time_part_spectra.py --basename test
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from unyt import Myr

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfzh

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def profile_time_part_spectra(basename):
    """Profile the cpu time usage of the particle spectra calculation."""
    start = time.time()

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the grid (normally this would be defined by an SPS grid)
    log10ages = np.arange(6.0, 10.5, 0.1)
    metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
    metal_dist = ZDist.Normal(0.01, 0.005)
    sfh = SFH.Constant(100 * Myr)  # constant star formation

    # Generate the star formation metallicity history
    mass = 10**10
    param_stars = ParametricStars(
        log10ages,
        metallicities,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=mass,
    )

    # Setup lists for times
    times = []

    # Loop over the number of stars
    for n in [10, 10**2, 10**3, 10**4, 10**5]:
        start = time.time()

        # Sample the SFZH, producing a Stars object
        stars = sample_sfzh(
            param_stars.sfzh,
            param_stars.log10ages,
            param_stars.log10metallicities,
            n,
            current_masses=np.full(n, 10**8.7 / n),
            redshift=1,
        )

        stars.get_particle_spectra_incident(grid)

        print(f"{n} stars took", time.time() - start)
        times.append(time.time() - start)

    np.savetxt(f"{basename}_particle_time_prof.txt", times)


if __name__ == "__main__":
    # Get the command line args
    args = argparse.ArgumentParser()

    args.add_argument(
        "--basename",
        type=str,
        default="test",
        help="The basename of the output files.",
    )

    args = args.parse_args()

    profile_time_part_spectra(args.basename)
