"""A script to test the strong scaling of the particle spectra calculation.

Usage:
    python part_spectra_strong_scaling.py --basename test --max_threads 8
       --nstars 10**5
"""

import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.stars import sample_sfhz
from unyt import Myr

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(42)


def part_spectra_strong_scaling(basename, max_threads=8, nstars=10**5):
    """Profile the cpu time usage of the integrated spectra calculation."""
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

    # Sample the SFZH, producing a Stars object
    stars = sample_sfhz(
        param_stars.sfzh,
        param_stars.log10ages,
        param_stars.log10metallicities,
        nstars,
        current_masses=np.full(nstars, 10**8.7 / nstars),
        redshift=1,
    )

    # Get spectra in serial first to get over any overhead due to linking
    # the first time the function is called
    stars.get_particle_spectra_incident(grid, nthreads=1)
    print("++++++++++++++++++++++++++++++++++++++++++++++")
    print()

    # Setup lists for times
    times = []
    threads = []

    # Loop over the number of threads
    nthreads = 1
    while nthreads <= max_threads:
        spec_start = time.time()
        stars.get_particle_spectra_incident(grid, nthreads=nthreads)
        print(
            f"{nstars} stars with {nthreads} threads took",
            time.time() - spec_start,
        )

        times.append(time.time() - spec_start)
        threads.append(nthreads)

        nthreads *= 2

    # Make sure we test the max_threads in case it wasn't a power of 2
    if max_threads not in threads:
        spec_start = time.time()
        stars.get_particle_spectra_incident(grid, nthreads=max_threads)
        print(
            f"{nstars} stars with {max_threads} threads took",
            time.time() - spec_start,
        )

        times.append(time.time() - spec_start)
        threads.append(max_threads)

    # Combine times and threads into a single array
    times = np.array([threads, times])

    np.savetxt(f"{basename}_particle_strong_scaling_{nstars}.txt", times)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(times[0], times[1], "o-")
    ax.set_xlabel("Number of Threads")
    ax.set_ylabel("Time (s)")
    ax.set_title(f"Particle Spectra Strong Scaling ({nstars} stars)")
    ax.grid(True)
    plt.show()


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

    args = args.parse_args()

    part_spectra_strong_scaling(args.basename, args.max_threads, args.nstars)
