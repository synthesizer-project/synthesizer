"""
This example generates a sample of star particles from a 2D SFZH, generates an
SED for each particle and then generates images in a number of Webb bands.
"""
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from unyt import yr, Myr
import gc

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.galaxy.particle import ParticleGalaxy as Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import SVOFilterCollection as Filters
from synthesizer.kernel_functions import quintic

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

if __name__ == '__main__':

    # Set the seed
    np.random.seed(42)

    start = time.time()

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the grid (normally this would be defined by an SPS grid)
    log10ages = np.arange(6., 10.5, 0.1)
    metallicities = 10**np.arange(-5., -1.5, 0.1)
    Z_p = {'Z': 0.01}
    Zh = ZH.deltaConstant(Z_p)
    sfh_p = {'duration': 100 * Myr}
    sfh = SFH.Constant(sfh_p)  # constant star formation
    sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)

    filter_start = time.time()

    # Define filter list
    filter_codes = ["JWST/NIRCam.F090W", "JWST/NIRCam.F150W",
                    "JWST/NIRCam.F200W"]

    # Set up filter object
    filters = Filters(filter_codes, new_lam=grid.lam)

    print("Filters created, took:", time.time() - filter_start)

    # Create list of particle numbers
    ns = [20, 50, 100, 500, 1000, 2000, 5000, 10000, 50000, 100000]

    # Create arrays to store runtimes
    create_stars = np.zeros(len(ns))
    create_gals = np.zeros(len(ns))
    create_spec = np.zeros(len(ns))
    create_hist = np.zeros(len(ns))
    create_smooth = np.zeros(len(ns))

    # Loop over particle numbers
    for ind, n in enumerate(ns):

        stars_start = time.time()

        # Create stars object
        coords = CoordinateGenerator.generate_3D_gaussian(n)
        stars = sample_sfhz(sfzh, n)
        stars.coordinates = coords
        stars.smoothing_lengths = np.ones(n) / 4

        # Compute width of stellar distribution
        width = np.max(coords) - np.min(coords)

        create_stars[ind] = time.time() - stars_start

        galaxy_start = time.time()

        # Create galaxy object
        galaxy = Galaxy(stars=stars)

        create_gals[ind] = time.time() - galaxy_start

        spectra_start = time.time()

        # Calculate the stars SEDs
        galaxy.generate_intrinsic_spectra(grid, update=True, integrated=False)

        create_spec[ind] = time.time() - spectra_start

        # Define image propertys
        resolution = (width + 1) / 100

        img_start = time.time()

        # Get the image
        hist_img = galaxy.make_image(resolution, fov=width + 1,
                                     img_type="hist",
                                     sed=galaxy.spectra_array["intrinsic"],
                                     filters=filters,
                                     kernel_func=quintic, rest_frame=True)

        create_hist[ind] = time.time() - img_start
        img_start = time.time()

        # Get the image
        smooth_img = galaxy.make_image(resolution, fov=width + 1,
                                       img_type="smoothed",
                                       sed=galaxy.spectra_array["intrinsic"],
                                       filters=filters,
                                       kernel_func=quintic, rest_frame=True)

        create_smooth[ind] = time.time() - img_start

        print("Finished N_filters=%d" % n)
        gc.collect()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.grid(True)

    # Plot each timed section
    for arr, lab in zip([create_stars, create_gals, create_spec,
                         create_hist, create_smooth],
                        ["Stars", "Galaxy", "SED",
                         "Image (hist)", "Image (smoothed)"]):
        ax.loglog(ns, arr, label=lab)

    # Label axes
    ax.set_ylabel("Wallclock / [s]")
    ax.set_xlabel("$N_\star$")

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              fancybox=True, shadow=True, ncol=5)

    # Plot the image
    plt.savefig("../particle_scaling_test.png", bbox_inches="tight", dpi=300)
