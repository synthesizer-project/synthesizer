"""
This example generates a sample of star particles from a 2D SFZH, generates an
SED for each particle and then generates images in a number of Webb bands.
"""
import os
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from unyt import yr, Myr, kpc, arcsec
from astropy.cosmology import Planck18 as cosmo

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import FilterCollection as Filters


plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

if __name__ == "__main__":

    # Set the seed
    np.random.seed(42)

    start = time.time()

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the grid (normally this would be defined by an SPS grid)
    log10ages = np.arange(6.0, 10.5, 0.1)
    metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
    Z_p = {"Z": 0.01}
    Zh = ZH.deltaConstant(Z_p)
    sfh_p = {"duration": 100 * Myr}
    sfh = SFH.Constant(sfh_p)  # constant star formation
    sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh, stellar_mass=10**9)

    print("SFHZ sampled, took:", time.time() - start)

    stars_start = time.time()

    # Create stars object
    n = 100000  # number of particles for sampling
    coords = CoordinateGenerator.generate_3D_gaussian(n)
    stars = sample_sfhz(sfzh, n)
    stars.coordinates = coords
    stars.coord_units = kpc
    stars.initial_masses = np.full(n, 10**9 / n)
    cent = np.mean(coords, axis=0)  # define geometric centre
    rs = np.sqrt(
        (coords[:, 0] - cent[0]) ** 2
        + (coords[:, 1] - cent[1]) ** 2
        + (coords[:, 2] - cent[2]) ** 2
    )  # calculate radii
    rs[rs < 0.1] = 0.4  # Set a lower bound on the "smoothing length"
    stars.smoothing_lengths = rs / 4  # convert radii into smoothing lengths
    stars.redshift = 1

    # Compute width of stellar distribution
    width = np.max(coords) - np.min(coords)

    print("Stars created, took:", time.time() - stars_start)

    galaxy_start = time.time()

    # Create galaxy object
    galaxy = Galaxy(stars=stars)

    print("Galaxy created, took:", time.time() - galaxy_start)

    spectra_start = time.time()

    # Calculate the stars SEDs
    sed = galaxy.get_particle_spectra_stellar(grid)
    sed.get_fnu(cosmo, stars.redshift, igm=None)

    print("Spectra created, took:", time.time() - spectra_start)

    filter_start = time.time()

    # Define filter list
    filter_codes = [
        "JWST/NIRCam.F070W",
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F200W",
    ]

    # Set up filter object
    filters = Filters(filter_codes, new_lam=grid.lam)

    print("Filters created, took:", time.time() - filter_start)

    img_start = time.time()

    # Define image propertys
    redshift = 1
    resolution = ((width + 1) / 100) * kpc
    width = (width + 1) * kpc

    # Get the image
    hist_img = galaxy.make_images(
        resolution,
        fov=width,
        img_type="hist",
        sed=sed,
        filters=filters,
        rest_frame=False,
        cosmo=cosmo,
    )

    print("Histogram images made, took:", time.time() - img_start)
    img_start = time.time()

    # Get the image
    smooth_img = galaxy.make_images(
        resolution,
        fov=width,
        img_type="smoothed",
        sed=sed,
        filters=filters,
        rest_frame=False,
        cosmo=cosmo,
    )

    print("Smoothed images made, took:", time.time() - img_start)

    hist_imgs = hist_img.imgs
    smooth_imgs = smooth_img.imgs

    print("Sucessfuly made images for:", [key for key in hist_imgs])

    print("Total runtime (not including plotting):", time.time() - start)

    # Lets make a plot of the histogram images
    fig, ax = hist_img.plot_image(img_type="standard")

    fig.savefig(script_path + "/plots/flux_in_filters_RGB_test_hist.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Lets make a plot of the smoothed images
    fig, ax = smooth_img.plot_image(img_type="standard")

    fig.savefig(script_path + "/plots/flux_in_filters_RGB_test_hist_smoothed.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)

    # Lets make a plot of a single filter image with arcsinh scaling
    fig, ax = smooth_img.plot_image(img_type="standard",
                                    filter_code=filter_codes[0],
                                    scaling_func=np.arcsinh)
    fig.savefig(script_path + "/plots/flux_in_filters_RGB_test_hist_smoothed_"
                "single_filter.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


    # Also, lets make an RGB image
    smooth_img.make_rgb_image(
        rgb_filters={"R": ["JWST/NIRCam.F200W",],
                     "G": ["JWST/NIRCam.F150W",],
                     "B": ["JWST/NIRCam.F090W",]},
        img_type="standard",
    )
    fig, ax, rgb_img = smooth_img.plot_rgb_image()
    fig.savefig(script_path + "/plots/flux_in_filters_RGB_test.png",
                bbox_inches="tight", dpi=300)
    plt.close(fig)
