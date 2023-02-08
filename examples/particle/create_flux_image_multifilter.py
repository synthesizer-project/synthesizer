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
from synthesizer.galaxy.particle import ParticleGalaxy as Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import FilterCollection as Filters
from synthesizer.kernel_functions import quintic


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
    n = 100  # number of particles for sampling
    coords = CoordinateGenerator.generate_3D_gaussian(n)
    stars = sample_sfhz(sfzh, n)
    stars.coordinates = coords
    stars.coord_units = kpc
    cent = np.mean(coords, axis=0)  # define geometric centre
    rs = np.sqrt(
        (coords[:, 0] - cent[0]) ** 2
        + (coords[:, 1] - cent[1]) ** 2
        + (coords[:, 2] - cent[2]) ** 2
    )  # calculate radii
    rs[rs < 0.1] = 0.4  # Set a lower bound on the "smoothing length"
    stars.smoothing_lengths = rs / 4  # convert radii into smoothing lengths
    stars.redshift = 1
    print(stars)

    # Compute width of stellar distribution
    width = np.max(coords) - np.min(coords)

    print("Stars created, took:", time.time() - stars_start)

    galaxy_start = time.time()

    # Create galaxy object
    galaxy = Galaxy(stars=stars)

    print("Galaxy created, took:", time.time() - galaxy_start)

    spectra_start = time.time()

    # Calculate the stars SEDs
    sed = galaxy.generate_intrinsic_particle_spectra(grid, sed_object=True)

    print("Spectra created, took:", time.time() - spectra_start)

    filter_start = time.time()

    # Define filter list
    filter_codes = [
        "JWST/NIRCam.F090W",
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
    hist_img = galaxy.make_image(
        resolution,
        fov=width,
        img_type="hist",
        sed=sed,
        filters=filters,
        kernel_func=quintic,
        rest_frame=False,
        cosmo=cosmo,
    )

    print("Histogram images made, took:", time.time() - img_start)
    img_start = time.time()

    # Get the image
    smooth_img = galaxy.make_image(
        resolution,
        fov=width,
        img_type="smoothed",
        sed=sed,
        filters=filters,
        kernel_func=quintic,
        rest_frame=False,
        cosmo=cosmo,
    )

    print("Smoothed images made, took:", time.time() - img_start)

    hist_imgs = hist_img.imgs
    smooth_imgs = smooth_img.imgs

    print("Sucessfuly made images for:", [key for key in hist_imgs])

    print("Total runtime (not including plotting):", time.time() - start)

    # Set up plot
    fig = plt.figure(figsize=(4 * len(filters), 4 * 2))
    gs = gridspec.GridSpec(2, len(filters))

    # Create top row
    axes = []
    for i in range(len(filters)):
        axes.append(fig.add_subplot(gs[0, i]))

    # Loop over images plotting them
    for ax, fcode in zip(axes, filter_codes):
        ax.imshow(hist_imgs[fcode])
        ax.set_title(fcode)

    # Set y axis label on left most plot
    axes[0].set_ylabel("Histogram")

    # Create bottom row
    axes = []
    for i in range(len(filters)):
        axes.append(fig.add_subplot(gs[1, i]))

    # Loop over images plotting them
    for ax, fcode in zip(axes, filter_codes):
        ax.imshow(smooth_imgs[fcode])

    # Set y axis label on left most plot
    axes[0].set_ylabel("Smoothed")

    # Plot the image
    plt.savefig("../flux_in_filters_test.png", bbox_inches="tight", dpi=300)
