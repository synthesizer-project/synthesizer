"""
This example generates a sample of star particles from a 2D SFZH, generates an
SED for each particle and then generates images in a number of Webb bands.
"""
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec 
from unyt import yr, Myr

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.galaxy.particle import ParticleGalaxy as Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import FilterCollection as Filters
from synthesizer.kernel_functions import quintic
from astropy.cosmology import Planck18 as cosmo

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

if __name__ == '__main__':

    # Set the seed
    np.random.seed(42)

    start = time.time()

    # Define the grid
    grid_name = 'test_grid'
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

    print("SFHZ sampled, took:", time.time() - start)

    stars_start = time.time()

    # Create stars object
    n = 10000  # number of particles for sampling
    coords = CoordinateGenerator.generate_3D_gaussian(n)
    stars = sample_sfhz(sfzh, n)
    stars.coordinates = coords
    cent = np.mean(coords, axis=0)  # define geometric centre
    rs = np.sqrt((coords[:, 0] - cent[0]) ** 2
                 + (coords[:, 1] - cent[1]) ** 2
                 + (coords[:, 2] - cent[2]) ** 2)  # calculate radii
    rs[rs < 0.1] = 0.4  # Set a lower bound on the "smoothing length"
    stars.smoothing_lengths = rs / 4  # convert radii into smoothing lengths
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
    galaxy.generate_intrinsic_spectra(grid, update=True, integrated=False)

    print("Spectra created, took:", time.time() - spectra_start)

    filter_start = time.time()

    # Define filter list
    filter_codes = ["JWST/NIRCam.F090W", "JWST/NIRCam.F150W", "JWST/NIRCam.F200W",
                    "JWST/NIRCam.F444W", "JWST/MIRI.F1000W", "JWST/MIRI.F1500W"]

    # Set up filter object
    filters = Filters(filter_codes, new_lam=grid.lam)

    print("Filters created, took:", time.time() - filter_start)

    img_start = time.time()

    # Define image propertys
    resolution = (width + 1) / 100
    redshift = 1

    # Get the image
    hist_img = galaxy.make_image(resolution, npix=None, fov=width + 1,
                                 img_type="hist",
                                 sed=galaxy.spectra_array["intrinsic"],
                                 survey=None, filters=filters, pixel_values=None,
                                 with_psf=False, with_noise=False,
                                 kernel_func=quintic, rest_frame=False,
                                 redshift=redshift, cosmo=cosmo, igm=None)

    print("Histogram images made, took:", time.time() - img_start)
    img_start = time.time()

    # Get the image
    smooth_img = galaxy.make_image(resolution, npix=None, fov=width + 1,
                                   img_type="smoothed",
                                   sed=galaxy.spectra_array["intrinsic"],
                                   survey=None, filters=filters, pixel_values=None,
                                   with_psf=False, with_noise=False,
                                   kernel_func=quintic, rest_frame=False,
                                   redshift=redshift, cosmo=cosmo, igm=None)

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
