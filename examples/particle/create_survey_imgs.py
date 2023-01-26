"""
This example shows how to create a survey of fake galaxies generated using a
2D SFZH, and make images of each of these galaxies.
"""
import os
import time
import numpy as np
from scipy import signal
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from unyt import yr, Myr, kpc, arcsec

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.galaxy.particle import ParticleGalaxy as Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import FilterCollection as Filters
from synthesizer.kernel_functions import quintic
from synthesizer.imaging.survey import Survey
from astropy.cosmology import Planck18 as cosmo

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Set the seed
np.random.seed(42)

# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = script_path + "/../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Create an empty Survey object
survey = Survey(super_resolution_factor=2, fov=8)

# Lets make filter sets for two different instruments
hst_filter_codes = ["HST/WFC3_IR.F105W", "HST/WFC3_IR.F125W"]
webb_filter_codes = ["JWST/NIRCam.F090W", "JWST/NIRCam.F150W",
                     "JWST/NIRCam.F200W"]
hst_filters = Filters(hst_filter_codes, new_lam=grid.lam)
webb_filters = Filters(webb_filter_codes, new_lam=grid.lam)

# Create a fake PSF for each instrument (normalising the kernels)
hst_psf = np.outer(signal.windows.gaussian(25, 2),
                   signal.windows.gaussian(25, 2))
hst_psf /= np.sum(hst_psf)
webb_psf = np.outer(signal.windows.gaussian(50, 3),
                    signal.windows.gaussian(50, 3))
webb_psf /= np.sum(webb_psf)
hst_psfs = {f: hst_psf for f in hst_filters.filter_codes}
webb_psfs = {f: webb_psf for f in webb_filters.filter_codes}

# Lets define some depths in magnitudes
hst_depths = {f: 33.0 for f in hst_filters.filter_codes}
webb_depths = {f: 33.0 for f in webb_filters.filter_codes}

# Let's add these instruments to the survey
survey.add_photometric_instrument(filters=hst_filters, resolution=0.1 * arcsec,
                                  label="HST/WFC3_IR", psfs=hst_psfs,
                                  depths=hst_depths, snrs=5, apertures=0.5)
survey.add_photometric_instrument(filters=webb_filters, resolution=0.05 * arcsec,
                                  label="JWST/NIRCam", psfs=webb_psfs,
                                  depths=webb_depths, snrs=5, apertures=0.5)

# We need to convert the our depths into flux to be consistent with the images.
survey.convert_mag_depth_to_fnu()

# Define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6., 10.5, 0.1)
metallicities = 10**np.arange(-5., -1.5, 0.1)
Z_p = {'Z': 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {'duration': 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation

# Define a FOV to be updated by the particle distribution
fov = survey.fov

# Make some fake galaxies
ngalaxies = 4
galaxies = []
for igal in range(ngalaxies):

    # Generate the star formation metallicity history
    sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)

    # Create stars object
    n = 10000
    coords = CoordinateGenerator.generate_3D_gaussian(n)
    stars = sample_sfhz(sfzh, n)
    stars.coordinates = coords
    stars.coord_units = kpc
    cent = np.mean(coords, axis=0)  # define geometric centre
    rs = np.sqrt((coords[:, 0] - cent[0]) ** 2
                 + (coords[:, 1] - cent[1]) ** 2
                 + (coords[:, 2] - cent[2]) ** 2)  # calculate radii
    rs[rs < 0.1] = 0.4  # Set a lower bound on the "smoothing length"
    stars.smoothing_lengths = rs / 4  # convert radii into smoothing lengths
    stars.redshift = 1

    # Compute width of stellar distribution
    width = np.max(coords) - np.min(coords)

    # Update the FOV
    if width > fov:
        fov = width

    # Create galaxy object
    galaxy = Galaxy("Galaxy%d" % igal, stars=stars, redshift=1)

    # Calculate the SEDs of stars in this galaxy
    galaxy.generate_intrinsic_spectra(grid, update=True, integrated=False)

    # Include this galaxy
    galaxies.append(galaxy)

# Define image propertys
redshift = 1
fov = (width + 1) * cosmo.arcsec_per_kpc_proper(redshift).value * arcsec

# Set the fov in the survey
print("Image FOV:", fov)
survey.fov = fov

# Store galaxies in the survey
survey.add_galaxies(galaxies)

# Make images for each galaxy in this survey
survey.make_images(img_type="smoothed", spectra_type="intrinsic",
                   kernel_func=quintic, rest_frame=False, cosmo=cosmo)

# Set up plot
fig = plt.figure(figsize=(3.5 * survey.nfilters, 3.5 * survey.ngalaxies))
gs = gridspec.GridSpec(survey.ngalaxies, survey.nfilters)

# Create top row
axes = np.empty((survey.ngalaxies, survey.nfilters), dtype=object)
for i in range(survey.ngalaxies):
    for j in range(survey.nfilters):
        axes[i, j] = fig.add_subplot(gs[i, j])

# Create a mask for which plots are populated
populated = np.zeros((survey.ngalaxies, survey.nfilters))

# Loop over instruments
for inst in survey.imgs:

    # Loop over galaxies
    for i, img in enumerate(survey.imgs[inst]):

        # Find the next cell in this row
        j = 0
        while populated[i, j] != 0:
            j += 1

        # Label the edge
        if j == 0:
            axes[i, j].set_ylabel(survey.galaxies[i].name)

        # Loop over filters in this instrument
        for fcode in img.imgs:
            axes[i, j].imshow(img.imgs_noise[fcode], cmap="Greys_r")

            # Label the top row
            if i == 0:
                axes[i, j].set_title(fcode)

            # Record that we put an image here
            populated[i, j] = 1
            j += 1

# Plot the image
plt.savefig("../survey_img_test.png",
            bbox_inches="tight", dpi=300)
