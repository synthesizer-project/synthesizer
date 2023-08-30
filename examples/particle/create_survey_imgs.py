"""
Create survey images
====================

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
from unyt import yr, Myr, kpc, arcsec, Mpc

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import FilterCollection as Filters
from synthesizer.kernel_functions import quintic
from synthesizer.survey import Survey
from astropy.cosmology import Planck18 as cosmo

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

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

# What redshift are we testing?
redshift = 4

# Create an empty Survey object
survey = Survey(super_resolution_factor=2, fov=15)

# Lets make filter sets for two different instruments
hst_filter_codes = ["HST/WFC3_IR.F105W", "HST/WFC3_IR.F125W"]
webb_filter_codes = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
]
hst_filters = Filters(hst_filter_codes, new_lam=grid.lam)
webb_filters = Filters(webb_filter_codes, new_lam=grid.lam)

# Create a fake PSF for each instrument (normalising the kernels)
hst_psf = np.outer(
    signal.windows.gaussian(25, 4), signal.windows.gaussian(25, 5)
)
hst_psf /= np.sum(hst_psf)
webb_psf = np.outer(
    signal.windows.gaussian(50, 6), signal.windows.gaussian(50, 6)
)
webb_psf /= np.sum(webb_psf)
hst_psfs = {f: hst_psf for f in hst_filters.filter_codes}
webb_psfs = {f: webb_psf for f in webb_filters.filter_codes}

# Lets define some depths in magnitudes
hst_depths = {f: 46.0 for f in hst_filters.filter_codes}
webb_depths = {f: 46.0 for f in webb_filters.filter_codes}

# Let's add these instruments to the survey
survey.add_photometric_instrument(
    filters=hst_filters,
    resolution=0.5 * Mpc,
    label="HST/WFC3_IR",
    psfs=hst_psfs,
    depths=hst_depths,
    snrs=5,
    apertures=3,
)
survey.add_photometric_instrument(
    filters=webb_filters,
    resolution=0.1 * Mpc,
    label="JWST/NIRCam",
    psfs=webb_psfs,
    depths=webb_depths,
    snrs=5,
    apertures=3,
)

# We need to convert the our depths into flux to be consistent with the images.
survey.convert_mag_depth_to_lnu(redshift)

# Define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6.0, 10.5, 0.1)
metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
Z_p = {"Z": 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation

# Define a FOV to be updated by the particle distribution
fov = survey.fov

# Make some fake galaxies
ngalaxies = 5
galaxies = []
for igal in range(ngalaxies):

    # Generate the star formation metallicity history
    mass = 10**10
    sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh, stellar_mass=mass)

    # Define the number of stars
    n = 1000
    
    # Generate some random coordinates
    coords = CoordinateGenerator.generate_3D_gaussian(n)

    # Calculate the smoothing lengths from radii
    cent = np.mean(coords, axis=0)
    rs = np.sqrt(
            (coords[:, 0] - cent[0]) ** 2
            + (coords[:, 1] - cent[1]) ** 2
            + (coords[:, 2] - cent[2]) ** 2
    )
    rs[rs < 0.2] = 0.6  # Set a lower bound on the "smoothing length"

    # Sample the SFZH, producing a Stars object
    # we will also pass some keyword arguments for attributes
    # we will need for imaging
    stars = sample_sfhz(sfzh, n, coordinates=coords, 
                        current_masses=np.full(n, 10**8.7 / n), 
                        smoothing_lengths=rs / 2, redshift=1)

    # Compute width of stellar distribution
    width = (np.max(coords) - np.min(coords)) * Mpc

    # Update the FOV
    if width > fov:
        fov = width

    # Create galaxy object
    galaxy = Galaxy("Galaxy%d" % igal, stars=stars, redshift=1)

    # Include this galaxy
    galaxies.append(galaxy)

# Define image properties
fov = (width.value + 1) * Mpc

# Set the fov in the survey
print("Image FOV:", fov)
survey.fov = fov

# Store galaxies in the survey
survey.add_galaxies(galaxies)

# Calculate the SEDs
survey.get_particle_spectra(grid, "incident", redshift=redshift, rest_frame=False)

# Make images for each galaxy in this survey
survey.make_images(
    img_type="smoothed",
    spectra_type="incident",
    kernel_func=quintic,
    rest_frame=False,
    cosmo=cosmo,
)

print("Total runtime (including creation, not including plotting):",
      time.time() - start)

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
            print(img.imgs_noise[fcode].max())
            print(img.noise_arrs[fcode].max())

            # Label the top row
            if i == 0:
                axes[i, j].set_title(fcode)

            # Record that we put an image here
            populated[i, j] = 1
            j += 1

# Plot the image
plt.savefig(script_path + "/plots/survey_img_test.png", bbox_inches="tight", dpi=300)
