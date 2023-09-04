"""
Create survey photometry
========================

This example shows how to create a survey of fake galaxies generated using a
2D SFZH, and make images of each of these galaxies.
"""
import os
import time
import random
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
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import FilterCollection as Filters
from synthesizer.kernel_functions import quintic
from synthesizer.survey import Survey
from astropy.cosmology import Planck18 as cosmo

plt.rcParams["font.family"] = "DeJavu Serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

# Set the seed
np.random.seed(45)

start = time.time()

# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
# script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

print("Grid dimensions: (%d, %d)" % (grid.log10age.size,
                                     grid.metallicity.size))

# What redshift are we testing?
redshift = 8

# Create an empty Survey object
survey = Survey(super_resolution_factor=1)

# Lets make filter sets for two different instruments
hst_filter_codes = ["HST/WFC3_IR.F105W", "HST/WFC3_IR.F125W"]
webb_filter_codes = [
    "JWST/NIRCam.F090W",
    "JWST/NIRCam.F150W",
    "JWST/NIRCam.F200W",
]
hst_filters = Filters(hst_filter_codes, new_lam=grid.lam)
webb_filters = Filters(webb_filter_codes, new_lam=grid.lam)

# Let's add these instruments to the survey
survey.add_photometric_instrument(filters=hst_filters, label="HST/WFC3_IR")
survey.add_photometric_instrument(filters=webb_filters, label="JWST/NIRCam")

# Define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6.0, 10.5, 0.1)
metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
Z_p = {"Z": 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation

# Make some fake galaxies
ngalaxies = 1000
galaxies = []
for igal in range(ngalaxies):
    # Generate the star formation metallicity history
    mass = np.random.uniform(10**6, 10**10, 1)[0]
    sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh, stellar_mass=mass)

    # Create stars object
    n = random.randint(100, 1000)

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
    stars = sample_sfhz(
        sfzh,
        n,
        coordinates=coords,
        current_masses=np.full(n, 10**8.7 / n),
        smoothing_lengths=rs / 2,
        redshift=1,
    )

    # Create galaxy object
    galaxy = Galaxy(name="Galaxy%d" % igal, stars=stars, redshift=1)

    # Include this galaxy
    galaxies.append(galaxy)

# Store galaxies in the survey
survey.add_galaxies(galaxies)

# Get the SEDs
survey.get_spectra(grid, spectra_type="incident", rest_frame=False, redshift=redshift)

# Make images for each galaxy in this survey
survey.get_photometry(spectra_type="incident")

print(
    "Total runtime (including creation, not including plotting):", time.time() - start
)

# Get stellar masses
ms = []
for gal in galaxies:
    ms.append(gal.stellar_mass)

# Set up plot
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.grid(True)
ax.loglog()

# Loop over filters
for f in survey.photometry:
    # Get photometry
    phot = survey.photometry[f]

    # Plot the scatter for this filter
    ax.scatter(ms, phot, marker=".", label=f)

ax.set_ylabel("$F /$ [nJy] ")
ax.set_xlabel("$M / \mathrm{M}_\odot$")

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    fancybox=True,
    shadow=True,
    ncol=2,
)

# Plot the image
# plt.savefig(script_path + "/plots/survey_photometry_test.png", bbox_inches="tight", dpi=300)
plt.show()
