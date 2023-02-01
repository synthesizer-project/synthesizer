"""
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
from unyt import yr, Myr

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.galaxy.particle import ParticleGalaxy as Galaxy
from synthesizer.particle.particles import CoordinateGenerator
from synthesizer.filters import FilterCollection as Filters
from synthesizer.imaging.survey import Survey

plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Set the seed
np.random.seed(42)
random.seed(42)

start = time.time()

# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = script_path + "/../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6., 10.5, 0.1)
metallicities = 10**np.arange(-5., -1.5, 0.1)
Z_p = {'Z': 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {'duration': 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation

# Generate the star formation metallicity history
sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)

# Set up plot
fig = plt.figure(figsize=(3.5, 3.5))
ax = fig.add_subplot(111)
ax.grid(True)
ax.loglog()

# Define the number of particles in a galaxy
nparts = [10, 100, 1000, 10000, 100000]

# Lets make filter sets for two different instruments
hst_filter_codes = ["HST/WFC3_IR.F105W", "HST/WFC3_IR.F125W"]
webb_filter_codes = ["JWST/NIRCam.F090W", "JWST/NIRCam.F150W",
                     "JWST/NIRCam.F200W"]
hst_filters = Filters(hst_filter_codes, new_lam=grid.lam)
webb_filters = Filters(webb_filter_codes, new_lam=grid.lam)

# Loop over nparts
for npart in nparts:

    # Define the number of galaxies array and timings results array
    gal_res = 5
    ngals = np.logspace(1, 3, gal_res, dtype=int)
    times = np.zeros(gal_res)

    # Create stars object
    n = npart
    stars = sample_sfhz(sfzh, n)

    # Loop over n galaxies
    for i, ngalaxies in enumerate(ngals):

        # Create an empty Survey object
        survey = Survey(super_resolution_factor=1)

        # Let's add these instruments to the survey
        survey.add_photometric_instrument(filters=hst_filters,
                                          label="HST/WFC3_IR")
        survey.add_photometric_instrument(filters=webb_filters,
                                          label="JWST/NIRCam")

        # Create galaxy object
        galaxy = Galaxy("Galaxy%d" % 0, stars=stars, redshift=1)

        # Make some fake galaxiesx
        spec_time = 0
        galaxies = np.empty(ngalaxies, dtype=object)
        for igal in range(ngalaxies):

            # Calculate the SEDs of stars in this galaxy
            start = time.time()
            galaxy.generate_intrinsic_spectra(grid, update=True,
                                              integrated=True)
            spec_time += time.time() - start

            # Include this galaxy
            galaxies[igal] = galaxy

        # Store galaxies in the survey
        survey.add_galaxies(galaxies)

        # Make images for each galaxy in this survey
        start = time.time()
        survey.get_photometry(spectra_type="intrinsic")

        times[i] = spec_time + time.time() - start
        print("Completed N_galaxies=%d with N_part=%d in %.2f" % (len(galaxies),
                                                                  npart,
                                                                  times[i]))

    # Plot the scatter for this filter
    ax.plot(ngals, times, marker=".",
            label="$N_\star=10^{%d}$" % int(np.log10(npart)))

# Label axes
ax.set_ylabel("Wallclock / [s]")
ax.set_xlabel("$N_\mathrm{gal}$")

ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=2)

# Plot the image
plt.savefig("../photometry_scaling_test.png",
            bbox_inches="tight", dpi=300)
