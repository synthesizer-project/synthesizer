"""
This example generates a sample of star particles from a 2D SFZH and then
generates an image of the mass distribution.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr, kpc

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.galaxy.particle import ParticleGalaxy as Galaxy
from synthesizer.particle.particles import CoordinateGenerator


# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6.0, 10.5, 0.1)
metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
Z_p = {"Z": 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation
sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)

# Create stars object
N = 10000  # number of particles for sampling
coords = CoordinateGenerator.generate_3D_gaussian(N)
stars = sample_sfhz(sfzh, N)
stars.coordinates = coords
stars.coord_units = kpc
print(stars)

# Create galaxy object
galaxy = Galaxy(stars=stars)

# Define image properties
resolution = 0.05 * kpc

# Get the image
img = galaxy.create_stellarmass_hist(resolution, npix=100)

# Plot the image
plt.imshow(img)
plt.savefig(script_path + "/plots/stellarmass_test.png")
