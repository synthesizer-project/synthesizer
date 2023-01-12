# --- this example generates a sample of star particles from a 2D SFZH and then generates a spectral cube.


import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.particles import CoordinateGenerator


# --- define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6., 10.5, 0.1)
metallicities = 10**np.arange(-5., -1.5, 0.1)
Z_p = {'Z': 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {'duration': 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation
sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh)


# -----------------------------v
# --- create stars object

N = 1000  # number of particles for sampling

coords = CoordinateGenerator.generate_3D_gaussian(N)
stars = sample_sfhz(sfzh, N)
stars.coordinates = coords
print(stars)


# # --- open grid

# grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'
# grid = Grid(grid_name)

# --- create galaxy object

galaxy = Galaxy(stars=stars)

# Define image propertys
resolution = 0.1

cube = galaxy.create_stellarmass_hist(resolution, npix=100)

# image = cube.create_image(500) # create an image at wavelength = lam[500]
# image.make_image_plot(show=True)

# image = cube.create_image(1500.) # create an image at the wavelength closest to 1500\AA. I think it would be better to use quantities here
# image.make_image_plot(show=True)

# cube.animate() # create animation of the spectral grid
