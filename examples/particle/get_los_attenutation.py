"""
This example shows how to compute line of sight dust surface densities.
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
from synthesizer.particle.gas import Gas
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.particles import CoordinateGenerator
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

# Define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6.0, 10.5, 0.1)
metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)
Z_p = {"Z": 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation

# Generate the star formation metallicity history
mass = 10**10
sfzh = generate_sfzh(log10ages, metallicities, sfh, Zh, stellar_mass=mass)

for n, ngas in zip([10, 100, 1000] * 3,
                   [100, 100, 100,
                    1000, 1000, 1000,
                    10000, 10000, 10000]):

    # Make a fake galaxy

    # First make the stars

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

    # Now make the gas

    # Generate some random coordinates
    coords = CoordinateGenerator.generate_3D_gaussian(ngas)

    # Calculate the smoothing lengths from radii
    cent = np.mean(coords, axis=0)
    rs = np.sqrt(
        (coords[:, 0] - cent[0]) ** 2
        + (coords[:, 1] - cent[1]) ** 2
        + (coords[:, 2] - cent[2]) ** 2
    )
    rs[rs < 0.2] = 0.6  # Set a lower bound on the "smoothing length"

    gas = Gas(masses=np.random.uniform(10**6, 10**6.5, ngas),
              metallicities=np.random.uniform(0.01, 0.05, ngas),
              coordinates=coords,
              smoothing_lengths=rs / 4,
              dust_to_metal_ratio=0.2)

    # Create galaxy object
    galaxy = Galaxy("Galaxy", stars=stars, gas=gas, redshift=1)

    # Create a fake kernel
    kernel = np.random.normal(0.5, 0.25, 100)
    kernel = np.sort(kernel)[::-1]
    kernel /= np.sum(kernel)

    # Calculate the tau_vs
    start = time.time()
    tau_v = galaxy.calculate_los_tau_v(kappa=0.5, kernel=kernel)
    print("Calculation took %.2f seconds for nstar=%d and ngas=%d" % (
        time.time() - start, n, ngas))
