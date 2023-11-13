"""
Create sampled SED
==================

this example generates a sample of star particles from a 2D SFZH. In this case it is generated from a parametric star formation history with constant star formation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr

from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy


# --- define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6.0, 10.5, 0.1)
metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)

# --- define the parameters of the star formation and metal enrichment histories

Z_p = {"Z": 0.01}
metal_dist = ZDist.DeltaConstant(**Z_p)

sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(**sfh_p)  # constant star formation
sfzh = generate_sfzh(log10ages, metallicities, sfh, metal_dist)
print(sfzh)
# sfzh.plot()


# --- create stars object

N = 100  # number of particles for sampling
stars = sample_sfhz(sfzh, N)


# --- open grid

# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
# script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# --- create galaxy object

galaxy = Galaxy(stars=stars)

# --- this generates stellar and intrinsic spectra
# galaxy.generate_intrinsic_spectra(grid, fesc=0.0) # calculate only integrated SEDs
# calculates for every star particle, slow but necessary for LOS.
galaxy.stars.get_spectra_incident(grid)

# --- generate dust screen
# galaxy.get_screen(0.5) # tau_v

# --- generate CF00 variable dust screen
# galaxy.get_CF00(grid, 0.5, 0.5) # grid, tau_v_BC, tau_v_ISM

# --- generate for los model
# TODO: to be implemented
# tau_vs = np.ones(N) * 0.5
# galaxy.get_los(tau_vs)  # grid, tau_v_BC, tau_v_ISM

for sed_type, sed in galaxy.spectra.items():
    plt.plot(np.log10(sed.lam), np.log10(sed.lnu), label=sed_type)

plt.legend()
plt.xlim([2, 5])
plt.ylim([10, 24])
plt.show()
