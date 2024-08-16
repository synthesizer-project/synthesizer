"""
Create sampled SED
==================

this example generates a sample of star particles from a 2D SFZH.
In this case it is generated from a parametric star formation history
with constant star formation.
"""

import matplotlib.pyplot as plt
import numpy as np
from unyt import Myr

from synthesizer.emission_models import IncidentEmission
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.stars import sample_sfhz

# Define the grid
grid_name = "test_grid"
grid_dir = "../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# Define the emission model
model = IncidentEmission(grid)

# define the grid (normally this would be defined by an SPS grid)
log10ages = np.arange(6.0, 10.5, 0.1)
metallicities = 10 ** np.arange(-5.0, -1.5, 0.1)

# define the parameters of the star formation and metal enrichment histories

Z_p = {"metallicity": 0.01}
metal_dist = ZDist.DeltaConstant(**Z_p)

sfh_p = {"duration": 100 * Myr}
sfh = SFH.Constant(**sfh_p)  # constant star formation
sfzh = ParametricStars(
    log10ages, metallicities, sf_hist=sfh, metal_dist=metal_dist
)
print(sfzh)
# sfzh.plot()


# --- create stars object

N = 100  # number of particles for sampling
stars = sample_sfhz(sfzh.sfzh, sfzh.log10ages, sfzh.log10metallicities, N)

# --- create galaxy object

galaxy = Galaxy(stars=stars)

""" this generates stellar and intrinsic spectra
galaxy.generate_intrinsic_spectra(grid, fesc=0.0)
calculate only integrated SEDs """
galaxy.stars.get_spectra(model)

# --- generate dust screen
# galaxy.get_screen(0.5) # tau_v

# --- generate CF00 variable dust screen
# galaxy.get_CF00(grid, 0.5, 0.5) # grid, tau_v_BC, tau_v_ISM

# --- generate for los model
# TODO: to be implemented
# tau_vs = np.ones(N) * 0.5
# galaxy.get_los(tau_vs)  # grid, tau_v_BC, tau_v_ISM

for sed_type, sed in galaxy.stars.spectra.items():
    plt.plot(np.log10(sed.lam), np.log10(sed.lnu), label=sed_type)

plt.legend()
plt.xlim([2, 5])
plt.ylim([10, 24])
plt.show()
