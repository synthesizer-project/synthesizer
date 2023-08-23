"""
Compare parametric and particle SEDs
=====================================

This example compares a sampled and binned (parametric) SED for different numbers of particles
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.stars import sample_sfhz
from synthesizer.particle.stars import Stars
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy


# --- initialise the SPS grid
# grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'
# grid = Grid(grid_name)

# Get the location of this script, __file__ is the absolute path of this
# script, however we just want to directory
script_path = os.path.abspath(os.path.dirname(__file__))

# Define the grid
grid_name = "test_grid"
grid_dir = script_path + "/../../tests/test_grid/"
grid = Grid(grid_name, grid_dir=grid_dir)

# --- define the binned (parametric star formation history)

Z_p = {'Z': 0.01}
Zh = ZH.deltaConstant(Z_p)
sfh_p = {'duration': 100 * Myr}
sfh = SFH.Constant(sfh_p)  # constant star formation
sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)


# --------------------------------------------
# CREATE PARAMETRIC SED

parametric_galaxy = ParametricGalaxy(sfzh)
parametric_galaxy.get_spectra_stellar(grid)
sed = parametric_galaxy.spectra['stellar']
plt.plot(np.log10(sed.lam), np.log10(sed.lnu), label='parametric', lw=4, c='k', alpha=0.3)


# --------------------------------------------
# CREATE PARTICLE SED

for N in [1, 10, 100, 1000]:

    # --- create stars object
    stars = sample_sfhz(sfzh, N)
    # ensure that the total mass = 1 irrespective of N. This can be also acheived by setting the mass of the star particles in sample_sfhz but this will be easier most of the time.
    stars.renormalise_mass(1.)

    # --- create galaxy object
    particle_galaxy = ParticleGalaxy(stars=stars)

    # --- this generates stellar and intrinsic spectra
    # particle_galaxy.generate_spectra(grid, fesc=0.0, integrated=True)
    
    # Calculate the stars SEDs
    particle_galaxy.get_spectra_stellar(grid)


    sed = particle_galaxy.spectra['stellar']
    plt.plot(np.log10(sed.lam), np.log10(sed.lnu), label=f'particle (N={N})')


plt.legend()
plt.xlim([2, 5])
plt.ylim([10, 22])
plt.savefig(script_path + '/plots/compare_parametric_particle.png', dpi=200, bbox_inches='tight'); plt.close()
# plt.show()
