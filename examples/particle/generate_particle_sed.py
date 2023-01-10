

import numpy as np
import matplotlib.pyplot as plt

from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy


initial_masses = np.array([1.]*5)/5
ages = np.array([1E6, 1E7, 1E8, 1E9, 1E10])
metallicities = np.array([0.01]*5)
tauVs = np.array([0.5]*5)

grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'

grid = Grid(grid_name)

galaxy = Galaxy()
galaxy.load_stars(initial_masses, ages, metallicities)


# --- this generates stellar and intrinsic spectra
galaxy.generate_intrinsic_spectra(grid, fesc=0.0) # calculate only integrated SEDs
galaxy.generate_intrinsic_spectra(grid, fesc=0.0, integrated = False) # calculates for every star particle, slow but necessary for LOS.

# --- generate dust screen
# galaxy.get_screen(0.5) # tauV

# --- generate CF00 variable dust screen
# galaxy.get_CF00(grid, 0.5, 0.5) # grid, tauV_BC, tauV_ISM

# --- generate for los model
galaxy.get_los(tauVs) # grid, tauV_BC, tauV_ISM


for sed_type, sed in galaxy.spectra.items():
    print(sed_type, sed, sed.lnu.shape)
    plt.plot(np.log10(sed.lam), np.log10(sed.lnu), label = sed_type)

plt.legend()
plt.xlim([2,5])
plt.ylim([10,22])
plt.show()
