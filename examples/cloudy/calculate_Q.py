

from synthesizer.sed import convert_fnu_to_flam, calculate_Q
from synthesizer.grid import Grid

from unyt import c, h, angstrom, erg, s, Hz, eV, unyt_array

import numpy as np


grid_dir = '../../tests/test_grid'
grid_name = 'test_grid'

grid = Grid(grid_name, grid_dir=grid_dir)

sed = grid.get_sed(0, 0)  # get sed for the lowest metallicity / youngest grid point

lnu = sed.lnu * erg/s/Hz
lam = sed.lam * angstrom

llam = lnu * c / lam**2

print(calculate_Q(lam, lnu))  # by default calculate Q for HI


# these two function are deprecated in favour of the above
# from synthesizer.cloudy import measure_Q
# from synthesizer.sed import calculate_Q_deprecated

# print(calculate_Q_deprecated(lam.to('angstrom').value, lnu.to('erg/s/Hz').value)) # deprecated
# print(measure_Q(lam, llam.to('erg/s/angstrom').value)) # deprecated

ionisation_energy = 54.4 * eV
print(calculate_Q(lam, lnu, ionisation_energy=ionisation_energy))  # by default calculate Q for HI
