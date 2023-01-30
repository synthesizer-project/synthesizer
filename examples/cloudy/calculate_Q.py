
""" Tests calculate_Q including doing some performance testing compared to earlier functions """


from synthesizer.cloudy import measure_Q
import timeit
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

N = 10  # number of iterations for timing


name_func = [
    ('measure_Q, HI', measure_Q, sed.lam, llam.to('erg/s/angstrom').value, {}),
    ('calculate_Q, HI', calculate_Q, lam, lnu, {'ionisation_energy': 13.6 * eV}),
    ('calculate_Q, HI - no units', calculate_Q, sed.lam, sed.lnu, {'ionisation_energy': 13.6 * eV}),
    ('calculate_Q, HeII', calculate_Q, lam, lnu, {'ionisation_energy': 54.4 * eV}),
]


for name, func, lam, l, kwargs in name_func:

    print(name, '-'*10)
    print(f'value: {np.log10(func(lam, l, **kwargs))}')

    def f():
        func(lam, l, **kwargs)

    print(f'time: {timeit.timeit(f, number=N)/N}')
