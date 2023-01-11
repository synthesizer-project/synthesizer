

# Create a model SED


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synthesizer.grid import Grid

import flare.plt as fplt


log10Z = -2.
log10age = 6.0

sps_name = 'bpass-v2.2.1_chab100-bin_cloudy-v17.0_logUref-2'

fesc_LyA = 0.5

grid = Grid(sps_name)


iZ, log10Z = grid.get_nearest_log10Z(log10Z)
print(f'closest metallicity: {log10Z:.2f}')
ia, log10age = grid.get_nearest_log10age(log10age)
print(f'closest age: {log10age:.2f}')


fig = plt.figure(figsize = (3.5, 5.))

left  = 0.15
height = 0.8
bottom = 0.1
width = 0.8

ax = fig.add_axes((left, bottom, width, height))

Lnu = grid.spectra['linecont'][iZ, ia, :]
ax.plot(np.log10(grid.lam), np.log10(Lnu), lw=2, alpha = 0.3, c='k')


idx = grid.get_nearest_index(1216., grid.lam)
Lnu[idx] *= fesc_LyA

ax.plot(np.log10(grid.lam), np.log10(Lnu), lw=1, alpha = 1, c='k')


ax.set_xlim([2.8, 3.6])
ax.set_ylim([18., 23])
ax.legend(fontsize = 8, labelspacing = 0.0)
ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')

plt.show()
