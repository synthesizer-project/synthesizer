

import flare.plt as fplt
from synthesizer.grid import Grid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


if __name__ == '__main__':

    grid_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'
    grid_name = 'blackbody'

    # initialise grid
    grid = Grid(grid_name, grid_dir=grid_dir)

    # plot different spectra for a single grid point

    # choose age and metallicity
    log10T = 5.0  # log10(age/yr)
    log10Z = 0.01  # metallicity
    log10U = -2

    # get the grid point for this combination of parameters
    grid_point = grid.get_grid_point((log10T, log10Z, log10U))

    sed = grid.get_sed(grid_point, spec_name='total')
    normalisation = np.sum(sed.lnu)
    plt.plot(np.log10(sed.lam), np.log10(sed.lnu/normalisation),
             c='k', lw=2, alpha=1, label='TOTAL')

    # loop over available spectra and plot
    for spec_name in grid.spec_names:
        # get Sed object
        sed = grid.get_sed(grid_point, spec_name=spec_name)
        plt.plot(np.log10(sed.lam), np.log10(sed.lnu/normalisation),
                 lw=1, alpha=0.8, label=spec_name)

    plt.xlim([1., 4.])
    plt.ylim([-8, -2])
    plt.legend(fontsize=8, labelspacing=0.0)
    plt.xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')
    plt.show()

    # plot both total and incident for a range of temperatures

    log10Ts = [4., 4.5, 5., 5.5, 6.]

    log10Z = 0.01  # metallicity
    log10U = -3

    colours = cmr.take_cmap_colors('cmr.ember', len(log10Ts), cmap_range=(0.1, 1))

    for log10T, c in zip(log10Ts, colours):  # :

        # get the grid point for this combination of parameters
        grid_point = grid.get_grid_point((log10T, log10Z, log10U))

        sed = grid.get_sed(grid_point, spec_name='total')
        normalisation = np.sum(sed.lnu)
        plt.plot(np.log10(sed.lam), np.log10(sed.lnu/normalisation),
                 c=c, lw=1, alpha=1, zorder=2)

        sed = grid.get_sed(grid_point, spec_name='incident')
        plt.plot(np.log10(sed.lam), np.log10(sed.lnu/normalisation),
                 c=c, lw=3, alpha=0.2, zorder=1)

    plt.xlim([1., 4.])
    plt.ylim([-8, -2])
    plt.legend(fontsize=8, labelspacing=0.0)
    plt.xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')
    plt.show()
