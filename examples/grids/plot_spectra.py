

from synthesizer.grid import Grid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import os
import sys


if __name__ == '__main__':

    """
    This example demonstrates how to extract a spectra directly from a grid and plots all the available spectra.
    """


    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"

    # initialise grid
    grid = Grid(grid_name, grid_dir=grid_dir)

    # choose age and metallicity
    log10age = 6.0  # log10(age/yr)
    Z = 0.01  # metallicity

    # get the grid point for this log10age and metallicity
    grid_point = grid.get_grid_point((log10age, Z))

    # loop over available spectra and plot
    for spec_name in grid.spec_names:
        # get Sed object
        sed = grid.get_sed(grid_point, spec_name=spec_name)
        print(sed)
        plt.plot(np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=spec_name)

    plt.xlim([2., 4.])
    plt.ylim([18., 23])
    plt.legend(fontsize=8, labelspacing=0.0)
    plt.xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    plt.ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')
    plt.show()
