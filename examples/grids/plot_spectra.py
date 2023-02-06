

import flare.plt as fplt
from synthesizer.grid import Grid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def plot_spectra(grid, log10Z=-2.0, log10age=6.0, spec_names=None):
    """
    Plots spectra at the closest grid point to a choice of metallicity and log10age.


    """

    iZ, log10Z = grid.get_nearest_log10Z(log10Z)
    print(f'closest metallicity: {log10Z:.2f}')
    ia, log10age = grid.get_nearest_log10age(log10age)
    print(f'closest age: {log10age:.2f}')

    if not spec_names:
        spec_names = grid.spec_names

    fig = plt.figure(figsize=(3.5, 5.))

    left = 0.15
    height = 0.8
    bottom = 0.1
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    for spec_name in spec_names:
        Lnu = grid.spectra[spec_name][ia, iZ, :]
        ax.plot(np.log10(grid.lam), np.log10(Lnu),
                lw=1, alpha=0.8, label=spec_name)

    ax.set_xlim([2., 4.])
    ax.set_ylim([18., 23])
    ax.legend(fontsize=8, labelspacing=0.0)
    ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    ax.set_ylabel(
        r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')

    return fig, ax


if __name__ == '__main__':

    # -------------------------------------------------
    # --- define choise of SPS model and initial mass function (IMF)

    log10Z = -2.  # log10(metallicity)
    log10age = 6.0  # log10(age/yr)

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"

    # grid_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'
    # grid_name = 'bpass-2.2.1-bin_chabrier03-0.1,100.0_cloudy'
    # grid_name = 'bpass-2.2.1-bin_chabrier03-0.1,100.0'

    grid = Grid(grid_name, grid_dir=grid_dir)

    # fig, ax = plot_spectra(grid, log10Z = log10Z, log10age = log10age, spec_names = ['linecont'])
    fig, ax = plot_spectra(grid, log10Z=log10Z, log10age=log10age)

    # plt.show()
    # fig.savefig(f'figs/spectra_type_{sps_name}.pdf')
