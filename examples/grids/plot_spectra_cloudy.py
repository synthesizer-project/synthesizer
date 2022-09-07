

# Create a model SED


import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synthesizer.grid_sw import Grid

import flare.plt as fplt




def plot_spectra(grid, log10Z = -2.0, log10age = 6.0, spec_names = None):

    iZ, log10Z = grid.get_nearest_log10Z(log10Z)
    print(f'closest metallicity: {log10Z:.2f}')
    ia, log10age = grid.get_nearest_log10age(log10age)
    print(f'closest age: {log10age:.2f}')


    if not spec_names: spec_names = grid.spec_names

    fig = plt.figure(figsize = (3.5, 5.))

    left  = 0.15
    height = 0.8
    bottom = 0.1
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    for spec_name in spec_names:
        Lnu = grid.spectra[spec_name][iZ, ia, :]
        ax.plot(np.log10(grid.lam), np.log10(Lnu), lw=1, alpha = 0.8, label = spec_name)


    ax.set_xlim([2., 4.])
    ax.set_ylim([18., 23])
    ax.legend(fontsize = 8, labelspacing = 0.0)
    ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')

    return fig, ax


if __name__ == '__main__':

    # -------------------------------------------------
    # --- define choise of SPS model and initial mass function (IMF)

    sps_names = ['bpass-v2.2.1_chab100-bin', 'bpass-v2.2.1_chab100-bin_cloudy-v17.0_logUref-2']

    log10Z = -2. # log10(metallicity)
    log10age = 6.0 # log10(age/yr)



    for sps_name in sps_names:

        grid = Grid(sps_name)

        # fig, ax = plot_spectra(grid, log10Z = log10Z, log10age = log10age, spec_names = ['linecont'])
        fig, ax = plot_spectra(grid, log10Z = log10Z, log10age = log10age)

        fig.savefig(f'figs/spectra_type_{sps_name}.pdf')
