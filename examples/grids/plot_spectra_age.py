

# Create a model SED


import flare.plt as fplt
from synthesizer.sed import convert_fnu_to_flam
from synthesizer.grid import SpectralGrid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


norm = mpl.colors.Normalize(vmin=5., vmax=11.)
cmap = cmr.bubblegum


# -------------------------------------------------
# --- define choise of SPS model and initial mass function (IMF)


def plot_spectra_age(grid, log10Z=-2.0, spec_name='stellar'):

    iZ, log10Z = grid.get_nearest_log10Z(log10Z)
    print(f'closest metallicity: {log10Z:.2f}')

    fig = plt.figure(figsize=(3.5, 5.))

    left = 0.15
    height = 0.8
    bottom = 0.1
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    cax = fig.add_axes((left, bottom+height, width, 0.02))

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax,
                        orientation='horizontal')  # add the colourbar
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    cax.set_xlabel(r'$\rm \log_{10}(age/yr)$')

    for ia, log10age in enumerate(grid.log10ages):
        Lnu = grid.spectra[spec_name][ia, iZ, :]
        # Lnu = convert_fnu_to_flam(grid.lam, Lnu)
        ax.plot(np.log10(grid.lam), np.log10(Lnu), c=cmap(norm(log10age)), lw=1, alpha=0.8)

    for wv in [912., 3646.]:
        ax.axvline(np.log10(wv), c='k', lw=1, alpha=0.5)

    ax.set_xlim([2., 4.])
    ax.set_ylim([10., 22])
    ax.legend(fontsize=5, labelspacing=0.0)
    ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
    ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')

    return fig, ax


if __name__ == '__main__':

    log10Z = -2.
    sps_names = [
        # 'bpass-v2.2.1-bin_chab-100',
        # 'fsps-v3.2_Chabrier03',
        # 'bc03-2016-BaSeL_chabrier03',
        # 'bc03-2016-Miles_chabrier03',
        # 'bc03-2016-Stelib_chabrier03',
        f'fsps-v3.2_imf3:2.3',
        # f'fsps-v3.2_Salpeter',
    ]

    for sps_name in sps_names:

        grid = SpectralGrid(sps_name)

        fig, ax = plot_spectra_age(grid, log10Z=log10Z)

        fig.savefig(f'figs/spectra_age_{sps_name}.pdf')
