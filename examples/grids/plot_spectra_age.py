# Create a model SED


import flare.plt as fplt
from synthesizer.sed import convert_fnu_to_flam
from synthesizer.grid import Grid
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cmasher as cmr

import sys
import os

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


norm = mpl.colors.Normalize(vmin=5.0, vmax=11.0)
cmap = cmr.bubblegum


# -------------------------------------------------
# --- define choise of SPS model and initial mass function (IMF)


def plot_spectra_age(grid, log10Z=-2.0, spec_name="stellar"):

    iZ, log10Z = grid.get_nearest_log10Z(log10Z)
    print(f"closest metallicity: {log10Z:.2f}")

    fig = plt.figure(figsize=(3.5, 5.0))

    left = 0.15
    height = 0.8
    bottom = 0.1
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    cax = fig.add_axes((left, bottom + height, width, 0.02))

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation="horizontal",
    )  # add the colourbar
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position("top")
    cax.set_xlabel(r"$\rm \log_{10}(age/yr)$")

    for ia, log10age in enumerate(grid.log10ages):
        Lnu = grid.spectra[spec_name][ia, iZ, :]
        # Lnu = convert_fnu_to_flam(grid.lam, Lnu)
        ax.plot(
            np.log10(grid.lam),
            np.log10(Lnu),
            c=cmap(norm(log10age)),
            lw=1,
            alpha=0.8,
        )

    for wv in [912.0, 3646.0]:
        ax.axvline(np.log10(wv), c="k", lw=1, alpha=0.5)

    ax.set_xlim([2.0, 4.0])
    ax.set_ylim([10.0, 22])
    ax.legend(fontsize=5, labelspacing=0.0)
    ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
    ax.set_ylabel(
        r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$"
    )

    return fig, ax


if __name__ == "__main__":

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"

    grid = Grid(grid_name, grid_dir=grid_dir)
    log10Z = -2.0

    fig, ax = plot_spectra_age(grid, log10Z=log10Z)
    # plt.show()
    # fig.savefig(f'figs/spectra_age_{sps_name}.pdf')
