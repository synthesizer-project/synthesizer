"""
Plot quiescent UVJ diagram
==========================

"""

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from synthesizer.filters import UVJ
from synthesizer.grid import Grid


def simple_UVJ(grid, target_metallicity=0.01):
    """Calculate UVJ colours as a function of age for single metallicity.

    Args:
        grid (Grid): The grid object.
        target_metallicity (float): The metallicity to use for the calculation.
    """
    iZ = grid.get_nearest_index(target_metallicity, grid.metallicity)

    fc = UVJ(new_lam=grid.lam)
    # fc.plot_transmission_curves()

    for ia, log10age in enumerate(grid.log10age):
        sed = grid.get_sed_at_grid_point(
            (ia, iZ)
        )  # creates an SED object from a given grid point

        # --- now calculate the observed frame spectra

        sed.get_fnu0()  # generate dummy observed frame spectra.

        # --- measure broadband fluxes
        sed.get_photo_fnu(fc)

        print(
            (
                f"log10(age/Myr): {log10age - 6:.1f} "
                f"U-V: {sed.measure_colour('U', 'V'):.2f} "
                f"V-J: {sed.measure_colour('V', 'J'):.2f}"
            )
        )


def UVJ_metallicity(grid):
    """Calculate UVJ as a function of metallicity and save as a .ecsv file.

    Args:
        grid (Grid): The grid object.
    """
    fc = UVJ(new_lam=grid.lam)

    table = Table()
    table.meta["metallicities"] = list(grid.metallicity)
    table["log10ages"] = grid.log10age

    for iZ, Z in enumerate(grid.metallicity):
        for f in "UVJ":
            table[f"{Z}_{f}"] = np.zeros(len(grid.log10age))

        for ia, log10age in enumerate(grid.log10age):
            sed = grid.get_sed_at_grid_point(
                (ia, iZ)
            )  # creates an SED object from a given grid point

            # --- now calculate the observed frame spectra

            sed.get_fnu0()  # generate dummy observed frame spectra.

            # --- measure broadband fluxes
            sed.get_photo_fnu(fc)

            for f in "UVJ":
                table[f"{Z}_{f}"][ia] = sed.photo_fnu[f]

    # --- make plot

    fig, axes = plt.subplots(
        2, 1, figsize=(3.5, 4.5), sharex=True, sharey=True
    )
    plt.subplots_adjust(
        left=0.15, top=0.975, bottom=0.1, right=0.95, wspace=0.0, hspace=0.0
    )

    colors = cmr.take_cmap_colors("cmr.bubblegum", len(grid.metallicity))

    for Z, c in zip(grid.metallicity, colors):
        x = table["log10ages"] - 6.0

        for i, (f1, f2) in enumerate(["UV", "VJ"]):
            y = 2.5 * np.log10(table[f"{Z}_{f2}"] / table[f"{Z}_{f1}"])
            axes[i].plot(x, y, color=c, lw=1, label=f"Z={Z}")

    for i, (f1, f2) in enumerate(["UV", "VJ"]):
        axes[i].set_ylabel(rf"$\rm {f1}-{f2}$")

    axes[0].legend(fontsize=6, labelspacing=0.0)
    axes[1].set_xlabel(r"$\rm \log_{10}(age/Myr)$")

    return table, fig, axes


if __name__ == "__main__":
    grid_dir = "../../tests/test_grid"
    grid_name = "test_grid"

    grid = Grid(grid_name, grid_dir=grid_dir)

    simple_UVJ(grid)

    table, fig, axes = UVJ_metallicity(grid)
    plt.show()

    # table.write(f'data/{grid_name}_UVJ.ecsv', overwrite=True)
