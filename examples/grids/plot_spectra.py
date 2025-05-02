"""
Plot spectra example
====================

This example demonstrates how to extract a spectra directly from a grid and
plots all the available spectra.

NOTE: this only works on 2D grids at the moment
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from synthesizer.grid import Grid

if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # define the test grid dir
    # test_grid_dir = script_path + "/../../tests/test_grid/"
    test_grid_dir = "../../tests/test_grid/"

    # initialise argument parser
    parser = argparse.ArgumentParser(
        description=(
            "Create a plot of all spectra types for a given metallicity and \
            age"
        )
    )

    # The name of the grid. Defaults to the test grid.
    parser.add_argument(
        "-grid_name",
        "--grid_name",
        type=str,
        required=False,
        default="test_grid",
    )

    # The path to the grid directory. Defaults to the test grid directory.
    parser.add_argument(
        "-grid_dir",
        "--grid_dir",
        type=str,
        required=False,
        default=test_grid_dir,
    )

    # The target metallicity. The code function will find the closest
    # metallicity and report it back. The rationale behind this is
    # that this code can easily be adapted to explore other grids.
    parser.add_argument(
        "-metallicity", type=float, required=False, default=0.01
    )

    # The target log10(age/yr). The code function will find the closest
    # metallicity and report it back. The rationale behind this is that
    # this code can easily be adapted to explore other grids.
    parser.add_argument("-log10age", type=float, required=False, default=6.0)

    # Get dictionary of arguments
    args = parser.parse_args()

    # initialise grid
    grid = Grid(args.grid_name, grid_dir=args.grid_dir)

    # get the grid point for this log10age and metallicity
    grid_point = grid.get_grid_point(
        log10ages=args.log10age,
        metallicity=args.metallicity,
    )

    # loop over available spectra and plot
    for spec_name in grid.available_spectra:
        # get Sed object
        sed = grid.get_sed_at_grid_point(grid_point, spectra_type=spec_name)
        # print summary of SED object
        print(sed)
        plt.plot(
            np.log10(sed.lam),
            np.log10(sed.lnu),
            lw=1,
            alpha=0.8,
            label=spec_name,
        )

    plt.xlim([2.0, 4.0])
    plt.ylim([18.0, 23])
    plt.legend(fontsize=8, labelspacing=0.0)
    plt.xlabel(r"$\rm log_{10}(\lambda/\AA)$")
    plt.ylabel(r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$")
    plt.show()
