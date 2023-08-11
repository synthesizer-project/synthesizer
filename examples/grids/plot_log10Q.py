
import os
import argparse
import matplotlib.pyplot as plt
from synthesizer.grid import Grid
from synthesizer.plots import plot_log10Q

if __name__ == '__main__':

    """
    Makes a plot of the specific ionising luminosity for a given choice of grid and ion
    """


    # Get the location of this script, __file__ is the absolute path of this script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # define the test grid dir
    test_grid_dir = script_path + "/../../tests/test_grid/"

    # initialise argument parser
    parser = argparse.ArgumentParser(description=('Create a plot of all spectra types for a given metallicity and age'))

    # The name of the grid. Defaults to the test grid.
    parser.add_argument("-grid_name", "--grid_name", type=str, required=False, default="test_grid")

    # The path to the grid directory. Defaults to the test grid directory.
    parser.add_argument("-grid_dir", "--grid_dir", type=str, required=False, default=test_grid_dir)

    # The desired ion.
    parser.add_argument("-ion", type=str, required=False, default='HI')
    
    # Get dictionary of arguments
    args = parser.parse_args()


    # initialise grid
    grid = Grid(args.grid_name, grid_dir=args.grid_dir)

    # plot grid of HI ionising luminosities
    fig, ax = plot_log10Q(grid, ion=args.ion)
    plt.show()


