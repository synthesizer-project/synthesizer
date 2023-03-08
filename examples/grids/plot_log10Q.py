"""
plot grid (age / metallicity) of HI and HeII ionissing emissivity
"""
import os
import matplotlib.pyplot as plt

from synthesizer.grid import Grid
from synthesizer.plots import plot_log10Q

if __name__ == '__main__':

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"

    grid = Grid(grid_name, grid_dir=grid_dir)

    # plot grid of HI ionising luminosities
    fig, ax = plot_log10Q(grid, ion='HI')
    plt.show()

    # plot grid of HeII ionising luminosities
    fig, ax = plot_log10Q(grid, ion='HeII')
    plt.show()
