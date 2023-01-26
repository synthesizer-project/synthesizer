"""
Create a model SED
"""
import os
import matplotlib.pyplot as plt

from synthesizer.grid import Grid

if __name__ == '__main__':

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"

    grid = Grid(grid_name, grid_dir=grid_dir)

    fig, ax = grid.plot_log10Q()
    # plt.show()
