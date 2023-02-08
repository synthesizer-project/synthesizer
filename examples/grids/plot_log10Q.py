"""
Create a model SED
"""

import matplotlib.pyplot as plt

from synthesizer.grid import Grid
from synthesizer.plots import plot_log10Q

if __name__ == '__main__':

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'

    grid = Grid(grid_name, grid_dir=grid_dir)

    # plot grid of HI ionising luminosities
    fig, ax = plot_log10Q(grid, ion='HI')
    plt.show()

    # plot grid of HeII ionising luminosities
    fig, ax = plot_log10Q(grid, ion='HeII')
    plt.show()
