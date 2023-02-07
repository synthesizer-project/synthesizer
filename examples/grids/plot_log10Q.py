"""
Create a model SED
"""

import matplotlib.pyplot as plt

from synthesizer.grid import Grid

if __name__ == '__main__':

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'

    grid = Grid(grid_name, grid_dir=grid_dir)

    fig, ax = grid.plot_log10Q(ion='HI')
    plt.show()

    fig, ax = grid.plot_log10Q(ion='HeII')
    plt.show()
