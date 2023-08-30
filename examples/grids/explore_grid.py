"""
Explore a grid
==============

This example allows us to load a HDF5 grid file and explore the corresponding Grid object.
"""

import h5py
import os
from synthesizer.grid import Grid

if __name__ == "__main__":

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"

    # explore HDF5 grid
    with h5py.File(f'{grid_dir}/{grid_name}.hdf5', 'r') as hf:

        for k, v in hf.attrs.items():
            print('    -', k, ':', v)

        hf.visit(print)

    # open grid object
    grid = Grid(grid_name, grid_dir=grid_dir)
    print(grid)
