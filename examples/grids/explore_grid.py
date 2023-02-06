
"""
This example allows us to explore a HDF5 and the correspinding Grid object.
"""

from synthesizer.grid import Grid, get_available_lines
from synthesizer.utils import explore_hdf5_grid


import h5py


if __name__ == "__main__":

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'

    grid_dir = '/Users/stephenwilkins/Dropbox/Research/data/synthesizer/grids'
    grid_name = 'bc03-2016-BaSeL_chabrier03-0.1,100'

    # explore HDF5 grid
    with h5py.File(f'{grid_dir}/{grid_name}.hdf5', 'r') as hf:

        for k, v in hf.attrs.items():
            print('    -', k, ':', v)

        hf.visititems(explore_hdf5_grid)

    grid = Grid(grid_name, grid_dir=grid_dir)
    print(grid)
