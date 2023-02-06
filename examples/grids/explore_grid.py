"""
This example allows us to explore a HDF5 and the correspinding Grid object.
"""

from synthesizer.grid import Grid, get_available_lines
import h5py


def explore_hdf5_grid(name, item):
    """
    A simple function for exploring HDF5 grid files.

    NOTE: this should be moved to some kind of utilities.
    TODO: modify to not loop over every line.
    """

    split = name.split("/")
    name_ = "    " * (len(split) - 1) + split[-1]
    print(name_, item)

    for k, v in item.attrs.items():
        print("    " * (len(split) - 1), k, ":", v)


if __name__ == "__main__":

    grid_dir = "../../tests/test_grid"
    grid_name = "test_grid"

    # explore HDF5 grid
    with h5py.File(f"{grid_dir}/{grid_name}.hdf5", "r") as hf:

        for k, v in hf.attrs.items():
            print("    -", k, ":", v)

        # hf.visititems(explore_hdf5_grid)

    grid = Grid(grid_name, grid_dir=grid_dir)
    print(grid)
