"""
Example for generating the equivalent width for a set of UV indices from a parametric galaxy
- build a parametric galaxy (see make_sfzh)
- calculate equivalent width (see sed.py)
"""

import matplotlib.pyplot as plt

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.particle.galaxy import Galaxy as Galaxy
from synthesizer.sed import uv_indices as index
from unyt import yr, Myr


def get_ew(index, Z, imf, grid, EqW, mode):
    sfh_p = {'duration': 100 * Myr}

    Z_p = {'Z': Z}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = imf

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

    # --- create a galaxy object
    galaxy = Galaxy(sfzh)

    # --- generate equivalent widths
    if mode == 0:
        galaxy.get_particle_spectra_stellar(grid)
    else:
        galaxy.get_particle_spectra_intrinsic(grid, fesc=0.5)

    EqW.append(galaxy.get_equivalent_width(index))
    return EqW


def equivalent_width(grids, index, index_uv):
    # -- Calculate the equivalent width for each defined index
    for i in range(0, len(index)):
        grid = Grid(grids, grid_dir=grid_dir)

        # --- define the parameters of the star formation and metal enrichment histories
        Z = grid.metallicities
        stellar_mass = 1E8
        EqW = []

        # Compute each index for each metallicity in the grid.
        for k in range(0, len(Z)):
            EqW = get_ew(index[i], Z[k], stellar_mass, grid, EqW, 0)

        # Configure plot figure
        plt.rcParams['figure.dpi'] = 200
        plt.subplot(3, 3, i + 1)
        plt.grid(True)

        if i == 0 or i == 3 or i == 6:
            plt.ylabel('EW (\u212B)', fontsize=8)
        if i > 5:
            plt.xlabel('Z', fontsize=8)

        if index_uv[i] == 1501 or index_uv[i] == 1719:
            label = 'UV_' + str(index_uv[i])
        else:
            label = 'F' + str(index_uv[i])

        _, y_max = plt.ylim()

        plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8)

        plt.scatter(grid.metallicities, EqW, color='white', edgecolors='grey', alpha=1.0,
                    zorder=10, linewidth=0.5, s=10)
        plt.semilogx(grid.metallicities, EqW, linewidth=0.75, color='grey')
        EqW.clear()

        plt.tight_layout()

        if i == len(index) - 1:
            plt.show()


def get_indices():
    # -- Extract indices and pseudo continuum window from external csv file
    index_grid = index()
    temp = []
    key = []

    for i in range(0, len(index_grid)):
        temp.append(index_grid[i][1:])
        key.append(index_grid[i][0])

    index_grid = temp

    return index_grid, key


if __name__ == '__main__':
    grid_dir = '../tests/test_grid'  # Change this directory to your own.
    grid_name = 'test_grid'  # Change this to the appropriate .hdf5

    indices, uv_index = get_indices()  # Retrieve UV indices

    equivalent_width(grid_name, indices, uv_index)
