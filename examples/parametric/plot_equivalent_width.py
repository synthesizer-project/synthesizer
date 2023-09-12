"""
Plot equivalent width for UV indices
====================================

Example for generating the equivalent width for a set of UV indices from a parametric galaxy
- build a parametric galaxy (see make_sfzh)
- calculate equivalent width (see sed.py)
"""

import matplotlib.pyplot as plt

from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.galaxy import Galaxy as Galaxy
from synthesizer.sed import Sed
from unyt import yr, Myr


def set_index():
    """
    A function to define a dictionary of uv indices.

    Each index has a defined absorption window.
    A pseudo-continuum is defined, made up of a blue and red shifted window.

    Returns:
        tuple: A tuple containing the following lists:
            - index (int): List of UV indices.
            - index_window (int): List of absorption window bounds.
            - blue_window (int): List of blue shifted window bounds.
            - red_window (int): List of red shifted window bounds.
    """

    index = [1370, 1400, 1425, 1460, 1501, 1533, 1550, 1719, 1853]
    index_window = [[1360, 1380], [1385, 1410], [1413, 1435], [1450, 1470], [1496, 1506], [1530, 1537], [1530, 1560], [1705, 1729], [1838, 1858]]
    blue_window = [[1345, 1354], [1345, 1354], [1345, 1354], [1436, 1447], [1482, 1491], [1482, 1491], [1482, 1491], [1675, 1684], [1797, 1807]]
    red_window = [[1436, 1447], [1436, 1447], [1436, 1447], [1482, 1491], [1583, 1593], [1583, 1593], [1583, 1593], [1751, 1761], [1871, 1883]]

    return index, index_window, blue_window, red_window


def get_ew(index, feature, blue, red, Z, smass, grid, EqW, mode):
    """
    Calculate equivalent width for a specified UV index.

    Args:
        index (int): The UV index for which the equivalent width is calculated.
        Z (float): Metallicity.
        smass (float): Stellar mass.
        grid (Grid): The grid object.
        EqW (float): Initial equivalent width.
        mode (str): Calculation mode.

    Returns:
        float: The calculated equivalent width.

    Raises:
        ValueError: If mode is invalid.
    """
    
    sfh_p = {"duration": 100 * Myr}

    Z_p = {"Z": Z}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = smass

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(
        grid.log10age, grid.metallicity, sfh, Zh, stellar_mass=stellar_mass
    )

    # --- create a galaxy object
    galaxy = Galaxy(sfzh)

    # --- generate equivalent widths
    if mode == 0:
        galaxy.get_spectra_incident(grid)
    else:
        galaxy.get_spectra_intrinsic(grid, fesc=0.5)

    EqW.append(galaxy.get_equivalent_width(feature, blue, red))
    return EqW


def equivalent_width(grids, uv_index, index_window, blue_window, red_window):
    """
    Calculate equivalent widths for specified UV indices.

    Args:
        grids (str): Grid name.
        uv_index (list): List of UV indices to calculate equivalent widths for.
        index_window (list): List of index window bounds.
        blue_window (list): List of blue shifted window bounds.
        red_window (list): List of red shifted window bounds.

    Returns:
        None
    """
    
    # -- Calculate the equivalent width for each defined index
    for i, index in enumerate(uv_index):
        grid = Grid(grids, grid_dir=grid_dir)

        # --- define the parameters of the star formation and metal enrichment histories
        Z = grid.metallicity
        stellar_mass = 1e8
        EqW = []

        # Compute each index for each metallicity in the grid.
        feature, blue, red = index_window[i], blue_window[i], red_window[i]

        for k in range(0, len(Z)):
            EqW = get_ew(index, feature, blue, red, Z[k], stellar_mass, grid, EqW, 0)

        print(EqW)

        # Configure plot figure
        plt.rcParams["figure.dpi"] = 200
        plt.subplot(3, 3, i + 1)
        plt.grid(True)

        if i == 0 or i == 3 or i == 6:
            plt.ylabel("EW (\u212B)", fontsize=8)
        if i > 5:
            plt.xlabel("Z", fontsize=8)

        if index == 1501 or index == 1719:
            label = "UV_" + str(index)
        else:
            label = "F" + str(index)

        _, y_max = plt.ylim()

        plt.title(label, fontsize=8, transform=plt.gca().transAxes, y=0.8)

        plt.scatter(
            grid.metallicity,
            EqW,
            color="white",
            edgecolors="grey",
            alpha=1.0,
            zorder=10,
            linewidth=0.5,
            s=10,
        )
        plt.semilogx(grid.metallicity, EqW, linewidth=0.75, color="grey")
        EqW.clear()

        plt.tight_layout()

        if i == len(uv_index) - 1:
            plt.show()


if __name__ == "__main__":
    grid_dir = "../../tests/test_grid"  # Change this directory to your own.
    grid_name = "test_grid"  # Change this to the appropriate .hdf5

    index, index_window, blue_window, red_window = set_index()  # Retrieve UV indices

    equivalent_width(grid_name, index, index_window, blue_window, red_window)
