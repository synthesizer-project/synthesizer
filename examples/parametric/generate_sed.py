import os
import numpy as np

import matplotlib.pyplot as plt

from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.galaxy.parametric import ParametricGalaxy as Galaxy
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr
from astropy.cosmology import Planck18 as cosmo


if __name__ == '__main__':

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # --- define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 10 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities,
                         sfh, Zh, stellar_mass=stellar_mass)

    # --- create a galaxy object
    galaxy = Galaxy(sfzh)

    # # --- generate pure stellar spectra alone
    # galaxy.get_stellar_spectra(grid)
    # galaxy.plot_spectra()

    # # --- generate intrinsic spectra (which includes reprocessing by gas)
    # galaxy.get_intrinsic_spectra(grid, fesc = 0.5)
    # galaxy.plot_spectra()

    # # --- simple dust and gas screen
    # galaxy.get_screen_spectra(grid, tauV = 0.1, fesc = 0.5)
    # galaxy.plot_spectra()

    # # --- pacman model
    # galaxy.get_pacman_spectra(grid, tauV = 0.1, fesc = 0.5)
    # galaxy.plot_spectra()

    # # --- pacman model (no Lyman-alpha escapes and no dust)
    # galaxy.get_pacman_spectra(grid, fesc = 0.0, fesc_LyA = 0.0)
    # galaxy.plot_spectra()

    # # --- pacman model (complex)
    galaxy.get_pacman_spectra(grid, fesc=0.5, fesc_LyA=0.5, tauV=0.1)
    galaxy.plot_spectra()

    # # --- CF00 model NOT YET IMPLEMENTED
    # galaxy.get_pacman_spectra(grid, tauV = 0.1, fesc = 0.5)
    # galaxy.plot_spectra()

    print(galaxy)
