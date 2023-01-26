import os
import numpy as np
import matplotlib.pyplot as plt
from unyt import yr, Myr

from synthesizer.filters import UVJ
from synthesizer.galaxy import ParametricGalaxy as Galaxy
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.morphology import Sersic2D
from synthesizer.grid import Grid


if __name__ == '__main__':

    # ------------------------------------------------
    # --- define morphology

    morphology_parameters = {'r_eff': 5., 'n': 1.}
    morph = Sersic2D(morphology_parameters)
    # morph.plot(pixel_size=21, pixel_scale=1)  # --- show quick plot of morphology

    # ------------------------------------------------
    # --- define SFZH

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
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities,
                         sfh, Zh, stellar_mass=stellar_mass)

    # ------------------------------------------------
    # --- create galaxy

    galaxy = Galaxy(sfzh, morph=morph)

    galaxy.get_stellar_spectra(grid)

    filter_collection = UVJ(new_lam=grid.lam)

    sed = galaxy.spectra['stellar'].get_broadband_luminosities(
        filter_collection)

    resolution = 0.5
    npix = 25

    images = galaxy.make_images(
        'stellar', filter_collection, resolution, npix=npix)

    images.plot('U')

    images.plot_rgb(['J', 'V', 'U'])
