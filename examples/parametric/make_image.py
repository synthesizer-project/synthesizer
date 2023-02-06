"""
Example for generating ta rest-frame physical scale image. This example will:
- build a parametric galaxy (see make_sfzh and make_sed)
- define its morphology
- calculate rest-frame luminosities for the UVJ bands
- make an image of the galaxy, including an RGB image.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from unyt import kpc, yr, Myr, mas

from synthesizer.filters import UVJ
from synthesizer.galaxy import ParametricGalaxy as Galaxy
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.morphology import Sersic2D
from synthesizer.grid import Grid


if __name__ == '__main__':

    # define morphology
    # r_eff could be defined in terms of an angle instead.
    morph = Sersic2D({'r_eff': 1 * kpc, 'n': 1.})

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # define SFZH
    # define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 10 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # Define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # Created a BinnedSFZH object containing the 2D star formation and metal
    # enrichment history
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities,
                         sfh, Zh, stellar_mass=stellar_mass)

    # initialise Galaxy
    galaxy = Galaxy(sfzh, morph=morph)

    # generate stellar spectra
    galaxy.get_stellar_spectra(grid)

    # define filter set
    filter_collection = UVJ(new_lam=grid.lam)

    sed = galaxy.spectra['stellar'].get_broadband_luminosities(
        filter_collection)

    # define geometry of images
    resolution = 0.1 * kpc  # resolution in kpc
    npix = 25  # width of image in pixels

    images = galaxy.make_images(
        'stellar', filter_collection, resolution, npix=npix)

    print(images)

    images.make_ascii()
    # images.plot()  #  plot base image
    # images.plot('U')  #  plot U-band image
    # images.plot_rgb(['J', 'V', 'U'])  #  plot RGB image
