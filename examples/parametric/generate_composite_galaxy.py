import os
from synthesizer.grid import Grid
# from synthesizer.parametric.morphology import Sersic2D
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.galaxy import ParametricGalaxy as Galaxy
from synthesizer.filters import UVJ
from unyt import yr, Myr
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    # resolution = 0.5
    # npix = 25

    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    filter_collection = UVJ(new_lam=grid.lam)

    # ------------------------------------------------
    # --- DISK

    # morphology_parameters = {'r_eff': 5., 'n': 1., 'ellip': 0.5, 'theta': 35.}
    # morph = Sersic2D(morphology_parameters)

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

    # disk = Galaxy(morph=morph, SFZH=sfzh)
    disk = Galaxy(sfzh=sfzh)

    disk.get_stellar_spectra(grid)

    sed = disk.spectra['stellar'].get_broadband_luminosities(filter_collection)

    # images = disk.make_images('stellar', filter_collection, resolution, npix=npix)
    # images.plot_rgb(['J', 'V', 'U'])

    # ------------------------------------------------
    # --- BULGE

    # morphology_parameters = {'r_eff': 5., 'n': 4.}
    # morph = Sersic2D(morphology_parameters)

    # --- define the parameters of the star formation and metal enrichment histories
    stellar_mass = 1E9
    sfzh = generate_instant_sfzh(
        grid.log10ages, grid.metallicities, 10., 0.01, stellar_mass=stellar_mass)

    # bulge = Galaxy(morph=morph, SFZH=sfzh)
    bulge = Galaxy(sfzh=sfzh)

    bulge.get_stellar_spectra(grid)

    sed = bulge.spectra['stellar'].get_broadband_luminosities(
        filter_collection)

    total = disk + bulge

    print(disk)
    print(bulge)
    print(total)

    # images = bulge.make_images('stellar', filter_collection, resolution, npix=npix)
    #
    # images.plot_rgb(['J', 'V', 'U'])

    # total = disk.images['stellar'].rgb_img + bulge.images['stellar'].rgb_img
    #
    # total /= np.max(total)
    #
    # plt.figure()
    # plt.imshow(total, origin='lower', interpolation='nearest')
    # plt.show()
