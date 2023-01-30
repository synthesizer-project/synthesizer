import numpy as np

import matplotlib.pyplot as plt
from unyt import kpc, yr, Myr, mas

from synthesizer.filters import UVJ
from synthesizer.galaxy import ParametricGalaxy as Galaxy
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.morphology import Sersic2D
from synthesizer.grid import Grid


if __name__ == '__main__':

    # ------------------------------------------------
    # --- define morphology

    # r_eff could be defined in terms of (physical) kpc instead
    morph = Sersic2D({'r_eff': 1 * kpc, 'n': 1.})
    # morph.plot(pixel_size=21, pixel_scale=1)  # --- show quick plot of morphology

    # ------------------------------------------------
    # --- define SFZH

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'

    grid = Grid(grid_name, grid_dir=grid_dir)

    # --- define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 10 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

    # ------------------------------------------------
    # --- create galaxy

    galaxy = Galaxy(sfzh, morph=morph)

    galaxy.get_stellar_spectra(grid)

    filter_collection = UVJ(new_lam=grid.lam)

    print(filter_collection)

    sed = galaxy.spectra['stellar'].get_broadband_luminosities(filter_collection)

    resolution = 0.1 * kpc  # resolution in kpc
    npix = 25

    images = galaxy.make_images('stellar', filter_collection, resolution, npix=npix)

    print(images)

    images.plot('U')

    images.plot_rgb(['J', 'V', 'U'])
