"""
Example for generating a composite galaxy
photometry. This example will:
- build two parametric "galaxies" (see make_sfzh)
- calculate spectral luminosity density of each
TODO: add image creation
"""


from synthesizer.grid import Grid
from synthesizer.parametric.morphology import Sersic2D
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh, generate_instant_sfzh
from synthesizer.galaxy import Galaxy
from synthesizer.filters import UVJ
from unyt import yr, Myr, kpc, mas
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':

    resolution = 0.05 * kpc
    npix = 50

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'
    grid = Grid(grid_name, grid_dir=grid_dir)

    filter_collection = UVJ(new_lam=grid.lam)

    # DISK

    # define morphology
    morphology_parameters = {'r_eff': 1. * kpc, 'n': 1., 'ellip': 0.5, 'theta': 35.}
    morph = Sersic2D(morphology_parameters)

    # define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 10 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

    # initialise Galaxy object
    disk = Galaxy(morph=morph, sfzh=sfzh)

    # generate stellar spectra
    disk.get_spectra_stellar(grid)

    # generate broadband luminosities
    sed = disk.spectra['stellar'].get_broadband_luminosities(filter_collection)

    # make images
    images = disk.make_images('stellar', filtercollection=filter_collection,
                              resolution=resolution, npix=npix)

    print(disk)
    images.plot_rgb(['J', 'V', 'U'])

    # BULGE

    morphology_parameters = {'r_eff': 1. * kpc, 'n': 4.}
    morph = Sersic2D(morphology_parameters)

    # define the parameters of the star formation and metal enrichment histories
    stellar_mass = 1E9
    sfzh = generate_instant_sfzh(
        grid.log10ages, grid.metallicities, 10., 0.01, stellar_mass=stellar_mass)

    bulge = Galaxy(morph=morph, sfzh=sfzh)

    bulge.get_spectra_stellar(grid)

    sed = bulge.spectra['stellar'].get_broadband_luminosities(filter_collection)

    # make images
    images = bulge.make_images('stellar', filter_collection, resolution, npix=npix)

    print(bulge)

    images.plot_rgb(['J', 'V', 'U'])

    # TOTAL

    total = disk + bulge

    print(total)

    # images = total.make_images('stellar', filter_collection, resolution, npix=npix)
    
    # images.plot_rgb(['J', 'V', 'U'])

    total = disk.images['stellar'].rgb_img + bulge.images['stellar'].rgb_img
    
    total /= np.max(total)
    
    plt.figure()
    plt.imshow(total, origin='lower', interpolation='nearest')
    plt.show()
