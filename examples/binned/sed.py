

import numpy as np

import matplotlib.pyplot as plt

from synthesizer.filters import FilterCollection
from synthesizer.grid_sw import Grid
from synthesizer.binned import SFH, ZH, generate_sfzh, SEDGenerator
from synthesizer.plt import single, single_histxy, mlabel

from astropy.cosmology import Planck18 as cosmo



if __name__ == '__main__':


    grid_name = 'bpass-v2.2.1_chab100-bin_cloudy-v17.0_logUref-2'

    grid = Grid(grid_name)

    # --- define the parameters of the star formation and metal enrichment histories
    sfh_p = [1E8] # [duration/yr]
    Z_p = [0.01]
    stellar_mass = 1E8

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(*sfh_p) # constant star formation
    Zh = ZH.deltaConstant(*Z_p) # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    SFZH = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass = stellar_mass)

    galaxy = SEDGenerator(grid, SFZH)


    # # --- simple dust and gas screen
    # galaxy.screen(tauV = 0.1)
    # galaxy.plot_spectra()

    # # --- should be identical to above
    # galaxy.pacman(tauV = 0.1)
    # galaxy.plot_spectra()

    # # --- half of light escapes without nebular reprocessing
    # galaxy.pacman(fesc = 0.5)
    # galaxy.plot_spectra()

    # --- no Lyman-alpha escapes
    # galaxy.pacman(fesc = 0.0, fesc_LyA = 0.0)
    # galaxy.plot_spectra()
    # galaxy.plot_spectra(spectra_to_plot = ['total'])

    # --- everything
    galaxy.pacman(fesc = 0.5, fesc_LyA = 0.5, tauV = 0.2)
    # galaxy.plot_spectra()

    z = 4

    sed = galaxy.spectra['total']
    sed.get_fnu(cosmo, z) # generate observed frame spectra


    # --- calculate broadband luminosities
    filter_codes = [f'JWST/NIRCam.{f}' for f in ['F090W', 'F115W','F150W','F200W','F277W','F356W','F444W']] # define a list of filter codes
    filter_codes += [f'JWST/MIRI.{f}' for f in ['F770W']]
    fc = FilterCollection(filter_codes, new_lam = sed.lamz)

    galaxy.plot_observed_spectra(cosmo, z, fc = fc, spectra_to_plot = ['total'])
