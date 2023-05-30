"""
Example for generating the *observed* spectrum for a parametric galaxy including
photometry. This example will:
- build a parametric galaxy (see make_sfzh and make_sed)
- calculate spectral luminosity density (see make_sed)
- calculate observed frame spectra (requires comsology and redshift)
- calculate observed frame fluxes
"""

import numpy as np
import matplotlib.pyplot as plt

from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.galaxy.parametric import ParametricGalaxy as Galaxy
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr
from synthesizer.igm import Madau96, Inoue14
from astropy.cosmology import Planck18 as cosmo


if __name__ == '__main__':

    grid_dir = '../../tests/test_grid'
    grid_name = 'test_grid'

    grid = Grid(grid_name, grid_dir=grid_dir)

    # define the parameters of the star formation and metal enrichment histories
    sfh_p = {'duration': 10 * Myr}
    Z_p = {'log10Z': -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1E8

    # define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass=stellar_mass)

    # create a galaxy object
    galaxy = Galaxy(sfzh)

    # generate spectra using pacman model (complex)
    sed = galaxy.get_pacman_spectra(grid, fesc=0.5, fesc_LyA=0.5, tauV=0.1, sed_object=True)

    # now calculate the observed frame spectra
    z = 10.  # redshift
    sed.get_fnu(cosmo, z, igm=Madau96())  # generate observed frame spectra

    # define filters
    filter_codes = [f'JWST/NIRCam.{f}' for f in ['F090W', 'F115W', 'F150W',
                                                 'F200W', 'F277W', 'F356W', 'F444W']]  # define a list of filter codes
    filter_codes += [f'JWST/MIRI.{f}' for f in ['F770W']]
    fc = FilterCollection(filter_codes, new_lam=sed.lamz)

    # print(sed.fnu)
    # print(sed.lnu)
    # print(sed._lnu)

    # measure broadband fluxes
    fluxes = sed.get_broadband_fluxes(fc)

    # print broadband fluxes
    for filter, flux in fluxes.items():
        print(f'{filter}: {flux:.2f}')

    # make plot of observed including broadband fluxes (if filter collection object given)
    galaxy.plot_observed_spectra(cosmo, z, fc=fc, spectra_to_plot=['total'])
