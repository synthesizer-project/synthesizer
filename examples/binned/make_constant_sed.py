
""" make a series of SEDs corresponding to different durations of constant star formation """



import numpy as np

import matplotlib.pyplot as plt

from synthesizer.filters import SVOFilterCollection
from synthesizer.grid import SpectralGrid
from synthesizer.binned.sfzh import SFH, ZH, generate_sfzh
from synthesizer.binned.galaxy import SEDGenerator
from synthesizer.plt import single, single_histxy, mlabel
from unyt import yr, Myr

from astropy.cosmology import Planck18 as cosmo


if __name__ == '__main__':

    grid_name = 'bpass-v2.2.1-bin_chab-100_cloudy-v17.03_log10Uref-2'

    grid = SpectralGrid(grid_name)

    fig, ax = single()

    for duration in [10, 100, 1000]:

        # --- define the parameters of the star formation and metal enrichment histories
        sfh_p = {'duration': duration * Myr }
        Z_p = {'log10Z': -2.0} # can also use linear metallicity e.g. {'Z': 0.01}

        # --- define the functional form of the star formation and metal enrichment histories
        sfh = SFH.Constant(sfh_p) # constant star formation
        sfh.summary() # print sfh summary
        Zh = ZH.deltaConstant(Z_p) # constant metallicity

        # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
        sfzh = generate_sfzh(grid.log10ages, grid.metallicities, sfh, Zh)

        galaxy = SEDGenerator(grid, sfzh)
        galaxy.pacman()

        ax.plot(np.log10(galaxy.lam), np.log10(galaxy.spectra['total'].lnu))

    ax.set_xlim([3., 4.])
    ax.set_ylim([18., 22])


    fig.savefig('figs/constant.pdf')
