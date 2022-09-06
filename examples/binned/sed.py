

import numpy as np



from synthesizer.grid_sw import Grid
from synthesizer.binned import sfzh, binned
from synthesizer.plt import single, single_histxy, mlabel





if __name__ == '__main__':


    grid_name = 'bpass-v2.2.1_chab100-bin_cloudy-v17.0_logUref-2'

    grid = Grid(grid_name)

    # --- define the parameters of the star formation and metal enrichment histories
    sfh_p = [1E8] # [duration/yr]
    Z_p = [0.01]
    stellar_mass = 1E8

    # --- define the functional form of the star formation and metal enrichment histories
    sfh = sfzh.SFH.Constant(*sfh_p) # constant star formation
    Zh = sfzh.ZH.deltaConstant(*Z_p) # constant metallicity

    # --- get the 2D star formation and metal enrichment history for the given SPS grid. This is (age, Z).
    SFZH = sfzh.Binned.sfzh(grid.log10ages, grid.metallicities, sfh, Zh, stellar_mass = stellar_mass)

    galaxy = binned.SEDGenerator(grid, SFZH)


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
    galaxy.plot_spectra()
