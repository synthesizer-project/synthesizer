"""
Generate parametric galaxy SED
===============================

Example for generating the rest-frame spectrum for a parametric galaxy including
photometry. This example will:
- build a parametric galaxy (see make_sfzh)
- calculate spectral luminosity density
"""
import os
import numpy as np

from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.galaxy import Galaxy
from unyt import yr, Myr


if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = script_path + "/../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # define the parameters of the star formation and metal enrichment histories
    sfh_p = {"duration": 10 * Myr}
    Z_p = {"log10Z": -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1e8

    # define the functional form of the star formation and metal enrichment histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS grid.
    sfzh = generate_sfzh(
        grid.log10age, grid.metallicity, sfh, Zh, stellar_mass=stellar_mass
    )

    # create a galaxy object
    galaxy = Galaxy(sfzh)

    # generate pure stellar spectra alone
    galaxy.get_spectra_incident(grid)
    print("Pure stellar spectra")
    galaxy.plot_spectra(show=True)

    # generate intrinsic spectra (which includes reprocessing by gas)
    galaxy.get_spectra_intrinsic(grid, fesc=0.5)
    print("Intrinsic spectra")
    galaxy.plot_spectra(show=True)

    # # --- simple dust and gas screen
    galaxy.get_spectra_screen(grid, tau_v=0.1, fesc=0.5)
    print("Simple dust and gas screen")
    galaxy.plot_spectra(show=True)

    # --- CF00 model
    galaxy.get_spectra_CharlotFall(
        grid, tau_v_ISM=0.1, tau_v_BC=0.1, alpha_ISM=-0.7, alpha_BC=-1.3
    )
    print("CF00 model")
    galaxy.plot_spectra(show=True)

    # # --- pacman model
    galaxy.get_spectra_pacman(grid, tau_v=0.1, fesc=0.5)
    print("Pacman model")
    galaxy.plot_spectra(show=True)

    # pacman model (no Lyman-alpha escape and no dust)
    galaxy.get_spectra_pacman(grid, fesc=0.0, fesc_LyA=0.0)
    print("Pacman model (no Ly-alpha escape, and no dust)")
    galaxy.plot_spectra(show=True)

    # # --- pacman model (complex)
    galaxy.get_spectra_pacman(grid, fesc=0.0, fesc_LyA=0.5, tau_v=0.6)
    print("Pacman model (complex)")
    galaxy.plot_spectra(show=True)

    # --- CF00 model implemented within pacman model
    galaxy.get_spectra_pacman(
        grid, fesc=0.1, fesc_LyA=0.1, tau_v=[1.0, 1.0], alpha=[-1, -1], CF00=True
    )
    print("CF00 implemented within the Pacman model")
    galaxy.plot_spectra()

    # print galaxy summary
    print(galaxy)

    sed = galaxy.spectra["total"]
    print(sed)

    # generate broadband photometry
    tophats = {
        "U": {"lam_eff": 3650, "lam_fwhm": 660},
        "V": {"lam_eff": 5510, "lam_fwhm": 880},
        "J": {"lam_eff": 12200, "lam_fwhm": 2130},
    }
    fc = FilterCollection(tophat_dict=tophats, new_lam=grid.lam)

    bb_lnu = sed.get_broadband_luminosities(fc)
    print(bb_lnu)
