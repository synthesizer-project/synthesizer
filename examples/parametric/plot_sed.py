"""
Generate parametric galaxy SED
===============================

Example for generating the rest-frame spectrum for a parametric galaxy
including photometry. This example will:
- build a parametric galaxy (see make_sfzh)
- calculate spectral luminosity density
"""


from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, ZDist, Stars
from synthesizer.parametric.galaxy import Galaxy
from unyt import Myr, yr


if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # define the parameters of the star formation and metal enrichment
    # histories
    sfh_p = {"duration": 10 * Myr}
    Z_p = {"log10metallicity": -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1e8

    # define the functional form of the star formation and metal enrichment
    # histories
    sfh = SFH.Constant(**sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    metal_dist = ZDist.DeltaConstant(**Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS
    # grid.
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=stellar_mass,
    )

    # create a galaxy object
    galaxy = Galaxy(stars)

    # generate pure stellar spectra alone
    galaxy.stars.get_spectra_incident(grid)
    print("Pure stellar spectra")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # generate intrinsic spectra (which includes reprocessing by gas)
    galaxy.stars.get_spectra_reprocessed(grid, fesc=0.5)
    print("Intrinsic spectra")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # # --- simple dust and gas screen
    galaxy.stars.get_spectra_screen(grid, tau_v=0.1)
    print("Simple dust and gas screen")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # --- CF00 model
    galaxy.stars.get_spectra_CharlotFall(
        grid, tau_v_ISM=0.1, tau_v_BC=0.1, alpha_ISM=-0.7, alpha_BC=-1.3
    )
    print("CF00 model")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # # --- pacman model
    galaxy.stars.get_spectra_pacman(grid, tau_v=0.1, fesc=0.5)
    print("Pacman model")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # pacman model (no Lyman-alpha escape and no dust)
    galaxy.stars.get_spectra_pacman(grid, fesc=0.0, fesc_LyA=0.0)
    print("Pacman model (no Ly-alpha escape, and no dust)")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # # --- pacman model (complex)
    galaxy.stars.get_spectra_pacman(grid, fesc=0.0, fesc_LyA=0.5, tau_v=0.6)
    print("Pacman model (complex)")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # --- CF00 model implemented within pacman model
    galaxy.stars.get_spectra_pacman(
        grid,
        fesc=0.1,
        fesc_LyA=0.1,
        tau_v=[1.0, 1.0],
        alpha=[-1, -1],
        young_old_thresh=1e7 * yr,
    )
    print("CF00 implemented within the Pacman model")
    galaxy.plot_spectra(show=True, combined_spectra=False, stellar_spectra=True)

    # print galaxy summary
    print(galaxy)

    sed = galaxy.stars.spectra["attenuated"]
    print(sed)

    # generate broadband photometry using 3 top-hat filters
    tophats = {
        "U": {"lam_eff": 3650, "lam_fwhm": 660},
        "V": {"lam_eff": 5510, "lam_fwhm": 880},
        "J": {"lam_eff": 12200, "lam_fwhm": 2130},
    }
    fc = FilterCollection(tophat_dict=tophats, new_lam=grid.lam)

    bb_lnu = sed.get_rest_photometry(fc)
    print(bb_lnu)
