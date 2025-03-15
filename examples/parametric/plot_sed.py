"""
Generate parametric galaxy SED
===============================

Example for generating the rest-frame spectrum for a parametric galaxy
including photometry. This example will:
- build a parametric galaxy (see make_sfzh)
- calculate spectral luminosity density
"""

from unyt import Msun, Myr, angstrom

from synthesizer.emission_models import (
    AttenuatedEmission,
    BimodalPacmanEmission,
    CharlotFall2000,
    IncidentEmission,
    PacmanEmission,
    ReprocessedEmission,
)
from synthesizer.emission_models.attenuation import PowerLaw
from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

if __name__ == "__main__":
    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the parameters of the star formation and metal enrichment
    # histories
    sfh_p = {"max_age": 10 * Myr}
    Z_p = {
        "log10metallicity": -2.0
    }  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1e8 * Msun

    # Define the functional form of the star formation and metal enrichment
    # histories
    sfh = SFH.Constant(**sfh_p)  # constant star formation
    print(sfh)  # print sfh summary

    metal_dist = ZDist.DeltaConstant(**Z_p)  # constant metallicity

    # Get the 2D star formation and metal enrichment history for the given SPS
    # grid.
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=stellar_mass,
    )

    # Create a galaxy object
    galaxy = Galaxy(stars)

    # Generate pure stellar spectra alone
    incident = IncidentEmission(grid)
    galaxy.stars.get_spectra(incident)
    print("Pure stellar spectra")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # Generate intrinsic spectra (which includes reprocessing by gas)
    reprocessed = ReprocessedEmission(grid, fesc=0.5)
    galaxy.stars.get_spectra(reprocessed)
    print("Intrinsic spectra")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # Simple dust and gas screen
    attenuated = AttenuatedEmission(
        tau_v=0.1,
        apply_to=reprocessed,
        dust_curve=PowerLaw(slope=-1),
        emitter="stellar",
    )
    galaxy.stars.get_spectra(attenuated)
    print("Simple dust and gas screen")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # --- CF00 model
    cf00 = CharlotFall2000(
        grid=grid,
        tau_v_ism=0.1,
        tau_v_birth=0.1,
        dust_curve_ism=PowerLaw(slope=-0.7),
        dust_curve_birth=PowerLaw(slope=-1.3),
    )
    galaxy.stars.get_spectra(cf00)
    print("CF00 model")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # # --- pacman model
    pc_fesc = PacmanEmission(
        grid,
        tau_v=0.1,
        dust_curve=PowerLaw(slope=-1),
        fesc=0.5,
    )
    galaxy.stars.get_spectra(pc_fesc)
    print("Pacman model")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # Pacman model (no Lyman-alpha escape and no dust)
    pc_lya = PacmanEmission(
        grid,
        tau_v=0.1,
        dust_curve=PowerLaw(slope=-1),
        fesc_ly_alpha=0.0,
    )
    galaxy.stars.get_spectra(pc_lya)
    print("Pacman model (no Ly-alpha escape, and no dust)")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # # --- pacman model (complex)
    pc_complex = PacmanEmission(
        grid,
        tau_v=0.6,
        dust_curve=PowerLaw(slope=-1),
        fesc=0.0,
        fesc_ly_alpha=0.5,
    )
    galaxy.stars.get_spectra(pc_complex)
    print("Pacman model (complex)")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # --- CF00 model implemented within pacman model
    cf_with_fesc = BimodalPacmanEmission(
        grid,
        tau_v_ism=0.1,
        tau_v_birth=0.1,
        dust_curve_ism=PowerLaw(slope=-1),
        dust_curve_birth=PowerLaw(slope=-1),
        fesc=0.1,
        fesc_ly_alpha=0.1,
    )
    galaxy.stars.get_spectra(cf_with_fesc)
    print("CF00 implemented within the Pacman model")
    galaxy.plot_spectra(
        show=True, combined_spectra=False, stellar_spectra=True
    )

    # Print galaxy summary
    print(galaxy)

    sed = galaxy.stars.spectra["attenuated"]
    print(sed)

    # Generate broadband photometry using 3 top-hat filters
    tophats = {
        "U": {"lam_eff": 3650 * angstrom, "lam_fwhm": 660 * angstrom},
        "V": {"lam_eff": 5510 * angstrom, "lam_fwhm": 880 * angstrom},
        "J": {"lam_eff": 12200 * angstrom, "lam_fwhm": 2130 * angstrom},
    }
    fc = FilterCollection(tophat_dict=tophats, new_lam=grid.lam)

    bb_lnu = sed.get_photo_lnu(fc)
    print(bb_lnu)
