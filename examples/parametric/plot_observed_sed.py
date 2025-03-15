"""
Generate parametric observed SED
================================

Example for generating the *observed* spectrum for a parametric galaxy
including photometry. This example will:
- build a parametric galaxy (see make_sfzh and make_sed)
- calculate spectral luminosity density (see make_sed)
- calculate observed frame spectra (requires comsology and redshift)
- calculate observed frame fluxes
"""

from astropy.cosmology import Planck18 as cosmo
from unyt import Msun, Myr

from synthesizer.emission_models import PacmanEmission
from synthesizer.emission_models.attenuation import Madau96, PowerLaw
from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy

if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # Define the emission model
    model = PacmanEmission(
        grid,
        tau_v=0.1,
        fesc=0.5,
        fesc_ly_alpha=0.5,
        dust_curve=PowerLaw(slope=-1),
    )

    # define filters
    filter_codes = [
        f"JWST/NIRCam.{f}"
        for f in [
            "F090W",
            "F115W",
            "F150W",
            "F200W",
            "F277W",
            "F356W",
            "F444W",
        ]
    ]  # define a list of filter codes
    filter_codes += [f"JWST/MIRI.{f}" for f in ["F770W"]]
    fc = FilterCollection(filter_codes, new_lam=grid.lam)

    # define the parameters of the star formation and metal enrichment
    # histories
    sfh_p = {"max_age": 10 * Myr}
    Z_p = {
        "log10metallicity": -2.0
    }  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1e8 * Msun

    # define the functional form of the star formation and metal enrichment
    # histories
    sfh = SFH.Constant(**sfh_p)  # constant star formation
    metal_dist = ZDist.DeltaConstant(**Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS
    # grid. This is (age, Z).
    stars = Stars(
        grid.log10age,
        grid.metallicity,
        sf_hist=sfh,
        metal_dist=metal_dist,
        initial_mass=stellar_mass,
    )

    # Define redshift
    z = 10.0

    # create a galaxy object
    galaxy = Galaxy(stars, redshift=z)

    # generate spectra using pacman model (complex)
    sed = galaxy.stars.get_spectra(model)

    # now calculate the observed frame spectra
    sed.get_fnu(
        cosmo,
        z,
        igm=Madau96,
    )  # generate observed frame spectra, assume Madau96 IGM model

    # measure broadband fluxes
    fluxes = sed.get_photo_fnu(fc)

    # print broadband fluxes
    for filter, flux in fluxes.items():
        print(f"{filter}: {flux:.2f}")

    # Calculate the observed spectra for all stellar spectra
    galaxy.get_observed_spectra(cosmo, igm=Madau96)

    # make plot of observed including broadband fluxes (if filter collection
    # object given)
    galaxy.plot_observed_spectra(
        filters=fc,
        show=True,
        combined_spectra=False,
        stellar_spectra=True,
    )
