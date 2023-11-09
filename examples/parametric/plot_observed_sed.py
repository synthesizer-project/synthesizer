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


from synthesizer.filters import FilterCollection
from synthesizer.grid import Grid
from synthesizer.parametric.sfzh import SFH, ZH, generate_sfzh
from synthesizer.parametric.galaxy import Galaxy
from unyt import Myr
from synthesizer.igm import Madau96
from astropy.cosmology import Planck18 as cosmo


if __name__ == "__main__":
    # Get the location of this script, __file__ is the absolute path of this
    # script, however we just want to directory
    # script_path = os.path.abspath(os.path.dirname(__file__))

    # Define the grid
    grid_name = "test_grid"
    grid_dir = "../../tests/test_grid/"
    grid = Grid(grid_name, grid_dir=grid_dir)

    # define filters
    filter_codes = [
        f"JWST/NIRCam.{f}"
        for f in ["F090W", "F115W", "F150W", "F200W", "F277W", "F356W", 
                  "F444W"]
    ]  # define a list of filter codes
    filter_codes += [f"JWST/MIRI.{f}" for f in ["F770W"]]
    fc = FilterCollection(filter_codes)

    # define the parameters of the star formation and metal enrichment 
    # histories
    sfh_p = {"duration": 10 * Myr}
    Z_p = {"log10Z": -2.0}  # can also use linear metallicity e.g. {'Z': 0.01}
    stellar_mass = 1e8

    # define the functional form of the star formation and metal enrichment 
    # histories
    sfh = SFH.Constant(sfh_p)  # constant star formation
    Zh = ZH.deltaConstant(Z_p)  # constant metallicity

    # get the 2D star formation and metal enrichment history for the given SPS 
    # grid. This is (age, Z).
    sfzh = generate_sfzh(
        grid.log10age, grid.metallicity, sfh, Zh, stellar_mass=stellar_mass
    )

    # create a galaxy object
    galaxy = Galaxy(sfzh)

    # generate spectra using pacman model (complex)
    sed = galaxy.stars.get_spectra_pacman(grid, fesc=0.5, fesc_LyA=0.5, tau_v=0.1)

    # now calculate the observed frame spectra
    z = 10.0  # redshift
    sed.get_fnu(
        cosmo, z, igm=Madau96
    )  # generate observed frame spectra, assume Madau96 IGM model

    # measure broadband fluxes
    fluxes = sed.get_broadband_fluxes(fc)

    # print broadband fluxes
    for filter, flux in fluxes.items():
        print(f"{filter}: {flux:.2f}")

    # make plot of observed including broadband fluxes (if filter collection 
    # object given)
    galaxy.plot_observed_spectra(cosmo,
                                 z,
                                 fc=fc,
                                 # spectra_to_plot=["total"],
                                 show=True)
