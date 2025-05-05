"""A collection of fixtures for testing the synthesizer package."""

import numpy as np
import pytest
from scipy import signal
from unyt import (
    Hz,
    Mpc,
    Msun,
    Myr,
    angstrom,
    cm,
    erg,
    km,
    kpc,
    s,
    unyt_array,
    yr,
)

from synthesizer.emission_models import (
    BimodalPacmanEmission,
    IncidentEmission,
    IntrinsicEmission,
    NebularEmission,
    PacmanEmission,
    ReprocessedEmission,
    TemplateEmission,
    TransmittedEmission,
)
from synthesizer.emission_models.attenuation import Inoue14, Madau96
from synthesizer.emission_models.transformers.dust_attenuation import PowerLaw
from synthesizer.emissions import LineCollection, Sed
from synthesizer.grid import Grid, Template
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.instruments.filters import UVJ
from synthesizer.kernel_functions import Kernel
from synthesizer.parametric.stars import Stars as ParametricStars
from synthesizer.particle import BlackHoles, Galaxy, Gas, Stars
from synthesizer.photometry import PhotometryCollection
from synthesizer.pipeline import Pipeline

# ================================== GRID =====================================


@pytest.fixture
def test_grid():
    """Return a Grid object."""
    return Grid("test_grid.hdf5", grid_dir="tests/test_grid")


@pytest.fixture
def test_template():
    """Return a Template object."""
    lam = unyt_array(np.linspace(1000, 10000, 100), "angstrom")
    lnu = unyt_array(np.ones_like(lam.value), "erg/s/Hz")
    return Template(lam, lnu)


@pytest.fixture
def lam():
    """Return a wavelength array.

    This function generates a logarithmically spaced array of wavelengths
    ranging from 10^2 to 10^6 angstroms, with 1000 points in total.

    Returns:
        np.ndarray:
            A numpy array containing the generated wavelengths with
            angstrom units.
    """
    return np.logspace(2, 6, 1000) * angstrom


# ================================= MODELS ====================================


@pytest.fixture
def nebular_emission_model(test_grid):
    """Return a NebularEmission object."""
    # First need a grid to pass to the NebularEmission object
    return NebularEmission(grid=test_grid)


@pytest.fixture
def incident_emission_model(test_grid):
    """Return a IncidentEmission object."""
    # First need a grid to pass to the IncidentEmission object
    return IncidentEmission(grid=test_grid)


@pytest.fixture
def transmitted_emission_model(test_grid):
    """Return a TransmittedEmission object."""
    # First need a grid to pass to the IncidentEmission object
    return TransmittedEmission(grid=test_grid)


@pytest.fixture
def reprocessed_emission_model(test_grid):
    """Return a ReprocessedEmission object."""
    # First need a grid to pass to the IncidentEmission object
    return ReprocessedEmission(grid=test_grid)


@pytest.fixture
def intrinsic_emission_model(test_grid):
    """Return an IntrinsicEmission object."""
    return IntrinsicEmission(grid=test_grid)


@pytest.fixture
def pacman_emission_model(test_grid):
    """Return a PacmanEmission object."""
    return PacmanEmission(grid=test_grid)


@pytest.fixture
def bimodal_pacman_emission_model(test_grid):
    """Return a BimodalPacmanEmission object."""
    return BimodalPacmanEmission(
        grid=test_grid,
        dust_curve_ism=PowerLaw(slope=-0.7),
        dust_curve_birth=PowerLaw(slope=-1.3),
    )


@pytest.fixture
def template_emission_model_bh(test_template):
    """Return a TemplateEmission object."""
    return TemplateEmission(test_template, "blackhole")


# ================================= IGMS ======================================


@pytest.fixture
def i14():
    """Return an Inoue14 IGM object."""
    return Inoue14()


@pytest.fixture
def m96():
    """Return a Madau96 IGM object."""
    return Madau96()


# ================================= STARS =====================================


@pytest.fixture
def particle_stars_A():
    """Return a particle Stars object."""
    return Stars(
        initial_masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
        ages=np.array([1.0, 2.0, 3.0]) * Myr,
        metallicities=np.array([0.01, 0.02, 0.03]),
        redshift=1.0,
        tau_v=np.array([0.1, 0.2, 0.3]),
        coordinates=np.random.rand(3, 3) * kpc,
        dummy_attr=1.0,
    )


@pytest.fixture
def particle_stars_B():
    """Return a particle Stars object."""
    return Stars(
        initial_masses=np.array([4.0, 5.0, 6.0, 7.0]) * 1e6 * Msun,
        ages=np.array([4.0, 5.0, 6.0, 7.0]) * Myr,
        metallicities=np.array([0.04, 0.05, 0.06, 0.07]),
        redshift=1.0,
        tau_v=np.array([0.4, 0.5, 0.6, 0.7]),
        coordinates=np.random.rand(4, 3) * Mpc,
        dummy_attr=1.2,
    )


@pytest.fixture
def unit_mass_stars():
    """Return a particle Stars object with unit masses for weighting tests."""
    return Stars(
        initial_masses=np.ones(3) * Msun,
        ages=np.array([1.0, 2.0, 3.0]) * Myr,
        metallicities=np.array([0.01, 0.02, 0.03]),
        redshift=1.0,
        tau_v=np.array([0.1, 0.2, 0.3]),
        coordinates=np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )
        * kpc,
        current_masses=np.ones(3) * Msun,
    )


@pytest.fixture
def unit_emission_stars():
    """Return a particle Stars object with unit masses for weighting tests."""
    stars = Stars(
        initial_masses=np.ones(3) * Msun,
        ages=np.array([1.0, 2.0, 3.0]) * Myr,
        metallicities=np.array([0.01, 0.02, 0.03]),
        redshift=1.0,
        tau_v=np.array([0.1, 0.2, 0.3]),
        coordinates=np.array(
            [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        )
        * kpc,
        current_masses=np.ones(3) * Msun,
    )
    stars.particle_photo_lnu["FAKE"] = PhotometryCollection(
        filters=None,
        fake=np.ones(3) * erg / s / Hz,
    )
    stars.particle_photo_fnu["FAKE"] = PhotometryCollection(
        filters=None,
        fake=np.ones(3) * erg / s / cm**2 / Hz,
    )
    return stars


@pytest.fixture
def random_part_stars():
    """Return a particle Stars object with velocities."""
    # Randomly generate the attribute we'll need for the stars
    nstars = np.random.randint(5, 10)
    initial_masses = np.random.uniform(0.1, 10, nstars) * 1e6 * Msun
    ages = np.random.uniform(4, 7, nstars) * Myr
    metallicities = np.random.uniform(0.01, 0.1, nstars)
    redshift = np.random.randint(0, 10)
    tau_v = np.random.uniform(0.1, 0.9, nstars)
    coordinates = (
        np.random.normal(
            0.1,
            np.random.rand(1) * 100,
            (nstars, 3),
        )
        * Mpc
    )
    velocities = (
        np.random.normal(
            np.random.uniform(-100, 100),
            np.random.rand(1) * 200,
            (nstars, 3),
        )
        * km
        / s
    )
    smls = np.random.uniform(0.1, 1, nstars) * Mpc

    return Stars(
        initial_masses=initial_masses,
        ages=ages,
        metallicities=metallicities,
        redshift=redshift,
        tau_v=tau_v,
        coordinates=coordinates,
        velocities=velocities,
        smoothing_lengths=smls,
    )


@pytest.fixture
def single_star_particle():
    """Return a particle Stars object with a single star."""
    return Stars(
        initial_masses=np.array([1.0]) * Msun,
        ages=np.array([1e7]) * yr,
        metallicities=np.array([0.01]),
        redshift=1.0,
        tau_v=np.array([0.1]),
        coordinates=np.random.rand(1, 3) * kpc,
    )


@pytest.fixture
def single_star_parametric(test_grid):
    """Return a parametric Stars object with a single star."""
    return ParametricStars(
        test_grid.log10age,
        test_grid.metallicity,
        sf_hist=1e7 * yr,
        metal_dist=0.01,
        initial_mass=1 * Msun,
    )


# ================================= GAS =======================================


@pytest.fixture
def particle_gas_A():
    """Return a particle Gas object."""
    return Gas(
        masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
        metallicities=np.array([0.01, 0.02, 0.03]),
        redshift=1.0,
        coordinates=np.random.rand(3, 3) * Mpc,
        dust_to_metal_ratio=0.3,
    )


@pytest.fixture
def particle_gas_B():
    """Return a particle Gas object."""
    return Gas(
        masses=np.array([4.0, 5.0, 6.0, 7.0]) * 1e6 * Msun,
        metallicities=np.array([0.04, 0.05, 0.06, 0.07]),
        redshift=1.0,
        coordinates=np.random.rand(4, 3) * Mpc,
        dust_to_metal_ratio=0.3,
    )


@pytest.fixture
def random_particle_gas():
    """Return a particle Gas object with velocities."""
    # Randomly generate the attribute we'll need for the gas
    ngas = np.random.randint(2, 10)
    masses = np.random.uniform(0.1, 10, ngas) * 1e6 * Msun
    metallicities = np.random.uniform(0.01, 0.1, ngas)
    redshift = np.random.randint(0, 10)
    coordinates = (
        np.random.normal(
            0.1,
            np.random.rand(1) * 100,
            (ngas, 3),
        )
        * Mpc
    )
    velocities = (
        np.random.normal(
            np.random.uniform(-100, 100),
            np.random.rand(1) * 200,
            (ngas, 3),
        )
        * km
        / s
    )
    smls = np.random.uniform(0.1, 1, ngas) * Mpc

    return Gas(
        masses=masses,
        metallicities=metallicities,
        redshift=redshift,
        coordinates=coordinates,
        velocities=velocities,
        dust_to_metal_ratio=0.3,
        smoothing_lengths=smls,
    )


# ================================== AGN ======================================


@pytest.fixture
def particle_black_hole():
    """Return a particle BlackHole object."""
    return BlackHoles(
        masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
        accretion_rates=np.array([1.0, 2.0, 3.0]) * Msun / yr,
        redshift=1.0,
        coordinates=np.random.rand(3, 3) * Mpc,
    )


@pytest.fixture
def single_particle_black_hole():
    """Return a particle BlackHole object with a single black hole."""
    return BlackHoles(
        masses=np.array([1.0]) * 1e6 * Msun,
        accretion_rates=np.array([1.0]) * Msun / yr,
        redshift=1.0,
        coordinates=np.random.rand(1, 3) * Mpc,
    )


@pytest.fixture
def single_particle_black_hole_scalars():
    """Return a particle BlackHole object with a single black hole."""
    return BlackHoles(
        masses=1.0 * 1e6 * Msun,
        accretion_rates=1.0 * Msun / yr,
        redshift=1.0,
        coordinates=np.random.rand(1, 3) * Mpc,
    )


@pytest.fixture
def random_particle_black_hole():
    """Return a particle BlackHole object with velocities."""
    # Randomly generate the attribute we'll need for the black holes
    nblackholes = np.random.randint(2, 5)
    masses = np.random.uniform(0.1, 10, nblackholes) * 1e6 * Msun
    accretion_rates = np.random.uniform(0.1, 10, nblackholes) * Msun / yr
    redshift = np.random.randint(0, 10)
    coordinates = (
        np.random.normal(
            0.1,
            np.random.rand(1) * 100,
            (nblackholes, 3),
        )
        * Mpc
    )
    velocities = (
        np.random.normal(
            np.random.uniform(-100, 100),
            np.random.rand(1) * 200,
            (nblackholes, 3),
        )
        * km
        / s
    )
    smls = np.random.uniform(0.1, 1, nblackholes) * Mpc

    return BlackHoles(
        masses=masses,
        accretion_rates=accretion_rates,
        redshift=redshift,
        coordinates=coordinates,
        velocities=velocities,
        smoothing_lengths=smls,
    )


# ================================= GALAXIES ==================================


@pytest.fixture
def random_particle_galaxy(
    random_particle_gas,
    random_part_stars,
    random_particle_black_hole,
):
    """Return a particle Galaxy object with random particles."""
    # Unify the redshifts of the component
    redshift = random_part_stars.redshift
    random_particle_gas.redshift = redshift
    random_particle_black_hole.redshift = redshift
    centre = random_part_stars.coordinates.mean(axis=0)
    return Galaxy(
        stars=random_part_stars,
        gas=random_particle_gas,
        black_holes=random_particle_black_hole,
        redshift=redshift,
        centre=centre,
    )


@pytest.fixture
def list_of_random_particle_galaxies(random_particle_galaxy):
    """Return a list of particle Galaxy objects with random particles."""
    # Unify the redshifts of the component
    return [random_particle_galaxy for _ in range(3)]


# ================================ FILTERS ====================================


@pytest.fixture
def filters_UVJ(lam):
    """Return a dictionary of UVJ filters."""
    return UVJ(new_lam=lam)


@pytest.fixture
def nircam_filters(lam):
    """Return a dictionary of NIRCam filters."""
    return FilterCollection(
        filter_codes=[
            f"JWST/NIRCam.{f}"
            for f in ["F090W", "F150W", "F200W", "F277W", "F356W", "F444W"]
        ],
        new_lam=lam,
    )


# =============================== INSTRUMENTS =================================


@pytest.fixture
def uvj_instrument(filters_UVJ):
    """Return a UVJ instrument object."""
    return Instrument("UVJ", filters=filters_UVJ)


@pytest.fixture
def nircam_instrument(nircam_filters):
    """Return a NIRCAM instrument object."""
    # Create a fake PSF for each filter
    psf = np.outer(
        signal.windows.gaussian(100, 3),
        signal.windows.gaussian(100, 3),
    )
    return Instrument(
        "JWST",
        filters=nircam_filters,
        resolution=1 * Mpc,
        psfs={f: psf for f in nircam_filters.filter_codes},
    )


@pytest.fixture
def nircam_instrument_no_psf(nircam_filters):
    """Return a NIRCAM instrument object without PSF."""
    return Instrument(
        "JWST",
        filters=nircam_filters,
        resolution=1 * Mpc,
    )


@pytest.fixture
def spectroscopy_instrument(test_grid):
    """Return a generic spectroscopy instrument object."""
    return Instrument("GenericSpec", lam=test_grid.lam)


@pytest.fixture
def spatial_spec_instrument(test_grid):
    """Return a generic spatial spectroscopy instrument object."""
    return Instrument("GenericIFU", lam=test_grid.lam, resolution=1 * Mpc)


@pytest.fixture
def spectroscopy_instruments(spectroscopy_instrument, spatial_spec_instrument):
    """Return a dictionary of spectroscopy instruments."""
    return spectroscopy_instrument + spatial_spec_instrument


@pytest.fixture
def uvj_nircam_insts(uvj_instrument, nircam_instrument):
    """Return a dictionary of UVJ and NIRCAM instruments."""
    return uvj_instrument + nircam_instrument


# ================================ PIPELINE ===================================


@pytest.fixture
def base_pipeline(nebular_emission_model):
    """Return an empty pipeline."""
    return Pipeline(
        emission_model=nebular_emission_model,
        nthreads=1,
        verbose=0,
    )


@pytest.fixture
def pipeline_with_galaxies(
    nebular_emission_model,
    list_of_random_particle_galaxies,
):
    """Return an empty pipeline."""
    p = Pipeline(
        emission_model=nebular_emission_model,
        nthreads=1,
        verbose=0,
    )
    p.add_galaxies(list_of_random_particle_galaxies)
    return p


@pytest.fixture
def pipeline_with_galaxies_per_particle(
    nebular_emission_model,
    list_of_random_particle_galaxies,
):
    """Return an empty pipeline."""
    # Make the emisison model per particle
    nebular_emission_model.set_per_particle(True)
    p = Pipeline(
        emission_model=nebular_emission_model,
        nthreads=1,
        verbose=0,
    )
    p.add_galaxies(list_of_random_particle_galaxies)
    return p


# ================================ SPECTRA ====================================


@pytest.fixture
def unit_sed(test_grid):
    """Return a unit Sed object."""
    return Sed(
        lam=test_grid.lam,
        lnu=np.ones_like(test_grid._lam) * erg / s / Hz,
    )


@pytest.fixture
def empty_sed(lam):
    """Return an Sed instance."""
    return Sed(lam=lam)


# ================================= LINES =====================================


@pytest.fixture
def simple_line_collection():
    """Return a simple LineCollection with two emission lines."""
    return LineCollection(
        line_ids=["O III 5007 A", "H 1 6563 A"],
        lam=np.array([5007, 6563]) * angstrom,
        lum=np.array([1e40, 1e39]) * erg / s,
        cont=np.array([1e38, 1e37]) * erg / s / Hz,
    )


@pytest.fixture
def multi_dimension_line_collection():
    """Return a LineCollection with multidimensional arrays of lines."""
    return LineCollection(
        line_ids=["O III 5007 A", "H 1 6563 A", "H 1 4861 A"],
        lam=np.array([5007, 6563, 4861]) * angstrom,
        lum=np.array([[1e40, 1e39, 1e38], [2e40, 2e39, 2e38]]) * erg / s,
        cont=np.array([[1e38, 1e37, 1e36], [2e38, 2e37, 2e36]]) * erg / s / Hz,
    )


@pytest.fixture
def line_ratio_collection(test_grid):
    """Return a LineCollection with lines needed for common ratios."""
    return test_grid.get_lines((1, 1))


# ================================== KERNEL ===================================


@pytest.fixture
def kernel():
    """Return a Kernel object."""
    sph_kernel = Kernel()
    return sph_kernel.get_kernel()
