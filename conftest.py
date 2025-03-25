"""A collection of fixtures for testing the synthesizer package."""

import numpy as np
import pytest
from unyt import Hz, Mpc, Msun, Myr, angstrom, erg, km, kpc, s, yr

from synthesizer.emission_models import (
    BimodalPacmanEmission,
    IncidentEmission,
    IntrinsicEmission,
    NebularEmission,
    PacmanEmission,
    ReprocessedEmission,
    TransmittedEmission,
)
from synthesizer.emission_models.attenuation import Inoue14, Madau96
from synthesizer.emission_models.transformers.dust_attenuation import PowerLaw
from synthesizer.emissions import LineCollection, Sed
from synthesizer.grid import Grid
from synthesizer.instruments.filters import UVJ
from synthesizer.parametric.stars import Stars as ParametricStars
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars

# ================================== GRID =====================================


@pytest.fixture
def test_grid():
    """Return a Grid object."""
    return Grid("test_grid.hdf5", grid_dir="tests/test_grid")


@pytest.fixture
def lam():
    """
    Return a wavelength array.

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


# ================================= IGMS ======================================


@pytest.fixture
def i14():
    return Inoue14()


@pytest.fixture
def m96():
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
def random_part_stars():
    """Return a particle Stars object with velocities."""
    # Randomly generate the attribute we'll need for the stars
    nstars = np.random.randint(1, 10)
    initial_masses = np.random.uniform(0.1, 10, nstars) * 1e6 * Msun
    ages = np.random.uniform(4, 7, nstars) * Myr
    metallicities = np.random.uniform(0.01, 0.1, nstars)
    redshift = np.random.randint(0, 10)
    tau_v = np.random.uniform(0.1, 0.9, nstars)
    coordinates = (
        np.random.normal(
            0,
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

    return Stars(
        initial_masses=initial_masses,
        ages=ages,
        metallicities=metallicities,
        redshift=redshift,
        tau_v=tau_v,
        coordinates=coordinates,
        velocities=velocities,
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


# ================================ FILTERS ====================================


@pytest.fixture
def filters_UVJ(test_grid):
    """Return a dictionary of UVJ filters."""
    return UVJ(new_lam=test_grid.lam)


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
