import numpy as np
import pytest
from unyt import Mpc, Msun, Myr, km, kpc, s

from synthesizer.emission_models import (
    IncidentEmission,
    NebularEmission,
    TransmittedEmission,
)
from synthesizer.grid import Grid
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars


@pytest.fixture
def test_grid():
    """Return a Grid object."""
    return Grid("test_grid.hdf5", grid_dir="tests/test_grid")


@pytest.fixture
def nebular_emission_model():
    """Return a NebularEmission object."""
    # First need a grid to pass to the NebularEmission object
    grid = Grid("test_grid.hdf5", grid_dir="tests/test_grid")
    return NebularEmission(grid=grid)


@pytest.fixture
def incident_emission_model():
    """Return a IncidentEmission object."""
    # First need a grid to pass to the IncidentEmission object
    grid = Grid("test_grid.hdf5", grid_dir="tests/test_grid")
    return IncidentEmission(grid=grid)


@pytest.fixture
def transmitted_emission_model():
    """Return a TransmittedEmission object."""
    # First need a grid to pass to the IncidentEmission object
    grid = Grid("test_grid.hdf5", grid_dir="tests/test_grid")
    return TransmittedEmission(grid=grid)


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
