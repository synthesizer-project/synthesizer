import numpy as np
import pytest
from unyt import Mpc, Msun, Myr

from synthesizer.grid import Grid
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars


@pytest.fixture
def test_grid():
    """
    Returns a Grid object
    """
    return Grid("test_grid.hdf5", grid_dir="tests/test_grid")


@pytest.fixture
def particle_stars_A():
    return Stars(
        initial_masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
        ages=np.array([1.0, 2.0, 3.0]) * Myr,
        metallicities=np.array([0.01, 0.02, 0.03]),
        redshift=1.0,
        tau_v=np.array([0.1, 0.2, 0.3]),
        coordinates=np.random.rand(3, 3) * Mpc,
        dummy_attr=1.0,
    )


@pytest.fixture
def particle_stars_B():
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
    return Gas(
        masses=np.array([1.0, 2.0, 3.0]) * 1e6 * Msun,
        metallicities=np.array([0.01, 0.02, 0.03]),
        redshift=1.0,
        coordinates=np.random.rand(3, 3) * Mpc,
        dust_to_metal_ratio=0.3,
    )


@pytest.fixture
def particle_gas_B():
    return Gas(
        masses=np.array([4.0, 5.0, 6.0, 7.0]) * 1e6 * Msun,
        metallicities=np.array([0.04, 0.05, 0.06, 0.07]),
        redshift=1.0,
        coordinates=np.random.rand(4, 3) * Mpc,
        dust_to_metal_ratio=0.3,
    )
