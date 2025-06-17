"""Test the LOS column density calculations."""

import numpy as np
import pytest
from unyt import Mpc, Msun, Myr

from synthesizer.exceptions import InconsistentArguments
from synthesizer.particle import Galaxy, Gas, Stars


@pytest.fixture
def one_star():
    """Single star at z=1 Mpc, zero xy."""
    # Create Stars object with one particle
    star = Stars(
        initial_masses=np.array([1.0]) * Msun,
        ages=np.array([1.0]) * Myr,
        metallicities=np.array([0.02]),
        redshift=0.0,
        tau_v=np.array([0.0]),
        coordinates=np.array([[0.0, 0.0, 1.0]]) * Mpc,
    )
    # Assign dummy arrays needed by LOS
    star.smoothing_lengths = np.array([1.0]) * Mpc
    return star


@pytest.fixture
def one_gas_front():
    """Single gas in front of star."""
    gas = Gas(
        masses=np.array([1e6]) * Msun,
        metallicities=np.array([0.01]),
        redshift=0.0,
        coordinates=np.array([[0.0, 0.0, 0.0]]) * Mpc,
        dust_to_metal_ratio=1.0,
        smoothing_lengths=np.array([1.0]) * Mpc,
    )
    return gas


@pytest.fixture
def one_gas_behind():
    """Single gas behind star: z=2 Mpc."""
    gas = Gas(
        masses=np.array([1e6]) * Msun,
        metallicities=np.array([0.01]),
        redshift=0.0,
        coordinates=np.array([[0.0, 0.0, 2.0]]) * Mpc,
        dust_to_metal_ratio=1.0,
        smoothing_lengths=np.array([1.0]) * Mpc,
    )
    return gas


class TestLOSColumnDensity:
    """Test the line of sight column density calculations."""

    def test_column_density_in_front(self, one_star, one_gas_front):
        """Test Gas particle in front column density and tau_v."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )
        # Simple kernel of length kdim=1
        kernel = np.array([1.0])
        kappa = 2.0
        # Force serial loop by setting force_loop and min_count high
        tau = gal.get_stellar_los_tau_v(
            kappa=kappa,
            kernel=kernel,
            force_loop=1,
            min_count=10,
        )
        # For one gas: surf_density = dust_masses/(sml**2) * kernel[0]
        # dust_masses = mass * metallicity * dust_to_metal_ratio
        #            = 1e6 Msun * 0.01 * 1.0 = 1e4
        # sml = 1 Mpc → surf_density = 1e4
        # τ = kappa * surf_density / (1e6)**2
        #   = 2.0 * 1e4 / 1e12 = 2e-8
        expected = np.array([2e-8])
        assert np.allclose(tau, expected), (
            f"Expected tau {expected}, got {tau}"
        )
        assert np.allclose(one_star.tau_v, expected), (
            f"Expected star tau_v {expected}, got {one_star.tau_v}"
        )

    def test_column_density_behind_zero(self, one_star, one_gas_behind):
        """Test Gas particle behind column density."""
        gal = Galaxy(
            stars=one_star, gas=one_gas_behind, redshift=0.0, centre=None
        )
        kernel = np.array([1.0])
        kappa = 5.0
        tau = gal.get_stellar_los_tau_v(
            kappa=kappa,
            kernel=kernel,
            force_loop=1,
            min_count=10,
        )
        assert np.allclose(tau, 0.0)
        assert np.allclose(one_star.tau_v, 0.0)

    def test_missing_components(self, one_star, one_gas_front):
        """Raises when stars or gas missing."""
        gal_no_gas = Galaxy(
            stars=one_star, gas=None, redshift=0.0, centre=None
        )
        with pytest.raises(InconsistentArguments):
            gal_no_gas.get_stellar_los_tau_v(kappa=1.0, kernel=np.array([1.0]))
        gal_no_star = Galaxy(
            stars=None, gas=one_gas_front, redshift=0.0, centre=None
        )
        with pytest.raises(InconsistentArguments):
            gal_no_star.get_stellar_los_tau_v(
                kappa=1.0, kernel=np.array([1.0])
            )
