"""Test the LOS column density calculations."""

import numpy as np
import pytest
from unyt import Mpc, Msun, Myr

from synthesizer.exceptions import (
    InconsistentArguments,
    UnimplementedFunctionality,
)
from synthesizer.kernel_functions import Kernel
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

    @staticmethod
    def _make_star(z=1.0, smoothing_length=1.0):
        """Create a single star particle for LOS tests."""
        star = Stars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([1.0]) * Myr,
            metallicities=np.array([0.02]),
            redshift=0.0,
            tau_v=np.array([0.0]),
            coordinates=np.array([[0.0, 0.0, z]]) * Mpc,
        )
        star.smoothing_lengths = np.array([smoothing_length]) * Mpc
        return star

    @staticmethod
    def _make_gas(z=0.0, smoothing_length=1.0):
        """Create a single gas particle for LOS tests."""
        return Gas(
            masses=np.array([1e6]) * Msun,
            metallicities=np.array([0.01]),
            redshift=0.0,
            coordinates=np.array([[0.0, 0.0, z]]) * Mpc,
            dust_to_metal_ratio=1.0,
            smoothing_lengths=np.array([smoothing_length]) * Mpc,
        )

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

    def test_column_density_as_points_default(self, one_star, one_gas_front):
        """Test the default point-particle LOS path remains unchanged."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )
        kernel = np.array([1.0])

        tau_default = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            force_loop=1,
            min_count=10,
        )
        tau_explicit = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=True,
            force_loop=1,
            min_count=10,
        )

        assert np.allclose(tau_default, tau_explicit)

    def test_column_density_smoothed_input_changes_result(
        self, one_star, one_gas_front
    ):
        """Test the smoothed-input LOS path differs from point-particle LOS."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )
        kernel = Kernel(name="uniform", binsize=32)

        tau_points = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=True,
            force_loop=1,
            min_count=10,
        )
        tau_smoothed = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )

        assert np.all(np.isfinite(tau_smoothed))
        assert np.all(tau_smoothed > 0.0)
        assert not np.allclose(tau_points, tau_smoothed, rtol=1e-3, atol=0.0)

    def test_column_density_accepts_kernel_object(
        self, one_star, one_gas_front
    ):
        """Test the point-particle LOS path accepts Kernel instances."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )
        kernel = Kernel(name="uniform", binsize=8)

        tau_object = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            force_loop=1,
            min_count=10,
        )
        tau_array = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel.get_kernel(),
            force_loop=1,
            min_count=10,
        )

        assert np.allclose(tau_object, tau_array)

    def test_column_density_smoothed_input_requires_smoothing_lengths(
        self, one_gas_front
    ):
        """Test smoothed-input LOS requires input smoothing lengths."""
        star = Stars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([1.0]) * Myr,
            metallicities=np.array([0.02]),
            redshift=0.0,
            tau_v=np.array([0.0]),
            coordinates=np.array([[0.0, 0.0, 1.0]]) * Mpc,
        )
        gal = Galaxy(
            stars=star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )

        with pytest.raises(InconsistentArguments):
            gal.get_stellar_los_tau_v(
                kappa=2.0,
                kernel=Kernel(name="uniform", binsize=8),
                as_points=False,
                force_loop=1,
                min_count=10,
            )

    def test_column_density_smoothed_input_requires_kernel_object(
        self, one_star, one_gas_front
    ):
        """Test smoothed-input LOS requires a kernel object."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )

        with pytest.raises(InconsistentArguments):
            gal.get_stellar_los_tau_v(
                kappa=2.0,
                kernel=np.array([1.0]),
                as_points=False,
                force_loop=1,
                min_count=10,
            )

    def test_column_density_smoothed_input_front_non_zero(
        self, one_star, one_gas_front
    ):
        """Test the staged smoothed-input LOS path gives a finite result."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )

        tau = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=Kernel(name="uniform", binsize=32),
            as_points=False,
            force_loop=1,
            min_count=10,
        )

        assert np.all(np.isfinite(tau))
        assert np.all(tau > 0.0)

    def test_column_density_smoothed_input_threshold_changes_result(
        self, one_star, one_gas_front
    ):
        """Test smoothed-input LOS honours the support threshold."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )

        tau_full = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=Kernel(name="uniform", binsize=32),
            as_points=False,
            threshold=1.0,
            force_loop=1,
            min_count=10,
        )
        tau_compact = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=Kernel(name="uniform", binsize=32),
            as_points=False,
            threshold=0.5,
            force_loop=1,
            min_count=10,
        )

        assert np.all(np.isfinite(tau_full))
        assert np.all(np.isfinite(tau_compact))
        assert np.all(tau_compact >= 0.0)
        assert not np.allclose(tau_full, tau_compact, rtol=1e-3, atol=0.0)

    def test_column_density_smoothed_input_force_loop_only(
        self, one_star, one_gas_front
    ):
        """Test the staged smoothed-input LOS path requires force_loop."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )

        with pytest.raises(UnimplementedFunctionality):
            gal.get_stellar_los_tau_v(
                kappa=2.0,
                kernel=Kernel(name="uniform", binsize=8),
                as_points=False,
                force_loop=0,
                min_count=10,
            )

    def test_column_density_smoothed_input_front_saturates(self):
        """Test fully front contributors give the same smoothed result."""
        kernel = Kernel(name="uniform", binsize=32)
        tau_front_far = Galaxy(
            stars=self._make_star(),
            gas=self._make_gas(z=-3.0),
            redshift=0.0,
            centre=None,
        ).get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )
        tau_front_edge = Galaxy(
            stars=self._make_star(),
            gas=self._make_gas(z=-2.0),
            redshift=0.0,
            centre=None,
        ).get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )

        assert np.allclose(tau_front_far, tau_front_edge, rtol=1e-3, atol=0.0)

    def test_column_density_smoothed_input_behind_zero(self):
        """Test fully behind contributors give zero smoothed attenuation."""
        tau = Galaxy(
            stars=self._make_star(),
            gas=self._make_gas(z=3.0),
            redshift=0.0,
            centre=None,
        ).get_stellar_los_tau_v(
            kappa=2.0,
            kernel=Kernel(name="uniform", binsize=32),
            as_points=False,
            force_loop=1,
            min_count=10,
        )

        assert np.allclose(tau, 0.0)

    def test_column_density_smoothed_input_straddling_is_intermediate(self):
        """Test straddling contributors interpolate between front and back."""
        kernel = Kernel(name="uniform", binsize=32)
        tau_front = Galaxy(
            stars=self._make_star(),
            gas=self._make_gas(z=-3.0),
            redshift=0.0,
            centre=None,
        ).get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )
        tau_straddle = Galaxy(
            stars=self._make_star(),
            gas=self._make_gas(z=0.0),
            redshift=0.0,
            centre=None,
        ).get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )
        tau_back = Galaxy(
            stars=self._make_star(),
            gas=self._make_gas(z=2.5),
            redshift=0.0,
            centre=None,
        ).get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )

        assert np.all(tau_front > tau_straddle)
        assert np.all(tau_straddle > tau_back)
        assert np.all(tau_back > 0.0)

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
