"""Test the LOS column density calculations."""

import numpy as np
import pytest
from unyt import Mpc, Msun, Myr, pc, unyt_array

from synthesizer.exceptions import InconsistentArguments
from synthesizer.kernel_functions import Kernel
from synthesizer.particle import Galaxy, Gas, Stars
from synthesizer.units import unyt_to_ndview


@pytest.fixture
def one_star():
    """Single star at z=1 Mpc, zero xy."""
    star = Stars(
        initial_masses=np.array([1.0]) * Msun,
        ages=np.array([1.0]) * Myr,
        metallicities=np.array([0.02]),
        redshift=0.0,
        tau_v=np.array([0.0]),
        coordinates=np.array([[0.0, 0.0, 1.0]]) * Mpc,
    )
    star.smoothing_lengths = np.array([1.0]) * Mpc
    return star


@pytest.fixture
def one_gas_front():
    """Single gas in front of star."""
    return Gas(
        masses=np.array([1e6]) * Msun,
        metallicities=np.array([0.01]),
        redshift=0.0,
        coordinates=np.array([[0.0, 0.0, 0.0]]) * Mpc,
        dust_to_metal_ratio=1.0,
        smoothing_lengths=np.array([1.0]) * Mpc,
    )


@pytest.fixture
def one_gas_behind():
    """Single gas behind star: z=2 Mpc."""
    return Gas(
        masses=np.array([1e6]) * Msun,
        metallicities=np.array([0.01]),
        redshift=0.0,
        coordinates=np.array([[0.0, 0.0, 2.0]]) * Mpc,
        dust_to_metal_ratio=1.0,
        smoothing_lengths=np.array([1.0]) * Mpc,
    )


class TestLOSColumnDensity:
    """Test the line of sight column density calculations."""

    _UNIFORM_W0 = 1.0 / ((4.0 / 3.0) * np.pi)
    _UNIFORM_PROJECTED_CENTRE = 3.0 / (2.0 * np.pi)

    @staticmethod
    def _kernel(name="uniform", binsize=32, **kwargs):
        """Construct a Kernel instance for LOS tests."""
        return Kernel(name=name, binsize=binsize, **kwargs)

    @staticmethod
    def _evaluate_kernel_direct(kernel_name, r):
        """Evaluate a kernel directly without using lookup tables."""
        r = np.asarray(r)
        values = np.zeros_like(r, dtype=np.float64)

        if kernel_name == "uniform":
            values[r < 1.0] = 1.0 / ((4.0 / 3.0) * np.pi)
            return values

        if kernel_name == "sph_anarchy":
            mask = r <= 1.0
            rm = 1.0 - r[mask]
            values[mask] = (
                (21.0 / (2.0 * np.pi)) * rm**4 * (1.0 + 4.0 * r[mask])
            )
            return values

        if kernel_name == "gadget_2":
            inner = r < 0.5
            outer = (r >= 0.5) & (r < 1.0)
            values[inner] = (8.0 / np.pi) * (
                1.0 - 6.0 * r[inner] ** 2 + 6.0 * r[inner] ** 3
            )
            values[outer] = (16.0 / np.pi) * (1.0 - r[outer]) ** 3
            return values

        if kernel_name == "cubic":
            inner = r < 0.5
            outer = (r >= 0.5) & (r < 1.0)
            values[inner] = (
                2.546479089470
                + 15.278874536822 * (r[inner] - 1.0) * r[inner] ** 2
            )
            values[outer] = 5.092958178941 * (1.0 - r[outer]) ** 3
            return values

        if kernel_name == "quintic":
            inner = r < 0.333333333
            middle = (r >= 0.333333333) & (r < 0.666666667)
            outer = (r >= 0.666666667) & (r < 1.0)
            values[inner] = 27.0 * (
                6.4457752 * r[inner] ** 4 * (1.0 - r[inner])
                - 1.4323945 * r[inner] ** 2
                + 0.17507044
            )
            values[middle] = 27.0 * (
                3.2228876 * r[middle] ** 4 * (r[middle] - 3.0)
                + 10.7429587 * r[middle] ** 3
                - 5.01338071 * r[middle] ** 2
                + 0.5968310366 * r[middle]
                + 0.1352817016
            )
            values[outer] = (
                27.0
                * 0.64457752
                * (
                    -(r[outer] ** 5)
                    + 5.0 * r[outer] ** 4
                    - 10.0 * r[outer] ** 3
                    + 10.0 * r[outer] ** 2
                    - 5.0 * r[outer]
                    + 1.0
                )
            )
            return values

        raise ValueError(f"Unsupported kernel {kernel_name}")

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

    @classmethod
    def _uniform_overlap_reference(
        cls,
        star_position,
        star_smoothing_length,
        gas_position,
        gas_smoothing_length,
        gas_dust_mass,
        nsample=90,
    ):
        """Compute a high-resolution reference for the uniform kernel."""
        mids = np.linspace(-1.0 + 1.0 / nsample, 1.0 - 1.0 / nsample, nsample)
        qx, qy, qz = np.meshgrid(mids, mids, mids, indexing="ij")
        mask = qx * qx + qy * qy + qz * qz < 1.0

        qx = qx[mask]
        qy = qy[mask]
        qz = qz[mask]

        x = star_position[0] + star_smoothing_length * qx
        y = star_position[1] + star_smoothing_length * qy
        z = star_position[2] + star_smoothing_length * qz

        dx = gas_position[0] - x
        dy = gas_position[1] - y
        projected_q2 = (dx * dx + dy * dy) / (gas_smoothing_length**2)
        inside = projected_q2 < 1.0

        values = np.zeros_like(x)
        if np.any(inside):
            z_extent = np.sqrt(1.0 - projected_q2[inside])
            z_upper = (z[inside] - gas_position[2]) / gas_smoothing_length
            upper = np.minimum(z_extent, z_upper)
            foreground_length = np.maximum(0.0, upper + z_extent)
            values[inside] = (
                gas_dust_mass
                / (gas_smoothing_length**2)
                * cls._UNIFORM_W0
                * foreground_length
            )

        return values.mean()

    @classmethod
    def _direct_overlap_reference(
        cls,
        kernel_name,
        star_position,
        star_smoothing_length,
        gas_position,
        gas_smoothing_length,
        gas_dust_mass,
        nsample=18,
        nz=1025,
    ):
        """Compute a direct numerical overlap reference."""
        mids = np.linspace(-1.0 + 1.0 / nsample, 1.0 - 1.0 / nsample, nsample)
        qx, qy, qz = np.meshgrid(mids, mids, mids, indexing="ij")
        qr = np.sqrt(qx * qx + qy * qy + qz * qz)
        mask = qr < 1.0

        qx = qx[mask]
        qy = qy[mask]
        qz = qz[mask]
        qr = qr[mask]

        weights = cls._evaluate_kernel_direct(kernel_name, qr)

        x = star_position[0] + star_smoothing_length * qx
        y = star_position[1] + star_smoothing_length * qy
        z = star_position[2] + star_smoothing_length * qz

        dx = gas_position[0] - x
        dy = gas_position[1] - y
        projected_q = np.sqrt(dx * dx + dy * dy) / gas_smoothing_length

        values = np.zeros_like(weights)
        inside = projected_q < 1.0
        if np.any(inside):
            zeta = np.linspace(-1.0, 1.0, nz)
            radius = np.sqrt(
                projected_q[inside, None] ** 2 + zeta[None, :] ** 2
            )
            integrand = cls._evaluate_kernel_direct(kernel_name, radius)
            z_upper = (z[inside] - gas_position[2]) / gas_smoothing_length
            integrand = np.where(
                zeta[None, :] <= z_upper[:, None], integrand, 0.0
            )
            values[inside] = (
                gas_dust_mass
                / (gas_smoothing_length**2)
                * np.trapezoid(integrand, zeta, axis=1)
            )

        return np.sum(weights * values) / np.sum(weights)

    def _assert_uniform_overlap_matches_reference(
        self,
        star_position,
        gas_position,
        star_smoothing_length=1.0,
        gas_smoothing_length=1.0,
        rtol=7e-3,
        expect_non_zero=True,
    ):
        """Assert the smoothed LOS result matches an independent reference."""
        star = self._make_star(
            z=star_position[2],
            smoothing_length=star_smoothing_length,
        )
        star.coordinates = np.array([star_position]) * Mpc

        gas = self._make_gas(
            z=gas_position[2],
            smoothing_length=gas_smoothing_length,
        )
        gas.coordinates = np.array([gas_position]) * Mpc

        galaxy = Galaxy(
            stars=star,
            gas=gas,
            redshift=0.0,
            centre=None,
        )
        kernel = Kernel(name="uniform", binsize=128)

        measured = galaxy.stars.get_los_column_density(
            galaxy.gas,
            "dust_masses",
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )[0]
        reference = self._uniform_overlap_reference(
            np.array(star_position),
            star_smoothing_length,
            np.array(gas_position),
            gas_smoothing_length,
            gas.dust_masses[0].value,
        )

        if expect_non_zero:
            assert measured > 0.0
            assert reference > 0.0
        else:
            assert np.isclose(measured, 0.0)
            assert np.isclose(reference, 0.0)

        assert np.isclose(measured, reference, rtol=rtol, atol=0.0), (
            f"Expected {reference:.8e}, got {measured:.8e} for star "
            f"at {star_position} and gas at {gas_position}."
        )

    def _assert_direct_overlap_matches_reference(
        self,
        kernel_name,
        star_position,
        gas_position,
        star_smoothing_length=1.0,
        gas_smoothing_length=1.0,
        rtol=2e-2,
    ):
        """Assert the smoothed overlap matches a direct numerical reference."""
        star = self._make_star(
            z=star_position[2],
            smoothing_length=star_smoothing_length,
        )
        star.coordinates = np.array([star_position]) * Mpc

        gas = self._make_gas(
            z=gas_position[2],
            smoothing_length=gas_smoothing_length,
        )
        gas.coordinates = np.array([gas_position]) * Mpc

        measured = star.get_los_column_density(
            gas,
            "dust_masses",
            kernel=Kernel(name=kernel_name, binsize=128),
            as_points=False,
            force_loop=1,
            min_count=10,
        )[0]
        reference = self._direct_overlap_reference(
            kernel_name,
            np.array(star_position),
            star_smoothing_length,
            np.array(gas_position),
            gas_smoothing_length,
            gas.dust_masses[0].value,
        )

        assert measured > 0.0
        assert reference > 0.0
        assert np.isclose(measured, reference, rtol=rtol, atol=0.0), (
            f"Expected {reference:.8e}, got {measured:.8e} for kernel "
            f"{kernel_name}, star at {star_position}, and gas at "
            f"{gas_position}."
        )

    def test_column_density_in_front(self, one_star, one_gas_front):
        """Test Gas particle in front column density and tau_v."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )
        kernel = self._kernel()
        tau = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            force_loop=1,
            min_count=10,
        )
        expected = np.array(
            [2.0 * 1e4 * self._UNIFORM_PROJECTED_CENTRE / 1e12]
        )
        assert np.allclose(tau, expected)
        assert np.allclose(one_star.tau_v, expected)

    def test_column_density_returns_units_and_stores_attr(
        self, one_star, one_gas_front
    ):
        """LOS column density keeps the correct surface-density units."""
        col_den = one_star.get_los_column_density(
            one_gas_front,
            "masses",
            self._kernel(),
            column_density_attr="sigmalos_mass",
            force_loop=1,
            min_count=10,
        )

        expected_units = (
            one_gas_front.masses.units / one_gas_front.coordinates.units**2
        )

        assert isinstance(col_den, unyt_array)
        assert col_den.units == expected_units
        assert one_star.sigmalos_mass.units == expected_units
        assert col_den is one_star.sigmalos_mass

    def test_column_density_zero_particle_returns_unitful_array(
        self, one_gas_front
    ):
        """Zero-particle LOS returns still carry the right units."""
        empty_stars = Stars(
            initial_masses=np.array([]) * Msun,
            ages=np.array([]) * Myr,
            metallicities=np.array([]),
            redshift=0.0,
            tau_v=np.array([]),
            coordinates=np.empty((0, 3)) * Mpc,
            smoothing_lengths=np.array([]) * Mpc,
        )

        col_den = empty_stars.get_los_column_density(
            one_gas_front,
            "masses",
            self._kernel(),
            column_density_attr="sigmalos_mass",
            force_loop=1,
            min_count=10,
        )

        expected_units = (
            one_gas_front.masses.units / one_gas_front.coordinates.units**2
        )

        assert isinstance(col_den, unyt_array)
        assert col_den.units == expected_units
        assert col_den.size == 0
        assert empty_stars.sigmalos_mass.units == expected_units

    def test_column_density_zero_particle_mask_returns_masked_shape(
        self, one_gas_front
    ):
        """Zero-particle early returns respect the masked output shape."""
        empty_stars = Stars(
            initial_masses=np.array([]) * Msun,
            ages=np.array([]) * Myr,
            metallicities=np.array([]),
            redshift=0.0,
            tau_v=np.array([]),
            coordinates=np.empty((0, 3)) * Mpc,
            smoothing_lengths=np.array([]) * Mpc,
        )

        col_den = empty_stars.get_los_column_density(
            one_gas_front,
            "masses",
            self._kernel(),
            mask=np.array([], dtype=bool),
            column_density_attr="sigmalos_mass",
            force_loop=1,
            min_count=10,
        )

        assert col_den.shape == (0,)
        assert empty_stars.sigmalos_mass.shape == (0,)

    def test_column_density_empty_source_returns_unitful_array(self, one_star):
        """Empty source particles still yield a unitful zero array."""
        empty_gas = Gas(
            masses=np.array([]) * Msun,
            metallicities=np.array([]),
            redshift=0.0,
            coordinates=np.empty((0, 3)) * Mpc,
            dust_to_metal_ratio=1.0,
            smoothing_lengths=np.array([]) * Mpc,
        )

        col_den = one_star.get_los_column_density(
            empty_gas,
            "masses",
            self._kernel(),
            column_density_attr="sigmalos_mass",
            force_loop=1,
            min_count=10,
        )

        expected_units = (
            empty_gas.masses.units / empty_gas.coordinates.units**2
        )

        assert isinstance(col_den, unyt_array)
        assert col_den.units == expected_units
        assert np.allclose(col_den.value, np.zeros(one_star.nparticles))
        assert one_star.sigmalos_mass.units == expected_units

    def test_column_density_empty_source_mask_returns_masked_shape(
        self, one_star
    ):
        """Empty source early returns match the number of masked targets."""
        empty_gas = Gas(
            masses=np.array([]) * Msun,
            metallicities=np.array([]),
            redshift=0.0,
            coordinates=np.empty((0, 3)) * Mpc,
            dust_to_metal_ratio=1.0,
            smoothing_lengths=np.array([]) * Mpc,
        )
        mask = np.array([True])

        col_den = one_star.get_los_column_density(
            empty_gas,
            "masses",
            self._kernel(),
            mask=mask,
            column_density_attr="sigmalos_mass",
            force_loop=1,
            min_count=10,
        )

        assert col_den.shape == (mask.sum(),)
        assert one_star.sigmalos_mass.shape == (mask.sum(),)

    def test_column_density_missing_attr_raises_consistent_error(
        self, one_star
    ):
        """Missing density attributes should raise the LOS validation error."""
        bad_gas = Gas(
            masses=np.array([1e6]) * Msun,
            metallicities=np.array([0.01]),
            redshift=0.0,
            coordinates=np.array([[0.0, 0.0, 0.0]]) * Mpc,
            dust_to_metal_ratio=1.0,
            smoothing_lengths=np.array([1.0]) * Mpc,
        )
        bad_gas.masses = None

        with pytest.raises(InconsistentArguments):
            one_star.get_los_column_density(
                bad_gas,
                "masses",
                self._kernel(),
                force_loop=1,
                min_count=10,
            )

    def test_column_density_unitless_attr_raises_consistent_error(
        self, one_star
    ):
        """Unitless density attributes should fail before unit access."""
        bad_gas = Gas(
            masses=np.array([1e6]) * Msun,
            metallicities=np.array([0.01]),
            redshift=0.0,
            coordinates=np.array([[0.0, 0.0, 0.0]]) * Mpc,
            dust_to_metal_ratio=1.0,
            smoothing_lengths=np.array([1.0]) * Mpc,
        )
        bad_gas.unitless_density = np.array([1.0])

        with pytest.raises(InconsistentArguments, match="unitless_density"):
            one_star.get_los_column_density(
                bad_gas,
                "unitless_density",
                self._kernel(),
                force_loop=1,
                min_count=10,
            )

    def test_tau_v_unit_conversion_uses_surface_density_units(
        self, one_star, one_gas_front
    ):
        """Tau_v conversion uses unit conversion rather than raw relabeling."""
        gal = Galaxy(
            stars=one_star,
            gas=one_gas_front,
            redshift=0.0,
            centre=None,
        )

        los_dustsds = one_star.get_los_column_density(
            one_gas_front,
            "dust_masses",
            self._kernel(),
            force_loop=1,
            min_count=10,
        )

        expected_units = (
            one_gas_front.dust_masses.units
            / one_gas_front.coordinates.units**2
        )

        assert los_dustsds.units == expected_units
        assert np.allclose(
            unyt_to_ndview(los_dustsds.copy(), Msun / pc**2),
            [1e-8 * self._UNIFORM_PROJECTED_CENTRE],
        )

        tau = gal.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=self._kernel(),
            force_loop=1,
            min_count=10,
        )

        assert np.allclose(
            tau,
            np.array([2e-8 * self._UNIFORM_PROJECTED_CENTRE]),
        )
        assert not hasattr(tau, "units")
        assert not hasattr(one_star.tau_v, "units")

    def test_column_density_behind_zero(self, one_star, one_gas_behind):
        """Test Gas particle behind column density."""
        gal = Galaxy(
            stars=one_star, gas=one_gas_behind, redshift=0.0, centre=None
        )
        tau = gal.get_stellar_los_tau_v(
            kappa=5.0,
            kernel=self._kernel(),
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
        kernel = self._kernel()

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

    def test_column_density_point_los_requires_kernel_object(
        self, one_star, one_gas_front
    ):
        """Point LOS should reject raw projected-kernel arrays."""
        with pytest.raises(InconsistentArguments, match="Kernel instance"):
            one_star.get_los_column_density(
                one_gas_front,
                "dust_masses",
                np.array([1.0]),
                force_loop=1,
                min_count=10,
            )

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

    def test_column_density_smoothed_input_tree_matches_force_loop(self):
        """Test the smoothed-input tree path matches the loop reference."""
        rng = np.random.default_rng(42)
        nstars = 4
        ngas = 24

        stars = Stars(
            initial_masses=np.ones(nstars) * Msun,
            ages=np.ones(nstars) * Myr,
            metallicities=np.full(nstars, 0.02),
            redshift=0.0,
            tau_v=np.zeros(nstars),
            coordinates=rng.normal(0.0, 0.25, size=(nstars, 3)) * Mpc,
        )
        stars.smoothing_lengths = np.full(nstars, 0.3) * Mpc

        gas = Gas(
            masses=np.full(ngas, 1e6) * Msun,
            metallicities=np.full(ngas, 0.01),
            redshift=0.0,
            coordinates=rng.normal(0.0, 0.3, size=(ngas, 3)) * Mpc,
            dust_to_metal_ratio=1.0,
            smoothing_lengths=np.full(ngas, 0.25) * Mpc,
        )

        galaxy = Galaxy(
            stars=stars,
            gas=gas,
            redshift=0.0,
            centre=None,
        )
        kernel = Kernel(name="uniform", binsize=32)

        tau_loop = galaxy.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=1,
            min_count=10,
        )
        tau_tree = galaxy.get_stellar_los_tau_v(
            kappa=2.0,
            kernel=kernel,
            as_points=False,
            force_loop=0,
            min_count=4,
        )

        assert np.allclose(tau_tree, tau_loop, rtol=2e-2, atol=0.0)

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

    def test_uniform_overlap_matches_reference_fully_front(self):
        """Test a fully front uniform-kernel overlap against a reference."""
        self._assert_uniform_overlap_matches_reference(
            star_position=(0.0, 0.0, 1.0),
            gas_position=(0.0, 0.0, -1.0),
        )

    def test_uniform_overlap_matches_reference_fully_front_offset(self):
        """Test an offset fully front overlap against a reference."""
        self._assert_uniform_overlap_matches_reference(
            star_position=(0.0, 0.0, 1.0),
            gas_position=(0.6, 0.0, -1.0),
        )

    def test_uniform_overlap_matches_reference_straddling(self):
        """Test a straddling uniform-kernel overlap against a reference."""
        self._assert_uniform_overlap_matches_reference(
            star_position=(0.0, 0.0, 1.0),
            gas_position=(0.0, 0.0, 0.5),
        )

    def test_uniform_overlap_matches_reference_straddling_offset(self):
        """Test an offset straddling overlap against a reference."""
        self._assert_uniform_overlap_matches_reference(
            star_position=(0.0, 0.0, 1.0),
            gas_position=(0.6, 0.0, 0.5),
        )

    def test_uniform_overlap_matches_reference_no_projected_overlap(self):
        """Test a non-overlapping projected geometry against a reference."""
        self._assert_uniform_overlap_matches_reference(
            star_position=(0.0, 0.0, 1.0),
            gas_position=(2.5, 0.0, -1.0),
            expect_non_zero=False,
        )

    def test_uniform_overlap_matches_reference_different_smoothing_lengths(
        self,
    ):
        """Test different smoothing lengths against a reference."""
        self._assert_uniform_overlap_matches_reference(
            star_position=(0.0, 0.0, 1.0),
            gas_position=(0.3, 0.0, 0.4),
            star_smoothing_length=0.6,
            gas_smoothing_length=1.2,
        )

    @pytest.mark.parametrize(
        ("label", "star_position", "gas_position", "expect_non_zero"),
        [
            (
                "input wholly in front",
                (0.0, 0.0, 5.15),
                (0.45, 0.0, 3.2),
                True,
            ),
            (
                "input wholly behind",
                (0.0, 0.0, 1.15),
                (0.45, 0.0, 3.2),
                False,
            ),
            (
                "z overlap, source in front",
                (0.0, 0.0, 3.75),
                (0.45, 0.0, 3.2),
                True,
            ),
            (
                "z overlap, source behind",
                (0.0, 0.0, 2.65),
                (0.45, 0.0, 3.2),
                True,
            ),
        ],
        ids=[
            "wholly-front",
            "wholly-behind",
            "overlap-source-front",
            "overlap-source-behind",
        ],
    )
    def test_uniform_overlap_matches_reference_example_geometries(
        self,
        label,
        star_position,
        gas_position,
        expect_non_zero,
    ):
        """Match the four overlap geometries used in the example figure."""
        del label
        self._assert_uniform_overlap_matches_reference(
            star_position=star_position,
            gas_position=gas_position,
            star_smoothing_length=0.8,
            gas_smoothing_length=1.0,
            expect_non_zero=expect_non_zero,
        )

    @pytest.mark.parametrize(
        "kernel_name",
        ["uniform", "sph_anarchy", "gadget_2", "cubic", "quintic"],
    )
    def test_direct_overlap_matches_reference_all_kernels(self, kernel_name):
        """Test the overlap table against a direct reference."""
        self._assert_direct_overlap_matches_reference(
            kernel_name=kernel_name,
            star_position=(0.0, 0.0, 1.0),
            gas_position=(0.35, 0.0, 0.4),
            star_smoothing_length=0.7,
            gas_smoothing_length=1.1,
        )

    @pytest.mark.parametrize(
        "kernel_name",
        ["uniform", "sph_anarchy", "gadget_2", "cubic", "quintic"],
    )
    def test_direct_overlap_matches_reference_fully_front_all_kernels(
        self,
        kernel_name,
    ):
        """Test fully front overlap saturation against a direct reference."""
        self._assert_direct_overlap_matches_reference(
            kernel_name=kernel_name,
            star_position=(0.0, 0.0, 1.0),
            gas_position=(0.2, 0.0, -1.4),
            star_smoothing_length=0.8,
            gas_smoothing_length=1.0,
        )

    def test_missing_components(self, one_star, one_gas_front):
        """Raises when stars or gas missing."""
        gal_no_gas = Galaxy(
            stars=one_star, gas=None, redshift=0.0, centre=None
        )
        with pytest.raises(InconsistentArguments):
            gal_no_gas.get_stellar_los_tau_v(kappa=1.0, kernel=self._kernel())

        gal_no_star = Galaxy(
            stars=None, gas=one_gas_front, redshift=0.0, centre=None
        )
        with pytest.raises(InconsistentArguments):
            gal_no_star.get_stellar_los_tau_v(kappa=1.0, kernel=self._kernel())

    def test_column_density_inside_source_kernel_is_partial(self):
        """A star inside a foreground gas kernel sees a partial column."""
        star = Stars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([1.0]) * Myr,
            metallicities=np.array([0.02]),
            redshift=0.0,
            tau_v=np.array([0.0]),
            coordinates=np.array([[0.0, 0.0, 0.5]]) * Mpc,
        )
        star.smoothing_lengths = np.array([1.0]) * Mpc

        gas = Gas(
            masses=np.array([1e6]) * Msun,
            metallicities=np.array([0.01]),
            redshift=0.0,
            coordinates=np.array([[0.0, 0.0, 0.0]]) * Mpc,
            dust_to_metal_ratio=1.0,
            smoothing_lengths=np.array([1.0]) * Mpc,
        )

        full_star = Stars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([1.0]) * Myr,
            metallicities=np.array([0.02]),
            redshift=0.0,
            tau_v=np.array([0.0]),
            coordinates=np.array([[0.0, 0.0, 2.0]]) * Mpc,
        )
        full_star.smoothing_lengths = np.array([1.0]) * Mpc

        partial = star.get_los_column_density(
            gas,
            "dust_masses",
            self._kernel(),
            force_loop=1,
            min_count=10,
        )
        full = full_star.get_los_column_density(
            gas,
            "dust_masses",
            self._kernel(),
            force_loop=1,
            min_count=10,
        )

        assert partial[0] > 0.0
        assert partial[0] < full[0]

    def test_point_los_respects_truncated_q_and_z_bins(
        self, one_star, one_gas_front
    ):
        """Point LOS should use the dedicated truncated-table dimensions."""
        coarse = self._kernel(
            binsize=32,
            truncated_q_binsize=8,
            truncated_z_binsize=17,
        )
        fine = self._kernel(
            binsize=32,
            truncated_q_binsize=64,
            truncated_z_binsize=257,
        )

        coarse_tau = one_star.get_los_column_density(
            one_gas_front,
            "dust_masses",
            coarse,
            force_loop=1,
            min_count=10,
        )
        fine_tau = one_star.get_los_column_density(
            one_gas_front,
            "dust_masses",
            fine,
            force_loop=1,
            min_count=10,
        )

        assert np.all(np.isfinite(coarse_tau))
        assert np.all(np.isfinite(fine_tau))
        assert np.all(coarse_tau > 0.0)
        assert np.all(fine_tau > 0.0)
