"""Tests for particle spatial resampling (Gas, Stars, BlackHoles)."""

import numpy as np
import pytest
from unyt import Mpc, Msun, Myr, km, s, unyt_array

from synthesizer import check_atomic_timing, check_openmp, exceptions
from synthesizer.kernel_functions import Kernel
from synthesizer.particle.blackholes import BlackHoles
from synthesizer.particle.gas import Gas
from synthesizer.particle.particles import Particles
from synthesizer.particle.resample_utils import (
    RESAMPLE_MODES,
    add_velocity_dispersion,
    deterministic_kernel_offsets,
    resample_by_mode,
    sample_kernel_positions,
    validate_mask,
    validate_resample_factor,
)
from synthesizer.particle.sph_density import evaluate_sph_field
from synthesizer.particle.stars import Stars, sample_sfzh
from synthesizer.utils.operation_timers import OperationTimers


def _resample_gas(gas, factor, **kwargs):
    """Helper: call the Gas instance method."""
    kwargs.setdefault("kernel", Kernel("cubic"))
    return gas.spatially_resample(factor, **kwargs)


def _resample_stars(stars, factor, **kwargs):
    """Helper: call the Stars instance method."""
    kwargs.setdefault("kernel", Kernel("cubic"))
    return stars.spatially_resample(factor, **kwargs)


def _make_gas(n=20):
    rng = np.random.default_rng(42)
    return Gas(
        masses=np.ones(n) * 1e6 * Msun,
        metallicities=rng.uniform(0.001, 0.03, n),
        coordinates=rng.uniform(-10, 10, (n, 3)) * Mpc,
        velocities=rng.uniform(-100, 100, (n, 3)) * km / s,
        smoothing_lengths=np.ones(n) * 0.5 * Mpc,
        softening_lengths=np.ones(n) * 0.1 * Mpc,
        redshift=0.1,
    )


def _make_stars(n=15):
    rng = np.random.default_rng(42)
    return Stars(
        initial_masses=np.ones(n) * 1e6 * Msun,
        ages=rng.uniform(10, 1000, n) * Myr,
        metallicities=rng.uniform(0.001, 0.03, n),
        coordinates=rng.uniform(-10, 10, (n, 3)) * Mpc,
        velocities=rng.uniform(-100, 100, (n, 3)) * km / s,
        smoothing_lengths=np.ones(n) * 0.5 * Mpc,
        softening_lengths=np.ones(n) * 0.1 * Mpc,
        redshift=0.1,
    )


def _make_overlap_gas():
    """Create two overlapping gas particles with contrasting metallicities."""
    return Gas(
        masses=np.array([1.0, 1.0]) * Msun,
        metallicities=np.array([0.0, 1.0]),
        coordinates=np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]) * Mpc,
        velocities=np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]]) * km / s,
        smoothing_lengths=np.array([1.0, 1.0]) * Mpc,
        softening_lengths=np.array([0.1, 0.1]) * Mpc,
        redshift=0.1,
    )


def _make_overlap_stars():
    """Create two overlapping stars with contrasting ages and metallicities."""
    return Stars(
        initial_masses=np.array([1.0, 1.0]) * Msun,
        ages=np.array([100.0, 1000.0]) * Myr,
        metallicities=np.array([0.001, 0.03]),
        coordinates=np.array([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]]) * Mpc,
        velocities=np.array([[5.0, 0.0, 0.0], [-5.0, 0.0, 0.0]]) * km / s,
        smoothing_lengths=np.array([1.0, 1.0]) * Mpc,
        softening_lengths=np.array([0.1, 0.1]) * Mpc,
        redshift=0.1,
    )


def _set_omp_threads(monkeypatch, nthreads):
    """Set the OpenMP thread count for the duration of a test."""
    monkeypatch.setenv("OMP_NUM_THREADS", str(nthreads))
    monkeypatch.setenv("OMP_DYNAMIC", "FALSE")


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------


class TestSampleKernelPositions:
    """Tests for :func:`sample_kernel_positions`."""

    def test_shape(self):
        """Output shape is (N, n_samples, 3) and dtype is float64."""
        kernel = Kernel("cubic")
        smls = np.ones(10, dtype=np.float64)
        offsets = sample_kernel_positions(kernel, smls, 3, seed=42)
        assert offsets.shape == (10, 3, 3)
        assert offsets.dtype == np.float64

    def test_radii_within_smoothing_length(self):
        """All sampled offsets lie within the kernel extent (h)."""
        kernel = Kernel("cubic")
        smls = np.ones(5, dtype=np.float64) * 2.0
        offsets = sample_kernel_positions(kernel, smls, 100, seed=42)
        radii = np.linalg.norm(offsets, axis=2)
        assert np.all(radii <= 2.0)
        assert np.any(radii > 0.0)


class TestValidateResampleFactor:
    """Tests for :func:`validate_resample_factor`."""

    def test_valid(self):
        """factor=2 is accepted."""
        validate_resample_factor(2)

    def test_rejects_one(self):
        """factor=1 raises ValueError."""
        with pytest.raises(ValueError, match=">= 2"):
            validate_resample_factor(1)

    def test_rejects_zero(self):
        """factor=0 raises ValueError."""
        with pytest.raises(ValueError, match=">= 2"):
            validate_resample_factor(0)

    def test_rejects_float(self):
        """Non-integer factors raise ValueError."""
        with pytest.raises(ValueError, match="must be an integer"):
            validate_resample_factor(2.5)


class TestDeterministicKernelOffsets:
    """Tests for deterministic field-mode kernel offsets."""

    def test_repeatable(self):
        """Repeated calls return identical offset patterns."""
        kernel = Kernel("cubic")
        first = deterministic_kernel_offsets(kernel, 8)
        second = deterministic_kernel_offsets(kernel, 8)
        assert np.allclose(first, second)

    def test_within_unit_support(self):
        """All deterministic offsets remain inside the unit support radius."""
        kernel = Kernel("cubic")
        offsets = deterministic_kernel_offsets(kernel, 32)
        assert np.all(np.linalg.norm(offsets, axis=1) < 1.0)


class TestValidateMask:
    """Tests for :func:`validate_mask`."""

    def test_valid(self):
        """A correct-length bool mask is returned as-is."""
        mask = np.array([True, False, True])
        result = validate_mask(mask, 3)
        assert result.dtype == bool
        assert np.array_equal(result, mask)

    def test_wrong_length(self):
        """A mask with the wrong length raises InconsistentArguments."""
        with pytest.raises(exceptions.InconsistentArguments):
            validate_mask(np.array([True, False]), 3)

    def test_rejects_non_boolean_dtype(self):
        """Non-boolean masks raise InconsistentArguments."""
        with pytest.raises(exceptions.InconsistentArguments, match="boolean"):
            validate_mask(np.array([0, 2, -1]), 3)


class TestResampleByMode:
    """Tests for all five resampling modes in :func:`resample_by_mode`."""

    def setup_method(self):
        """Create a shared RNG and test array for all tests."""
        self.rng = np.random.default_rng(42)
        self.arr = np.array([10.0, 20.0, 30.0])
        self.factor = 3

    def test_known_modes(self):
        """All five mode names are registered in RESAMPLE_MODES."""
        assert "duplicated" in RESAMPLE_MODES
        assert "proportional" in RESAMPLE_MODES
        assert "normal" in RESAMPLE_MODES
        assert "lognormal" in RESAMPLE_MODES
        assert "conserved_normal" in RESAMPLE_MODES

    def test_unknown_mode_raises(self):
        """An unrecognised mode string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown resampling mode"):
            resample_by_mode(self.arr, "blorp", self.factor, self.rng)

    def test_duplicated_values(self):
        """``"duplicated"`` repeats each element factor times."""
        out = resample_by_mode(self.arr, "duplicated", self.factor, self.rng)
        assert out.shape == (9,)
        assert np.allclose(out, np.repeat(self.arr, self.factor))

    def test_proportional_sum_conserved(self):
        """``"proportional"`` conserves the total sum."""
        out = resample_by_mode(self.arr, "proportional", self.factor, self.rng)
        assert out.shape == (9,)
        assert np.isclose(out.sum(), self.arr.sum())
        assert np.allclose(out, np.repeat(self.arr / self.factor, self.factor))

    def test_normal_shape_and_scatter(self):
        """``"normal"`` produces scattered (not identical) children."""
        out = resample_by_mode(self.arr, "normal", self.factor, self.rng)
        assert out.shape == (9,)
        assert not np.allclose(out, np.repeat(self.arr, self.factor))

    def test_normal_custom_sigma(self):
        """sigma=0 in ``"normal"`` mode recovers duplication."""
        out = resample_by_mode(
            self.arr, ("normal", 0.0), self.factor, self.rng
        )
        assert np.allclose(out, np.repeat(self.arr, self.factor))

    def test_lognormal_positive(self):
        """``"lognormal"`` produces strictly positive children."""
        out = resample_by_mode(self.arr, "lognormal", self.factor, self.rng)
        assert out.shape == (9,)
        assert np.all(out > 0)

    def test_lognormal_rejects_non_positive_inputs(self):
        """``"lognormal"`` rejects zero and negative parent values."""
        with pytest.raises(ValueError, match="strictly positive"):
            resample_by_mode(
                np.array([1.0, 0.0, -1.0]),
                "lognormal",
                self.factor,
                self.rng,
            )

    def test_conserved_normal_sum(self):
        """``"conserved_normal"`` children sum exactly to each parent."""
        out = resample_by_mode(
            self.arr, "conserved_normal", self.factor, self.rng
        )
        assert out.shape == (9,)
        for i, parent in enumerate(self.arr):
            child_sum = out[i * self.factor : (i + 1) * self.factor].sum()
            assert np.isclose(child_sum, parent, rtol=1e-10)

    def test_unyt_units_preserved(self):
        """Unyt array input → unyt array output with same units."""
        arr = unyt_array([10.0, 20.0], "Msun")
        out = resample_by_mode(arr, "proportional", 2, self.rng)
        assert hasattr(out, "units")
        assert str(out.units) == str(arr.units)

    def test_bool_dtype_preserved_duplicated(self):
        """Boolean arrays stay boolean so they can be used as masks."""
        arr = np.array([True, False, True])
        out = resample_by_mode(arr, "duplicated", 2, self.rng)
        assert out.dtype == np.bool_
        assert np.array_equal(out, np.repeat(arr, 2))


class TestAddVelocityDispersion:
    """Tests for :func:`add_velocity_dispersion`."""

    def test_shape_and_noise(self):
        """Output shape matches input; noise adds non-zero std."""
        vels = np.zeros((10, 3), dtype=np.float64)
        result = add_velocity_dispersion(vels, 1.0, seed=42)
        assert result.shape == (10, 3)
        assert not np.allclose(result, 0.0)
        assert np.std(result) > 0.0


class TestSphFieldEvaluation:
    """Tests for the SPH field-evaluation helpers."""

    def test_overlap_case_returns_mixed_field_values(self):
        """Overlapping kernels produce sensible mixed field values."""
        gas = _make_overlap_gas()
        query_positions = (
            np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=np.float64)
            * Mpc
        )
        attributes = {
            "metallicities": gas.metallicities,
            "velocities": gas.velocities,
        }

        density, interpolated = evaluate_sph_field(
            query_positions=query_positions,
            particle_positions=gas.coordinates,
            smoothing_lengths=gas.smoothing_lengths,
            masses=gas.masses,
            kernel=Kernel("cubic"),
            attributes=attributes,
        )

        assert density.shape == (2,)
        assert np.all(density.value > 0.0)
        assert interpolated["metallicities"].shape == (2,)
        assert interpolated["velocities"].shape == (2, 3)
        assert np.all(interpolated["metallicities"] > 0.0)
        assert np.all(interpolated["metallicities"] < 1.0)

    @pytest.mark.skipif(
        not check_openmp(),
        reason="Requires the C++ SPH evaluator with OpenMP enabled.",
    )
    def test_cpp_wrapper_matches_between_thread_counts(self, monkeypatch):
        """Threaded and serial evaluator runs agree numerically."""
        gas = _make_gas(n=24)
        query_positions = gas.coordinates[:12]
        attributes = {
            "metallicities": gas.metallicities,
            "velocities": gas.velocities,
        }

        _set_omp_threads(monkeypatch, 1)
        density_single, attrs_single = evaluate_sph_field(
            query_positions=query_positions,
            particle_positions=gas.coordinates,
            smoothing_lengths=gas.smoothing_lengths,
            masses=gas.masses,
            kernel=Kernel("cubic"),
            attributes=attributes,
            nthreads=1,
        )

        _set_omp_threads(monkeypatch, 4)
        density_multi, attrs_multi = evaluate_sph_field(
            query_positions=query_positions,
            particle_positions=gas.coordinates,
            smoothing_lengths=gas.smoothing_lengths,
            masses=gas.masses,
            kernel=Kernel("cubic"),
            attributes=attributes,
            nthreads=4,
        )

        assert np.allclose(density_single.value, density_multi.value)
        assert np.allclose(
            attrs_single["metallicities"], attrs_multi["metallicities"]
        )
        assert np.allclose(
            attrs_single["velocities"].value,
            attrs_multi["velocities"].value,
        )

    @pytest.mark.skipif(
        not check_atomic_timing(),
        reason="Requires ATOMIC_TIMING to be enabled.",
    )
    def test_field_evaluator_records_operation_timers(self):
        """The public and C++ SPH evaluator timings appear."""
        timers = OperationTimers()
        timers.reset()

        gas = _make_overlap_gas()
        _density, _attrs = evaluate_sph_field(
            query_positions=gas.coordinates,
            particle_positions=gas.coordinates,
            smoothing_lengths=gas.smoothing_lengths,
            masses=gas.masses,
            kernel=Kernel("cubic"),
            attributes={"metallicities": gas.metallicities},
        )

        names = set(timers.keys())
        assert "particle.evaluate_sph_field" in names
        assert "evaluate_sph_density_cpp" in names


# ---------------------------------------------------------------------------
# Gas.spatially_resample tests
# ---------------------------------------------------------------------------


class TestGasSpatialResample:
    """Tests for :meth:`Gas.spatially_resample`."""

    def test_basic_resample(self):
        """Factor-2 resample doubles the particle count and conserves mass."""
        gas = _make_gas()
        g = _resample_gas(gas, 2, seed=42)
        assert g.nparticles == 40
        assert np.isclose(np.sum(g.masses), np.sum(gas.masses))
        assert g.coordinates.shape == (40, 3)
        assert g.velocities.shape == (40, 3)

    def test_factor_3(self):
        """Factor-3 resample triples particle count."""
        gas = _make_gas(n=10)
        g = _resample_gas(gas, 3, seed=42)
        assert g.nparticles == 30

    def test_factor_10(self):
        """Factor-10 resample on 5 particles yields 50."""
        gas = _make_gas(n=5)
        g = _resample_gas(gas, 10, seed=42)
        assert g.nparticles == 50

    def test_mass_conservation(self):
        """Total gas mass is conserved for factors 2, 3, 5."""
        for factor in [2, 3, 5]:
            gas = _make_gas(n=10)
            g = _resample_gas(gas, factor, seed=42)
            assert np.isclose(np.sum(g.masses), np.sum(gas.masses), rtol=1e-10)

    def test_smoothing_length_scaling(self):
        """Smoothing lengths are scaled by factor^{-1/3}."""
        gas = _make_gas()
        g = _resample_gas(gas, 4, seed=42)
        factor = 4.0
        expected_sml = np.repeat(
            gas.smoothing_lengths.value / (factor ** (1.0 / 3.0)), 4
        )
        assert np.allclose(g.smoothing_lengths.value, expected_sml)

    def test_metallicities_duplicated(self):
        """Metallicities are duplicated by default."""
        gas = _make_gas()
        g = _resample_gas(gas, 3, seed=42)
        expected = np.repeat(gas.metallicities, 3)
        assert np.allclose(g.metallicities, expected)

    def test_star_forming_duplicated(self):
        """Star-forming flags are duplicated by default."""
        gas = _make_gas()
        g = _resample_gas(gas, 2, seed=42)
        if gas.star_forming is not None:
            expected = np.repeat(gas.star_forming, 2)
            assert np.array_equal(g.star_forming, expected)

    def test_velocity_dispersion(self):
        """Velocity dispersion adds scatter to the duplicated velocities."""
        gas = _make_gas()
        g = _resample_gas(gas, 2, seed=42, velocity_dispersion=10 * km / s)
        tiled = np.repeat(gas.velocities.value, 2, axis=0)
        assert not np.allclose(g.velocities.value, tiled)

    def test_mask_subset(self):
        """Only masked particles are resampled; others are kept as-is."""
        gas = _make_gas(n=20)
        mask = np.zeros(20, dtype=bool)
        mask[:5] = True
        g = _resample_gas(gas, 2, mask=mask, seed=42)
        assert g.nparticles == 15 + 5 * 2

    def test_mask_all_false(self):
        """A fully-False mask returns the original particle set."""
        gas = _make_gas()
        mask = np.zeros(gas.nparticles, dtype=bool)
        g = _resample_gas(gas, 2, mask=mask, seed=42)
        assert g.nparticles == gas.nparticles
        assert np.isclose(np.sum(g.masses), np.sum(gas.masses))

    def test_mask_all_true_equivalent(self):
        """A fully-True mask produces the same result as no mask."""
        gas = _make_gas()
        mask = np.ones(gas.nparticles, dtype=bool)
        g_masked = _resample_gas(gas, 2, mask=mask, seed=42)
        g_all = _resample_gas(gas, 2, seed=42)
        assert g_masked.nparticles == g_all.nparticles
        assert np.isclose(np.sum(g_masked.masses), np.sum(g_all.masses))

    def test_invalid_mask_shape(self):
        """A mask with the wrong length raises InconsistentArguments."""
        gas = _make_gas()
        with pytest.raises(exceptions.InconsistentArguments):
            _resample_gas(gas, 2, mask=np.array([True, False]))

    def test_no_smoothing_lengths(self):
        """Resample without smoothing_lengths raises InconsistentArguments."""
        rng = np.random.default_rng(42)
        gas = Gas(
            masses=np.ones(5) * Msun,
            metallicities=np.ones(5) * 0.01,
            coordinates=rng.uniform(-1, 1, (5, 3)) * Mpc,
            velocities=rng.uniform(-1, 1, (5, 3)) * km / s,
            smoothing_lengths=None,
            redshift=0.1,
        )
        with pytest.raises(exceptions.InconsistentArguments):
            _resample_gas(gas, 2)

    def test_attr_modes_normal_metallicities(self):
        """``"normal"`` mode scatters metallicity around parent values."""
        gas = _make_gas(n=10)
        g = _resample_gas(
            gas,
            2,
            seed=42,
            attr_modes={"metallicities": "normal"},
        )
        assert g.nparticles == 20
        assert not np.allclose(
            g.metallicities, np.repeat(gas.metallicities, 2)
        )

    def test_attr_modes_proportional_metallicities(self):
        """``"proportional"`` overrides default ``"duplicated"`` for Z."""
        gas = _make_gas(n=10)
        g = _resample_gas(
            gas,
            2,
            seed=42,
            attr_modes={"metallicities": "proportional"},
        )
        expected = np.repeat(gas.metallicities / 2, 2)
        assert np.allclose(g.metallicities, expected)

    def test_attr_modes_conserved_normal_masses(self):
        """``"conserved_normal"`` scatters masses while conserving total."""
        gas = _make_gas(n=5)
        g = _resample_gas(
            gas,
            3,
            seed=42,
            attr_modes={"masses": "conserved_normal"},
        )
        assert np.isclose(g.masses.sum().value, gas.masses.sum().value)

    def test_tau_v_array_is_tiled(self):
        """Per-particle tau_v is duplicated by default."""
        rng = np.random.default_rng(42)
        n = 5
        gas = Gas(
            masses=np.ones(n) * Msun,
            metallicities=np.ones(n) * 0.01,
            coordinates=rng.uniform(-1, 1, (n, 3)) * Mpc,
            velocities=rng.uniform(-1, 1, (n, 3)) * km / s,
            smoothing_lengths=np.ones(n) * 0.5 * Mpc,
            softening_lengths=np.ones(n) * 0.1 * Mpc,
            redshift=0.1,
            tau_v=rng.uniform(0.1, 2.0, n),
        )
        g = _resample_gas(gas, 3, seed=42)
        expected = np.repeat(gas.tau_v, 3)
        assert np.allclose(g.tau_v, expected)

    def test_scalar_custom_attr_is_preserved(self):
        """Scalar custom attrs are copied without resampling."""
        gas = Gas(
            masses=np.ones(4) * Msun,
            metallicities=np.ones(4) * 0.01,
            coordinates=np.zeros((4, 3)) * Mpc,
            velocities=np.zeros((4, 3)) * km / s,
            smoothing_lengths=np.ones(4) * 0.5 * Mpc,
            softening_lengths=np.ones(4) * 0.1 * Mpc,
            redshift=0.1,
            metadata=7,
        )
        resampled = _resample_gas(gas, 2, seed=42)
        assert resampled.metadata == 7

    def test_field_mode_is_repeatable(self):
        """Field mode returns identical coordinates across repeated runs."""
        gas = _make_gas(n=8)
        first = _resample_gas(gas, 3, method="field", seed=1)
        second = _resample_gas(gas, 3, method="field", seed=999)
        assert np.allclose(first.coordinates.value, second.coordinates.value)
        assert np.allclose(first.metallicities, second.metallicities)

    def test_field_mode_conserves_mass(self):
        """Field mode still preserves the total gas mass exactly."""
        gas = _make_gas(n=8)
        resampled = _resample_gas(gas, 4, method="field")
        assert np.isclose(resampled.masses.sum().value, gas.masses.sum().value)

    def test_field_mode_interpolates_overlap_metallicity(self):
        """Overlapping gas kernels yield mixed metallicities in field mode."""
        gas = _make_overlap_gas()
        resampled = _resample_gas(gas, 4, method="field")
        assert np.all(resampled.metallicities >= 0.0)
        assert np.all(resampled.metallicities <= 1.0)
        assert np.any(resampled.metallicities > 0.1)
        assert np.any(resampled.metallicities < 0.9)

    @pytest.mark.skipif(
        not check_openmp(),
        reason="Requires the C++ SPH evaluator with OpenMP enabled.",
    )
    def test_field_mode_matches_between_thread_counts(self, monkeypatch):
        """Gas field-mode resampling agrees across thread counts."""
        gas = _make_gas(n=12)

        _set_omp_threads(monkeypatch, 1)
        single = _resample_gas(gas, 3, method="field", nthreads=1)

        _set_omp_threads(monkeypatch, 4)
        multi = _resample_gas(gas, 3, method="field", nthreads=4)

        assert np.allclose(single.coordinates.value, multi.coordinates.value)
        assert np.allclose(single.metallicities, multi.metallicities)
        assert np.allclose(single.masses.value, multi.masses.value)
        assert np.allclose(single.velocities.value, multi.velocities.value)


# ---------------------------------------------------------------------------
# Stars.spatially_resample tests
# ---------------------------------------------------------------------------


class TestStarsSpatialResample:
    """Tests for :meth:`Stars.spatially_resample` (non-SFZH path)."""

    def test_basic_resample(self):
        """Factor-2 resample doubles particle count and conserves mass."""
        stars = _make_stars()
        s = _resample_stars(stars, 2, seed=42)
        assert s.nparticles == 30
        assert np.isclose(
            np.sum(s.initial_masses), np.sum(stars.initial_masses)
        )

    def test_coordinates_have_units(self):
        """Resampled coordinates retain their units."""
        stars = _make_stars()
        s = _resample_stars(stars, 2, seed=42)
        assert hasattr(s.coordinates, "units")

    def test_velocity_dispersion(self):
        """Velocity dispersion adds noise to duplicated stellar velocities."""
        import unyt as _unyt

        stars = _make_stars()
        resampled = _resample_stars(
            stars,
            2,
            seed=42,
            velocity_dispersion=10 * _unyt.km / _unyt.s,
        )
        tiled = np.repeat(stars.velocities.value, 2, axis=0)
        assert not np.allclose(resampled.velocities.value, tiled)

    def test_mask_subset(self):
        """Partial mask resamples only the selected stars."""
        stars = _make_stars(n=15)
        mask = np.zeros(15, dtype=bool)
        mask[:3] = True
        s = _resample_stars(stars, 3, mask=mask, seed=42)
        assert s.nparticles == 12 + 3 * 3

    def test_mask_all_false(self):
        """A fully-False mask returns the original stars unchanged."""
        stars = _make_stars()
        mask = np.zeros(stars.nparticles, dtype=bool)
        s = _resample_stars(stars, 2, mask=mask, seed=42)
        assert s.nparticles == stars.nparticles

    def test_no_smoothing_lengths(self):
        """Missing smoothing_lengths raises InconsistentArguments."""
        rng = np.random.default_rng(42)
        stars = Stars(
            initial_masses=np.ones(5) * Msun,
            ages=np.ones(5) * 100 * Myr,
            metallicities=np.ones(5) * 0.01,
            coordinates=rng.uniform(-1, 1, (5, 3)) * Mpc,
            velocities=rng.uniform(-1, 1, (5, 3)) * km / s,
            smoothing_lengths=None,
            redshift=0.1,
        )
        with pytest.raises(exceptions.InconsistentArguments):
            _resample_stars(stars, 2)

    def test_attr_modes_lognormal_ages(self):
        """``"lognormal"`` scatters ages while keeping them positive."""
        stars = _make_stars(n=10)
        s = _resample_stars(
            stars,
            2,
            seed=42,
            attr_modes={"ages": "lognormal"},
        )
        assert s.nparticles == 20
        assert np.all(s.ages.value > 0)
        assert not np.allclose(s.ages.value, np.repeat(stars.ages.value, 2))

    def test_attr_modes_proportional_metallicities(self):
        """``"proportional"`` overrides the default ``"duplicated"`` for Z."""
        stars = _make_stars(n=10)
        s = _resample_stars(
            stars,
            2,
            seed=42,
            attr_modes={"metallicities": "proportional"},
        )
        expected = np.repeat(stars.metallicities / 2, 2)
        assert np.allclose(s.metallicities, expected)

    def test_non_sfzh_ages_metallicities_duplicated(self):
        """Without SFZH, ages and metallicities are duplicated by default."""
        stars = _make_stars()
        s = _resample_stars(stars, 2, seed=42)
        assert np.allclose(s.ages.value, np.repeat(stars.ages.value, 2))
        assert np.allclose(s.metallicities, np.repeat(stars.metallicities, 2))

    def test_non_sfzh_mass_conservation(self):
        """Initial and current masses conserve total across factors."""
        for factor in [2, 3, 5]:
            stars = _make_stars(n=10)
            s = _resample_stars(stars, factor, seed=42)
            assert np.isclose(
                np.sum(s.initial_masses),
                np.sum(stars.initial_masses),
                rtol=1e-10,
            )

    def test_non_sfzh_current_masses(self):
        """Current masses are divided proportionally and conserve total."""
        rng = np.random.default_rng(42)
        stars = Stars(
            initial_masses=np.ones(5) * Msun,
            ages=np.ones(5) * 100 * Myr,
            metallicities=np.ones(5) * 0.01,
            coordinates=rng.uniform(-1, 1, (5, 3)) * Mpc,
            velocities=rng.uniform(-1, 1, (5, 3)) * km / s,
            smoothing_lengths=np.ones(5) * 0.5 * Mpc,
            softening_lengths=np.ones(5) * 0.1 * Mpc,
            redshift=0.1,
            current_masses=np.ones(5) * 0.8 * Msun,
        )
        resampled = _resample_stars(stars, 2, seed=42)
        assert np.isclose(
            np.sum(resampled.current_masses),
            np.sum(stars.current_masses),
        )

    def test_array_redshift_is_resampled(self):
        """Per-particle redshift arrays are duplicated cleanly."""
        stars = Stars(
            initial_masses=np.ones(4) * Msun,
            ages=np.ones(4) * 100 * Myr,
            metallicities=np.ones(4) * 0.01,
            coordinates=np.zeros((4, 3)) * Mpc,
            velocities=np.zeros((4, 3)) * km / s,
            smoothing_lengths=np.ones(4) * 0.5 * Mpc,
            softening_lengths=np.ones(4) * 0.1 * Mpc,
            redshift=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        resampled = _resample_stars(stars, 2, seed=42)
        assert np.allclose(resampled.redshift, np.repeat(stars.redshift, 2))

    def test_fesc_attrs_are_preserved(self):
        """Per-particle escape fractions survive resampling."""
        stars = Stars(
            initial_masses=np.ones(4) * Msun,
            ages=np.ones(4) * 100 * Myr,
            metallicities=np.ones(4) * 0.01,
            coordinates=np.zeros((4, 3)) * Mpc,
            velocities=np.zeros((4, 3)) * km / s,
            smoothing_lengths=np.ones(4) * 0.5 * Mpc,
            softening_lengths=np.ones(4) * 0.1 * Mpc,
            redshift=0.1,
            fesc=np.array([0.1, 0.2, 0.3, 0.4]),
            fesc_ly_alpha=np.array([0.4, 0.3, 0.2, 0.1]),
        )
        resampled = _resample_stars(stars, 2, seed=42)
        assert np.allclose(resampled.fesc, np.repeat(stars.fesc, 2))
        assert np.allclose(
            resampled.fesc_ly_alpha,
            np.repeat(stars.fesc_ly_alpha, 2),
        )

    def test_scalar_custom_attr_is_preserved(self):
        """Scalar custom attrs are copied without resampling."""
        stars = Stars(
            initial_masses=np.ones(4) * Msun,
            ages=np.ones(4) * 100 * Myr,
            metallicities=np.ones(4) * 0.01,
            coordinates=np.zeros((4, 3)) * Mpc,
            velocities=np.zeros((4, 3)) * km / s,
            smoothing_lengths=np.ones(4) * 0.5 * Mpc,
            softening_lengths=np.ones(4) * 0.1 * Mpc,
            redshift=0.1,
            metadata=7,
        )
        resampled = _resample_stars(stars, 2, seed=42)
        assert resampled.metadata == 7

    def test_field_mode_is_repeatable(self):
        """Field mode returns identical stellar coordinates across reruns."""
        stars = _make_stars(n=8)
        first = _resample_stars(stars, 3, method="field", seed=7)
        second = _resample_stars(stars, 3, method="field", seed=123)
        assert np.allclose(first.coordinates.value, second.coordinates.value)
        assert np.allclose(first.ages.value, second.ages.value)

    def test_field_mode_interpolates_overlap_properties(self):
        """Field mode mixes ages and metallicities in overlap regions."""
        stars = _make_overlap_stars()
        resampled = _resample_stars(stars, 4, method="field")
        assert np.all(resampled.metallicities >= stars.metallicities.min())
        assert np.all(resampled.metallicities <= stars.metallicities.max())
        assert np.any(resampled.metallicities > stars.metallicities.min())
        assert np.any(resampled.metallicities < stars.metallicities.max())
        assert np.any(resampled.ages.value > stars.ages.min().value)
        assert np.any(resampled.ages.value < stars.ages.max().value)

    @pytest.mark.skipif(
        not check_openmp(),
        reason="Requires the C++ SPH evaluator with OpenMP enabled.",
    )
    def test_field_mode_matches_between_thread_counts(self, monkeypatch):
        """Stellar field-mode resampling agrees across thread counts."""
        stars = _make_stars(n=10)

        _set_omp_threads(monkeypatch, 1)
        single = _resample_stars(stars, 3, method="field", nthreads=1)

        _set_omp_threads(monkeypatch, 4)
        multi = _resample_stars(stars, 3, method="field", nthreads=4)

        assert np.allclose(single.coordinates.value, multi.coordinates.value)
        assert np.allclose(single.ages.value, multi.ages.value)
        assert np.allclose(single.metallicities, multi.metallicities)
        assert np.allclose(
            single.initial_masses.value, multi.initial_masses.value
        )
        assert np.allclose(single.velocities.value, multi.velocities.value)


class TestStarsSpatialResampleWithSFZH:
    """Stars.spatially_resample with an SFZH distribution."""

    @pytest.fixture
    def grid(self):
        """Load the test grid for SFZH sampling."""
        from synthesizer.grid import Grid

        return Grid("test_grid")

    @pytest.fixture
    def sfzh(self, grid):
        """Create a parametric SFZH distribution."""
        from synthesizer.parametric import Stars as ParaStars

        return ParaStars(
            grid.log10ages,
            grid.metallicities,
            sf_hist=1e7 * Myr,
            metal_dist=0.01,
            initial_mass=1e6 * Msun,
        )

    def test_sfzh_resample(self, sfzh):
        """SFZH resample doubles particle count and conserves total mass."""
        stars = _make_stars(n=10)
        s = _resample_stars(stars, 2, sfzh=sfzh, seed=42)
        assert s.nparticles == 20
        assert np.isclose(
            np.sum(s.initial_masses),
            np.sum(stars.initial_masses),
            rtol=1e-10,
        )

    def test_sfzh_ages_sampled(self, sfzh):
        """Ages are sampled from the SFZH and carry units."""
        stars = _make_stars(n=10)
        s = _resample_stars(stars, 2, sfzh=sfzh, seed=42)
        assert hasattr(s.ages, "units")
        assert np.all(s.ages.value > 0)

    def test_sfzh_metallicities_sampled(self, sfzh):
        """Metallicities are sampled from the SFZH and non-negative."""
        stars = _make_stars(n=10)
        s = _resample_stars(stars, 2, sfzh=sfzh, seed=42)
        assert np.all(s.metallicities >= 0)

    def test_sfzh_with_mask(self, sfzh):
        """SFZH resampling with a partial mask produces correct counts."""
        stars = _make_stars(n=10)
        mask = np.zeros(10, dtype=bool)
        mask[:4] = True
        s = _resample_stars(stars, 3, sfzh=sfzh, mask=mask, seed=42)
        assert s.nparticles == 6 + 4 * 3

    def test_sample_sfzh_converts_logged_metallicity_axis(self, sfzh):
        """Public sample_sfzh converts logged metallicity axes to linear Z."""
        stars = sample_sfzh(
            sfzh.sfzh,
            sfzh.log10ages,
            sfzh.log10metallicities,
            20,
        )
        assert np.all(stars.metallicities > 0)
        assert np.all(stars.metallicities >= sfzh.metallicities.min())
        assert np.all(stars.metallicities <= sfzh.metallicities.max())


# ---------------------------------------------------------------------------
# Instance method and _custom_attr_names tests
# ---------------------------------------------------------------------------


class TestInstanceResample:
    """Tests for the instance-method spatial resampling API."""

    def test_gas_instance_method(self):
        """Gas instance method doubles particle count and conserves mass."""
        gas = _make_gas()
        g = gas.spatially_resample(2, seed=42, kernel=Kernel("cubic"))
        assert g.nparticles == gas.nparticles * 2
        assert np.isclose(np.sum(g.masses), np.sum(gas.masses))

    def test_stars_instance_method(self):
        """Stars instance method doubles particle count and conserves mass."""
        stars = _make_stars()
        s = stars.spatially_resample(2, seed=42, kernel=Kernel("cubic"))
        assert s.nparticles == stars.nparticles * 2
        assert np.isclose(
            np.sum(s.initial_masses), np.sum(stars.initial_masses)
        )

    def test_custom_attr_names_registered(self):
        """Extra kwargs at construction appear in ``_custom_attr_names``."""
        rng = np.random.default_rng(42)
        extra = rng.uniform(0, 1, 20)
        gas = Gas(
            masses=np.ones(20) * 1e6 * Msun,
            metallicities=rng.uniform(0.001, 0.03, 20),
            coordinates=rng.uniform(-10, 10, (20, 3)) * Mpc,
            velocities=rng.uniform(-100, 100, (20, 3)) * km / s,
            smoothing_lengths=np.ones(20) * 0.5 * Mpc,
            softening_lengths=np.ones(20) * 0.1 * Mpc,
            redshift=0.1,
            my_custom_field=extra,
        )
        assert "my_custom_field" in gas._custom_attr_names

    def test_custom_attr_propagated_through_instance_resample(self):
        """Custom attrs survive resampling with correct output shape."""
        rng = np.random.default_rng(42)
        n = 20
        extra = rng.uniform(0, 1, n)
        gas = Gas(
            masses=np.ones(n) * 1e6 * Msun,
            metallicities=rng.uniform(0.001, 0.03, n),
            coordinates=rng.uniform(-10, 10, (n, 3)) * Mpc,
            velocities=rng.uniform(-100, 100, (n, 3)) * km / s,
            smoothing_lengths=np.ones(n) * 0.5 * Mpc,
            softening_lengths=np.ones(n) * 0.1 * Mpc,
            redshift=0.1,
            my_custom_field=extra,
        )
        g = gas.spatially_resample(2, seed=42, kernel=Kernel("cubic"))
        assert hasattr(g, "my_custom_field")
        assert g.my_custom_field.shape[0] == n * 2


class TestErrorCases:
    """Tests for error handling in spatial resampling."""

    def test_black_holes_resample_raises(self):
        """BlackHoles resample raises NotImplementedError."""
        bh = BlackHoles(
            masses=np.ones(1) * Msun,
            redshift=0.0,
            coordinates=np.zeros((1, 3)) * Mpc,
        )
        with pytest.raises(NotImplementedError) as excinfo:
            bh.spatially_resample(2)
        assert "nonsensical" in str(excinfo.value).lower()

    def test_particles_base_resample_raises(self):
        """Base Particles resample raises NotImplementedError."""
        p = Particles(
            coordinates=np.zeros((1, 3)) * Mpc,
            velocities=np.zeros((1, 3)) * km / s,
            masses=np.ones(1) * Msun,
            redshift=0.0,
            softening_lengths=np.ones(1) * Mpc,
            nparticles=1,
            centre=np.zeros(3) * Mpc,
        )
        with pytest.raises(NotImplementedError):
            p.spatially_resample(2)

    def test_invalid_factor_gas(self):
        """Gas resample with factor=1 raises ValueError."""
        gas = _make_gas(n=5)
        with pytest.raises(ValueError, match=">= 2"):
            _resample_gas(gas, 1)

    def test_invalid_factor_stars(self):
        """Stars resample with factor=1 raises ValueError."""
        stars = _make_stars(n=5)
        with pytest.raises(ValueError, match=">= 2"):
            _resample_stars(stars, 1)
