"""Tests for the particle spectra C extension."""

import numpy as np
import pytest
from synthesizer.extensions.doppler_particle_spectra import (
    compute_part_seds_with_vel_shift,
)
from synthesizer.extensions.particle_spectra import compute_particle_seds


def _particle_spectra_inputs():
    """Build a small deterministic 1D grid and particle sample."""
    nlam = 5
    axis = np.ascontiguousarray([0.0, 1.0, 2.0], dtype=np.float64)
    grid_spectra = np.ascontiguousarray(
        np.arange(axis.size * nlam, dtype=np.float64).reshape(axis.size, nlam)
        + 1.0
    )
    part_props = (
        np.ascontiguousarray([0.25, 1.5, -1.0, 3.0], dtype=np.float64),
    )
    weights = np.ascontiguousarray([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
    grid_dims = np.ascontiguousarray([axis.size], dtype=np.int32)

    return grid_spectra, (axis,), part_props, weights, grid_dims


def _expected_ngp_spectra(grid_spectra, weights):
    """Return the expected NGP per-particle spectra."""
    grid_indices = np.array([0, 2, 0, 2], dtype=np.int64)
    return grid_spectra[grid_indices] * weights[:, None]


def _expected_cic_spectra(grid_spectra, weights):
    """Return the expected CIC per-particle spectra."""
    expected = np.zeros(
        (weights.size, grid_spectra.shape[-1]), dtype=np.float64
    )
    expected[0] = (0.75 * grid_spectra[0] + 0.25 * grid_spectra[1]) * weights[
        0
    ]
    expected[1] = (0.5 * grid_spectra[1] + 0.5 * grid_spectra[2]) * weights[1]
    expected[2] = grid_spectra[0] * weights[2]
    expected[3] = grid_spectra[2] * weights[3]
    return expected


@pytest.mark.parametrize(
    ("method", "expected_func"),
    [
        ("ngp", _expected_ngp_spectra),
        ("cic", _expected_cic_spectra),
    ],
)
def test_compute_particle_seds_matches_expected(method, expected_func):
    """Test particle spectra against a deterministic manual calculation."""
    grid_spectra, axes, part_props, weights, grid_dims = (
        _particle_spectra_inputs()
    )

    part_spectra = compute_particle_seds(
        grid_spectra,
        axes,
        part_props,
        weights,
        grid_dims,
        len(axes),
        weights.size,
        grid_spectra.shape[-1],
        method,
        1,
        None,
        None,
        False,
        np.float64,
        ("x",),
    )

    expected = expected_func(grid_spectra, weights)
    np.testing.assert_allclose(part_spectra, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("method", ["ngp", "cic"])
def test_compute_particle_seds_lam_mask_dispatch(method):
    """Test explicit wavelength-mask dispatch in the C extension."""
    grid_spectra, axes, part_props, weights, grid_dims = (
        _particle_spectra_inputs()
    )
    nlam = grid_spectra.shape[-1]

    unmasked_part_spectra = compute_particle_seds(
        grid_spectra,
        axes,
        part_props,
        weights,
        grid_dims,
        len(axes),
        weights.size,
        nlam,
        method,
        1,
        None,
        None,
        False,
        np.float64,
        ("x",),
    )

    all_lam_mask = np.ascontiguousarray(np.ones(nlam, dtype=np.bool_))
    all_mask_part_spectra = compute_particle_seds(
        grid_spectra,
        axes,
        part_props,
        weights,
        grid_dims,
        len(axes),
        weights.size,
        nlam,
        method,
        1,
        None,
        all_lam_mask,
        True,
        np.float64,
        ("x",),
    )

    np.testing.assert_allclose(
        all_mask_part_spectra, unmasked_part_spectra, rtol=0.0, atol=0.0
    )

    partial_lam_mask = np.ascontiguousarray(
        [True, False, True, False, True], dtype=np.bool_
    )
    partial_part_spectra = compute_particle_seds(
        grid_spectra,
        axes,
        part_props,
        weights,
        grid_dims,
        len(axes),
        weights.size,
        nlam,
        method,
        1,
        None,
        partial_lam_mask,
        True,
        np.float64,
        ("x",),
    )

    expected_part_spectra = unmasked_part_spectra.copy()
    expected_part_spectra[:, ~partial_lam_mask] = 0.0

    np.testing.assert_allclose(
        partial_part_spectra, expected_part_spectra, rtol=0.0, atol=0.0
    )


@pytest.mark.parametrize("method", ["ngp", "cic"])
def test_compute_particle_seds_threaded_matches_serial(method):
    """Test threaded and serial particle spectra agree."""
    grid_spectra, axes, part_props, weights, grid_dims = (
        _particle_spectra_inputs()
    )
    nlam = grid_spectra.shape[-1]
    lam_mask = np.ascontiguousarray(
        [True, False, True, True, False], dtype=np.bool_
    )

    serial_part_spectra = compute_particle_seds(
        grid_spectra,
        axes,
        part_props,
        weights,
        grid_dims,
        len(axes),
        weights.size,
        nlam,
        method,
        1,
        None,
        lam_mask,
        True,
        np.float64,
        ("x",),
    )
    threaded_part_spectra = compute_particle_seds(
        grid_spectra,
        axes,
        part_props,
        weights,
        grid_dims,
        len(axes),
        weights.size,
        nlam,
        method,
        2,
        None,
        lam_mask,
        True,
        np.float64,
        ("x",),
    )

    np.testing.assert_allclose(
        threaded_part_spectra, serial_part_spectra, rtol=0.0, atol=0.0
    )


def test_compute_particle_seds_supports_float32_input_and_output():
    """Particle spectra should preserve float32 when requested."""
    nlam = 5
    axis = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    grid_spectra = np.arange(axis.size * nlam, dtype=np.float32).reshape(
        axis.size, nlam
    ) + np.float32(1.0)
    part_props = (np.array([0.25, 1.5, -1.0, 3.0], dtype=np.float32),)
    weights = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    grid_dims = np.array([axis.size], dtype=np.int32)

    part_spectra = compute_particle_seds(
        grid_spectra,
        (axis,),
        part_props,
        weights,
        grid_dims,
        1,
        weights.size,
        nlam,
        "ngp",
        1,
        None,
        None,
        False,
        np.float32,
        ("x",),
    )

    expected = _expected_ngp_spectra(grid_spectra, weights).astype(np.float32)
    np.testing.assert_allclose(part_spectra, expected, rtol=0.0, atol=0.0)
    assert part_spectra.dtype == np.float32


def test_compute_particle_seds_supports_float64_output_from_float32_input():
    """Particle spectra output dtype should be independently configurable."""
    nlam = 5
    axis = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    grid_spectra = np.arange(axis.size * nlam, dtype=np.float32).reshape(
        axis.size, nlam
    ) + np.float32(1.0)
    part_props = (np.array([0.25, 1.5, -1.0, 3.0], dtype=np.float32),)
    weights = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float32)
    grid_dims = np.array([axis.size], dtype=np.int32)

    part_spectra = compute_particle_seds(
        grid_spectra,
        (axis,),
        part_props,
        weights,
        grid_dims,
        1,
        weights.size,
        nlam,
        "ngp",
        1,
        None,
        None,
        False,
        np.float64,
        ("x",),
    )

    expected = _expected_ngp_spectra(grid_spectra, weights).astype(np.float64)
    np.testing.assert_allclose(part_spectra, expected, rtol=0.0, atol=0.0)
    assert part_spectra.dtype == np.float64


class TestDopplerParticleSpectraExtension:
    """Tests for the Doppler particle spectra C extension."""

    @staticmethod
    def _inputs():
        """Build a small deterministic 1D grid and particle sample."""
        nlam = 5
        axis = np.ascontiguousarray([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
        wavelength = np.ascontiguousarray(
            [100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64
        )
        grid_spectra = np.ascontiguousarray(
            np.arange(axis.size * nlam, dtype=np.float64).reshape(
                axis.size, nlam
            )
            + 1.0
        )
        part_props = (np.ascontiguousarray([0.2, 1.5, 2.2], dtype=np.float64),)
        weights = np.ascontiguousarray([2.0, 3.0, 4.0], dtype=np.float64)
        velocities = np.ascontiguousarray([0.0, 10.0, -20.0], dtype=np.float64)
        grid_dims = np.ascontiguousarray([axis.size], dtype=np.int32)
        lam_mask = np.ascontiguousarray(np.ones(nlam, dtype=np.bool_))

        return (
            grid_spectra,
            wavelength,
            (axis,),
            part_props,
            weights,
            velocities,
            grid_dims,
            lam_mask,
        )

    @pytest.mark.parametrize("method", ["ngp", "cic"])
    def test_threaded_matches_serial(self, method):
        """Test threaded and serial Doppler particle spectra agree."""
        (
            grid_spectra,
            wavelength,
            axes,
            part_props,
            weights,
            velocities,
            grid_dims,
            lam_mask,
        ) = self._inputs()

        serial_part_spectra, serial_spectra = compute_part_seds_with_vel_shift(
            grid_spectra,
            wavelength,
            axes,
            part_props,
            weights,
            velocities,
            grid_dims,
            len(axes),
            weights.size,
            grid_spectra.shape[-1],
            method,
            1,
            299792458.0,
            None,
            lam_mask,
            np.float64,
            ("x",),
        )
        threaded_part_spectra, threaded_spectra = (
            compute_part_seds_with_vel_shift(
                grid_spectra,
                wavelength,
                axes,
                part_props,
                weights,
                velocities,
                grid_dims,
                len(axes),
                weights.size,
                grid_spectra.shape[-1],
                method,
                2,
                299792458.0,
                None,
                lam_mask,
                np.float64,
                ("x",),
            )
        )

        np.testing.assert_allclose(
            threaded_part_spectra, serial_part_spectra, rtol=0.0, atol=1e-12
        )
        np.testing.assert_allclose(
            threaded_spectra, serial_spectra, rtol=0.0, atol=1e-12
        )

    def test_supports_float32_input_and_output(self):
        """Doppler particle spectra should preserve float32 when requested."""
        nlam = 5
        axis = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
        wavelength = np.array(
            [100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32
        )
        grid_spectra = np.arange(axis.size * nlam, dtype=np.float32).reshape(
            axis.size, nlam
        ) + np.float32(1.0)
        part_props = (np.array([0.2, 1.5, 2.2], dtype=np.float32),)
        weights = np.array([2.0, 3.0, 4.0], dtype=np.float32)
        velocities = np.array([0.0, 10.0, -20.0], dtype=np.float32)
        grid_dims = np.array([axis.size], dtype=np.int32)
        lam_mask = np.ones(nlam, dtype=np.bool_)

        part_spectra, spectra = compute_part_seds_with_vel_shift(
            grid_spectra,
            wavelength,
            (axis,),
            part_props,
            weights,
            velocities,
            grid_dims,
            1,
            weights.size,
            nlam,
            "ngp",
            1,
            299792458.0,
            None,
            lam_mask,
            np.float32,
            ("x",),
        )

        assert part_spectra.dtype == np.float32
        assert spectra.dtype == np.float32

    @pytest.mark.parametrize("method", ["ngp", "cic"])
    def test_uses_z_velocity_for_3d_velocities(self, method):
        """Test 3D velocities are shifted using the z-axis component."""
        (
            grid_spectra,
            wavelength,
            axes,
            part_props,
            weights,
            velocities,
            grid_dims,
            lam_mask,
        ) = self._inputs()
        velocities_3d = np.ascontiguousarray(
            np.column_stack(
                (
                    velocities + 1000.0,
                    velocities - 1000.0,
                    velocities,
                )
            ),
            dtype=np.float64,
        )

        los_part_spectra, los_spectra = compute_part_seds_with_vel_shift(
            grid_spectra,
            wavelength,
            axes,
            part_props,
            weights,
            velocities,
            grid_dims,
            len(axes),
            weights.size,
            grid_spectra.shape[-1],
            method,
            1,
            100.0,
            None,
            lam_mask,
            np.float64,
            ("x",),
        )
        vel3d_part_spectra, vel3d_spectra = compute_part_seds_with_vel_shift(
            grid_spectra,
            wavelength,
            axes,
            part_props,
            weights,
            velocities_3d,
            grid_dims,
            len(axes),
            weights.size,
            grid_spectra.shape[-1],
            method,
            1,
            100.0,
            None,
            lam_mask,
            np.float64,
            ("x",),
        )

        np.testing.assert_allclose(
            vel3d_part_spectra, los_part_spectra, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            vel3d_spectra, los_spectra, rtol=0.0, atol=0.0
        )

    @pytest.mark.parametrize("method", ["ngp", "cic"])
    def test_velocity_shift_matches_expected_interpolation(self, method):
        """Test Doppler shifts scatter into expected wavelength bins."""
        wavelength = np.ascontiguousarray(
            [100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float64
        )
        axis = np.ascontiguousarray([0.0, 1.0], dtype=np.float64)
        grid_spectra = np.ascontiguousarray(
            [[10.0, 20.0, 30.0, 40.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        part_props = (np.ascontiguousarray([0.0], dtype=np.float64),)
        weights = np.ascontiguousarray([1.0], dtype=np.float64)
        velocities = np.ascontiguousarray([25.0], dtype=np.float64)
        grid_dims = np.ascontiguousarray([axis.size], dtype=np.int32)
        lam_mask = np.ascontiguousarray(
            np.ones(wavelength.size, dtype=np.bool_)
        )

        part_spectra, spectra = compute_part_seds_with_vel_shift(
            grid_spectra,
            wavelength,
            (axis,),
            part_props,
            weights,
            velocities,
            grid_dims,
            1,
            weights.size,
            wavelength.size,
            method,
            1,
            100.0,
            None,
            lam_mask,
            np.float64,
            ("x",),
        )

        expected = np.array([[7.5, 12.5, 17.5, 22.5, 40.0]])
        np.testing.assert_allclose(part_spectra, expected, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(spectra, expected[0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("method", ["ngp", "cic"])
def test_compute_particle_seds_mixed_part_grid_dtypes(method):
    """Mixed particle/grid dtypes must match the all-float64 result.

    This is a regression test for the grid axes being reinterpreted at the
    particle dtype's width when the two families differ.
    """
    grid_spectra, axes, part_props, weights, grid_dims = (
        _particle_spectra_inputs()
    )
    nlam = grid_spectra.shape[-1]

    def run(props, w):
        return compute_particle_seds(
            grid_spectra,
            axes,
            props,
            w,
            grid_dims,
            len(axes),
            w.size,
            nlam,
            method,
            1,
            None,
            None,
            False,
            np.float64,
            ("x",),
        )

    ref = run(part_props, weights)
    mixed = run(
        (part_props[0].astype(np.float32),), weights.astype(np.float32)
    )
    np.testing.assert_allclose(mixed, ref, rtol=1e-6)


@pytest.mark.parametrize("method", ["ngp", "cic"])
def test_compute_integrated_sed_mixed_part_grid_dtypes(method):
    """Integrated spectra must accept mixed particle/grid dtypes."""
    from synthesizer.extensions.integrated_spectra import (
        compute_integrated_sed,
    )

    grid_spectra, axes, part_props, weights, grid_dims = (
        _particle_spectra_inputs()
    )
    nlam = grid_spectra.shape[-1]

    def run(props, w):
        spec, _ = compute_integrated_sed(
            grid_spectra,
            axes,
            props,
            w,
            grid_dims,
            len(axes),
            w.size,
            nlam,
            method,
            1,
            None,
            None,
            None,
            np.float64,
            ("x",),
        )
        return spec

    ref = run(part_props, weights)
    mixed = run(
        (part_props[0].astype(np.float32),), weights.astype(np.float32)
    )
    np.testing.assert_allclose(mixed, ref, rtol=1e-6)
