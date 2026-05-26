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
        ("x",),
    )

    np.testing.assert_allclose(
        threaded_part_spectra, serial_part_spectra, rtol=0.0, atol=0.0
    )


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
                ("x",),
            )
        )

        np.testing.assert_allclose(
            threaded_part_spectra, serial_part_spectra, rtol=0.0, atol=0.0
        )
        np.testing.assert_allclose(
            threaded_spectra, serial_spectra, rtol=0.0, atol=0.0
        )
