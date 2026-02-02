"""Tests for C++ extension error handling."""

import numpy as np
import pytest

from synthesizer.extensions.column_density import compute_column_density
from synthesizer.extensions.doppler_particle_spectra import (
    compute_part_seds_with_vel_shift,
)
from synthesizer.extensions.integrated_spectra import compute_integrated_sed
from synthesizer.extensions.integration import trapz_last_axis
from synthesizer.extensions.particle_spectra import compute_particle_seds
from synthesizer.extensions.sfzh import compute_sfzh
from synthesizer.imaging.extensions.circular_aperture import (
    calculate_circular_overlap,
)
from synthesizer.imaging.extensions.image import make_img
from synthesizer.utils.precision import get_numpy_dtype


def test_integration_rejects_wrong_dtype():
    """Ensure C++ integration rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    xs = np.array([0.0, 1.0, 2.0], dtype=wrong_dtype)
    ys = np.array([[0.0, 1.0, 4.0]], dtype=wrong_dtype)

    with pytest.raises(TypeError, match="incorrect dtype"):
        trapz_last_axis(xs, ys, 1)


def test_integration_rejects_non_contiguous():
    """Ensure C++ integration rejects non-contiguous arrays."""
    dtype = get_numpy_dtype()
    xs = np.array([0.0, 1.0, 2.0], dtype=dtype)

    ys_full = np.array([[0.0, 1.0, 4.0], [0.0, 2.0, 8.0]], dtype=dtype)
    ys = ys_full[:, ::2]

    if ys.flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous test array")

    with pytest.raises(ValueError, match="C contiguous"):
        trapz_last_axis(xs, ys, 1)


def test_make_img_rejects_wrong_dtype():
    """Ensure make_img rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    pix_values = np.array([1.0, 2.0], dtype=wrong_dtype)
    smoothing_lengths = np.array([0.1, 0.1], dtype=wrong_dtype)
    coords = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=wrong_dtype)
    kernel = np.array([1.0, 0.5, 0.0], dtype=wrong_dtype)

    with pytest.raises(TypeError, match="incorrect dtype"):
        make_img(
            pix_values,
            smoothing_lengths,
            coords,
            kernel,
            1.0,
            4,
            4,
            2,
            1.0,
            kernel.size,
            1,
            1,
        )


def test_make_img_rejects_non_contiguous():
    """Ensure make_img rejects non-contiguous arrays."""
    dtype = get_numpy_dtype()

    pix_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    smoothing_lengths = np.array([0.1, 0.1, 0.1, 0.1], dtype=dtype)
    coords_full = np.array(
        [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=dtype
    )
    coords = coords_full[::2, :]
    kernel = np.array([1.0, 0.5, 0.0], dtype=dtype)

    if coords.flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous coords array")

    with pytest.raises(ValueError, match="C contiguous"):
        make_img(
            pix_values[:2],
            smoothing_lengths[:2],
            coords,
            kernel,
            1.0,
            4,
            4,
            2,
            1.0,
            kernel.size,
            1,
            1,
        )


def test_circular_aperture_rejects_wrong_dtype():
    """Ensure circular_aperture rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    img = np.zeros((4, 4), dtype=wrong_dtype)
    cent = np.array([1.0, 1.0], dtype=wrong_dtype)

    with pytest.raises(TypeError, match="incorrect dtype"):
        calculate_circular_overlap(1.0, 4, 4, 1.0, img, cent, 1)


def test_circular_aperture_rejects_non_contiguous():
    """Ensure circular_aperture rejects non-contiguous arrays."""
    dtype = get_numpy_dtype()
    img_full = np.zeros((4, 4), dtype=dtype)
    img = img_full[:, ::2]
    cent = np.array([1.0, 1.0], dtype=dtype)

    if img.flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous image")

    with pytest.raises(ValueError, match="C contiguous"):
        calculate_circular_overlap(1.0, 4, 4, 1.0, img, cent, 1)


def test_column_density_rejects_wrong_dtype():
    """Ensure column_density rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    kernel = np.array([1.0, 0.5, 0.0], dtype=wrong_dtype)
    pos_i = np.array([[0.0, 0.0, 0.0]], dtype=wrong_dtype)
    pos_j = np.array([[0.0, 0.0, 0.0]], dtype=wrong_dtype)
    smls = np.array([1.0], dtype=wrong_dtype)
    surf_den = np.array([1.0], dtype=wrong_dtype)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_column_density(
            kernel,
            pos_i,
            pos_j,
            smls,
            surf_den,
            1,
            1,
            kernel.size,
            1.0,
            0,
            1,
            1,
        )


def test_column_density_rejects_non_contiguous():
    """Ensure column_density rejects non-contiguous arrays."""
    dtype = get_numpy_dtype()
    kernel = np.array([1.0, 0.5, 0.0], dtype=dtype)
    pos_full = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=dtype)
    pos_i = pos_full[:1, :]
    pos_j = pos_full[::2, :]
    smls = np.array([1.0, 1.0], dtype=dtype)
    surf_den = np.array([1.0, 1.0], dtype=dtype)

    if pos_j.flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous positions")

    with pytest.raises(ValueError, match="C contiguous"):
        compute_column_density(
            kernel,
            pos_i,
            pos_j,
            smls,
            surf_den,
            1,
            2,
            kernel.size,
            1.0,
            1,
            1,
            1,
        )


def test_particle_spectra_rejects_wrong_dtype():
    """Ensure particle_spectra rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    grid_spectra = np.zeros((1, 2), dtype=wrong_dtype)
    grid_axes = (np.array([0.0], dtype=target_dtype),)
    part_props = (np.array([0.0], dtype=target_dtype),)
    part_mass = np.array([1.0], dtype=target_dtype)
    ndims = np.array([1], dtype=np.int32)
    mask = np.array([True], dtype=np.bool_)
    lam_mask = np.array([True, True], dtype=np.bool_)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_particle_seds(
            grid_spectra,
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            2,
            "cic",
            1,
            mask,
            lam_mask,
        )


def test_particle_spectra_rejects_non_contiguous():
    """Ensure particle_spectra rejects non-contiguous arrays."""
    dtype = get_numpy_dtype()
    grid_spectra_full = np.zeros((2, 2), dtype=dtype)
    grid_spectra = grid_spectra_full[:, ::2]
    grid_axes = (np.array([0.0], dtype=dtype),)
    part_props = (np.array([0.0], dtype=dtype),)
    part_mass = np.array([1.0], dtype=dtype)
    ndims = np.array([1], dtype=np.int32)
    mask = np.array([True], dtype=np.bool_)
    lam_mask = np.array([True, True], dtype=np.bool_)

    if grid_spectra.flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous grid spectra")

    with pytest.raises(ValueError, match="C contiguous"):
        compute_particle_seds(
            grid_spectra,
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            2,
            "cic",
            1,
            mask,
            lam_mask,
        )


def test_particle_spectra_rejects_mask_dtype():
    """Ensure particle_spectra rejects non-bool masks."""
    dtype = get_numpy_dtype()
    grid_spectra = np.zeros((1, 2), dtype=dtype)
    grid_axes = (np.array([0.0], dtype=dtype),)
    part_props = (np.array([0.0], dtype=dtype),)
    part_mass = np.array([1.0], dtype=dtype)
    ndims = np.array([1], dtype=np.int32)
    mask = np.array([1], dtype=np.int32)
    lam_mask = np.array([True, True], dtype=np.bool_)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_particle_seds(
            grid_spectra,
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            2,
            "cic",
            1,
            mask,
            lam_mask,
        )


def test_integrated_sed_rejects_wrong_dtype():
    """Ensure integrated_sed rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    grid_spectra = np.zeros((1, 2), dtype=wrong_dtype)
    grid_axes = (np.array([0.0], dtype=target_dtype),)
    part_props = (np.array([0.0], dtype=target_dtype),)
    part_mass = np.array([1.0], dtype=target_dtype)
    ndims = np.array([1], dtype=np.int32)
    grid_weights = np.zeros((1,), dtype=target_dtype)
    mask = np.array([True], dtype=np.bool_)
    lam_mask = np.array([True, True], dtype=np.bool_)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_integrated_sed(
            grid_spectra,
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            2,
            "cic",
            1,
            grid_weights,
            mask,
            lam_mask,
        )


def test_integrated_sed_rejects_non_contiguous():
    """Ensure integrated_sed rejects non-contiguous arrays."""
    dtype = get_numpy_dtype()
    grid_spectra_full = np.zeros((2, 2), dtype=dtype)
    grid_spectra = grid_spectra_full[:, ::2]
    grid_axes = (np.array([0.0], dtype=dtype),)
    part_props = (np.array([0.0], dtype=dtype),)
    part_mass = np.array([1.0], dtype=dtype)
    ndims = np.array([1], dtype=np.int32)
    grid_weights = np.zeros((1,), dtype=dtype)
    mask = np.array([True], dtype=np.bool_)
    lam_mask = np.array([True, True], dtype=np.bool_)

    if grid_spectra.flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous grid spectra")

    with pytest.raises(ValueError, match="C contiguous"):
        compute_integrated_sed(
            grid_spectra,
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            2,
            "cic",
            1,
            grid_weights,
            mask,
            lam_mask,
        )


def test_doppler_particle_spectra_rejects_wrong_dtype():
    """Ensure doppler_particle_spectra rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    grid_spectra = np.zeros((1, 2), dtype=wrong_dtype)
    grid_lam = np.array([1.0, 2.0], dtype=target_dtype)
    grid_axes = (np.array([0.0], dtype=target_dtype),)
    part_props = (np.array([0.0], dtype=target_dtype),)
    part_mass = np.array([1.0], dtype=target_dtype)
    velocities = np.zeros((1, 3), dtype=target_dtype)
    ndims = np.array([1], dtype=np.int32)
    mask = np.array([True], dtype=np.bool_)
    lam_mask = np.array([True, True], dtype=np.bool_)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_part_seds_with_vel_shift(
            grid_spectra,
            grid_lam,
            grid_axes,
            part_props,
            part_mass,
            velocities,
            ndims,
            1,
            1,
            2,
            "cic",
            1,
            3e5,
            mask,
            lam_mask,
        )


def test_doppler_particle_spectra_rejects_non_contiguous():
    """Ensure doppler_particle_spectra rejects non-contiguous arrays."""
    dtype = get_numpy_dtype()
    grid_spectra_full = np.zeros((2, 2), dtype=dtype)
    grid_spectra = grid_spectra_full[:, ::2]
    grid_lam = np.array([1.0, 2.0], dtype=dtype)
    grid_axes = (np.array([0.0], dtype=dtype),)
    part_props = (np.array([0.0], dtype=dtype),)
    part_mass = np.array([1.0], dtype=dtype)
    velocities = np.zeros((1, 3), dtype=dtype)
    ndims = np.array([1], dtype=np.int32)
    mask = np.array([True], dtype=np.bool_)
    lam_mask = np.array([True, True], dtype=np.bool_)

    if grid_spectra.flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous grid spectra")

    with pytest.raises(ValueError, match="C contiguous"):
        compute_part_seds_with_vel_shift(
            grid_spectra,
            grid_lam,
            grid_axes,
            part_props,
            part_mass,
            velocities,
            ndims,
            1,
            1,
            2,
            "cic",
            1,
            3e5,
            mask,
            lam_mask,
        )


def test_compute_grid_weights_rejects_wrong_dtype():
    """Ensure compute_grid_weights rejects incorrect dtype arrays."""
    compute_weights_module = pytest.importorskip(
        "synthesizer.extensions.weights",
        reason="weights extension not available",
    )
    compute_grid_weights = compute_weights_module.compute_grid_weights
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    grid_axes = (np.array([0.0], dtype=target_dtype),)
    part_props = (np.array([0.0], dtype=target_dtype),)
    part_mass = np.array([1.0], dtype=wrong_dtype)
    ndims = np.array([1], dtype=np.int32)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_grid_weights(
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            "cic",
            1,
        )


def test_compute_grid_weights_rejects_non_contiguous():
    """Ensure compute_grid_weights rejects non-contiguous arrays."""
    compute_weights_module = pytest.importorskip(
        "synthesizer.extensions.weights",
        reason="weights extension not available",
    )
    compute_grid_weights = compute_weights_module.compute_grid_weights
    dtype = get_numpy_dtype()
    grid_axes = (np.array([0.0], dtype=dtype),)
    part_props_full = np.array([[0.0, 1.0]], dtype=dtype)
    part_props = (part_props_full[:, ::2],)
    part_mass = np.array([1.0], dtype=dtype)
    ndims = np.array([1], dtype=np.int32)

    if part_props[0].flags["C_CONTIGUOUS"]:
        pytest.skip("Could not create non-contiguous part props")

    with pytest.raises(ValueError, match="C contiguous"):
        compute_grid_weights(
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            "cic",
            1,
        )


def test_compute_sfzh_rejects_wrong_dtype():
    """Ensure compute_sfzh rejects incorrect dtype arrays."""
    target_dtype = get_numpy_dtype()
    wrong_dtype = np.float64 if target_dtype == np.float32 else np.float32

    grid_axes = (np.array([0.0], dtype=target_dtype),)
    part_props = (np.array([0.0], dtype=target_dtype),)
    part_mass = np.array([1.0], dtype=wrong_dtype)
    ndims = np.array([1], dtype=np.int32)
    mask = np.array([True], dtype=np.bool_)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_sfzh(
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            "cic",
            1,
            mask,
        )


def test_compute_sfzh_rejects_mask_dtype():
    """Ensure compute_sfzh rejects non-bool masks."""
    dtype = get_numpy_dtype()
    grid_axes = (np.array([0.0], dtype=dtype),)
    part_props = (np.array([0.0], dtype=dtype),)
    part_mass = np.array([1.0], dtype=dtype)
    ndims = np.array([1], dtype=np.int32)
    mask = np.array([1], dtype=np.int32)

    with pytest.raises(TypeError, match="incorrect dtype"):
        compute_sfzh(
            grid_axes,
            part_props,
            part_mass,
            ndims,
            1,
            1,
            "cic",
            1,
            mask,
        )
