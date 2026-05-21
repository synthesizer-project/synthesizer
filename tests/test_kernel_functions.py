"""Tests for the LOS kernel lookup helpers and SPH kernel functions."""

import h5py
import numpy as np
import pytest
from scipy import integrate

from synthesizer import kernel_functions
from synthesizer.kernel_functions import Kernel

KERNEL_NAMES = (
    "uniform",
    "sph_anarchy",
    "gadget_2",
    "cubic",
    "quartic",
    "quintic",
)

KERNEL_BREAKPOINTS = {
    "uniform": (),
    "sph_anarchy": (),
    "gadget_2": (0.5,),
    "cubic": (0.5,),
    "quartic": (0.2, 0.6),
    "quintic": (1.0 / 3.0, 2.0 / 3.0),
}


def _quad_projected_kernel(kernel_name, binsize):
    """Build a projected kernel table using scipy quad as reference."""
    kernel = Kernel(name=kernel_name, binsize=binsize)
    bins = kernel._get_bins()
    reference = np.zeros(binsize + 1)

    for ii, impact_parameter in enumerate(bins[:-1]):
        value, _ = integrate.quad(
            kernel._integral_func(impact_parameter),
            0,
            np.sqrt(1.0 - impact_parameter**2),
        )
        reference[ii] = value * 2.0

    return reference


def _reference_kernel(kernel_name, r):
    """Evaluate the expected unit-support 3D kernel shape."""
    r = np.asarray(r, dtype=np.float64)
    values = np.zeros_like(r)

    if kernel_name == "uniform":
        values[r < 1.0] = 3.0 / (4.0 * np.pi)
        return values

    if kernel_name == "sph_anarchy":
        mask = r <= 1.0
        rm = 1.0 - r[mask]
        values[mask] = (21.0 / (2.0 * np.pi)) * rm**4 * (1.0 + 4.0 * r[mask])
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
        values[inner] = (8.0 / np.pi) * (
            1.0 - 6.0 * r[inner] ** 2 + 6.0 * r[inner] ** 3
        )
        values[outer] = (16.0 / np.pi) * (1.0 - r[outer]) ** 3
        return values

    if kernel_name == "quartic":
        q = 2.5 * r
        norm = 25.0 / (32.0 * np.pi)
        inner = q < 0.5
        middle = (q >= 0.5) & (q < 1.5)
        outer = (q >= 1.5) & (q < 2.5)
        values[inner] = norm * (
            (2.5 - q[inner]) ** 4
            - 5.0 * (1.5 - q[inner]) ** 4
            + 10.0 * (0.5 - q[inner]) ** 4
        )
        values[middle] = norm * (
            (2.5 - q[middle]) ** 4 - 5.0 * (1.5 - q[middle]) ** 4
        )
        values[outer] = norm * (2.5 - q[outer]) ** 4
        return values

    if kernel_name == "quintic":
        inner = r < 1.0 / 3.0
        middle = (r >= 1.0 / 3.0) & (r < 2.0 / 3.0)
        outer = (r >= 2.0 / 3.0) & (r < 1.0)
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
        values[outer] = 27.0 * 0.64457752 * (1.0 - r[outer]) ** 5
        return values

    raise ValueError(f"Unknown kernel {kernel_name}")


@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_projected_kernel_matches_quad_reference(kernel_name):
    """The C++ projected-kernel builder should match the quad reference."""
    binsize = 64
    kernel = Kernel(name=kernel_name, binsize=binsize)

    projected = kernel.get_kernel()
    reference = _quad_projected_kernel(kernel_name, binsize)

    assert np.allclose(projected, reference, rtol=1e-5, atol=1e-7)


def test_kernel_hdf5_round_trip_preserves_tables(tmp_path):
    """Kernel lookup tables and metadata should round-trip via HDF5."""
    filepath = tmp_path / "kernel_uniform.hdf5"
    kernel = Kernel(
        name="uniform",
        binsize=16,
        truncated_q_binsize=9,
        truncated_z_binsize=11,
        overlap_q_binsize=4,
        overlap_u_binsize=6,
        overlap_eta_binsize=3,
        overlap_eta_min=0.5,
        overlap_eta_max=2.0,
        overlap_build_ndim=4,
        projected_integration_steps=32,
    )

    expected_projected = kernel.get_kernel()
    expected_truncated, expected_truncated_q, expected_truncated_z = (
        kernel.get_truncated_los_kernel()
    )
    (
        expected_overlap,
        expected_overlap_q,
        expected_overlap_u,
        expected_overlap_eta,
    ) = kernel.get_overlap_kernel()

    kernel.create_kernel(filepath=filepath)

    with h5py.File(filepath, "r") as hdf:
        assert "Header" in hdf
        assert "Kernel" in hdf
        assert hdf["Header"].attrs["type"] == "Kernel"
        assert hdf["Header"].attrs["format"] == "kernel_lookup"

        group = hdf["Kernel"]
        assert group.attrs["name"] == "uniform"
        assert group.attrs["binsize"] == 16
        assert group.attrs["truncated_q_binsize"] == 9
        assert group.attrs["truncated_z_binsize"] == 11
        assert group.attrs["overlap_q_binsize"] == 4
        assert group.attrs["overlap_u_binsize"] == 6
        assert group.attrs["overlap_eta_binsize"] == 3
        assert group.attrs["overlap_eta_min"] == 0.5
        assert group.attrs["overlap_eta_max"] == 2.0
        assert group.attrs["overlap_build_ndim"] == 4
        assert group.attrs["projected_integration_steps"] == 32

        np.testing.assert_allclose(
            group["projected_kernel"][...], expected_projected
        )
        np.testing.assert_allclose(
            group["projected_bins"][...], kernel._get_bins()
        )
        np.testing.assert_allclose(
            group["truncated_kernel"][...], expected_truncated
        )
        np.testing.assert_allclose(
            group["truncated_q"][...], expected_truncated_q
        )
        np.testing.assert_allclose(
            group["truncated_z"][...], expected_truncated_z
        )
        np.testing.assert_allclose(
            group["overlap_kernel"][...], expected_overlap
        )
        np.testing.assert_allclose(group["overlap_q"][...], expected_overlap_q)
        np.testing.assert_allclose(group["overlap_u"][...], expected_overlap_u)
        np.testing.assert_allclose(
            group["overlap_eta"][...], expected_overlap_eta
        )

    reloaded = Kernel.load(filepath)

    np.testing.assert_allclose(reloaded.get_kernel(), expected_projected)

    truncated_kernel, truncated_q, truncated_z = (
        reloaded.get_truncated_los_kernel()
    )
    np.testing.assert_allclose(truncated_kernel, expected_truncated)
    np.testing.assert_allclose(truncated_q, expected_truncated_q)
    np.testing.assert_allclose(truncated_z, expected_truncated_z)

    overlap_kernel, overlap_q, overlap_u, overlap_eta = (
        reloaded.get_overlap_kernel()
    )
    np.testing.assert_allclose(overlap_kernel, expected_overlap)
    np.testing.assert_allclose(overlap_q, expected_overlap_q)
    np.testing.assert_allclose(overlap_u, expected_overlap_u)
    np.testing.assert_allclose(overlap_eta, expected_overlap_eta)


@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_kernel_wrappers_match_reference_shapes(kernel_name):
    """Public kernel wrappers should match their analytic definitions."""
    radii = np.array(
        [0.0, 0.1, 0.2, 1.0 / 3.0, 0.49, 0.5, 0.6, 0.7, 0.9, 1.0, 1.1]
    )
    wrapper = getattr(kernel_functions, kernel_name)

    np.testing.assert_allclose(
        wrapper(radii),
        _reference_kernel(kernel_name, radii),
        rtol=1e-10,
        atol=1e-10,
    )
    assert wrapper(0.25) == pytest.approx(
        _reference_kernel(kernel_name, 0.25).item(),
        rel=1e-10,
        abs=1e-10,
    )


@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_kernel_shapes_are_physical(kernel_name):
    """Kernels should be finite, non-negative, and compactly supported."""
    kernel = Kernel(name=kernel_name, binsize=16)
    radii = np.linspace(0.0, 1.2, 128)
    values = kernel.f(radii)

    assert np.all(np.isfinite(values))
    assert np.all(values >= -1e-14)
    np.testing.assert_allclose(values[radii >= 1.0], 0.0, atol=1e-14)


@pytest.mark.parametrize("kernel_name", KERNEL_NAMES)
def test_kernel_is_unit_normalized(kernel_name):
    """The 3D kernels should integrate to unity over their support."""
    kernel = Kernel(name=kernel_name, binsize=16)

    integral, _ = integrate.quad(
        lambda r: 4.0 * np.pi * r * r * kernel.f(r),
        0.0,
        1.0,
        points=KERNEL_BREAKPOINTS[kernel_name],
    )

    assert integral == pytest.approx(1.0, rel=1e-4, abs=1e-6)
