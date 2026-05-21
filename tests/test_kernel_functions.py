"""Tests for the LOS kernel lookup helpers."""

import h5py
import numpy as np
import pytest
from scipy import integrate

from synthesizer.kernel_functions import Kernel


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


@pytest.mark.parametrize(
    "kernel_name",
    ["uniform", "sph_anarchy", "gadget_2", "cubic", "quartic", "quintic"],
)
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
