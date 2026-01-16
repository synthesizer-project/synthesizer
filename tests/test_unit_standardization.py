"""Tests for unit standardization in imaging.

This module tests the _standardize_imaging_units function which ensures
that resolution, fov, and emitter coordinates are all in consistent units
for imaging operations.
"""

import numpy as np
import pytest
from astropy.cosmology import Planck18
from unyt import arcsecond, kpc, unyt_array

from synthesizer import exceptions
from synthesizer.imaging.image_generators import _standardize_imaging_units


class MockEmitter:
    """Mock emitter class for testing."""

    def __init__(self, coords, smoothing_lengths=None, redshift=None):
        """Initialize a mock emitter.

        Args:
            coords (unyt_array):
                Centered coordinates of the emitter.
            smoothing_lengths (unyt_array, optional):
                Smoothing lengths of the emitter. Default is None.
            redshift (float, optional):
                Redshift of the emitter. Default is None.
        """
        self.centered_coordinates = coords
        self.smoothing_lengths = smoothing_lengths
        self.redshift = redshift


class TestStandardizeImagingUnits:
    """Test suite for _standardize_imaging_units function."""

    def test_all_cartesian_no_conversion(self):
        """Test that no conversion occurs when all inputs are Cartesian."""
        # Setup
        resolution = 0.1 * kpc
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        smls = unyt_array(np.random.rand(100), "kpc")
        emitter = MockEmitter(coords, smls, redshift=1.0)

        # Execute
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=True,
        )

        # Assert
        assert res_out == resolution
        assert np.allclose(fov_out.value, fov.value)
        assert fov_out.units == fov.units
        assert np.allclose(coords_out.value, coords.value)
        assert coords_out.units == coords.units
        assert np.allclose(smls_out.value, smls.value)
        assert smls_out.units == smls.units

        # Ensure returned arrays are copies, not references
        assert coords_out is not coords
        assert smls_out is not smls

    def test_all_angular_no_conversion(self):
        """Test that no conversion occurs when all inputs are angular."""
        # Setup
        resolution = 0.1 * arcsecond
        fov = 10.0 * arcsecond
        coords = (
            unyt_array(np.random.rand(100, 3), "arcsecond") - 0.5 * arcsecond
        )
        smls = unyt_array(np.random.rand(100), "arcsecond")
        emitter = MockEmitter(coords, smls, redshift=1.0)

        # Execute
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=True,
        )

        # Assert
        assert res_out == resolution
        assert np.allclose(fov_out.value, fov.value)
        assert fov_out.units == fov.units
        assert np.allclose(coords_out.value, coords.value)
        assert coords_out.units == coords.units
        assert np.allclose(smls_out.value, smls.value)
        assert smls_out.units == smls.units

        # Ensure returned arrays are copies
        assert coords_out is not coords
        assert smls_out is not smls

    def test_cartesian_resolution_angular_fov_converts(self):
        """Test conversion when resolution is Cartesian but fov is angular."""
        # Setup
        resolution = 0.1 * kpc
        fov = 10.0 * arcsecond
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        emitter = MockEmitter(coords, redshift=2.0)

        # Execute
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=False,
        )

        # Assert units are now consistent
        assert res_out.units == kpc
        assert fov_out.units == kpc
        assert coords_out.units == kpc
        assert smls_out is None

    def test_angular_resolution_cartesian_fov_converts(self):
        """Test conversion when resolution is angular but fov is Cartesian."""
        # Setup
        resolution = 0.1 * arcsecond
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        smls = unyt_array(np.random.rand(100), "kpc")
        emitter = MockEmitter(coords, smls, redshift=2.0)

        # Execute
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=True,
        )

        # Assert units are now consistent (all angular)
        assert res_out.units == arcsecond
        assert fov_out.units == arcsecond
        assert coords_out.units == arcsecond
        assert smls_out.units == arcsecond

    def test_mixed_units_no_cosmo_raises_error(self):
        """Test that missing cosmology raises an error with mixed units."""
        # Setup
        resolution = 0.1 * arcsecond
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        emitter = MockEmitter(coords, redshift=2.0)

        # Execute and assert
        with pytest.raises(exceptions.InconsistentArguments):
            _standardize_imaging_units(
                resolution=resolution,
                fov=fov,
                emitter=emitter,
                cosmo=None,  # No cosmology provided
                include_smoothing_lengths=False,
            )

    def test_mixed_units_no_redshift_raises_error(self):
        """Test that missing redshift raises an error with mixed units."""
        # Setup
        resolution = 0.1 * arcsecond
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        emitter = MockEmitter(coords, redshift=None)  # No redshift

        # Execute and assert
        with pytest.raises(exceptions.InconsistentArguments):
            _standardize_imaging_units(
                resolution=resolution,
                fov=fov,
                emitter=emitter,
                cosmo=Planck18,
                include_smoothing_lengths=False,
            )

    def test_fov_single_value_expands_to_array(self):
        """Test that a single fov value is expanded to a 2D array."""
        # Setup - use a scalar fov
        resolution = 0.1 * kpc
        fov = 10.0 * kpc  # Single value
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        emitter = MockEmitter(coords, redshift=1.0)

        # Execute
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=False,
        )

        # Assert fov is now a 2-element array
        assert fov_out.size == 2
        assert np.allclose(fov_out.value, [10.0, 10.0])

    def test_no_smoothing_lengths_returns_none(self):
        """Test that None is returned for smoothing lengths."""
        # Setup
        resolution = 0.1 * kpc
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        smls = unyt_array(np.random.rand(100), "kpc")
        emitter = MockEmitter(coords, smls, redshift=1.0)

        # Execute without requesting smoothing lengths
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=False,
        )

        # Assert
        assert smls_out is None

    def test_conversion_preserves_coordinate_centering(self):
        """Test that coordinate centering is preserved after conversion."""
        # Setup - create centered coordinates
        resolution = 0.1 * arcsecond
        fov = 10.0 * kpc
        coords = (
            unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        )  # Centered at origin
        emitter = MockEmitter(coords, redshift=2.0)

        # Execute
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=False,
        )

        # Assert coordinates are still centered (mean should be near 0)
        # Use a more relaxed tolerance since unit conversion can introduce
        # small numerical errors
        assert np.allclose(coords_out.mean(axis=0).value, 0.0, atol=0.01)

    def test_incompatible_resolution_units_raises_error(self):
        """Test that incompatible resolution units raise an error."""
        # Setup with units that are incompatible with both kpc and arcsecond
        # Note: 'meter' is technically spatial but not the expected unit for
        # astronomical imaging, however the function checks if resolution
        # is compatible with kpc/arcsec, not if it's "valid" for imaging
        # So let's use a truly incompatible unit like seconds
        from unyt import s

        resolution = 0.1 * s  # Time units - truly incompatible
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        emitter = MockEmitter(coords, redshift=1.0)

        # Execute and assert
        with pytest.raises(exceptions.InconsistentArguments):
            _standardize_imaging_units(
                resolution=resolution,
                fov=fov,
                emitter=emitter,
                cosmo=Planck18,
                include_smoothing_lengths=False,
            )

    def test_empty_coordinates(self):
        """Test handling of empty coordinate arrays."""
        # Setup
        resolution = 0.1 * kpc
        fov = 10.0 * kpc
        coords = unyt_array([], "kpc").reshape(0, 3)
        emitter = MockEmitter(coords, redshift=1.0)

        # Execute
        res_out, fov_out, coords_out, smls_out = _standardize_imaging_units(
            resolution=resolution,
            fov=fov,
            emitter=emitter,
            cosmo=Planck18,
            include_smoothing_lengths=False,
        )

        # Assert
        assert res_out == resolution
        assert np.allclose(fov_out.value, fov.value)
        assert coords_out.size == 0
        assert coords_out.units == coords.units
        assert smls_out is None

    def test_invalid_resolution_type_raises_error(self):
        """Test that non-unyt resolution raises an error."""
        # Setup - resolution without units
        resolution = 0.1  # Plain float, no units
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        emitter = MockEmitter(coords, redshift=1.0)

        # Execute and assert
        with pytest.raises(exceptions.InconsistentArguments) as excinfo:
            _standardize_imaging_units(
                resolution=resolution,
                fov=fov,
                emitter=emitter,
                cosmo=Planck18,
                include_smoothing_lengths=False,
            )
        assert "Resolution must be a unyt_quantity" in str(excinfo.value)

    def test_invalid_fov_type_raises_error(self):
        """Test that non-unyt fov raises an error."""
        # Setup - fov without units
        resolution = 0.1 * kpc
        fov = 10.0  # Plain float, no units
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc
        emitter = MockEmitter(coords, redshift=1.0)

        # Execute and assert
        with pytest.raises(exceptions.InconsistentArguments) as excinfo:
            _standardize_imaging_units(
                resolution=resolution,
                fov=fov,
                emitter=emitter,
                cosmo=Planck18,
                include_smoothing_lengths=False,
            )
        assert "Field of view (fov) must be a unyt_quantity" in str(
            excinfo.value
        )

    def test_missing_centered_coordinates_raises_error(self):
        """Test that emitter without centered_coordinates raises error."""
        # Setup - emitter without centered_coordinates attribute
        resolution = 0.1 * kpc
        fov = 10.0 * kpc

        class BadEmitter:
            """Emitter without centered_coordinates."""

            def __init__(self):
                self.redshift = 1.0

        emitter = BadEmitter()

        # Execute and assert
        with pytest.raises(exceptions.InconsistentArguments) as excinfo:
            _standardize_imaging_units(
                resolution=resolution,
                fov=fov,
                emitter=emitter,
                cosmo=Planck18,
                include_smoothing_lengths=False,
            )
        assert "centered_coordinates" in str(excinfo.value)

    def test_missing_smoothing_lengths_raises_error(self):
        """Test that missing smoothing_lengths raises error when required."""
        # Setup - emitter without smoothing_lengths
        resolution = 0.1 * kpc
        fov = 10.0 * kpc
        coords = unyt_array(np.random.rand(100, 3), "kpc") - 0.5 * kpc

        class EmitterWithoutSmls:
            """Emitter without smoothing_lengths."""

            def __init__(self, coords):
                self.centered_coordinates = coords
                self.redshift = 1.0

        emitter = EmitterWithoutSmls(coords)

        # Execute and assert
        with pytest.raises(exceptions.InconsistentArguments) as excinfo:
            _standardize_imaging_units(
                resolution=resolution,
                fov=fov,
                emitter=emitter,
                cosmo=Planck18,
                include_smoothing_lengths=True,
            )
        assert "smoothing_lengths" in str(excinfo.value)
        assert "include_smoothing_lengths=True" in str(excinfo.value)
