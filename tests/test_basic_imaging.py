"""Test suite for basic generic functionality in the imaging module.

This module contains unit tests for the imaging functionality of the
synthesizer package. It tests the creation and manipulation of images,
spectral cubes, and image collections, ensuring that the imaging
functionality works as expected with various inputs and configurations.
"""

import numpy as np
import pytest
from unyt import (
    Hz,
    arcsecond,
    erg,
    kpc,
    s,
    unyt_array,
)

from synthesizer import exceptions
from synthesizer.imaging.base_imaging import ImagingBase
from synthesizer.imaging.image import Image


class DummyImaging(ImagingBase):
    """Minimal concrete class for testing ImagingBase geometry.

    Exposes the shape property as the image dimensions.
    """

    @property
    def shape(self):
        """Return the image shape as a tuple of pixel counts."""
        return tuple(self.npix)


class TestImagingGeometry:
    """Unit tests for ImagingBase geometry operations."""

    def test_init_cartesian(self):
        """Test initialization with Cartesian units."""
        res = 1 * kpc
        fov = 10 * kpc
        img = DummyImaging(resolution=res, fov=fov)

        assert img.has_cartesian_units, "Should have Cartesian units"
        assert not img.has_angular_units, "Should not have angular units"

        assert img.cart_resolution == res, (
            "stored cart_resolution should be same as input"
        )
        assert img.ang_resolution is None, "should not have angular resolution"
        assert np.allclose(img.cart_fov, unyt_array([10, 10], kpc)), (
            "fov should be same"
        )
        assert img.ang_fov is None, "should not have angular fov"

        # npix = ceil(fov / resolution) = [10, 10]
        assert np.array_equal(img.npix, np.array([10, 10], dtype=np.int32))
        assert img.shape == (10, 10)

        # orig_* preserved
        assert img.orig_resolution == res
        assert np.array_equal(img.orig_npix, img.npix)

    def test_init_angular(self):
        """Test initialization with angular units."""
        res = 2 * arcsecond
        fov = 100 * arcsecond
        img = DummyImaging(resolution=res, fov=fov)

        assert img.has_angular_units, "Should have angular units"
        assert not img.has_cartesian_units, "Should not have Cartesian units"

        assert img.ang_resolution == res
        assert img.cart_resolution is None
        assert np.allclose(img.ang_fov, unyt_array([100, 100], arcsecond))
        assert img.cart_fov is None

        # npix = ceil(100 / 2) = [50, 50]
        assert np.array_equal(img.npix, np.array([50, 50], dtype=np.int32))

    def test_init_tuple_fov(self):
        """Test initialization accepts tuple FOV and computes npix per axis."""
        res = 1 * kpc
        fov = unyt_array([10, 20], kpc)
        img = DummyImaging(resolution=res, fov=fov)
        assert np.array_equal(img.npix, np.array([10, 20], dtype=np.int32))

    def test_init_inconsistent_units_raises(self):
        """Test that inconsistent units raise an error."""
        with pytest.raises(exceptions.InconsistentArguments):
            DummyImaging(resolution=1 * kpc, fov=100 * arcsecond)

    def test_set_resolution(self):
        """Test setting a new resolution updates npix while preserving FOV."""
        img = DummyImaging(1 * kpc, 10 * kpc)
        img.set_resolution(2 * kpc)

        assert img.cart_resolution == 2 * kpc
        assert np.allclose(img.cart_fov, unyt_array([10, 10], kpc))
        assert np.array_equal(img.npix, np.array([5, 5], dtype=np.int32))

    def test_set_fov(self):
        """Test setting a new FOV updates npix while preserving resolution."""
        img = DummyImaging(1 * kpc, 10 * kpc)
        img.set_fov(20 * kpc)

        assert np.allclose(img.cart_fov, unyt_array([20, 20], kpc)), (
            f"FOV should be same as arguments but found {img.cart_fov}"
        )
        assert img.cart_resolution == 1 * kpc, (
            "resolution should be same as arguments"
        )
        assert np.array_equal(img.npix, np.array([20, 20], dtype=np.int32)), (
            f"npix should be same as arguments but found {img.npix}"
        )

    def test_set_npix(self):
        """Test setting npix updates resolution and FOV consistently."""
        img = DummyImaging(1 * kpc, 10 * kpc)
        img.set_npix(5)

        assert np.array_equal(img.npix, np.array([5, 5], dtype=np.int32)), (
            f"npix should be same as arguments but found {img.npix}"
        )
        assert img.cart_resolution == 2 * kpc, (
            f"resolution should be same as arguments but found "
            f"{img.cart_resolution}"
        )

    def test_resample_resolution(self):
        """Test resampling resolution scales resolution and npix correctly."""
        img = DummyImaging(1 * kpc, 10 * kpc)
        img._resample_resolution(2)

        assert img.cart_resolution == 0.5 * kpc
        assert np.array_equal(img.npix, np.array([20, 20], dtype=np.int32))

    def test_invalid_set_resolution_type_raises(self):
        """Test that setting resolution without units raises an error."""
        img = DummyImaging(1 * kpc, 10 * kpc)
        with pytest.raises(exceptions.InconsistentArguments):
            img.set_resolution(5)  # no units

    def test_invalid_set_fov_type_raises(self):
        """Test that setting FOV without units raises an error."""
        img = DummyImaging(1 * kpc, 10 * kpc)
        with pytest.raises(exceptions.InconsistentArguments):
            img.set_fov(5)  # no units

    def test_invalid_set_npix_type_raises(self):
        """Test that setting npix with non-integer type raises an error."""
        img = DummyImaging(1 * kpc, 10 * kpc)
        with pytest.raises(exceptions.InconsistentArguments):
            img.set_npix(5.5)  # not int/tuple


class TestImageCreation:
    """Test suite for Image class instantiation and basic operations."""

    def test_image_init_cartesian(self):
        """Test Image initialization with Cartesian units."""
        from synthesizer.imaging.image import Image

        res = 0.1 * kpc
        fov = 1.0 * kpc
        img = Image(resolution=res, fov=fov)

        assert img.has_cartesian_units
        assert img.cart_resolution == res
        assert np.allclose(img.cart_fov, unyt_array([1.0, 1.0], kpc))
        assert np.array_equal(img.npix, np.array([10, 10], dtype=np.int32))
        assert img.arr is None  # No image data yet
        assert img.units is None

    def test_image_init_angular(self):
        """Test Image initialization with angular units."""
        from synthesizer.imaging.image import Image

        res = 0.1 * arcsecond
        fov = 1.0 * arcsecond
        img = Image(resolution=res, fov=fov)

        assert img.has_angular_units, (
            f"Should have angular units but found {img.units}"
        )
        assert img.ang_resolution == res, (
            f"Stored ang_resolution should be same as input but "
            f"found {img.ang_resolution}"
        )
        # FOV might be stored in different units, so convert for comparison
        expected_fov = unyt_array([1.0, 1.0], arcsecond).to("degree")
        assert np.allclose(img.fov, expected_fov), (
            f"FOV should be same as arguments but found {img.ang_fov} "
            f"and expected {expected_fov}"
        )
        assert np.array_equal(img.npix, np.array([10, 10], dtype=np.int32)), (
            f"npix should be same as arguments but found {img.npix}"
        )

    def test_image_init_with_array(self):
        """Test Image initialization with existing array data."""
        from unyt import Hz, erg, s

        from synthesizer.imaging.image import Image

        res = 0.1 * kpc
        fov = 1.0 * kpc
        test_array = unyt_array(np.random.rand(10, 10), erg / s / Hz)

        img = Image(resolution=res, fov=fov, img=test_array)

        assert img.arr is not None
        assert np.array_equal(img.arr, test_array.value)
        assert img.units == test_array.units

    def test_image_init_with_plain_array(self):
        """Test Image initialization with plain numpy array."""
        from synthesizer.imaging.image import Image

        res = 0.1 * kpc
        fov = 1.0 * kpc
        test_array = np.random.rand(10, 10)

        img = Image(resolution=res, fov=fov, img=test_array)

        assert img.arr is not None
        assert np.array_equal(img.arr, test_array)
        assert img.units is None


class TestImageBasics:
    """Test basic image creation and properties."""

    def test_image_creation_cartesian(self):
        """Test image creation with Cartesian coordinates."""
        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)

        assert img.has_cartesian_units, "Should have Cartesian units"
        assert not img.has_angular_units, "Should not have angular units"
        assert np.all(img.cart_resolution == 0.1 * kpc), (
            "Stored cart_resolution should be same as input but "
            f"found {img.cart_resolution}"
        )
        assert np.all(img.shape == (10, 10)), (
            f"Image shape should be (10, 10) but found {img.shape}"
        )

    def test_image_creation_angular(self):
        """Test image creation with angular coordinates."""
        img = Image(resolution=0.1 * arcsecond, fov=1.0 * arcsecond)

        assert img.has_angular_units, "Should have angular units"
        assert not img.has_cartesian_units, "Should not have Cartesian units"
        assert img.ang_resolution == 0.1 * arcsecond, (
            "Stored ang_resolution should be same as input but "
            f"found {img.ang_resolution}"
        )
        assert np.all(img.shape == (10, 10)), (
            f"Image shape should be (10, 10) but found {img.shape}"
        )

    def test_image_with_data(self):
        """Test image creation with existing data."""
        data = np.random.rand(20, 20) * erg / s / Hz
        img = Image(resolution=0.1 * kpc, fov=2.0 * kpc, img=data)

        assert img.arr is not None, (
            "Image array should not be None after initialization"
        )
        assert np.all(img.arr.shape == (20, 20)), (
            f"Image shape should be (20, 20) but found {img.arr.shape}"
        )
        assert np.array_equal(img.arr, data.value)
        assert img.units == data.units
