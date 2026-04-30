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
from synthesizer.instruments.photometric_noise import (
    _cf_periodicity_dilution_correction_standalone,
    _generate_noise_from_rootps_standalone,
    _model_and_apply_correlated_noise,
)


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


class TestCorrelatedNoiseCore:
    """Tests for the internal correlated noise generation functions."""

    def test_cf_periodicity_correction_shape(self):
        """Correction factor array has the same shape as the input."""
        shape = (32, 48)
        correction = _cf_periodicity_dilution_correction_standalone(shape)
        assert correction.shape == shape

    def test_cf_periodicity_correction_positive(self):
        """All correction factor values are positive."""
        correction = _cf_periodicity_dilution_correction_standalone((16, 16))
        assert np.all(correction > 0)

    def test_cf_periodicity_correction_dc_is_one(self):
        """DC component (index [0,0]) of the correction equals 1.0."""
        correction = _cf_periodicity_dilution_correction_standalone((20, 20))
        assert np.isclose(correction[0, 0], 1.0)

    def test_generate_noise_shape(self):
        """Generated noise field has the shape passed to the function."""
        rng = np.random.default_rng(0)
        shape = (30, 40)
        rootps = np.ones((shape[0], shape[1] // 2 + 1), dtype=complex)
        noise = _generate_noise_from_rootps_standalone(rng, shape, rootps)
        assert noise.shape == shape

    def test_generate_noise_is_real(self):
        """Generated noise field is a real-valued array."""
        rng = np.random.default_rng(0)
        shape = (16, 16)
        rootps = np.ones((shape[0], shape[1] // 2 + 1), dtype=complex)
        noise = _generate_noise_from_rootps_standalone(rng, shape, rootps)
        assert np.isrealobj(noise)

    def test_model_apply_output_target_shape(self):
        """Output array has the same shape as the target image."""
        rng = np.random.default_rng(1)
        source = rng.normal(size=(20, 20))
        target = np.zeros((30, 30))
        result = _model_and_apply_correlated_noise(source, target, rng_seed=1)
        assert result.shape == target.shape

    def test_model_apply_reproducible_with_seed(self):
        """Identical seeds produce identical noise realisations."""
        source = np.random.default_rng(99).normal(size=(24, 24))
        target = np.zeros((24, 24))
        out1 = _model_and_apply_correlated_noise(source, target, rng_seed=7)
        out2 = _model_and_apply_correlated_noise(source, target, rng_seed=7)
        assert np.array_equal(out1, out2)

    def test_model_apply_different_seeds_differ(self):
        """Different seeds produce different noise realisations."""
        source = np.random.default_rng(99).normal(size=(24, 24))
        target = np.zeros((24, 24))
        out1 = _model_and_apply_correlated_noise(source, target, rng_seed=1)
        out2 = _model_and_apply_correlated_noise(source, target, rng_seed=2)
        assert not np.array_equal(out1, out2)

    def test_model_apply_subtract_mean_runs(self):
        """subtract_mean=True completes without error."""
        source = np.random.default_rng(0).normal(size=(16, 16))
        target = np.zeros((16, 16))
        result = _model_and_apply_correlated_noise(
            source, target, subtract_mean=True, rng_seed=0
        )
        assert result.shape == target.shape

    def test_model_apply_no_periodicity_correction(self):
        """correct_periodicity=False completes without error."""
        source = np.random.default_rng(0).normal(size=(16, 16))
        target = np.zeros((16, 16))
        result = _model_and_apply_correlated_noise(
            source, target, correct_periodicity=False, rng_seed=0
        )
        assert result.shape == target.shape

    def test_model_apply_source_target_different_shapes(self):
        """Source and target images may have different pixel dimensions."""
        source = np.random.default_rng(0).normal(size=(40, 40))
        target = np.zeros((20, 20))
        result = _model_and_apply_correlated_noise(source, target, rng_seed=0)
        assert result.shape == (20, 20)

    def test_model_apply_non2d_raises(self):
        """Non-2D inputs raise a ValueError."""
        with pytest.raises(ValueError):
            _model_and_apply_correlated_noise(
                np.ones((4, 4, 4)), np.zeros((4, 4))
            )
        with pytest.raises(ValueError):
            _model_and_apply_correlated_noise(
                np.ones((4, 4)), np.zeros((4, 4, 4))
            )


class TestImageCorrelatedNoise:
    """Tests for Image.apply_correlated_noise (instrument-based API)."""

    @pytest.fixture
    def noise_source(self):
        """A plain 2D array serving as the observed noise template."""
        return np.random.default_rng(7).normal(size=(32, 32))

    @pytest.fixture
    def instrument(self, noise_source):
        """Instrument with a single noise map keyed by filter code."""
        from synthesizer.instruments import Instrument

        return Instrument(
            label="test_inst",
            noise_maps={"F150W": noise_source},
        )

    @pytest.fixture
    def base_image(self):
        """An Image with a randomly-filled pixel array."""
        rng = np.random.default_rng(42)
        arr = rng.normal(size=(32, 32))
        img = Image(resolution=0.1 * kpc, fov=3.2 * kpc)
        img.arr = arr
        return img

    def test_returns_image_instance(self, base_image, instrument):
        """apply_correlated_noise returns an Image object."""
        result = base_image.apply_correlated_noise(instrument, "F150W")
        assert isinstance(result, Image)

    def test_output_differs_from_input(self, base_image, instrument):
        """The returned image array differs from the original array."""
        original = base_image.arr.copy()
        result = base_image.apply_correlated_noise(instrument, "F150W")
        assert not np.array_equal(result.arr, original)

    def test_noise_arr_is_set(self, base_image, instrument):
        """noise_arr attribute is populated on the returned image."""
        result = base_image.apply_correlated_noise(instrument, "F150W")
        assert result.noise_arr is not None
        assert result.noise_arr.shape == base_image.arr.shape

    def test_weight_map_is_set(self, base_image, instrument):
        """weight_map attribute is a positive scalar on the returned image."""
        result = base_image.apply_correlated_noise(instrument, "F150W")
        assert result.weight_map is not None
        assert result.weight_map > 0

    def test_output_preserves_shape(self, base_image, instrument):
        """Returned image has the same pixel dimensions as the input."""
        result = base_image.apply_correlated_noise(instrument, "F150W")
        assert result.arr.shape == base_image.arr.shape

    def test_subtract_mean_option(self, base_image, instrument):
        """subtract_mean=True runs without error and returns an Image."""
        result = base_image.apply_correlated_noise(
            instrument, "F150W", subtract_mean=True
        )
        assert isinstance(result, Image)

    def test_no_periodicity_correction_option(self, base_image, instrument):
        """correct_periodicity=False runs without error and returns Image."""
        result = base_image.apply_correlated_noise(
            instrument, "F150W", correct_periodicity=False
        )
        assert isinstance(result, Image)

    def test_noise_source_different_shape(self, base_image):
        """Noise template with a larger shape than the image is accepted."""
        from synthesizer.instruments import Instrument

        noise_source_big = np.random.default_rng(5).normal(size=(64, 64))
        inst = Instrument(
            label="test_inst_big",
            noise_maps={"F150W": noise_source_big},
        )
        result = base_image.apply_correlated_noise(inst, "F150W")
        assert result.arr.shape == base_image.arr.shape

    def test_cf_is_cached_after_first_call(self, base_image, instrument):
        """The CF is cached on the instrument after the first call."""
        base_image.apply_correlated_noise(instrument, "F150W")
        cache_key = ("F150W", False, True)
        assert cache_key in instrument._correlated_noise_cf_cache

    def test_cache_reused_on_second_call(self, base_image, instrument):
        """A second call for the same filter hits the cache (no recompute)."""
        base_image.apply_correlated_noise(instrument, "F150W")
        cf_first = instrument._correlated_noise_cf_cache[
            ("F150W", False, True)
        ]
        base_image.apply_correlated_noise(instrument, "F150W")
        cf_second = instrument._correlated_noise_cf_cache[
            ("F150W", False, True)
        ]
        assert cf_first is cf_second  # identical object — cache was reused

    def test_missing_filter_raises(self, base_image, instrument):
        """Requesting an absent filter raises InconsistentArguments."""
        with pytest.raises(exceptions.InconsistentArguments):
            base_image.apply_correlated_noise(instrument, "NONEXISTENT")

    def test_no_noise_maps_raises(self, base_image):
        """An instrument without noise_maps raises MissingArgument."""
        from synthesizer.instruments import Instrument

        inst = Instrument(label="no_noise")
        with pytest.raises(exceptions.MissingArgument):
            base_image.apply_correlated_noise(inst, "F150W")
