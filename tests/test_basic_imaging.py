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
    angstrom,
    arcsecond,
    erg,
    kpc,
    s,
    unyt_array,
)

from synthesizer import exceptions
from synthesizer.imaging.base_imaging import ImagingBase
from synthesizer.imaging.image import Image
from synthesizer.imaging.image_collection import ImageCollection
from synthesizer.instruments import FilterCollection, Instrument
from synthesizer.instruments.photometric_noise import (
    CorrelatedNoiseModel,
    _cf_periodicity_dilution_correction_standalone,
    _generate_noise_from_rootps_standalone,
)


def make_test_imager(filter_codes, resolution=0.1 * kpc, **kwargs):
    """Create a minimal photometric imager for imaging tests."""
    # Build a small synthetic filter set matching the requested test filters.
    filters = FilterCollection(
        generic_dict={
            filter_code: np.ones(1000) for filter_code in filter_codes
        },
        new_lam=np.linspace(4000, 8000, 1000) * angstrom,
    )

    # Construct a real imaging-capable instrument so tests follow the current
    # factory contract.
    return Instrument(
        kwargs.pop("label", "test_inst"),
        filters=filters,
        resolution=resolution,
        **kwargs,
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
    """Tests for the correlated-noise modelling helpers."""

    def test_model_initialisation_stores_source(self):
        """The model retains the original source noise map."""
        source = np.random.default_rng(3).normal(size=(10, 12))
        model = CorrelatedNoiseModel(source)
        assert model.source_noise_map is source

    def test_model_non2d_source_raises(self):
        """A non-2D source noise map raises a ValueError."""
        with pytest.raises(ValueError):
            CorrelatedNoiseModel(np.ones((4, 4, 4)))

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
        model = CorrelatedNoiseModel(source)
        noise = model.generate_noise_array(target.shape, rng_seed=1)
        assert (target + noise).shape == target.shape

    def test_model_apply_reproducible_with_seed(self):
        """Identical seeds produce identical noise realisations."""
        source = np.random.default_rng(99).normal(size=(24, 24))
        model = CorrelatedNoiseModel(source)
        out1 = model.generate_noise_array((24, 24), rng_seed=7)
        out2 = model.generate_noise_array((24, 24), rng_seed=7)
        assert np.array_equal(out1, out2)

    def test_model_apply_different_seeds_differ(self):
        """Different seeds produce different noise realisations."""
        source = np.random.default_rng(99).normal(size=(24, 24))
        model = CorrelatedNoiseModel(source)
        out1 = model.generate_noise_array((24, 24), rng_seed=1)
        out2 = model.generate_noise_array((24, 24), rng_seed=2)
        assert not np.array_equal(out1, out2)

    def test_model_apply_subtract_mean_runs(self):
        """Low-level model generation supports explicit mean subtraction."""
        source = np.random.default_rng(0).normal(size=(16, 16))
        model = CorrelatedNoiseModel(source)
        result = model.generate_noise_array(
            (16, 16), subtract_mean=True, rng_seed=0
        )
        assert result.shape == (16, 16)

    def test_model_apply_no_periodicity_correction(self):
        """correct_periodicity=False completes without error."""
        source = np.random.default_rng(0).normal(size=(16, 16))
        model = CorrelatedNoiseModel(source)
        result = model.generate_noise_array(
            (16, 16), correct_periodicity=False, rng_seed=0
        )
        assert result.shape == (16, 16)

    def test_model_apply_source_target_different_shapes(self):
        """Source and target images may have different pixel dimensions."""
        source = np.random.default_rng(0).normal(size=(40, 40))
        model = CorrelatedNoiseModel(source)
        result = model.generate_noise_array((20, 20), rng_seed=0)
        assert result.shape == (20, 20)

    def test_model_apply_non2d_raises(self):
        """Non-2D inputs raise a ValueError."""
        with pytest.raises(ValueError):
            CorrelatedNoiseModel(np.ones((4, 4, 4)))

    def test_generate_noise_array_preserves_units(self):
        """Generated noise carries units from the source template."""
        source = unyt_array(
            np.random.default_rng(4).normal(size=(18, 18)), erg / s / Hz
        )
        model = CorrelatedNoiseModel(source)
        noise = model.generate_noise_array((12, 12), rng_seed=0)
        assert isinstance(noise, unyt_array)
        assert noise.units == source.units
        assert noise.shape == (12, 12)


class TestImageCorrelatedNoise:
    """Tests for Image.apply_correlated_noise (instrument-based API)."""

    @pytest.fixture
    def noise_source(self):
        """A plain 2D array serving as the observed noise template."""
        return np.random.default_rng(7).normal(size=(32, 32))

    @pytest.fixture
    def instrument(self, noise_source):
        """Instrument with a single correlated-noise source map."""
        return make_test_imager(
            filter_codes=("F150W",),
            noise_source_maps={"F150W": noise_source},
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

    def test_no_periodicity_correction_option(self, base_image, instrument):
        """correct_periodicity=False runs without error and returns Image."""
        result = base_image.apply_correlated_noise(
            instrument, "F150W", correct_periodicity=False
        )
        assert isinstance(result, Image)

    def test_noise_source_different_shape(self, base_image):
        """Noise template with a larger shape than the image is accepted."""
        noise_source_big = np.random.default_rng(5).normal(size=(64, 64))
        inst = make_test_imager(
            filter_codes=("F150W",),
            label="test_inst_big",
            noise_source_maps={"F150W": noise_source_big},
        )
        result = base_image.apply_correlated_noise(inst, "F150W")
        assert result.arr.shape == base_image.arr.shape

    def test_inplace_updates_original_image(self, base_image, instrument):
        """inplace=True updates and returns the original image object."""
        original = base_image.arr.copy()
        result = base_image.apply_correlated_noise(
            instrument, "F150W", inplace=True
        )
        assert result is base_image
        assert not np.array_equal(base_image.arr, original)
        assert base_image.noise_arr is not None

    def test_missing_filter_raises(self, base_image, instrument):
        """Requesting an absent filter raises InconsistentArguments."""
        with pytest.raises(exceptions.InconsistentArguments):
            base_image.apply_correlated_noise(instrument, "NONEXISTENT")

    def test_no_noise_source_maps_raises(self, base_image):
        """An instrument without source maps raises MissingArgument."""
        inst = make_test_imager(filter_codes=("F150W",), label="no_noise")
        with pytest.raises(exceptions.MissingArgument):
            base_image.apply_correlated_noise(inst, "F150W")

    def test_fixed_noise_maps_are_not_correlated_models(self, base_image):
        """Fixed noise arrays do not satisfy the correlated-noise API."""
        fixed_noise = np.ones((32, 32))
        inst = make_test_imager(
            filter_codes=("F150W",),
            label="fixed_noise",
            noise_maps={"F150W": fixed_noise},
        )

        with pytest.raises(exceptions.MissingArgument):
            base_image.apply_correlated_noise(inst, "F150W")

    def test_instrument_apply_noise_uses_fixed_noise_maps_directly(
        self, base_image
    ):
        """Instrument.apply_noise applies fixed arrays without modelling."""
        fixed_noise = np.arange(32 * 32, dtype=float).reshape(32, 32)
        inst = make_test_imager(
            filter_codes=("F150W",),
            label="fixed_noise",
            noise_maps={"F150W": fixed_noise},
        )

        result = inst.apply_noise(base_image, "F150W")

        assert np.array_equal(result.noise_arr, fixed_noise)
        assert np.array_equal(result.arr, base_image.arr + fixed_noise)


class TestImageCollectionCorrelatedNoise:
    """Tests for ImageCollection.apply_correlated_noise."""

    @pytest.fixture
    def instrument(self):
        """Instrument with correlated-noise source maps for two filters."""
        rng = np.random.default_rng(21)
        return make_test_imager(
            filter_codes=("F090W", "F150W"),
            noise_source_maps={
                "F090W": rng.normal(size=(32, 32)),
                "F150W": rng.normal(size=(32, 32)),
            },
        )

    @pytest.fixture
    def image_collection(self):
        """Two-filter image collection for correlated-noise application."""
        rng = np.random.default_rng(22)
        imgs = {
            "F090W": Image(
                resolution=0.1 * kpc,
                fov=3.2 * kpc,
                img=rng.normal(size=(32, 32)),
            ),
            "F150W": Image(
                resolution=0.1 * kpc,
                fov=3.2 * kpc,
                img=rng.normal(size=(32, 32)),
            ),
        }
        return ImageCollection(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            imgs=imgs,
        )

    def test_returns_image_collection(self, image_collection, instrument):
        """apply_correlated_noise returns an ImageCollection."""
        result = image_collection.apply_correlated_noise(instrument)
        assert isinstance(result, ImageCollection)

    def test_applies_noise_to_each_image(self, image_collection, instrument):
        """Each image in the collection receives a generated noise array."""
        result = image_collection.apply_correlated_noise(instrument)
        for f in image_collection.filter_codes:
            assert result.imgs[f].noise_arr is not None
            assert (
                result.imgs[f].arr.shape == image_collection.imgs[f].arr.shape
            )

    def test_inplace_updates_collection(self, image_collection, instrument):
        """inplace=True updates and returns the original collection."""
        original = {
            f: image_collection.imgs[f].arr.copy()
            for f in image_collection.filter_codes
        }
        result = image_collection.apply_correlated_noise(
            instrument, inplace=True
        )

        assert result is image_collection
        for f in image_collection.filter_codes:
            assert not np.array_equal(
                image_collection.imgs[f].arr, original[f]
            )
            assert image_collection.imgs[f].noise_arr is not None

    def test_missing_filter_model_raises(self, image_collection):
        """Missing a model for one collection filter raises an error."""
        inst = make_test_imager(
            filter_codes=("F090W",),
            noise_source_maps={
                "F090W": np.random.default_rng(23).normal(size=(32, 32))
            },
        )

        with pytest.raises(exceptions.InconsistentArguments):
            image_collection.apply_correlated_noise(inst)

    def test_rng_seed_is_forwarded(self, image_collection, instrument):
        """Providing the same rng_seed reproduces the same collection noise."""
        out1 = image_collection.apply_correlated_noise(instrument, rng_seed=11)
        out2 = image_collection.apply_correlated_noise(instrument, rng_seed=11)

        for f in image_collection.filter_codes:
            assert np.array_equal(
                out1.imgs[f].noise_arr, out2.imgs[f].noise_arr
            )

    def test_instrument_apply_noises_uses_independent_filter_noise(
        self, image_collection, instrument
    ):
        """apply_noises should not reuse the same random stream per filter."""
        result = instrument.apply_noises(image_collection, rng_seed=7)

        assert not np.array_equal(
            result.imgs["F090W"].noise_arr,
            result.imgs["F150W"].noise_arr,
        )


class TestPhotometricImagerPsfApplication:
    """Tests for instrument-owned PSF application on photometric imagers."""

    @pytest.fixture
    def instrument(self):
        """Instrument with PSFs defined for two imaging filters."""
        psf = np.zeros((3, 3), dtype=float)
        psf[1, 1] = 1.0
        return make_test_imager(
            filter_codes=("F090W", "F150W"),
            label="psf_inst",
            psfs={"F090W": psf, "F150W": psf},
        )

    @pytest.fixture
    def image_collection(self):
        """Two-filter image collection for instrument-side PSF application."""
        imgs = {
            "F090W": Image(
                resolution=0.1 * kpc,
                fov=3.2 * kpc,
                img=np.arange(32 * 32, dtype=float).reshape(32, 32),
            ),
            "F150W": Image(
                resolution=0.1 * kpc,
                fov=3.2 * kpc,
                img=np.eye(32, dtype=float),
            ),
        }
        return ImageCollection(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            imgs=imgs,
        )

    def test_apply_psf_uses_filter_specific_psf(self, instrument):
        """apply_psf should use the PSF matching the requested filter."""
        image = Image(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            img=np.arange(32 * 32, dtype=float).reshape(32, 32),
        )

        result = instrument.apply_psf(image, "F090W")

        assert np.allclose(result.arr, image.arr)

    def test_apply_psf_returns_new_image_by_default(self, instrument):
        """apply_psf should return a fresh image by default."""
        image = Image(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            img=np.arange(32 * 32, dtype=float).reshape(32, 32),
        )

        result = instrument.apply_psf(image, "F090W")

        assert result is not image

    def test_apply_psf_can_update_image_inplace(self, instrument):
        """apply_psf should support explicit in-place updates."""
        image = Image(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            img=np.arange(32 * 32, dtype=float).reshape(32, 32),
        )

        result = instrument.apply_psf(image, "F090W", inplace=True)

        assert result is image

    def test_apply_psf_raises_without_psf_configuration(self):
        """apply_psf should fail explicitly when no PSFs are configured."""
        instrument = make_test_imager(
            filter_codes=("F090W",),
            label="no_psf",
        )
        image = Image(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            img=np.ones((32, 32), dtype=float),
        )

        with pytest.raises(exceptions.MissingArgument):
            instrument.apply_psf(image, "F090W")

    def test_apply_psfs_returns_new_image_collection(
        self, instrument, image_collection
    ):
        """apply_psfs should return a new image collection."""
        result = instrument.apply_psfs(image_collection)

        assert isinstance(result, ImageCollection)
        assert result is not image_collection
        for filter_code in image_collection.filter_codes:
            assert (
                result.imgs[filter_code]
                is not image_collection.imgs[filter_code]
            )

    def test_apply_psfs_can_update_collection_inplace(
        self, instrument, image_collection
    ):
        """apply_psfs should support explicit in-place updates."""
        result = instrument.apply_psfs(image_collection, inplace=True)

        assert result is image_collection

    def test_image_container_no_longer_owns_psf_application(self):
        """Image containers should no longer expose PSF application methods."""
        image = Image(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            img=np.ones((32, 32), dtype=float),
        )

        assert not hasattr(image, "apply_psf")

    def test_image_collection_container_no_longer_owns_psf_application(self):
        """ImageCollection should no longer expose collection PSF methods."""
        collection = ImageCollection(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            imgs={
                "F090W": Image(
                    resolution=0.1 * kpc,
                    fov=3.2 * kpc,
                    img=np.ones((32, 32), dtype=float),
                )
            },
        )

        assert not hasattr(collection, "apply_psfs")

    def test_apply_psfs_with_resampling_does_not_mutate_input(
        self, instrument, image_collection
    ):
        """Instrument-side PSF application should not mutate inputs."""
        original_resolution = image_collection.resolution
        original_npix = image_collection.npix.copy()
        original_arrays = {
            filter_code: image_collection.imgs[filter_code].arr.copy()
            for filter_code in image_collection.filter_codes
        }

        instrument.apply_psfs(image_collection, psf_resample_factor=2)

        assert image_collection.resolution == original_resolution
        assert np.array_equal(image_collection.npix, original_npix)
        for filter_code in image_collection.filter_codes:
            assert np.array_equal(
                image_collection.imgs[filter_code].arr,
                original_arrays[filter_code],
            )


class TestImageCollectionResampling:
    """Tests for image-collection geometry resampling helpers."""

    def test_supersample_updates_collection_geometry(self):
        """Supersample should keep collection geometry in sync."""
        collection = ImageCollection(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            imgs={
                "F090W": Image(
                    resolution=0.1 * kpc,
                    fov=3.2 * kpc,
                    img=np.ones((32, 32), dtype=float),
                )
            },
        )

        # Supersample the collection and its contained image together.
        collection.supersample(2)

        assert np.array_equal(collection.npix, np.array([64, 64]))
        assert collection.imgs["F090W"].arr.shape == (64, 64)
        assert np.array_equal(collection.imgs["F090W"].npix, collection.npix)

    def test_downsample_updates_collection_geometry(self):
        """Downsample should keep collection geometry in sync."""
        collection = ImageCollection(
            resolution=0.1 * kpc,
            fov=3.2 * kpc,
            imgs={
                "F090W": Image(
                    resolution=0.1 * kpc,
                    fov=3.2 * kpc,
                    img=np.ones((32, 32), dtype=float),
                )
            },
        )

        # First supersample so the downsampling path has work to do.
        collection.supersample(2)

        # Downsample back to the original resolution and shape.
        collection.downsample(0.5)

        assert np.array_equal(collection.npix, np.array([32, 32]))
        assert collection.imgs["F090W"].arr.shape == (32, 32)
        assert np.array_equal(collection.imgs["F090W"].npix, collection.npix)
