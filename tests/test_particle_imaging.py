"""Test suite for imaging module.

This module contains unit tests for the imaging functionality of the
synthesizer package. It tests the creation and manipulation of images,
spectral cubes, and image collections, ensuring that the imaging
functionality works as expected with various inputs and configurations.
"""

import numpy as np
import pytest
from unyt import (
    Hz,
    Msun,
    Myr,
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
from synthesizer.imaging.spectral_cube import SpectralCube
from synthesizer.instruments import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.photometry import PhotometryCollection


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


class TestImageGeneration:
    """Test image generation methods."""

    @pytest.fixture
    def mock_particles(self):
        """Create mock particle data."""
        n_particles = 20
        coords = unyt_array(
            np.random.uniform(-0.4, 0.4, (n_particles, 3)), kpc
        )
        signal = unyt_array(
            np.random.uniform(1e28, 1e30, n_particles), erg / s
        )
        smoothing_lengths = unyt_array(np.full(n_particles, 0.05), kpc)
        return coords, signal, smoothing_lengths

    def test_get_img_hist(self, mock_particles):
        """Test histogram image generation."""
        coords, signal, _ = mock_particles
        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)

        img.get_img_hist(signal, coords)

        assert img.arr is not None
        assert np.all(img.shape == (10, 10)), (
            f"Image shape should be (10, 10) but found {img.shape}"
        )
        assert np.sum(img.arr) >= 0

    def test_get_img_smoothed(self, mock_particles):
        """Test smoothed image generation."""
        coords, signal, smoothing_lengths = mock_particles
        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        kernel = Kernel().get_kernel()

        img.get_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        assert img.arr is not None
        assert np.all(img.shape == (10, 10)), (
            f"Image shape should be (10, 10) but found {img.shape}"
        )
        assert np.sum(img.arr) >= 0

    def test_empty_particle_arrays(self):
        """Test with empty particle arrays."""
        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        coords = unyt_array(np.empty((0, 3)), kpc)
        signal = unyt_array(np.empty(0), erg / s)

        img.get_img_hist(signal, coords)

        assert img.arr is not None
        assert np.sum(img.arr) == 0


class TestImageOperations:
    """Test image operations and manipulations."""

    @pytest.fixture
    def test_image(self):
        """Create a test image with known data."""
        data = np.ones((10, 10)) * 5.0
        data[4:6, 4:6] = 20.0  # Central bright region
        test_data = unyt_array(data, erg / s / Hz)

        return Image(resolution=0.1 * kpc, fov=1.0 * kpc, img=test_data)

    def test_image_arithmetic_addition(self, test_image):
        """Test image addition."""
        img1 = test_image
        img2 = Image(
            img1.resolution, img1.fov, img=img1.arr.copy() * img1.units
        )

        result = img1 + img2
        expected_sum = 2 * np.sum(img1.arr)

        assert np.sum(result.arr) == pytest.approx(expected_sum)

    def test_image_arithmetic_multiplication(self, test_image):
        """Test image scalar multiplication."""
        img = test_image
        original_data = img.arr.copy()

        result = img * 2.0

        assert np.allclose(result.arr, 2.0 * original_data)

    def test_aperture_photometry(self, test_image):
        """Test aperture photometry."""
        img = test_image

        # Use pixel coordinates for aperture center (center of 10x10 image
        # is [5, 5])
        aperture_center = np.array([5.0, 5.0])  # Image center in pixels
        radius = 2.0 * kpc  # Radius in physical units

        signal = img.get_signal_in_aperture(radius, aperture_center)

        assert signal > 0
        assert isinstance(signal, (unyt_array, float))

    def test_apply_noise(self, test_image):
        """Test noise application."""
        img = test_image
        original_data = img.arr.copy()

        # Apply noise
        noise_std = 1.0 * erg / s / Hz
        new_img = img.apply_noise_from_std(noise_std)

        # Image should be different but roughly similar magnitude
        assert not np.allclose(new_img.arr, original_data, atol=0.1), (
            "Image should change after noise application"
        )
        mean_diff = np.abs(np.mean(img.arr) - np.mean(original_data))
        assert mean_diff < 2.0, (
            "Mean difference should be within expected noise range"
        )

    def test_apply_psf(self, test_image):
        """Test PSF application."""
        img = test_image
        original_sum = np.sum(img.arr)

        # Create simple PSF kernel
        psf_size = 3
        psf = np.ones((psf_size, psf_size))
        psf = psf / np.sum(psf)  # Normalize

        img.apply_psf(psf)

        # Total flux should be approximately conserved
        new_sum = np.sum(img.arr)
        assert np.abs(new_sum - original_sum) / original_sum < 0.01


class TestImageCollection:
    """Test ImageCollection functionality."""

    @pytest.fixture
    def mock_photometry(self):
        """Create mock photometry data."""
        n_particles = 15
        return PhotometryCollection(
            FilterCollection(
                generic_dict={
                    "g_band": np.ones(1000),
                    "r_band": np.ones(1000) * 1.2,
                    "i_band": np.ones(1000) * 1.5,
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            ),
            **{
                "g_band": unyt_array(
                    np.random.rand(n_particles) * 1e30, erg / s / Hz
                ),
                "r_band": unyt_array(
                    np.random.rand(n_particles) * 1.2e30, erg / s / Hz
                ),
                "i_band": unyt_array(
                    np.random.rand(n_particles) * 1.5e30, erg / s / Hz
                ),
            },
        )

    @pytest.fixture
    def mock_coordinates(self):
        """Create mock coordinates."""
        n_particles = 15
        return unyt_array(np.random.uniform(-0.4, 0.4, (n_particles, 3)), kpc)

    def test_collection_creation(self):
        """Test ImageCollection creation."""
        collection = ImageCollection(resolution=0.1 * kpc, fov=1.0 * kpc)

        assert collection.has_cartesian_units
        assert len(collection.imgs) == 0

    def test_get_imgs_hist(self, mock_photometry, mock_coordinates):
        """Test histogram image generation for collection."""
        collection = ImageCollection(resolution=0.1 * kpc, fov=1.0 * kpc)

        collection.get_imgs_hist(mock_photometry, mock_coordinates)

        assert len(collection.imgs) == 3
        for band_name in ["g_band", "r_band", "i_band"]:
            assert band_name in collection.imgs
            assert collection.imgs[band_name].arr is not None

    def test_get_imgs_smoothed(self, mock_photometry, mock_coordinates):
        """Test smoothed image generation for collection."""
        collection = ImageCollection(resolution=0.1 * kpc, fov=1.0 * kpc)
        n_particles = len(mock_coordinates)
        smoothing_lengths = unyt_array(np.full(n_particles, 0.05), kpc)
        kernel = Kernel().get_kernel()

        collection.get_imgs_smoothed(
            mock_photometry, mock_coordinates, smoothing_lengths, kernel
        )

        assert len(collection.imgs) == 3
        for band_name in ["g_band", "r_band", "i_band"]:
            assert band_name in collection.imgs
            assert collection.imgs[band_name].arr is not None

    def test_make_rgb_image(self, mock_photometry, mock_coordinates):
        """Test RGB image creation."""
        collection = ImageCollection(resolution=0.1 * kpc, fov=1.0 * kpc)
        collection.get_imgs_hist(mock_photometry, mock_coordinates)

        rgb_filters = {
            "R": ("r_band",),
            "G": ("g_band",),
            "B": ("i_band",),
        }
        rgb_img = collection.make_rgb_image(rgb_filters)

        assert np.all(rgb_img.shape == (10, 10, 3))


class TestSpectralCube:
    """Test SpectralCube functionality."""

    @pytest.fixture
    def basic_cube(self):
        """Create a basic spectral cube."""
        wavelengths = np.linspace(5000, 6000, 10) * angstrom
        return SpectralCube(
            resolution=0.2 * kpc, fov=1.0 * kpc, lam=wavelengths
        )

    def test_cube_creation(self, basic_cube):
        """Test SpectralCube creation."""
        assert basic_cube.has_cartesian_units, (
            f"Should have Cartesian units but found {basic_cube.units}"
        )
        assert basic_cube.arr is None, "Cube should be None initially"
        assert len(basic_cube.lam) == 10, (
            f"Should have 10 wavelengths but found {len(basic_cube.lam)}"
        )

    def test_get_data_cube_hist(self, basic_cube):
        """Test data cube generation with histogram method."""
        from synthesizer.emissions.sed import Sed

        # Create simple SED data
        n_particles = 5
        n_wavelengths = len(basic_cube.lam)
        spectra = np.random.rand(n_particles, n_wavelengths)
        sed = Sed(lam=basic_cube.lam, lnu=spectra * erg / s / Hz)

        coords = unyt_array(
            np.random.uniform(-0.4, 0.4, (n_particles, 3)), kpc
        )

        basic_cube.get_data_cube_hist(sed, coords)

        assert basic_cube.cube is not None
        assert basic_cube.cube.shape == (5, 5, 10)
        assert np.sum(basic_cube.cube) >= 0


class TestImageIntegration:
    """Integration tests for complete workflows."""

    def test_complete_multiband_workflow(self):
        """Test complete multiband imaging workflow."""
        # Create mock galaxy data
        n_particles = 30
        coords = unyt_array(
            np.random.uniform(-0.3, 0.3, (n_particles, 3)), kpc
        )

        # Multi-band photometry
        photometry = PhotometryCollection(
            FilterCollection(
                generic_dict={
                    "F435W": np.ones(1000),
                    "F606W": np.ones(1000) * 1.2,
                    "F814W": np.ones(1000) * 1.5,
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            ),
            **{
                "F435W": unyt_array(
                    np.random.rand(n_particles) * 1e30, erg / s / Hz
                ),
                "F606W": unyt_array(
                    np.random.rand(n_particles) * 1.5e30, erg / s / Hz
                ),
                "F814W": unyt_array(
                    np.random.rand(n_particles) * 2e30, erg / s / Hz
                ),
            },
        )

        # Create image collection
        collection = ImageCollection(resolution=0.05 * kpc, fov=1.0 * kpc)

        # Generate images
        collection.get_imgs_hist(photometry, coords)

        # Apply noise to make it realistic
        noise_stds = {
            "F435W": 0.1 * erg / s / Hz,
            "F606W": 0.08 * erg / s / Hz,
            "F814W": 0.05 * erg / s / Hz,
        }
        collection.apply_noise_from_stds(noise_stds)

        # Create RGB composite
        rgb_filters = {"R": ("F606W",), "G": ("F435W",), "B": ("F814W",)}
        rgb_image = collection.make_rgb_image(rgb_filters)

        # Verify workflow completed successfully
        assert len(collection.imgs) == 3
        assert rgb_image.shape == (20, 20, 3)

        # All images should have data
        for img in collection.imgs.values():
            assert img.arr is not None
            assert np.sum(img.arr) >= 0

    def test_spectral_cube_workflow(self):
        """Test spectral cube creation and manipulation."""
        from synthesizer.emissions.sed import Sed

        # Create spectral cube
        wavelengths = np.linspace(4000, 8000, 20) * angstrom
        cube = SpectralCube(
            resolution=0.2 * kpc, fov=1.0 * kpc, lam=wavelengths
        )

        # Create mock SED data
        n_particles = 8
        n_wavelengths = len(wavelengths)
        # Simple declining spectrum
        base_spectrum = 1e30 * (wavelengths.value / 5000.0) ** -1.5

        spectra = np.zeros((n_particles, n_wavelengths))
        for i in range(n_particles):
            spectra[i] = base_spectrum * np.random.uniform(0.8, 1.2)

        sed = Sed(lam=wavelengths, lnu=spectra * erg / s / Hz)
        coords = unyt_array(
            np.random.uniform(-0.4, 0.4, (n_particles, 3)), kpc
        )

        # Generate data cube
        cube.get_data_cube_hist(sed, coords)

        # Verify cube structure
        assert cube.cube is not None
        assert cube.cube.shape == (5, 5, 20)
        assert np.sum(cube.cube) > 0

        # Test that spectral structure is preserved
        # Sum over spatial dimensions to get integrated spectrum
        integrated_spectrum = np.sum(cube.cube, axis=(0, 1))
        assert len(integrated_spectrum) == n_wavelengths
        assert np.sum(integrated_spectrum) > 0


class TestGalaxyImagingSingleParticle:
    """Test imaging functionality for a galaxy with a single particle.

    This is the simplest test case, ensuring that the imaging
    functionality is behaving as expected.
    """

    @pytest.fixture
    def one_part_galaxy(self, incident_emission_model):
        """Create a mock galaxy with one particle."""
        from synthesizer import Galaxy
        from synthesizer.particle.stars import Stars

        # Create a galaxy with one star particle
        stars = Stars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([30.0]) * Myr,
            metallicities=np.array([0.02]),
            coordinates=np.array([[1.05, 1.05, 0.0]]) * kpc,
            smoothing_lengths=np.array([0.01]) * kpc,
            tau_v=np.array([0.7]),
        )
        galaxy = Galaxy(
            stars=stars, centre=np.array([0.0, 0.0, 0.0]) * kpc, redshift=0.0
        )

        incident_emission_model.set_per_particle(True)

        # Generate spectra and photometry for the galaxy
        galaxy.get_spectra(incident_emission_model)

        # Create a fake filter colleciton to get photometry for
        trans = np.zeros(1000)
        trans[400:600] = 1.0
        generic_dict = {
            "filter_r": trans,
        }
        new_lam = np.logspace(
            np.log10(4000 * angstrom),
            np.log10(7000 * angstrom),
            1000,
        )
        filters = FilterCollection(
            generic_dict=generic_dict,
            new_lam=new_lam * angstrom,
        )

        # Get photometry for the galaxy
        galaxy.get_photo_lnu(filters)

        return galaxy

    def test_single_particle_image(
        self, one_part_galaxy, incident_emission_model
    ):
        """Test image generation for a galaxy with one particle."""
        # Define the image properties
        resolution = 0.1 * kpc
        fov = 3.0 * kpc

        incident_emission_model.set_per_particle(True)

        kernel = Kernel().get_kernel()

        # Create an image for the galaxy
        galaxy_image = one_part_galaxy.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            kernel=kernel,
        )["filter_r"]

        assert galaxy_image is not None, (
            "Galaxy image should not be None after generation"
        )
        assert galaxy_image.arr is not None, (
            "Image array should not be None after generation"
        )

    def test_compare_hist_smoothed_single_particle(
        self, one_part_galaxy, incident_emission_model
    ):
        """Test histogram vs smoothed image for a single particle."""
        # Define the image properties
        resolution = 0.1 * kpc
        fov = 3.0 * kpc

        incident_emission_model.set_per_particle(True)

        kernel = Kernel().get_kernel()

        # Create histogram image
        hist_image = one_part_galaxy.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            kernel=kernel,
            img_type="hist",
        )["filter_r"]

        # Create smoothed image
        smoothed_image = one_part_galaxy.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            kernel=kernel,
            img_type="smoothed",
        )["filter_r"]

        assert hist_image is not None, (
            "Histogram image should not be None after generation"
        )
        assert smoothed_image is not None, (
            "Smoothed image should not be None after generation"
        )

        assert np.all(hist_image.arr >= 0), (
            "Histogram image array should contain non-negative values"
        )
        assert np.all(smoothed_image.arr >= 0), (
            "Smoothed image array should contain non-negative values"
        )

        # Ensure the images are identical for a single particle
        assert np.array_equal(hist_image.arr, smoothed_image.arr), (
            "Histogram and smoothed images should be identical for "
            "a single particle"
        )

    def test_orientation(self, one_part_galaxy, incident_emission_model):
        """Test image generation with different orientations."""
        # Define the image properties
        resolution = 0.1 * kpc
        fov = 3.0 * kpc

        incident_emission_model.set_per_particle(True)

        kernel = Kernel().get_kernel()

        # Get the image
        galaxy_image = one_part_galaxy.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            kernel=kernel,
        )["filter_r"]

        # Ensure the pixel populated is (25, 25), i.e. in the bottom left
        # quadrant of the image
        assert galaxy_image.arr[25, 25] > 0, (
            "Image should have a pixel populated at (25, 25)"
        )


class TestImagingFluxConservation:
    """Test flux conservation in imaging operations."""

    def test_flux_conservation_histogram(
        self,
        random_part_stars,
        incident_emission_model,
    ):
        """Test flux conservation in histogram imaging."""
        # Define the image properties
        resolution = 0.5 * kpc
        fov = 200 * kpc

        incident_emission_model.set_per_particle(True)

        random_part_stars.get_spectra(incident_emission_model)
        random_part_stars.get_particle_photo_lnu(
            FilterCollection(
                generic_dict={
                    "filter_r": np.ones(1000),
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )
        random_part_stars.get_photo_lnu(
            FilterCollection(
                generic_dict={
                    "filter_r": np.ones(1000),
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )

        random_part_stars.centre = np.array([0.0, 0.0, 0.0]) * kpc

        # Create histogram image
        hist_image = random_part_stars.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            img_type="hist",
        )["filter_r"]

        # Get the sum of the histogram image
        hist_flux = np.sum(hist_image.arr)

        # Get the true photometry
        expected_flux = random_part_stars.photo_lnu["incident"]["filter_r"]

        # Compare the fluxes
        assert np.isclose(hist_flux, expected_flux, rtol=1e-3), (
            f"Histogram flux {hist_flux} does not match expected flux "
            f"{expected_flux} within tolerance"
        )

    def test_flux_conservation_smoothed(
        self,
        random_part_stars,
        incident_emission_model,
    ):
        """Test flux conservation in smoothed imaging."""
        # Define the image properties
        resolution = 0.5 * kpc
        fov = 200 * kpc

        incident_emission_model.set_per_particle(True)

        random_part_stars.get_spectra(incident_emission_model)
        random_part_stars.get_particle_photo_lnu(
            FilterCollection(
                generic_dict={
                    "filter_r": np.ones(1000),
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )
        random_part_stars.get_photo_lnu(
            FilterCollection(
                generic_dict={
                    "filter_r": np.ones(1000),
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )

        random_part_stars.centre = np.array([0.0, 0.0, 0.0]) * kpc

        kernel = Kernel().get_kernel()

        # Create smoothed image
        smoothed_image = random_part_stars.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            img_type="smoothed",
            kernel=kernel,
        )["filter_r"]

        # Get the sum of the smoothed image
        smoothed_flux = np.sum(smoothed_image.arr)

        # Get the true photometry
        expected_flux = random_part_stars.photo_lnu["incident"]["filter_r"]

        # Compare the fluxes
        assert np.isclose(smoothed_flux, expected_flux, rtol=1e-3), (
            f"Smoothed flux {smoothed_flux} does not match expected flux "
            f"{expected_flux} within tolerance"
        )

    def test_threaded_vs_serial_smoothed_images(
        self,
        random_part_stars,
        incident_emission_model,
    ):
        """Test threaded vs serial smoothed image generation."""
        # Define the image properties
        resolution = 0.5 * kpc
        fov = 200 * kpc

        incident_emission_model.set_per_particle(True)

        random_part_stars.get_spectra(incident_emission_model)
        random_part_stars.get_particle_photo_lnu(
            FilterCollection(
                generic_dict={
                    "filter_r": np.ones(1000),
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )
        random_part_stars.get_photo_lnu(
            FilterCollection(
                generic_dict={
                    "filter_r": np.ones(1000),
                },
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )

        random_part_stars.centre = np.array([0.0, 0.0, 0.0]) * kpc

        kernel = Kernel().get_kernel()

        # Create smoothed image using threading
        threaded_image = random_part_stars.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            img_type="smoothed",
            kernel=kernel,
            nthreads=4,
        )["filter_r"]

        # Create smoothed image without threading
        serial_image = random_part_stars.get_images_luminosity(
            emission_model=incident_emission_model,
            resolution=resolution,
            fov=fov,
            img_type="smoothed",
            kernel=kernel,
        )["filter_r"]

        # Compare the two images
        assert np.allclose(threaded_image.arr, serial_image.arr), (
            "Threaded and serial smoothed images should be identical"
        )
