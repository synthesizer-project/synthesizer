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
    erg,
    kpc,
    s,
    unyt_array,
)

from synthesizer.imaging.image import Image
from synthesizer.imaging.image_collection import ImageCollection
from synthesizer.imaging.spectral_cube import SpectralCube
from synthesizer.instruments import FilterCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.photometry import PhotometryCollection


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
