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
            [
                [-0.2, -0.1, 0.0],
                [0.2, 0.1, 0.0],
                [-0.1, 0.2, 0.0],
                [0.1, -0.2, 0.0],
                [0.0, 0.0, 0.0],
            ],
            kpc,
        )

        basic_cube.get_data_cube_hist(sed, coords)

        assert basic_cube.cube is not None
        assert basic_cube.cube.shape == (5, 5, 10)
        assert np.sum(basic_cube.cube) >= 0

    def test_spectral_cube_flux_conservation_smoothed(self):
        """Test flux conservation in smoothed spectral cubes.

        Flux in each wavelength slice should be conserved when using SPH
        smoothing.
        """
        from synthesizer.emissions.sed import Sed

        # Create spectral cube
        wavelengths = np.linspace(5000, 6000, 5) * angstrom
        cube = SpectralCube(
            resolution=0.2 * kpc, fov=2.0 * kpc, lam=wavelengths
        )

        # Create particles with known spectra
        n_particles = 10
        n_wavelengths = len(wavelengths)

        # Simple constant SED per particle for easy flux tracking
        spectra_values = np.ones((n_particles, n_wavelengths)) * 1e30
        sed = Sed(lam=wavelengths, lnu=spectra_values * erg / s / Hz)

        coords = unyt_array(
            np.random.uniform(-0.8, 0.8, (n_particles, 3)), kpc
        )
        coords[:, 2] = 0.0  # Keep z=0

        smoothing_lengths = unyt_array([0.3] * n_particles, kpc)

        # Generate smoothed cube
        kernel = Kernel().get_kernel()
        cube.get_data_cube_smoothed(
            sed, coords, smoothing_lengths, kernel=kernel
        )

        # Check flux conservation for each wavelength slice
        expected_flux_per_wavelength = np.sum(spectra_values, axis=0)

        for i in range(n_wavelengths):
            wavelength_slice = cube.cube[:, :, i]
            slice_flux = np.sum(wavelength_slice)

            # FLUX MUST BE CONSERVED - 1% tolerance for numerical precision
            assert np.isclose(
                slice_flux, expected_flux_per_wavelength[i], rtol=0.01
            ), (
                f"Wavelength slice {i} flux {slice_flux} != expected "
                f"{expected_flux_per_wavelength[i]} - FLUX MUST BE CONSERVED!"
            )

    def test_spectral_cube_very_small_smoothing_lengths(self):
        """Test spectral cube with very small smoothing lengths.

        When smoothing length << pixel size, flux should still be conserved.
        """
        from synthesizer.emissions.sed import Sed

        # Create spectral cube
        wavelengths = np.linspace(5000, 6000, 3) * angstrom
        resolution = 0.5 * kpc
        cube = SpectralCube(
            resolution=resolution, fov=5.0 * kpc, lam=wavelengths
        )

        # Create particles with very small smoothing lengths
        n_particles = 5
        n_wavelengths = len(wavelengths)

        spectra_values = np.ones((n_particles, n_wavelengths)) * 1e30
        sed = Sed(lam=wavelengths, lnu=spectra_values * erg / s / Hz)

        # Place particles slightly off pixel boundaries to avoid numerical
        # issues. Spread particles around origin to satisfy centering
        # requirement
        coords = unyt_array(
            [
                [-0.02, -0.02, 0.0],
                [-0.01, -0.01, 0.0],
                [0.0, 0.03, 0.0],
                [0.01, 0.01, 0.0],
                [0.02, 0.02, 0.0],
            ],
            kpc,
        )
        smoothing_lengths = unyt_array(
            [0.01] * n_particles, kpc
        )  # 50x smaller

        kernel = Kernel().get_kernel()
        cube.get_data_cube_smoothed(
            sed, coords, smoothing_lengths, kernel=kernel
        )

        # Check total flux conservation
        expected_total_flux = np.sum(spectra_values)
        actual_total_flux = np.sum(cube.cube)

        # Flux must be conserved to within 1% (numerical precision)
        assert np.isclose(actual_total_flux, expected_total_flux, rtol=0.01), (
            f"Total flux {actual_total_flux} != expected {expected_total_flux}"
            f" - more than 1% flux lost!"
        )

    def test_spectral_cube_large_smoothing_lengths(self):
        """Test spectral cube with large smoothing lengths.

        When smoothing length >> pixel size, flux should still be conserved.
        """
        from synthesizer.emissions.sed import Sed

        # Create spectral cube with large FOV to contain kernel support
        wavelengths = np.linspace(5000, 6000, 4) * angstrom
        resolution = 0.2 * kpc
        # FOV must be large enough to contain particles + kernel support
        # Kernel radius = smoothing_length * threshold ≈ 0.6 * 1.825 = 1.1
        # So FOV = 2 * (max_pos + kernel_radius) = 2 * (0.5 + 1.1) = 3.2
        cube = SpectralCube(
            resolution=resolution, fov=4.0 * kpc, lam=wavelengths
        )

        # Create particles with large smoothing lengths
        n_particles = 8
        n_wavelengths = len(wavelengths)

        spectra_values = np.ones((n_particles, n_wavelengths)) * 1e30
        sed = Sed(lam=wavelengths, lnu=spectra_values * erg / s / Hz)

        # Keep particles well away from FOV edges
        coords = unyt_array(
            np.random.uniform(-0.5, 0.5, (n_particles, 3)), kpc
        )
        coords[:, 2] = 0.0

        # Use smoothing length that's large compared to pixels but fits in FOV
        smoothing_lengths = unyt_array(
            [0.6] * n_particles, kpc
        )  # 3x larger than res

        kernel = Kernel().get_kernel()
        cube.get_data_cube_smoothed(
            sed, coords, smoothing_lengths, kernel=kernel
        )

        # Check total flux conservation
        expected_total_flux = np.sum(spectra_values)
        actual_total_flux = np.sum(cube.cube)

        # Flux must be conserved to within 1% (numerical precision)
        assert np.isclose(actual_total_flux, expected_total_flux, rtol=0.01), (
            f"Total flux {actual_total_flux} != expected {expected_total_flux}"
            f" - more than 1% flux lost!"
        )

    def test_spectral_cube_threading_consistency(self):
        """Test that spectral cubes are consistent with different threading.

        Serial and parallel execution should give identical results.
        """
        from synthesizer.emissions.sed import Sed

        # Create spectral cube
        wavelengths = np.linspace(5000, 6000, 6) * angstrom
        n_particles = 20
        n_wavelengths = len(wavelengths)

        # Create reproducible SED data
        np.random.seed(42)
        spectra_values = np.random.rand(n_particles, n_wavelengths) * 1e30
        sed = Sed(lam=wavelengths, lnu=spectra_values * erg / s / Hz)

        coords = unyt_array(
            np.random.uniform(-0.8, 0.8, (n_particles, 3)), kpc
        )
        coords[:, 2] = 0.0

        smoothing_lengths = unyt_array(
            np.random.uniform(0.1, 0.5, n_particles), kpc
        )

        kernel = Kernel().get_kernel()

        # Test different thread counts
        results = {}
        for nthreads in [1, 2, 4]:
            cube = SpectralCube(
                resolution=0.15 * kpc, fov=2.0 * kpc, lam=wavelengths
            )
            cube.get_data_cube_smoothed(
                sed,
                coords,
                smoothing_lengths,
                kernel=kernel,
                nthreads=nthreads,
            )
            results[nthreads] = cube.cube.copy()

        # All thread counts should give identical results
        for nthreads in [2, 4]:
            assert np.allclose(results[1], results[nthreads], rtol=1e-10), (
                f"Results with {nthreads} threads differ from serial"
            )

    def test_spectral_cube_wavelength_independence(self):
        """Test that each wavelength slice is computed independently.

        Different wavelength slices should not affect each other.
        """
        from synthesizer.emissions.sed import Sed

        # Create spectral cube
        wavelengths = np.linspace(5000, 6000, 3) * angstrom
        cube = SpectralCube(
            resolution=0.2 * kpc, fov=2.0 * kpc, lam=wavelengths
        )

        # Create particles with varying spectra
        n_particles = 5
        n_wavelengths = len(wavelengths)

        # Different flux at each wavelength
        spectra_values = np.zeros((n_particles, n_wavelengths))
        spectra_values[:, 0] = 1e30  # Only first wavelength has flux
        spectra_values[:, 1] = 2e30  # Only second wavelength has flux
        spectra_values[:, 2] = 3e30  # Only third wavelength has flux

        sed = Sed(lam=wavelengths, lnu=spectra_values * erg / s / Hz)

        coords = unyt_array(
            np.random.uniform(-0.8, 0.8, (n_particles, 3)), kpc
        )
        coords[:, 2] = 0.0

        smoothing_lengths = unyt_array([0.3] * n_particles, kpc)

        kernel = Kernel().get_kernel()
        cube.get_data_cube_smoothed(
            sed, coords, smoothing_lengths, kernel=kernel
        )

        # Check each wavelength slice independently
        for i in range(n_wavelengths):
            expected_flux = np.sum(spectra_values[:, i])
            actual_flux = np.sum(cube.cube[:, :, i])

            assert np.isclose(actual_flux, expected_flux, rtol=0.01), (
                f"Wavelength {i} flux {actual_flux} != "
                f"expected {expected_flux}"
            )


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
        # Coordinates MUST be centered (mean at zero) for imaging to work
        # correctly. Place particle slightly off-center to test non-trivial
        # positioning.
        particle_pos = np.array([[0.05, 0.05, 0.0]]) * kpc
        centre_pos = particle_pos[
            0
        ]  # Centre = particle position for single particle

        stars = Stars(
            initial_masses=np.array([1.0]) * Msun,
            ages=np.array([30.0]) * Myr,
            metallicities=np.array([0.02]),
            coordinates=particle_pos,
            smoothing_lengths=np.array([0.5])
            * kpc,  # Larger smoothing length for better coverage
            tau_v=np.array([0.7]),
        )
        galaxy = Galaxy(stars=stars, centre=centre_pos, redshift=0.0)

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
        from synthesizer.instruments import Instrument

        resolution = 0.1 * kpc
        fov = 3.0 * kpc

        incident_emission_model.set_per_particle(True)

        kernel = Kernel().get_kernel()

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Create an image for the galaxy using the photometry label
        galaxy_image = one_part_galaxy.get_images_luminosity(
            "incident",  # Use the emission model label as positional argument
            fov=fov,
            instrument=instrument,
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
        """Test flux conservation in histogram vs smoothed image.

        Both histogram and smoothed imaging methods should conserve total
        flux for a single particle, though the spatial distributions may
        differ.
        """
        # Define the image properties
        from synthesizer.instruments import Instrument

        resolution = 0.1 * kpc
        fov = 3.0 * kpc

        incident_emission_model.set_per_particle(True)

        kernel = Kernel().get_kernel()

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Create histogram image
        hist_image = one_part_galaxy.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            kernel=kernel,
            img_type="hist",
        )["filter_r"]

        # Create smoothed image
        smoothed_image = one_part_galaxy.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
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

        # Get expected flux from photometry
        expected_flux = one_part_galaxy.stars.photo_lnu["incident"]["filter_r"]

        # Both methods should conserve total flux
        # Use 2% tolerance to account for edge effects with small smoothing
        # lengths
        hist_flux = np.sum(hist_image.arr)
        smoothed_flux = np.sum(smoothed_image.arr)

        # Attach units if present and compare in the expected photometry units
        if getattr(hist_image, "units", None) is not None:
            hist_flux = hist_flux * hist_image.units
        if getattr(smoothed_image, "units", None) is not None:
            smoothed_flux = smoothed_flux * smoothed_image.units

        hist_flux = hist_flux.to(expected_flux.units)
        smoothed_flux = smoothed_flux.to(expected_flux.units)

        hist_diff = 100 * abs(hist_flux - expected_flux) / expected_flux
        assert np.isclose(hist_flux, expected_flux, rtol=0.02), (
            f"Histogram flux {hist_flux} != expected {expected_flux} "
            f"(diff={hist_diff:.2f}%)."
        )
        smooth_diff = 100 * abs(smoothed_flux - expected_flux) / expected_flux
        assert np.isclose(smoothed_flux, expected_flux, rtol=0.02), (
            f"Smoothed flux {smoothed_flux} != expected {expected_flux} "
            f"(diff={smooth_diff:.2f}%)."
        )

    def test_orientation(self, one_part_galaxy, incident_emission_model):
        """Test image generation with different orientations."""
        # Define the image properties
        from synthesizer.instruments import Instrument

        resolution = 0.1 * kpc
        fov = 3.0 * kpc

        incident_emission_model.set_per_particle(True)

        kernel = Kernel().get_kernel()

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Get the image
        galaxy_image = one_part_galaxy.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            kernel=kernel,
        )["filter_r"]

        # Ensure the image has non-zero flux
        assert np.sum(galaxy_image.arr) > 0, (
            "Image should have non-zero total flux"
        )

        # Ensure at least one pixel is populated near the centre
        # (particle is at [0.05, 0.05, 0] kpc which is near centre)
        centre_region = galaxy_image.arr[
            13:17, 13:17
        ]  # 4x4 region near centre
        assert np.sum(centre_region) > 0, (
            "Image should have flux in the central region with the particle"
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
        from synthesizer.instruments import Instrument

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

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Create histogram image
        hist_image = random_part_stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
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
        from synthesizer.instruments import Instrument

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

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Create smoothed image
        smoothed_image = random_part_stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
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
        from synthesizer.instruments import Instrument

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

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Create smoothed image using threading
        threaded_image = random_part_stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
            nthreads=4,
        )["filter_r"]

        # Create smoothed image without threading
        serial_image = random_part_stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )["filter_r"]

        # Compare the two images
        assert np.allclose(threaded_image.arr, serial_image.arr), (
            "Threaded and serial smoothed images should be identical"
        )


class TestPixelOverlapFix:
    """Tests for pixel overlap fix in smoothed imaging.

    These tests verify that particles contribute to pixels that overlap
    with their kernel, even when the pixel centers fall outside the kernel
    support radius. This is especially important when smoothing lengths
    are comparable to or smaller than the pixel size.
    """

    def test_small_smoothing_length_pixel_overlap(self):
        """Test that particles with small smoothing lengths populate pixels.

        This is a regression test for the bug where particles whose kernel
        radius was smaller than the distance to pixel centers would not
        contribute to any pixels, even though the pixels overlapped with
        the kernel.
        """
        # Create a single particle at a specific position (centered coords)
        coords = unyt_array([[0.0, 0.0, 0.0]], kpc)
        signal = unyt_array([1e30], erg / s)
        # Small smoothing length comparable to pixel size
        smoothing_lengths = unyt_array([0.08], kpc)

        # Create image with resolution such that particle is near pixel edge
        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        kernel = Kernel().get_kernel()

        img.get_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        # The particle should contribute to multiple pixels
        non_zero_pixels = np.sum(img.arr > 0)
        assert non_zero_pixels > 0, (
            "Particle should contribute to at least one pixel"
        )
        assert non_zero_pixels >= 4, (
            f"Particle near pixel edge should contribute to multiple pixels, "
            f"but only {non_zero_pixels} pixels were populated"
        )

        # Total flux should be conserved
        total_flux = np.sum(img.arr)
        assert total_flux > 0, "Total flux should be positive"

    def test_particle_at_pixel_edge(self):
        """Test particle positioned exactly at pixel edges.

        When a particle is at a pixel edge, its kernel should contribute
        to neighboring pixels even if their centers are > 1 smoothing
        length away.
        """
        # Position mirrored particles near opposite pixel boundaries to keep
        # the centred coordinate requirement satisfied
        coords = unyt_array([[-0.15, 0.15, 0.0], [0.15, -0.15, 0.0]], kpc)
        signal = unyt_array([1e30, 1e30], erg / s)
        smoothing_lengths = unyt_array([0.1, 0.1], kpc)

        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        kernel = Kernel().get_kernel()

        img.get_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        non_zero_pixels = np.sum(img.arr > 0)
        assert non_zero_pixels >= 4, (
            "Edge particles should populate neighboring pixels; "
            f"populated {non_zero_pixels}"
        )

    def test_flux_conservation_with_small_kernels(self):
        """Test that flux is conserved when kernel is smaller than pixels.

        The total flux in the image should approximately equal the input
        signal, even when using small kernels that require pixel overlap
        detection.
        """
        # Single particle with known signal (centered)
        coords = unyt_array([[0.0, 0.0, 0.0]], kpc)
        signal_value = 1e30
        signal = unyt_array([signal_value], erg / s)
        smoothing_lengths = unyt_array([0.08], kpc)

        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        kernel = Kernel().get_kernel()

        img.get_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        total_flux = np.sum(img.arr)
        # With proper pixel integration, flux conservation should be
        # much better. We expect to conserve most of the flux even
        # with small smoothing lengths.
        assert total_flux > 0, (
            f"No flux in image: input={signal_value:.2e}, "
            f"output={total_flux:.2e}"
        )
        # Check we're capturing at least 50% of flux (conservative bound)
        assert total_flux > 0.5 * signal_value, (
            f"Too much flux lost: input={signal_value:.2e}, "
            f"output={total_flux:.2e} ({100 * total_flux / signal_value:.2f}%)"
        )
        # Also check we're not wildly over-counting (should be < 2x input)
        assert total_flux < 2 * signal_value, (
            f"Flux over-counted: input={signal_value:.2e}, "
            f"output={total_flux:.2e}"
        )

    def test_multiple_particles_with_varying_smoothing_lengths(self):
        """Test imaging with particles having different smoothing lengths.

        Ensures the pixel overlap fix works correctly when particles have
        varying smoothing lengths, including some smaller than pixel size.
        """
        n_particles = 5
        coords = unyt_array(
            [
                [-0.3, -0.3, 0.0],
                [-0.1, -0.1, 0.0],
                [0.1, -0.3, 0.0],
                [-0.3, 0.1, 0.0],
                [0.0, 0.0, 0.0],
            ],
            kpc,
        )
        signal = unyt_array([1e30] * n_particles, erg / s)
        # Mix of smoothing lengths
        smoothing_lengths = unyt_array([0.05, 0.08, 0.12, 0.15, 0.20], kpc)

        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        kernel = Kernel().get_kernel()

        img.get_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        # All particles should contribute
        assert np.sum(img.arr) > 0, "Image should have non-zero flux"
        non_zero_pixels = np.sum(img.arr > 0)
        assert non_zero_pixels >= n_particles, (
            f"Expected at least {n_particles} non-zero pixels, "
            f"got {non_zero_pixels}"
        )

    def test_very_small_smoothing_length(self):
        """Test that very small smoothing lengths are handled correctly.

        Even with very small smoothing lengths, the kernel integration
        should still work properly and give sensible results.
        """
        coords = unyt_array([[0.0, 0.0, 0.0]], kpc)
        signal_value = 1e30
        signal = unyt_array([signal_value], erg / s)
        # Very small smoothing length
        smoothing_lengths = unyt_array([0.04], kpc)  # < res/2

        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        kernel = Kernel().get_kernel()

        img.get_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        # Should have at least 1 non-zero pixel
        non_zero_pixels = np.sum(img.arr > 0)
        assert non_zero_pixels >= 1, (
            f"Very small smoothing length should populate at least 1 pixel, "
            f"got {non_zero_pixels}"
        )

        # Check we get some flux
        total_flux = np.sum(img.arr)
        assert total_flux > 0, (
            "Should have non-zero flux for very small smoothing length"
        )

    def test_comparison_with_large_kernels(self):
        """Test that fix doesn't break imaging with large kernels.

        Particles with large smoothing lengths should still work correctly
        and produce similar results to before the fix.
        """
        coords = unyt_array([[0.0, 0.0, 0.0]], kpc)
        signal = unyt_array([1e30], erg / s)
        # Large smoothing length
        smoothing_lengths = unyt_array([0.3], kpc)

        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        kernel = Kernel().get_kernel()

        img.get_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        # Large kernel should populate many pixels
        non_zero_pixels = np.sum(img.arr > 0)
        assert non_zero_pixels > 16, (
            f"Large kernel should populate many pixels, got {non_zero_pixels}"
        )

        # Check that the peak is at the center
        peak_idx = np.unravel_index(np.argmax(img.arr), img.arr.shape)
        # Particle is at (0, 0) kpc, which is the center pixel (5, 5)
        # in 10x10 image
        expected_peak = (5, 5)
        # Allow some tolerance
        assert np.abs(peak_idx[0] - expected_peak[0]) <= 1, (
            "Peak should be near particle position"
        )
        assert np.abs(peak_idx[1] - expected_peak[1]) <= 1, (
            "Peak should be near particle position"
        )


class TestComprehensiveImagingCoverage:
    """Comprehensive tests for imaging flux conservation and edge cases.

    This test suite ensures complete coverage of all imaging branches and
    validates flux conservation across different regimes and edge cases.
    """

    def test_flux_conservation_very_small_smoothing_length(self):
        """Test flux conservation with very small smoothing lengths (< pixel).

        This tests the pixel overlap fix - when smoothing length is much
        smaller than the pixel size, the pixel overlap approach should still
        capture most of the flux.
        """
        from synthesizer.emission_models import IncidentEmission
        from synthesizer.grid import Grid
        from synthesizer.particle import Stars as ParticleStars

        # Create a simple grid
        grid = Grid("test_grid")

        # Create particles with very small smoothing lengths
        n_stars = 100
        coords = unyt_array(np.random.uniform(-50, 50, (n_stars, 3)), kpc)
        ages = unyt_array([100.0] * n_stars, Myr)
        masses = unyt_array([1e6] * n_stars, Msun)
        metallicities = np.array([0.01] * n_stars)
        # Very small smoothing lengths - 50x smaller than pixel
        smoothing_lengths = unyt_array([0.01] * n_stars, kpc)

        stars = ParticleStars(
            ages=ages,
            initial_masses=masses,
            current_masses=masses,
            metallicities=metallicities,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            centre=unyt_array([0.0, 0.0, 0.0], kpc),
        )

        # Create emission model and get spectra
        emission_model = IncidentEmission(grid=grid, per_particle=True)
        stars.get_spectra(emission_model)
        stars.get_particle_photo_lnu(
            FilterCollection(
                generic_dict={"filter_r": np.ones(1000)},
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )
        stars.get_photo_lnu(
            FilterCollection(
                generic_dict={"filter_r": np.ones(1000)},
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )

        # Create image with large pixels relative to smoothing length
        from synthesizer.instruments import Instrument

        resolution = 0.5 * kpc
        fov = 200 * kpc
        kernel = Kernel().get_kernel()

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        smoothed_image = stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )["filter_r"]

        smoothed_flux = np.sum(smoothed_image.arr)
        expected_flux = stars.photo_lnu["incident"]["filter_r"]

        # Flux MUST be conserved - no exceptions!
        # Use 1% tolerance only to account for numerical precision
        assert np.isclose(smoothed_flux, expected_flux, rtol=0.01), (
            f"Smoothed flux {smoothed_flux} does not match expected flux "
            f"{expected_flux} - FLUX MUST BE CONSERVED! "
            f"Captured {smoothed_flux / expected_flux * 100:.1f}% of flux."
        )

    def test_flux_conservation_medium_smoothing_length(self):
        """Test flux conservation with medium smoothing lengths (~ pixel).

        When smoothing length equals pixel size, flux conservation should
        be good.
        """
        from synthesizer.emission_models import IncidentEmission
        from synthesizer.grid import Grid
        from synthesizer.particle import Stars as ParticleStars

        # Create a simple grid
        grid = Grid("test_grid")

        # Create particles with medium smoothing lengths
        n_stars = 100
        coords = unyt_array(np.random.uniform(-50, 50, (n_stars, 3)), kpc)
        ages = unyt_array([100.0] * n_stars, Myr)
        masses = unyt_array([1e6] * n_stars, Msun)
        metallicities = np.array([0.01] * n_stars)
        # Medium smoothing lengths - equal to pixel size
        smoothing_lengths = unyt_array([0.5] * n_stars, kpc)

        stars = ParticleStars(
            ages=ages,
            initial_masses=masses,
            current_masses=masses,
            metallicities=metallicities,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            centre=unyt_array([0.0, 0.0, 0.0], kpc),
        )

        # Create emission model and get spectra
        emission_model = IncidentEmission(grid=grid, per_particle=True)
        stars.get_spectra(emission_model)
        stars.get_particle_photo_lnu(
            FilterCollection(
                generic_dict={"filter_r": np.ones(1000)},
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )
        stars.get_photo_lnu(
            FilterCollection(
                generic_dict={"filter_r": np.ones(1000)},
                new_lam=np.linspace(4000, 8000, 1000) * angstrom,
            )
        )

        # Create image
        from synthesizer.instruments import Instrument

        resolution = 0.5 * kpc
        fov = 200 * kpc
        kernel = Kernel().get_kernel()

        # Create an instrument for the image
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        smoothed_image = stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )["filter_r"]

        smoothed_flux = np.sum(smoothed_image.arr)
        expected_flux = stars.photo_lnu["incident"]["filter_r"]

        # Flux MUST be conserved - no exceptions!
        # Use 1% tolerance only to account for numerical precision
        assert np.isclose(smoothed_flux, expected_flux, rtol=0.01), (
            f"Smoothed flux {smoothed_flux} does not match expected flux "
            f"{expected_flux} - FLUX MUST BE CONSERVED!"
        )

    def test_flux_conservation_large_smoothing_length(
        self,
        random_part_stars,
        incident_emission_model,
    ):
        """Test flux conservation with large smoothing lengths (>> pixel).

        When smoothing length is much larger than the pixel size, flux
        conservation should still be excellent.
        """
        # Use large smoothing lengths - much larger than pixel
        random_part_stars.smoothing_lengths = (
            np.ones(random_part_stars.nstars) * 5.0 * kpc
        )

        # Define the image properties
        resolution = 0.5 * kpc  # 10x smaller than smoothing length
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

        # Create an instrument for the image
        from synthesizer.instruments import Instrument

        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Create smoothed image
        smoothed_image = random_part_stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )["filter_r"]

        # Get the sum of the smoothed image
        smoothed_flux = np.sum(smoothed_image.arr)

        # Get the true photometry
        expected_flux = random_part_stars.photo_lnu["incident"]["filter_r"]

        # Large smoothing lengths should give excellent flux conservation
        assert np.isclose(smoothed_flux, expected_flux, rtol=1e-3), (
            f"Smoothed flux {smoothed_flux} does not match expected flux "
            f"{expected_flux} within 0.1% tolerance for large "
            f"smoothing lengths"
        )

    def test_edge_particles_flux_conservation(self):
        """Test flux conservation for particles near image boundaries.

        Particles at the edge of the image should have their flux properly
        conserved even when part of their kernel extends beyond the image.
        """
        # Create particles at various edge positions
        coords = unyt_array(
            [
                [-4.5, 0.0, 0.0],  # Left edge
                [4.5, 0.0, 0.0],  # Right edge
                [0.0, -4.5, 0.0],  # Bottom edge
                [0.0, 4.5, 0.0],  # Top edge
                [-4.5, -4.5, 0.0],  # Bottom-left corner
                [4.5, 4.5, 0.0],  # Top-right corner
            ],
            kpc,
        )

        smoothing_lengths = unyt_array([1.0] * 6, kpc)

        # Simple signal - uniform for all particles
        signal = unyt_array([1e30] * 6, erg / s)
        expected_total = np.sum(signal)

        # Create image
        resolution = 0.5 * kpc
        fov = 10.0 * kpc
        kernel = Kernel().get_kernel()

        img = Image(resolution=resolution, fov=fov)
        img.get_img_smoothed(
            signal=signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        # The image will capture partial flux from edge particles
        # We expect at least 50% of total flux to be captured
        image_flux = np.sum(img.arr)
        assert image_flux >= 0.5 * expected_total, (
            f"Edge particles should contribute at least 50% of flux, "
            f"got {image_flux / expected_total * 100:.1f}%"
        )

    def test_overlapping_particles(self):
        """Test that overlapping particles correctly add their contributions.

        When multiple particles overlap in space, their flux contributions
        should add linearly.
        """
        # Create two particles at the same position with identical properties
        # Centered coordinates
        coords = unyt_array(
            [
                [-0.01, -0.01, 0.0],
                [0.01, 0.01, 0.0],
            ],
            kpc,
        )

        smoothing_lengths = unyt_array([1.0, 1.0], kpc)

        # Give both particles the same signal
        signal = unyt_array([1e30, 1e30], erg / s)

        # Create image
        resolution = 0.5 * kpc
        fov = 10.0 * kpc
        kernel = Kernel().get_kernel()

        img = Image(resolution=resolution, fov=fov)
        img.get_img_smoothed(
            signal=signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        # Total flux should be sum of both particles
        expected_flux = np.sum(signal)
        image_flux = np.sum(img.arr)

        # Flux MUST be conserved - 1% tolerance for numerical precision
        assert np.isclose(image_flux, expected_flux, rtol=0.01), (
            f"Overlapping particles flux {image_flux} should equal "
            f"sum of individual fluxes {expected_flux} - "
            f"FLUX MUST BE CONSERVED!"
        )

    def test_different_kernel_thresholds(self):
        """Test that different kernel thresholds don't break flux conservation.

        The kernel threshold determines the support radius. Different
        thresholds should still conserve flux properly.
        """
        # Create a single particle at center (use all zeros since it's
        # a single point and np.all(np.isclose(coords, 0)) will be True)
        coords = unyt_array([[0.0, 0.0, 0.0]], kpc)
        smoothing_lengths = unyt_array([1.0], kpc)
        signal = unyt_array([1e30], erg / s)

        # Test different thresholds
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            kernel = Kernel().get_kernel()

            resolution = 0.2 * kpc
            fov = 10.0 * kpc

            img = Image(resolution=resolution, fov=fov)
            img.get_img_smoothed(
                signal=signal,
                coordinates=coords,
                smoothing_lengths=smoothing_lengths,
                kernel=kernel,
                kernel_threshold=threshold,
            )

            image_flux = np.sum(img.arr)

            # Flux MUST be conserved regardless of kernel threshold
            # 1% tolerance for numerical precision only
            assert np.isclose(image_flux, signal[0], rtol=0.01), (
                f"Kernel threshold {threshold} gives flux {image_flux}, "
                f"expected {signal[0]} - FLUX MUST BE CONSERVED!"
            )

    def test_multifilter_imaging_flux_conservation(
        self,
        random_part_stars,
        incident_emission_model,
    ):
        """Test flux conservation when generating multiple filters.

        When creating images for multiple filters simultaneously, each
        filter should independently conserve flux.
        """
        # Define the image properties
        resolution = 0.5 * kpc
        fov = 200 * kpc

        incident_emission_model.set_per_particle(True)

        random_part_stars.get_spectra(incident_emission_model)

        # Create multiple filters
        filters = FilterCollection(
            generic_dict={
                "filter_r": np.ones(1000),
                "filter_g": np.ones(1000),
                "filter_b": np.ones(1000),
            },
            new_lam=np.linspace(4000, 8000, 1000) * angstrom,
        )

        random_part_stars.get_particle_photo_lnu(filters)
        random_part_stars.get_photo_lnu(filters)

        random_part_stars.centre = np.array([0.0, 0.0, 0.0]) * kpc

        kernel = Kernel().get_kernel()

        # Create an instrument for the image
        from synthesizer.instruments import Instrument

        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Create images for all filters
        images = random_part_stars.get_images_luminosity(
            "incident",
            fov=fov,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )

        # Check flux conservation for each filter
        for filter_name in ["filter_r", "filter_g", "filter_b"]:
            image_flux = np.sum(images[filter_name].arr)
            expected_flux = random_part_stars.photo_lnu["incident"][
                filter_name
            ]

            # Use 1% tolerance for multi-filter imaging
            assert np.isclose(image_flux, expected_flux, rtol=0.01), (
                f"Filter {filter_name} flux {image_flux} does not match "
                f"expected {expected_flux} within 1%"
            )

    def test_histogram_equals_smoothed_for_large_smoothing(self):
        """Test that histogram and smoothed converge for large smoothing.

        When smoothing length is very large relative to FOV, the smoothed
        image should approach a uniform distribution similar to histogram.
        """
        # Create particles spread across FOV (already centered)
        np.random.seed(42)  # For reproducibility
        n_particles = 50
        coords = unyt_array(np.random.uniform(-5, 5, (n_particles, 3)), kpc)
        coords[:, 2] = 0.0  # Keep z=0

        # Very large smoothing lengths - comparable to FOV
        smoothing_lengths = unyt_array([5.0] * n_particles, kpc)
        signal = unyt_array(np.ones(n_particles) * 1e30, erg / s)

        resolution = 0.5 * kpc
        fov = 50.0 * kpc
        kernel = Kernel().get_kernel()

        # Create histogram image
        hist_img = Image(resolution=resolution, fov=fov)
        hist_img.get_img_hist(signal=signal, coordinates=coords)

        # Create smoothed image
        smooth_img = Image(resolution=resolution, fov=fov)
        smooth_img.get_img_smoothed(
            signal=signal,
            coordinates=coords,
            smoothing_lengths=smoothing_lengths,
            kernel=kernel,
        )

        # Both should conserve total flux
        hist_flux = np.sum(hist_img.arr)
        smooth_flux = np.sum(smooth_img.arr)
        expected_flux = np.sum(signal)

        assert np.isclose(hist_flux, expected_flux, rtol=1e-10), (
            f"Histogram flux {hist_flux} != expected {expected_flux}"
        )
        # Flux MUST be conserved - 1% tolerance for numerical precision only
        assert np.isclose(smooth_flux, expected_flux, rtol=0.01), (
            f"Smoothed flux {smooth_flux} != expected {expected_flux} - "
            f"FLUX MUST BE CONSERVED!"
        )

    def test_single_particle_different_positions(self):
        """Test single particle imaging at different positions within a pixel.

        A particle at different sub-pixel positions should have similar
        total flux but potentially different pixel distributions.
        """
        signal_value = 1e30
        resolution = 0.5 * kpc
        fov = 10.0 * kpc
        kernel = Kernel().get_kernel()
        smoothing_length = 0.5 * kpc

        # Test particle at different sub-pixel positions (paired with mirror
        # to satisfy centering requirement). Keep offsets modest to avoid
        # numerical drift dominating the comparison.
        positions = [
            [0.0, 0.0, 0.0],  # Center of pixel
            [0.1, 0.1, 0.0],  # Small offset within pixel
            [0.08, -0.06, 0.0],  # Another small offset
        ]

        fluxes = []
        for pos in positions:
            coords = unyt_array([pos, [-pos[0], -pos[1], -pos[2]]], kpc)
            smoothing_lengths = unyt_array(
                [smoothing_length.value, smoothing_length.value], kpc
            )
            signal = unyt_array([signal_value / 2, signal_value / 2], erg / s)

            img = Image(resolution=resolution, fov=fov)
            img.get_img_smoothed(
                signal=signal,
                coordinates=coords,
                smoothing_lengths=smoothing_lengths,
                kernel=kernel,
            )

            fluxes.append(np.sum(img.arr))

        # All positions should conserve total flux within 1% tolerance
        # (accounting for numerical precision in different pixel distributions)
        for i, flux in enumerate(fluxes):
            assert np.isclose(flux, signal_value, rtol=0.01), (
                f"Position {positions[i]} gives flux {flux}, "
                f"expected {signal_value} - flux conservation failed"
            )

    def test_threading_consistency_detailed(
        self,
        random_part_stars,
        incident_emission_model,
    ):
        """Test that different thread counts give identical results.

        Threading should not affect the output - this tests serial vs
        multiple thread counts.
        """
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

        # Create an instrument for the image
        from synthesizer.instruments import Instrument

        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )

        # Generate images with different thread counts
        thread_counts = [1, 2, 4, 8]
        images = {}

        for nthreads in thread_counts:
            img = random_part_stars.get_images_luminosity(
                "incident",
                fov=fov,
                instrument=instrument,
                img_type="smoothed",
                kernel=kernel,
                nthreads=nthreads,
            )["filter_r"]
            images[nthreads] = img

        # All images should be identical
        reference_img = images[1].arr
        for nthreads in thread_counts[1:]:
            assert np.allclose(images[nthreads].arr, reference_img), (
                f"Image with {nthreads} threads differs from serial (1 thread)"
            )


class TestCombinedModelImaging:
    """Test imaging with combined emission models (e.g., nebular).

    This class tests the critical functionality of generating images for
    models that combine multiple sub-models, ensuring the model_param_cache
    is properly accessed across component boundaries.
    """

    def test_galaxy_nebular_imaging(
        self, nebular_emission_model, random_part_stars
    ):
        """Test galaxy-level imaging with nebular (combined) model.

        The nebular model combines nebular_line and nebular_continuum.
        This tests that:
        1. Component-level spectra/photometry are generated correctly
        2. The combined_cache is built properly at galaxy level
        3. _combine_image_collections can access the cache from components
        4. Images are correctly combined
        """
        from synthesizer.instruments import Instrument
        from synthesizer.particle.galaxy import Galaxy

        # Set up the model
        nebular_emission_model.set_per_particle(True)

        # Create a galaxy with stellar component
        galaxy = Galaxy(
            name="test_galaxy",
            stars=random_part_stars,
            redshift=0.1,
        )
        galaxy.stars.centre = np.array([0.0, 0.0, 0.0]) * kpc

        # Generate spectra for all models (nebular_line, nebular_continuum)
        galaxy.get_spectra(nebular_emission_model)

        # Generate photometry
        filters = FilterCollection(
            generic_dict={
                "filter_r": np.ones(1000),
            },
            new_lam=np.linspace(4000, 8000, 1000) * angstrom,
        )
        galaxy.get_photo_lnu(filters)

        # Create instrument
        resolution = 0.5 * kpc
        fov = 200 * kpc
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )
        kernel = Kernel().get_kernel()

        # Generate images - this should combine nebular_line and
        # nebular_continuum into nebular
        nebular_img = galaxy.get_images_luminosity(
            "nebular",  # Combined model (single label returns ImageCollection)
            fov=fov,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )

        # Verify the nebular image was created
        assert nebular_img is not None, "Nebular image is None"

        # Get filter names from the ImageCollection
        filter_names = list(nebular_img.imgs.keys())
        assert len(filter_names) > 0, "No filters in nebular ImageCollection"

        # Use the first filter (should be filter_r)
        filter_name = filter_names[0]
        assert nebular_img[filter_name].arr is not None, (
            "Nebular image array is None"
        )

        # Verify flux conservation - the combined image should have the
        # sum of line and continuum photometry
        nebular_line_phot = galaxy.stars.photo_lnu["nebular_line"][filter_name]
        nebular_continuum_phot = galaxy.stars.photo_lnu["nebular_continuum"][
            filter_name
        ]
        expected_total = nebular_line_phot + nebular_continuum_phot

        image_flux = np.sum(nebular_img[filter_name].arr)

        assert np.isclose(image_flux, expected_total, rtol=0.01), (
            f"Nebular combined image flux {image_flux} does not match "
            f"expected {expected_total} (line + continuum)"
        )

    def test_component_level_combined_imaging(
        self, nebular_emission_model, random_part_stars
    ):
        """Test component-level imaging with combined model.

        Components should be able to generate images for their own combined
        models using their own model_param_cache.
        """
        from synthesizer.instruments import Instrument

        nebular_emission_model.set_per_particle(True)

        # Generate spectra and photometry on the component directly
        random_part_stars.get_spectra(nebular_emission_model)
        filters = FilterCollection(
            generic_dict={
                "filter_r": np.ones(1000),
            },
            new_lam=np.linspace(4000, 8000, 1000) * angstrom,
        )
        random_part_stars.get_particle_photo_lnu(filters)
        random_part_stars.get_photo_lnu(filters)
        random_part_stars.centre = np.array([0.0, 0.0, 0.0]) * kpc

        # Create instrument
        resolution = 0.5 * kpc
        fov = 200 * kpc
        instrument = Instrument(
            label="test_inst",
            filters=None,
            resolution=resolution,
        )
        kernel = Kernel().get_kernel()

        # Generate images at component level (single label returns
        # ImageCollection)
        nebular_img = random_part_stars._get_images(
            "nebular",
            img_type="smoothed",
            instrument=instrument,
            kernel=kernel,
            fov=fov,
            resolution=resolution,
            phot_type="lnu",
        )

        # Verify the image was created and combined properly
        assert nebular_img is not None, (
            "Nebular image is None at component level"
        )

        # Get filter names from the ImageCollection
        filter_names = list(nebular_img.imgs.keys())
        assert len(filter_names) > 0, "No filters in nebular ImageCollection"

        # Use the first filter
        filter_name = filter_names[0]
        image_flux = np.sum(nebular_img[filter_name].arr)
        expected_flux = random_part_stars.photo_lnu["nebular"][filter_name]

        assert np.isclose(image_flux, expected_flux, rtol=0.01), (
            f"Component-level nebular image flux {image_flux} does not "
            f"match expected {expected_flux}"
        )
