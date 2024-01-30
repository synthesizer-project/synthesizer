"""Definitions for the SpectralCube class."""
import numpy as np
import warnings
from unyt import kpc, mas, unyt_array, unyt_quantity
from unyt.dimensions import angle

import synthesizer.exceptions as exceptions
from synthesizer.units import Quantity


class SpectralCube:
    """
    The Spectral data cube object.

    """

    # Define quantities
    lam = Quantity()
    resolution = Quantity()
    fov = Quantity()

    def __init__(
        self,
        resolution,
        lams,
        fov=None,
        npix=None,
    ):
        """
        Intialise the SpectralCube.

        """

        # Store the wavelengths
        self.lam = lams

        # Attribute to hold the IFU array. This is populated later and
        # allocated in the C extensions or when needed.
        self.arr = None

        # Define an attribute to hold the units
        self.units = None

    def _sample_spectra_onto_wavelength_grid(self, sed):
        """
        Sample spectra onto the wavelength grid.

        Args:
            spectra (array_like, float):
                The spectra to be sampled onto the wavelength grid.

        Returns:
            spectra (array_like, float):
                The spectra sampled onto the wavelength grid.
        """

        # Sample the spectra onto the wavelength grid
        resampled_spectra = sed.get_resampled_spectra(new_lam=self.lam)

        # If the requested quantity is a flux we need to call get_fnu on
        # this new spectra

        return spectra

    def get_img_hist(
        self,
        sed,
        coordinates=None,
        quantity="lnu",
    ):
        """
        Calculate an image with no smoothing.

        This is only applicable to particle based spectral cubes.

        Args:
            coordinates (unyt_array, float):
                The coordinates of the particles.

        Returns:
            img (array_like, float)
                A 2D array containing the pixel values sorted into the image.
                (npix, npix)
        """

        # Strip off and store the units on the spectra for later
        self.units = spectra.units
        spectra = spectra.value

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value

        # In case coordinates haven't been centered we need to centre them
        if not (coordinates.min() < 0 and coordinates.max() > 0):
            coordinates -= np.average(coordinates, axis=0, weights=signal)

        # Set up the IFU array
        self.ifu = np.zeros(
            (self.npix, self.npix, self.spectral_resolution), dtype=np.float64
        )

        # Loop over positions including the sed
        for ind in range(self.npart):
            # Skip particles outside the FOV
            if (
                coordinates[ind, 0] < 0
                or coordinates[ind, 1] < 0
                or coordinates[ind, 0] >= self.npix
                or coordinates[ind, 1] >= self.npix
            ):
                continue

            self.ifu[
                self.pix_pos[ind, 0], self.pix_pos[ind, 1], :
            ] += self.sed_values[ind, :]

        return self.ifu

        return self.arr * self.units if self.units is not None else self.arr

    def get_img_smoothed(
        self,
        spectra,
        coordinates=None,
        smoothing_lengths=None,
        kernel=None,
        kernel_threshold=1,
        density_grid=None,
    ):
        """
        Calculate a smoothed image.

        In the particle case this smooths each particle's signal over the SPH
        kernel defined by their smoothing length. This uses C extensions to
        calculate the image for each particle efficiently.

        In the parametric case the signal is smoothed over a density grid. This
        density grid is an array defining the weight in each pixel.

        Args:
            coordinates (unyt_array, float):
                The coordinates of the particles. (particle case only)
            smoothing_lengths (unyt_array, float):
                The smoothing lengths of the particles. (particle case only)
            kernel (str):
                The kernel to use for smoothing. (particle case only)
            kernel_threshold (float):
                The threshold for the kernel. (particle case only)
            density_grid (array_like, float):
                The density grid to smooth over. (parametric case only)

        Returns:
            img : array_like (float)
                A 2D array containing particles sorted into an image.
                (npix[0], npix[1])

        Raises:
            InconsistentArguments
                If conflicting particle and parametric arguments are passed
                or any arguments are missing an error is raised.
        """
        # Strip off and store the units on the signal if they are present
        if isinstance(signal, (unyt_quantity, unyt_array)):
            self.units = signal.units
            signal = signal.value

        # Ensure we have the right arguments
        if density_grid is not None and (
            coordinates is not None
            or smoothing_lengths is not None
            or kernel is not None
        ):
            raise exceptions.InconsistentArguments(
                "Parametric smoothed images only require a density grid. You "
                "Shouldn't have particle based quantities in conjunction with "
                "parametric properties, what are you doing?"
            )
        if density_grid is None and (
            coordinates is None or smoothing_lengths is None or kernel is None
        ):
            raise exceptions.InconsistentArguments(
                "Particle based smoothed images require the coordinates, "
                "smoothing_lengths, and kernel arguments to be passed."
            )

        # Handle the parametric case
        if density_grid is not None:
            # Multiply the density grid by the sed to get the IFU
            self.arr = density_grid[:, :] * spectra

            return (
                self.arr * self.units if self.units is not None else self.arr
            )

        from .extensions.spectral_cube import make_ifu

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value
        smoothing_lengths = smoothing_lengths.to(self.resolution.units).value

        # In case coordinates haven't been centered we need to centre them
        if not (coordinates.min() < 0 and coordinates.max() > 0):
            coordinates -= np.average(coordinates, axis=0, weights=signal)

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        spectra = np.ascontiguousarray(spectra, dtype=np.float64)
        smls = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
        xs = np.ascontiguousarray(coordinates[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(coordinates[:, 1], dtype=np.float64)

        self.ifu = make_ifu(
            spectra,
            smls,
            xs,
            ys,
            self.kernel,
            self.resolution,
            self.npix,
            self.coordinates.shape[0],
            self.spectral_resolution,
            self.kernel_threshold,
            self.kernel_dim,
        )

        return self.ifu

    def apply_psf(self):
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def apply_noise_array(self):
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def apply_noise_from_std(self):
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def apply_noise_from_snr(self):
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/flaresimulations/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )
