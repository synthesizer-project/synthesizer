"""Definitions for the SpectralCube class.

This file contains the definitions for the SpectralCube class. This class
is used to generate and store spectral data cubes. This can be done in two
ways: by sorting particle spectra into the data cube or by smoothing
particles/a density grid over the data cube.

This file is part of the synthesizer package and is distributed under the
terms of the MIT license. See the LICENSE.md file for details.

Example usage:
    # Create a data cube
    cube = SpectralCube(
        resolution=0.1,
        lam=np.arange(1000, 2000, 1),
        fov=1,
    )

    # Get a hist data cube
    cube.get_data_cube_hist(
        sed=sed,
        coordinates=coordinates,
    )

    # Get a smoothed data cube
    cube.get_data_cube_smoothed(
        sed=sed,
        coordinates=coordinates,
        smoothing_lengths=smoothing_lengths,
        kernel=kernel,
        kernel_threshold=kernel_threshold,
        quantity="lnu",
    )
"""
import numpy as np

import synthesizer.exceptions as exceptions
from synthesizer.units import Quantity


class SpectralCube:
    """
    The Spectral data cube object.

    This object is used to generate and store spectral data cube. This can be
    done in two ways: by sorting particle spectra into the data cube or by
    smoothing particles/a density grid over the data cube.

    Attributes:
        lam (unyt_array, float):
            The wavelengths of the data cube.
        resolution (unyt_quantity, float):
            The spatial resolution of the data cube.
        fov (unyt_array, float/tuple):
            The field of view of the data cube. If a single value is given,
            the FOV is assumed to be square.
        npix (unyt_array, int/tuple):
            The number of pixels in the data cube. If a single value is given,
            the number of pixels is assumed to be square.
        arr (array_like, float):
            A 3D array containing the data cube. (npix[0], npix[1], lam.size)
        units (unyt_quantity, float):
            The units of the data cube.
    """

    # Define quantities
    lam = Quantity()
    resolution = Quantity()
    fov = Quantity()

    def __init__(
        self,
        resolution,
        lam,
        fov=None,
        npix=None,
    ):
        """
        Intialise the SpectralCube.

        Either fov or npix must be given. If both are given, fov is used.

        Args:
            resolution (unyt_quantity, float):
                The spatial resolution of the data cube.
            lam (unyt_array, float):
                The wavelengths of the data cube.
            fov (unyt_array, float/tuple):
                The field of view of the data cube. If a single value is
                given, the FOV is assumed to be square.
            npix (unyt_array, int/tuple):
                The number of pixels in the data cube. If a single value is
                given, the number of pixels is assumed to be square.

        """
        # Attach resolution, fov, and npix
        self.resolution = resolution
        self.fov = fov
        self.npix = npix

        # If fov isn't a array, make it one
        if self.fov is not None and self.fov.size == 1:
            self.fov = np.array((self.fov, self.fov))

        # If npix isn't an array, make it one
        if npix is not None and not isinstance(npix, np.ndarray):
            if isinstance(npix, int):
                self.npix = np.array((npix, npix))
            else:
                self.npix = np.array(npix)

        # Keep track of the input resolution and and npix so we can handle
        # super resolution correctly.
        self.orig_resolution = resolution
        self.orig_npix = npix

        # Handle the different input cases
        if npix is None:
            self._compute_npix()
        else:
            self._compute_fov()

        # Store the wavelengths
        self.lam = lams

        # Attribute to hold the IFU array. This is populated later and
        # allocated in the C extensions or when needed.
        self.arr = None

        # Define an attribute to hold the units
        self.units = None

    @property
    def data_cube(self):
        """
        Return the data cube.

        This is a property to allow the data cube to be accessed as an
        attribute.

        Returns:
            array_like (float):
                A 3D array containing the data cube. (npix[0], npix[1],
                lam.size)
        """
        return self.arr * self.units

    def _compute_npix(self):
        """
        Compute the number of pixels in the FOV.

        When resolution and fov are given, the number of pixels is computed
        using this function. This can redefine the fov to ensure the FOV
        is an integer number of pixels.
        """
        # Compute how many pixels fall in the FOV
        self.npix = np.int32(np.ceil(self._fov / self._resolution))
        if self.orig_npix is None:
            self.orig_npix = np.int32(np.ceil(self._fov / self._resolution))

        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix

    def _compute_fov(self):
        """
        Compute the FOV, based on the number of pixels.

        When resolution and npix are given, the FOV is computed using this
        function.
        """
        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix

    def get_data_cube_hist(
        self,
        sed,
        coordinates=None,
        quantity="lnu",
    ):
        """
        Calculate a spectral data cube with no smoothing.

        This is only applicable to particle based spectral cubes.

        Args:
            sed (Sed):
                The Sed object containing the spectra to be sorted into the
                data cube.
            coordinates (unyt_array, float):
                The coordinates of the particles.
            quantity (str):
                The Sed attribute/quantity to sort into the data cube.

        Returns:
            array_like (float):
                A 3D array containing particle spectra sorted into the data
                cube. (npix[0], npix[1], lam.size)
        """
        # Sample the spectra onto the wavelength grid
        sed = sed.get_resampled_spectra(new_lam=self.lam)

        # Get the spectra we will be sorting into the spectral cube
        spectra = getattr(sed, quantity)

        # Strip off and store the units on the spectra for later
        self.units = spectra.units
        spectra = spectra.value

        from .extensions.spectral_cube import make_data_cube_hist

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value

        # In case coordinates haven't been centered we need to centre them
        if not (coordinates.min() < 0 and coordinates.max() > 0):
            coordinates -= np.average(
                coordinates, axis=0, weights=np.sum(spectra, axis=1)
            )

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        spectra = np.ascontiguousarray(spectra, dtype=np.float64)
        xs = np.ascontiguousarray(coordinates[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(coordinates[:, 1], dtype=np.float64)

        self.arr = make_data_cube_hist(
            spectra,
            xs,
            ys,
            self._resolution,
            self.npix[0],
            self.npix[1],
            coordinates.shape[0],
            self.lams.size,
        )

        return self.arr * self.units

    def get_data_cube_smoothed(
        self,
        sed,
        coordinates=None,
        smoothing_lengths=None,
        kernel=None,
        kernel_threshold=1,
        density_grid=None,
        quantity="lnu",
    ):
        """
        Calculate a spectral data cube with smoothing.

        In the particle case this smooths each particle's signal over the SPH
        kernel defined by their smoothing length. This uses C extensions to
        calculate the image for each particle efficiently.

        In the parametric case the signal is smoothed over a density grid. This
        density grid is an array defining the weight in each pixel.

        Args:
            sed (Sed):
                The Sed object containing the spectra to be sorted into the
                data cube.
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
            quantity (str):
                The Sed attribute/quantity to sort into the data cube.

        Returns:
            array_like (float):
                A 3D array containing particle spectra sorted into the data
                cube. (npix[0], npix[1], lam.size)

        Raises:
            InconsistentArguments
                If conflicting particle and parametric arguments are passed
                or any arguments are missing an error is raised.
        """
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

        # Sample the spectra onto the wavelength grid
        sed = sed.get_resampled_spectra(new_lam=self.lam)

        # Get the spectra we will be sorting into the spectral cube
        spectra = getattr(sed, quantity)

        # Strip off and store the units on the spectra for later
        self.units = spectra.units
        spectra = spectra.value

        # Handle the parametric case
        if density_grid is not None:
            # Multiply the density grid by the sed to get the IFU
            self.arr = density_grid[:, :] * spectra

            return self.arr * self.units

        from .extensions.spectral_cube import make_data_cube_smooth

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value
        smoothing_lengths = smoothing_lengths.to(self.resolution.units).value

        # In case coordinates haven't been centered we need to centre them
        if not (coordinates.min() < 0 and coordinates.max() > 0):
            coordinates -= np.average(
                coordinates, axis=0, weights=np.sum(spectra, axis=1)
            )

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        spectra = np.ascontiguousarray(spectra, dtype=np.float64)
        smls = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
        xs = np.ascontiguousarray(coordinates[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(coordinates[:, 1], dtype=np.float64)

        self.arr = make_data_cube_smooth(
            spectra,
            smls,
            xs,
            ys,
            kernel,
            self._resolution,
            self.npix[0],
            self.npix[1],
            coordinates.shape[0],
            self.lams.size,
            kernel_threshold,
            kernel.size,
        )

        return self.arr * self.units

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
