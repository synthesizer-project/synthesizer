"""Definitions for the SpectralCube class.

This file contains the definitions for the SpectralCube class. This class
is used to generate and store spectral data cubes. This can be done in two
ways: by sorting particle spectra into the data cube or by smoothing
particles/a density grid over the data cube.

This file is part of the synthesizer package and is distributed under the
terms of the MIT license. See the LICENSE.md file for details.

Example usage::

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

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize

from synthesizer import exceptions
from synthesizer.units import Quantity


class SpectralCube:
    """The Spectral data cube object.

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
        sed (Sed):
            The Sed used to generate the data cube.
        quantity (str):
            The Sed attribute/quantity to sort into the data cube, i.e.
            "lnu", "llam", "luminosity", "fnu", "flam" or "flux".
    """

    # Define quantities
    lam = Quantity("wavelength")
    resolution = Quantity("spatial")
    fov = Quantity("spatial")

    def __init__(
        self,
        resolution,
        lam,
        fov=None,
        npix=None,
    ):
        """Intialise the SpectralCube.

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
        self.lam = lam

        # Attribute to hold the IFU array. This is populated later and
        # allocated in the C extensions or when needed.
        self.arr = None

        # Define an attribute to hold the units
        self.units = None

        # Placeholders to store a pointer to the sed and quantity
        self.sed = None
        self.quantity = None

    @property
    def data_cube(self):
        """Return the data cube.

        This is a property to allow the data cube to be accessed as an
        attribute.

        Returns:
            array_like (float):
                A 3D array containing the data cube. (npix[0], npix[1],
                lam.size)
        """
        return self.arr * self.units

    def _compute_npix(self):
        """Compute the number of pixels in the FOV.

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
        """Compute the FOV, based on the number of pixels.

        When resolution and npix are given, the FOV is computed using this
        function.
        """
        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix

    def __add__(self, other):
        """Add two SpectralCubes together.

        This is done by adding the IFU arrays together but SpectralCubes can
        only be added if they have the same units, resolution, FOV, and
        wavelengths.

        Args:
            other (SpectralCube):
                The other spectral cube to add to this one.

        Returns:
            SpectralCube:
                The new spectral cube.
        """
        # Ensure there are data cubes to add
        if self.arr is None or other.arr is None:
            raise exceptions.InconsistentArguments(
                "Both spectral cubes must have been populated before they can "
                "be added together."
            )

        # Check the units are the same
        if self.units != other.units:
            raise exceptions.InconsistentArguments(
                "To add two spectral cubes together they must have the same "
                "units."
            )

        # Check the resolution is the same
        if self.resolution != other.resolution:
            raise exceptions.InconsistentArguments(
                "To add two spectral cubes together they must have the same "
                "resolution."
            )

        # Check the FOV is the same
        if np.any(self.fov != other.fov):
            raise exceptions.InconsistentArguments(
                "To add two spectral cubes together they must have the same "
                "FOV."
            )

        # Create the new spectral cube
        new_cube = SpectralCube(
            resolution=self.resolution,
            lam=self.lam,
            fov=self.fov,
        )

        # Add the data cube arrays together
        new_cube.arr = self.arr + other.arr

        # Add the attached seds
        new_cube.sed = self.sed + other.sed

        # Set the quantity
        new_cube.quantity = self.quantity

        return new_cube

    def get_data_cube_hist(
        self,
        sed,
        coordinates=None,
        quantity="lnu",
        nthreads=1,
    ):
        """Calculate a spectral data cube with no smoothing.

        This is only applicable to particle based spectral cubes.

        Args:
            sed (Sed):
                The Sed object containing the spectra to be sorted into the
                data cube.
            coordinates (unyt_array, float):
                The coordinates of the particles.
            quantity (str):
                The Sed attribute/quantity to sort into the data cube, i.e.
                "lnu", "llam", "luminosity", "fnu", "flam" or "flux".
            nthreads (int):
                The number of threads to use for the C extensions.

        Returns:
            array_like (float):
                A 3D array containing particle spectra sorted into the data
                cube. (npix[0], npix[1], lam.size)
        """
        # Sample the spectra onto the wavelength grid if we need to
        if not np.array_equal(self.lam, sed.lam):
            sed = sed.get_resampled_sed(new_lam=self.lam)

        # Store the Sed and quantity
        self.sed = sed
        self.quantity = quantity

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
        xs = np.ascontiguousarray(
            coordinates[:, 0] + (self._fov[0] / 2), dtype=np.float64
        )
        ys = np.ascontiguousarray(
            coordinates[:, 1] + (self._fov[1] / 2), dtype=np.float64
        )

        self.arr = make_data_cube_hist(
            spectra,
            xs,
            ys,
            self._resolution,
            self.npix[0],
            self.npix[1],
            coordinates.shape[0],
            self.lam.size,
            nthreads,
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
        nthreads=1,
    ):
        """Calculate a spectral data cube with smoothing.

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
                The Sed attribute/quantity to sort into the data cube, i.e.
                "lnu", "llam", "luminosity", "fnu", "flam" or "flux".
            nthreads (int):
                The number of threads to use for the C extensions. (particle
                case only).

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

        # Sample the spectra onto the wavelength grid if we need to
        if not np.array_equal(self.lam, sed.lam):
            sed = sed.get_resampled_sed(new_lam=self.lam)

        # Store the Sed and quantity
        self.sed = sed
        self.quantity = quantity

        # Get the spectra we will be sorting into the spectral cube
        spectra = getattr(sed, quantity)

        # Strip off and store the units on the spectra for later
        self.units = spectra.units
        spectra = spectra.value

        # Handle the parametric case
        if density_grid is not None:
            # Multiply the density grid by the sed to get the IFU
            self.arr = density_grid[:, :, None] * spectra

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
        xs = np.ascontiguousarray(
            coordinates[:, 0] + (self._fov[0] / 2), dtype=np.float64
        )
        ys = np.ascontiguousarray(
            coordinates[:, 1] + (self._fov[1] / 2), dtype=np.float64
        )
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
            self.lam.size,
            kernel_threshold,
            kernel.size,
            nthreads,
        )

        return self.arr * self.units

    def apply_psf(self):
        """Apply a PSF to the data cube.

        This is not yet implemented. Feel free to implement and raise a
        pull request.
        """
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/synthesizer-project/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def apply_noise_array(self):
        """Apply noise to the data cube.

        This is not yet implemented. Feel free to implement and raise a
        pull request.
        """
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/synthesizer-project/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def apply_noise_from_std(self):
        """Apply noise to the data cube.

        This is not yet implemented. Feel free to implement and raise a
        pull request.
        """
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/synthesizer-project/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def apply_noise_from_snr(self):
        """Apply noise to the data cube.

        This is not yet implemented. Feel free to implement and raise a
        pull request.
        """
        raise exceptions.UnimplementedFunctionality(
            "Not yet implemented! Feel free to implement and raise a "
            "pull request. Guidance for contributing can be found at "
            "https://github.com/synthesizer-project/synthesizer/blob/main/"
            "docs/CONTRIBUTING.md"
        )

    def animate_data_cube(
        self,
        show=False,
        save_path=None,
        fps=30,
        vmin=None,
        vmax=None,
    ):
        """Create an animation of the spectral cube.

        Each frame of the animation is a wavelength bin.

        Args:
            show (bool):
                Should the animation be shown?
            save_path (str, optional):
                Path to save the animation. If not specified, the
                animation is not saved.
            fps (int, optional):
                the number of frames per second in the output animation.
                Default is 30 frames per second.
            vmin (float):
                The minimum of the normalisation.
            vmax (float):
                The maximum of the normalisation.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        # Integrate the input Sed
        sed = self.sed.sum()

        # Create the figure and axes
        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0},
            figsize=(6, 8),
        )

        # Get the normalisation
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.percentile(self.arr, 99.9)

        # Define the norm
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

        # Create a placeholder image
        img = ax1.imshow(
            self.arr[:, :, 0],
            origin="lower",
            animated=True,
            norm=norm,
        )
        ax1.axis("off")

        # Second subplot for the spectra
        spectra = getattr(sed, self.quantity)
        if self.quantity in ("lnu", "llam", "luminosity"):
            ax2.semilogy(self.lam, spectra)
        else:
            ax2.semilogy(sed.obslam, spectra)
        (line,) = ax2.plot(
            [self.lam[0], self.lam[0]],
            ax2.get_ylim(),
            color="red",
        )

        # Get units for labels
        x_units = str(self.lam.units)
        y_units = str(spectra.units)
        x_units = x_units.replace("/", r"\ / \ ").replace("*", " ")
        y_units = y_units.replace("/", r"\ / \ ").replace("*", " ")

        # Label the spectra
        ax2.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")

        # Label the y axis handling all possibilities
        if self.quantity == "lnu":
            ax2.set_ylabel(r"$L_{\nu}/[\mathrm{" + y_units + r"}]$")
        elif self.quantity == "llam":
            ax2.set_ylabel(r"$L_{\lambda}/[\mathrm{" + y_units + r"}]$")
        elif self.quantity == "luminosity":
            ax2.set_ylabel(r"$L/[\mathrm{" + y_units + r"}]$")
        elif self.quantity == "fnu":
            ax2.set_ylabel(r"$F_{\nu}/[\mathrm{" + y_units + r"}]$")
        elif self.quantity == "flam":
            ax2.set_ylabel(r"$F_{\lambda}/[\mathrm{" + y_units + r"}]$")
        else:
            ax2.set_ylabel(r"$F/[\mathrm{" + y_units + r"}]$")

        def update(i):
            # Update the image for the ith frame
            img.set_data(self.arr[:, :, i])
            line.set_xdata([self.lam[i], self.lam[i]])
            return [img, line]

        # Calculate interval in milliseconds based on fps
        interval = 1000 / fps

        # Create the animation
        anim = FuncAnimation(
            fig, update, frames=self.lam.size, interval=interval, blit=False
        )

        # Save if a path is provided
        if save_path is not None:
            anim.save(save_path, writer="imagemagick")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return anim
