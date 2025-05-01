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
from unyt import angstrom

from synthesizer import exceptions
from synthesizer.imaging.base_imaging import ImagingBase
from synthesizer.imaging.image_generators import (
    _generate_ifu_parametric_smoothed,
    _generate_ifu_particle_hist,
    _generate_ifu_particle_smoothed,
)
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter


class SpectralCube(ImagingBase):
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

    @accepts(lam=angstrom)
    def __init__(
        self,
        resolution,
        fov,
        lam,
    ):
        """Intialise the SpectralCube.

        Either fov or npix must be given. If both are given, fov is used.

        Args:
            resolution (unyt_quantity, float):
                The spatial resolution of the data cube.
            fov (unyt_array, float/tuple):
                The field of view of the data cube. If a single value is
                given, the FOV is assumed to be square.
            lam (unyt_array, float):
                The wavelengths of the data cube.

        """
        # Instantiate the base class holding the geometry
        ImagingBase.__init__(self, resolution, fov)

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
        # Not applicable when the IFU hasn't been generated yet
        if self.arr is None:
            raise exceptions.MissingIFU(
                "The IFU array hasn't been generated yet. Please call "
                "get_data_cube_hist or get_data_cube_smoothed first."
            )
        return self.arr * self.units

    @property
    def ifu(self):
        """Return the IFU array.

        An alias for the data cube.

        This is a property to allow the IFU array to be accessed as an
        attribute.

        Returns:
            array_like (float):
                A 3D array containing the IFU array. (npix[0], npix[1],
                lam.size)
        """
        # Not applicable when the IFU hasn't been generated yet
        if self.arr is None:
            raise exceptions.MissingIFU(
                "The IFU array hasn't been generated yet. Please call "
                "get_data_cube_hist or get_data_cube_smoothed first."
            )
        return self.arr * self.units

    @property
    def shape(self):
        """Return the shape of the data cube.

        Returns:
            tuple (int):
                The shape of the data cube. (npix[0], npix[1], lam.size)
        """
        # Not applicable when the IFU hasn't been generated yet
        if self.arr is None:
            return ()
        return self.arr.shape

    def __str__(self):
        """Return a string representation of the SpectralCube object.

        Returns:
            table (str)
                A string representation of the SpectralCube object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("SpectralCube")

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
        return _generate_ifu_particle_hist(
            ifu=self,
            sed=sed,
            quantity=quantity,
            cent_coords=coordinates,
            nthreads=nthreads,
        )

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
        # Call the correct generation function (particle or parametric)
        if density_grid is not None and sed is not None:
            return _generate_ifu_parametric_smoothed(
                self,
                sed,
                quantity,
                density_grid,
            )
        elif (
            coordinates is not None
            and smoothing_lengths is not None
            and kernel is not None
            and kernel_threshold is not None
            and sed is not None
        ):
            return _generate_ifu_particle_smoothed(
                ifu=self,
                sed=sed,
                quantity=quantity,
                cent_coords=coordinates,
                smoothing_lengths=smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

        else:
            raise exceptions.InconsistentArguments(
                "Didn't find a valid set of arguments to generate images. "
                "Please provide either a density grid and photometry for "
                f"parametric imaging (found density_grid={type(density_grid)} "
                f"sed={type(sed)}) or coordinates, smoothing "
                f"lengths, kernel, and kernel_threshold for particle imaging "
                f"(found coordinates={type(coordinates)}, "
                f"smoothing_lengths={type(smoothing_lengths)}, "
                f"kernel={type(kernel)}, "
                f"kernel_threshold={type(kernel_threshold)}, "
                f"sed={type(sed)})"
            )

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
