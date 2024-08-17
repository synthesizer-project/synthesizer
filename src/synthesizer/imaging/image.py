"""A module containing the definition of an image.

This module contains the definition of an image.

An image can be generated from particle based data with or without smoothing,
and from parametric data with smoothing defined by a morphology derived
density grid.

Example Usage:
        # Particle based case
        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        img.get_img(signal, coordinates * kpc, smoothing_lengths * kpc, kernel)

        # Parametric case
        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        img.get_img(signal, density_grid)
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from scipy import signal
from scipy.ndimage import zoom
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.units import Quantity


class Image:
    """
    A class for generating images.

    This class is used to generate images from particle based data with or
    without smoothing, and from parametric data with smoothing defined by a
    morphology derived density grid.

    This can be used in isolation to generate singular images or generated by
    an ImageCollection to generate a collection of images in various filters.

    Attributes:
        resolution (unyt_quantity, float):
            The resolution of the image.
        fov (unyt_quantity, float):
            The field of view of the image.
        npix (tuple):
            The number of pixels in the image.
        arr (array_like, float):
            The array containing the image.
        units (unyt.Units):
            The units of the image.
    """

    # Define quantities
    resolution = Quantity()
    fov = Quantity()

    def __init__(
        self,
        resolution,
        fov,
        img=None,
    ):
        """
        Create an image with the images metadata.

        Args:
            resolution (unyt_quantity, float):
                The resolution of the image.
            fov (unyt_quantity, float):
                The field of view of the image. If a single value is passed
                then the FOV is assumed to be square.
            img (unyt_array/array_like, float):
                The image array. Only used to attach an existing image array
                to an image instance. Mostly used internally when methods
                make a new image instance for self.
        """
        # Set the quantities
        self.resolution = resolution
        self.fov = fov

        # If fov isn't a array, make it one
        if self.fov is not None and self.fov.size == 1:
            self.fov = np.array((self.fov, self.fov))

        # Calculate the shape of the image
        self._compute_npix()

        # Attribute to hold the image array itself
        self.arr = None

        # Attributes to hold the image units
        self.units = None

        # Attach an image if handed one
        if img is not None:
            self.arr = img.value if isinstance(img, unyt_array) else img
            self.units = img.units if isinstance(img, unyt_array) else None

        # Set up the noise array and weight map
        self.noise_arr = None
        self.weight_map = None

    def _compute_npix(self):
        """
        Compute the number of pixels in the FOV.

        When resolution and fov are given, the number of pixels is computed
        using this function. This can redefine the fov to ensure the FOV
        is an integer number of pixels.
        """
        # Compute how many pixels fall in the FOV
        self.npix = np.int32(np.ceil(self._fov / self._resolution))

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

    def resample(self, factor):
        """
        Resample the image by factor.

        Args:
            factor (float)
                The factor by which to resample the image, >1 increases
                resolution, <1 decreases resolution.
        """
        # Perform the conversion on the basic image properties
        self.resolution /= factor
        self._compute_npix()

        # Resample the image.
        # NOTE: skimage.transform.pyramid_gaussian is more efficient but adds
        #       another dependency.
        if self.arr is not None:
            self.arr = zoom(self.arr, (factor, factor), order=3)
            new_shape = self.arr.shape
        else:
            raise exceptions.MissingImage(
                "The image array hasn't been generated yet. Please run "
                "get_img_hist() or get_img_smoothed() before resampling."
            )

        # Handle the edge case where the conversion between resolutions has
        # messed with the FOV.
        if self.npix[0] != new_shape[0] or self.npix[1] != new_shape[1]:
            self.npix = new_shape
            self._compute_fov()

    def __add__(self, other_img):
        """
        Add 2 Images together.

        Args:
            other_img (Image)
                The other image to be added.

        Returns:
            Image
                The new image containing the added arrays.

        Raises:
            InconsistentArguments
                If the images have different resolutions or fovs, or if the
                combination of units is incompatible an error is raised.
        """
        # Ensure the images have the same resolution
        if self.resolution != other_img.resolution:
            raise exceptions.InconsistentAddition(
                "The images must have the same resolution to be added."
            )

        # Ensure the images have the same fov
        if np.any(self.fov != other_img.fov):
            raise exceptions.InconsistentAddition(
                "The images must have the same fov to be added."
            )

        # Hanlde if units are involved or not
        if self.units is None and other_img.units is None:
            return Image(
                self.resolution,
                self.fov,
                img=self.arr + other_img.arr,
            )
        elif self.units is not None and other_img.units is not None:
            return Image(
                self.resolution,
                self.fov,
                img=self.arr * self.units + other_img.arr * other_img.units,
            )
        else:
            s = "dimensionless"
            raise exceptions.InconsistentArguments(
                "Cannot add inconsistent units "
                f"({self.units if self.units is not None else s}, "
                f"{other_img.units if other_img.units is not None else s})."
            )

    def __mul__(self, mult):
        """
        Multiply the image by a multiplier.

        Args:
            mult (int/float/array-like)
                The number to multiply the image array by.

        Returns:
            Image
                The new image containing the multipled array.
        """
        # Create the new image
        new_img = Image(self.resolution, self.fov)

        # Associate the image array and units
        new_img.arr = self.arr
        new_img.units = self.units

        # Multiply the image array
        new_img.arr *= mult
        return new_img

    def get_img_hist(
        self,
        signal,
        coordinates=None,
    ):
        """
        Calculate an image with no smoothing.

        This is only applicable to particle based images and is just a
        wrapper for numpy.histogram2d.

        Args:
            signal (array_like, float):
                The signal to be sorted into the image.
            coordinates (unyt_array, float):
                The coordinates of the particles.

        Returns:
            img (array_like, float)
                A 2D array containing the pixel values sorted into the image.
                (npix, npix)
        """
        # Strip off and store the units on the signal if they are present
        if isinstance(signal, (unyt_quantity, unyt_array)):
            self.units = signal.units
            signal = signal.value

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value

        # In case coordinates haven't been centered we need to centre them
        if not (coordinates.min() < 0 and coordinates.max() > 0):
            coordinates -= np.average(coordinates, axis=0, weights=signal)

        self.arr = np.histogram2d(
            coordinates[:, 0],
            coordinates[:, 1],
            bins=(
                np.linspace(
                    -self._fov[0] / 2, self._fov[0] / 2, self.npix[0] + 1
                ),
                np.linspace(
                    -self._fov[1] / 2, self._fov[1] / 2, self.npix[1] + 1
                ),
            ),
            weights=signal,
        )[0]

        return self.arr * self.units if self.units is not None else self.arr

    def get_img_smoothed(
        self,
        signal,
        coordinates=None,
        smoothing_lengths=None,
        kernel=None,
        kernel_threshold=1,
        density_grid=None,
        nthreads=1,
    ):
        """
        Calculate a smoothed image.

        In the particle case this smooths each particle's signal over the SPH
        kernel defined by their smoothing length. This uses C extensions to
        calculate the image for each particle efficiently.

        In the parametric case the signal is smoothed over a density grid. This
        density grid is an array defining the weight in each pixel.

        Args:
            signal (array_like, float):
                The signal to be sorted into the image.
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
            nthreads (int):
                The number of threads to use for the C extension. (particle
                case only)

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
            self.arr = density_grid[:, :] * signal

            return (
                self.arr * self.units if self.units is not None else self.arr
            )

        from .extensions.image import make_img

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to(self.resolution.units).value
        smoothing_lengths = smoothing_lengths.to(self.resolution.units).value

        # In case coordinates haven't been centered we need to centre them
        if not (coordinates.min() < 0 and coordinates.max() > 0):
            coordinates -= np.average(coordinates, axis=0, weights=signal)

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        signal = np.ascontiguousarray(signal, dtype=np.float64)
        smls = np.ascontiguousarray(smoothing_lengths, dtype=np.float64)
        xs = np.ascontiguousarray(
            coordinates[:, 0] + (self._fov[0] / 2), dtype=np.float64
        )
        ys = np.ascontiguousarray(
            coordinates[:, 1] + (self._fov[1] / 2), dtype=np.float64
        )

        self.arr = make_img(
            signal,
            smls,
            xs,
            ys,
            kernel,
            self._resolution,
            self.npix[0],
            self.npix[1],
            coordinates.shape[0],
            kernel_threshold,
            kernel.size,
            nthreads,
        )

        return self.arr * self.units if self.units is not None else self.arr

    def apply_psf(self, psf):
        """
        Apply a Point Spread Function to this image.

        Args:
            psf (np.ndarray)
                An array describing the point spread function.

        Returns:
            Image
                The image convolved with the psf.
        """
        # Perform the convolution
        convolved_img = signal.fftconvolve(self.arr, psf, mode="same")

        # Include units if we have them
        if self.units is not None:
            convolved_img *= self.units

        return Image(
            resolution=self.resolution,
            fov=self.fov,
            img=convolved_img,
        )

    def apply_noise_array(self, noise_arr):
        """
        Apply a noise array.

        Args:
            noise_arr (np.ndarray)
                The noise array to add to the image.

        Returns:
            Image
                The image including the noise array
            np.ndarray
                The weight map, derived from 1 / std^2
        """
        # Add the noise array to the image
        if self.units is not None:
            noisy_img = self.arr * self.units + noise_arr
        else:
            noisy_img = self.arr + noise_arr

        # Make the new image
        new_img = Image(
            resolution=self.resolution,
            fov=self.fov,
            img=noisy_img,
        )

        # Store the noise array on the new image
        new_img.noise_arr = noise_arr

        # Calculate and store the weight map (if called from
        # apply_noise_from_std this will be overwritten with the
        # true standard deviation)
        new_img.weight_map = 1 / np.std(noise_arr) ** 2

        return new_img

    def apply_noise_from_std(self, noise_std):
        """
        Apply noise derived from a standard deviation.

        This creates noise with a normal distribution centred on 0 with the
        passed standard deviation.

        Args:
            noise_std (float)
                The standard deviation of the noise to add to the image.

        Returns:
            Image
                The image including the noise array
            np.ndarray
                The weight map.
        """
        # Strip off units if necessary
        if isinstance(noise_std, unyt_quantity):
            units = noise_std.units
            noise_std = noise_std.value
        else:
            units = None

        # Get the noise array
        noise_arr = np.random.normal(
            loc=0,
            scale=noise_std,
            size=self.npix,
        )

        # Reapply units if we have them
        if units is not None:
            noise_arr *= units

        # Add the noise to the image
        new_img = self.apply_noise_array(noise_arr)

        # Calculate the weight map
        new_img.weight_map = 1 / noise_std**2

        return new_img

    def apply_noise_from_snr(self, snr, depth, aperture_radius=None):
        """
        Apply noise derived from a SNR and depth.

        This can either be for a point source or an aperture if aperture_radius
        is passed.

        This assumes the SNR is defined as SNR = S / sqrt(noise_std)

        Args:

        Returns:
            Image
                The image including the noise array
            np.ndarray
                The noise array.
            np.ndarray
                The weight map.
        """
        # Convert aperture radius to consistent units if we have it
        if aperture_radius is not None:
            aperture_radius = aperture_radius.to(self.resolution.units).value

        # Ensure we have units if we need them
        if self.units is not None and not isinstance(depth, unyt_quantity):
            raise exceptions.InconsistentArguments(
                "If the Image has units then the depth must also be passed "
                f"with units. (image.units = {self.units})"
            )

        # Strip off units from the depth if we have them
        if isinstance(depth, unyt_quantity):
            units = depth.units
            depth = depth.value
        else:
            units = None

        # Calculate the noise array from an aperture or point source
        if aperture_radius is not None:
            # Calculate the total noise in the aperture
            # NOTE: this assumes SNR = S / app_noise
            app_noise = depth / snr

            # Calculate the aperture area in image coordinates
            app_area_coordinates = np.pi * aperture_radius**2

            # Convert the aperture area to per pixel
            app_area_pix = app_area_coordinates / self._resolution**2

            # Get the noise per pixel
            noise_std = app_noise / app_area_pix

        # Calculate the noise from the depth and snr for a point source.
        else:
            # Calculate noise in a pixel
            noise_std = depth / snr

        # Reapply units if we have them
        if units is not None:
            noise_std *= units

        return self.apply_noise_from_std(noise_std)

    def plot_img(
        self,
        show=False,
        cmap="Greys_r",
        norm=None,
        fig=None,
        ax=None,
    ):
        """
        Plot an image.

        Args:
            show (bool)
                Whether to show the plot or not (Default False).
            cmap (str)
                The name of the matplotlib colormap for image plotting. Can be
                any valid string that can be passed to the cmap argument of
                imshow. Defaults to "Greys_r".
            norm (function)
                A normalisation function. This can be custom made or one of
                matplotlib's normalisation functions. It must take an array and
                return the same array after normalisation.
            tick_formatter (matplotlib.ticker.FuncFormatter)
                An instance of the tick formatter for formatting the colorbar
                ticks.
            fig (matplotlib.pyplot.figure)
                The figure object to plot on. If None a new figure is created.
            ax (matplotlib.pyplot.figure.axis)
                The axis object to plot on. If None a new axis is created.

        Returns:
            matplotlib.pyplot.figure
                The figure object containing the plot
            matplotlib.pyplot.figure.axis
                The axis object containing the image.
        """
        # Get the image
        img = self.arr

        # Set up the figure
        if fig is None:
            fig = plt.figure(figsize=(3.5, 3.5))

        # Create the axis and turn off the ticks and frame
        if ax is None:
            ax = fig.add_subplot(111)
            ax.axis("off")

        # Plot the image and remove the surrounding axis
        ax.imshow(
            img,
            cmap=cmap,
            norm=norm,
        )

        if show:
            plt.show()

        return fig, ax

    def plot_map(
        self,
        show=False,
        extent=None,
        cmap="Greys_r",
        cbar_label=None,
        norm=None,
        tick_formatter=None,
        fig=None,
        ax=None,
    ):
        """
        Plot a map.

        Unlike an image we want a colorbar and axes for a map.

        Args:
            show (bool)
                Whether to show the plot or not (Default False).
            extent (array_like)
                The extent of the x and y axes.
            cmap (str)
                The name of the matplotlib colormap for image plotting. Can be
                any valid string that can be passed to the cmap argument of
                imshow. Defaults to "Greys_r".
            cbar_label (str)
                The label for the colorbar.
            norm (function)
                A normalisation function. This can be custom made or one of
                matplotlib's normalisation functions. It must take an array and
                return the same array after normalisation.
            tick_formatter (matplotlib.ticker.FuncFormatter)
                An instance of the tick formatter for formatting the colorbar
                ticks.
            fig (matplotlib.pyplot.figure)
                The figure object to plot on. If None a new figure is created.
            ax (matplotlib.pyplot.figure.axis)
                The axis object to plot on. If None a new axis is created.

        Returns:
            matplotlib.pyplot.figure
                The figure object containing the plot
            matplotlib.pyplot.figure.axis
                The axis object containing the image.
        """
        # Get the image
        img = self.arr

        # Set up the figure
        if fig is None:
            fig = plt.figure(figsize=(3.5, 3.5))

        # Create the axis
        if ax is None:
            ax = fig.add_subplot(111)

        # Plot the image and remove the surrounding axis
        im = ax.imshow(
            img,
            extent=extent,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
        )

        # Make the colorbar with the format if provided
        cbar = fig.colorbar(im, format=tick_formatter)
        if cbar_label is not None:
            cbar.set_label(cbar_label)

        if show:
            plt.show()

        return fig, ax

    def print_ascii(self):
        """Print an ASCII representation of an image."""
        # Define the possible ASCII symbols in density order
        scale = (
            "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft|()1{}[]?-_+~<>"
            "i!lI;:,\"^`'. "[::-1]
        )

        # Define the number of symbols
        nscale = len(scale)

        # Map the image onto a range of 0 -> nscale - 1
        img = (nscale - 1) * self.arr / np.max(self.arr)

        # Convert to integers for indexing
        img = img.astype(int)

        # Create the ASCII string image
        ascii_img = ""
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                ascii_img += 2 * scale[img[i, j]]
            ascii_img += "\n"

        print(ascii_img)

    def plot_unknown_pleasures(
        self,
        constrast=100,
        target_lines=50,
        figsize=(8, 8),
        title="SYNTHESIZER",
    ):
        """
        Create a representation of an image similar in style to Joy Division's
        seminal 1979 album Unknown Pleasures.

        Borrows some code from this matplotlib examples:
        https://matplotlib.org/stable/gallery/animation/unchained.html

        Arguments
            constrast (float)
                The contrast.
            target_lines (int)
                The target number of individual lines to use.
        """

        # extract data
        data = 1 * self.arr

        # normalise to the maximum
        data /= np.max(data)

        # log10
        data = np.log10(data)

        # set any -np.inf values to zero (once renormalised)
        data[data == -np.inf] = -np.log10(constrast)

        # define normalising function
        norm = Normalize(vmin=-np.log10(constrast), vmax=0.0)

        # normalise data
        data = norm(data) * 5

        # set any data <0.0 to zero
        data[data < 0.0] = 0.0

        # Unknown Pleasures works best with about 50 lines so reshape the data
        # to have approximately 50 lines.

        # Calcualate the eventual number of lines.
        nlines = int(data.shape[0] / (data.shape[0] // target_lines))

        # Reshape data to keep the x-axis resolution but reduced number of
        # lines.
        new_shape = nlines, data.shape[0] // nlines, data.shape[1], 1
        data = data.reshape(new_shape).mean(-1).mean(1)

        # Create new Figure with black background
        fig = plt.figure(figsize=figsize, facecolor="black")

        # Add a subplot with no frame
        ax = plt.subplot(111, frameon=False)

        X = np.linspace(-1, 1, data.shape[-1])

        # Generate line plots
        lines = []
        for i in range(data.shape[0]):
            # Small reduction of the X extents to get a cheap perspective
            # effect.
            xscale = 1 - i / 100.0
            # Same for linewidth (thicker strokes on bottom)
            lw = 1.5 - i / 100.0

            (line,) = ax.plot(xscale * X, i + data[i], color="w", lw=lw)
            lines.append(line)

        # Set y limit (or first line is cropped because of thickness)
        ax.set_ylim(-20, nlines + 20)

        # No ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add title
        if title:
            ax.text(
                0.5,
                0.8,
                title,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                color="w",
                family="sans-serif",
                fontweight="light",
                fontsize=16,
            )

        return fig, ax

    def get_signal_in_aperture(
        self,
        aperture_radius,
        aperture_cent=None,
        nthreads=1,
    ):
        """
        Return the sum of the image within an aperture.

        This uses fractional pixel coverage to calculate the overlap between
        the aperture and each pixel.

        Args:
            aperture_radius (unyt_quantity, float)
                The radius of the aperture.
            aperture_cent (unyt_array, float)
                The centre of the aperture (in pixel coordinates, i.e. the
                centre is [npix/2, npix/2], top left is [0, 0], and bottom
                right is [npix, npix]). If None then the centre is assumed to
                be the maximum pixel.
            nthreads (int)
                The number of threads to use for the calculation. Default is 1.

        Returns:
            float
                The sum of the image within the aperture.
        """
        # Convert the aperture radius to the correct units
        aperture_radius = aperture_radius.to(self.resolution.units).value

        # If the aperture centre isn't passed, assume it's the maximum
        # pixel
        if aperture_cent is None:
            max_pixel = np.unravel_index(self.arr.argmax(), self.arr.shape)
            aperture_cent = np.array(max_pixel) + 0.5

        from synthesizer.imaging.extensions.circular_aperture import (
            calculate_circular_overlap,
        )

        return (
            calculate_circular_overlap(
                self._resolution,
                self.npix[0],
                self.npix[1],
                np.float64(aperture_radius),
                self.arr,
                np.float64(aperture_cent),
                nthreads,
            )
            * self.units
        )
