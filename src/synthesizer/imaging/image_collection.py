"""Defintions for collections of generic images.

This module contains the definition for a generic ImageCollection class. This
provides the common functionality between particle and parametric imaging. The
user should not use this class directly, but rather use the
particle.imaging.Images and parametric.imaging.Images classes.

Example usage:
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from synthesizer import exceptions
from synthesizer.imaging.image import Image
from syntheiszer.units import Quantity


class ImageCollection:
    """
    A collection of Image objects.

    This contains all the generic methods for creating and manipulating
    images. Any particle or parametric functionality is defined in the
    particle and parametric mdoules respectively.
    """

    # Define quantities
    resolution = Quantity()
    fov = Quantity()
    orig_resolution = Quantity()

    def __init__(self, resolution, fov=None, npix=None):
        """Initialize the image collection.

        Either fov or npix must be specified.

        Args:
            resolution (unyt_quantity)
                The size of a pixel.
            fov (unyt_quantity/tuple, unyt_quantity)
                The width of the image.
            npix (int/tuple, int)
                The number of pixels in the image.
        """
        # Check the arguments
        self._check_args(resolution, fov, npix)

        # Attach resolution, fov, and npix
        self.resolution = resolution
        self.fov = fov
        self.npix = npix

        # If fov isn't a tuple, make it one
        if fov is not None and not isinstance(fov, tuple):
            self.fov = (fov, fov)

        # If npix isn't a tuple, make it one
        if npix is not None and not isinstance(npix, tuple):
            self.npix = (npix, npix)

        # Handle the different input cases
        if npix is None:
            self._compute_npix()
        else:
            self._compute_fov()

        # Keep track of the input resolution and and npix so we can handle
        # super resolution correctly.
        self.orig_resolution = resolution
        self.orig_npix = npix

        # Container for images (populated when image creation methods are
        # called)
        self.imgs = {}

    def _check_args(self, resolution, fov, npix):
        """
        Ensure we have a valid combination of inputs.

        Args:
            resolution (unyt_quantity)
                The size of a pixel.
            fov (unyt_quantity)
                The width of the image.
            npix (int)
                The number of pixels in the image.

        Raises:
            InconsistentArguments
               Errors when an incorrect combination of arguments is passed.
        """
        # Missing units on resolution
        if isinstance(resolution, float):
            raise exceptions.InconsistentArguments(
                "Resolution is missing units! Please include unyt unit "
                "information (e.g. resolution * arcsec or resolution * kpc)"
            )

        # Missing image size
        if fov is None and npix is None:
            raise exceptions.InconsistentArguments(
                "Either fov or npix must be specified!"
            )

    def _compute_npix(self):
        """
        Compute the number of pixels in the FOV.

        When resolution and fov are given, the number of pixels is computed
        using this function. This can redefine the fov to ensure the FOV
        is an integer number of pixels.
        """
        # Compute how many pixels fall in the FOV
        self.npix = int(math.ceil(self._fov / self._resolution))
        if self.orig_npix is None:
            self.orig_npix = int(math.ceil(self._fov / self._resolution))

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

    def downsample(self, factor):
        """
        Supersamples all images contained within this instance.

        Useful when applying a PSF to get more accurate convolution results.

        NOTE: It is more robust to create the initial "standard" image at high
        resolution and then downsample it after done with the high resolution
        version.

        Args:
            factor (float)
                The factor by which to resample the image, >1 increases
                resolution, <1 decreases resolution.

        Raises:
            ValueError
                If the incorrect resample function is called an error is raised
                to ensure the user does not erroneously resample.
        """
        # Check factor (NOTE: this doesn't actually cause an issue
        # mechanically but will ensure users are literal about resampling and
        # can't mistakenly resample in unintended ways).
        if factor > 1:
            raise ValueError("Using downsample method to supersample!")

        # Resample each image
        for f in self.imgs:
            self.imgs[f].resample(factor)

    def supersample(self, factor):
        """
        Supersample all images contained within this instance.

        Useful when applying a PSF to get more accurate convolution results.

        NOTE: It is more robust to create the initial "standard" image at high
        resolution and then downsample it after done with the high resolution
        version.

        Args:
            factor (float)
                The factor by which to resample the image, >1 increases
                resolution, <1 decreases resolution.

        Raises:
            ValueError
                If the incorrect resample function is called an error is raised
                to ensure the user does not erroneously resample.
        """
        # Check factor (NOTE: this doesn't actually cause an issue
        # mechanically but will ensure users are literal about resampling and
        # can't mistakenly resample in unintended ways).
        if factor < 1:
            raise ValueError("Using supersample method to downsample!")

        # Resample each image
        for f in self.imgs:
            self.imgs[f].resample(factor)

    def __add__(self, other_img):
        """
        Add two img objects together.

        This combines all images with a common key.

        The resulting image object inherits its attributes from self, i.e in
        img = img1 + img2, img will inherit the attributes of img1.

        Args:
            other_img (ImageCollection)
                The other image collection to be combined with self.

        Returns:
            composite_img (ImageCollection)
                A new Image object contain the composite image of self and
                other_img.

        Raises:
            InconsistentAddition
                If the ImageCollections can't be added and error is thrown.
        """
        # Make sure the images are compatible dimensions
        if (
            self.resolution != other_img.resolution
            or self.fov != other_img.fov
            or self.npix != other_img.npix
        ):
            raise exceptions.InconsistentAddition(
                f"Cannot add Images: resolution=({str(self.resolution)} + "
                f"{str(other_img.resolution)}), fov=({str(self.fov)} + "
                f"{str(other_img.fov)}), npix=({str(self.npix)} + "
                f"{str(other_img.npix)})"
            )

        # Initialise the composite image with the right type
        composite_img = ImageCollection(
            resolution=self.resolution,
            npix=None,
            fov=self.fov,
        )

        # Get common filters
        filters = set(list(self.imgs.keys())).intersection(
            set(list(other_img.imgs.keys()))
        )

        # Combine any common filters
        for f in filters:
            composite_img.imgs[f] = self.imgs[f] + other_img.imgs[f]

        return composite_img

    def get_imgs_hist(self, photometry, coordinates):
        """
        Calculate an image with no smoothing.

        Only applicable to particle based imaging.

        Args:
            photometry (PhotometryCollection)
                A dictionary of photometry for each filter.
            coordinates (unyt_array, float)
                The coordinates of the particles.
        """
        # Need to loop over filters, calculate photometry, and
        # return a dictionary of images
        for f in photometry.filters:
            # Create an Image object for this filter
            img = Image(self.resolution, self.fov)

            # Get the image for this filter
            img.get_img_hist(photometry[f])

            # Store the image
            self.imgs[f] = img

    def get_imgs_smoothed(
        self,
        photometry,
        coordinates=None,
        smoothing_lengths=None,
        kernel=None,
        density_grid=None,
    ):
        """
        Calculate an images from a smoothed distribution.

        In the particle case this smooths each particle's signal over the SPH
        kernel defined by their smoothing length. This uses C extensions to
        calculate the image for each particle efficiently.

        In the parametric case the signal is smoothed over a density grid. This
        density grid is an array defining the weight in each pixel.

        Args:


        Returns:
            img/imgs (array_like/dictionary, float)
                If pixel_values is provided: A 2D array containing particles
                smoothed and sorted into an image. (npix, npix)
                If a filter list is provided: A dictionary containing 2D array
                with particles smoothed and sorted into the image.
                (npix, npix)
        """
        # Loop over filters in the photometry making an image for each.
        for f in photometry.filters:
            # Create an Image object for this filter
            img = Image(self.resolution, self.fov)

            # Get the image for this filter
            img.get_img_hist(photometry[f])

            # Store the image
            self.imgs[f] = img

            # Get and store the image for this filter
            self.imgs[f] = self.get_img_smoothed(
                signal=photometry[f],
                coordinates=coordinates,
                smoothing_lengths=smoothing_lengths,
                kernel=kernel,
                density_grid=density_grid,
            )

    def apply_psf(self, psfs):
        """
        Convolve this ImageCollection's images with their PSFs.

        This function will handle the different cases for image creation. If
        there are multiple filters it will use the psf for each filters,
        unless a single psf is provided in which case each filter will be
        convolved with the singular psf. If the Image only contains a single
        image it will convolve the psf with that image.

        To more accurately apply the PSF we recommend using a super resolution
        image. This can be done via the supersample method and then
        downsampling to the native pixel scale after resampling. However, it
        is more efficient and robust to start at the super resolution initially
        and then downsample after the fact.

        Returns:
            img/imgs (array_like/dictionary, float)
                If pixel_values exists: A singular image convolved with the
                PSF. If a filter list exists: Each img in self.imgs is
                returned convolved with the corresponding PSF (or the single
                PSF if an array was supplied for psf).

        Raises:
            InconsistentArguments
                If a dictionary of PSFs is provided that doesn't match the
                filters an error is raised.
        """
        # Check we have a valid set of PSFs
        if len(self.filters) == 0 and isinstance(psfs, dict):
            raise exceptions.InconsistentArguments(
                "To convolve with a single image an array should be "
                "provided for the PSF not a dictionary."
            )
        elif len(self.filters) > 0 and isinstance(psfs, dict):
            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in psfs:
                filter_codes -= set(
                    [
                        key,
                    ]
                )

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single PSF or a dictionary with a PSF for each "
                    "filter must be given. PSFs are missing for filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        # Handle the possible cases (multiple filters or single image)
        if len(self.filters) == 0:
            self.img_psf = self._get_psfed_single_img(self.img, psfs)

            return self.img_psf

        # Otherwise, we need to loop over filters and return a dictionary of
        # convolved images.
        for f in self.filters:
            # Get the PSF
            if isinstance(psfs, dict):
                psf = psfs[f.filter_code]
            else:
                psf = psfs

            # Apply the PSF to this image
            self.imgs_psf[f.filter_code] = self._get_psfed_single_img(
                self.imgs[f.filter_code], psf
            )

        return self.imgs_psf

    def apply_noise(self, noises=None):
        """
        Make and add a noise array to each image in this Image object. The
        noise is defined by either a depth and signal-to-noise in an aperture
        or by an explicit noise pixel value.

        Note that the noise will be applied to the psfed images by default
        if they exist (those stored in self.imgs_psf). If those images do not
        exist then it will be applied to the standard images in self.imgs.

        Args:
            noises (float/dict, float)
                The standard deviation of the noise distribution. If noises is
                provided then depth, snr and aperture are ignored. Can either
                be a single value or a value per filter in a dictionary.
        Returns:
            noisy_img (array_like, float)
                The image with a noise contribution.

        Raises:
            InconsistentArguments
                If dictionaries are provided and each filter doesn't have an
                entry and error is thrown.
        """

        # Check we have a valid set of noise attributes
        if len(self.filters) == 0 and (
            isinstance(self.depths, dict)
            or isinstance(self.snrs, dict)
            or isinstance(self.apertures, dict)
            or isinstance(noises, dict)
        ):
            raise exceptions.InconsistentArguments(
                "If there is a single image then noise arguments should be "
                "floats not dictionaries."
            )
        if len(self.filters) > 0 and isinstance(self.depths, dict):
            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in self.depths:
                filter_codes -= set(
                    [
                        key,
                    ]
                )

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single depth or a dictionary of depths for each "
                    "filter must be given. Depths are missing for filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        if len(self.filters) > 0 and isinstance(self.snrs, dict):
            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in self.snrs:
                filter_codes -= set(
                    [
                        key,
                    ]
                )

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single SNR or a dictionary of SNRs for each "
                    "filter must be given. SNRs are missing for filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        if len(self.filters) > 0 and isinstance(self.apertures, dict):
            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in self.apertures:
                filter_codes -= set(
                    [
                        key,
                    ]
                )

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single aperture or a dictionary of apertures for"
                    " each filter must be given. Apertures are missing for "
                    "filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        if len(self.filters) > 0 and isinstance(noises, dict):
            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in noises:
                filter_codes -= set(
                    [
                        key,
                    ]
                )

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single noise or a dictionary of noises for each "
                    "filter must be given. Noises are missing for filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        # Handle the possible cases (multiple filters or single image)
        if len(self.filters) == 0:
            # Apply noise to the image
            noise_tuple = self._get_noisy_single_img(
                self.img_psf, self.depths, self.snrs, self.apertures, noises
            )

            self.img_noise, self.weight_map, self.noise_arr = noise_tuple

            return self.img_noise, self.weight_map, self.noise_arr

        # Otherwise, we need to loop over filters and return a dictionary of
        # convolved images.
        for f in self.filters:
            # Extract the arguments
            if isinstance(self.depths, dict):
                depth = self.depths[f.filter_code]
            else:
                depth = self.depths
            if isinstance(self.snrs, dict):
                snr = self.snrs[f.filter_code]
            else:
                snr = self.snrs
            if isinstance(self.apertures, dict):
                aperture = self.apertures[f.filter_code]
            else:
                aperture = self.apertures
            if isinstance(noises, dict):
                noise = noises[f.filter_code]
            else:
                noise = noises

            # Calculate and apply noise to this image
            if len(self.imgs_psf) > 0:
                noise_tuple = self._get_noisy_single_img(
                    self.imgs_psf[f.filter_code], depth, snr, aperture, noise
                )
            else:
                noise_tuple = self._get_noisy_single_img(
                    self.imgs[f.filter_code], depth, snr, aperture, noise
                )

            # Store the resulting noisy image, weight, and noise arrays
            self.imgs_noise[f.filter_code] = noise_tuple[0]
            self.weight_maps[f.filter_code] = noise_tuple[1]
            self.noise_arrs[f.filter_code] = noise_tuple[2]

        return self.imgs_noise, self.weight_maps, self.noise_arrs

    def plot_images(
        self,
        img_type="standard",
        filter_code=None,
        show=False,
        vmin=None,
        vmax=None,
        scaling_func=None,
        cmap="Greys_r",
    ):
        """
        Plot an image.

        If this image object contains multiple filters each with an image and
        the filter_code argument is not specified, then all images will be
        plotted in a grid of images. If only a single image exists within the
        image object or a filter has been specified via the filter_code
        argument, then only a single image will be plotted.

        Note: When plotting images in multiple filters, if normalisation
        (vmin, vmax) are not provided then the normalisation will be unique
        to each filter. If they are provided then then they will be global
        across all filters.

        Args:
            img_type (str)
                The type of images to combine. Can be "standard" for noiseless
                and psfless images (self.imgs), "psf" for images with psf
                (self.imgs_psf), or "noise" for images with noise
                (self.imgs_noise).
            filter_code (str)
                The filter code of the image to be plotted. If provided a plot
                is made only for this filter. This is not needed if the image
                object only contains a single image.
            show (bool)
                Whether to show the plot or not (Default False).
            vmin (float)
                The minimum value of the normalisation range.
            vmax (float)
                The maximum value of the normalisation range.
            scaling_func (function)
                A function to scale the image by. This function should take a
                single array and produce an array of the same shape but scaled
                in the desired manner.
            cmap (str)
                The name of the matplotlib colormap for image plotting. Can be
                any valid string that can be passed to the cmap argument of
                imshow. Defaults to "Greys_r".

        Returns:
            matplotlib.pyplot.figure
                The figure object containing the plot
            matplotlib.pyplot.figure.axis
                The axis object containing the image.

        Raises:
            UnknownImageType
                If the requested image type has not yet been created and
                stored in this image object an exception is raised.
        """

        # Handle the scaling function for less branches
        if scaling_func is None:

            def scaling_func(x):
                return x

        # What type of image are we plotting?
        if img_type == "standard":
            img = self.img
            imgs = self.imgs
        elif img_type == "psf":
            img = self.img_psf
            imgs = self.imgs_psf
        elif img_type == "noise":
            img = self.img_noise
            imgs = self.imgs_noise
        else:
            raise exceptions.UnknownImageType(
                "img_type can be 'standard', 'psf', or 'noise' "
                "not '%s'" % img_type
            )

        # Are we only plotting a single image from a set?
        if filter_code is not None:
            # Get that image
            img = imgs[filter_code]

        # Plot the single image
        if img is not None:
            # Set up the figure
            fig = plt.figure(figsize=(3.5, 3.5))

            # Create the axis
            ax = fig.add_subplot(111)

            # Set up minima and maxima
            if vmin is None:
                vmin = np.min(img)
            if vmax is None:
                vmax = np.max(img)

            # Normalise the image.
            img = (img - vmin) / (vmax - vmin)

            # Scale the image
            img = scaling_func(img)

            # Plot the image and remove the surrounding axis
            ax.imshow(img, origin="lower", interpolation="nearest", cmap=cmap)
            ax.axis("off")

        else:
            # Ok, plot a grid of filter images

            # Do we need to find the normalisation for each filter?
            unique_norm_min = vmin is None
            unique_norm_max = vmax is None

            # Set up the figure
            fig = plt.figure(
                figsize=(4 * 3.5, int(np.ceil(len(self.filters) / 4)) * 3.5)
            )

            # Create a gridspec grid
            gs = gridspec.GridSpec(
                int(np.ceil(len(self.filters) / 4)), 4, hspace=0.0, wspace=0.0
            )

            # Loop over filters making each image
            for ind, f in enumerate(self.filters):
                # Get the image
                img = imgs[f.filter_code]

                # Create the axis
                ax = fig.add_subplot(gs[int(np.floor(ind / 4)), ind % 4])

                # Set up minima and maxima
                if unique_norm_min:
                    vmin = np.min(img)
                if unique_norm_max:
                    vmax = np.max(img)

                # Normalise the image.
                img = (img - vmin) / (vmax - vmin)

                # Scale the image
                img = scaling_func(img)

                # Plot the image and remove the surrounding axis
                ax.imshow(
                    img, origin="lower", interpolation="nearest", cmap=cmap
                )
                ax.axis("off")

                # Place a label for which filter this ised_ASCII
                ax.text(
                    0.95,
                    0.9,
                    f.filter_code,
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc="w",
                        ec="k",
                        lw=1,
                        alpha=0.8,
                    ),
                    transform=ax.transAxes,
                    horizontalalignment="right",
                )

        if show:
            plt.show()

        return fig, ax

    def make_rgb_image(
        self, rgb_filters, img_type="standard", weights=None, scaling_func=None
    ):
        """
        Makes an rgb image of specified filters with optional weights in
        each filter.

        Args:
            rgb_filters (dict, array_like, str)
                A dictionary containing lists of each filter to combine to
                create the red, green, and blue channels.
                e.g. {"R": "Webb/NIRCam.F277W",
                "G": "Webb/NIRCam.F150W", "B": "Webb/NIRCam.F090W"}.
            img_type (str)
                The type of images to combine. Can be "standard" for noiseless
                and psfless images (self.imgs), "psf" for images with psf
                (self.imgs_psf), or "noise" for images with noise
                (self.imgs_noise).
            weights (dict, array_like, float)
                A dictionary of weights for each filter. Defaults to equal
                weights.
            scaling_func (function)
                A function to scale the image by. Defaults to arcsinh. This
                function should take a single array and produce an array of the
                same shape but scaled in the desired manner.

        Returns:
            array_like (float)
                The image array itself.
        """

        # Handle the scaling function for less branches
        if scaling_func is None:

            def scaling_func(x):
                return x

        # Handle the case where we haven't been passed weights
        if weights is None:
            weights = {}
            for rgb in rgb_filters:
                for f in rgb_filters[rgb]:
                    weights[f] = 1.0

        # Ensure weights sum to 1.0
        for rgb in rgb_filters:
            w_sum = 0
            for f in rgb_filters[rgb]:
                w_sum += weights[f]
            for f in rgb_filters[rgb]:
                weights[f] /= w_sum

        # Set up the rgb image
        rgb_img = np.zeros((self.npix, self.npix, 3), dtype=np.float64)

        for rgb_ind, rgb in enumerate(rgb_filters):
            for f in rgb_filters[rgb]:
                if img_type == "standard":
                    rgb_img[:, :, rgb_ind] += scaling_func(
                        weights[f] * self.imgs[f]
                    )
                elif img_type == "psf":
                    rgb_img[:, :, rgb_ind] += scaling_func(
                        weights[f] * self.imgs_psf[f]
                    )
                elif img_type == "noise":
                    rgb_img[:, :, rgb_ind] += scaling_func(
                        weights[f] * self.imgs_noise[f]
                    )
                else:
                    raise exceptions.UnknownImageType(
                        "img_type can be 'standard', 'psf', or 'noise' "
                        "not '%s'" % img_type
                    )

        self.rgb_img = rgb_img

        return rgb_img

    def plot_rgb_image(self, show=False, vmin=None, vmax=None):
        """
        Plot an RGB image.

        Args:
            show (bool)
                Whether to show the plot or not (Default False).
            vmin (float)
                The minimum value of the normalisation range.
            vmax (float)
                The maximum value of the normalisation range.

        Returns:
            matplotlib.pyplot.figure
                The figure object containing the plot
            matplotlib.pyplot.figure.axis
                The axis object containing the image.
            array_like (float)
                The rgb image array itself.

        Raises:
            MissingImage
                If the RGB image has not yet been created and stored in this
                image object an exception is raised.
        """

        # If the image hasn't been made throw an error
        if self.rgb_img is None:
            raise exceptions.MissingImage(
                "The rgb image hasn't been computed yet. Run "
                "Image.make_rgb_image to compute the RGB image before "
                "plotting."
            )

        # Set up minima and maxima
        if vmin is None:
            vmin = np.min(self.rgb_img)
        if vmax is None:
            vmax = np.max(self.rgb_img)

        # Normalise the image.
        rgb_img = (self.rgb_img - vmin) / (vmax - vmin)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(rgb_img, origin="lower", interpolation="nearest")
        ax.axis("off")

        if show:
            plt.show()

        return fig, ax, rgb_img

    def print_ascii(self, filter_code=None, img_type="standard"):
        """
        Print an ASCII representation of an image.

        Args:
        img_type : str
            The type of images to combine. Can be "standard" for noiseless
            and psfless images (self.imgs), "psf" for images with psf
            (self.imgs_psf), or "noise" for images with noise
            (self.imgs_noise).
        filter_code : str
            The filter code of the image to be plotted. If provided a plot is
            made only for this filter. This is not needed if the image object
            only contains a single image.
        """
        # Define the possible ASCII symbols in density order
        scale = (
            "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft|()1{}[]?-_+~<>"
            "i!lI;:,\"^`'. "[::-1]
        )

        # Define the number of symbols
        nscale = len(scale)

        # If a filter code has been provided extract that image, otherwise use
        # the standalone image
        if filter_code:
            img = self.imgs[filter_code]
        else:
            if self.img is None:
                raise exceptions.InconsistentArguments(
                    "A filter code needs to be supplied"
                )
            img = self.img

        # Map the image onto a range of 0 -> nscale - 1
        img = (nscale - 1) * img / np.max(img)

        # Convert to integers for indexing
        img = img.astype(int)

        # Create the ASCII string image
        ascii_img = ""
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                ascii_img += 2 * scale[img[i, j]]
            ascii_img += "\n"

        print(ascii_img)
