"""Defintions for collections of generic images.

This module contains the definition for a generic ImageCollection class. This
provides the common functionality between particle and parametric imaging. The
user should not use this class directly, but rather use the
particle.imaging.Images and parametric.imaging.Images classes.

Example usage:
    # Create an image collection
    img_coll = ImageCollection(resolution=0.1 * unyt.arcsec, npix=100)

    # Get histograms of the particle distribution
    img_coll.get_imgs_hist(photometry, coordinates)

    # Get smoothed images of the particle distribution
    img_coll.get_imgs_smoothed(
        photometry,
        coordinates,
        smoothing_lengths,
        kernel,
        kernel_threshold,
    )

    # Get smoothed images of a parametric distribution
    img_coll.get_imgs_smoothed(
        photometry,
        density_grid=density_grid,
    )

    # Apply PSFs to the images
    img_coll.apply_psfs(psfs)

    # Apply noise to the images
    img_coll.apply_noise_from_stds(noise_stds)

    # Plot the images
    img_coll.plot_images()

    # Make an RGB image
    img_coll.make_rgb_image(rgb_filters, weights)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from unyt import unyt_quantity

from synthesizer import exceptions
from synthesizer.imaging.image import Image
from synthesizer.units import Quantity


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

    def __init__(
        self,
        resolution,
        fov=None,
        npix=None,
        imgs=None,
        noise_maps=None,
        weight_maps=None,
    ):
        """Initialize the image collection.

        Either fov or npix must be specified.

        An ImageCollection can either generate images or be initialised with
        an image dictionary, and optionally noise and weight maps. In practice
        the latter approach is mainly used only internally when generating
        new images from an existing ImageCollection.

        Args:
            resolution (unyt_quantity)
                The size of a pixel.
            fov (unyt_quantity/tuple, unyt_quantity)
                The width of the image.
            npix (int/tuple, int)
                The number of pixels in the image.
            imgs (dict)
                A dictionary of images to be turned into a collection.
            noise_maps (dict)
                A dictionary of noise maps associated to imgs.
            weight_maps (dict)
                A dictionary of weight maps associated to imgs.
        """
        # Check the arguments
        self._check_args(resolution, fov, npix)

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

        # Container for images (populated when image creation methods are
        # called)
        self.imgs = {}

        # Create placeholders for any noise and weight maps
        self.noise_maps = None
        self.weight_maps = None

        # Attribute for looping
        self._current_ind = 0

        # Store the filter codes
        self.filter_codes = []

        # A place holder for the RGB image
        self.rgb_img = None

        # Attach any images
        if imgs is not None:
            for f, img in imgs.items():
                self.imgs[f] = img
                self.filter_codes.append(f)

        # Attach any noise and weight maps
        if noise_maps is not None:
            self.noise_maps = {}
            for f, noise in noise_maps.items():
                self.noise_maps[f] = noise
        if weight_maps is not None:
            self.weight_maps = {}
            for f, weight in weight_maps.items():
                self.weight_maps[f] = weight

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

    def __len__(self):
        """Overload the len operator to return how many images there are."""
        return len(self.imgs)

    def __getitem__(self, filter_code):
        """
        Enable dictionary key look up syntax.

        This allows the user to extract specific images with the following
        syntax: ImageCollection["JWST/NIRCam.F150W"].

        Args:
            filter_code (str)
                The filter code of the desired photometry.

        Returns:
            Image
                The image corresponding to the filter code.
        """

        # Perform the look up
        return self.imgs[filter_code]

    def keys(self):
        """
        Enable dict.keys() behaviour.
        """
        return self.imgs.keys()

    def values(self):
        """
        Enable dict.values() behaviour.
        """
        return self.imgs.values()

    def items(self):
        """
        Enables dict.items() behaviour.
        """
        return self.imgs.items()

    def __iter__(self):
        """
        Overload iteration to allow simple looping over Image objects.

        Combined with __next__ this enables for f in ImageCollection syntax
        """
        return self

    def __next__(self):
        """
        Overload iteration to allow simple looping over Image objects.

        Combined with __iter__ this enables for f in ImageCollection syntax
        """
        # Check we haven't finished
        if self._current_ind >= len(self):
            self._current_ind = 0
            raise StopIteration
        else:
            # Increment index
            self._current_ind += 1

            # Return the filter
            return self.imgs[self.filter_codes[self._current_ind - 1]]

    def __add__(self, other_img):
        """
        Add two ImageCollections together.

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
            or np.any(self.fov != other_img.fov)
            or np.any(self.npix != other_img.npix)
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
        for f in photometry.filter_codes:
            # Create an Image object for this filter
            img = Image(self.resolution, self.fov)

            # Get the image for this filter
            img.get_img_hist(photometry[f], coordinates)

            # Store the image
            self.imgs[f] = img
            self.filter_codes.append(f)

    def get_imgs_smoothed(
        self,
        photometry,
        coordinates=None,
        smoothing_lengths=None,
        kernel=None,
        kernel_threshold=1,
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
        for f in photometry.filter_codes:
            # Create an Image object for this filter
            img = Image(self.resolution, self.fov)

            # Get the image for this filter
            img.get_img_smoothed(
                signal=photometry[f],
                coordinates=coordinates,
                smoothing_lengths=smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                density_grid=density_grid,
            )
            self.filter_codes.append(f)

            # Store the image
            self.imgs[f] = img

    def apply_psfs(self, psfs):
        """
        Convolve this ImageCollection's images with their PSFs.

        To more accurately apply the PSF we recommend using a super resolution
        image. This can be done via the supersample method and then
        downsampling to the native pixel scale after resampling. However, it
        is more efficient and robust to start at the super resolution initially
        and then downsample after the fact.

        Args:
            psfs (dict)
                A dictionary with a point spread function for each image within
                the ImageCollection. The key of each PSF must be the
                filter_code of the image it should be applied to.
        Returns:
            ImageCollection
                A new image collection containing the images convolved with a
                PSF.

        Raises:
            InconsistentArguments
                If a dictionary of PSFs is provided that doesn't match the
                filters an error is raised.
        """
        # Check we have a valid set of PSFs
        if not isinstance(psfs, dict):
            raise exceptions.InconsistentArguments(
                "psfs must be a dictionary with a PSF for each image"
            )
        missing_psfs = [f for f in self.imgs.keys() if f not in psfs]
        if len(missing_psfs) > 0:
            raise exceptions.InconsistentArguments(
                f"Missing a psf for the following filters: {missing_psfs}"
            )

        # Loop over each images and perform the convolution
        psfed_imgs = {}
        for f in psfs:
            # Apply the PSF to this image
            psfed_imgs[f] = self.imgs[f].apply_psf(psfs[f])

        return ImageCollection(
            resolution=self.resolution,
            npix=self.npix,
            imgs=psfed_imgs,
        )

    def apply_noise_arrays(self, noise_arrs):
        """
        Apply an existing noise array to each image.

        Args:
            noise_arrs (dict)
                A dictionary with a noise array for each image within the
                ImageCollection. The key of each noise array must be the
                filter_code of the image it should be applied to.

        Returns:
            ImageCollection
                A new image collection containing the images with noise
                applied.

        Raises:
            InconsistentArguments
                If the noise arrays dict is missing arguments an error is
                raised.
        """
        # Check we have a valid set of noise arrays
        if not isinstance(noise_arrs, dict):
            raise exceptions.InconsistentArguments(
                "noise_arrs must be a dictionary with a noise "
                "array for each image"
            )
        missing = [f for f in self.filter_codes if f not in noise_arrs]
        if len(missing) > 0:
            raise exceptions.InconsistentArguments(
                "Missing a noise array for the following "
                f"filters: {missing}"
            )

        # Loop over each images getting the noisy version
        noisy_imgs = {}
        weight_maps = {}
        for f in noise_arrs:
            # Apply the noise to this image
            noisy_imgs[f], weight_maps[f] = self.imgs[f].apply_noise_array(
                noise_arrs[f]
            )

        return ImageCollection(
            resolution=self.resolution,
            npix=self.npix,
            imgs=noisy_imgs,
            noise_maps=noise_arrs,
            weight_maps=weight_maps,
        )

    def apply_noise_from_stds(self, noise_stds):
        """
        Apply noise based on standard deviations of the noise distribution.

        Args:
            noise_stds (dict)
                A dictionary with a standard deviation for each image within
                the ImageCollection. The key of each standard deviation must
                be the filter_code of the image it should be applied to.

        Returns:
            ImageCollection
                A new image collection containing the images with noise
                applied.


        Raises:
            InconsistentArguments
                If a standard deviation for an image is missing an error is
                raised.
        """
        # Check we have a valid set of noise standard deviations
        if not isinstance(noise_stds, dict):
            raise exceptions.InconsistentArguments(
                "noise_stds must be a dictionary with a standard "
                "deviation for each image"
            )
        missing = [f for f in self.filter_codes if f not in noise_stds]
        if len(missing) > 0:
            raise exceptions.InconsistentArguments(
                "Missing a standard deviation for the following "
                f"filters: {missing}"
            )

        # Loop over each image getting the noisy version
        noisy_imgs = {}
        noise_maps = {}
        weight_maps = {}
        for f in noise_stds:
            # Apply the noise to this image
            noisy_imgs[f], noise_maps[f], weight_maps[f] = self.imgs[
                f
            ].apply_noise_from_std(noise_stds[f])

        return ImageCollection(
            resolution=self.resolution,
            npix=self.npix,
            imgs=noisy_imgs,
            noise_maps=noise_maps,
            weight_maps=weight_maps,
        )

    def apply_noise_from_snrs(self, snrs, depths, aperture_radius=None):
        """
        Apply noise based on SNRs and depths for each image.

        Args:
            snrs (dict)
                A dictionary containing the signal to noise ratio for each
                image within the ImageCollection. The key of each SNR must
                be the filter_code of the image it should be applied to.
            depths (dict)
                A dictionary containing the depth for each image within the
                ImageCollection. The key of each dpeth must be the filter_code
                of the image it should be applied to.
            aperture_radius (unyt_quantity)
                The radius of the aperture in which the SNR and depth is
                defined. This must have units attached and be in the same
                system as the images resolution (e.g. cartesian or angular).
                If not set a point source depth and SNR is assumed.

        Returns:
            ImageCollection
                A new image collection containing the images with noise
                applied.

        Raises:
            InconsistentArguments
                If a snr or depth for an image is missing an error is raised.
        """
        # Check we have a valid set of noise standard deviations
        if not isinstance(snrs, dict):
            raise exceptions.InconsistentArguments(
                "snrs must be a dictionary with a SNR for each image"
            )
        if not isinstance(depths, dict):
            raise exceptions.InconsistentArguments(
                "depths must be a dictionary with a depth for each image"
            )
        missing_snrs = [f for f in self.filter_codes if f not in snrs]
        missing_depths = [f for f in self.filter_codes if f not in depths]
        if len(missing_snrs) > 0:
            raise exceptions.InconsistentArguments(
                "Missing a SNR for the following " f"filters: {missing_snrs}"
            )
        if len(missing_depths) > 0:
            raise exceptions.InconsistentArguments(
                "Missing a depth for the following "
                f"filters: {missing_depths}"
            )
        if aperture_radius is not None and not isinstance(
            aperture_radius, unyt_quantity
        ):
            raise exceptions.InconsistentArguments(
                "aperture_radius must be given with units"
            )

        # Loop over each image getting the noisy version
        noisy_imgs = {}
        noise_maps = {}
        weight_maps = {}
        for f in snrs:
            # Apply the noise to this image
            noisy_imgs[f], noise_maps[f], weight_maps[f] = self.imgs[
                f
            ].apply_noise_from_snr(
                snr=snrs[f], depth=depths[f], aperture_radius=aperture_radius
            )

        return ImageCollection(
            resolution=self.resolution,
            npix=self.npix,
            imgs=noisy_imgs,
            noise_maps=noise_maps,
            weight_maps=weight_maps,
        )

    def plot_images(
        self,
        show=False,
        vmin=None,
        vmax=None,
        scaling_func=None,
        cmap="Greys_r",
    ):
        """
        Plot all images.

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

        # Do we need to find the normalisation for each filter?
        unique_norm_min = vmin is None
        unique_norm_max = vmax is None

        # Set up the figure
        fig = plt.figure(
            figsize=(4 * 3.5, int(np.ceil(len(self.filter_codes) / 4)) * 3.5)
        )

        # Create a gridspec grid
        gs = gridspec.GridSpec(
            int(np.ceil(len(self.filter_codes) / 4)), 4, hspace=0.0, wspace=0.0
        )

        # Loop over filters making each image
        for ind, f in enumerate(self.filter_codes):
            # Get the image
            img = self.imgs[f].arr

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
            ax.imshow(img, origin="lower", interpolation="nearest", cmap=cmap)
            ax.axis("off")

            # Place a label for which filter this ised_ASCII
            ax.text(
                0.95,
                0.9,
                f,
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
        self,
        rgb_filters,
        weights=None,
        scaling_func=None,
    ):
        """
        Make an rgb image from the ImageCollection.

        The filters in each channel are defined via the rgb_filters dict,
        with the option of providing weights for each filter.

        Args:
            rgb_filters (dict, array_like, str)
                A dictionary containing lists of each filter to combine to
                create the red, green, and blue channels.
                e.g.
                {
                "R": "Webb/NIRCam.F277W",
                "G": "Webb/NIRCam.F150W",
                "B": "Webb/NIRCam.F090W",
                }.
            weights (dict, array_like, float)
                A dictionary of weights for each filter. Defaults to equal
                weights.
            scaling_func (function)
                A function to scale the image by. This function should take a
                single array and produce an array of the same shape but scaled
                in the desired manner. The scaling is done to each channel
                individually.

        Returns:
            np.ndarray
                The RGB image array.
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
        rgb_img = np.zeros((self.npix[0], self.npix[1], 3), dtype=np.float64)

        # Loop over each filter calcualting the RGB channels
        for rgb_ind, rgb in enumerate(rgb_filters):
            for f in rgb_filters[rgb]:
                rgb_img[:, :, rgb_ind] += scaling_func(
                    weights[f] * self.imgs[f].arr
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
            np.ndarray
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
                "ImageCollection.make_rgb_image to compute the RGB "
                "image before plotting."
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
