"""Defintions for collections of generic images.

This module contains the definition for a generic ImageCollection class. This
provides the common functionality between particle and parametric imaging. The
user should not use this class directly, but rather use the
particle.imaging.Images and parametric.imaging.Images classes.

Example usage::

    # Create an image collection
    img_coll = ImageCollection(
        resolution=0.1 * unyt.arcsec,
        fov=(10, 10) * unyt.arcsec,
    )

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

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from unyt import unyt_quantity

from synthesizer import exceptions
from synthesizer.extensions.timers import tic, toc
from synthesizer.imaging.base_imaging import ImagingBase
from synthesizer.imaging.image import Image
from synthesizer.imaging.image_generators import (
    _generate_images_parametric_smoothed,
    _generate_images_particle_hist,
    _generate_images_particle_smoothed,
)
from synthesizer.utils import TableFormatter


class ImageCollection(ImagingBase):
    """A collection of Image objects.

    This contains all the generic methods for creating and manipulating
    images. In addition to generating images it can also apply PSFs and noise.

    Both parametric and particle based imaging uses this class.

    Attributes:
        imgs (dict):
            A dictionary of images to be turned into a collection.
        noise_maps (dict):
            A dictionary of noise maps to be applied to the images.
        weight_maps (dict):
            A dictionary of weight maps to be applied to the images.
        filter_codes (list):
            A list of filter codes for each image in the collection.
        rgb_img (np.ndarray of float):
            The RGB image array.
    """

    def __init__(
        self,
        resolution,
        fov,
        imgs=None,
    ):
        """Initialize the image collection.

        An ImageCollection can either generate images or be initialised with
        an image dictionary, and optionally noise and weight maps. In practice
        the latter approach is mainly used only internally when generating
        new images from an existing ImageCollection.

        Args:
            resolution (unyt_quantity):
                The size of a pixel.
            fov (unyt_quantity/tuple, unyt_quantity):
                The width of the image. If a single value is given then the
                image is assumed to be square.
            imgs (dict):
                A dictionary of images to be turned into a collection.
        """
        start = tic()
        # Instantiate the base class holding the geometry
        ImagingBase.__init__(self, resolution, fov)

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

        toc("Creating ImageCollection", start)

    @property
    def shape(self):
        """Return the shape of the image collection.

        Returns:
            tuple: A tuple containing (number of images, height, width) if
                  images exist, or an empty tuple if no images are present.
        """
        if self.imgs is None:
            return ()
        return (len(self.imgs), self.npix[0], self.npix[1])

    def downsample(self, factor):
        """Supersamples all images contained within this instance.

        Useful when applying a PSF to get more accurate convolution results.

        NOTE: It is more robust to create the initial "standard" image at high
        resolution and then downsample it after done with the high resolution
        version.

        Args:
            factor (float):
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
        """Supersample all images contained within this instance.

        Useful when applying a PSF to get more accurate convolution results.

        NOTE: It is more robust to create the initial "standard" image at high
        resolution and then downsample it after done with the high resolution
        version.

        Args:
            factor (float):
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

    def __str__(self):
        """Return a string representation of the ImageCollection.

        Returns:
            table (str)
                A string representation of the ImageCollection.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("ImageCollection")

    def __getitem__(self, filter_code):
        """Enable dictionary key look up syntax.

        This allows the user to extract specific images with the following
        syntax: ImageCollection["JWST/NIRCam.F150W"].

        Args:
            filter_code (str):
                The filter code of the desired photometry.

        Returns:
            Image
                The image corresponding to the filter code.
        """
        # Perform the look up
        if filter_code in self.imgs:
            return self.imgs[filter_code]

        # We may be being asked for all the images for an observatory, e.g.
        # "JWST", in which case we should return a new ImageCollection with
        # just those images.
        out = ImageCollection(resolution=self.resolution, fov=self.fov)
        for f in self.imgs:
            if filter_code in f:
                out.imgs[f.replace(filter_code + "/", "")] = self.imgs[f]
                out.filter_codes.append(f)

        # if we have any images, return the new ImageCollection
        if len(out) > 0:
            return out

        # We don't have any images, raise an error
        raise KeyError(
            f"Filter code {filter_code} not found in ImageCollection"
        )

    def __setitem__(self, filter_code, img):
        """Store the image at filter_code in the imgs dictionary.

        This allows the user to store specific images with the following
        syntax: ImageCollection["JWST/NIRCam.F150W"] = img.

        Image can either be a numpy array or an Image object. If it is a numpy
        array it is converted to an Image object.

        Args:
            filter_code (str):
                The filter code of the desired photometry.
            img (Image/unyt_array):
                The image to be added to the collection.
        """
        if not isinstance(img, Image):
            # Convert ndarray → Image
            img = Image(self.resolution, self.fov, img=img)

        # Insert / update while keeping filter_codes unique
        self.imgs[filter_code] = img
        if filter_code not in self.filter_codes:
            self.filter_codes.append(filter_code)

    def keys(self):
        """Return the keys of the image collection.

        This enables dict.keys() behaviour.

        Returns:
            list:
                The keys of the image collection.
        """
        return self.imgs.keys()

    def values(self):
        """Return the values of the image collection.

        This enables dict.values() behaviour.

        Returns:
            list:
                The values of the image collection.
        """
        return self.imgs.values()

    def items(self):
        """Return the items of the image collection.

        This enables dict.items() behaviour.

        Returns:
            list:
                The items of the image collection.
        """
        return self.imgs.items()

    def __iter__(self):
        """Overload iteration to allow simple looping over Image objects.

        Combined with __next__ this enables for f in ImageCollection syntax
        """
        return self

    def __next__(self):
        """Overload iteration to allow simple looping over Image objects.

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
        """Add two ImageCollections together.

        This combines all images with a common key.

        The resulting image object inherits its attributes from self, i.e in
        img = img1 + img2, img will inherit the attributes of img1.

        Args:
            other_img (ImageCollection):
                The other image collection to be combined with self.

        Returns:
            composite_img (ImageCollection):
                A new Image object containing the composite image of self and
                other_img.

        Raises:
            InconsistentAddition:
                If the ImageCollections can't be added and error is thrown.
        """
        # Make sure the images are compatible dimensions
        if (
            not np.isclose(self.resolution, other_img.resolution)
            or not np.allclose(self.fov, other_img.fov)
            or not np.allclose(self.npix, other_img.npix)
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
            fov=self.fov,
        )

        # Get common filters
        filters = set(list(self.imgs.keys())).intersection(
            set(list(other_img.imgs.keys()))
        )

        # Combine any common filters
        for f in filters:
            composite_img.filter_codes.append(f)
            composite_img.imgs[f] = self.imgs[f] + other_img.imgs[f]

        return composite_img

    def get_imgs_hist(
        self,
        photometry,
        coordinates,
        normalisations=None,
    ):
        """Calculate an image with no smoothing.

        Only applicable to particle based imaging.

        Args:
            photometry (PhotometryCollection):
                A dictionary of photometry for each filter.
            coordinates (unyt_array of float):
                The coordinates of the particles.
            normalisations (array_like, float):
                The normalisation property for each image. This is multiplied
                by the signal before sorting, then normalised out.

        Returns:
            ImageCollection: The image collection containing the generated
                images.
        """
        # Generate the images
        return _generate_images_particle_hist(
            self,
            coordinates=coordinates,
            signals=photometry,
            normalisations=normalisations,
        )

    def get_imgs_smoothed(
        self,
        photometry,
        coordinates=None,
        smoothing_lengths=None,
        kernel=None,
        kernel_threshold=1,
        density_grid=None,
        nthreads=1,
        normalisations=None,
    ):
        """Calculate an images from a smoothed distribution.

        In the particle case this smooths each particle's signal over the SPH
        kernel defined by their smoothing length. This uses C extensions to
        calculate the image for each particle efficiently.

        In the parametric case the signal is smoothed over a density grid. This
        density grid is an array defining the weight in each pixel.

        Args:
            photometry (unyt_array, float):
                The signal of each particle to be sorted into pixels.
            coordinates (unyt_array, float):
                The centered coordinates of the particles. (Only applicable to
                particle imaging)
            smoothing_lengths (unyt_array, float):
                The smoothing lengths of the particles. (Only applicable to
                particle imaging)
            kernel (str):
                The array describing the kernel. This is dervied from the
                kernel_functions module. (Only applicable to particle imaging)
            kernel_threshold (float):
                The threshold for the kernel. Particles with a kernel value
                below this threshold are included in the image. (Only
                applicable to particle imaging)
            density_grid (np.ndarray of float):
                The density grid to be smoothed over. (Only applicable to
                parametric imaging).
            nthreads (int):
                The number of threads to use when smoothing the image. This
                only applies to particle imaging.
            normalisations (array_like, float):
                The normalisation property. This is multiplied by the signal
                before sorting, then normalised out. (Only applicable to
                particle imaging)

        Returns:
            ImageCollection: The image collection containing the generated
                images.
        """
        # Call the correct image generation function (particle or parametric)
        if density_grid is not None and photometry is not None:
            # Generate the images for the parametric case
            return _generate_images_parametric_smoothed(
                self,
                density_grid=density_grid,
                signals=photometry,
            )
        elif (
            coordinates is not None
            and smoothing_lengths is not None
            and kernel is not None
            and kernel_threshold is not None
            and photometry is not None
        ):
            # Generate the images for the particle case
            return _generate_images_particle_smoothed(
                self,
                photometry.photometry,
                cent_coords=coordinates,
                smoothing_lengths=smoothing_lengths,
                labels=photometry.filter_codes,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
                normalisations=normalisations,
            )
        else:
            raise exceptions.InconsistentArguments(
                "Didn't find a valid set of arguments to generate images. "
                "Please provide either a density grid and photometry for "
                f"parametric imaging (found density_grid={type(density_grid)} "
                f"photometry={type(photometry)}) or coordinates, smoothing "
                f"lengths, kernel, and kernel_threshold for particle imaging "
                f"(found coordinates={type(coordinates)}, "
                f"smoothing_lengths={type(smoothing_lengths)}, "
                f"kernel={type(kernel)}, "
                f"kernel_threshold={type(kernel_threshold)}, "
                f"photometry={type(photometry)})"
            )

    def apply_psfs(self, psfs):
        """Convolve this ImageCollection's images with their PSFs.

        To more accurately apply the PSF we recommend using a super resolution
        image. This can be done via the supersample method and then
        downsampling to the native pixel scale after resampling. However, it
        is more efficient and robust to start at the super resolution initially
        and then downsample after the fact.

        Args:
            psfs (dict):
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
            fov=self.fov,
            imgs=psfed_imgs,
        )

    def apply_noise_arrays(self, noise_arrs):
        """Apply an existing noise array to each image.

        Args:
            noise_arrs (dict):
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
                f"Missing a noise array for the following filters: {missing}"
            )

        # Loop over each images getting the noisy version
        noisy_imgs = {}
        for f in noise_arrs:
            # Apply the noise to this image
            noisy_imgs[f] = self.imgs[f].apply_noise_array(noise_arrs[f])

        return ImageCollection(
            resolution=self.resolution,
            fov=self.fov,
            imgs=noisy_imgs,
        )

    def apply_noise_from_stds(self, noise_stds):
        """Apply noise based on standard deviations of the noise distribution.

        Args:
            noise_stds (dict):
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
        for f in noise_stds:
            # Apply the noise to this image
            noisy_imgs[f] = self.imgs[f].apply_noise_from_std(noise_stds[f])

        return ImageCollection(
            resolution=self.resolution,
            fov=self.fov,
            imgs=noisy_imgs,
        )

    def apply_noise_from_snrs(self, snrs, depths, aperture_radius=None):
        """Apply noise based on SNRs and depths for each image.

        Args:
            snrs (dict):
                A dictionary containing the signal to noise ratio for each
                image within the ImageCollection. The key of each SNR must
                be the filter_code of the image it should be applied to.
            depths (dict):
                A dictionary containing the depth for each image within the
                ImageCollection. The key of each dpeth must be the filter_code
                of the image it should be applied to.
            aperture_radius (unyt_quantity):
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
                f"Missing a SNR for the following filters: {missing_snrs}"
            )
        if len(missing_depths) > 0:
            raise exceptions.InconsistentArguments(
                f"Missing a depth for the following filters: {missing_depths}"
            )
        if aperture_radius is not None and not isinstance(
            aperture_radius, unyt_quantity
        ):
            raise exceptions.InconsistentArguments(
                "aperture_radius must be given with units"
            )

        # Loop over each image getting the noisy version
        noisy_imgs = {}
        for f in snrs:
            # Apply the noise to this image
            noisy_imgs[f] = self.imgs[f].apply_noise_from_snr(
                snr=snrs[f], depth=depths[f], aperture_radius=aperture_radius
            )

        return ImageCollection(
            resolution=self.resolution,
            fov=self.fov,
            imgs=noisy_imgs,
        )

    def plot_images(
        self,
        show=False,
        vmin=None,
        vmax=None,
        scaling_func=None,
        cmap="Greys_r",
        filters=None,
        ncols=4,
        individual_norm=False,
    ):
        """Plot all images.

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
            show (bool):
                Whether to show the plot or not (Default False).
            vmin (float):
                The minimum value of the normalisation range.
            vmax (float):
                The maximum value of the normalisation range.
            scaling_func (function):
                A function to scale the image by. This function should take a
                single array and produce an array of the same shape but scaled
                in the desired manner.
            cmap (str):
                The name of the matplotlib colormap for image plotting. Can be
                any valid string that can be passed to the cmap argument of
                imshow. Defaults to "Greys_r".
            filters (list):
                A list of filter codes to plot. If None, all filters will
                be plotted.
            ncols (int):
                The number of columns to use when plotting multiple images.
            individual_norm (bool):
                If True, each image will be normalised individually. If
                False, and vmin and vmax are not provided, the images will be
                normalised to the global min and max of all images.
                Defaults to False.

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
        unique_norm_min = vmin is None and individual_norm
        unique_norm_max = vmax is None and individual_norm

        # Set up the minima and maxima
        if vmin is None and not unique_norm_min:
            vmin = np.inf
            for f in self.imgs:
                minimum = np.percentile(self.imgs[f].arr, 32)
                if minimum < vmin:
                    vmin = minimum
        if vmax is None and not unique_norm_max:
            vmax = -np.inf
            for f in self.imgs:
                maximum = np.percentile(self.imgs[f].arr, 99.9)
                if maximum > vmax:
                    vmax = maximum

        # Are we looping over a specified set of filters?
        if filters is not None:
            filter_codes = filters
        else:
            filter_codes = self.filter_codes

        # Set up the figure
        fig = plt.figure(
            figsize=(
                ncols * 3.5,
                int(np.ceil(len(filter_codes) / ncols)) * 3.5,
            )
        )

        # Create a gridspec grid
        gs = gridspec.GridSpec(
            int(np.ceil(len(filter_codes) / ncols)),
            ncols,
            hspace=0.0,
            wspace=0.0,
        )

        # Loop over filters making each image
        for ind, f in enumerate(filter_codes):
            # Get the image
            img = self.imgs[f].arr

            # Create the axis
            ax = fig.add_subplot(gs[int(np.floor(ind / ncols)), ind % ncols])

            # Set up minima and maxima
            if unique_norm_min:
                vmin = np.min(img)
            if unique_norm_max:
                vmax = np.max(img)

            # Scale the image
            img = scaling_func(img)

            # Define the normalisation
            norm = plt.Normalize(
                vmin=scaling_func(vmin),
                vmax=scaling_func(vmax),
                clip=True,
            )

            # Plot the image and remove the surrounding axis
            ax.imshow(
                img,
                origin="lower",
                interpolation="nearest",
                cmap=cmap,
                norm=norm,
            )
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
        """Make an rgb image from the ImageCollection.

        The filters in each channel are defined via the rgb_filters dict,
        with the option of providing weights for each filter.

        Args:
            rgb_filters (dict, array_like, str):
                A dictionary containing lists of each filter to combine to
                create the red, green, and blue channels.
                e.g.
                {
                "R": "Webb/NIRCam.F277W",
                "G": "Webb/NIRCam.F150W",
                "B": "Webb/NIRCam.F090W",
                }.
            weights (dict, array_like, float):
                A dictionary of weights for each filter. Defaults to equal
                weights.
            scaling_func (function):
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
        """Plot an RGB image.

        Args:
            show (bool):
                Whether to show the plot or not (Default False).
            vmin (float):
                The minimum value of the normalisation range.
            vmax (float):
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

        # Clip the image to the normalisation range
        self.rgb_img = np.clip(self.rgb_img, vmin, vmax)

        # Normalise the image to the range 0-1
        rgb_img = (self.rgb_img - vmin) / (vmax - vmin)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(rgb_img, origin="lower", interpolation="nearest")
        ax.axis("off")

        if show:
            plt.show()

        return fig, ax, rgb_img
