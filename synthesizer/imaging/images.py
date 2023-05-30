"""
Definitions for image objects
"""
import math
import numpy as np
import ctypes
from scipy import signal
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from unyt import unyt_quantity, kpc, mas, unyt_array, unyt_quantity
from unyt.dimensions import length, angle

import synthesizer.exceptions as exceptions
from synthesizer.imaging.scene import Scene, ParticleScene, ParametricScene
from synthesizer.imaging.spectral_cubes import (
    ParticleSpectralCube,
    ParametricSpectralCube,
)


class Image():
    """
    The generic Image object, containing attributes and methods for calculating
    and manipulating images.
    This is the base class used for both particle and parametric images,
    containing the functionality common to both. Images can be made with or
    without a PSF and noise.
    Attributes
    ----------
    ifu_obj : obj (ParticleSpectralCube/ParametricSpectralCube)
        A local attribute holding a SpectralCube object. This is only used
        when a filter based image is requested, if an array of pixel values is
        provided this is never populated and used.
    ifu : array-like (float)
        The ifu array from _ifu_obj. This simplifies syntax for filter
        application. (npix, npix, nwavelength)
    filters : obj (FilterCollection)
        An imutable collection of Filter objects. If provided images are made
        for each filter.
    img : array-like (float)
        An array containing an image. Only used if pixel_values is defined and
        a single image is created. (npix, npix)
    imgs : dict
        A dictionary containing filter_code keys and img values. Only used if a
        FilterCollection is passed.
    combined_imgs : list
        A list containing any other image objects that were combined to
        make a composite image object.
    depths : float/dict
        The depth of this observation. Can either be a single value or a
        value per filter in a dictionary.
    snrs : float/dict
        The desired signal to noise of this observation. Assuming a
        signal-to-noise ratio of the form SN R= S / N = S / sqrt(sigma).
        Can either be a single SNR or a SNR per filter in a dictionary.
    apertures : float/dict
        The radius of the aperture depth is defined in, if not a point
        source depth, in the same units as the image resolution. Can either
        be a single radius or a radius per filter in a dictionary.
    Methods
    -------
    get_psfed_imgs
        Applies a user provided PSF to the images contained within this object.
        Note that a PSF for each filter must be provided in a dictionary if
        images have been made for each filter.
    get_noisy_imgs
        Applies noise defied by the user to the images contained within this
        object. Note that noise can be defined in a number of ways see
        documentation for details.
    """

    def __init__(
        self,
        filters=(),
        psfs=None,
        depths=None,
        apertures=None,
        snrs=None,
    ):
        """
        Intialise the Image.
        Parameters
        ----------
        resolution : float
            The size a pixel.
        npix : int
            The number of pixels along an axis of the image or number of
            spaxels in the image plane of the IFU.
        fov : float
            The width of the image/ifu. If coordinates are being used to make
            the image this should have the same units as those coordinates.
        filters : obj (FilterCollection)
            An object containing the Filter objects for which images are
            required.
        sed : obj (SED)
            An sed object containing the spectra for this observation.
        depths : float/dict
            The depth of this observation. Can either be a single value or a
            value per filter in a dictionary.
        snrs : float/dict
            The desired signal to noise of this observation. Assuming a
            signal-to-noise ratio of the form SN R= S / N = S / sqrt(sigma).
            Can either be a single SNR or a SNR per filter in a dictionary.
        apertures : float/dict
            The radius of the aperture depth is defined in, if not a point
            source depth, in the same units as the image resolution. Can either
            be a single radius or a radius per filter in a dictionary.
        super_resolution_factor : int
            The factor by which the resolution is divided to make the super
            resolution image used for PSF convolution.
        """

        # Sanitize inputs
        if filters is None:
            filters = ()

        # Define attributes to hold the PSF information
        self.psfs = psfs
        self._normalise_psfs

        # Intialise IFU attributes
        self.ifu_obj = None
        self.ifu = None

        # Set up filter objects
        self.filters = filters

        # Set up img arrays. When multiple filters are provided we need a dict.
        self.img = None
        self.img_psf = None
        self.img_noise = None
        self.imgs = {}
        self.imgs_psf = {}
        self.imgs_noise = {}

        # Set up a list to hold combined images.
        self.combined_imgs = []

        # Define attributes containing information for noise production.
        self.depths = depths
        self.apertures = apertures
        self.snrs = snrs

        # Set up arrays and dicts to store the noise arrays.
        self.weight_map = None
        self.noise_arr = None
        self.noise_arrs = {}
        self.weight_maps = {}

    def __add__(self, other_img):
        """
        Adds two img objects together, combining all images in all filters (or
        single band/property images).
        If the images are incompatible in dimension an error is thrown.
        Note: Once a new composite Image object is returned this will contain
        the combined images in the combined_imgs dictionary.
        Parameters
        ----------
        other_img : obj (Image/ParticleImage/ParametricImage)
            The other image to be combined with self.
        Returns
        -------
        composite_img : obj (Image)
             A new Image object contain the composite image of self and
             other_img.
        """

        # Make sure the images are compatible dimensions
        if (
            self.resolution != other_img.resolution
            or self.fov != other_img.fov
            or self.npix != other_img.npix
        ):
            raise exceptions.InconsistentAddition(
                "Cannot add Images: resolution=("
                + str(self.resolution)
                + " + "
                + str(other_img.resolution)
                + "), fov=("
                + str(self.fov)
                + " + "
                + str(other_img.fov)
                + "), npix=("
                + str(self.npix)
                + " + "
                + str(other_img.npix)
                + ")"
            )

        # Make sure they contain compatible filters (but we allow one
        # filterless image to be added to a image object with filters)
        if len(self.filters) > 0 and len(other_img.filters) > 0:
            if self.filters != other_img.filters:
                raise exceptions.InconsistentAddition(
                    "Cannot add Images with incompatible filter sets!"
                    + "\nFilter set 1:"
                    + "[ "
                    + ", ".join([fstr for fstr in self.filters.filter_codes])
                    + " ]"
                    + "\nFilter set 2:"
                    + "[ "
                    + ", ".join(
                        [fstr for fstr in other_img.filters.filter_codes]
                    )
                    + " ]"
                )

        # Get the filter set for the composite, we have to handle the case
        # where one of the images is a single band/property image so can't
        # just take self.filters
        composite_filters = self.filters
        if len(composite_filters) == 0:
            composite_filters = other_img.filters

        # Initialise the composite image
        composite_img = Image(
            self.resolution * self.spatial_unit,
            npix=self.npix,
            fov=self.fov * self.spatial_unit,
            filters=composite_filters,
            sed=None,
        )

        # Store the original images in the composite extracting any
        # nested images.
        if len(self.combined_imgs) > 0:
            for img in self.combined_imgs:
                composite_img.combined_imgs.append(img)
        else:
            composite_img.combined_imgs.append(self)
        if len(other_img.combined_imgs) > 0:
            for img in other_img.combined_imgs:
                composite_img.combined_imgs.append(img)
        else:
            composite_img.combined_imgs.append(other_img)

        # Now we can actually combine them, start with the single band/property
        if self.img is not None and other_img.img is not None:
            composite_img.img = self.img + other_img.img

        # Are we adding a single band/property image to a dictionary?
        elif self.img is not None and len(other_img.imgs) > 0:
            for key, img in other_img.imgs.items():
                composite_img.imgs[key] = img + self.img
        elif other_img.img is not None and len(self.imgs) > 0:
            for key, img in self.imgs.items():
                composite_img.imgs[key] = other_img.img + self.imgs[key]

        # Otherwise, we are simply combining images in multiple filters
        else:
            for key, img in self.imgs.items():
                composite_img.imgs[key] = img + other_img.imgs[key]

        return composite_img

    def _normalise_psfs(self):
        """
        Normalise the PSF/s just to be safe. If the PSF is correctly normalised
        doing this will not be harmful.
        """

        # Handle the different sort of psfs we can be given
        if isinstance(self.psfs, dict):
            for key in self.psfs:
                self.psfs[key] /= np.sum(self.psfs[key])
        else:
            self.psfs /= np.sum(self.psfs)

    @staticmethod
    def resample_img(img, factor):
        """
        Convolve an image with a PSF using scipy.signal.fftconvolve.
        Parameters
        ----------
        img : array-like (float)
            The image to resample.
        factor : float
            The factor by which to resample the image, >1 increases resolution,
            <1 decreases resolution.
        spline_order : int
            The order of the spline used during interpolation of the image onto
            the resampled resolution.
        Returns
        -------
        resampled_img : array_like (float)
            The image resampled by factor.
        """

        # Resample the image. (uses the default cubic order for interpolation)
        # NOTE: skimage.transform.pyramid_gaussian is more efficient but adds
        #       another dependency.
        if factor != 1:
            resampled_img = zoom(img, factor)
        else:
            resampled_img = img

        return resampled_img

    def _get_psfed_single_img(self, img, psf):
        """
        Convolve an image with a PSF using scipy.signal.fftconvolve.
        Parameters
        ----------
        img : array-like (float)
            The image to convolve with the PSF.
        psf : array-like (float)
            The PSF to convolve with the image.
        Returns
        -------
        convolved_img : array_like (float)
            The image convolved with the PSF.
        """

        # Perform the convolution
        convolved_img = signal.fftconvolve(img, psf, mode="same")

        # Downsample the image back to native resolution.
        convolved_img = self.resample_img(
            convolved_img, 1 / self.super_resolution_factor
        )

        return convolved_img

    def get_psfed_imgs(self):
        """
        Convolve the imgs stored in this object with the set of psfs passed to
        this method.
        This function will handle the different cases for image creation. If
        there are multiple filters it will use the psf for each filters,
        unless a single psf is provided in which case each filter will be
        convolved with the singular psf. If the Image only contains a single
        image it will convolve the psf with that image.
        To more accurately apply the PSF a super resolution image is
        automatically used. If psfs are supplied the resolution of the original
        image is increased by Image.super_resolution_factor. Once the PSF is
        completed the original image and PSFed images are downsampled back to
        native resolution.
        Parameters
        ----------
        psfs : array-like (float)/dict
            Either A single array describing a PSF or a dictionary containing a
        Returns
        -------
        img/imgs : array_like (float)/dictionary
            If pixel_values exists: A singular image convolved with the PSF.
            If a filter list exists: Each img in self.imgs is returned
            convolved with the corresponding PSF (or the single PSF if an array
            was supplied for psf).
        Raises
        -------
        InconsistentArguments
            If a dictionary of PSFs is provided that doesn't match the filters
            an error is raised.
        """

        # Get a local variable for the psfs
        psfs = self.psfs

        # Check we have a valid set of PSFs
        if self.pixel_values is not None and isinstance(psfs, dict):
            raise exceptions.InconsistentArguments(
                "To convolve with a single image an array should be "
                "provided for the PSF not a dictionary."
            )
        elif self.filters is not None and isinstance(psfs, dict):

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
        if self.pixel_values is not None:

            self.img_psf = self._get_psfed_single_img(self.img, psfs)

            # Now that we are done with the convolution return the original
            # images to the native resolution.
            self._super_to_native_resolution()

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

        # Now that we are done with the convolution return the original images
        # to the native resolution.
        self._super_to_native_resolution()

        return self.imgs_psf

    def _get_noisy_single_img(
        self, img, depth=None, snr=None, aperture=None, noise=None
    ):
        """
        Make and add a noise array to this image defined by either a depth and
        signal-to-noise in an aperture or by an explicit noise pixel value.
        Parameters
        ----------
        img : array-like (float)
            The image to add noise to.
        depth : float
            The depth of this observation.
        snr : float
            The desired signal to noise of this observation. Assuming a
            signal-to-noise ratio of the form SN R= S / N = S / sqrt(sigma).
        aperture : float
            The radius of the aperture depth is defined in, if not a point
            source depth, in the same units as the image resolution.
        noise : float
            The standard deviation of the noise distribution. If noise is
            provided then depth, snr and aperture are ignored.
        Returns
        -------
        noisy_img : array_like (float)
            The image with a noise contribution.
        Raises
        -------
        InconsistentArguments
            If noise isn't explictly stated and either depth or snr is
            missing an error is thrown.
        """

        # Ensure we have valid inputs
        if noise is None and (depth is None or snr is None):
            raise exceptions.InconsistentArguments(
                "Either a the explict standard deviation of the noise "
                "contribution (noise_sigma) or a signal-to-noise ratio and "
                "depth must be given."
            )

        # Calculate noise from the depth, aperture, and snr if given.
        if noise is None and aperture is not None:

            # Calculate the total noise in the aperture
            # NOTE: this assumes SNR = S / sqrt(app_noise)
            app_noise = (depth / snr) ** 2

            # Calculate the aperture area in image coordinates
            app_area_coords = np.pi * aperture**2

            # Convert the aperture area to units of pixels
            app_area_pix = app_area_coords / (self.resolution) ** 2

            # Get the noise per pixel
            # NOTE: here we remove the squaring done above.
            noise = np.sqrt(app_noise / app_area_pix)

        # Calculate the noise from the depth and snr for a point source.
        if noise is None and aperture is None:

            # Calculate noise in a pixel
            # NOTE: this assumes SNR = S / noise
            noise = depth / snr

        # Make the noise array and calculate the weight map
        noise_arr = noise * np.ones((self.npix, self.npix))
        weight_map = 1 / noise**2
        noise_arr *= np.random.randn(self.npix, self.npix)

        # Add the noise to the image
        if isinstance(noise_arr, np.ndarray):
            noisy_img = img + noise_arr
        else:
            noisy_img = img + noise_arr.value

        return noisy_img, weight_map, noise_arr

    def get_noisy_imgs(self, noises=None):
        """
        Make and add a noise array to each image in this Image object. The
        noise is defined by either a depth and signal-to-noise in an aperture
        or by an explicit noise pixel value.

        Note that the noise will be applied to the psfed images by default
        if they exist (those stored in self.imgs_psf). If those images do not
        exist then it will be applied to the standard images in self.imgs.
        Parameters
        ----------
        noises : float/dict
            The standard deviation of the noise distribution. If noises is
            provided then depth, snr and aperture are ignored. Can either be a
            single value or a value per filter in a dictionary.
        Returns
        -------
        noisy_img : array_like (float)
            The image with a noise contribution.
        Raises
        -------
        InconsistentArguments
            If dictionaries are provided and each filter doesn't have an entry
            and error is thrown.
        """

        # Check we have a valid set of PSFs
        # TODO: could move these to a check args function.
        if self.pixel_values is not None and (
            isinstance(self.depths, dict)
            or isinstance(self.snrs, dict)
            or isinstance(self.apertures, dict)
            or isinstance(noises, dict)
        ):
            raise exceptions.InconsistentArguments(
                "If there is a single image then noise arguments should be "
                "floats not dictionaries."
            )
        if self.filters is not None and isinstance(self.depths, dict):

            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in self.depths:
                filter_codes -= set([key, ])

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single depth or a dictionary of depths for each "
                    "filter must be given. Depths are missing for filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        if self.filters is not None and isinstance(self.snrs, dict):

            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in self.snrs:
                filter_codes -= set([key, ])

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single SNR or a dictionary of SNRs for each "
                    "filter must be given. SNRs are missing for filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        if self.filters is not None and isinstance(self.apertures, dict):

            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in self.apertures:
                filter_codes -= set([key, ])

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single aperture or a dictionary of apertures for"
                    " each filter must be given. Apertures are missing for "
                    "filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        if self.filters is not None and isinstance(noises, dict):

            # What filters are we missing psfs for?
            filter_codes = set(self.filters.filter_codes)
            for key in noises:
                filter_codes -= set([key, ])

            # If filters are missing raise an error saying which filters we
            # are missing
            if len(filter_codes) > 0:
                raise exceptions.InconsistentArguments(
                    "Either a single noise or a dictionary of noises for each "
                    "filter must be given. Noises are missing for filters: "
                    "[" + ", ".join(list(filter_codes)) + "]"
                )

        # Handle the possible cases (multiple filters or single image)
        if self.pixel_values is not None:

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
            if len(self.imgs_psf) > 
            noise_tuple = self._get_noisy_single_img(
                self.imgs_psf[f.filter_code], depth, snr, aperture, noise
            )

            # Store the resulting noisy image, weight, and noise arrays
            self.imgs_noise[f.filter_code] = noise_tuple[0]
            self.weight_maps[f.filter_code] = noise_tuple[1]
            self.noise_arrs[f.filter_code] = noise_tuple[2]

        return self.imgs_noise, self.weight_maps, self.noise_arrs

    def make_rgb_image(self, rgb_filters, img_type="standard", weights=None):
        """
        Makes an rgb image of specified filters.
        
        Parameters
        ----------
        r_filters : dict (str: array_like (str))
            A dictionary containing lists of each filter to combine to create
            the red, green, and blue channels. e.g. {"R": "Webb/NIRCam.F277W",
            "G": "Webb/NIRCam.F150W", "B": "Webb/NIRCam.F090W"}.
        img_type : str
            The type of images to combine. Can be "standard" for noiseless
            and psfless images (self.imgs), "psf" for images with psf
            (self.imgs_psf), or "noise" for images with noise \
            (self.imgs_noise).
        weights : dict (str: array_like (str))
            A dictionary of weights for each filter. Defaults to equal weights.
        Returns
        ----------
        array_like (float)
            The image array itself
        """

        # Handle the case where we haven't been passed weights
        for rgb in enumerate(rgb_filters):
            for f in rgb_filters[rgb]:
                weights[f] = 1.0

        # Ensure weights sum to 1.0
        for rgb in enumerate(rgb_filters):
            w_sum = 0
            for f in rgb_filters[rgb]:
                w_sum += weights[f]
            for f in rgb_filters[rgb]:
                weights[f] /= w_sum

        # Set up the rgb image
        rgb_img = np.zeros((self.npix, self.npix, 3), dtype=np.float64)

        for rgb_ind, rgb in enumerate(rgb_filters):
            for f in rgb:
                if img_type == "standard":
                    rgb_img[:, :, rgb_ind] += weights[f] * self.imgs[f]
                elif img_type == "psf":
                    rgb_img[:, :, rgb_ind] += weights[f] * self.imgs_psf[f]
                elif img_type == "noise":
                    rgb_img[:, :, rgb_ind] += weights[f] * self.imgs_noise[f]
                else:
                    raise exceptions.UnknownImageType(
                        "img_type can be 'standard', 'psf', or 'noise' "
                        "not '%s'" % img_type
                    )

        self.rgb_img = rgb_img

        return rgb_img

    def plot_rgb(self, rgb_filters, img_type="intrinsic", weights=None,
                 show=False):
        """
        Plot an RGB image.

        If one has not already been created one will be made.
        
        Parameters
        ----------
        r_filters : dict (str: array_like (str))
            A dictionary containing lists of each filter to combine to create
            the red, green, and blue channels. e.g. {"R": "Webb/NIRCam.F277W",
            "G": "Webb/NIRCam.F150W", "B": "Webb/NIRCam.F090W"}.
        img_type : str
            The type of images to combine. Can be "standard" for noiseless
            and psfless images (self.imgs), "psf" for images with psf
            (self.imgs_psf), or "noise" for images with noise \
            (self.imgs_noise).
        weights : dict (str: array_like (str))
            A dictionary of weights for each filter. Defaults to equal weights.
        show : bool
            Whether to show the plot or not (Default False).

        Returns
        ----------
        matplotlib.pyplot.figure
            The figure object containing the plot
        matplotlib.pyplot.figure.axis
            The axis object containing the image.
        """

        if self.rgb_img is None:
            self.make_rgb_image(rgb_filters)

        # Normalise the image.
        # TODO: allow the user to state minima and maxima
        rgb_img /= np.max(rgb_img)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(rgb_img, origin="lower", interpolation="nearest")
        ax.axis("off")

        if show:
            plt.show()

        return fig, ax


class ParticleImage(ParticleScene, Image):
    """
    The Image object used when creating images from particle distributions.
    This can either be used by passing explict arrays of positions and values
    to sort into pixels or by passing SED and Stars Synthesizer objects. Images
    can be created with or without a PSF and noise.
    Methods
    -------
    get_hist_img
        Sorts particles into singular pixels. If an array of pixel_values is
        passed then this is just a wrapper for numpy.histogram2d. Based on the
        inputs this function will either create multiple images (when filters
        is not None), storing them in a dictionary that is returned, or create
        a single image which is returned as an array.
    get_smoothed_img
        Sorts particles into pixels, smoothing by a user provided kernel. Based
        on the inputs this function will either create multiple images (when
        filters is not None), storing them in a dictionary that is returned,
        or create a single image which is returned as an array.
    """

    def __init__(
            self,
            resolution,
            npix=None,
            fov=None,
            sed=None,
            stars=None,
            filters=(),
            positions=None,
            pixel_values=None,
            smoothing_lengths=None,
            centre=None,
            rest_frame=True,
            redshift=None,
            cosmo=None,
            psfs=None,
            depths=None,
            apertures=None,
            snrs=None,
            super_resolution_factor=1,
    ):
        """
        Intialise the ParticleImage.
        Parameters
        ----------
        resolution : float
            The size a pixel.
        npix : int
            The number of pixels along an axis of the image or number of
            spaxels in the image plane of the IFU.
        fov : float
            The width of the image/ifu. If coordinates are being used to make
            the image this should have the same units as those coordinates.
        sed : obj (SED)
            An sed object containing the spectra for this observation.
        stars : obj (Stars)
            The object containing the stars to be placed in a image.
        filters : obj (FilterCollection)
            An object containing the Filter objects for which images are
            required.
        survey : obj (Survey)
            WorkInProgress
        positons : array-like (float)
            The position of particles to be sorted into the image.
        pixel_values : array-like (float)
            The values to be sorted/smoothed into pixels. Only needed if an sed
            and filters are not used.
        smoothing_lengths : array-like (float)
            The values describing the size of the smooth kernel for each
            particle. Only needed if star objects are not passed.
        centre : array-like (float)
            The centre to use for the image if not the geometric centre of
            the particle distribution.
        rest_frame : bool
            Are we making an observation in the rest frame?
        redshift : float
            The redshift of the observation. Used when converting rest frame
            luminosity to flux.bserv
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        igm : obj (Inoue14/Madau96)
            Object containing the absorbtion due to an intergalactic medium.
        """

        # Clean up arguments
        if filters is None:
            filters = ()

        # Initilise the parent classes
        ParticleScene.__init__(
            self,
            resolution=resolution,
            npix=npix,
            fov=fov,
            sed=sed,
            stars=stars,
            positions=positions,
            smoothing_lengths=smoothing_lengths,
            centre=centre,
            super_resolution_factor=super_resolution_factor,
            cosmo=cosmo,
            rest_frame=rest_frame,
        )
        Image.__init__(
            self,
            filters=filters,
            psfs=psfs,
            depths=depths,
            apertures=apertures,
            snrs=snrs,
        )

        # If we have a list of filters make an IFU
        if len(filters) > 0:
            self.ifu_obj = ParticleSpectralCube(
                sed=self.sed,
                resolution=self.orig_resolution,
                npix=self.orig_npix,
                fov=fov,
                stars=self.stars,
                rest_frame=rest_frame,
                cosmo=cosmo,
                super_resolution_factor=super_resolution_factor,
            )

        # Set up standalone arrays used when Synthesizer objects are not
        # passed.
        if isinstance(pixel_values, unyt_array):
            self.pixel_values = pixel_values.value
        else:
            self.pixel_values = pixel_values

    def _get_hist_img_single_filter(self):
        """
        A generic method to calculate an image with no smoothing.
        Just a wrapper for numpy.histogram2d utilising ParticleImage
        attributes.
        Returns
        -------
        img : array_like (float)
            A 2D array containing the pixel values sorted into the image.
            (npix, npix)
        """

        self.img = np.histogram2d(
            self.pix_pos[:, 0],
            self.pix_pos[:, 1],
            bins=self.npix,
            range=((0, self.npix), (0, self.npix)),
            weights=self.pixel_values,
        )[0]

        return self.img

    def _get_smoothed_img_single_filter(self, kernel_func):
        """
        A generic method to calculate an image where particles are smoothed over
        a kernel.
        Parameters
        ----------
        kernel_func : function
            A function describing the smoothing kernel that returns a single
            number between 0 and 1. This function can be imported from the
            options in kernel_functions.py or can be user defined. If user
            defined the function must return the kernel value corredsponding
            to the position of a particle with smoothing length h at distance
            r from the centre of the kernel (r/h).
        Returns
        -------
        img : array_like (float)
            A 2D array containing particles sorted into an image.
            (npix, npix)
        """

        from .extensions.sph_kernel_calc import make_img

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        # TODO: more memory efficient to pass the position array and handle C
        #       extraction.
        pix_vals = np.ascontiguousarray(self.pixel_values, dtype=np.float64)
        smls = np.ascontiguousarray(self.smoothing_lengths, dtype=np.float64)
        xs = np.ascontiguousarray(self.coords[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(self.coords[:, 1], dtype=np.float64)
        zs = np.ascontiguousarray(self.coords[:, 2], dtype=np.float64)

        self.img = make_img(pix_vals, smls, xs, ys, zs,
                            self.resolution, self.npix,
                            self.coords.shape[0])

        return self.img

    def get_hist_img(self):
        """
        A generic function to calculate an image with no smoothing.
        Parameters
        ----------
        None
        Returns
        -------
        img/imgs : array_like (float)/dictionary
            If pixel_values is provided: A 2D array containing particles
            smoothed and sorted into an image. (npix, npix)
            If a filter list is provided: A dictionary containing 2D array with
            particles smoothed and sorted into the image. (npix, npix)
        """

        # Handle the possible cases (multiple filters or single image)
        if self.pixel_values is not None:

            return self._get_hist_img_single_filter()

        # Calculate IFU "image"
        self.ifu = self.ifu_obj.get_hist_ifu()

        # Otherwise, we need to loop over filters and return a dictionary
        for f in self.filters:

            # Apply this filter to the IFU
            if self.rest_frame:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, self.ifu_obj.sed.nu
                )
            else:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, self.ifu_obj.sed.nuz
                )

        return self.imgs

    def get_smoothed_img(self, kernel_func):
        """
        A generic method to calculate an image where particles are smoothed over
        a kernel.
        If pixel_values is defined then a single image is made and returned,
        if a filter list has been provided a image is made for each filter and
        returned in a dictionary. If neither of these situations has happened
        an error will have been produced at earlier stages.
        Parameters
        ----------
        kernel_func : function
            A function describing the smoothing kernel that returns a single
            number between 0 and 1. This function can be imported from the
            options in kernel_functions.py or can be user defined. If user
            defined the function must return the kernel value corredsponding
            to the position of a particle with smoothing length h at distance
            r from the centre of the kernel (r/h).
        Returns
        -------
        img/imgs : array_like (float)/dictionary
            If pixel_values is provided: A 2D array containing particles
            smoothed and sorted into an image. (npix, npix)
            If a filter list is provided: A dictionary containing 2D array with
            particles smoothed and sorted into the image. (npix, npix)
        """

        # Handle the possible cases (multiple filters or single image)
        if self.pixel_values is not None:

            return self._get_smoothed_img_single_filter(kernel_func)

        # Calculate IFU "image"
        self.ifu = self.ifu_obj.get_smoothed_ifu(kernel_func)

        # Otherwise, we need to loop over filters and return a dictionary
        for f in self.filters:

            # Apply this filter to the IFU
            if self.rest_frame:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, self.ifu_obj.sed.nu
                )
            else:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, self.ifu_obj.sed.nuz
                )

        return self.imgs


class ParametricImage(ParametricScene, Image):
    """
    The Image object, containing attributes and methods for calculating images.
    WorkInProgress
    Attributes
    ----------
    Methods
    -------
    """

    def __init__(
        self,
        morphology,
        resolution,
        filters=None,
        sed=None,
        npix=None,
        fov=None,
        cosmo=None,
        redshift=None,
        rest_frame=True,
        psfs=None,
        depths=None,
        apertures=None,
        snrs=None,
        super_resolution_factor=None,
    ):
        """
        Intialise the ParametricImage.
        Parameters
        ----------
        resolution : float
            The size a pixel.
        filter_collection : obj (FilterCollection)
            An object containing a collection of filters.
        npix : int
            The number of pixels along an axis of the image or number of
            spaxels in the image plane of the IFU.
        fov : float
            The width of the image/ifu. If coordinates are being used to make
            the image this should have the same units as those coordinates.
        filters : obj (FilterCollection)
            An object containing the Filter objects for which images are
            required.
        survey : obj (Survey)
            WorkInProgress
        """

        # Initilise the parent classes
        ParametricScene.__init__(
            self,
            resolution=resolution,
            npix=npix,
            fov=fov,
            sed=sed,
            super_resolution_factor=super_resolution_factor,
            rest_frame=rest_frame,
        )
        Image.__init__(
            self,
            resolution=resolution,
            npix=npix,
            fov=fov,
            filters=filters,
            psfs=psfs,
            depths=depths,
            apertures=apertures,
            snrs=snrs,
        )

        # If we have a list of filters make an IFU
        if len(filters) > 0:
            self._ifu_obj = ParametricSpectralCube(sed, resolution, 
                                                   npix=npix, fov=fov)

        self.rest_frame = rest_frame

        # check resolution has units and convert to desired units
        if isinstance(resolution, unyt_quantity):
            if resolution.units.dimensions == angle:
                resolution = resolution.to("mas")
            elif resolution.units.dimensions == length:
                resolution = resolution.to("kpc")
            else:
                # raise exception, don't understand units
                pass
            _resolution = resolution.value
        else:
            # raise exception, resolution must have units
            pass

        # check morphology has the correct method
        # this might not be generic enough
        if (resolution.units == kpc) & (not morphology.model_kpc):

            if (cosmo != None) & (redshift != None):
                morphology.update(morphology.p, cosmo=cosmo, z=redshift)
            else:
                """raise exception, morphology is defined in mas but image
                resolution in kpc. Please provide cosmology (cosmo) and redshift.
                """
                pass

        if (resolution.units == mas) & (not morphology.model_mas):

            if (cosmo != None) & (redshift != None):
                morphology.update(morphology.p, cosmo=cosmo, z=redshift)
            else:
                """raise exception, morphology is defined in kpc but image
                resolution in mas. Please provide cosmology (cosmo) and redshift.
                """
                pass

        # Define 1D bin centres of each pixel
        bin_centres = _resolution * np.linspace(
            -(npix - 1) / 2, (npix - 1) / 2, npix
        )

        # As above but for the 2D grid
        self.xx, self.yy = np.meshgrid(bin_centres, bin_centres)

        # define the base image
        self.img = morphology.img(self.xx, self.yy, units=resolution.units)
        self.img /= np.sum(self.img)  # normalise this image to 1

    def create_images(self, sed=None):
        """
        Create multiband images
        Parameters
        ----------
        sed : obj (SED)
            An sed object containing the spectra for this observation.
        Returns
        ----------
        dictionary array
            a dictionary of images
        """

        if not sed:
            if self.sed:
                sed = self.sed
            else:
                # raise exception if no Sed object available
                pass

        # check if all filters have fluxes calculated
        if self.rest_frame:
            if sed.broadband_luminosities:
                filters = list(sed.broadband_luminosities.keys())
            else:
                # raise exception if broadband luminosities not generated
                pass
        else:
            if sed.broadband_fluxes:
                filters = list(sed.broadband_fluxes.keys())
            else:
                # raise exception if broadband fluxes not generated
                pass

        for filter_ in filters:
            self.imgs[filter_] = sed.broadband_luminosities[filter_] * self.img

        return self.imgs

    def plot(self, filter_code=None):
        """
        Make a simple plot of the image
        Parameters
        ----------
        filter_code : str
            The filter code
        """

        # if filter code provided use broadband image, else use base image
        if filter_code:
            img = self.imgs[filter_code]
        else:
            img = self.img

        plt.figure()

        plt.imshow(np.log10(img), origin="lower", interpolation="nearest")
        plt.show()

    def make_ascii(self, filter_code=None):
        """
        Make an ascii art image
        Parameters
        ----------
        filter_code : str
            The filter code
        """

        scale = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft|()1{}[]?-_+~<>i!lI;:,\"^`'. "[
            ::-1
        ]
        # scale = " .:-=+*#%@"
        nscale = len(scale)

        # if filter code provided use broadband image, else use base image
        if filter_code:
            img = self.imgs[filter_code]
        else:
            img = self.img

        img = (nscale - 1) * img / np.max(img)  # maps image onto a
        img = img.astype(int)

        ascii_img = ""
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                ascii_img += 2 * scale[img[i, j]]
            ascii_img += "\n"

        print(ascii_img)


