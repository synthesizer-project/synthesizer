"""
Definitions for image objects
"""
import math
import numpy as np
import ctypes
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from unyt import unyt_quantity, kpc, mas, unyt_array, unyt_quantity
from unyt.dimensions import length, angle

import synthesizer.exceptions as exceptions
from synthesizer.imaging.scene import Scene, ParticleScene
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
    get_hist_imgs
        Sorts particles into singular pixels. If an array of pixel_values is
        passed then this is just a wrapper for numpy.histogram2d. Based on the
        inputs this function will either create multiple images (when filters
        is not None), storing them in a dictionary that is returned, or create
        a single image which is returned as an array.
    get_imgs
        Sorts particles into pixels, smoothing by a user provided kernel. Based
        on the inputs this function will either create multiple images (when
        filters is not None), storing them in a dictionary that is returned,
        or create a single image which is returned as an array.
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
        """

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
        self.rgb_img = None

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
        single band/property images). The resulting image object inherits its
        attributes from self, i.e in img = img1 + img2, img will inherit the
        attributes of img1.
        
        If the images are incompatible in dimension an error is thrown.
        
        Note: Once a new composite Image object is returned this will contain
        the combined image objects in the combined_imgs dictionary.
        
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

        # Initialise the composite image with the right type
        if isinstance(self, ParametricImage):
            composite_img = ParametricImage(
                morphology=None,
                resolution=self.resolution * self.spatial_unit,
                filters=self.filters,
                sed=self.sed,
                npix=None,
                fov=self.fov * self.spatial_unit,
                cosmo=self.cosmo,
                redshift=self.redshift,
                rest_frame=self.rest_frame,
                psfs=self.psfs,
                depths=self.depths,
                apertures=self.apertures,
                snrs=self.snrs,
            )
        else:
            composite_img = ParticleImage(
                resolution=self.resolution * self.spatial_unit,
                npix=None,
                fov=self.fov * self.spatial_unit,
                sed=self.sed,
                stars=self.stars,
                filters=None,
                positions=self.coords * self.coord_unit,
                pixel_values=None,
                smoothing_lengths=None,
                centre=None,
                rest_frame=self.rest_frame,
                cosmo=self.cosmo,
                redshift=self.redshift,
                psfs=self.psfs,
                depths=self.depths,
                apertures=self.apertures,
                snrs=self.snrs,
            )

        # Get the filter set for the composite, we have to handle the case
        # where one of the images is a single band/property image so can't
        # just take self.filters
        composite_filters = self.filters
        if len(composite_filters) == 0:
            composite_filters = other_img.filters
        elif len(other_img.filters) > 0:
            composite_filters += other_img.filters
        composite_img.filters = composite_filters

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

    def _get_hist_img_single_filter(self):
        """
        A place holder to be overloaded on child classes for making histogram
        images.
        """
        raise exceptions.UnimplementedFunctionality(
            "Image._get_hist_img_single_filter should be overloaded by child "
            "class. It is not designed to be called directly."
        )

    def _get_img_single_filter(self):
        """
        A place holder to be overloaded on child classes for making smoothed
        images.
        """
        raise exceptions.UnimplementedFunctionality(
            "Image._get_img_single_filter should be overloaded by "
            "child class. It is not designed to be called directly."
        )

    def get_hist_imgs(self):
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
        if len(self.filters) == 0:

            return self._get_hist_img_single_filter()

        # Calculate IFU "image"
        self.ifu = self.ifu_obj.get_hist_ifu()

        # Otherwise, we need to loop over filters and return a dictionary
        for f in self.filters:

            # Apply this filter to the IFU
            if self.rest_frame:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, nu=self.ifu_obj.sed._nu
                )

            else:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, nu=self.ifu_obj.sed.nuz
                )

        return self.imgs

    def get_imgs(self):
        """
        A generic method to calculate an image where particles are smoothed over
        a kernel.
        If pixel_values is defined then a single image is made and returned,
        if a filter list has been provided a image is made for each filter and
        returned in a dictionary. If neither of these situations has happened
        an error will have been produced at earlier stages.
        Returns
        -------
        img/imgs : array_like (float)/dictionary
            If pixel_values is provided: A 2D array containing particles
            smoothed and sorted into an image. (npix, npix)
            If a filter list is provided: A dictionary containing 2D array with
            particles smoothed and sorted into the image. (npix, npix)
        """

        # Handle the possible cases (multiple filters or single image)
        if len(self.filters) == 0:

            return self._get_img_single_filter()

        # Calculate IFU "image"
        self.ifu = self.ifu_obj.get_ifu()

        # Otherwise, we need to loop over filters and return a dictionary
        for f in self.filters:

            # Apply this filter to the IFU
            if self.rest_frame:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, nu=self.ifu_obj.sed._nu
                )
            else:
                self.imgs[f.filter_code] = f.apply_filter(
                    self.ifu, nu=self.ifu_obj.sed.nuz
                )

        return self.imgs

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
        
        To more accurately apply the PSF we recommend using a super resolution
        image. This can be done via the supersample method and then
        downsampling to the native pixel scale after resampling. However, it
        is more efficient and robust to start at the super resolution initially
        and then downsample after the fact.
        
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
                filter_codes -= set([key, ])

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
                filter_codes -= set([key, ])

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

        if len(self.filters) > 0 and isinstance(noises, dict):

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

    def plot_image(self, img_type="standard", filter_code=None, show=False,
                   vmin=None, vmax=None, scaling_func=None, cmap="Greys_r"):
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
        
        Parameters
        ----------
        img_type : str
            The type of images to combine. Can be "standard" for noiseless
            and psfless images (self.imgs), "psf" for images with psf
            (self.imgs_psf), or "noise" for images with noise 
            (self.imgs_noise).
        filter_code : str
            The filter code of the image to be plotted. If provided a plot is
            made only for this filter. This is not needed if the image object
            only contains a single image.
        show : bool
            Whether to show the plot or not (Default False).
        vmin : float
            The minimum value of the normalisation range.
        vmax : float
            The maximum value of the normalisation range.
        scaling_func : function
            A function to scale the image by. This function should take a
            single array and produce an array of the same shape but scaled in
            the desired manner.
        cmap : str
            The name of the matplotlib colormap for image plotting. Can be any
            valid string that can be passed to the cmap argument of imshow.
            Defaults to "Greys_r".
        
        Returns
        ----------
        matplotlib.pyplot.figure
            The figure object containing the plot
        matplotlib.pyplot.figure.axis
            The axis object containing the image.

        Raises
        ----------
        MissingImage
            If the requested image type has not yet been created and stored in
            this image object an exception is raised.
        """

        # Handle the scaling function for less branches
        if scaling_func is None:
            scaling_func = lambda x: x

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
            ax.imshow(img, origin="lower", interpolation="nearest",
                      cmap=cmap)
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
            gs = gridspec.GridSpec(int(np.ceil(len(self.filters) / 4)), 4,
                                   hspace=0.0, wspace=0.0)

            # Loop over filters making each image
            for ind, f in enumerate(self.filters):

                # Get the image
                img = imgs[f.filter_code]
                
                # Create the axis
                ax = fig.add_subplot(gs[int(np.floor(ind / 4)),
                                        ind % 4])

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
                ax.imshow(img, origin="lower", interpolation="nearest",
                          cmap=cmap)
                ax.axis("off")

                # Place a label for which filter this ised_ASCII
                ax.text(0.95, 0.9, f.filter_code,
                        bbox=dict(boxstyle="round,pad=0.3", fc='w', ec="k",
                                  lw=1, alpha=0.8), transform=ax.transAxes,
                        horizontalalignment='right')

        if show:
            plt.show()

        return fig, ax

    def make_rgb_image(self, rgb_filters, img_type="standard", weights=None,
                       scaling_func=None):
        """
        Makes an rgb image of specified filters with optional weights in
        each filter.
        
        Parameters
        ----------
        r_filters : dict (str: array_like (str))
            A dictionary containing lists of each filter to combine to create
            the red, green, and blue channels. e.g. {"R": "Webb/NIRCam.F277W",
            "G": "Webb/NIRCam.F150W", "B": "Webb/NIRCam.F090W"}.
        img_type : str
            The type of images to combine. Can be "standard" for noiseless
            and psfless images (self.imgs), "psf" for images with psf
            (self.imgs_psf), or "noise" for images with noise 
            (self.imgs_noise).
        weights : dict (str: array_like (str))
            A dictionary of weights for each filter. Defaults to equal weights.
        scaling_func : function
            A function to scale the image by. Defaults to arcsinh. This
            function should take a single array and produce an array of the
            same shape but scaled in the desired manner.
        Returns
        ----------
        array_like (float)
            The image array itself
        """

        # Handle the scaling function for less branches
        if scaling_func is None:
            scaling_func = lambda x: x

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
        
        Parameters
        ----------
        show : bool
            Whether to show the plot or not (Default False).
        vmin : float
            The minimum value of the normalisation range.
        vmax : float
            The maximum value of the normalisation range.
        
        Returns
        ----------
        matplotlib.pyplot.figure
            The figure object containing the plot
        matplotlib.pyplot.figure.axis
            The axis object containing the image.
        array_like (float)
            The rgb image array itself.

        Raises
        ----------
        MissingImage
            If the RGB image has not yet been created and stored in this image
            object an exception is raised.
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
        
        Parameters
        ----------
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
        scale = ("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft|()1{}[]?-_+~<>"
                 "i!lI;:,\"^`'. "[::-1])

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


class ParticleImage(ParticleScene, Image):
    """
    The Image object used when creating images from particle distributions.
    This can either be used by passing explict arrays of positions and values
    to sort into pixels or by passing SED and Stars Synthesizer objects. Images
    can be created with or without a PSF and noise.
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
            luminosity to flux.
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        igm : obj (Inoue14/Madau96)
            Object containing the absorbtion due to an intergalactic medium.
        """

        # Sanitize inputs
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
            cosmo=cosmo,
            redshift=redshift,
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

    def _get_img_single_filter(self):
        """
        A generic method to calculate an image where particles are smoothed over
        a kernel. This uses C extensions to calculate the image for each
        particle efficiently.
        
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


class ParametricImage(Scene, Image):
    """
    The Image object, containing attributes and methods for calculating images
    from parametric morphologies.
    
    Attributes
    ----------
    morphology : obj (BaseMorphology and children)
        The object that describes the parameters and creates the density grid
        for a particular morphology.
    
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
    ):
        """
        Intialise the ParametricImage.
        
        Parameters
        ----------
        morphology : obj (Morphology)
            The object that describes the parameters and creates the density
            grid for the desired morphology to be imaged.
        resolution : float
            The size a pixel.
        filters : obj (FilterCollection)
            An object containing a collection of filters.
        sed : obj (SED)
            An sed object containing the spectra for this observation.
        npix : int
            The number of pixels along an axis of the image or number of
            spaxels in the image plane of the IFU.
        fov : float
            The width of the image/ifu. If coordinates are being used to make
            the image this should have the same units as those coordinates.
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        redshift : float
            The redshift of the observation. Used when converting rest frame
            luminosity to flux.
        rest_frame : bool
            Are we making an observation in the rest frame?
        psfs : array-like (float)/dict
            The Point Spread Function to convolve with the image. Can be a
            single array or a dictionary with a PSF for each Filter.
        depths : float/dict
            The depth of this observation. Can either be a single value or a
            value per filter in a dictionary.
        apertures : float/dict
            The radius of the aperture depth is defined in, if not a point
            source depth, in the same units as the image resolution. Can either
            be a single radius or a radius per filter in a dictionary.
        snrs : float/dict
            The desired signal to noise of this observation. Assuming a
            signal-to-noise ratio of the form SN R= S / N = S / sqrt(sigma).
            Can either be a single SNR or a SNR per filter in a dictionary.
        super_resolution_factor : int
            The factor by which the resolution is divided to make the super
            resolution image used for PSF convolution.
        """

        # Sanitize inputs
        if filters is None:
            filters = ()

        # Initilise the parent classes
        Scene.__init__(
            self,
            resolution=resolution,
            npix=npix,
            fov=fov,
            sed=sed,
            rest_frame=rest_frame,
            cosmo=cosmo,
            redshift=redshift,
        )
        Image.__init__(
            self,
            filters=filters,
            psfs=psfs,
            depths=depths,
            apertures=apertures,
            snrs=snrs,
        )

        # Check our inputs
        self._check_parametric_img_args(morphology)

        # Store the morphology object
        self.morphology = morphology
        
        # Make the IFU, even if we only have 1 filter this is a minor overhead
        self.ifu_obj = ParametricSpectralCube(morphology, sed, resolution, 
                                              npix=npix, fov=fov,
                                              rest_frame=rest_frame)

    def _check_parametric_img_args(self, morphology):
        """
        Checks all arguments agree and do not conflict.

        Raises
        ----------
        """

        # For now ensure we have filters. Filterless images are not currently
        # supported.
        if self.filters is None:
            raise exceptions.UnimplementedFunctionality(
                "Parametric images are currently only supported for "
                "photometry when using filters. Provide a FilterCollection."
            )
        if len(self.filters) == 0:
            raise exceptions.UnimplementedFunctionality(
                "Parametric images are currently only supported for "
                "photometry when using filters. Provide a FilterCollection."
            )




