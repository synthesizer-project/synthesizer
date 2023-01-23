""" Definitions for image objects
"""
import math
import numpy as np
import matplotlib.pyplot as plt
import synthesizer.exceptions as exceptions
from synthesizer.imaging.observation import (Observation, ParticleObservation,
                                             ParametricObservation)
from synthesizer.imaging.spectral_cubes import (ParticleSpectralCube,
                                                ParametricSpectralCube)


class Image(Observation):
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

    Methods
    -------
    apply_filter
        Applies the transmission curve stored in Filter objects to the ifu and
        stores the resulting image in imgs.
    get_psfed_img
        NotYetImplemented
    get_noisy_img
        NotYetImplemented

    """

    def __init__(self, resolution, npix=None, fov=None, filters=(), sed=None,
                 survey=None):
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
        survey : obj (Survey)
            WorkInProgress

        """

        # Sanitize inputs
        if filters is None:
            filters = ()

        # Initilise the parent class
        Observation.__init__(self, resolution=resolution, npix=npix, fov=fov,
                             sed=sed, survey=survey)

        # Intialise IFU attributes
        self.ifu_obj = None
        self.ifu = None

        # Set up filter objects
        self.filters = filters

        # Set up img arrays. When multiple filters are provided we need a dict.
        self.img = np.zeros((self.npix, self.npix), dtype=np.float64)
        self.imgs = {}

    def apply_filter(self, f):
        """
        Applies a filters transmission curve defined by a Filter object to an
        IFU to get a single band image.

        Constructing an IFU first and applying the filter to the image will
        always be faster than calculating particle photometry and making
        multiple images when there are multiple bands. This way the image is
        made once as an IFU and the filter application which is much faster
        is done for each band.

        Parameters
        ----------
        f : obj (Filter)
            The Filter object containing all filter information.

        Returns
        -------
        img : array-like (float)
             A single band image in this filter with shape (npix, npix).
        """

        # Get the mask that removes wavelengths we don't currently care about
        in_band = f.t > 0

        # Multiply the IFU by the filter transmission curve
        transmitted = self.ifu[:, :, in_band] * f.t[in_band]

        # Sum over the final axis to "collect" transmission in this filer
        img = np.sum(transmitted, axis=-1)

        return img

    def get_psfed_img(self):
        pass

    def get_noisy_img(self):
        pass


class ParticleImage(ParticleObservation, Image):
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

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 filters=(), survey=None, positions=None, pixel_values=None,
                 rest_frame=True, redshift=None, cosmo=None, igm=None):
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

        # Initilise the parent classes
        ParticleObservation.__init__(self, resolution=resolution, npix=npix,
                                     fov=fov, sed=sed, stars=stars,
                                     survey=survey, positions=positions)
        Image.__init__(self, resolution=resolution, npix=npix, fov=fov,
                       filters=filters, sed=sed, survey=survey)

        # If we have a list of filters make an IFU
        if len(filters) > 0:
            self.ifu_obj = ParticleSpectralCube(sed=self.sed,
                                                resolution=self.resolution,
                                                npix=self.npix, fov=self.fov,
                                                stars=self.stars,
                                                survey=self.survey,
                                                rest_frame=rest_frame,
                                                redshift=redshift, cosmo=cosmo,
                                                igm=igm)

        # Set up pixel values
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

        self.img = np.histogram2d(self.pix_pos[:, 0], self.pix_pos[:, 1],
                                  bins=self.npix,
                                  range=((0, self.npix), (0, self.npix)),
                                  weights=self.pixel_values)[0]

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

        # Get the size of a pixel
        res = self.resolution

        # Loop over positions including the sed
        for ind in range(self.npart):

            # Get this particles smoothing length and position
            smooth_length = self.stars.smoothing_lengths[ind]
            pos = self.shifted_sim_pos[ind]

            # How many pixels are in the smoothing length?
            delta_pix = math.ceil(smooth_length / self.resolution) + 1

            # Loop over a square aperture around this particle
            # NOTE: This includes "pixels" in front of and behind the image
            #       plane since the kernel is by defintion 3D
            # TODO: Would be considerably more accurate to integrate over the
            #       kernel in z axis since this is not quantised into pixels
            #       like the axes in the image plane.
            for i in range(self.pix_pos[ind, 0] - delta_pix,
                           self.pix_pos[ind, 0] + delta_pix + 1):

                # Skip if outside of image
                if i < 0 or i >= self.npix:
                    continue

                # Compute the x separation
                x_dist = (i * res) + (res / 2) - pos[0]

                for j in range(self.pix_pos[ind, 1] - delta_pix,
                               self.pix_pos[ind, 1] + delta_pix + 1):

                    # Skip if outside of image
                    if j < 0 or j >= self.npix:
                        continue

                    # Compute the y separation
                    y_dist = (j * res) + (res / 2) - pos[1]

                    for k in range(self.pix_pos[ind, 2] - delta_pix,
                                   self.pix_pos[ind, 2] + delta_pix + 1):

                        # Compute the z separation
                        z_dist = (k * res) + (res / 2) - pos[2]

                        # Compute the distance between the centre of this pixel
                        # and the particle.
                        dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)

                        # Get the value of the kernel here
                        kernel_val = kernel_func(dist / smooth_length)

                        # Add this pixel's contribution
                        self.img[i, j] += self.pixel_values[ind] * kernel_val

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
            self.imgs[f.filter_code] = self.apply_filter(f)

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
            self.imgs[f.filter_code] = self.apply_filter(f)

        return self.imgs


class ParametricImage(ParametricObservation, Image):
    """
    The Image object, containing attributes and methods for calculating images.

    WorkInProgress

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, filters, resolution, morphology, npix=None, fov=None, survey=None):
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
        ParametricObservation.__init__(self, resolution=resolution, npix=npix,
                                       fov=fov, survey=survey)
        Image.__init__(self, resolution=resolution, npix=npix, fov=fov,
                       filters=filters, survey=survey)

        # Define 1D bin centres of each pixel
        bin_centres = resolution * np.linspace(-(npix-1)/2, (npix-1)/2, npix)

        # As above but for the 2D grid
        self.xx, self.yy = np.meshgrid(bin_centres, bin_centres)

        # define the base image
        self.img = morphology.img(xx, yy)
        self.img /= np.sum(self.img)  # normalise this image to 1

    def create_images(self, sed, filters=None):
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

        sed_filters = list(sed.broadband_luminosities.keys())

        # if filters not given read from sed object
        if not filters:
            filters = sed_filters

        # check if all filters have fluxes calculated

        for filter_ in filters:
            self.imgs[filter_] = sed.broadband_luminosities[filter_] * self.img

        return self.imgs

    def plot(self, filter_code):
        """
        Make a simple plot of the image

        Parameters
        ----------
        filter_code : str
            The filter code
        """

        plt.figure()
        plt.imshow(np.log10(self.imgs[filter_code]), origin='lower', interpolation='nearest')
        plt.show()

    def make_rgb_image(self, rgb_filters, update=True):
        """
        Make an rgb image

        Parameters
        ----------
        filter_code : str
            rgb_filters
        """

        rgb_img = np.array([self.imgs[filter_code] for filter_code in rgb_filters]).T

        if update:
            self.rgb_img = rgb_img

        return rgb_img

    def plot_rgb(self, rgb_filters):
        """
        Make a simple rgb plot

        Parameters
        ----------
        filter_code : str
            rgb_filters
        """

        rgb_img = self.make_rgb_image(rgb_filters)

        rgb_img /= np.max(rgb_img)

        plt.figure()
        plt.imshow(rgb_img, origin='lower', interpolation='nearest')
        plt.show()
