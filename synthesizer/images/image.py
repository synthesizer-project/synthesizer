""" Definitions for image objects
"""
import math
import numpy as np
import synthesizer.exceptions as exceptions


class Observation:
    """
    The parent class used for creation of 2D images and IFUs (data cubes)

    Attributes
    ----------


    Methods
    -------

    """

    # # Define slots to reduce memory overhead of this class
    # __slots__ = ["res", "width", "img_sum", "npart", "sim_pos",
    #              "shifted_sim_pos", "part_val", "pix_pos", "pos_offset",
    #              "img"]

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 survey=None, positions=None):
        """
        Intialise the Observation.

        Parameters
        ----------
        sed : SED object
           An sed object to make the image.
        fov : float
            The width of the image.
        npix : int
            The number of pixels in the image.
        stars : Stars object
            The object containing the stars to be placed in a image.
        positons : array-like (float)
            The position in the image of pixel values.

        Returns
        -------
        string
           a value in a string

        Raises
        ------
        KeyError
           when a key error
        OtherError
           when an other error
        """

        # Check what we've been given
        self._check_args(fov, npix, stars, positions)

        # Image metadata
        self.resulution = resolution
        self.fov = fov
        self.npix = npix

        # Attributes containing data
        self.sed = sed
        self.stars = stars
        self.survey = survey

        # Handle the different input cases
        if npix is None:
            self._compute_npix()
        else:
            self._compute_fov()

        # Handle the particle positions
        if stars is not None:
            self.sim_coords = stars.coods
            self.shifted_sim_pos = stars.coods

        else:
            self.sim_coords = positions
            self.shifted_sim_pos = positions

        # How many particle are there?
        self.npart = self.sim_coords.shape[0]

        # Are the positions centered?
        if np.min(self.sim_coords) < 0:

            # If so compute that offset and shift particles to start at 0
            self.pos_offset = np.min(self.sim_coords, axis=0)
            self.shifted_sim_pos -= self.pos_offset

        # Run instantiation methods
        self.pix_pos = np.zeros(self.sim_coords.shape, dtype=np.int32)
        self.get_pixel_pos()

        # # Include noise related attributes
        # self.pixel_noise = pixel_noise

        # # Set up noisy img
        # self.noisy_img = np.zeros(self.res, dtype=np.float64)

    def _check_args(self, fov, npix, stars, positions):
        """
        Ensures we have a valid combination of inputs.


        Parameters
        ----------
        fov : float
            The width of the image.
        npix : int
            The number of pixels in the image.
        stars : Stars object
            The object containing the stars to be placed in a image.
        positons : array-like (float)
            The position in the image of pixel values.

        Returns
        -------
        None

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Missing image size
        if fov is None and npix is None:
            raise exceptions.InconsistentArguments(
                "Either fov or npix must be specified!"
            )

        # Missing positions
        if stars is None and positions is None:
            raise exceptions.InconsistentArguments(
                "Either stars or positions must be specified!"
            )

    def _get_pixel_pos(self):
        """
        Convert particle positions to the pixel reference frame.
        """

        # TODO: Can threadpool this.

        # Convert sim positions to pixel positions
        self.pix_pos[:, 0] = self.shifted_sim_pos[:, 0] / self.width
        self.pix_pos[:, 1] = self.shifted_sim_pos[:, 1] / self.width
        self.pix_pos[:, 2] = self.shifted_sim_pos[:, 2] / self.width

    def _compute_npix(self):
        """
        Compute the number of pixels in the FOV, ensuring the FOV is an
        integer number of pixels.
        """

        # Compute how many pixels fall in the FOV
        self.npix = math.ceil(self.fov / self.resolution)

        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix

    def _compute_fov(self):
        """
        Compute the FOV, based on the number of pixels.
        """

        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix


class SpectralCube(Observation):
    """
    The IFU/Spectral data cube object, containing attributes and methods for
    calculating IFUs.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, sed, resolution, npix=None, fov=None, stars=None,
                 survey=None, positions=None):

        # Initilise the parent class
        super().__init__(resolution, npix, fov, sed, stars, survey, positions)

        # Set up the data cube dimensions
        self.nwlengths = sed.lam.size

        # Assign pixel values
        self.pixel_values = sed.fnu

        # Set up the image itself (populated later)
        self.ifu = np.zeros((self.npix, self.npix, self.nwlengths),
                            dtype=np.float64)

    def get_hist_ifu(self):
        """
        A generic function to calculate an image with no smoothing.


        Parameters
        ----------
        None

        Returns
        -------
        img : array_like (float)
            A 3D array containing the pixel values sorted into the image.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Loop over positions including the sed
        for ind in range(self.npart):

            self.ifu[self.pix_pos[ind, 0],
                     self.pix_pos[ind, 1], :] += self.pixel_values[ind, :]

        return self.ifu

    def get_smoothed_ifu(self):
        pass

    def get_psfed_ifu(self):
        pass

    def get_noisy_ifu(self):
        pass


class Image(Observation):
    """
    The Image object, containing attributes and methods for calculating images.

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 filters=(), survey=None, positions=None, pixel_values=None):

        # Check what we've been given
        self._check_args(sed, pixel_values, filters)

        # Initilise the parent class
        super().__init__(resolution, npix, fov, sed, stars, survey, positions)

        # Set up pixel values
        self.pixel_values = pixel_values

        # Set up filter objects
        self.filters = filters

        # If we have a list of filters make an IFU
        self._ifu_obj = None
        self.ifu = None
        if len(filters) > 0:
            self._ifu_obj = SpectralCube(sed, resolution, npix, fov, stars,
                                         survey, positions)

        # Set up img arrays. When multiple filters are provided we need a dict.
        self.img = np.zeros((self.npix, self.npix), dtype=np.float64)
        self.imgs = {f: np.zeros((self.npix, self.npix), dtype=np.float64)
                     for f in filters}

    def _check_args(self, sed, pixel_values, filters):
        """
        Ensures we have a valid combination of inputs.


        Parameters
        ----------
        sed : SED object
           An sed object to make the image.
        pixel_values : array-like (float)
            values to be interested in an image if sed is not present.

        Returns
        -------
        None

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Missing pixel_values
        if sed is None and pixel_values is None:
            raise exceptions.InconsistentArguments(
                "Either sed or pixel_values must be specified!"
            )

        # Missing filters
        if sed is not None and len(filters) == 0:
            raise exceptions.InconsistentArguments(
                "Filters must be specified when using an SED!"
            )

    def apply_filter(self, f):
        pass

    def _get_hist_img_single_filter(self):
        """
        A generic function to calculate an image with no smoothing. This
        function returns a single image.
        Just a wrapper for numpy's histogramming function.


        Parameters
        ----------
        None

        Returns
        -------
        img : array_like (float)
            A 2D array containing the pixel values sorted into the image.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        self.img = np.histogram2d(self.pix_pos[:, 0], self.pix_pos[:, 1],
                                  bins=self.npix,
                                  range=((0, self.fov), (0, self.fov)),
                                  weights=self.pixel_values)

        return self.img

    def get_smoothed_img_single_filter(self):
        pass

    def get_psfed_img_single_filter(self):
        pass

    def get_noisy_img_single_filter(self):
        pass

    def get_hist_img(self):
        """
        A generic function to calculate an image with no smoothing. This
        function returns a single image.
        Just a wrapper for numpy's histogramming function.


        Parameters
        ----------
        None

        Returns
        -------
        img : array_like (float)
            A 2D array containing the pixel values sorted into the image.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Handle the possible cases (multiple filters or single image)
        if self.pixel_values is not None:

            return self._get_hist_img_single_filter()

        # Calculate IFU "image"
        self.ifu = self._ifu_obj.get_hist_ifu()

        # Otherwise, we need to loop over filters and return a dictionary
        for f in self.filters:

            # Apply this filter to the IFU
            self.imgs[f] = self.apply_filter(f)

        return self.imgs

    def get_smoothed_img(self):
        pass

    def get_psfed_img(self):
        pass

    def get_noisy_img(self):
        pass
