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
                 survey=None):
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
        self._check_args(fov, npix)

        # Image metadata
        self.resolution = resolution
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

        # # Include noise related attributes
        # self.pixel_noise = pixel_noise

        # # Set up noisy img
        # self.noisy_img = np.zeros(self.res, dtype=np.float64)

    def _check_args(self, fov, npix):
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


class ParticleObservation(Observation):
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
        None

        Raises
        ------
        None

        """

        # Check what we've been given
        self._check_args(stars, positions)

        # Initilise the parent class
        Observation.__init__(resolution, npix, fov, sed, stars, survey)

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
        if np.min(self.sim_coords) != 0:

            # If so compute that offset and shift particles to start at 0
            self.pos_offset = np.min(self.sim_coords, axis=0)
            self.shifted_sim_pos -= self.pos_offset

        # Run instantiation methods
        self.pix_pos = np.zeros(self.sim_coords.shape, dtype=np.int32)
        self._get_pixel_pos()

    def _check_args(self, stars, positions):
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


class ParametricObservation(Observation):
    """
    The parent class used for creation of 2D images and IFUs (data cubes)

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 survey=None):
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
        None

        Raises
        ------
        None

        """

        # Initilise the parent class
        Observation.__init__(resolution, npix, fov, sed, stars, survey)
