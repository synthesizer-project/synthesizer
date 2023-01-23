""" Definitions for image objects
"""
import math
import numpy as np
import synthesizer.exceptions as exceptions


class Scene:
    """
    The parent class for all "observations". These include:
    - Flux/rest frame luminosity images in photometric bands.
    - Images of underlying properties such as SFR, stellar mass, etc.
    - Data cubes (IFUs) containing spatially resolved spectra.

    This parent contains the attributes and methods common to all observation
    types to reduce boilerplate.

    Attributes
    ----------
    resolution : float
        The size a pixel.
    npix : int
        The number of pixels along an axis of the image or number of spaxels
        in the image plane of the IFU. 
    fov : float
        The width of the image/ifu. If coordinates are being used to make the
        image this should have the same units as those coordinates.
    sed : obj (SED)
        An sed object containing the spectra for this observation.
    survey : obj (Survey)
        WorkInProgress

    Raises
    ----------
    InconsistentArguments
        If an incompatible combination of arguments is provided an error is
        raised.

    """

    # Define slots to reduce memory overhead of this class and limit the
    # possible attributes.
    __slots__ = ["resolution", "fov", "npix", "sed", "survey"]

    def __init__(self, resolution, npix=None, fov=None, sed=None,
                 super_resolution_factor=None):
        """
        Intialise the Observation.

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
        survey : obj (Survey)
            WorkInProgress

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Check what we've been given
        self._check_obs_args(fov, npix)

        # Scene resolution, width and pixel information
        self.resolution = resolution
        self.fov = fov
        self.npix = npix

        # Define the super resolution (used to resample the scene to a higher
        # resolution if convolving with a PSF)
        self.super_resolution_factor = super_resolution_factor

        # Attributes containing data
        self.sed = sed

        # Handle the different input cases
        if npix is None:
            self._compute_npix()
        else:
            self._compute_fov()

        # # Include noise related attributes
        # self.pixel_noise = pixel_noise

        # # Set up noisy img
        # self.noisy_img = np.zeros(self.res, dtype=np.float64)

    def _check_obs_args(self, fov, npix):
        """
        Ensures we have a valid combination of inputs.


        Parameters
        ----------
        fov : float
            The width of the image.
        npix : int
            The number of pixels in the image.
        
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

    def _super_to_native_resolution(self):
        """
        Converts the super resolution resolution and npix into the native
        equivalents after PSF convolution.
        """

        # Perform conversion
        self.resolution *= self.super_resolution_factor
        self.npix //= self.super_resolution_factor

    def _native_to_super_resolution(self):
        """
        Converts the native resolution and npix into the super resolution
        equivalents used in PSF convolution.
        """

        # Perform conversion
        self.resolution /= self.super_resolution_factor
        self.npix *= self.super_resolution_factor

    def _compute_npix(self):
        """
        Compute the number of pixels in the FOV, ensuring the FOV is an
        integer number of pixels.

        There are multiple ways to define the dimensions of an image, this
        handles the case where the resolution and FOV is given.
        """

        # Compute how many pixels fall in the FOV
        self.npix = math.ceil(self.fov / self.resolution)

        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix

    def _compute_fov(self):
        """
        Compute the FOV, based on the number of pixels.

        There are multiple ways to define the dimensions of an image, this
        handles the case where the resolution and number of pixels is given.
        """

        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix


class ParticleScene(Scene):
    """
    The parent class for all "observations". These include:
    - Flux/rest frame luminosity images in photometric bands.
    - Images of underlying properties such as SFR, stellar mass, etc.
    - Data cubes (IFUs) containing spatially resolved spectra.

    This parent contains the attributes and methods common to all observation
    types to reduce boilerplate.

    Attributes
    ----------
    stars : obj (Stars)
        The object containing the stars to be placed in a image.
    coords : array-like (float)
        The position of particles to be sorted into the image.
    centre : array-like (float)
        The coordinates around which the image will be centered.
    pix_pos : array-like (float)
        The integer coordinates of particles in pixel units.
    npart : int
        The number of stellar particles.

    Raises
    ----------
    InconsistentArguments
        If an incompatible combination of arguments is provided an error is
        raised.

    """

    # Define slots to reduce memory overhead of this class
    __slots__ = ["stars", "coords", "centre", "pix_pos", "npart"]

    def __init__(self, resolution, npix=None, fov=None, sed=None, stars=None,
                 positions=None, centre=None, super_resolution_factor=None):
        """
        Intialise the ParticleObservation.

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
        survey : obj (Survey)
            WorkInProgress
        positons : array-like (float)
            The position of particles to be sorted into the image.
        centre : array-like (float)
            The coordinates around which the image will be centered. The if one
            is not provided then the geometric centre is calculated and used.

        Raises
        ------
        InconsistentArguments
            If an incompatible combination of arguments is provided an error is
            raised.

        """

        # Check what we've been given
        self._check_part_args(stars, positions, centre)

        # Initilise the parent class
        Scene.__init__(self, resolution=resolution, npix=npix, fov=fov,
                       sed=sed, super_resolution_factor=super_resolution_factor)

        # Initialise stars attribute
        self.stars = stars

        # Handle the particle positions, here we make a copy to avoid changing
        # the original values
        if self.stars is not None:
            self.coords = np.copy(self.stars.coordinates)
        else:
            self.coords = np.copy(positions)

        # If the positions are not already centred centre them
        self.centre = centre
        self._centre_coords()

        # Shift positions to start at 0
        self.coords += self.fov / 2

        # Run instantiation methods
        self.pix_pos = np.zeros(self.coords.shape, dtype=np.int32)
        self._get_pixel_pos()

        # How many particle are there?
        self.npart = self.coords.shape[0]

    def _check_part_args(self, stars, positions, centre):
        """
        Ensures we have a valid combination of inputs.

        Parameters
        ----------
        stars : obj (Stars)
            The object containing the stars to be placed in a image.
        positons : array-like (float)
            The position of particles to be sorted into the image.
        centre : array-like (float)
            The coordinates around which the image will be centered.

        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        InconsistentCoordinates
           If the centre does not lie within the range of coordinates an error
           is raised.
        """

        # Ensure we haven't been handed a resampled set of stars
        if stars is not None:
            if stars.resampled:
                raise exceptions.UnimplementedFunctionality(
                    "Functionality to make images from resampled stellar "
                    "distributions is currently unsupported. Contact the "
                    "authors if you wish to contribute this behaviour."
                )

        # Missing positions
        if stars is None and positions is None:
            raise exceptions.InconsistentArguments(
                "Either stars or positions must be specified!"
            )

        # The passed centre does not lie within the range of positions
        if centre is not None:
            if stars is None:
                pos = positions
            else:
                pos = stars.coordinates
                if (centre[0] < np.min(pos[:, 0]) or
                    centre[0] > np.max(pos[:, 0]) or
                    centre[1] < np.min(pos[:, 1]) or
                    centre[1] > np.max(pos[:, 1]) or
                    centre[2] < np.min(pos[:, 2]) or
                    centre[2] > np.max(pos[:, 2])):
                    raise exceptions.InconsistentCoordinates(
                        "The centre lies outside of the coordinate range. "
                        "Are they already centred?"
                    )

    def _centre_coords(self):
        """
        Centre coordinates on the geometric mean or the user provided centre
        """

        # Calculate the centre if necessary
        if self.centre is None:
            self.centre = np.mean(self.coords, axis=0)

        # Centre the coordinates
        self.coords -= self.centre

    def _get_pixel_pos(self):
        """
        Convert particle positions to interger pixel coordinates.

        These later help speed up sorting particles into pixels since their
        index is precomputed here.
        """

        # Convert sim positions to pixel positions
        self.pix_pos[:, 0] = self.coords[:, 0] / self.resolution
        self.pix_pos[:, 1] = self.coords[:, 1] / self.resolution
        self.pix_pos[:, 2] = self.coords[:, 2] / self.resolution


class ParametricScene(Scene):
    """
    The parent class for all parametric "observations". These include:
    - Flux/rest frame luminosity images in photometric bands.
    - Images of underlying properties such as SFR, stellar mass, etc.
    - Data cubes (IFUs) containing spatially resolved spectra.

    This parent contains all functionality needed for parametric observations.

    WorkInProgress

    Attributes
    ----------

    Raises
    ----------
    InconsistentArguments
        If an incompatible combination of arguments is provided an error is
        raised.

    """

    def __init__(self, resolution, npix=None, fov=None, sed=None,
                 super_resolution_factor=None):
        """
        Intialise the ParametricObservation.

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
        survey : obj (Survey)
            WorkInProgress

        """

        # Initilise the parent class
        Scene.__init__(self, resolution, npix, fov, sed,
                       super_resolution_factor=super_resolution_factor)
