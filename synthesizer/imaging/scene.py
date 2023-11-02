""" Definitions for image objects
"""
import math
import numpy as np
import unyt
from unyt import arcsec, kpc
from scipy.ndimage import zoom

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
    spatial_unit : obj (unyt.unit)
        The units of the spatial image/IFU information (coordinates,
        resolution, fov, apertures, etc.).
    Raises
    ----------
    InconsistentArguments
        If an incompatible combination of arguments is provided an error is
        raised.
    """

    # Define slots to reduce memory overhead of this class and limit the
    # possible attributes.
    __slots__ = [
        "resolution",
        "fov",
        "npix",
        "sed",
        "survey",
        "spatial_unit",
        "orig_resolution",
        "orig_npix",
    ]

    def __init__(
        self,
        resolution,
        npix=None,
        fov=None,
        sed=None,
        rest_frame=True,
        cosmo=None,
        redshift=None,
    ):
        """
        Intialise the Observation.
        Parameters
        ----------
        resolution : Quantity (float * unyt.unit)
            The size a pixel.
        npix : int
            The number of pixels along an axis of the image or number of
            spaxels in the image plane of the IFU.
        fov : Quantity (float * unyt.unit)
            The width of the image/ifu. If coordinates are being used to make
            the image this should have the same units as those coordinates.
        sed : obj (SED)
            An sed object containing the spectra for this observation.
        rest_frame : bool
            Is the observation in the rest frame or observer frame. Default
            is rest frame (True).
        Raises
        ------
        InconsistentArguments
           Errors when an incorrect combination of arguments is passed.
        """

        # Check what we've been given
        self._check_scene_args(resolution, fov, npix)

        # Define the spatial units of the image
        self.spatial_unit = resolution.units

        # Scene resolution, width and pixel information
        self.resolution = resolution.value
        self.fov = fov
        self.npix = npix

        # Check unit consistency
        self._convert_scene_coords()

        # Attributes containing data
        self.sed = sed

        # Store the cosmology object and redshift
        self.cosmo = cosmo
        self.redshift = redshift

        # Keep track of the input resolution and and npix so we can handle
        # super resolution correctly.
        self.orig_resolution = resolution
        self.orig_npix = npix

        # Handle the different input cases
        if npix is None:
            self._compute_npix()
        elif fov is None:
            self._compute_fov()

        # What frame are we observing in?
        self.rest_frame = rest_frame

    def _check_scene_args(self, resolution, fov, npix):
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
        Compute the number of pixels in the FOV, ensuring the FOV is an
        integer number of pixels.
        There are multiple ways to define the dimensions of an image, this
        handles the case where the resolution and FOV is given.
        """

        # Compute how many pixels fall in the FOV
        self.npix = int(math.ceil(self.fov / self.resolution))
        if self.orig_npix is None:
            self.orig_npix = int(math.ceil(self.fov / self.resolution))

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

    def _convert_scene_coords(self):
        """
        Ensures that the resolution and FOV are in consistent units.
        """

        # If we haven't been given an FOV there's nothing to do.
        if self.fov is None:
            return

        # Otherwise, do we need to convert the FOV to the correct units?
        elif self.fov.units != self.spatial_unit:
            # If the units have the same dimension convert them
            if self.fov.units.same_dimensions_as(self.spatial_unit):
                self.fov.to(self.spatial_unit)

            # For now raise an error, in the future we could handle length to
            # angle conversions.
            else:
                raise exceptions.UnimplementedFunctionality(
                    "Conversion between length and angular units for "
                    "resolution and FOV not yet supported."
                )

        # Strip off the units
        self.fov = self.fov.value

    def _resample(self, factor):
        """
        Helper function to resample all images contained within this instance
        by the stated factor using interpolation.

        Parameters
        ----------
        factor : float
            The factor by which to resample the image, >1 increases resolution,
            <1 decreases resolution.
        """

        # Perform the conversion on the basic image properties
        self.resolution /= factor
        self._compute_npix()

        # Resample the image/s using the scipy default cubic order for
        # interpolation.
        # NOTE: skimage.transform.pyramid_gaussian is more efficient but adds
        #       another dependency.
        if self.img is not None:
            self.img = zoom(self.img, factor)
            new_shape = self.img.shape
        if len(self.imgs) > 0:
            for f in self.imgs:
                self.imgs[f] = zoom(self.imgs[f], factor)
                new_shape = self.imgs[f].shape
        if self.img_psf is not None:
            self.img_psf = zoom(self.img_psf, factor)
        if len(self.imgs_psf) > 0:
            for f in self.imgs_psf:
                self.imgs_psf[f] = zoom(self.imgs_psf[f], factor)
        if self.img_noise is not None:
            self.img_noise = zoom(self.img_noise, factor)
        if len(self.imgs_noise) > 0:
            for f in self.imgs_noise:
                self.imgs_noise[f] = zoom(self.imgs_noise[f], factor)

        # Handle the edge case where the conversion between resolutions has
        # messed with Scene properties.
        if self.npix != new_shape[0]:
            self.npix = new_shape
            self._compute_fov()

    def downsample(self, factor):
        """
        Supersamples all images contained within this instance by the stated
        factor using interpolation. Useful when applying a PSF to get more
        accurate convolution results.

        NOTE: It is more robust to create the initial "standard" image at high
        resolution and then downsample it after done with the high resolution
        version.

        Parameters
        ----------
        factor : float
            The factor by which to resample the image, >1 increases resolution,
            <1 decreases resolution.
        Raises
        -------
        ValueError
            If the incorrect resample function is called an error is raised to
            ensure the user does not erroneously resample.
        """

        # Check factor (NOTE: this doesn't actually cause an issue
        # mechanically but will ensure users are literal about resampling and
        # can't mistakenly resample in unintended ways).
        if factor > 1:
            raise ValueError("Using downsample method to supersample!")

        # Resample the images
        self._resample(factor)

    def supersample(self, factor):
        """
        Supersamples all images contained within this instance by the stated
        factor using interpolation. Useful when applying a PSF to get more
        accurate convolution results.

        NOTE: It is more robust to create the initial "standard" image at high
        resolution and then downsample it after done with the high resolution
        version.

        Parameters
        ----------
        factor : float
            The factor by which to resample the image, >1 increases resolution,
            <1 decreases resolution.
        Raises
        -------
        ValueError
            If the incorrect resample function is called an error is raised to
            ensure the user does not erroneously resample.
        """

        # Check factor (NOTE: this doesn't actually cause an issue
        # mechanically but will ensure users are literal about resampling and
        # can't mistakenly resample in unintended ways).
        if factor < 1:
            raise ValueError("Using supersample method to downsample!")

        # Resample the images
        self._resample(factor)


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
    kernel (array-like, float)
        The values from one of the kernels from the kernel_functions module.
        Only used for smoothed images.
    kernel_dim:
        The number of elements in the kernel.
    kernel_threshold (float)
        The kernel's impact parameter threshold (by default 1).
    Raises
    ----------
    InconsistentArguments
        If an incompatible combination of arguments is provided an error is
        raised.
    """

    # Define slots to reduce memory overhead of this class
    __slots__ = ["stars", "coords", "centre", "pix_pos", "npart", "smoothing_lengths"]

    def __init__(
        self,
        resolution,
        npix=None,
        fov=None,
        sed=None,
        stars=None,
        positions=None,
        smoothing_lengths=None,
        centre=None,
        cosmo=None,
        redshift=None,
        rest_frame=True,
        kernel=None,
        kernel_threshold=1,
    ):
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
        smoothing_lengths : array-like (float)
            The values describing the size of the smooth kernel for each
            particle. Only needed if star objects are not passed.
        centre : array-like (float)
            The coordinates around which the image will be centered. The if one
            is not provided then the geometric centre is calculated and used.
        kernel (array-like, float)
            The values from one of the kernels from the kernel_functions module.
            Only used for smoothed images.
        kernel_threshold (float)
            The kernel's impact parameter threshold (by default 1).
        Raises
        ------
        InconsistentArguments
            If an incompatible combination of arguments is provided an error is
            raised.
        """

        # Check what we've been given
        self._check_part_args(resolution, stars, positions, centre, cosmo, sed)

        # Initilise the parent class
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

        # Initialise stars attribute
        self.stars = stars

        # Handle the particle positions, here we make a copy to avoid changing
        # the original values
        if self.stars is not None:
            self.coords = np.copy(self.stars._coordinates)
            self.coord_unit = self.stars.coordinates.units
        else:
            self.coords = np.copy(positions)
            self.coord_unit = positions.units

        # If the positions are not already centred centre them
        self.centre = centre
        self._centre_coords()

        # Store the smoothing lengths because we again need a copy
        if self.stars is not None:
            self.smoothing_lengths = np.copy(self.stars._smoothing_lengths)
            self.smooth_unit = self.stars.smoothing_lengths.units
        else:
            if smoothing_lengths is not None:
                self.smoothing_lengths = np.copy(smoothing_lengths)
                self.smooth_unit = smoothing_lengths.units
            else:
                self.smoothing_lengths = None
                self.smooth_unit = None

        # Convert coordinates to the image's spatial units
        self._convert_to_img_units()

        # Shift positions to start at 0
        self.coords += self.fov / 2

        # Calculate the position of particles in pixel coordinates
        self.pix_pos = np.zeros(self.coords.shape, dtype=np.int32)
        self._get_pixel_pos()

        # How many particle are there?
        self.npart = self.coords.shape[0]

        # Set up the kernel attributes we need
        if kernel is not None:
            self.kernel = kernel
            self.kernel_dim = kernel.size
            self.kernel_threshold = kernel_threshold
        else:
            self.kernel = None
            self.kernel_dim = None
            self.kernel_threshold = None

    def _check_part_args(self, resolution, stars, positions, centre, cosmo, sed):
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

        # TODO: check units work for array based images.

        # Get the spatial units
        spatial_unit = resolution.units

        # Have we been given an integrated SED by accident?
        if sed is not None:
            if len(sed.lnu.shape) == 1:
                raise exceptions.InconsistentArguments(
                    "Particle Spectra are required for imaging, an integrated "
                    "spectra has been passed."
                )

        # Check the stars we have been given.
        if stars is not None:
            # Ensure we haven't been handed a resampled set of stars
            if stars.resampled:
                raise exceptions.UnimplementedFunctionality(
                    "Functionality to make images from resampled stellar "
                    "distributions is currently unsupported. Contact the "
                    "authors if you wish to contribute this behaviour."
                )

            # If we are working in terms of angles we need redshifts for the
            # stars.
            if spatial_unit.same_dimensions_as(arcsec) and stars.redshift is None:
                raise exceptions.InconsistentArguments(
                    "When working in an angular unit system the provided "
                    "stars need a redshift associated to them. Stars.redshift"
                    " can either be a single redshift for all stars or an "
                    "array of redshifts for each star."
                )

            # Need to ensure we have a per particle SED
            if sed is not None:
                if sed.lnu.shape[0] != stars.nparticles:
                    raise exceptions.InconsistentArguments(
                        "The shape of the SED array:",
                        sed.lnu.shape,
                        "does not agree with the number of stellar particles "
                        "(%d)" % stars.nparticles,
                    )

        # Missing cosmology
        if spatial_unit.same_dimensions_as(arcsec) and cosmo is None:
            raise exceptions.InconsistentArguments(
                "When working in an angular unit system a cosmology object"
                " must be given."
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
                if (
                    centre[0] < np.min(pos[:, 0])
                    or centre[0] > np.max(pos[:, 0])
                    or centre[1] < np.min(pos[:, 1])
                    or centre[1] > np.max(pos[:, 1])
                    or centre[2] < np.min(pos[:, 2])
                    or centre[2] > np.max(pos[:, 2])
                ):
                    raise exceptions.InconsistentCoordinates(
                        "The centre lies outside of the coordinate range. "
                        "Are they already centred?"
                    )

        # Need to ensure we have a per particle SED
        if sed is not None and positions is not None:
            if sed.lnu.shape[0] != positions.shape[0]:
                raise exceptions.InconsistentArguments(
                    "The shape of the SED array:",
                    sed.lnu.shape,
                    "does not agree with the number of coordinates "
                    "(%d)" % positions.shape[0],
                )

    def _centre_coords(self):
        """
        Centre coordinates on the geometric mean or the user provided centre.

        TODO: Fix angular conversion. Need to cleanly handle different cases.
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
        self.pix_pos = np.int32(np.floor(self.coords / self.resolution))

    def _convert_to_img_units(self):
        """
        Convert the passed coordinates (and smoothing lengths) to the scene's
        spatial unit system.
        """

        # TODO: missing redshift information in Scene

        # Is there anything to do here?
        if self.coord_unit == self.spatial_unit:
            return

        # If they are the same dimension do the conversion.
        elif self.spatial_unit.same_dimensions_as(self.coord_unit):
            # First do the coordinates
            self.coords *= self.coord_unit
            self.coords.convert_to_units(self.spatial_unit)
            self.coords = self.coords.value

        # # Otherwise, handle conversion between length and angle
        # elif self.spatial_unit.same_dimensions_as(
        #     arcsec
        # ) and self.stars.coord_units.same_dimensions_as(kpc):

        #     # Convert coordinates from comoving to physical coordinates
        #     self.coords *= 1 / (1 + self.stars.redshift)

        #     # Get the conversion factor for arcsec and kpc at this redshift
        #     arcsec_per_kpc_proper = self.cosmo.arcsec_per_kpc_proper(
        #         self.stars.redshift
        #     ).value

        #     # First we need to convert to kpc
        #     if self.stars.coord_units != kpc:

        #         # First do the coordinates
        #         self.coords *= self.stars.coord_units
        #         self.coords.convert_to_units(kpc)
        #         self.coords = self.coords.value

        #     # Now we can convert to arcsec
        #     self.coords *= arcsec_per_kpc_proper * arcsec

        #     # Finally convert to the image unit system if needed
        #     if self.spatial_unit != arcsec:
        #         self.coords.convert_to_units(self.spatial_unit)

        #     # And strip off the unit
        #     self.coords = self.coords.value

        else:
            raise exceptions.UnimplementedFunctionality(
                "Unrecognised combination of image and coordinate units. Feel "
                "free to raise an issue on Github. Currently supported input"
                " dimensions are length or angle for image units and only "
                "length for coordinate units."
            )

        # And now do the smoothing lengths
        if self.smoothing_lengths is not None:
            # If they are the same dimension do the conversion.
            if self.spatial_unit.same_dimensions_as(self.smooth_unit):
                self.smoothing_lengths *= self.smooth_unit
                self.smoothing_lengths.convert_to_units(self.spatial_unit)
                self.smoothing_lengths = self.smoothing_lengths.value

            # # Otherwise, handle conversion between length and angle
            # elif self.spatial_unit.same_dimensions_as(
            #     arcsec
            # ) and self.stars.coord_units.same_dimensions_as(kpc):

            #     # Convert coordinates from comoving to physical coordinates
            #     self.smoothing_lengths *= 1 / (1 + self.stars.redshift)

            #     # First we need to convert to kpc
            #     if self.stars.coord_units != kpc:
            #         self.smoothing_lengths *= self.stars.coord_units
            #         self.smoothing_lengths.convert_to_units(kpc)
            #         self.smoothing_lengths = self.smoothing_lengths.value

            #     # Now we can convert to arcsec
            #     self.smoothing_lengths *= arcsec_per_kpc_proper * arcsec

            #     # Finally convert to the image unit system if needed
            #     if self.spatial_unit != arcsec:
            #         self.coords.convert_to_units(self.spatial_unit)
            #         self.smoothing_lengths.convert_to_units(self.spatial_unit)

            #     # And strip off the unit
            #     self.smoothing_lengths = self.smoothing_lengths.value
