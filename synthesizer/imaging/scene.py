""" Definitions for image scene objects

These should not be explictly used by the user. Instead the user interfaces with
Images and Galaxys
"""
import math
import numpy as np
import unyt
from unyt import arcsec, kpc
from scipy.ndimage import zoom

from synthesizer.particle import Stars
import synthesizer.exceptions as exceptions
from synthesizer.units import Quantity


class Scene:
    """
    The parent class for all "images" containing all information related to the
    "scene" being imaged.

    Attributes:
        resolution (Quantity, float)
            The size a pixel.
        npix (int)
            The number of pixels along an axis of the image or number of spaxels
            in the image plane of the IFU.
        fov (Quantity, float)
            The width of the image/ifu. If coordinates are being used to make the
            image this should have the same units as those coordinates.
        sed (Sed)
            An sed object containing the spectra for this observation.
        orig_resolution (Quantity, float)
            The original resolution (only used when images are resampled).
        orig_npix (int)
            The original npic (only used when images are resampled).
        cosmo (astropy.cosmology)
            The Astropy object containing the cosmological model.
        redshift (float)
            The redshift of the observation.
    """

    # Define slots to reduce memory overhead of this class and limit the
    # possible attributes.
    __slots__ = [
        "_resolution",
        "_fov",
        "npix",
        "sed",
        "_orig_resolution",
        "orig_npix",
        "cosmo",
        "redshift",
    ]

    # Define quantities
    resolution = Quantity()
    fov = Quantity()
    orig_resolution = Quantity()

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
        Intialise the Scene.

        Args:
            resolution (unyt_quantity)
                The size a pixel.
            npix (int)
                The number of pixels along an axis of the image or number of
                spaxels in the image plane of the IFU.
            fov (unyt_quantity)
                The width of the image/ifu. If coordinates are being used to make
                the image this should have the same units as those coordinates.
            sed (Sed)
                An sed object containing the spectra for this observation.
            rest_frame (bool)
                Is the observation in the rest frame or observer frame. Default
                is rest frame (True).
            cosmo (astropy.cosmology)
                The Astropy object containing the cosmological model.
            redshift (float)
                The redshift of the observation.

        Raises:
            InconsistentArguments
                Errors when an incorrect combination of arguments is passed.
        """

        # Check what we've been given
        self._check_scene_args(resolution, fov, npix)

        # Scene resolution, width and pixel information
        self.resolution = resolution.value
        self.fov = fov
        self.npix = npix

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
        Compute the number of pixels in the FOV, ensuring the FOV is an
        integer number of pixels.
        There are multiple ways to define the dimensions of an image, this
        handles the case where the resolution and FOV is given.
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
        There are multiple ways to define the dimensions of an image, this
        handles the case where the resolution and number of pixels is given.
        """

        # Redefine the FOV based on npix
        self.fov = self.resolution * self.npix

    def _resample(self, factor):
        """
        Helper function to resample all images contained within this instance
        by the stated factor using interpolation.

        Args:
            factor (float)
                The factor by which to resample the image, >1 increases
                resolution, <1 decreases resolution.
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

        # Resample the images
        self._resample(factor)


class ParticleScene(Scene):
    """
    The parent class for all "images" of particles, containing all information
    related to the "scene" being imaged.

    Attributes:
        coordinates (Quantity, array-like, float)
            The position of particles to be sorted into the image.
        centre (Quantity, array-like, float)
            The coordinates around which the image will be centered.
        pix_pos (array-like, float)
            The integer coordinates of particles in pixel units.
        npart (int)
            The number of stellar particles.
        smoothing_lengths (Quantity, array-like, float)
            The smoothing lengths describing each particles SPH kernel.
        kernel (array-like, float)
            The values from one of the kernels from the kernel_functions module.
            Only used for smoothed images.
        kernel_dim (int)
            The number of elements in the kernel.
        kernel_threshold (float)
            The kernel's impact parameter threshold (by default 1).

    Raises:
        InconsistentArguments
            If an incompatible combination of arguments is provided an error is
            raised.
    """

    # Define slots to reduce memory overhead of this class
    __slots__ = [
        "_coordinates",
        "_centre",
        "pix_pos",
        "npart",
        "_smoothing_lengths",
        "kernel",
        "kernel_threshold",
        "kernel_dim",
    ]

    # Define quantities
    coordinates = Quantity()
    centre = Quantity()
    smoothing_lengths = Quantity()

    def __init__(
        self,
        resolution,
        npix=None,
        fov=None,
        sed=None,
        coordinates=None,
        smoothing_lengths=None,
        centre=None,
        rest_frame=True,
        cosmo=None,
        redshift=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Intialise the ParticleScene.

        Args:
            resolution (float)
                The size a pixel.
            npix (int)
                The number of pixels along an axis of the image or number of
                spaxels in the image plane of the IFU.
            fov (float)
                The width of the image/ifu. If coordinates are being used to make
                the image this should have the same units as those coordinates.
            sed (Sed)
                An sed object containing the spectra for this observation.
            coordinates (array-like, float)
                The position of particles to be sorted into the image.
            smoothing_lengths (array-like, float)
                The values describing the size of the smooth kernel for each
                particle. Only needed if star objects are not passed.
            centre (array-like, float)
                The coordinates around which the image will be centered. The if one
                is not provided then the geometric centre is calculated and used.
            rest_frame (bool)
                Is the observation in the rest frame or observer frame. Default
                is rest frame (True).
            cosmo (astropy.cosmology)
                The Astropy object containing the cosmological model.
            redshift (float)
                The redshift of the observation.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Raises:
            InconsistentArguments
                If an incompatible combination of arguments is provided an error is
                raised.
        """

        # Check what we've been given
        self._check_part_args(
            resolution, coordinates, centre, cosmo, sed, kernel, smoothing_lengths
        )

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

        # Handle the particle coordinates, here we make a copy to avoid changing
        # the original values
        self.coordinates = np.copy(coordinates)

        # If the coordinates are not already centred centre them
        self.centre = centre
        self._centre_coordinates()

        # Store the smoothing lengths because we again need a copy
        if smoothing_lengths is not None:
            self.smoothing_lengths = np.copy(smoothing_lengths)
        else:
            self.smoothing_lengths = None

        # Shift coordinates to start at 0
        self.coordinates += self.fov / 2

        # Calculate the position of particles in pixel coordinates
        self.pix_pos = np.zeros(self.coordinates.shape, dtype=np.int32)
        self._get_pixel_pos()

        # How many particle are there?
        self.npart = self.coordinates.shape[0]

        # Set up the kernel attributes we need
        if kernel is not None:
            self.kernel = kernel
            self.kernel_dim = kernel.size
            self.kernel_threshold = kernel_threshold
        else:
            self.kernel = None
            self.kernel_dim = None
            self.kernel_threshold = None

    def _check_part_args(
        self, resolution, coordinates, centre, cosmo, sed, kernel, smoothing_lengths
    ):
        """
        Ensures we have a valid combination of inputs.

        Args:
            resolution (float)
                The size a pixel.
            coordinates (array-like, float)
                The position of particles to be sorted into the image.
            centre (array-like, float)
                The coordinates around which the image will be centered. The if one
                is not provided then the geometric centre is calculated and used.
            cosmo (astropy.cosmology)
                The Astropy object containing the cosmological model.
            sed (Sed)
                An sed object containing the spectra for this observation.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            smoothing_lengths (array-like, float)
                The values describing the size of the smooth kernel for each
                particle. Only needed if star objects are not passed.

        Raises:
            InconsistentArguments
               Errors when an incorrect combination of arguments is passed.
            InconsistentCoordinates
               If the centre does not lie within the range of coordinates an error
               is raised.
        """

        # Get the spatial units
        spatial_unit = resolution.units

        # Have we been given an integrated SED by accident?
        if sed is not None:
            if len(sed.lnu.shape) == 1:
                raise exceptions.InconsistentArguments(
                    "Particle Spectra are required for imaging, an integrated "
                    "spectra has been passed."
                )

        # If we are working in terms of angles we need redshifts for the
        # particles.
        if spatial_unit.same_dimensions_as(arcsec) and self.redshift is None:
            raise exceptions.InconsistentArguments(
                "When working in an angular unit system the provided "
                "particles need a redshift associated to them. Particles.redshift"
                " can either be a single redshift for all particles or an "
                "array of redshifts for each star."
            )

        # Need to ensure we have a per particle SED
        if sed is not None:
            if sed.lnu.shape[0] != coordinates.shape[0]:
                raise exceptions.InconsistentArguments(
                    "The shape of the SED array:",
                    sed.lnu.shape,
                    "does not agree with the number of stellar particles "
                    "(%d)" % coordinates.shape[0],
                )

        # Missing cosmology
        if spatial_unit.same_dimensions_as(arcsec) and cosmo is None:
            raise exceptions.InconsistentArguments(
                "When working in an angular unit system a cosmology object"
                " must be given."
            )
        # The passed centre does not lie within the range of coordinates
        if centre is not None:
            if (
                centre[0] < np.min(coordinates[:, 0])
                or centre[0] > np.max(coordinates[:, 0])
                or centre[1] < np.min(coordinates[:, 1])
                or centre[1] > np.max(coordinates[:, 1])
                or centre[2] < np.min(coordinates[:, 2])
                or centre[2] > np.max(coordinates[:, 2])
            ):
                raise exceptions.InconsistentCoordinates(
                    "The centre lies outside of the coordinate range. "
                    "Are they already centred?"
                )

        # Need to ensure we have a per particle SED
        if sed is not None:
            if sed.lnu.shape[0] != coordinates.shape[0]:
                raise exceptions.InconsistentArguments(
                    "The shape of the SED array:",
                    sed.lnu.shape,
                    "does not agree with the number of coordinates "
                    "(%d)" % coordinates.shape[0],
                )

        # Ensure we aren't trying to smooth particles without smoothing lengths
        if kernel is not None and smoothing_lengths is None:
            raise exceptions.InconsistentArguments(
                "Trying to smooth particles which don't have smoothing lengths!"
            )

    def _centre_coordinates(self):
        """
        Centre coordinates on the geometric mean or the user provided centre.

        TODO: Fix angular conversion. Need to cleanly handle different cases.
        """

        # Calculate the centre if necessary
        if self.centre is None:
            self.centre = np.mean(self._coordinates, axis=0)

        # Centre the coordinates
        self.coordinates -= self._centre

    def _get_pixel_pos(self):
        """
        Convert particle coordinates to interger pixel coordinates.
        These later help speed up sorting particles into pixels since their
        index is precomputed here.
        """

        # Convert sim coordinates to pixel coordinates
        self.pix_pos = np.int32(np.floor(self._coordinates / self._resolution))
