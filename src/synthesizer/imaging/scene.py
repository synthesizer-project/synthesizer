""" Definitions for image scene objects

These should not be explictly used by the user. Instead the user
interfaces with Images and Galaxys
"""
import numpy as np
from unyt import arcsec

import synthesizer.exceptions as exceptions
from synthesizer.units import Quantity


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
            The values from one of the kernels from the kernel_functions
            module. Only used for smoothed images.
        kernel_dim (int)
            The number of elements in the kernel.
        kernel_threshold (float)
            The kernel's impact parameter threshold (by default 1).

    Raises:
        InconsistentArguments
            If an incompatible combination of arguments is provided an error is
            raised.
    """

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
                The width of the image/ifu. If coordinates are being used to
                make the image this should have the same units as those
                coordinates.
            sed (Sed)
                An sed object containing the spectra for this observation.
            coordinates (array-like, float)
                The position of particles to be sorted into the image.
            smoothing_lengths (array-like, float)
                The values describing the size of the smooth kernel for each
                particle. Only needed if star objects are not passed.
            centre (array-like, float)
                The coordinates around which the image will be centered. The
                if one is not provided then the geometric centre is calculated
                and used.
            rest_frame (bool)
                Is the observation in the rest frame or observer frame. Default
                is rest frame (True).
            cosmo (astropy.cosmology)
                The Astropy object containing the cosmological model.
            redshift (float)
                The redshift of the observation.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Raises:
            InconsistentArguments
                If an incompatible combination of arguments is provided an
                error is raised.
        """

        # Check what we've been given
        self._check_part_args(
            resolution,
            coordinates,
            centre,
            cosmo,
            sed,
            kernel,
            smoothing_lengths,
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

        # Handle the particle coordinates, here we make a copy to
        # avoid changing the original values
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
        self.pix_pos = np.zeros(self._coordinates.shape, dtype=np.int32)
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
        self,
        resolution,
        coordinates,
        centre,
        cosmo,
        sed,
        kernel,
        smoothing_lengths,
    ):
        """
        Ensures we have a valid combination of inputs.

        Args:
            resolution (float)
                The size a pixel.
            coordinates (array-like, float)
                The position of particles to be sorted into the image.
            centre (array-like, float)
                The coordinates around which the image will be centered. If
                one is not provided then the geometric centre is
                calculated and used.
            cosmo (astropy.cosmology)
                The Astropy object containing the cosmological model.
            sed (Sed)
                An sed object containing the spectra for this observation.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            smoothing_lengths (array-like, float)
                The values describing the size of the smooth kernel for each
                particle. Only needed if star objects are not passed.

        Raises:
            InconsistentArguments
               Errors when an incorrect combination of arguments is passed.
            InconsistentCoordinates
               If the centre does not lie within the range of coordinates an
               error is raised.
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
                "particles need a redshift associated to them. "
                "Particles.redshift can either be a single redshift "
                "for all particles or an array of redshifts for each star."
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
                "Trying to smooth particles which "
                "don't have smoothing lengths!"
            )

    def _centre_coordinates(self):
        """
        Centre coordinates on the geometric mean or the user provided centre.

        TODO: Fix angular conversion. Need to cleanly handle different cases.
        """

        # Calculate the centre if necessary
        if self.centre is None:
            self.centre = np.mean(self.coordinates, axis=0)

        # Centre the coordinates
        self.coordinates -= self.centre

    def _get_pixel_pos(self):
        """
        Convert particle coordinates to interger pixel coordinates.
        These later help speed up sorting particles into pixels since their
        index is precomputed here.
        """

        # Convert sim coordinates to pixel coordinates
        self.pix_pos = np.int32(np.floor(self._coordinates / self._resolution))
