"""A base class to encompass common functionality for all imaging classes.

This class is not intended to be used directly, but rather as a base class for
other imaging classes. It provides a common interface and shared functionality
for all imaging related methods.

It mostly handles the attributes and operations related to the image's/spectral
cube's resolution and dimensions but also provides some useful flagging
properties and methods.
"""

from abc import ABC, abstractmethod

import numpy as np
from unyt import arcsecond, kpc, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.units import Quantity, accepts, unit_is_compatible


class ImagingBase(ABC):
    """A base class to encompass common functionality for all imaging classes.

    This base classes handles the common geometry operations and related
    information. Of particular importance is the abstraction of
    angular vs cartesian unit systems.

    Attributes:
        cart_resolution (unyt_quantity): The spatial resolution of the image
            in Cartesian coordinates. If the image is in angular coordinates,
            this is None.
        ang_resolution (unyt_quantity): The spatial resolution of the image
            in angular coordinates. If the image is in Cartesian coordinates,
            this is None.
        cart_fov (unyt_quantity): The field of view of the image in Cartesian
            coordinates. If the image is in angular coordinates, this is None.
        ang_fov (unyt_quantity): The field of view of the image in angular
            coordinates. If the image is in Cartesian coordinates, this is
            None.
        npix (np.ndarray): The number of pixels in the image. This is a 2D
            array with the number of pixels in each dimension.
        orig_resolution (unyt_quantity): The original resolution of the image
            in the units of the image. This is used to keep track of the
            original resolution when resampling the image.
        orig_npix (np.ndarray): The original number of pixels in the image.
            This is used to keep track of the original number of pixels when
            resampling the image.
        has_cartesian_units (bool): True if the image has Cartesian units,
            False otherwise.
        has_angular_units (bool): True if the image has angular units, False
            otherwise.
        fov (unyt_quantity): The field of view of the image in the units of
            the image. This is a property that returns the field of view in
            the units of the image.
        resolution (unyt_quantity): The resolution of the image in the units
            of the image. This is a property that returns the resolution in
            the units of the image.
    """

    # Define quantities
    cart_resolution = Quantity("spatial")
    ang_resolution = Quantity("angular_resolution")
    cart_fov = Quantity("spatial")
    ang_fov = Quantity("angle")

    @accepts(resolution=(kpc, arcsecond), fov=(kpc, arcsecond))
    def __init__(
        self,
        resolution,
        fov,
    ):
        """Initialize the imaging base class.

        This class is a simple interface to the shared imaging properties
        any imaging class should have. It is not intended to be used directly.

        Args:
            resolution (unyt_quantity):
                The size of a pixel. Either in angular or Cartesian units.
            fov (unyt_quantity/tuple, unyt_quantity):
                The width of the image. If a single value is given then the
                image is assumed to be square.
        """
        # Ensure the fov has an entry for each axis if it doesn't already
        # (e.g. if it is a single value)
        if fov.size == 1:
            fov = unyt_array((fov.value, fov.value), fov.units)

        # Set the imaging quantities based on whether they are angular or
        # Cartesian
        if unit_is_compatible(resolution, kpc):
            self.cart_resolution = resolution
            self.ang_resolution = None
        else:
            self.cart_resolution = None
            self.ang_resolution = resolution
        if unit_is_compatible(fov, kpc):
            self.cart_fov = fov
            self.ang_fov = None
        else:
            self.cart_fov = None
            self.ang_fov = fov

        # Ensure that the resolution and fov are compatible (i.e. both
        # are angular or both are Cartesian)
        if not unit_is_compatible(self.resolution, self.fov.units):
            raise exceptions.InconsistentArguments(
                "The resolution and FOV must be in compatible units. "
                f"Found resolution={self.resolution.units}, "
                f"and fov={self.fov.units}."
            )

        # Compute the number of pixels in the FOV
        self._compute_npix()

        # Keep track of the input resolution and and npix so we can handle
        # super resolution correctly.
        self.orig_resolution = resolution.copy()
        self.orig_npix = self.npix.copy()

    def _compute_npix(self, compute_fov=True):
        """Compute the number of pixels given the resolution and fov.

        Args:
            compute_fov (bool): If True, compute the fov based on the
                resolution and new npix. Defaults to True.
        """
        # Compute how many pixels fall in the FOV
        self.npix = np.int32(self.fov / self.resolution)

        # Ensure that the npix is an array of 2 values
        if self.npix.size == 1:
            self.npix = np.array((self.npix, self.npix), dtype=np.int32)

        # Redefine the FOV based on npix
        if compute_fov:
            self._compute_fov(compute_npix=False)

    def _compute_fov(self, compute_npix=True):
        """Compute the FOV given the resolution and npix.

        Args:
            compute_npix (bool): If True, compute the npix based on the
                resolution and new fov. Defaults to True.
        """
        if self.has_cartesian_units:
            self.cart_fov = self.cart_resolution * self.npix
        else:
            self.ang_fov = self.ang_resolution * self.npix

        # Redefine the npix based on the FOV if requested
        if compute_npix:
            self._compute_npix(compute_fov=False)

    def _compute_resolution(self, compute_fov=True):
        """Compute the resolution given the FOV and npix.

        Args:
            compute_fov (bool): If True, compute the fov based on the
                resolution and new npix. Defaults to True.
        """
        if self.has_cartesian_units:
            self.cart_resolution = self.cart_fov[0] / self.npix[0]
        else:
            self.ang_resolution = self.ang_fov[0] / self.npix[0]

        # Redefine the fov based on npix
        if compute_fov:
            self._compute_fov(compute_npix=False)

    def _resample_resolution(self, factor):
        """Resample the resolution by a given factor.

        Args:
            factor (float): The factor to resample the resolution by.
        """
        # If the factor is 1, do nothing
        if factor == 1:
            return

        # Ensure the factor is a positive number
        if factor <= 0:
            raise exceptions.InconsistentArguments(
                "The factor must be a positive number."
            )

        # Resample the resolution and npix
        if self.has_cartesian_units:
            self.cart_resolution = self.cart_resolution / factor
        else:
            self.ang_resolution = self.ang_resolution / factor

        # Compute the new npix
        self._compute_npix(compute_fov=False)

    def set_resolution(self, resolution):
        """Set the resolution of the image.

        This will also update the FOV and npix to reflect the new resolution.

        Args:
            resolution (unyt_quantity): The new resolution of the image.
        """
        # Ensure we have units
        if not isinstance(resolution, (unyt_array, unyt_quantity)):
            raise exceptions.InconsistentArguments(
                "The resolution must be given with units."
            )

        # Ensure the resolution is compatible with the current fov
        if not unit_is_compatible(resolution, self.fov.units):
            raise exceptions.InconsistentArguments(
                "Expected a resolution in units compatible "
                f"with {self.fov.units}, but got {resolution.units}. To "
                "change the units of the FOV and resolution, make a new "
                "imaging object."
            )

        # Set the resolution
        if unit_is_compatible(resolution, kpc):
            self.cart_resolution = resolution
            self.ang_resolution = None
        else:
            self.cart_resolution = None
            self.ang_resolution = resolution

        # Compute the new npix
        self._compute_npix(compute_fov=True)

    def set_fov(self, fov):
        """Set the field of view of the image.

        This will also update the resolution and npix to reflect the new
        fov.

        Note that this will be modified to ensure an integer number of pixels
        tessellate the image.

        Args:
            fov (unyt_quantity): The new field of view of the image.
        """
        # Ensure we have units
        if not isinstance(fov, (unyt_array, unyt_quantity)):
            raise exceptions.InconsistentArguments(
                "The fov must be given with units."
            )

        # Ensure the fov is compatible with the current resolution
        if not unit_is_compatible(fov, self.resolution.units):
            raise exceptions.InconsistentArguments(
                "Expected a fov in units compatible "
                f"with {self.resolution.units}, but got {fov.units}. To "
                "change the units of the FOV and resolution, make a new "
                "imaging object."
            )

        # Set the fov
        if unit_is_compatible(fov, kpc):
            self.cart_fov = fov
            self.ang_fov = None
        else:
            self.cart_fov = None
            self.ang_fov = fov

        # Compute the new npix
        self._compute_npix(compute_fov=True)

    def set_npix(self, npix):
        """Set the number of pixels in the image.

        This will also update the resolution and fov to reflect the new npix.

        Args:
            npix (int/tuple): The new number of pixels in the image. If a
                single value is given then the image is assumed to be square.
        """
        # Ensure we have a number of pix per axis
        if isinstance(npix, int):
            npix = np.array((npix, npix), dtype=np.int32)
        elif isinstance(npix, tuple):
            if len(npix) != 2:
                raise exceptions.InconsistentArguments(
                    "npix must contain exactly two elements (nx, ny)."
                )
            npix = np.array(npix, dtype=np.int32)
        else:
            raise exceptions.InconsistentArguments(
                "The npix must be given as an int or tuple."
            )

        # Set the npix
        self.npix = npix

        # Update the resolution and fov
        self._compute_resolution(compute_fov=True)

    @property
    def has_angular_units(self):
        """Check if the image has angular units."""
        return self.ang_resolution is not None and self.ang_fov is not None

    @property
    def has_cartesian_units(self):
        """Check if the image has Cartesian units."""
        return self.cart_resolution is not None and self.cart_fov is not None

    @property
    @abstractmethod
    def shape(self):
        """The shape of the image."""
        pass

    @property
    def resolution(self):
        """The resolution of the image."""
        # Return the resolution in the correct units
        if self.has_cartesian_units:
            return self.cart_resolution
        return self.ang_resolution

    @property
    def fov(self):
        """The field of view of the image."""
        # Return the fov in the correct units
        if self.has_cartesian_units:
            return self.cart_fov
        return self.ang_fov

    @property
    def _resolution(self):
        """The resolution of the image without units.

        This alias is used to bring the underlying functionality of the
        angular and Cartesian Quantities to the front end friendly resolution
        property.
        """
        if self.has_cartesian_units:
            return self._cart_resolution
        return self._ang_resolution

    @property
    def _fov(self):
        """The field of view of the image without units.

        This alias is used to bring the underlying functionality of the
        angular and Cartesian Quantities to the front end friendly fov
        property.
        """
        if self.has_cartesian_units:
            return self._cart_fov
        return self._ang_fov
