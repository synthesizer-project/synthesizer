"""A submodule for defining parametric morphologies for use in making images.

This module provides a base class for defining parametric morphologies, and
specific classes for the Sersic profile and point sources. The base class
provides a common interface for defining morphologies, and the specific classes
provide the functionality for the Sersic profile and point sources.

Example usage::

    # Import the module
    from synthesizer import morphology

    # Define a Sersic profile
    sersic = morphology.Sersic(r_eff=10.0, sersic_index=4, ellipticity=0.5)

    # Define a point source
    point_source = morphology.PointSource(offset=[0.0, 0.0])
"""

from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import scipy.special
from unyt import kpc, mas, unyt_array
from unyt.dimensions import angle, length

from synthesizer import exceptions


class MorphologyBase(ABC):
    """A base class holding common methods for parametric morphologies.

    Attributes:
        r_eff_kpc (float): The effective radius in kpc.
        r_eff_mas (float): The effective radius in milliarcseconds.
        sersic_index (float): The Sersic index.
        ellipticity (float): The ellipticity.
        theta (float): The rotation angle.
        cosmo (astropy.cosmology): The cosmology object.
        redshift (float): The redshift.
        model_kpc (astropy.modeling.models.Sersic2D): The Sersic2D model in
            kpc.
        model_mas (astropy.modeling.models.Sersic2D): The Sersic2D model in
    """

    def plot_density_grid(self, resolution, npix):
        """Make a quick density plot.

        Args:
            resolution (float):
                The resolution (in the same units provded to the child class).
            npix (int):
                The number of pixels.
        """
        bins = resolution * np.arange(-npix / 2, npix / 2)

        xx, yy = np.meshgrid(bins, bins)

        img = self.compute_density_grid(xx, yy)

        plt.figure()
        plt.imshow(
            np.log10(img),
            origin="lower",
            interpolation="nearest",
            vmin=-1,
            vmax=2,
        )
        plt.show()

    @abstractmethod
    def compute_density_grid(self, *args):
        """Compute the density grid from coordinate grids.

        This is a place holder method to be overwritten by child classes.
        """
        pass

    def get_density_grid(self, resolution, npix):
        """Get the density grid based on resolution and npix.

        Args:
            resolution (unyt_quantity):
                The resolution of the grid.
            npix (tuple, int):
                The number of pixels in each dimension.
        """
        # Define 1D bin centres of each pixel
        if resolution.units.dimensions == angle:
            res = resolution.to("mas")
        else:
            res = resolution.to("kpc")
        xbin_centres = res.value * np.linspace(
            -npix[0] / 2, npix[0] / 2, npix[0]
        )
        ybin_centres = res.value * np.linspace(
            -npix[1] / 2, npix[1] / 2, npix[1]
        )

        # Convert the 1D grid into 2D grids coordinate grids
        xx, yy = np.meshgrid(xbin_centres, ybin_centres)

        # Extract the density grid from the morphology function
        density_grid = self.compute_density_grid(xx, yy, units=res.units)

        # And normalise it...
        return density_grid / np.sum(density_grid)


class PointSource(MorphologyBase):
    """A class holding a PointSource profile.

    This is a morphology where a single cell of the density grid is populated.

    Attributes:
        cosmo (astropy.cosmology):
            The cosmology object.
        redshift (float):
            The redshift.
        offset_kpc (float):
            The offset of the point source relative to the centre of the
            image in kpc.
    """

    def __init__(
        self,
        offset=np.array([0.0, 0.0]) * kpc,
        cosmo=None,
        redshift=None,
    ):
        """Initialise the morphology.

        Args:
            offset (unyt_array/float):
                The [x,y] offset in angular or physical units from the centre
                of the image. The default (0,0) places the source in the centre
                of the image.
            cosmo (astropy.cosmology.Cosmology):
                astropy cosmology object.
            redshift (float):
                Redshift.

        """
        # Check units of r_eff and convert if necessary
        if isinstance(offset, unyt_array):
            if offset.units.dimensions == length:
                self.offset_kpc = offset.to("kpc").value
            elif offset.units.dimensions == angle:
                self.offset_mas = offset.to("mas").value
            else:
                raise exceptions.IncorrectUnits(
                    "The units of offset must have length or angle dimensions"
                )
        else:
            raise exceptions.MissingUnits(
                "The offset must be provided with units"
            )

        # Associate the cosmology and redshift to this object
        self.cosmo = cosmo
        self.redshift = redshift

        # If cosmology and redshift have been provided we can calculate both
        # models
        if cosmo is not None and redshift is not None:
            # Compute conversion
            kpc_proper_per_mas = (
                self.cosmo.kpc_proper_per_arcmin(redshift).to("kpc/mas").value
            )

            # Calculate one offset from the other depending on what
            # we've been given.
            if self.offset_kpc is not None:
                self.offset_mas = self.offset_kpc / kpc_proper_per_mas
            else:
                self.offset_kpc = self.offset_mas * kpc_proper_per_mas

    def compute_density_grid(self, xx, yy, units=kpc):
        """Compute the density grid.

        This acts as a wrapper to astropy functionality (defined above) which
        only work in units of kpc or milliarcseconds (mas)

        Args:
            xx: array-like (float):
                x values on a 2D grid.
            yy: array-like (float):
                y values on a 2D grid.
            units : unyt.unit
                The units in which the coordinate grids are defined.

        Returns:
            density_grid : np.ndarray
                The density grid produced
        """
        # Create empty density grid
        image = np.zeros((len(xx), len(yy)))

        if units == kpc:
            # find the pixel corresponding to the supplied offset
            i = np.argmin(np.fabs(xx[0] - self.offset_kpc[0]))
            j = np.argmin(np.fabs(yy[:, 0] - self.offset_kpc[1]))
            # set the pixel value to 1.0
            image[i, j] = 1.0
            return image

        elif units == mas:
            # find the pixel corresponding to the supplied offset
            i = np.argmin(np.fabs(xx[0] - self.offset_mas[0]))
            j = np.argmin(np.fabs(yy[:, 0] - self.offset_mas[1]))
            # set the pixel value to 1.0
            image[i, j] = 1.0
            return image
        else:
            raise exceptions.InconsistentArguments(
                "Only kpc and milliarcsecond (mas) units are supported "
                "for morphologies."
            )


class Gaussian2D(MorphologyBase):
    """A class holding a 2-dimensional Gaussian distribution.

    This is a morphology where a 2-dimensional Gaussian density grid is
    populated based on provided x and y values.

    Attributes:
        x_mean: (float):
            The mean of the Gaussian along the x-axis.
        y_mean: (float):
            The mean of the Gaussian along the y-axis.
        stddev_x: (float):
            The standard deviation along the x-axis.
        stddev_y: (float):
            The standard deviation along the y-axis.
        rho: (float):
            The population correlation coefficient between x and y.
    """

    def __init__(self, x_mean, y_mean, stddev_x, stddev_y, rho=0):
        """Initialise the morphology.

        Args:
            x_mean: (float):
                The mean of the Gaussian along the x-axis.
            y_mean: (float):
                The mean of the Gaussian along the y-axis.
            stddev_x: (float):
                The standard deviation along the x-axis.
            stddev_y: (float):
                The standard deviation along the y-axis.
            rho: (float):
                The population correlation coefficient between x and y.
        """
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.stddev_x = stddev_x
        self.stddev_y = stddev_y
        self.rho = rho

    def compute_density_grid(self, x, y, units=None):
        """Compute density grid.

        Args:
            x (unyt_array of float):
                x values on a 2D grid.
            y (unyt_array of float):
                y values on a 2D grid.
            units (unyt.unit):
                The units in which the coordinate grids are defined.
                If None, defaults to kpc.

        Returns:
            g_2d_mat: np.ndarray:
                A 2D array representing the Gaussian density values at each
                (x, y) point.

        Raises:
            ValueError:
                If either x or y is None.
        """
        if units is None:
            units = kpc

        self.x_mean = unyt_array(self.x_mean, units)
        self.y_mean = unyt_array(self.y_mean, units)
        self.stddev_x = unyt_array(self.stddev_x, units)
        self.stddev_y = unyt_array(self.stddev_y, units)

        # Error for x, y = None
        if x is None or y is None:
            raise ValueError("x and y grids must be provided.")

        # Define covariance matrix
        cov_mat = np.array(
            [
                [self.stddev_x**2, (self.rho * self.stddev_x * self.stddev_y)],
                [(self.rho * self.stddev_x * self.stddev_y), self.stddev_y**2],
            ]
        )

        # Invert covariant matrix
        inv_cov = np.linalg.inv(cov_mat)

        # Determinant of covariance matrix
        det_cov = np.linalg.det(cov_mat)

        # Stack position deviation along third axis
        stack = np.dstack((x - self.x_mean, y - self.y_mean))

        # Define coefficient of Gaussian
        coeff = 1 / (2 * np.pi * (np.sqrt(det_cov)))

        # Define exponent of Gaussian
        exp = np.einsum("...k, kl, ...l->...", stack, inv_cov, stack)

        # Calc Gaussian vals
        g_2d_mat = coeff * np.exp(-0.5 * exp)

        return g_2d_mat


class Sersic2D(MorphologyBase):
    """A class holding a 2D Sersic profile.

    Attributes:
        r_eff_kpc (float): The effective radius in kpc.
        r_eff_mas (float): The effective radius in milliarcseconds.
        sersic_index (float): The Sersic index.
        ellipticity (float): The ellipticity.
        theta (float): The rotation angle.
        cosmo (astropy.cosmology): The cosmology object.
        redshift (float): The redshift.
        model_kpc : The 2D Sersic model in kpc.
        model_mas : The 2D Sersic model in milliarcseconds.
    """

    def __init__(
        self,
        r_eff,
        amplitude=1,
        sersic_index=1,
        x_0=0,
        y_0=0,
        theta=0,
        ellipticity=0,
        cosmo=None,
        redshift=None,
    ):
        """Initialise the morphology.

        Args:
            r_eff (unyt_array of float):
                Effective radius. This is converted as required.
            amplitude (float):
                Surface brightness at r_eff.
            sersic_index (float):
                Sersic index.
            x_0 (unyt_quantity of float):
                x offset from the centre of the image.
            y_0 (unyt_quantity of float):
                y offset from the centre of the image.
            ellipticity (float):
                Ellipticity.
            theta (float):
                Theta, the rotation angle.
            cosmo (astro.cosmology.Cosmology):
                astropy cosmology object.
            redshift (float):
                Redshift.

        """
        self.r_eff_mas = None
        self.r_eff_kpc = None

        # Check units of r_eff and convert if necessary.
        if isinstance(r_eff, unyt_array):
            if r_eff.units.dimensions == length:
                self.r_eff_kpc = r_eff.to("kpc").value
            elif r_eff.units.dimensions == angle:
                self.r_eff_mas = r_eff.to("mas").value
            else:
                raise exceptions.IncorrectUnits(
                    "The units of r_eff must have length or angle dimensions"
                )
            self.r_eff = r_eff
        else:
            raise exceptions.MissingAttribute(
                "The effective radius must be provided"
            )

        self.amplitude = amplitude
        self.r_eff = r_eff
        self.sersic_index = sersic_index
        self.x_0 = x_0
        self.y_0 = y_0
        self.theta = theta
        self.ellipticity = ellipticity

        # Associate the cosmology and redshift to this object
        self.cosmo = cosmo
        self.redshift = redshift

        # Check inputs
        self._check_args()

        # If cosmology and redshift have been provided we can calculate both
        # models
        if cosmo is not None and redshift is not None:
            # Compute conversion
            kpc_proper_per_mas = (
                self.cosmo.kpc_proper_per_arcmin(redshift).to("kpc/mas").value
            )

            # Calculate one effective radius from the other depending on what
            # we've been given.
            if self.r_eff_kpc is not None:
                self.r_eff_mas = self.r_eff_kpc / kpc_proper_per_mas
            else:
                self.r_eff_kpc = self.r_eff_mas * kpc_proper_per_mas

    def _check_args(self):
        """Test the inputs to ensure they are a valid combination."""
        # Ensure at least one effective radius has been passed
        if self.r_eff_kpc is None and self.r_eff_mas is None:
            raise exceptions.InconsistentArguments(
                "An effective radius must be defined in either kpc (r_eff_kpc)"
                "or milliarcseconds (mas)"
            )

        # Ensure cosmo has been provided if redshift has been passed
        if self.redshift is not None and self.cosmo is None:
            raise exceptions.InconsistentArguments(
                "Astropy.cosmology object is missing, cannot perform "
                "comoslogical calculations."
            )

    def compute_density_grid(self, x, y, units=kpc):
        """Compute the density grid.

        Args:
            x: array-like (float):
                x values on a 2D grid.
            y: array-like (float):
                y values on a 2D grid.
            units : unyt.unit
                The units in which the coordinate grids are defined.

        Returns:
            density_grid : np.ndarray
                The density grid produced from either
                the kpc or mas Sersic profile.
        """
        # Error for x, y = None
        if x is None or y is None:
            raise ValueError("x and y grids must be provided.")

        # Compute coordinate offset from x, y axes
        a = (x - self.x_0) * np.cos(self.theta) + (y - self.y_0) * np.sin(
            self.theta
        )

        b = -(x - self.x_0) * np.sin(self.theta) + (y - self.y_0) * np.cos(
            self.theta
        )

        # Compute radius from adjusted x, y coordinates
        radius = np.sqrt(a**2 + (b / (1 - self.ellipticity)) ** 2)

        # Define coefficient of Sersic profile from Sersic index
        b_n = scipy.special.gammaincinv(2 * self.sersic_index, 0.5)

        # Compute kpc model
        if self.r_eff_kpc is not None:
            self.model_kpc = self.amplitude * np.exp(
                -b_n * (radius / self.r_eff_kpc) ** (1 / self.sersic_index) - 1
            )
        else:
            self.model_kpc = None

        # Compute mas model
        if self.r_eff_mas is not None:
            self.model_mas = self.amplitude * np.exp(
                -b_n * (radius / self.r_eff_mas) ** (1 / self.sersic_index) - 1
            )
        else:
            self.model_mas = None

        if units == kpc and self.model_kpc is None:
            raise exceptions.InconsistentArguments(
                "Morphology has not been initialised with a kpc method. "
                "Reinitialise the model or use milliarcseconds."
            )
        elif units == mas and self.model_mas is None:
            raise exceptions.InconsistentArguments(
                "Morphology has not been initialised with a milliarcsecond "
                "method. Reinitialise the model or use kpc."
            )

        # Call the appropriate model function
        if units == kpc:
            return self.model_kpc
        elif units == mas:
            return self.model_mas
        else:
            raise exceptions.InconsistentArguments(
                "Only kpc and milliarcsecond (mas) units are supported "
                "for morphologies."
            )
