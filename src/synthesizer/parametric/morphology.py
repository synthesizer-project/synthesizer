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
from synthesizer.units import accepts, unit_is_compatible
from synthesizer.utils import TableFormatter


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
        grid (astropy.modeling.models.Sersic2D): The Sersic2D model in
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

    @accepts(resolution=(kpc, mas))
    def get_density_grid(self, resolution, npix, **kwargs):
        """Get the density grid based on resolution and npix.

        Args:
            resolution (unyt_quantity):
                The resolution of the grid.
            npix (tuple, int):
                The number of pixels in each dimension.
            **kwargs:
                Additional keyword arguments to pass to the
                compute_density_grid method.
        """
        # Define 1D bin centres of each pixel
        xbin_centres = resolution.value * np.linspace(
            -npix[0] / 2, npix[0] / 2, npix[0]
        )
        ybin_centres = resolution.value * np.linspace(
            -npix[1] / 2, npix[1] / 2, npix[1]
        )

        # Convert the 1D grid into 2D grids coordinate grids
        xx, yy = np.meshgrid(xbin_centres, ybin_centres) * resolution.units

        # Extract the density grid from the morphology function
        density_grid, norm = self.compute_density_grid(xx, yy, **kwargs)

        # And normalise it...
        return density_grid / norm

    def __str__(self):
        """Return a summary of the morphology.

        Returns:
            str: A string representation of the morphology.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table(self.__class__.__name__)


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

    @accepts(offset=(kpc, mas))
    def __init__(
        self,
        offset=np.array([0.0, 0.0]) * kpc,
        cosmo=None,
        redshift=None,
    ):
        """Initialise the morphology.

        If a cosmology and redshift are provided, the offset will be
        converted to both kpc and milliarcseconds (mas) for use in the
        compute_density_grid method. If only one is provided then the
        inputs to compute_density_grid must match the units of the

        Args:
            offset (unyt_array/float):
                The [x,y] offset in angular or physical units from the centre
                of the image. The default (0,0) places the source in the centre
                of the image.
            cosmo (astropy.cosmology.Cosmology):
                The cosmology object. Only required for conversions that make
                both kpc and mas models available.
            redshift (float):
                The redshift of the source. Only required for conversions that
                make both kpc and mas models available.

        """
        # Store offset in kpc or mas
        self.offset_kpc = None
        self.offset_mas = None
        if offset.units.dimensions == length:
            self.offset_kpc = offset
        elif offset.units.dimensions == angle:
            self.offset_mas = offset

        # Associate the cosmology and redshift to this object
        self.cosmo = cosmo
        self.redshift = redshift

        # If cosmology and redshift have been provided we can calculate both
        # models
        if cosmo is not None and redshift is not None:
            # Compute conversion
            kpc_proper_per_mas = (
                self.cosmo.kpc_proper_per_arcmin(redshift).to("kpc/mas").value
                * kpc
                / mas
            )

            # Calculate one offset from the other depending on what
            # we've been given.
            if self.offset_kpc is not None:
                self.offset_mas = self.offset_kpc / kpc_proper_per_mas
            else:
                self.offset_kpc = self.offset_mas * kpc_proper_per_mas

    @accepts(xx=(kpc, mas), yy=(kpc, mas))
    def compute_density_grid(self, xx, yy):
        """Compute the density grid.

        This acts as a wrapper to astropy functionality (defined above) which
        only work in units of kpc or milliarcseconds (mas)

        Args:
            xx: array-like (unyt_array of float):
                x values on a 2D grid.
            yy: array-like (unyt_array of float):
                y values on a 2D grid.

        Returns:
            np.ndarray:
                The density grid produced
            float:
                The normalisation factor for the density grid.
        """
        # Create empty density grid
        image = np.zeros((len(xx), len(yy)))

        # Get the units of the coordinate grids
        units = xx.units
        if not unit_is_compatible(yy, units):
            raise exceptions.InconsistentUnits(
                f" xx and yy in incompatible units: {xx.units} and {yy.units}"
            )

        if units == kpc and self.offset_kpc is not None:
            # find the pixel corresponding to the supplied offset
            i = np.argmin(np.fabs(xx[0] - self.offset_kpc[0]))
            j = np.argmin(np.fabs(yy[:, 0] - self.offset_kpc[1]))
            # set the pixel value to 1.0
            image[i, j] = 1.0
            return image, 1.0
        elif units == kpc and self.offset_kpc is None:
            raise exceptions.InconsistentArguments(
                "A kpc offset must be provided to the PointSource "
                "morphology if the coordinate grids are in kpc units."
            )

        elif units == mas and self.offset_mas is not None:
            # find the pixel corresponding to the supplied offset
            i = np.argmin(np.fabs(xx[0] - self.offset_mas[0]))
            j = np.argmin(np.fabs(yy[:, 0] - self.offset_mas[1]))
            # set the pixel value to 1.0
            image[i, j] = 1.0
            return image, 1.0
        elif units == mas and self.offset_mas is None:
            raise exceptions.InconsistentArguments(
                "A mas offset must be provided to the PointSource "
                "morphology if the coordinate grids are in mas units."
            )

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

    @accepts(
        x_mean=(kpc, mas),
        y_mean=(kpc, mas),
        stddev_x=(kpc, mas),
        stddev_y=(kpc, mas),
    )
    def __init__(self, x_mean, y_mean, stddev_x, stddev_y, rho=0):
        """Initialise the morphology.

        Args:
            x_mean (unyt_quantity of float): The mean of the Gaussian along
                the x-axis.
            y_mean (unyt_quantity of float): The mean of the Gaussian along
                the y-axis.
            stddev_x (unyt_quantity of float): The standard deviation along
                the x-axis.
            stddev_y (unyt_quantity of float): The standard deviation along
                the y-axis.
            rho: (float):
                The population correlation coefficient between x and y.
        """
        self.x_mean = x_mean
        self.y_mean = y_mean
        self.stddev_x = stddev_x
        self.stddev_y = stddev_y
        self.rho = rho

    @accepts(x=(kpc, mas), y=(kpc, mas))
    def compute_density_grid(self, x, y):
        """Compute density grid.

        Args:
            x (unyt_array of float):
                x values on a 2D grid.
            y (unyt_array of float):
                y values on a 2D grid.

        Returns:
            np.ndarray:
                A 2D array representing the Gaussian density values at each
                (x, y) point.
            float:
                The normalisation factor for the Gaussian density grid.

        Raises:
            ValueError:
                If either x or y is None.
        """
        # Define covariance matrix
        cov_mat = unyt_array(
            [
                [self.stddev_x**2, (self.rho * self.stddev_x * self.stddev_y)],
                [(self.rho * self.stddev_x * self.stddev_y), self.stddev_y**2],
            ],
            self.stddev_x.units,
        )

        # Ensure x and y are in compatiable units with x_mean and y_mean
        if not unit_is_compatible(x, self.x_mean.units):
            raise exceptions.InconsistentUnits(
                f"x units ({x.units}) must be compatible with "
                f"x_mean units ({self.x_mean.units})"
            )
        if not unit_is_compatible(y, self.y_mean.units):
            raise exceptions.InconsistentUnits(
                f"y units ({y.units}) must be compatible with "
                f"y_mean units ({self.y_mean.units})"
            )

        # Invert covariant matrix
        inv_cov = np.linalg.inv(cov_mat)

        # Determinant of covariance matrix
        det_cov = np.linalg.det(cov_mat)

        # Get the relative position of each pixel in the grid
        dx = x - self.x_mean
        dy = y - self.y_mean

        # Stack position deviation along third axis
        stack = np.dstack((dx, dy))

        # Define coefficient of Gaussian
        coeff = 1 / (2 * np.pi * (np.sqrt(det_cov)))

        # Define exponent of Gaussian (need to remove units to avoid incorrect
        # error!)
        exp = np.einsum(
            "...k, kl, ...l->...",
            stack.value,
            inv_cov.value,
            stack.value,
        )

        # Calc Gaussian vals
        g_2d_mat = coeff * np.exp(-0.5 * exp)

        return g_2d_mat.value, np.sum(g_2d_mat.value)


class Gaussian2DAnnuli(Gaussian2D):
    """A subclass of Gaussian2D that supports masking of concentric annuli.

    Attributes:
        x_mean: (float): The mean of the Gaussian along the x-axis.
        y_mean: (float): The mean of the Gaussian along the y-axis.
        stddev_x: (float): The standard deviation along the x-axis.
        stddev_y: (float): The standard deviation along the y-axis.
        rho: (float): The population correlation coefficient between x and y.
        radii (list of float): The radii defining the annuli.
        annulus_index (int): Index of the annulus to be used.
    """

    @accepts(
        x_mean=(kpc, mas),
        y_mean=(kpc, mas),
        stddev_x=(kpc, mas),
        stddev_y=(kpc, mas),
        radii=(kpc, mas),
    )
    def __init__(
        self,
        x_mean,
        y_mean,
        stddev_x,
        stddev_y,
        radii,
        rho=0,
    ):
        """Initialise the Gaussian morphology with optional annulus masking.

        Args:
            x_mean (unyt_quantity of float): The mean of the Gaussian along
                the x-axis.
            y_mean (unyt_quantity of float): The mean of the Gaussian along
                the y-axis.
            stddev_x (unyt_quantity of float): The standard deviation along
                the x-axis.
            stddev_y (unyt_quantity of float): The standard deviation along
                the y-axis.
            radii (unyt_array of float): The radii defining the annuli.
            rho (float): The correlation coefficient between x and y.
        """
        # Initialise the parent class
        Gaussian2D.__init__(self, x_mean, y_mean, stddev_x, stddev_y, rho)

        # Ensure x_mean and y_mean are in compatible units with radii
        if not unit_is_compatible(radii, self.x_mean.units):
            raise exceptions.InconsistentUnits(
                f"radii units ({radii.units}) must be compatible with "
                f"x_mean units ({self.x_mean.units})"
            )
        if not unit_is_compatible(radii, self.y_mean.units):
            raise exceptions.InconsistentUnits(
                f"radii units ({radii.units}) must be compatible with "
                f"y_mean units ({self.y_mean.units})"
            )

        # Attach the radii for annuli
        self.radii = radii

        # Add an infinite outer radius for the last annulus (this is safe
        # if the user has already defined the last radius as infinity, we
        # will just never touch this new entry)
        self.radii = np.append(self.radii, np.inf * radii.units)

        # How many annuli are there?
        self.n_annuli = len(radii)

    @accepts(x=(kpc, mas), y=(kpc, mas))
    def compute_density_grid(self, x, y, annulus):
        """Compute the Gaussian density grid with optional annulus masking.

        Args:
            x (array-like): x values on a 2D grid.
            y (array-like): y values on a 2D grid.
            annulus (int): Index of the annulus to be used.

        Returns:
            np.ndarray: The masked Gaussian density grid.
            float: The normalisation factor for the density grid.
        """
        # Ensure the annulus index is valid
        if annulus < 0 or annulus >= self.n_annuli:
            raise exceptions.InconsistentArguments(
                f"Invalid annulus index: {annulus}. "
                f"Must be between 0 and {self.n_annuli - 2}."
            )

        # Get the whole density grid first
        density_grid, norm = super().compute_density_grid(x, y)

        # Compute elliptical radius from (x, y)
        dx = x - self.x_mean
        dy = y - self.y_mean
        radius = np.sqrt(dx**2 + dy**2)

        # Get the inner and outer radii for the annulus
        inner_radius = self.radii[annulus]
        outer_radius = self.radii[annulus + 1]

        # Create a mask for the annulus
        mask = (radius >= inner_radius) & (radius < outer_radius)
        density_grid = np.where(mask, density_grid, 0)

        return density_grid, norm


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
        grid : The 2D Sersic model in kpc.
        model_mas : The 2D Sersic model in milliarcseconds.
    """

    @accepts(
        r_eff=(kpc, mas),
        x_0=(kpc, mas),
        y_0=(kpc, mas),
    )
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

        If a cosmology and redshift are provided, the effective radius
        will be converted to both kpc and milliarcseconds (mas) for use in the
        compute_density_grid method. If only one is provided then the
        inputs to compute_density_grid must match the units of the
        effective radius.

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
                Cosmology object for conversions between kpc and mas.
                Only required if both kpc and mas models are needed.
            redshift (float):
                Redshift.

        """
        self.r_eff_mas = None
        self.r_eff_kpc = None

        # Check units of r_eff and convert if necessary.
        if r_eff.units.dimensions == length:
            self.r_eff_kpc = r_eff
        elif r_eff.units.dimensions == angle:
            self.r_eff_mas = r_eff
        self.r_eff = r_eff

        # Ensure r_eff and x_0, y_0 are in compatible units
        if not unit_is_compatible(x_0, self.r_eff.units):
            raise exceptions.InconsistentUnits(
                f"x_0 units ({x_0.units}) must be compatible with "
                f"r_eff units ({self.r_eff.units})"
            )
        if not unit_is_compatible(y_0, self.r_eff.units):
            raise exceptions.InconsistentUnits(
                f"y_0 units ({y_0.units}) must be compatible with "
                f"r_eff units ({self.r_eff.units})"
            )

        # Set the other parameters
        self.amplitude = amplitude
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
                * kpc
                / mas
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
                " or milliarcseconds (mas)"
            )

        # Ensure cosmo has been provided if redshift has been passed
        if self.redshift is not None and self.cosmo is None:
            raise exceptions.InconsistentArguments(
                "Astropy.cosmology object is missing, cannot perform "
                "comoslogical calculations."
            )

    @accepts(x=(kpc, mas), y=(kpc, mas))
    def compute_density_grid(self, x, y):
        """Compute the density grid.

        Args:
            x: array-like (float):
                x values on a 2D grid.
            y: array-like (float):
                y values on a 2D grid.

        Returns:
            np.ndarray:
                The density grid produced from either
                the kpc or mas Sersic profile.
            float:
                The normalisation factor for the density grid.
        """
        # Ensure x and y are in compatible units with x_0 and y_0
        if not unit_is_compatible(x, self.x_0.units):
            raise exceptions.InconsistentUnits(
                f"x units ({x.units}) must be compatible with "
                f"x_0 units ({self.x_0.units})"
            )
        if not unit_is_compatible(y, self.y_0.units):
            raise exceptions.InconsistentUnits(
                f"y units ({y.units}) must be compatible with "
                f"y_0 units ({self.y_0.units})"
            )

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

        # Compute and return the Sersic profile based on the radius
        if radius.units == kpc and self.r_eff_kpc is not None:
            grid = self.amplitude * np.exp(
                -b_n * (radius / self.r_eff_kpc) ** (1 / self.sersic_index) - 1
            )
        elif radius.units == mas and self.r_eff_mas is not None:
            grid = self.amplitude * np.exp(
                -b_n * (radius / self.r_eff_mas) ** (1 / self.sersic_index) - 1
            )
        elif radius.units == kpc and self.r_eff_kpc is None:
            raise exceptions.InconsistentArguments(
                "A kpc effective radius must be provided to the Sersic2D "
                "morphology if the coordinate grids are in kpc units."
            )
        elif radius.units == mas and self.r_eff_mas is None:
            raise exceptions.InconsistentArguments(
                "A mas effective radius must be provided to the Sersic2D "
                "morphology if the coordinate grids are in mas units."
            )
        else:
            # Accepts means we will never get here, but just in case
            raise exceptions.InconsistentArguments(
                f"Unrecognised units for radius: {radius.units}."
            )

        return grid, np.sum(grid)


class Sersic2DAnnuli(Sersic2D):
    """A subclass of Sersic2D that supports masking of concentric annuli.

    Attributes:
        r_eff_kpc (float): The effective radius in kpc.
        r_eff_mas (float): The effective radius in milliarcseconds.
        sersic_index (float): The Sersic index.
        ellipticity (float): The ellipticity.
        theta (float): The rotation angle.
        cosmo (astropy.cosmology): The cosmology object.
        redshift (float): The redshift.
        grid : The 2D Sersic model in kpc.
        model_mas : The 2D Sersic model in milliarcseconds.
        radii (list of float): The radii defining the annuli.
        annulus_index (int): Index of the annulus to be used.
    """

    @accepts(
        r_eff=(kpc, mas),
        radii=(kpc, mas),
        x_0=(kpc, mas),
        y_0=(kpc, mas),
    )
    def __init__(
        self,
        r_eff,
        radii,
        amplitude=1,
        sersic_index=1,
        x_0=0,
        y_0=0,
        theta=0,
        ellipticity=0,
        cosmo=None,
        redshift=None,
    ):
        """Initialise the morphology with optional annulus masking.

        Args:
            r_eff (unyt_array of float): Effective radius.
            radii (unyt_array of float): The radii defining the annuli.
            amplitude (float): Surface brightness at r_eff.
            sersic_index (float): Sersic index.
            x_0 (unyt_quantity of float): x centre of the Sersic profile.
            y_0 (unyt_quantity of float): y centre of the Sersic profile.
            theta (float): Inclination angle.
            ellipticity (float): Ellipticity.
            cosmo (astropy.cosmology.Cosmology): astropy cosmology object.
            redshift (float): Redshift.
        """
        Sersic2D.__init__(
            self,
            r_eff,
            amplitude,
            sersic_index,
            x_0,
            y_0,
            theta,
            ellipticity,
            cosmo,
            redshift,
        )

        # Attach the radii for annuli
        self.radii = radii

        # Add an infinite outer radius for the last annulus (this is safe
        # if the user has already defined the last radius as infinity, we
        # will just never touch this new entry)
        self.radii = np.append(self.radii, np.inf * radii.units)

        # How many annuli are there?
        self.n_annuli = len(radii)

    @accepts(x=(kpc, mas), y=(kpc, mas))
    def compute_density_grid(self, x, y, annulus):
        """Compute the density grid with optional annulus masking.

        Args:
            x (array-like): x values on a 2D grid.
            y (array-like): y values on a 2D grid.
            annulus (int): Index of the annulus to be used.

        Returns:
            np.ndarray: The computed density grid, optionally masked by annuli.
            float: The normalisation factor for the density grid.
        """
        # Ensure the annulus index is valid
        if annulus < 0 or annulus >= self.n_annuli:
            raise exceptions.InconsistentArguments(
                f"Invalid annulus index: {annulus}. "
                f"Must be between 0 and {self.n_annuli - 2}."
            )

        # Get the density grid for the whole profile
        density_grid, norm = super().compute_density_grid(x, y)

        # Compute the radius of each grid cell in the full profile.
        a = (x - self.x_0) * np.cos(self.theta) + (y - self.y_0) * np.sin(
            self.theta
        )
        b = -(x - self.x_0) * np.sin(self.theta) + (y - self.y_0) * np.cos(
            self.theta
        )
        radius = np.sqrt(a**2 + (b / (1 - self.ellipticity)) ** 2)

        # Define the inner and outer radius of the annulus
        inner_radius = self.radii[annulus]
        outer_radius = self.radii[annulus + 1]

        # Apply annulus mask
        mask = (radius >= inner_radius) & (radius < outer_radius)
        density_grid = np.where(mask, density_grid, 0)

        return density_grid, norm
