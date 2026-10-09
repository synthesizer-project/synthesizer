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
from synthesizer.synth_warnings import deprecation, warn
from synthesizer.units import accepts, unit_is_compatible
from synthesizer.utils import TableFormatter

# The number of samples per pixel along each axis used to integrate a profile
# over each pixel (rather than sampling it at the pixel centre)
DEFAULT_SUPERSAMPLE = 5

# The relative tolerance each pixel of a parametric image is integrated to
# (see synthesizer.imaging.extensions.parametric_image)
IMAGE_TOLERANCE = 1e-4

# The outer radius the imaging extension treats as unbounded (it can't use
# infinities, see parametric_image.cpp)
_UNBOUNDED = 1e300


def _make_images(rows, signals, resolution, npix, nthreads=1, tolerance=None):
    """Make the images of populations with the parametric imaging extension.

    Args:
        rows (list of np.ndarray):
            The profile parameters of each population (see
            _get_image_rows), in the units of the resolution.
        signals (np.ndarray):
            The signal of each population, with shape (npop, nsignal).
        resolution (unyt_quantity):
            The resolution of the grid.
        npix (tuple, int):
            The number of pixels in each dimension.
        nthreads (int):
            The number of threads to use.
        tolerance (float):
            The relative tolerance of each pixel's integral, defaults to
            IMAGE_TOLERANCE.

    Returns:
        np.ndarray:
            The images, with shape (npix[1], npix[0], nsignal).
    """
    from synthesizer.imaging.extensions.parametric_image import (
        make_parametric_images,
    )

    signals = np.asarray(signals)
    if signals.dtype not in (np.float32, np.float64):
        signals = signals.astype(np.float64)
    signals = signals.reshape(len(rows), -1)

    # An image with no pixels (e.g. a field of view smaller than a pixel)
    if npix[0] <= 0 or npix[1] <= 0:
        return np.zeros(
            (max(int(npix[1]), 0), max(int(npix[0]), 0), signals.shape[1]),
            dtype=signals.dtype,
        )
    return make_parametric_images(
        np.ascontiguousarray(rows, dtype=np.float64),
        np.ascontiguousarray(signals),
        float(resolution.value),
        int(npix[0]),
        int(npix[1]),
        IMAGE_TOLERANCE if tolerance is None else tolerance,
        nthreads,
    )


def _get_image_rows(morphology, resolution):
    """Get the imaging extension's profile parameters of a morphology.

    Args:
        morphology (MorphologyBase):
            The morphology (a single profile, Annuli or PerPopulation).
        resolution (unyt_quantity):
            The resolution of the grid (whose units the parameters are in).

    Returns:
        list of np.ndarray or None:
            The parameters of each population, or None if the extension
            can't image the morphology (e.g. a custom profile).
    """
    if isinstance(morphology, PerPopulation):
        rows = []
        for morph in morphology.morphologies:
            morph_rows = _get_image_rows(morph, resolution)
            if morph_rows is None:
                return None
            rows.extend(morph_rows)
        return rows
    if isinstance(morphology, Annuli):
        profile_rows = _get_image_rows(morphology.profile, resolution)
        if profile_rows is None or len(profile_rows) != 1:
            return None
        radii = np.minimum(
            morphology.radii.to(resolution.units).value, _UNBOUNDED
        )
        rows = []
        for lo, hi in zip(radii[:-1], radii[1:]):
            row = profile_rows[0].copy()
            row[_RLO], row[_RHI] = lo, hi
            rows.append(row)
        return rows
    get_params = getattr(morphology, "_get_image_params", None)
    if get_params is None:
        return None
    return [get_params(resolution)]


# The indices of the parameters in an imaging extension profile row (see
# parametric_image.cpp)
(
    _TYPE,
    _X0,
    _Y0,
    _COS,
    _SIN,
    _Q,
    _SCALE,
    _INDEX,
    _BN,
    _RLO,
    _RHI,
    _CUSP,
    _NPARAMS,
) = range(13)


def _new_image_row(profile_type, x0, y0, angle=0.0, axis_ratio=1.0):
    """Start an imaging extension profile row.

    Args:
        profile_type (int): 0 (Sersic), 1 (Gaussian) or 2 (point source).
        x0 (float): The x coordinate of the centre.
        y0 (float): The y coordinate of the centre.
        angle (float): The angle of the major axis (radians).
        axis_ratio (float): The minor to major axis ratio.

    Returns:
        np.ndarray: The row, covering the whole profile (no annulus).
    """
    row = np.zeros(_NPARAMS)
    row[_TYPE] = profile_type
    row[_X0], row[_Y0] = x0, y0
    row[_COS], row[_SIN] = np.cos(angle), np.sin(angle)
    row[_Q] = axis_ratio
    row[_RHI] = _UNBOUNDED
    return row


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
        bins = resolution * (np.arange(npix) - (npix - 1) / 2)

        xx, yy = np.meshgrid(bins, bins)

        img = self.compute_density_grid(xx, yy)[0]

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

    @staticmethod
    def _get_coordinate_grids(resolution, npix, offset=(0.0, 0.0)):
        """Get the coordinates of each pixel centre.

        Args:
            resolution (unyt_quantity):
                The resolution of the grid.
            npix (tuple, int):
                The number of pixels in each dimension.
            offset (tuple of float):
                An offset from each pixel centre in units of the pixel size
                (used to sample within each pixel).

        Returns:
            tuple of unyt_array:
                The x and y coordinate of each pixel centre.
        """
        # Define 1D bin centres of each pixel, spaced by the resolution and
        # symmetric about the origin
        xbin_centres = resolution.value * (
            np.arange(npix[0]) - (npix[0] - 1) / 2 + offset[0]
        )
        ybin_centres = resolution.value * (
            np.arange(npix[1]) - (npix[1] - 1) / 2 + offset[1]
        )

        # Convert the 1D grid into 2D grids coordinate grids
        xx, yy = np.meshgrid(xbin_centres, ybin_centres) * resolution.units
        return xx, yy

    @staticmethod
    def _get_subpixel_offsets(supersample):
        """Get the offsets of the samples within a pixel.

        Args:
            supersample (int):
                The number of samples along each axis.

        Returns:
            list of tuple:
                The (x, y) offset of each sample in units of the pixel size.
        """
        steps = (np.arange(supersample) + 0.5) / supersample - 0.5
        return [(dx, dy) for dx in steps for dy in steps]

    def get_total(self, units):
        """Get the integral of the profile over the whole plane.

        Children with a closed form total override this, otherwise images
        are normalised to the light inside the field of view.

        Args:
            units (unyt.Unit):
                The units the coordinates (and the area) are in.

        Returns:
            float or None:
                The total, or None if unknown.
        """
        return None

    @accepts(resolution=(kpc, mas))
    def get_density_grid(
        self, resolution, npix, supersample=None, nthreads=1, **kwargs
    ):
        """Get the fraction of the light falling in each pixel.

        The profile is integrated over each pixel and normalised by its total
        over the whole plane, so light outside the field of view is lost
        rather than pushed back into it. The profiles the parametric imaging
        extension knows (Sersic2D, Gaussian2D and PointSource) are
        integrated there to a relative tolerance of IMAGE_TOLERANCE (see
        synthesizer.imaging.extensions.parametric_image). Any other profile
        is sampled on a supersample x supersample grid within each pixel,
        and normalised by its total if it has one, otherwise to the light
        inside the field of view.

        Args:
            resolution (unyt_quantity):
                The resolution of the grid.
            npix (tuple, int):
                The number of pixels in each dimension.
            supersample (int):
                The number of samples per pixel along each axis for profiles
                the imaging extension doesn't know (defaults to
                DEFAULT_SUPERSAMPLE).
            nthreads (int):
                The number of threads to use.
            **kwargs:
                Additional keyword arguments to pass to the
                compute_density_grid method.

        Returns:
            np.ndarray:
                The fraction of the light in each pixel.
        """
        if supersample is None:
            supersample = DEFAULT_SUPERSAMPLE

        # The profiles the imaging extension knows are integrated there (a
        # population morphology needs a signal per population, see
        # get_weighted_density_grid)
        if not kwargs and not isinstance(self, PopulationMorphology):
            rows = _get_image_rows(self, resolution)
            if rows is not None:
                return _make_images(
                    rows, np.ones((1, 1)), resolution, npix, nthreads
                )[..., 0]

        # Average the profile over samples within each pixel
        density_grid = 0.0
        offsets = self._get_subpixel_offsets(supersample)
        for offset in offsets:
            xx, yy = self._get_coordinate_grids(resolution, npix, offset)
            grid, _ = self.compute_density_grid(xx, yy, **kwargs)
            density_grid = density_grid + np.asarray(grid)
        density_grid = density_grid / len(offsets)

        # Normalise by the total over the plane if we know it
        total = None if kwargs else self.get_total(resolution.units)
        if total is None:
            return density_grid / np.sum(density_grid)
        return density_grid * resolution.value**2 / total

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

    def _get_image_params(self, resolution):
        """Get the imaging extension's parameters of the point source.

        Args:
            resolution (unyt_quantity): The resolution of the grid.

        Returns:
            np.ndarray: The profile parameters (see _get_image_rows).
        """
        units = resolution.units
        offset = (
            self.offset_kpc if units.dimensions == length else self.offset_mas
        )
        if offset is None:
            raise exceptions.InconsistentArguments(
                f"A {units} offset must be provided to the PointSource "
                f"morphology for a {units} grid."
            )
        offset = offset.to(units).value
        return _new_image_row(2, offset[0], offset[1])

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
        image = np.zeros(xx.shape)

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
            image[j, i] = 1.0
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
            image[j, i] = 1.0
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

    def _get_covariance(self):
        """Get the covariance matrix (in the units of stddev_x).

        Returns:
            np.ndarray:
                The 2x2 covariance matrix.
        """
        sx = self.stddev_x.value
        sy = self.stddev_y.to(self.stddev_x.units).value
        return np.array(
            [[sx**2, self.rho * sx * sy], [self.rho * sx * sy, sy**2]]
        )

    def get_total(self, units):
        """Get the integral of the profile over the whole plane.

        The profile is a normalised probability density (per unit area in
        the units of stddev_x).

        Args:
            units (unyt.Unit):
                The units the coordinates (and the area) are in.

        Returns:
            float:
                The total.
        """
        return (1 * self.stddev_x.units).to(units).value ** 2

    def get_radius(self, x, y):
        """Get the (elliptical) radius of each point.

        Like Sersic2D this is the semi-major axis of the ellipse of constant
        density through the point, so contours of the profile are contours
        of the radius.

        Args:
            x (unyt_array): x values on a 2D grid.
            y (unyt_array): y values on a 2D grid.

        Returns:
            unyt_array: The radius of each point.
        """
        units = self.stddev_x.units
        dx = (x - self.x_mean).to(units).value
        dy = (y - self.y_mean).to(units).value
        cov = self._get_covariance()
        inv_cov = np.linalg.inv(cov)
        mahalanobis_sq = (
            inv_cov[0, 0] * dx**2
            + 2 * inv_cov[0, 1] * dx * dy
            + inv_cov[1, 1] * dy**2
        )
        sigma_major = np.sqrt(np.max(np.linalg.eigvalsh(cov)))
        return np.sqrt(mahalanobis_sq) * sigma_major * units

    def _get_ellipse(self, units):
        """Get the centre, orientation and axis ratio of the contours.

        Args:
            units (unyt.Unit): The units of the coordinates.

        Returns:
            tuple: The centre x and y, the angle of the major axis (radians)
                and the minor to major axis ratio.
        """
        eigvals, eigvecs = np.linalg.eigh(self._get_covariance())
        major = eigvecs[:, 1]
        return (
            self.x_mean.to(units).value,
            self.y_mean.to(units).value,
            np.arctan2(major[1], major[0]),
            np.sqrt(eigvals[0] / eigvals[1]),
        )

    def _get_image_params(self, resolution):
        """Get the imaging extension's parameters of the profile.

        Args:
            resolution (unyt_quantity): The resolution of the grid.

        Returns:
            np.ndarray: The profile parameters (see _get_image_rows).
        """
        units = resolution.units
        x0, y0, angle, axis_ratio = self._get_ellipse(units)
        eigvals = np.linalg.eigvalsh(self._get_covariance())
        scale = (1 * self.stddev_x.units).to(units).value
        row = _new_image_row(1, x0, y0, angle, axis_ratio)
        row[_SCALE] = np.sqrt(eigvals[1]) * scale

        # A profile narrower than a pixel is integrated in shells near its
        # centre
        row[_CUSP] = float(np.sqrt(eigvals[0]) * scale < resolution.value)
        return row

    def get_enclosed_fraction(self, radius):
        """Get the fraction of the light within an (elliptical) radius.

        Args:
            radius (unyt_array): The radii (see get_radius).

        Returns:
            np.ndarray: The fraction of the total within each radius.
        """
        sigma_major = np.sqrt(
            np.max(np.linalg.eigvalsh(self._get_covariance()))
        )
        r = radius.to(self.stddev_x.units).value / sigma_major
        return 1.0 - np.exp(-0.5 * r**2)


class Gaussian2DAnnuli(Gaussian2D):
    """A subclass of Gaussian2D that supports masking of concentric annuli.

    Attributes:
        x_mean: (float): The mean of the Gaussian along the x-axis.
        y_mean: (float): The mean of the Gaussian along the y-axis.
        stddev_x: (float): The standard deviation along the x-axis.
        stddev_y: (float): The standard deviation along the y-axis.
        rho: (float): The population correlation coefficient between x and y.
        radii (list of float): The radii defining the annuli.
        annulus (int): Index of the annulus to be used when imaging.
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
        annulus=None,
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
            annulus (int): The index of the annulus this morphology
                describes, used when imaging.
        """
        deprecation(
            "Gaussian2DAnnuli is deprecated, give a Stars one population "
            "per annulus and use Annuli(Gaussian2D(...), radii) instead."
        )

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

        # Add an infinite outer radius for the last annulus unless the user
        # has already defined the last radius as infinity
        if np.isfinite(self.radii[-1]):
            self.radii = np.append(self.radii, np.inf * radii.units)

        # How many annuli are there? (each is bounded by consecutive radii)
        self.n_annuli = len(self.radii) - 1

        # The annulus this morphology describes, used when no annulus is
        # passed to get_density_grid (e.g. when imaging)
        self.annulus = annulus

    # The imaging extension can't apply the annulus these mask by
    _get_image_params = None

    def get_total(self, units):
        """Annuli are normalised over their own pixels.

        Args:
            units (unyt.Unit): Unused.

        Returns:
            None
        """
        return None

    @accepts(x=(kpc, mas), y=(kpc, mas))
    def compute_density_grid(self, x, y, annulus=None):
        """Compute the Gaussian density grid with optional annulus masking.

        Args:
            x (array-like): x values on a 2D grid.
            y (array-like): y values on a 2D grid.
            annulus (int): Index of the annulus to be used. Defaults to
                the annulus given at instantiation.

        Returns:
            np.ndarray: The masked Gaussian density grid.
            float: The normalisation factor for the density grid.
        """
        # Fall back on the annulus given at instantiation
        if annulus is None:
            annulus = self.annulus
        if annulus is None:
            raise exceptions.InconsistentArguments(
                "No annulus index given. Pass annulus when instantiating the "
                "morphology or when getting the density grid."
            )

        # Ensure the annulus index is valid
        if annulus < 0 or annulus >= self.n_annuli:
            raise exceptions.InconsistentArguments(
                f"Invalid annulus index: {annulus}. "
                f"Must be between 0 and {self.n_annuli - 1}."
            )

        # Get the whole density grid first
        density_grid, _ = super().compute_density_grid(x, y)

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

        # Normalise over the annulus itself since an annulus population only
        # holds the mass within that annulus
        norm = np.sum(density_grid)
        if norm == 0:
            warn(
                f"Annulus {annulus} contains no pixel centres at this "
                "resolution, its emission will be missing from the image."
            )
            norm = 1.0

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
        x_0=0 * kpc,
        y_0=0 * kpc,
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

        # Compute the (elliptical) radius of each point
        radius = self.get_radius(x, y)

        # Define coefficient of Sersic profile from Sersic index
        b_n = scipy.special.gammaincinv(2 * self.sersic_index, 0.5)

        # Compute and return the Sersic profile based on the radius
        if radius.units == kpc and self.r_eff_kpc is not None:
            # Correct Sersic law: I(R) = I_e exp[-b_n * ((R/R_e)^(1/n) - 1)]
            grid = self.amplitude * np.exp(
                -b_n
                * (
                    (radius / self.r_eff_kpc) ** (1.0 / self.sersic_index)
                    - 1.0
                )
            )
        elif radius.units == mas and self.r_eff_mas is not None:
            grid = self.amplitude * np.exp(
                -b_n
                * (
                    (radius / self.r_eff_mas) ** (1.0 / self.sersic_index)
                    - 1.0
                )
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

    def get_radius(self, x, y):
        """Get the (elliptical) radius of each point.

        Args:
            x (unyt_array): x values on a 2D grid.
            y (unyt_array): y values on a 2D grid.

        Returns:
            unyt_array: The radius of each point.
        """
        # Compute coordinate offset from x, y axes
        a = (x - self.x_0) * np.cos(self.theta) + (y - self.y_0) * np.sin(
            self.theta
        )
        b = -(x - self.x_0) * np.sin(self.theta) + (y - self.y_0) * np.cos(
            self.theta
        )

        # Compute radius from adjusted x, y coordinates
        return np.sqrt(a**2 + (b / (1 - self.ellipticity)) ** 2)

    def _get_r_eff(self, units):
        """Get the effective radius in a set of units.

        Args:
            units (unyt.Unit): Length or angle units.

        Returns:
            float: The effective radius.
        """
        r_eff = (
            self.r_eff_kpc if units.dimensions == length else self.r_eff_mas
        )
        if r_eff is None:
            raise exceptions.InconsistentArguments(
                f"A {units} effective radius must be provided to the "
                f"Sersic2D morphology for {units} coordinates."
            )
        return r_eff.to(units).value

    def get_total(self, units):
        """Get the integral of the profile over the whole plane.

        Args:
            units (unyt.Unit):
                The units the coordinates (and the area) are in.

        Returns:
            float:
                The total.
        """
        n = self.sersic_index
        b_n = scipy.special.gammaincinv(2 * n, 0.5)
        return (
            self.amplitude
            * 2
            * np.pi
            * n
            * (1 - self.ellipticity)
            * np.exp(b_n)
            * b_n ** (-2 * n)
            * scipy.special.gamma(2 * n)
            * self._get_r_eff(units) ** 2
        )

    def _get_ellipse(self, units):
        """Get the centre, orientation and axis ratio of the contours.

        Args:
            units (unyt.Unit): The units of the coordinates.

        Returns:
            tuple: The centre x and y, the angle of the major axis (radians)
                and the minor to major axis ratio.
        """
        return (
            self.x_0.to(units).value,
            self.y_0.to(units).value,
            self.theta,
            1 - self.ellipticity,
        )

    def _get_image_params(self, resolution):
        """Get the imaging extension's parameters of the profile.

        Args:
            resolution (unyt_quantity): The resolution of the grid.

        Returns:
            np.ndarray: The profile parameters (see _get_image_rows).
        """
        units = resolution.units
        x0, y0, angle, axis_ratio = self._get_ellipse(units)
        row = _new_image_row(0, x0, y0, angle, axis_ratio)
        row[_SCALE] = self._get_r_eff(units)
        row[_INDEX] = self.sersic_index
        row[_BN] = scipy.special.gammaincinv(2 * self.sersic_index, 0.5)

        # The centre of a Sersic profile is a cusp (or, for n < 1, a peak
        # much sharper than a Gaussian), so it is integrated in shells
        row[_CUSP] = 1.0
        return row

    def get_enclosed_fraction(self, radius):
        """Get the fraction of the light within an (elliptical) radius.

        Args:
            radius (unyt_array): The radii (see get_radius).

        Returns:
            np.ndarray: The fraction of the total within each radius.
        """
        n = self.sersic_index
        b_n = scipy.special.gammaincinv(2 * n, 0.5)
        r = radius.value / self._get_r_eff(radius.units)
        return scipy.special.gammainc(2 * n, b_n * r ** (1.0 / n))


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
        annulus (int): Index of the annulus to be used when imaging.
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
        x_0=0 * kpc,
        y_0=0 * kpc,
        theta=0,
        ellipticity=0,
        cosmo=None,
        redshift=None,
        annulus=None,
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
            annulus (int): The index of the annulus this morphology
                describes, used when imaging.
        """
        deprecation(
            "Sersic2DAnnuli is deprecated, give a Stars one population "
            "per annulus and use Annuli(Sersic2D(...), radii) instead."
        )

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

        # Add an infinite outer radius for the last annulus unless the user
        # has already defined the last radius as infinity
        if np.isfinite(self.radii[-1]):
            self.radii = np.append(self.radii, np.inf * radii.units)

        # How many annuli are there? (each is bounded by consecutive radii)
        self.n_annuli = len(self.radii) - 1

        # The annulus this morphology describes, used when no annulus is
        # passed to get_density_grid (e.g. when imaging)
        self.annulus = annulus

    # The imaging extension can't apply the annulus these mask by
    _get_image_params = None

    def get_total(self, units):
        """Annuli are normalised over their own pixels.

        Args:
            units (unyt.Unit): Unused.

        Returns:
            None
        """
        return None

    @accepts(x=(kpc, mas), y=(kpc, mas))
    def compute_density_grid(self, x, y, annulus=None):
        """Compute the density grid with optional annulus masking.

        Args:
            x (array-like): x values on a 2D grid.
            y (array-like): y values on a 2D grid.
            annulus (int): Index of the annulus to be used. Defaults to
                the annulus given at instantiation.

        Returns:
            np.ndarray: The computed density grid, optionally masked by annuli.
            float: The normalisation factor for the density grid.
        """
        # Fall back on the annulus given at instantiation
        if annulus is None:
            annulus = self.annulus
        if annulus is None:
            raise exceptions.InconsistentArguments(
                "No annulus index given. Pass annulus when instantiating the "
                "morphology or when getting the density grid."
            )

        # Ensure the annulus index is valid
        if annulus < 0 or annulus >= self.n_annuli:
            raise exceptions.InconsistentArguments(
                f"Invalid annulus index: {annulus}. "
                f"Must be between 0 and {self.n_annuli - 1}."
            )

        # Get the density grid for the whole profile
        density_grid, _ = super().compute_density_grid(x, y)

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

        # Normalise over the annulus itself since an annulus population only
        # holds the mass within that annulus
        norm = np.sum(density_grid)
        if norm == 0:
            warn(
                f"Annulus {annulus} contains no pixel centres at this "
                "resolution, its emission will be missing from the image."
            )
            norm = 1.0

        return density_grid, norm


class PopulationMorphology(MorphologyBase):
    """A base class for morphologies describing several populations at once.

    A multi population parametric Stars can give each population its own
    spatial distribution. An image is then the sum over populations of each
    population's (normalised) density grid weighted by its signal (e.g. its
    luminosity in a filter, or its spectrum for a data cube).

    Attributes:
        npop (int): The number of populations described.
    """

    def compute_density_grid(self, *args, **kwargs):
        """Population morphologies need a signal for each population.

        Raises:
            InconsistentArguments:
                Always, use get_weighted_density_grid instead.
        """
        raise exceptions.InconsistentArguments(
            f"{self.__class__.__name__} describes {self.npop} populations, "
            "so it needs a signal per population. Use "
            "get_weighted_density_grid instead."
        )

    @abstractmethod
    def get_weighted_density_grid(self, resolution, npix, signals, nthreads=1):
        """Get the sum of each population's density grid times its signal.

        Args:
            resolution (unyt_quantity):
                The resolution of the grid.
            npix (tuple, int):
                The number of pixels in each dimension.
            signals (np.ndarray):
                The signal of each population, with shape (npop, ...). Any
                trailing axes (e.g. wavelength) are kept in the result.
            nthreads (int):
                The number of threads to use.

        Returns:
            np.ndarray:
                The weighted density grid with shape (npix[1], npix[0], ...).
        """
        pass


class PerPopulation(PopulationMorphology):
    """A separate morphology for each population (or group of populations).

    This is what a Stars combined from populations with different
    morphologies (e.g. a bulge and a disk) uses. Each entry is either a
    single morphology for one population or a population morphology (e.g.
    Annuli) covering as many populations as it describes.

    Attributes:
        morphologies (list of MorphologyBase): The morphology of each
            population (or group of populations).
        npop (int): The number of populations.
    """

    def __init__(self, morphologies):
        """Initialise the morphology.

        Args:
            morphologies (list of MorphologyBase):
                The morphology of each population (or group of populations),
                in population order. Nested PerPopulation morphologies are
                flattened.
        """
        self.morphologies = []
        for morphology in morphologies:
            if isinstance(morphology, PerPopulation):
                self.morphologies.extend(morphology.morphologies)
            else:
                self.morphologies.append(morphology)
        self._counts = [
            getattr(morph, "npop", 1)
            if isinstance(morph, PopulationMorphology)
            else 1
            for morph in self.morphologies
        ]
        self.npop = int(np.sum(self._counts))

    def get_population_morphology(self, index):
        """Get the morphology of a single population.

        Args:
            index (int):
                The population's index.

        Returns:
            MorphologyBase:
                The population's morphology.
        """
        start = 0
        for morphology, count in zip(self.morphologies, self._counts):
            if index < start + count:
                if isinstance(morphology, PopulationMorphology):
                    return morphology.get_population_morphology(index - start)
                return morphology
            start += count
        raise exceptions.InconsistentArguments(
            f"Population {index} doesn't exist ({self.npop} populations)."
        )

    def get_weighted_density_grid(self, resolution, npix, signals, nthreads=1):
        """Get the sum of each population's density grid times its signal.

        When the imaging extension knows every population's profile all the
        populations are imaged in a single call.

        Args:
            resolution (unyt_quantity):
                The resolution of the grid.
            npix (tuple, int):
                The number of pixels in each dimension.
            signals (np.ndarray):
                The signal of each population, with shape (npop, ...).
            nthreads (int):
                The number of threads to use.

        Returns:
            np.ndarray:
                The weighted density grid with shape (npix[1], npix[0], ...).
        """
        signals = np.asarray(signals)
        rows = _get_image_rows(self, resolution)
        if rows is not None:
            images = _make_images(rows, signals, resolution, npix, nthreads)
            return images.reshape((npix[1], npix[0]) + signals.shape[1:])

        out = None
        start = 0
        for morphology, count in zip(self.morphologies, self._counts):
            group = signals[start : start + count]
            start += count
            if isinstance(morphology, PopulationMorphology):
                contribution = morphology.get_weighted_density_grid(
                    resolution, npix, group, nthreads=nthreads
                )
            else:
                density = morphology.get_density_grid(
                    resolution, npix, nthreads=nthreads
                )
                contribution = (
                    density.reshape(density.shape + (1,) * (group.ndim - 1))
                    * group[0]
                )
            out = contribution if out is None else out + contribution
        return out


class Annuli(PopulationMorphology):
    """A profile split into annuli, one population per annulus.

    Each annulus holds only the light of the profile within it (an annulus
    population holds only the mass within that annulus), so each is
    normalised by the profile's light within the annulus. Pixels straddling
    an annulus edge are split between the annuli exactly.

    Attributes:
        profile (MorphologyBase): The underlying profile (e.g. Sersic2D),
            which must provide get_radius.
        radii (unyt_array): The edges of the annuli (one more than the number
            of annuli, the last may be infinite).
        npop (int): The number of annuli (populations).
    """

    @accepts(radii=(kpc, mas))
    def __init__(self, profile, radii):
        """Initialise the morphology.

        Args:
            profile (MorphologyBase):
                The underlying profile, which must provide get_radius (e.g.
                Sersic2D or Gaussian2D).
            radii (unyt_array):
                The edges of the annuli, increasing, with one more edge than
                there are annuli. The last may be infinite.
        """
        if not hasattr(profile, "get_radius"):
            raise exceptions.InconsistentArguments(
                f"{profile.__class__.__name__} doesn't define a radius, so "
                "it can't be split into annuli."
            )
        if radii.size < 2 or np.any(np.diff(radii.value) <= 0):
            raise exceptions.InconsistentArguments(
                "radii must be at least two increasing annulus edges."
            )
        self.profile = profile
        self.radii = radii
        self.npop = radii.size - 1

    def get_population_morphology(self, index):
        """Get the morphology of a single annulus.

        Args:
            index (int):
                The annulus' index.

        Returns:
            Annuli:
                The single annulus.
        """
        return Annuli(self.profile, self.radii[index : index + 2])

    def get_weighted_density_grid(
        self, resolution, npix, signals, supersample=None, nthreads=1
    ):
        """Get the sum of each annulus' density grid times its signal.

        For profiles the imaging extension knows, every annulus is imaged in
        a single call, with each pixel integrated (and split exactly between
        the annuli it straddles) and each annulus normalised by the
        profile's light within it, so light outside the field of view is
        lost. Any other profile is sampled within each pixel and each
        annulus normalised over its own pixels.

        Args:
            resolution (unyt_quantity):
                The resolution of the grid.
            npix (tuple, int):
                The number of pixels in each dimension.
            signals (np.ndarray):
                The signal of each annulus, with shape (npop, ...).
            supersample (int):
                The number of samples per pixel along each axis for profiles
                the imaging extension doesn't know (defaults to
                DEFAULT_SUPERSAMPLE).
            nthreads (int):
                The number of threads to use.

        Returns:
            np.ndarray:
                The weighted density grid with shape (npix[1], npix[0], ...).
        """
        if supersample is None:
            supersample = DEFAULT_SUPERSAMPLE
        signals = np.asarray(signals)
        npixels = npix[0] * npix[1]

        rows = _get_image_rows(self, resolution)
        if rows is not None:
            images = _make_images(rows, signals, resolution, npix, nthreads)
            return images.reshape((npix[1], npix[0]) + signals.shape[1:])

        # Integrate the profile over the part of each pixel in each annulus
        weights = np.zeros(npixels * self.npop)
        offsets = self._get_subpixel_offsets(supersample)
        pixel = np.arange(npixels)
        for offset in offsets:
            xx, yy = self._get_coordinate_grids(resolution, npix, offset)
            profile, _ = self.profile.compute_density_grid(xx, yy)
            profile = np.asarray(profile).ravel()
            radius = self.profile.get_radius(xx, yy)
            radius = radius.to(self.radii.units).value.ravel()
            label = np.searchsorted(self.radii.value, radius, side="right") - 1
            inside = (label >= 0) & (label < self.npop)
            weights += np.bincount(
                pixel[inside] * self.npop + label[inside],
                weights=profile[inside],
                minlength=weights.size,
            )
        weights = weights.reshape(npixels, self.npop) / len(offsets)

        # Normalise each annulus over its own pixels
        return self._sum_annuli(
            weights, np.sum(weights, axis=0), signals, npix
        )

    def _sum_annuli(self, weights, norm, signals, npix):
        """Sum each annulus' normalised light times its signal.

        Args:
            weights (np.ndarray):
                The light of each annulus in each pixel, (npixels, npop).
            norm (np.ndarray):
                The total light of each annulus.
            signals (np.ndarray):
                The signal of each annulus, with shape (npop, ...).
            npix (tuple, int):
                The number of pixels in each dimension.

        Returns:
            np.ndarray:
                The weighted density grid with shape (npix[1], npix[0], ...).
        """
        covered = np.sum(weights, axis=0) > 0
        if np.any(
            ~covered & np.any(signals != 0, axis=tuple(range(1, signals.ndim)))
        ):
            warn(
                f"{np.sum(~covered)} annuli don't overlap the image, their "
                "emission will be missing from it."
            )
        scale = signals / np.where(norm > 0, norm, 1.0).reshape(
            (self.npop,) + (1,) * (signals.ndim - 1)
        )
        scale[norm <= 0] = 0.0

        # Sum each annulus' signal over the pixels it covers
        out = weights @ scale.reshape(self.npop, -1)
        return out.reshape((npix[1], npix[0]) + signals.shape[1:])
