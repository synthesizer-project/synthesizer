"""A module for creating and manipulating parametric stellar populations.

This is the parametric analog of particle.Stars. It not only computes and holds
the SFZH grid but everything describing a parametric Galaxy's stellar
component.

Example usage::

    stars = Stars(log10ages, metallicities, sfzh=sfzh)
    stars.get_spectra(emission_model)
    stars.plot_spectra()
"""

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from unyt import (
    Hz,
    Msun,
    erg,
    nJy,
    s,
    unyt_array,
    unyt_quantity,
    yr,
)

from synthesizer import exceptions
from synthesizer.components.stellar import StarsComponent
from synthesizer.emission_models.utils import get_param
from synthesizer.grid import Grid
from synthesizer.parametric.metal_dist import Common as ZDistCommon
from synthesizer.parametric.sf_hist import Common as SFHCommon
from synthesizer.synth_warnings import warn
from synthesizer.units import Quantity, accepts, convert_in_place
from synthesizer.utils.operation_timers import timed
from synthesizer.utils.plt import single_histxy
from synthesizer.utils.stats import weighted_mean, weighted_median
from synthesizer.utils.util_funcs import get_attr_unit_conversion

# The number of bins each interval between axis points is split into when
# binning an SFH or metallicity distribution function. Mass is uniform within
# each bin, so the error from this converges at second order: with 8 the
# spectra are within 0.2% of the converged result for a narrow (sigma = 4 Myr)
# young Gaussian SFH, and within 0.04% for typical SFHs.
SFZH_SUBDIVISIONS = 8

# The number of pieces each age bin is split into to average an age dependent
# distribution (e.g. Z(t)) over the mass formed within the bin
AGE_DEPENDENCE_SAMPLES = 8

# The bound used for an open end of a mask's range. The extensions are
# compiled with -ffast-math, which makes infinities undefined behaviour, so
# masks use the largest finite float instead
_MASK_BOUND = np.finfo(np.float64).max


def union_edges(edges, other_edges):
    """Get the union of two sets of bin edges.

    A value repeated in either set (i.e. a zero width bin) is repeated in the
    union as many times as in the set repeating it most, so every bin of
    both sets can be rebuilt from bins of the union.

    Args:
        edges (np.ndarray of float):
            The first (non-decreasing) bin edges.
        other_edges (np.ndarray of float):
            The second (non-decreasing) bin edges.

    Returns:
        np.ndarray of float:
            The union of the edges.
    """
    values = np.union1d(edges, other_edges)

    def count(arr):
        return np.searchsorted(arr, values, "right") - np.searchsorted(
            arr, values, "left"
        )

    return np.repeat(
        values, np.maximum(np.maximum(count(edges), count(other_edges)), 1)
    )


def rebin_axis(masses, axis, edges, new_edges):
    """Rebin masses along one axis onto finer edges.

    The new edges must contain every one of the old edges (e.g. the union
    of two sets of edges), so every new bin lies within a single old bin. A
    finite bin's mass is uniform within it, so each new finite bin takes the
    fraction of its old bin's mass given by their widths. A zero width bin's
    mass moves to the zero width bin at the same value (the first to the
    first, the second to the second and so on when a value has several).

    Args:
        masses (np.ndarray):
            The masses.
        axis (int):
            The axis of masses to rebin.
        edges (np.ndarray of float):
            The current bin edges along that axis.
        new_edges (np.ndarray of float):
            The new bin edges along that axis.

    Returns:
        np.ndarray:
            The rebinned masses.
    """
    lo, hi = edges[:-1], edges[1:]
    new_lo, new_hi = new_edges[:-1], new_edges[1:]
    nbins = lo.size

    # Find the old bin holding each new finite bin (searching to the right
    # of any repeated edge lands in the finite bin after a zero width one)
    source = np.searchsorted(edges, new_lo, side="right") - 1
    inside = (source >= 0) & (source < nbins) & (new_hi > new_lo)
    source = np.clip(source, 0, nbins - 1)
    inside &= (hi[source] > lo[source]) & (new_hi <= hi[source])
    fraction = np.where(
        inside,
        (new_hi - new_lo) / np.where(inside, hi[source] - lo[source], 1.0),
        0.0,
    )

    # Zero width bins take the mass of the zero width bin at the same value.
    # A value can have several zero width bins (e.g. every bin clipped to
    # an age of zero), so the nth new one at a value takes the nth old one
    points = np.flatnonzero(lo == hi)
    new_points = np.flatnonzero(new_lo == new_hi)
    if points.size > 0:
        values, new_values = lo[points], new_lo[new_points]
        rank = np.arange(new_values.size) - np.searchsorted(
            new_values, new_values
        )
        matches = np.searchsorted(values, new_values) + rank
        found = matches < np.searchsorted(values, new_values, side="right")
        source[new_points[found]] = points[matches[found]]
        fraction[new_points[found]] = 1.0

    shape = [1] * masses.ndim
    shape[axis] = fraction.size
    return np.take(masses, source, axis=axis) * fraction.reshape(shape)


class Stars(StarsComponent):
    """The parametric stellar population object.

    This class holds a binned star formation and metal enrichment history
    describing the age and metallicity of the stellar population, an
    optional morphology model describing the distribution of those stars,
    and various other important attributes for defining a parametric
    stellar population.

    Attributes:
        ages (np.ndarray of float):
            The array of ages defining the age axis of the SFZH.
        metallicities (np.ndarray of float):
            The array of metallicitities defining the metallicity axes of
            the SFZH.
        initial_mass (unyt_quantity/float)
            The total initial stellar mass. This is the total mass of stars
            formed, i.e. the integral of the SFH over time, and is not the
            same as the surviving mass which accounts for stellar evolution.
        surviving_mass (unyt_quantity/float):
            The total surviving stellar mass. This is the total mass of stars
            currently alive, i.e. the integral of the SFH over time weighted
            by the fraction of stars that survive to each age, and is not the
            same as the initial mass which does not account for stellar
            evolution.
        morphology (morphology.* e.g. Sersic2D)
            An instance of one of the morphology classes describing the
            stellar population's morphology. This can be any of the family
            of morphology classes from synthesizer.morphology.
        sfzh (np.ndarray of float):
            The mass formed at each (age, metallicity) point. If provided
            all following arguments are ignored.
        sf_hist (np.ndarray of float):
            An array describing the star formation history.
        metal_dist (np.ndarray of float):
            An array describing the metallicity distribution.
        sf_hist_func (SFH.*)
            An instance of one of the child classes of SFH. This will be
            used to calculate sf_hist and takes precedence over a passed
            sf_hist if both are present.
        metal_dist_func (ZH.*)
            An instance of one of the child classes of ZH. This will be
            used to calculate metal_dist and takes precedence over a
            passed metal_dist if both are present.
        instant_sf (float):
            An age at which to compute an instantaneous SFH, i.e. all
            stellar mass populating a single SFH bin.
        instant_metallicity (float):
            A metallicity at which to compute an instantaneous ZH, i.e. all
            stellar populating a single ZH bin.
        log10ages_lims (array_like_float)
            The log10(age) limits of the SFZH grid.
        metallicities_lims (np.ndarray of float):
            The metallicity limits of the SFZH grid.
        log10metallicities_lims (np.ndarray of float):
            The log10(metallicity) limits of the SFZH grid.
        metallicity_grid_type (str):
            The type of gridding for the metallicity axis. Either:
                - Regular linear ("Z")
                - Regular logspace ("log10Z")
                - Irregular (None)
    """

    # Define quantities
    initial_mass = Quantity("mass")
    surviving_mass = Quantity("mass")
    age_offset = Quantity("time")

    @accepts(
        surviving_mass=Msun.in_base("galactic"),
        initial_mass=Msun.in_base("galactic"),
    )
    @timed("ParametricStars.__init__")
    def __init__(
        self,
        log10ages,
        metallicities,
        initial_mass=None,
        surviving_mass=None,
        grid=None,
        morphology=None,
        sfzh=None,
        sf_hist=None,
        metal_dist=None,
        fesc=None,
        fesc_ly_alpha=None,
        extra_axes=None,
        **kwargs,
    ):
        """Initialise the parametric stellar population.

        Can either be instantiated by:
        - Passing a SFZH grid explictly.
        - Passing instant_sf and instant_metallicity to get an instantaneous
          SFZH.
        - Passing functions that describe the SFH and ZH.
        - Passing arrays that describe the SFH and ZH.
        - Passing any combination of SFH and ZH instant values, arrays
          or functions.

        Args:
            log10ages (np.ndarray of float):
                The array of ages defining the log10(age) axis of the SFZH.
            metallicities (np.ndarray of float):
                The array of metallicitities defining the metallicity axes of
                the SFZH.
            initial_mass (unyt_quantity/float):
                The total initial stellar mass. If provided the SFZH grid will
                be rescaled to obey this total mass.
            surviving_mass (unyt_quantity/float):
                The total surviving stellar mass. If provided the SFZH grid
                will be rescaled to obey this total mass.
            grid (Grid):
                A synthesizer Grid object. This is only used to provide
                stellar_fraction when initialising using a surviving mass, and
                is ignored if surviving_mass is not provided.
            morphology (morphology.* e.g. Sersic2D):
                An instance of one of the morphology classes describing the
                stellar population's morphology. This can be any of the family
                of morphology classes from synthesizer.morphology.
            sfzh (np.ndarray of float):
                The mass formed at each (age, metallicity) point. If provided
                all following arguments are ignored.
            sf_hist (float/unyt_quantity/np.ndarray of float/SFH.*):
                Either:
                    - An age at which to compute an instantaneous SFH, i.e. all
                      stellar mass populating a single SFH bin.
                    - An array describing the star formation history.
                    - An instance of one of the child classes of SFH. This
                      will be used to calculate an array describing the SFH.
            metal_dist (float/unyt_quantity/np.ndarray of float/ZDist.*):
                Either:
                    - A metallicity at which to compute an instantaneous
                      ZH, i.e. all stellar mass populating a single Z bin.
                    - An array describing the metallicity distribution.
                    - An instance of one of the child classes of ZH. This
                      will be used to calculate an array describing the
                      metallicity distribution.
            fesc (float):
                The escape fraction of incident radiation from the stars.
            fesc_ly_alpha (float):
                The escape fraction of Ly-alpha radiation from the stars.
            extra_axes (dict):
                Distributions along axes other than age and metallicity
                (e.g. ionisation_parameter), keyed by the axis name, each
                combined with the age and metallicity bins as an outer
                product (see _add_extra_axes). Each is a single value
                (float or unyt_quantity), or a tuple of bin edges and either
                the weight in each bin or a distribution with a
                get_bin_weights(edges) method (e.g. a ZDist). Grid axes
                with no bins are otherwise taken from the emission model or
                this object at extraction, like particle attributes.
            **kwargs (dict):
                Arbitrary keyword arguments to be set as attributes on the
                Stars instance.
        """
        # Instantiate the parent
        StarsComponent.__init__(
            self,
            10**log10ages * yr,
            metallicities,
            _star_type="parametric",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )

        # The names of the parameters describing the population(s), these
        # become per population parameters when populations are combined
        self._parameter_names = {"fesc", "fesc_ly_alpha", *kwargs}

        # The (optional) names of the populations
        self.population_names = None

        # Containers for the emission of each population, one row per
        # population. The emission models store per population emission
        # under the same names as per particle emission, the
        # population_spectra, population_lines and population_photo_*
        # properties are the names to use
        self.particle_spectra = {}
        self.particle_lines = {}
        self.particle_photo_lnu = {}
        self.particle_photo_fnu = {}

        # Set the age grid lims
        self.log10ages_lims = [self.log10ages[0], self.log10ages[-1]]

        # Set the metallicity grid lims
        self.metallicities_lims = [
            self.metallicities[0],
            self.metallicities[-1],
        ]
        self.log10metallicities_lims = [
            self.log10metallicities[0],
            self.log10metallicities[-1],
        ]

        # Store the SFH we've been given, this is either...
        if issubclass(type(sf_hist), SFHCommon):
            self.sf_hist_func = sf_hist  # a SFH function
            self.sf_hist = None
            self._instant_sf = None
        elif isinstance(sf_hist, (unyt_quantity, float)):
            self._instant_sf = sf_hist  # an instantaneous SFH
            self.sf_hist_func = None
            self.sf_hist = None
        elif isinstance(sf_hist, (unyt_array, np.ndarray)):
            self.sf_hist = sf_hist  # a numpy array
            self.sf_hist_func = None
            self._instant_sf = None
        elif sf_hist is None:
            self.sf_hist = None  # we must have been passed a SFZH
            self.sf_hist_func = None
            self._instant_sf = None
        else:
            raise exceptions.InconsistentArguments(
                f"Unrecognised sf_hist type ({type(sf_hist)}! This should be"
                " either a float, an instance of a SFH function from the "
                "SFH module, or a single float."
            )

        # Store the metallicity distribution we've been given, either...
        if issubclass(type(metal_dist), ZDistCommon):
            self.metal_dist_func = metal_dist  # a ZDist function
            self.metal_dist = None
            self._instant_metallicity = None
        elif isinstance(metal_dist, (unyt_quantity, float, np.floating)):
            self._instant_metallicity = metal_dist  # an instantaneous SFH
            self.metal_dist_func = None
            self.metal_dist = None
        elif isinstance(metal_dist, (unyt_array, np.ndarray)):
            self.metal_dist = metal_dist  # a numpy array
            self.metal_dist_func = None
            self._instant_metallicity = None
        elif metal_dist is None:
            self.metal_dist = None  # we must have been passed a SFZH
            self.metal_dist_func = None
            self._instant_metallicity = None
        else:
            raise exceptions.InconsistentArguments(
                f"Unrecognised metal_dist type ({type(metal_dist)}! This "
                "should be either a float, an instance of a ZDist function "
                "from the ZDist module, or a single float."
            )

        # Store the total initial stellar mass
        self.initial_mass = initial_mass
        self.surviving_mass = surviving_mass

        # Raise exception if both initial mass and surviving mass are
        # provided, this is not physically consistent and we don't want to
        # guess which one the user meant to provide
        if self.surviving_mass is not None and self.initial_mass is not None:
            raise exceptions.InconsistentArguments(
                "Cannot specify both initial_mass and surviving_mass! Please"
                " specify only one of these or neither."
            )

        # Raise exception if surviving mass is provided but no grid is
        # provided, we need the stellar fraction from the grid to rescale
        # the SFZH to obey the surviving mass
        if self.surviving_mass is not None and grid is None:
            raise exceptions.InconsistentArguments(
                "If surviving_mass is specified then a Grid object must be "
                "provided to get the stellar_fraction for rescaling the SFZH!"
            )

        # Raise exception if we have been given the surviving mass and grid
        # but that grid doesn't have a stellar_fraction attribute
        if self.surviving_mass is not None and grid is not None:
            if not hasattr(grid, "stellar_fraction"):
                raise exceptions.InconsistentArguments(
                    "If surviving_mass is specified then a Grid object with a"
                    "stellar_fraction attribute must be provided to rescale"
                    "the SFZH!"
                )

        # Get the stellar fraction from the grid if we have been given a
        # surviving mass, this is needed to rescale the SFZH to obey the
        # surviving mass constraint. This is the fraction of the initial mass
        # that survives to the present day, and is used to calculate the
        # initial mass from the surviving mass.
        if self.surviving_mass is not None and grid is not None:
            self.stellar_fraction = grid.stellar_fraction

        # Get the bins describing the population, either from an explicit
        # SFZH (defined at the points of the axes) or from the SFH and
        # metallicity distribution. Either way the masses are a new array,
        # so normalising them in place never touches the caller's SFZH
        if sfzh is not None:
            edges, masses = self._sfzh_to_bins(sfzh)
        else:
            edges, masses = self._get_bins()

        # Add any distributions along other axes
        edges, masses, self._bin_units = self._add_extra_axes(
            edges,
            masses,
            extra_axes,
            age_samples=(
                self._get_age_samples(edges["ages"], self.sf_hist_func)
                if extra_axes
                else None
            ),
        )

        # Check the masses don't contain any NaN or Inf values, this can
        # happen if the SFH or ZH functions return NaN or Inf values for some
        # reason, and this will cause all kinds of problems downstream if we
        # don't catch it here.
        if np.any(~np.isfinite(masses)):
            raise exceptions.InconsistentArguments(
                "SFZH grid contains NaN or Inf values! "
                "Please check the input parameters."
            )

        # We can't normalise bins with no mass (e.g. an SFH or ZDist peaking
        # far outside the axes), doing so would silently make them NaN
        if (
            self.surviving_mass is not None or self.initial_mass is not None
        ) and np.sum(masses) <= 0:
            raise exceptions.InconsistentArguments(
                "The SFH and metallicity distribution put no mass on the "
                "age and metallicity axes, so the SFZH can't be normalised. "
                "Please check the input parameters."
            )

        # Normalise the masses if needs be, and calculate the initial mass
        # if we have been given a surviving mass. If we have been given an
        # initial mass we just need to rescale the masses to obey this
        # constraint, and if we have been given neither then we can just sum
        # the masses to get the total initial mass.
        if self.surviving_mass is not None:
            current_surviving_mass = np.sum(
                self._bins_to_grid(grid, edges, masses) * grid.stellar_fraction
            )
            self.sfzh_normalisation = (
                self._surviving_mass / current_surviving_mass
            )
            masses *= self.sfzh_normalisation

            # now calculate the initial mass
            self.initial_mass = (np.sum(masses) * Msun).in_base("galactic")

        elif self.initial_mass is not None:
            self.sfzh_normalisation = self._initial_mass / np.sum(masses)
            masses *= self.sfzh_normalisation

        else:
            # Otherwise calculate the total initial mass by just summing the
            # masses
            self.initial_mass = (np.sum(masses) * Msun).in_base("galactic")

        # Store the population's bins (which also sets sf_hist and
        # metal_dist from them)
        self._set_bins(edges, masses, self._bin_units)

        # Attach the morphology model
        self.morphology = morphology

        # Check if metallicities are uniformly binned in log10metallicity or
        # linear metallicity or not at all (e.g. BPASS)
        if (
            len(np.unique(self.metallicities[:-1] - self.metallicities[1:]))
            == 1
        ):
            # Regular linearly
            self.metallicity_grid_type = "Z"

        elif (
            len(
                np.unique(
                    self.log10metallicities[:-1] - self.log10metallicities[1:]
                )
            )
            == 1
        ):
            # Regular in logspace
            self.metallicity_grid_type = "log10Z"

        else:
            # Irregular
            self.metallicity_grid_type = None

    @staticmethod
    def _get_fine_edges(nodes, breakpoints=()):
        """Get fine bin edges spanning zero to the last of a set of nodes.

        Each interval between nodes is split into SFZH_SUBDIVISIONS pieces
        (log spaced when the nodes are positive), below the first node there
        is a single bin from zero, and any breakpoints inside the range are
        added as edges so no bin straddles one.

        NOTE: Mass is spread uniformly within each bin, so the subdivisions
        only need to resolve how the SFH (or metallicity distribution) varies
        across each interval between nodes.

        Args:
            nodes (np.ndarray of float):
                The (linear) axis points.
            breakpoints (np.ndarray of float):
                Values where the distribution changes abruptly.

        Returns:
            np.ndarray of float:
                The bin edges.
        """
        # The fractional position of each subdivision within an interval
        nodes = np.asarray(nodes, dtype=np.float64)
        steps = np.arange(SFZH_SUBDIVISIONS) / SFZH_SUBDIVISIONS

        # Subdivide each interval, in log space when every node is positive
        # (the grids' axes are log spaced) and linearly otherwise
        if nodes[0] > 0:
            log_nodes = np.log10(nodes)
            fine = 10 ** (
                log_nodes[:-1, None] + np.diff(log_nodes)[:, None] * steps
            )
        else:
            fine = nodes[:-1, None] + np.diff(nodes)[:, None] * steps

        # Only breakpoints strictly inside (0, last node) add new edges
        breakpoints = np.asarray(breakpoints, dtype=np.float64)
        breakpoints = breakpoints[
            (breakpoints > 0) & (breakpoints < nodes[-1])
        ]

        # Combine everything into sorted, unique edges starting at zero and
        # ending at the last node
        return np.unique(
            np.concatenate(([0.0], fine.ravel(), [nodes[-1]], breakpoints))
        )

    @staticmethod
    def _points_to_bins(points, point_masses):
        """Convert masses at a set of points along one axis into bins.

        Each point becomes a zero width bin holding its mass, with an empty
        finite bin between each pair of points.

        Args:
            points (np.ndarray of float):
                The (linear) points.
            point_masses (np.ndarray of float):
                The mass at each point.

        Returns:
            tuple:
                The bin edges and the mass in each bin.
        """
        # Repeating each point gives a zero width bin at every even index
        # with an empty finite bin between each pair of points
        masses = np.zeros(2 * len(points) - 1)
        masses[::2] = point_masses
        return np.repeat(np.asarray(points, dtype=np.float64), 2), masses

    def _get_age_bins(self, offset):
        """Get the age bins and the mass formed in each.

        Args:
            offset (float):
                The offset (in years) to apply to the ages.

        Returns:
            tuple:
                The age bin edges (in years) and the mass in each bin.

        Raises:
            InconsistentArguments:
                If no SFH has been defined.
        """
        # Work with the ages in years
        ages = convert_in_place(self.ages, yr).ndview

        # An SFH function is integrated exactly over fine bins of the axis
        if self.sf_hist_func is not None:
            edges = self._get_fine_edges(
                ages, self.sf_hist_func._get_breakpoints() - offset
            )
            return edges, self.sf_hist_func.get_bin_masses(edges + offset)

        # An instantaneous SFH is a single point
        if self._instant_sf is not None:
            age = convert_in_place(self._instant_sf, yr).value + offset
            return np.array([age, age]), np.array([1.0])

        # An array is defined at the points of the axis
        if self.sf_hist is not None:
            return self._points_to_bins(ages, self.sf_hist)

        # Without any of the above we have no SFH
        raise exceptions.InconsistentArguments(
            "A method for defining both the SFH and ZH must be provided!\n"
            "For each either an instantaneous"
            " value, a SFH/ZH object, or an array must be passed"
        )

    def _get_metal_bins(self):
        """Get the metallicity bins and the weight of each.

        Returns:
            tuple:
                The metallicity bin edges and the weight in each bin.

        Raises:
            InconsistentArguments:
                If no metallicity distribution has been defined.
        """
        # Work with the metallicities as a plain float array
        metals = np.asarray(self.metallicities, dtype=np.float64)

        # A delta function is a single point
        if self.metal_dist_func is not None and (
            self.metal_dist_func.name == "DeltaConstant"
        ):
            metal = float(self.metal_dist_func.get_metallicity())
            return np.array([metal, metal]), np.array([1.0])

        # A metallicity distribution is integrated exactly over fine bins of
        # the axis
        if self.metal_dist_func is not None:
            edges = self._get_fine_edges(metals)
            return edges, self.metal_dist_func.get_bin_weights(edges)

        # An instantaneous metallicity is a single point
        if self._instant_metallicity is not None:
            metal = float(self._instant_metallicity)
            return np.array([metal, metal]), np.array([1.0])

        # An array is defined at the points of the axis
        if self.metal_dist is not None:
            return self._points_to_bins(metals, self.metal_dist)

        # Without any of the above we have no metallicity distribution
        raise exceptions.InconsistentArguments(
            "A method for defining both the SFH and ZH must be provided!\n"
            "For each either an instantaneous"
            " value, a SFH/ZH object, or an array must be passed"
        )

    @timed("ParametricStars._get_bins")
    def _get_bins(self):
        """Get the bins describing the population from the SFH and ZH.

        The SFH and metallicity distribution are binned separately and the
        population is their product.

        Returns:
            tuple:
                The bin edges keyed by axis name and the (1, n_age, n_Z) bin
                masses.
        """
        # If no units assume unit system
        if self._instant_sf is not None and not isinstance(
            self._instant_sf, unyt_quantity
        ):
            self._instant_sf = self._instant_sf * self.ages.units

        # Bin the ages
        age_edges, age_masses = self._get_age_bins(0.0)

        # A metallicity distribution that depends on age is averaged over
        # the mass formed in each age bin, so it differs between age bins
        dist = self.metal_dist_func
        if dist is not None and dist.depends_on_age:
            metals = np.asarray(self.metallicities, dtype=np.float64)
            ages, weights = self._get_age_samples(age_edges, self.sf_hist_func)
            if dist.name == "DeltaConstant":
                # Each metallicity is a point, split linearly between the
                # points of a fine metallicity axis
                points = self._get_fine_edges(metals)[1:]
                metal_edges = np.repeat(points, 2)
                metal_weights = self._average_over_samples(
                    self._deposit_on_points(
                        dist.get_metallicities(ages * yr), points
                    ),
                    weights,
                )
            else:
                metal_edges = self._get_fine_edges(metals)
                metal_weights = self._average_over_samples(
                    self._get_sample_bin_weights(dist, ages, metal_edges),
                    weights,
                )
            edges = {"ages": age_edges, "metallicities": metal_edges}
            return edges, (age_masses[:, None] * metal_weights)[None]

        # Otherwise bin the metallicities separately
        metal_edges, metal_weights = self._get_metal_bins()

        # The population is the outer product of the two, with a leading
        # axis for the single population
        edges = {"ages": age_edges, "metallicities": metal_edges}
        masses = (age_masses[:, None] * metal_weights[None, :])[None]
        return edges, masses

    @staticmethod
    def _get_age_samples(age_edges, sf_hist_func=None):
        """Sample each age bin to average an age dependent distribution.

        Each bin is split into AGE_DEPENDENCE_SAMPLES pieces (log spaced
        when the bin is at positive ages), each weighted by the mass formed
        in it: from the SFH function if there is one, otherwise uniformly
        (the mass is spread uniformly within each bin). A zero width bin is
        a single sample.

        Args:
            age_edges (np.ndarray of float):
                The increasing age bin edges (in yr).
            sf_hist_func (SFH.*):
                The SFH function, or None.

        Returns:
            tuple:
                The age (in yr) of each sample and the
                fraction of the bin's mass in it, each (n_age_bins,
                AGE_DEPENDENCE_SAMPLES).
        """
        lo = age_edges[:-1, None]
        hi = age_edges[1:, None]
        steps = np.arange(AGE_DEPENDENCE_SAMPLES + 1) / AGE_DEPENDENCE_SAMPLES
        with np.errstate(divide="ignore", invalid="ignore"):
            sub_edges = np.where(
                lo > 0, lo * (hi / lo) ** steps, lo + (hi - lo) * steps
            )
        ages = 0.5 * (sub_edges[:, 1:] + sub_edges[:, :-1])

        # Weight each sample by the mass formed in it
        if sf_hist_func is not None:
            cdf = sf_hist_func.get_cdf(sub_edges.ravel())
            weights = np.diff(cdf.reshape(sub_edges.shape), axis=1)
        else:
            weights = np.diff(sub_edges, axis=1)

        # A zero width bin is a single sample, and a bin with no mass is
        # weighted uniformly (it doesn't matter how)
        weights[(hi == lo)[:, 0]] = 0.0
        weights[(hi == lo)[:, 0], 0] = 1.0
        total = np.sum(weights, axis=1, keepdims=True)
        weights = np.where(
            total > 0, weights / np.where(total > 0, total, 1.0), 0.0
        )
        weights[total[:, 0] <= 0] = 1.0 / AGE_DEPENDENCE_SAMPLES
        return ages, weights

    @staticmethod
    def _average_over_samples(sample_weights, weights):
        """Average per sample bin weights over the samples of each age bin.

        Args:
            sample_weights (np.ndarray):
                The bin weights for each sample, (n_age_bins, n_samples,
                n_bins).
            weights (np.ndarray):
                The fraction of each age bin's mass in each sample,
                (n_age_bins, n_samples).

        Returns:
            np.ndarray:
                The (n_age_bins, n_bins) average weights, each row
                normalised.
        """
        average = np.einsum("aib,ai->ab", sample_weights, weights)
        total = np.sum(average, axis=1, keepdims=True)
        return average / np.where(total > 0, total, 1.0)

    @staticmethod
    def _get_sample_bin_weights(dist, ages, edges):
        """Get the bin weights of an age dependent distribution per sample.

        Args:
            dist (object):
                The distribution, with at_age and get_bin_weights methods.
            ages (np.ndarray of float):
                The age (in yr) of each sample, (n_age_bins, n_samples).
            edges (np.ndarray of float):
                The bin edges.

        Returns:
            np.ndarray:
                The (n_age_bins, n_samples, n_bins) bin weights.
        """
        return np.array(
            [
                dist.at_age(age * yr).get_bin_weights(edges)
                for age in ages.ravel()
            ]
        ).reshape(ages.shape + (edges.size - 1,))

    @staticmethod
    def _deposit_on_points(values, points):
        """Split values linearly between their neighbouring points.

        The split is linear in log10 when the points are positive (as the
        grids' axes are), so for any grid whose points are a subset of these
        the weights the grid gets are exactly those of the values themselves
        (cloud in cell is exact for piecewise linear hat functions). Values
        beyond the points are clamped onto the end points.

        Args:
            values (np.ndarray of float):
                The values (any shape).
            points (np.ndarray of float):
                The increasing points.

        Returns:
            np.ndarray:
                The weights (summing to one) on the zero width bins at the
                points, with the empty bins between them, shape
                values.shape + (2 * len(points) - 1,).
        """
        weights = np.zeros(np.shape(values) + (2 * points.size - 1,))
        if points.size == 1:
            weights[..., 0] = 1.0
            return weights
        log = points[0] > 0
        coords = np.log10(points) if log else points
        values = np.asarray(values, dtype=np.float64)
        if log:
            # Values at or below zero are below the points
            positive = values > 0
            x = np.log10(np.where(positive, values, 1.0))
            x = np.where(positive, x, coords[0])
        else:
            x = values
        x = np.clip(x, coords[0], coords[-1])
        j = np.clip(np.searchsorted(coords, x, side="right") - 1, 0, None)
        j = np.minimum(j, points.size - 2)
        frac = (x - coords[j]) / (coords[j + 1] - coords[j])
        index = np.indices(np.shape(values))
        weights[(*index, 2 * j)] = 1.0 - frac
        weights[(*index, 2 * j + 2)] += frac
        return weights

    @classmethod
    def _add_extra_axes(
        cls, edges, masses, extra_axes, units=None, age_samples=None
    ):
        """Add distributions along other axes to a set of bins.

        Each distribution is combined with the existing bins as an outer
        product, so it is the same for every existing bin (use from_binned
        with extra_edges for a joint distribution).

        Args:
            edges (dict):
                The existing bin edges keyed by axis name.
            masses (np.ndarray):
                The existing bin masses.
            extra_axes (dict):
                The distribution along each new axis keyed by its name. Each
                is a single value (float or unyt_quantity), or a tuple of
                bin edges and either the weight in each bin or a
                distribution with a get_bin_weights(edges) method. Weights
                are normalised, so the mass is unchanged. A value that
                depends on age is a tuple of points and a function of age
                (called with the age as a unyt_array), split linearly
                between the points, and a distribution can depend on age
                (see ZDist); both are averaged over the mass formed in each
                age bin.
            units (dict):
                The units of the existing edges keyed by axis name (ages
                default to yr).
            age_samples (tuple):
                The age samples of each age bin and their weights (see
                _get_age_samples), needed for age dependent distributions.

        Returns:
            tuple:
                The bin edges, the bin masses and the units of the edges
                (None if unitless) keyed by axis name.
        """
        edges = dict(edges)
        units = {
            axis: (units or {}).get(axis, yr if axis == "ages" else None)
            for axis in edges
        }
        for axis, dist in (extra_axes or {}).items():
            if axis in edges:
                raise exceptions.InconsistentArguments(
                    f"There are already bins along {axis}."
                )

            # Split the distribution into edges and weights
            if callable(dist):
                raise exceptions.InconsistentArguments(
                    f"An age dependent {axis} needs the points to place it "
                    "on: give (points, function) instead."
                )
            if isinstance(dist, tuple):
                axis_edges, weights = dist
            else:
                axis_edges, weights = [dist, dist], np.ones(1)

            # Strip (and remember) any units
            axis_units = None
            if isinstance(axis_edges, unyt_array):
                axis_units = axis_edges.units
                axis_edges = axis_edges.ndview
            elif isinstance(axis_edges[0], unyt_quantity):
                axis_units = axis_edges[0].units
                axis_edges = [e.to(axis_units).value for e in axis_edges]
            axis_edges = np.asarray(axis_edges, dtype=np.float64)
            if np.any(np.diff(axis_edges) < 0):
                raise exceptions.InconsistentArguments(
                    f"The {axis} bin edges must be increasing."
                )

            # An age dependent value or distribution differs between age
            # bins, so it is averaged over the mass formed in each
            if callable(weights) or getattr(weights, "depends_on_age", False):
                if age_samples is None:
                    raise exceptions.InconsistentArguments(
                        f"Can't give {axis} an age dependence here."
                    )
                ages, sample_weights = age_samples
                if callable(weights):
                    values = weights(ages * yr)
                    if isinstance(values, unyt_array) and axis_units:
                        values = values.to(axis_units)
                    if isinstance(values, unyt_array):
                        values = values.to("dimensionless").ndview
                    per_sample = cls._deposit_on_points(values, axis_edges)
                    axis_edges = np.repeat(axis_edges, 2)
                else:
                    per_sample = cls._get_sample_bin_weights(
                        weights, ages, axis_edges
                    )
                age_weights = cls._average_over_samples(
                    per_sample, sample_weights
                )
                shape = (1, age_weights.shape[0]) + (1,) * (masses.ndim - 2)
                edges[axis] = axis_edges
                units[axis] = axis_units
                masses = masses[..., None] * age_weights.reshape(
                    shape + (age_weights.shape[1],)
                )
                continue

            # Get the weights of a distribution in the bins
            if hasattr(weights, "get_bin_weights"):
                weights = weights.get_bin_weights(axis_edges)
            weights = np.asarray(weights, dtype=np.float64)
            if weights.shape != (axis_edges.size - 1,):
                raise exceptions.InconsistentArguments(
                    f"Got {weights.size} {axis} weights for "
                    f"{axis_edges.size - 1} bins."
                )

            edges[axis] = axis_edges
            units[axis] = axis_units
            masses = masses[..., None] * (weights / np.sum(weights))

        return edges, masses, units

    @staticmethod
    def _marginalise(edges, masses, keep):
        """Sum bins over every axis not in keep.

        Args:
            edges (dict):
                The bin edges keyed by axis name, in the order of the masses'
                axes (after the population axis).
            masses (np.ndarray):
                The bin masses.
            keep (iterable of str):
                The axes to keep.

        Returns:
            tuple:
                The edges and masses along the kept axes.
        """
        axes = list(edges)
        drop = tuple(i + 1 for i, axis in enumerate(axes) if axis not in keep)
        if drop:
            masses = np.sum(masses, axis=drop)
        return {axis: edges[axis] for axis in axes if axis in keep}, masses

    def _get_bins_at_earlier_time(self, age_offset):
        """Get the bins describing this population at an earlier time.

        At a lookback time of age_offset every star was younger by
        age_offset and stars younger than it hadn't formed. So the bins are
        clipped to ages above age_offset (keeping only the mass in the part
        of each bin that had formed) and shifted down by it. Mass exactly at
        the offset is kept, with an age of zero.

        Args:
            age_offset (unyt_quantity):
                The lookback time.

        Returns:
            tuple:
                The bin edges keyed by axis name and the bin masses.
        """
        # Get the offset in years, snapped onto a bin edge it matches so
        # that edge lands exactly on zero rather than a round off either
        # side of it
        offset = self._snap_to_bin_edge(
            "ages", convert_in_place(age_offset, yr).value
        )

        # Get the fraction of each age bin that had formed by then. Mass is
        # uniform within a finite bin, and a zero width bin has formed if
        # it isn't younger than the offset
        ages = self._bin_edges["ages"]
        lo, hi = ages[:-1], ages[1:]
        finite = hi > lo
        formed = np.where(
            finite,
            np.clip((hi - offset) / np.where(finite, hi - lo, 1.0), 0.0, 1.0),
            lo >= offset,
        )

        # Clip the age edges to the offset and shift them down by it
        edges = dict(self._bin_edges)
        edges["ages"] = np.maximum(ages, offset) - offset

        # Keep only the mass that had formed
        shape = [1] * self._bin_masses.ndim
        shape[self.bin_axes.index("ages") + 1] = formed.size
        return edges, self._bin_masses * formed.reshape(shape)

    @accepts(age_offset=yr)
    def get_at_earlier_time(self, age_offset):
        """Get a Stars object representing the population at an earlier time.

        The new Stars object holds the bins of this population shifted to
        the earlier time (see _get_bins_at_earlier_time), so the stars that
        hadn't formed yet are removed exactly.

        Args:
            age_offset (unyt_quantity):
                The offset to apply to the age grid.

        Returns:
            Stars:
                New Stars object on the requested grid.
        """
        return self._from_bins(
            *self._get_bins_at_earlier_time(age_offset),
            names=self.population_names,
            **self._get_population_params(),
        )

    def _set_bins(self, edges, masses, units=None):
        """Set the bins describing this population.

        Args:
            edges (dict):
                The bin edges along each axis, in linear units (ages in yr),
                keyed by axis name in the order of the masses' axes (ages and
                metallicities first).
            masses (np.ndarray):
                The mass in each bin (Msun) with shape (npop, nbins_0, ...,
                nbins_N).
            units (dict):
                The units of the edges (None if unitless) keyed by axis
                name, defaults to the current units (ages in yr).
        """
        # Store the units of the edges along each axis
        old_units = getattr(self, "_bin_units", {})
        self._bin_units = {
            axis: (units or {}).get(
                axis, old_units.get(axis, yr if axis == "ages" else None)
            )
            for axis in edges
        }

        # Store the bins
        self._bin_edges = edges
        self._bin_masses = masses

        # Anything derived from the bins is now stale
        self._sfzh_view = None
        self._grid_weights = {"cic": {}, "ngp": {}}

        # Keep the SFH and metallicity distribution on the axes current
        self.sf_hist = np.sum(self.sfzh, axis=1)
        self.metal_dist = np.sum(self.sfzh, axis=0)

    def _sfzh_to_bins(self, sfzh):
        """Convert an SFZH defined at the points of the axes into bins.

        Each point of the age and metallicity axes becomes a zero width bin,
        with the (empty) finite bins between them. Mapping these onto any
        grid is then exactly cloud in cell, and onto this object's own axes
        it returns the SFZH unchanged.

        Args:
            sfzh (np.ndarray):
                The mass (Msun) at each (age, metallicity) point.

        Returns:
            tuple:
                The bin edges keyed by axis name and the bin masses.
        """
        # Repeating each point gives a zero width bin at it, with a finite
        # bin between each pair of points
        age_edges = np.repeat(convert_in_place(self.ages, yr).ndview, 2)
        metal_edges = np.repeat(
            np.asarray(self.metallicities, dtype=np.float64), 2
        )

        # Put the SFZH in the zero width bins (even indices on both axes),
        # leaving the finite bins between them empty
        masses = np.zeros((1, age_edges.size - 1, metal_edges.size - 1))
        masses[0, ::2, ::2] = sfzh
        return {"ages": age_edges, "metallicities": metal_edges}, masses

    @timed("ParametricStars._bins_to_axes")
    def _bins_to_axes(self, edges, masses, mask=None):
        """Map bins onto this object's age and metallicity axes.

        Args:
            edges (dict):
                The bin edges keyed by axis name.
            masses (np.ndarray):
                The bin masses.
            mask (np.ndarray of float):
                The allowed range along each bin axis (see get_mask), or
                None.

        Returns:
            np.ndarray:
                The mass (Msun) at each (age, metallicity) point.
        """
        # Imported here to avoid importing the extension at module load
        from synthesizer.extensions.parametric_spectra import (
            compute_parametric_weights,
        )

        # Use log10 metallicity axes like the grids unless a metallicity of
        # zero makes that impossible
        log_metals = bool(np.all(self.metallicities > 0))
        metal_axis = (
            self.log10metallicities if log_metals else self.metallicities
        )

        # Sum over any other axes, the mask then only applies to the ages
        # and metallicities
        edges, masses = self._marginalise(
            edges, masses, ("ages", "metallicities")
        )
        if mask is not None:
            mask = np.ascontiguousarray(mask[:2])

        # Spread the bins onto the axis points (ages are always log10)
        return compute_parametric_weights(
            (
                np.ascontiguousarray(self.log10ages, dtype=np.float64),
                np.ascontiguousarray(metal_axis, dtype=np.float64),
            ),
            tuple(
                np.ascontiguousarray(edges[axis], dtype=np.float64)
                for axis in ("ages", "metallicities")
            ),
            np.ascontiguousarray(masses, dtype=np.float64),
            (True, log_metals),
            (0, 1),
            (1.0, 1.0),
            1,
            mask,
            (
                "log10ages",
                "log10metallicities" if log_metals else "metallicities",
            ),
        )

    def _get_unbinned_axis_value(self, grid, axis_name, model=None):
        """Get the value along a grid axis this population has no bins on.

        The value is looked up exactly as particle attributes are (on the
        emission model's fixed parameters, then on this object).

        Args:
            grid (Grid):
                The grid the axis belongs to.
            axis_name (str):
                The name of the grid's axis (with any log10 prefix).
            model (EmissionModel):
                The emission model, or None.

        Returns:
            np.ndarray of float:
                The linear value(s) in the grid's units.

        Raises:
            MissingAttribute:
                If there is no value for the axis.
        """
        # Get the name of the axis without any log10 prefix
        log = axis_name.startswith("log10")
        axis = axis_name[5:] if log else axis_name

        # Look the value up on the model then this object
        try:
            value = get_param(axis_name, model, None, self)
        except exceptions.MissingAttribute:
            raise exceptions.MissingAttribute(
                f"This parametric Stars has no bins or value along {axis}. "
                "Give it extra_axes, or set a value on the emission model "
                "or the Stars."
            ) from None

        # Get the linear value in the grid's units
        if isinstance(value, (unyt_array, unyt_quantity)):
            value = convert_in_place(value, grid._axes_units[axis]).ndview
        value = np.atleast_1d(np.asarray(value, dtype=np.float64))
        if log:
            value = 10**value
        return value

    def get_fraction_outside_grid(self, grid, model=None):
        """Get the fraction of the mass outside a grid's axes.

        Mass outside the grid is clamped onto the grid's edges when the
        emission is extracted (as it is for particles), so this is the
        fraction of the mass whose emission is only approximate. Mass is
        uniform within each bin, so a bin straddling a grid edge is only
        partly outside.

        Args:
            grid (Grid):
                The grid to compare to.
            model (EmissionModel):
                The emission model, for values along axes with no bins, or
                None.

        Returns:
            float:
                The fraction of the mass outside the grid's axes.

        Raises:
            InconsistentArguments:
                If a grid axis without bins doesn't have one value or one
                per population.
        """
        # Nothing is outside the grid if there is no mass
        total = np.sum(self._bin_masses)
        if total == 0.0:
            return 0.0

        # Scale the masses by the fraction inside the grid along each of
        # its axes in turn
        inside = self._bin_masses
        for axis_name in grid._extract_axes:
            # Get the (linear) range of the grid along this axis
            log = axis_name.startswith("log10")
            axis = axis_name[5:] if log else axis_name
            values = grid._extract_axes_values[axis_name]
            gmin, gmax = (
                (10 ** values[0], 10 ** values[-1])
                if log
                else (values[0], values[-1])
            )

            # Without bins along this axis each population is entirely
            # inside or outside at its value
            if axis not in self._bin_edges:
                value = self._get_unbinned_axis_value(grid, axis_name, model)
                if value.size not in (1, self.npop):
                    raise exceptions.InconsistentArguments(
                        f"Got {value.size} values of {axis} for "
                        f"{self.npop} populations."
                    )
                frac = (value >= gmin) & (value <= gmax)
                shape = [-1] + [1] * (inside.ndim - 1)
                inside = inside * frac.reshape(shape)
                continue

            # Otherwise get the bin edges in the grid's units
            units = self._bin_units[axis]
            edges = self._bin_edges[axis]
            if units is not None:
                edges = edges * get_attr_unit_conversion(
                    units, grid._axes_units[axis]
                )

            # Get the fraction of each bin inside the grid, a zero width
            # bin is either inside or not
            lo, hi = edges[:-1], edges[1:]
            width = hi - lo
            overlap = np.clip(
                np.minimum(hi, gmax) - np.maximum(lo, gmin), 0.0, None
            )
            frac = np.where(
                width > 0,
                overlap / np.where(width > 0, width, 1.0),
                (lo >= gmin) & (lo <= gmax),
            )
            shape = [1] * inside.ndim
            shape[self.bin_axes.index(axis) + 1] = -1
            inside = inside * frac.reshape(shape)

        return float(1.0 - np.sum(inside) / total)

    def _bins_to_grid(self, grid, edges, masses):
        """Map bins onto a grid's axes.

        Bin axes the grid doesn't have are summed over, and a grid axis with
        no bins has all the mass at a single value (see
        _get_unbinned_axis_value).

        Args:
            grid (Grid):
                The grid to map onto, whose axes are matched to the bin axes
                by name (with any log10 prefix removed).
            edges (dict):
                The bin edges keyed by axis name.
            masses (np.ndarray):
                The bin masses.

        Returns:
            np.ndarray:
                The mass (Msun) at each grid point.

        Raises:
            MissingAttribute:
                If the grid has an axis this object has no bins or value
                along.
            InconsistentArguments:
                If a grid axis without bins has a value per population.
        """
        # Imported here to avoid importing the extension at module load
        from synthesizer.extensions.parametric_spectra import (
            compute_parametric_weights,
        )

        # Sum over the bin axes the grid doesn't have
        grid_bin_axes = [
            a[5:] if a.startswith("log10") else a for a in grid._extract_axes
        ]
        edges, masses = self._marginalise(edges, masses, grid_bin_axes)
        axes = list(edges)

        # Set up lists for the bin edges along each grid axis, whether the
        # grid axis is log10, the bin axis matching it and the factor
        # converting the edges into the grid's units
        grid_edges = []
        log_flags = []
        bin_axes = []
        unit_scales = []

        # Loop over the grid's axes matching each to a bin axis
        for axis_name, bin_axis in zip(grid._extract_axes, grid_bin_axes):
            log_flags.append(axis_name.startswith("log10"))

            # Without bins along this axis all the mass is at one value
            # (already in the grid's units), on a new trailing axis
            if bin_axis not in edges:
                value = self._get_unbinned_axis_value(grid, axis_name)
                if value.size != 1:
                    # TODO: no value per population here, expand the
                    # masses as the extractor does if this is ever needed
                    raise exceptions.InconsistentArguments(
                        f"Mapping onto a grid needs a single {bin_axis}, "
                        f"got one per population."
                    )
                grid_edges.append(np.repeat(value, 2))
                bin_axes.append(masses.ndim - 1)
                unit_scales.append(1.0)
                masses = masses[..., None]
                continue

            # Otherwise use the bin edges, with the factor converting them
            # into the grid's units if they have any
            grid_edges.append(
                np.ascontiguousarray(edges[bin_axis], dtype=np.float64)
            )
            bin_axes.append(axes.index(bin_axis))
            units = self._bin_units[bin_axis]
            unit_scales.append(
                1.0
                if units is None
                else get_attr_unit_conversion(
                    units, grid._axes_units[bin_axis]
                )
            )

        # Spread the bins onto the grid points
        return compute_parametric_weights(
            tuple(grid._extract_axes_values[a] for a in grid._extract_axes),
            tuple(grid_edges),
            np.ascontiguousarray(masses, dtype=np.float64),
            tuple(log_flags),
            tuple(bin_axes),
            tuple(unit_scales),
            1,
            None,
            tuple(grid._extract_axes),
        )

    def _get_population_params(self):
        """Get the parameters and morphology describing the populations.

        Returns:
            dict:
                The arguments giving a new Stars the same parameters and
                morphology as this one.
        """
        params = {
            name: getattr(self, name, None) for name in self._parameter_names
        }
        params["morphology"] = self.morphology
        return params

    def _from_bins(
        self,
        edges,
        masses,
        log10ages=None,
        metallicities=None,
        names=None,
        units=None,
        **kwargs,
    ):
        """Make a new Stars holding the given bins.

        Args:
            edges (dict):
                The bin edges keyed by axis name.
            masses (np.ndarray):
                The bin masses.
            log10ages (np.ndarray of float):
                The log10 age axis, defaults to this object's.
            metallicities (np.ndarray of float):
                The metallicity axis, defaults to this object's.
            names (list of str):
                A name for each population, or None.
            units (dict):
                The units of the edges keyed by axis name, defaults to this
                object's.
            **kwargs (dict):
                Any other arguments for the new Stars.

        Returns:
            Stars:
                The new Stars.
        """
        if log10ages is None:
            log10ages = self.log10ages
        if metallicities is None:
            metallicities = self.metallicities
        new = Stars(
            log10ages,
            metallicities,
            sfzh=np.zeros((len(log10ages), len(metallicities))),
            **kwargs,
        )
        new._set_bins(
            edges, masses, self._bin_units if units is None else units
        )
        new.initial_mass = (np.sum(masses) * Msun).in_base("galactic")
        new.population_names = None if names is None else list(names)
        return new

    def get_population_photo_lnu(self, *args, **kwargs):
        """Calculate the luminosity photometry of each population.

        This takes the same arguments as Particles.get_particle_photo_lnu,
        with one row per population in place of one per particle.

        Returns:
            dict:
                The PhotometryCollection of each population's spectra, keyed
                by spectra label (also stored in population_photo_lnu).
        """
        from synthesizer.particle.particles import Particles

        return Particles.get_particle_photo_lnu(self, *args, **kwargs)

    def get_population_photo_fnu(self, *args, **kwargs):
        """Calculate the flux photometry of each population.

        This takes the same arguments as Particles.get_particle_photo_fnu,
        with one row per population in place of one per particle.

        Returns:
            dict:
                The PhotometryCollection of each population's spectra, keyed
                by spectra label (also stored in population_photo_fnu).
        """
        from synthesizer.particle.particles import Particles

        return Particles.get_particle_photo_fnu(self, *args, **kwargs)

    @property
    def population_spectra(self):
        """The spectra of each population, keyed by label.

        Returns:
            dict:
                The Sed of each label, with one row per population.
        """
        return self.particle_spectra

    @property
    def population_lines(self):
        """The lines of each population, keyed by label.

        Returns:
            dict:
                The LineCollection of each label, with one row per
                population.
        """
        return self.particle_lines

    @property
    def population_photo_lnu(self):
        """The luminosity photometry of each population, keyed by label.

        Returns:
            dict:
                The PhotometryCollection of each label, with one row per
                population.
        """
        return self.particle_photo_lnu

    @property
    def population_photo_fnu(self):
        """The flux photometry of each population, keyed by label.

        Returns:
            dict:
                The PhotometryCollection of each label, with one row per
                population.
        """
        return self.particle_photo_fnu

    @property
    def bin_axes(self):
        """The names of the axes the population is binned along.

        Returns:
            tuple of str:
                The bin axis names in the order of the bin_masses axes.
        """
        return tuple(self._bin_edges)

    @property
    def bin_masses(self):
        """The mass in each bin (Msun).

        Returns:
            np.ndarray:
                The masses with shape (npop, nbins_0, ..., nbins_N).
        """
        return self._bin_masses

    @property
    def npop(self):
        """The number of populations.

        Returns:
            int:
                The number of populations.
        """
        return self._bin_masses.shape[0]

    def get_bin_edges(self, axis, values_only=False):
        """Get the bin edges along an axis.

        Args:
            axis (str):
                The axis name (one of bin_axes).
            values_only (bool):
                Return the values without units.

        Returns:
            unyt_array/np.ndarray:
                The bin edges in linear units.

        Raises:
            MissingAttribute:
                If the population is not binned along the axis.
        """
        # Ensure we have bins along this axis
        if axis not in self._bin_edges:
            raise exceptions.MissingAttribute(
                f"This parametric Stars has no bins along {axis} (it is "
                f"binned along {self.bin_axes})."
            )
        # Attach the units of the edges (if they have any) to a view
        edges = self._bin_edges[axis]
        units = self._bin_units[axis]
        if values_only or units is None:
            return edges
        return unyt_array(edges, units, bypass_validation=True)

    @property
    def sfzh(self):
        """The SFZH on this object's age and metallicity axes.

        This is a read-only view of the bins mapped onto the points of the
        age and metallicity axes with the bin in cell approach (see
        synthesizer.extensions.parametric_spectra).

        Returns:
            np.ndarray:
                The mass (Msun) at each (age, metallicity) point.
        """
        # Compute the view lazily, caching it (read-only so the cache can't
        # drift from the bins) until the bins change
        if self._sfzh_view is None:
            view = self._bins_to_axes(self._bin_edges, self._bin_masses)
            view.setflags(write=False)
            self._sfzh_view = view
        return self._sfzh_view

    @timed("Stars.get_mask")
    def get_mask(
        self,
        attr,
        thresh,
        op,
        mask=None,
        attr_override_obj=None,
    ):
        """Return a mask based on the attribute and threshold.

        Will derive a mask of the form attr op thresh, e.g. age > 10 Myr.

        A condition on a bin axis (ages or metallicities, or their log10)
        can cut through a bin, so rather than flagging bins the mask is the
        allowed [lo, hi] range (inclusive, in the linear units of the bin
        edges) along each bin axis. The extraction clips each bin to this
        range and keeps the fraction of its mass inside it. A strict
        inequality moves the bound by one floating point step, so a zero
        width bin exactly on the threshold is excluded just as a particle
        would be. "!=" can't be expressed as a single range so isn't
        supported on a bin axis. An open end of a range is the largest
        finite float rather than infinity, which the extensions can't handle.

        Conditions on a population level value (e.g. a fixed model parameter)
        include or exclude whole populations, a failing one empties the
        range. A value per population can exclude only some of the
        populations, in which case the mask holds a range for each of them.

        Args:
            attr (str):
                The attribute to derive the mask from.
            thresh (float):
                The threshold value.
            op (str):
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=" ("!=" only on population level values).
            mask (np.ndarray of float):
                Optionally, a mask to combine with the new mask.
            attr_override_obj (object):
                An alternative object to check from the attribute. This
                is specifically used when an EmissionModel may have a
                fixed parameter override, but can be used more generally.

        Returns:
            mask (np.ndarray of float):
                The combined mask, the allowed range along each bin axis with
                shape (len(bin_axes), 2), or (npop, len(bin_axes), 2) if it
                treats the populations differently.

        Raises:
            InconsistentArguments:
                If the operator is unknown or "!=" is used on a bin axis, the
                mask to combine with has the wrong shape, or the attribute is
                neither a bin axis nor single valued.
        """
        # Reject unknown operators
        if op not in ("<", ">", "<=", ">=", "==", "!="):
            raise exceptions.InconsistentArguments(
                "Masking operation must be '<', '>', '<=', '>=', '==', or "
                f"'!=', not {op}"
            )

        # Start from a copy of the mask we're combining with, or an open
        # range along every axis
        nbin_axes = len(self.bin_axes)
        if mask is None:
            new_mask = np.empty((nbin_axes, 2))
            new_mask[:, 0] = -_MASK_BOUND
            new_mask[:, 1] = _MASK_BOUND
        elif np.shape(mask) in (
            (nbin_axes, 2),
            (self.npop, nbin_axes, 2),
        ):
            new_mask = np.array(mask, dtype=np.float64)
        else:
            raise exceptions.InconsistentArguments(
                "Parametric Stars masks must be the allowed range along each "
                f"bin axis with shape ({nbin_axes}, 2) or ({self.npop}, "
                f"{nbin_axes}, 2), got {np.shape(mask)}."
            )

        # Resolve a string threshold as an attribute alias on the emitter
        if isinstance(thresh, str):
            thresh = get_param(
                thresh, attr_override_obj, None, self, preserve_units=True
            )

        # A fixed model parameter applies to the whole population
        override = None
        if attr in getattr(attr_override_obj, "fixed_parameters", {}):
            override = get_param(
                attr, attr_override_obj, None, self, preserve_units=True
            )

        # Otherwise, is this a condition on a bin axis?
        axis = attr[5:] if attr.startswith("log10") else attr
        if override is None and axis in self.bin_axes:
            # "!=" would need a range with a hole in it
            if op == "!=":
                raise exceptions.InconsistentArguments(
                    f"Can't mask a parametric Stars on {attr} with '!=', "
                    "masks on the bin axes must be a single range."
                )

            # Narrow the axis' range, moving strict bounds by one step so a
            # value exactly on the threshold fails
            iaxis = self.bin_axes.index(axis)
            value = self._get_axis_threshold(thresh, attr, axis)
            if op in ("<", "<=", "=="):
                hi = np.nextafter(value, -np.inf) if op == "<" else value
                np.minimum(
                    new_mask[..., iaxis, 1], hi, out=new_mask[..., iaxis, 1]
                )
            if op in (">", ">=", "=="):
                lo = np.nextafter(value, np.inf) if op == ">" else value
                np.maximum(
                    new_mask[..., iaxis, 0], lo, out=new_mask[..., iaxis, 0]
                )
            return new_mask

        # Anything else must be a population level value, either one for all
        # populations or one per population
        value = (
            override
            if override is not None
            else get_param(attr, None, None, self, preserve_units=True)
        )
        if np.size(value) not in (1, self.npop):
            raise exceptions.InconsistentArguments(
                f"Can't mask a parametric Stars on {attr}: only the bin axes "
                f"({self.bin_axes}) and quantities with one value (or one "
                "per population) can be masked."
            )

        # Apply the operator
        if op == ">":
            include = value > thresh
        elif op == "<":
            include = value < thresh
        elif op == ">=":
            include = value >= thresh
        elif op == "<=":
            include = value <= thresh
        elif op == "==":
            include = value == thresh
        else:
            include = value != thresh

        # A failing population level condition excludes a whole population,
        # which we mark by emptying its first axis' range. Unless every
        # population agrees this needs a range per population
        include = np.atleast_1d(np.asarray(include, dtype=bool))
        if include.all():
            return new_mask
        if include.size > 1 and include.any() and new_mask.ndim == 2:
            new_mask = np.tile(new_mask, (self.npop, 1, 1))
        if new_mask.ndim == 3:
            new_mask[~np.broadcast_to(include, (self.npop,)), 0] = (
                _MASK_BOUND,
                -_MASK_BOUND,
            )
        else:
            new_mask[0] = (_MASK_BOUND, -_MASK_BOUND)
        return new_mask

    def get_population_mask(self, mask):
        """Get which populations a mask includes.

        The ranges along the bin axes act on the bins when the emission is
        extracted, so only a population level condition can exclude a whole
        population.

        Args:
            mask (np.ndarray of float):
                A mask from get_mask, or None.

        Returns:
            np.ndarray of bool:
                Whether each population is included, or None if the mask
                includes them all.
        """
        # No mask includes every population
        if mask is None:
            return None

        # A single set of ranges treats every population alike: an empty
        # first range (e.g. a condition they all fail) excludes them all
        if np.ndim(mask) == 2:
            if mask[0, 0] <= mask[0, 1]:
                return None
            return np.zeros(self.npop, dtype=bool)

        # An excluded population has an empty first range
        return mask[:, 0, 0] <= mask[:, 0, 1]

    def _get_axis_threshold(self, thresh, attr, axis):
        """Convert a mask threshold into the linear units of a bin axis.

        Args:
            thresh (float/unyt_quantity):
                The threshold as given to the mask.
            attr (str):
                The masked attribute (e.g. "log10ages" or "ages").
            axis (str):
                The bin axis the attribute refers to (e.g. "ages").

        Returns:
            float:
                The threshold in the units of the axis' bin edges, snapped
                onto a bin edge it matches to within round off.

        Raises:
            InconsistentArguments:
                If the axis has units but the threshold does not.
        """
        # A log10 attribute's threshold is the log10 of the linear value
        if attr.startswith("log10"):
            if isinstance(thresh, unyt_quantity):
                thresh = thresh.ndview
            return self._snap_to_bin_edge(axis, float(thresh), log=True)

        # Otherwise convert any units to the edge units
        elif isinstance(thresh, unyt_quantity):
            if self._bin_units[axis] is not None:
                value = float(
                    convert_in_place(thresh, self._bin_units[axis]).ndview
                )
            else:
                value = float(thresh.ndview)

        # A unitless threshold can't be compared with an axis that has units
        elif self._bin_units[axis] is not None:
            raise exceptions.InconsistentArguments(
                f"Masking attribute ({attr}) has units but threshold does not "
                f"({thresh})."
            )
        else:
            value = float(thresh)

        return self._snap_to_bin_edge(axis, value)

    def _snap_to_bin_edge(self, axis, value, log=False):
        """Snap a value onto a bin edge it matches to within round off.

        The edges and the value reach linear units by different routes (e.g.
        10**log10age or a unit conversion), which can differ in the last few
        bits. That would put a zero width bin lying exactly on the value on
        the wrong side of it, so a value matching an edge becomes that edge.

        The comparison is made in the space the value was given in, since a
        round off in a log10 value is amplified when it is made linear.

        Args:
            axis (str):
                The bin axis to snap onto (e.g. "ages").
            value (float):
                The value, in the units of the axis' bin edges (or their
                log10 if log is True).
            log (bool):
                Whether the value is the log10 of a point on the axis.

        Returns:
            float:
                The value in the linear units of the axis' bin edges, equal
                to a bin edge if it matched one.
        """
        # Get the linear value and the edges either side of it
        linear = 10**value if log else value
        edges = self._bin_edges[axis]
        upper = min(np.searchsorted(edges, linear), edges.size - 1)

        # Return an edge the value matches to within a few floating point
        # steps (an edge at zero has no log10 to match)
        tol = 4 * np.spacing(abs(value))
        for edge in (edges[max(upper - 1, 0)], edges[upper]):
            if log and edge <= 0:
                continue
            if abs((np.log10(edge) if log else edge) - value) <= tol:
                return float(edge)
        return linear

    def calculate_median_age(self):
        """Calculate the median age of the stellar population."""
        return weighted_median(self.ages, self.sf_hist) * self.ages.units

    def calculate_mean_age(self):
        """Calculate the mean age of the stellar population."""
        return weighted_mean(self.ages, self.sf_hist)

    def calculate_mean_metallicity(self):
        """Calculate the mean metallicity of the stellar population."""
        return weighted_mean(self.metallicities, self.metal_dist)

    @classmethod
    def from_binned(
        cls,
        log10ages,
        metallicities,
        age_edges,
        metallicity_edges,
        masses,
        names=None,
        extra_edges=None,
        extra_axes=None,
        **kwargs,
    ):
        """Create a Stars from binned masses, e.g. the output of a SAM.

        The mass in each (age, metallicity) bin is spread uniformly over the
        bin, so the bins are used exactly as given: nothing is interpolated
        onto the axes. A zero width bin (an edge repeated) is a single value,
        e.g. a single metallicity for the mass formed in an age bin.

        Args:
            log10ages (np.ndarray of float):
                The log10 age axis (used for the sfzh view and plots).
            metallicities (np.ndarray of float):
                The metallicity axis (used for the sfzh view and plots).
            age_edges (unyt_array/np.ndarray of float):
                The age bin edges (in years if unitless), either increasing or
                decreasing (as lookback time bins often are).
            metallicity_edges (np.ndarray of float):
                The increasing metallicity bin edges.
            masses (unyt_array/np.ndarray of float):
                The mass formed in each bin (in Msun if unitless), with shape
                (n_age_bins, n_metallicity_bins, ...) for one population or
                (npop, n_age_bins, n_metallicity_bins, ...) for several, with
                a trailing axis for each of extra_edges.
            names (list of str):
                A name for each population.
            extra_edges (dict):
                The increasing bin edges along any other axes the masses are
                binned on jointly (e.g. ionisation_parameter), keyed by axis
                name in the order of the masses' trailing axes. Edges with
                units are converted to the grid's units at extraction.
            extra_axes (dict):
                Distributions along further axes combined with the bins as an
                outer product (see Stars).
            **kwargs (dict):
                Any other arguments for the Stars, e.g. per population
                parameters such as tau_v (one value per population).

        Returns:
            Stars:
                The binned population(s).
        """
        # Get the edges and masses as doubles in yr and Msun (the masses
        # are copied so scaling them never touches the caller's array)
        if isinstance(age_edges, unyt_array):
            age_edges = convert_in_place(age_edges, yr).ndview
        age_edges = np.asarray(age_edges, dtype=np.float64)
        metallicity_edges = np.asarray(metallicity_edges, dtype=np.float64)
        if isinstance(masses, unyt_array):
            masses = convert_in_place(masses, Msun).ndview
        masses = np.array(masses, dtype=np.float64)

        # Collect the edges (and their units) along every axis
        edges = {"ages": age_edges, "metallicities": metallicity_edges}
        units = {"ages": yr, "metallicities": None}
        for axis, axis_edges in (extra_edges or {}).items():
            units[axis] = None
            if isinstance(axis_edges, unyt_array):
                units[axis] = axis_edges.units
                axis_edges = axis_edges.ndview
            edges[axis] = np.asarray(axis_edges, dtype=np.float64)

        # A single population has no population axis yet
        if masses.ndim == len(edges):
            masses = masses[None]

        # Accept decreasing (lookback) age edges
        if age_edges.size > 1 and age_edges[0] > age_edges[-1]:
            edges["ages"] = age_edges[::-1]
            masses = masses[:, ::-1]

        # Check the masses match the edges and the edges are monotonic
        nbins = tuple(axis_edges.size - 1 for axis_edges in edges.values())
        if masses.shape[1:] != nbins:
            raise exceptions.InconsistentArguments(
                f"masses has shape {masses.shape} but the edges define "
                f"{nbins} bins."
            )
        if any(np.any(np.diff(e) < 0) for e in edges.values()):
            raise exceptions.InconsistentArguments(
                "The bin edges must be monotonic."
            )

        # Add any distributions along further axes
        edges, masses, units = cls._add_extra_axes(
            edges,
            masses,
            extra_axes,
            units,
            age_samples=(
                cls._get_age_samples(edges["ages"]) if extra_axes else None
            ),
        )

        stars = cls(
            log10ages,
            metallicities,
            sfzh=np.zeros((len(log10ages), len(metallicities))),
            **kwargs,
        )
        stars._set_bins(
            {
                axis: np.ascontiguousarray(axis_edges)
                for axis, axis_edges in edges.items()
            },
            np.ascontiguousarray(masses),
            units,
        )
        stars.initial_mass = (np.sum(masses) * Msun).in_base("galactic")
        if names is not None:
            if len(names) != stars.npop:
                raise exceptions.InconsistentArguments(
                    f"Got {len(names)} names for {stars.npop} populations."
                )
            stars.population_names = list(names)
        return stars

    @classmethod
    def from_populations(cls, populations, names=None):
        """Combine Stars into a single Stars holding all their populations.

        Every population is kept separately (along the leading population
        axis of the bin masses), so the integrated emission of the combined
        Stars is extracted for all of them in a single call, exactly as for
        the particles of a particle Stars. The populations are rebinned onto
        the union of their edges along each axis (splitting a bin's uniformly
        spread mass between the pieces it is cut into) so they share edges.

        Parameters describing the populations (fesc, fesc_ly_alpha and any
        other keyword argument given to their Stars, e.g. tau_v) become per
        population parameters (one value per population) when they differ.
        Populations must share a morphology to keep it, otherwise the
        combined Stars has none.

        This will only work for Stars objects with the same axes.

        Args:
            populations (list of parametric.Stars):
                The Stars to combine.
            names (list of str):
                A name for each population, used to look populations up
                (e.g. stars["bulge"]). Defaults to the populations' own names
                if they all have them.

        Returns:
            Stars:
                A Stars holding every population.
        """
        first = populations[0]
        for other in populations[1:]:
            if not (
                np.array_equal(first.log10ages, other.log10ages)
                and np.array_equal(first.metallicities, other.metallicities)
            ):
                raise exceptions.InconsistentAddition(
                    "Stars can only be combined if they have the same axes"
                )
            if other._bin_units != first._bin_units:
                raise exceptions.InconsistentAddition(
                    "Stars can only be combined if they are binned along the "
                    f"same axes in the same units (got {first._bin_units} "
                    f"and {other._bin_units})"
                )

        # Rebin every population onto the union of all the edges
        edges = {}
        all_masses = [stars.bin_masses for stars in populations]
        for iaxis, axis in enumerate(first.bin_axes):
            axis_edges = [
                stars.get_bin_edges(axis, values_only=True)
                for stars in populations
            ]
            union = axis_edges[0]
            for other_edges in axis_edges[1:]:
                union = union_edges(union, other_edges)
            all_masses = [
                rebin_axis(masses, iaxis + 1, old, union)
                for masses, old in zip(all_masses, axis_edges)
            ]
            edges[axis] = union

        # Combine the parameters every population has, they become per
        # population parameters (one value per population) if they differ
        params = {}
        common = set.intersection(
            *[stars._parameter_names for stars in populations]
        )
        for name in sorted(common):
            given = [getattr(stars, name) for stars in populations]

            # A parameter no population has set stays unset
            if all(value is None for value in given):
                params[name] = None
                continue

            # One only some populations have set can't be combined
            if any(value is None for value in given):
                warn(
                    f"Only some of the populations have {name}, the "
                    "combined Stars won't have it."
                )
                continue

            # Get one value per population, in the units of the first
            units = getattr(given[0], "units", None)
            try:
                values = np.concatenate(
                    [
                        np.broadcast_to(
                            np.asarray(
                                value if units is None else value.to(units)
                            ),
                            (stars.npop,),
                        )
                        for stars, value in zip(populations, given)
                    ]
                )
            except (ValueError, TypeError, AttributeError):
                warn(
                    f"Can't combine {name} into one value per population, "
                    "the combined Stars won't have it."
                )
                continue

            # Keep a single value if the populations agree
            if np.all(values == values[0]):
                values = values[0]
            params[name] = values if units is None else values * units

        # Combine the morphologies
        params["morphology"] = cls._combine_morphologies(populations)

        # Name the populations
        if names is None and all(
            stars.population_names is not None for stars in populations
        ):
            names = [
                n for stars in populations for n in stars.population_names
            ]
        masses = np.concatenate(all_masses, axis=0)
        if names is not None and len(names) != masses.shape[0]:
            raise exceptions.InconsistentArguments(
                f"Got {len(names)} names for {masses.shape[0]} populations."
            )
        return first._from_bins(edges, masses, names=names, **params)

    @staticmethod
    def _combine_morphologies(populations):
        """Combine the morphologies of a set of populations.

        Args:
            populations (list of parametric.Stars):
                The Stars being combined.

        Returns:
            MorphologyBase:
                The shared morphology if every population has the same one,
                otherwise None.
        """
        morphologies = [stars.morphology for stars in populations]
        if all(morph is morphologies[0] for morph in morphologies):
            return morphologies[0]
        warn(
            "The combined Stars have different morphologies, the combined "
            "Stars will have no morphology."
        )
        return None

    def __getitem__(self, key):
        """Get a single population as its own Stars.

        Args:
            key (int/str):
                The population's index or name.

        Returns:
            Stars:
                The population, with its bins, parameters and morphology.
        """
        if isinstance(key, str):
            if self.population_names is None or (
                key not in self.population_names
            ):
                raise exceptions.InconsistentArguments(
                    f"No population named {key} (populations: "
                    f"{self.population_names})."
                )
            index = self.population_names.index(key)
        else:
            index = int(key)
            if index < 0:
                index += self.npop
            if not 0 <= index < self.npop:
                raise exceptions.InconsistentArguments(
                    f"Population {key} doesn't exist ({self.npop} "
                    "populations)."
                )

        # Pick out this population's parameters
        params = self._get_population_params()
        for name in self._parameter_names:
            value = params[name]
            if np.size(value) == self.npop and self.npop > 1:
                # Indexing an array directly keeps any units it has
                if not isinstance(value, np.ndarray):
                    value = np.asarray(value)
                params[name] = value[index]

        return self._from_bins(
            {axis: edges.copy() for axis, edges in self._bin_edges.items()},
            self.bin_masses[index : index + 1].copy(),
            names=None
            if self.population_names is None
            else [self.population_names[index]],
            **params,
        )

    def __add__(self, other_stars):
        """Add two Stars instances together.

        The result holds the populations of both (see from_populations).

        This will only work for Stars objects with the same axes.

        Args:
            other_stars (parametric.Stars):
                The other instance of Stars to add to this one.
        """
        return Stars.from_populations([self, other_stars])

    def __radd__(self, other_stars):
        """Add two Stars instances together (reflected addition).

        Overloads "reflected" addition to allow two Stars instances to be added
        together when in reverse order, i.e. second_stars + self.

        This will only work for Stars objects with the same SFZH grid axes.

        Args:
            other_stars (parametric.Stars):
                The other instance of Stars to add to this one.
        """
        return self.__add__(other_stars)

    @accepts(lum=erg / s / Hz)
    def scale_mass_by_luminosity(self, lum, scale_filter, spectra_type):
        """Scale the stellar mass to match a luminosity in a specific filter.

        NOTE: This will overwrite the initial mass attribute.

        Args:
            lum (unyt_quantity):
                The desired luminosity in scale_filter.
            scale_filter (Filter):
                The filter in which lum is measured.
            spectra_type (str):
                The spectra key with which to do this scaling, e.g. "incident"
                or "emergent".

        Raises:
            MissingSpectraType
                If the requested spectra doesn't exist an error is thrown.
        """
        # Check we have the spectra
        if spectra_type not in self.spectra:
            raise exceptions.MissingSpectraType(
                f"The requested spectra type ({spectra_type}) does not exist"
                " in this stellar population. Have you called the "
                "corresponding spectra method?"
            )

        # Calculate the current luminosity in scale_filter
        sed = self.spectra[spectra_type]
        current_lum = (
            scale_filter.apply_filter(sed.lnu, nu=sed.nu) * sed.lnu.units
        )

        # Calculate the conversion ratio between the requested and current
        # luminosity
        conversion = lum / current_lum

        # Apply conversion to the masses
        self._initial_mass *= conversion

        # Apply the conversion to all spectra
        for key in self.spectra:
            self.spectra[key]._lnu *= conversion
            if self.spectra[key]._fnu is not None:
                self.spectra[key]._fnu *= conversion

        # And to the spectra of each population
        for key in self.particle_spectra:
            self.particle_spectra[key]._lnu *= conversion
            if self.particle_spectra[key]._fnu is not None:
                self.particle_spectra[key]._fnu *= conversion

        # Apply correction to the SFZH
        self._bin_masses *= np.asarray(conversion)
        self._set_bins(self._bin_edges, self._bin_masses)

        # Ensure sf_hist and metal_dist reflect the rescaled sfzh
        self.sf_hist = np.sum(self.sfzh, axis=1)
        self.metal_dist = np.sum(self.sfzh, axis=0)

    @accepts(flux=nJy)
    def scale_mass_by_flux(self, flux, scale_filter, spectra_type):
        """Scale the stellar mass to match a flux in a specific filter.

        NOTE: This will overwrite the initial mass attribute.

        Args:
            flux (unyt_quantity):
                The desired flux in scale_filter.
            scale_filter (Filter):
                The filter in which flux is measured.
            spectra_type (str):
                The spectra key with which to do this scaling, e.g. "incident"
                or "emergent".

        Raises:
            MissingSpectraType
                If the requested spectra doesn't exist an error is thrown.
        """
        # Check we have the spectra
        if spectra_type not in self.spectra:
            raise exceptions.MissingSpectraType(
                f"The requested spectra type ({spectra_type}) does not exist"
                " in this stellar population. Have you called the "
                "corresponding spectra method?"
            )

        # Get the sed object
        sed = self.spectra[spectra_type]

        # Ensure we have a flux
        if sed.fnu is None:
            raise exceptions.MissingSpectraType(
                "{spectra_type} does not have a flux! Make sure to"
                " run Sed.get_fnu or Galaxy.get_observed_spectra"
            )

        # Calculate the current flux in scale_filter
        current_flux = (
            scale_filter.apply_filter(sed.fnu, nu=sed.obsnu) * sed.fnu.units
        )

        # Calculate the conversion ratio between the requested and current
        # flux
        conversion = flux / current_flux

        # Apply conversion to the masses
        self._initial_mass *= conversion

        # Apply the conversion to all spectra
        for key in self.spectra:
            self.spectra[key]._lnu *= conversion
            if self.spectra[key]._fnu is not None:
                self.spectra[key]._fnu *= conversion

        # And to the spectra of each population
        for key in self.particle_spectra:
            self.particle_spectra[key]._lnu *= conversion
            if self.particle_spectra[key]._fnu is not None:
                self.particle_spectra[key]._fnu *= conversion

        # Apply correction to the SFZH
        self._bin_masses *= np.asarray(conversion)
        self._set_bins(self._bin_edges, self._bin_masses)

        # Ensure sf_hist and metal_dist reflect the rescaled sfzh
        self.sf_hist = np.sum(self.sfzh, axis=1)
        self.metal_dist = np.sum(self.sfzh, axis=0)

    @timed("ParametricStars.get_sfzh")
    def get_sfzh(
        self,
        log10ages,
        metallicities,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """Get the binned SFZH at the provided axes.

        The returned Stars holds exactly the same bins as this one, only its
        axes differ, so its sfzh is these bins mapped onto the new axes with
        the bin in cell approach and nothing is lost by remapping.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFZH (bin centers, strictly
                monotonic).
            metallicities (np.ndarray of float):
                The metallicities of the desired SFZH (bin centers, strictly
                monotonic).
            grid_assignment_method (str):
                Unused, parametric populations are always mapped with the bin
                in cell approach. Kept for consistency with particle.Stars.
            nthreads (int):
                Unused, kept for consistency with particle.Stars.

        Returns:
            Stars: New Stars object on the requested axes.
        """
        return self._from_bins(
            {axis: edges.copy() for axis, edges in self._bin_edges.items()},
            self.bin_masses.copy(),
            log10ages=log10ages,
            metallicities=metallicities,
            names=self.population_names,
            **self._get_population_params(),
        )

    def plot_sfzh(
        self,
        show=True,
    ):
        """Plot the binned SZFH.

        Args:
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """
        # Create the figure and extra axes for histograms
        fig, ax, haxx, haxy = single_histxy()

        # Visulise the SFZH grid
        ax.pcolormesh(
            self.log10ages,
            self.log10metallicities,
            self.sfzh.T,
            cmap=cmr.sunburst,
        )

        # Add binned Z to right of the plot
        metal_dist = np.sum(self.sfzh, axis=0)
        haxy.fill_betweenx(
            self.log10metallicities,
            metal_dist / np.max(metal_dist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Add binned SF_HIST to top of the plot
        sf_hist = np.sum(self.sfzh, axis=1)
        haxx.fill_between(
            self.log10ages,
            sf_hist / np.max(sf_hist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Set plot limits
        haxy.set_xlim([0.0, 1.2])
        haxy.set_ylim(self.log10metallicities[0], self.log10metallicities[-1])
        haxx.set_ylim([0.0, 1.2])
        haxx.set_xlim(self.log10ages[0], self.log10ages[-1])

        # Set labels
        ax.set_xlabel(r"$\log_{10}(\mathrm{age}/\mathrm{yr})$")
        ax.set_ylabel(r"$\log_{10}(Z)$")

        # Set the limits so all axes line up
        ax.set_ylim(self.log10metallicities[0], self.log10metallicities[-1])
        ax.set_xlim(self.log10ages[0], self.log10ages[-1])

        # Shall we show it?
        if show:
            plt.show()

        return fig, ax

    def get_sfh(self):
        """Get the star formation history of the stellar population.

        Returns:
            unyt_array:
                The star formation history of the stellar population.
        """
        return self.sf_hist

    @property
    def sfh(self):
        """Alias for get_sfh."""
        return self.get_sfh()

    def plot_sfh(
        self,
        xlimits=(),
        ylimits=(),
        show=True,
    ):
        """Plot the star formation history of the stellar population.

        Args:
            xlimits (tuple):
                The limits of the x-axis.
            ylimits (tuple):
                The limits of the y-axis.
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.step(self.log10ages, self.sf_hist, where="mid", color="blue")
        ax.fill_between(
            self.log10ages,
            self.sf_hist,
            step="mid",
            color="blue",
            alpha=0.5,
        )

        ax.set_xlabel(r"$\log_{10}(\mathrm{age}/\mathrm{yr})$")
        ax.set_ylabel(r"SFH / M$_\odot$")

        if show:
            plt.show()

        return fig, ax

    def get_metal_dist(self):
        """Get the metallicity distribution of the stellar population.

        Returns:
            unyt_array:
                The metallicity distribution of the stellar population.
        """
        return self.metal_dist

    def plot_metal_dist(
        self,
        xlimits=(),
        ylimits=(),
        show=True,
    ):
        """Plot the metallicity distribution of the stellar population.

        Args:
            xlimits (tuple):
                The limits of the x-axis.
            ylimits (tuple):
                The limits of the y-axis.
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.step(self.metallicities, self.metal_dist, where="mid", color="red")
        ax.fill_between(
            self.metallicities,
            self.metal_dist,
            step="mid",
            color="red",
            alpha=0.5,
        )

        ax.set_xlabel(r"$Z$")
        ax.set_ylabel(r"Z_D / M$_\odot$")

        # Apply limits if provided
        if len(ylimits) > 0:
            ax.set_ylim(ylimits)
        if len(xlimits) > 0:
            ax.set_xlim(xlimits)

        if show:
            plt.show()

        return fig, ax

    def get_weighted_attr(self, attr):
        """Get a weighted attribute of the stellar population.

        Args:
            attr (str):
                The attribute to get.

        Returns:
            unyt_quantity:
                The weighted attribute.
        """
        # For now we need to raise an error if this is not an axis of the
        # SFZH grid
        if "age" not in attr and "metal" not in attr:
            raise exceptions.InconsistentArguments(
                "The attribute must be an axis of the SFZH grid"
            )

        # Get the attribute and the weights
        if "age" in attr:
            weight = self.sf_hist
        else:
            weight = self.metal_dist
        attr = getattr(self, attr)

        return weighted_mean(attr, weight)

    def calculate_average_sfr(self, t_range: tuple = (0, 1e8)):
        """Calculate the average SFR over a given age range.

        This is the mass formed between the two lookback ages divided by the
        time between them. The mass in each age bin is spread uniformly over
        it, so a bin straddling either limit contributes the fraction of its
        mass inside the range.

        Args:
            t_range (tuple[float, float]):
                The lookback age limits (t_start, t_end) in years over which
                to calculate the average SFR.

        Returns:
            unyt_quantity:
                The average SFR over the specified time range in Msun/yr.
        """
        # Support unyt quantities in t_range
        t_start, t_end = t_range
        if hasattr(t_start, "to"):
            t_start = t_start.to("yr").value
        if hasattr(t_end, "to"):
            t_end = t_end.to("yr").value
        if t_start >= t_end:
            raise ValueError("Start of t_range must be less than its end.")

        # Get the mass formed in the range
        mask = self.get_mask("ages", t_start * yr, ">=")
        mask = self.get_mask("ages", t_end * yr, "<", mask=mask)
        mass = (
            np.sum(
                self._bins_to_axes(
                    self._bin_edges, self._bin_masses, mask=mask
                )
            )
            * Msun
        ).in_base("galactic")

        return (mass / ((t_end - t_start) * yr)).to("Msun/yr")

    def calculate_surviving_sfzh(self, grid: Grid):
        """Calculate the surviving SFZH of the stellar population.

        This is the distribution of surviving stars in age and metallicity
        given the star formation and metal enrichment history, on the grid's
        axes.

        Args:
            grid (Grid):
                The grid to use for calculating the surviving SFZH. This is
                used to get the stellar fraction at each SFZH bin.

        Returns:
            np.ndarray: The surviving SFZH grid in Msun.
        """
        return (
            self._bins_to_grid(grid, self._bin_edges, self._bin_masses)
            * grid.stellar_fraction
        )

    def calculate_surviving_sfh(self, grid: Grid):
        """Calculate the surviving SFH of the stellar population.

        This is the distribution of surviving stars in age.

        Args:
            grid (Grid):
                The grid to use for calculating the surviving SFH. This is
                used to get the stellar fraction at each SFH bin.

        Returns:
            np.ndarray: The surviving SFH grid in Msun.
        """
        surviving_sfh = np.sum(self.calculate_surviving_sfzh(grid), axis=1)

        return surviving_sfh

    def calculate_surviving_mass(self, grid: Grid):
        """Calculate the surviving mass of the stellar population.

        This is the total mass of stars that have survived to the present day
        given the star formation and metal enrichment history.

        Args:
            grid (Grid):
                The grid to use for calculating the surviving mass.
                This is used to get the stellar fraction at each SFZH bin.

        Returns:
            unyt_quantity: The total surviving mass of the stellar
            population in Msun.
        """
        surviving_mass = np.sum(self.calculate_surviving_sfzh(grid))

        return (surviving_mass * Msun).in_base("galactic")

    def get_ionising_photon_luminosity(
        self,
        grid: Grid,
        ion: str = "HI",
    ) -> float:
        """Calculate the ionising photon luminosity from the grid.

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra.
            ion (str):
                The ion for which to calculate the ionising photon luminosity.
                Must be a recognised ion in the grid's
                log10_specific_ionising_lum dictionary.

        Returns:
             The ionising photon luminosity summed over the grid dimensions.
        """
        if ion not in grid.log10_specific_ionising_lum:
            raise exceptions.MissingGridProperty(
                f"The provided grid does not contain {ion} "
                "ionising luminosities"
            )
        weights = self._bins_to_grid(grid, self._bin_edges, self._bin_masses)
        return np.sum(10 ** grid.log10_specific_ionising_lum[ion] * weights)

    @accepts(age=yr)
    def calculate_initial_mass_at_age(self, age):
        """Calculate the initial mass of the stellar population at a given age.

        This is the total mass of stars formed that are older than the
        specified age.

        Args:
            age (float or unyt_quantity):
                The age at which to calculate the initial mass. This can be a
                float in years or a unyt quantity with time units.

        Returns:
            unyt_quantity:
                The total initial mass formed prior to this age.
        """
        _, masses = self._get_bins_at_earlier_time(age)
        return (np.sum(masses) * Msun).in_base("galactic")

    @accepts(age=yr)
    def calculate_surviving_mass_at_age(self, age, grid: Grid):
        """Calculate the surviving mass at a given age.

        This is the mass of stars older than the specified age that are
        surviving at the specified lookback time.

        Args:
            age (float or unyt_quantity):
                The age at which to calculate the surviving mass. This can be a
                float in years or a unyt quantity with time units.
            grid (Grid):
                The grid to use for calculating the surviving mass. This is
                used to get the stellar fraction at each SFZH bin.

        Returns:
            unyt_quantity:
                The surviving mass formed prior to this age.
        """
        edges, masses = self._get_bins_at_earlier_time(age)
        weights = self._bins_to_grid(grid, edges, masses)
        return (np.sum(weights * grid.stellar_fraction) * Msun).in_base(
            "galactic"
        )
