"""A module for creating and manipulating parametric stellar populations.

This is the parametric analog of particle.Stars. It not only computes and holds
the SFZH grid but everything describing a parametric Galaxy's stellar
component.

Example usage::

    stars = Stars(log10ages, metallicities, sf_hist=sfh, metal_dist=zdist)
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
from synthesizer.parametric.bin_mask import (
    BinMask,
    rebin_axis,
    to_axis_threshold,
    union_edges,
)
from synthesizer.parametric.metal_dist import Common as ZDistCommon
from synthesizer.parametric.morphology import (
    PerPopulation,
    PopulationMorphology,
)
from synthesizer.parametric.sf_hist import Common as SFHCommon
from synthesizer.synth_warnings import deprecation, warn
from synthesizer.units import Quantity, accepts
from synthesizer.utils.operation_timers import timed
from synthesizer.utils.plt import single_histxy
from synthesizer.utils.stats import weighted_mean, weighted_median

# The number of bins each interval between axis points is split into when
# binning an SFH or metallicity distribution function. Mass is uniform within
# each bin, so the error from this converges at second order: with 8 the
# spectra are within 0.2% of the converged result for a narrow (sigma = 4 Myr)
# young Gaussian SFH, and within 0.04% for typical SFHs.
SFZH_SUBDIVISIONS = 8


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
            Deprecated, use Stars.from_sfzh. The mass formed at each (age,
            metallicity) point. If provided all following arguments are
            ignored.
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
        **kwargs,
    ):
        """Initialise the parametric stellar population.

        Can either be instantiated by:
        - Passing a SFZH grid explictly (deprecated, use Stars.from_sfzh).
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
                Deprecated, use Stars.from_sfzh (or Stars.from_binned for
                bins that don't sit on the axes). The mass formed at each
                (age, metallicity) point. If provided all following arguments
                are ignored.
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
            **kwargs (dict):
                Arbitrary keyword arguments to be set as attributes on the
                Stars instance.
        """
        # Passing an SFZH is deprecated in favour of from_sfzh (which passes
        # it privately)
        if sfzh is not None:
            deprecation(
                "Passing sfzh to Stars is deprecated, use "
                "Stars.from_sfzh(log10ages, metallicities, sfzh) instead "
                "(or Stars.from_binned for bins that don't sit on the axes)."
            )
        sfzh = kwargs.pop("_point_sfzh", sfzh)

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

        # Containers for the emission of each population (stored like the
        # per particle emission of particle components, one row per
        # population)
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
        # SFZH (defined at the points of the axes, which we copy so
        # normalising doesn't modify the caller's array) or from the SFH and
        # metallicity distribution
        if sfzh is not None:
            edges, masses = self._get_point_bins(
                np.array(sfzh, dtype=np.float64)
            )
        else:
            edges, masses = self._get_bins()

        # Check the masses don't contain any NaN or Inf values, this can
        # happen if the SFH or ZH functions return NaN or Inf values for some
        # reason, and this will cause all kinds of problems downstream if we
        # don't catch it here.
        if np.any(~np.isfinite(masses)):
            raise exceptions.InconsistentArguments(
                "SFZH grid contains NaN or Inf values! "
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
            masses = masses * self.sfzh_normalisation

            # now calculate the initial mass
            self.initial_mass = np.sum(masses) * Msun

        elif self.initial_mass is not None:
            self.sfzh_normalisation = self._initial_mass / np.sum(masses)
            masses = masses * self.sfzh_normalisation

        else:
            # Otherwise calculate the total initial mass by just summing the
            # masses
            self.initial_mass = np.sum(masses) * Msun

        # Store the population's bins (which also sets sf_hist and
        # metal_dist from them)
        self._set_bins(edges, masses)

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
        nodes = np.asarray(nodes, dtype=np.float64)
        steps = np.arange(SFZH_SUBDIVISIONS) / SFZH_SUBDIVISIONS
        if nodes[0] > 0:
            log_nodes = np.log10(nodes)
            fine = 10 ** (
                log_nodes[:-1, None] + np.diff(log_nodes)[:, None] * steps
            )
        else:
            fine = nodes[:-1, None] + np.diff(nodes)[:, None] * steps
        breakpoints = np.asarray(breakpoints, dtype=np.float64)
        breakpoints = breakpoints[
            (breakpoints > 0) & (breakpoints < nodes[-1])
        ]
        return np.unique(
            np.concatenate(([0.0], fine.ravel(), [nodes[-1]], breakpoints))
        )

    @staticmethod
    def _get_points_as_bins(nodes, values):
        """Get zero width bins at a set of points.

        Each point becomes a zero width bin, with (empty) finite bins
        between them.

        Args:
            nodes (np.ndarray of float):
                The (linear) points.
            values (np.ndarray of float):
                The mass at each point.

        Returns:
            tuple:
                The bin edges and the mass in each bin.
        """
        masses = np.zeros(2 * len(nodes) - 1)
        masses[::2] = values
        return np.repeat(np.asarray(nodes, dtype=np.float64), 2), masses

    def _get_age_bins(self, offset):
        """Get the age bins and the mass formed in each.

        Args:
            offset (float):
                The offset (in years) to apply to the ages.

        Returns:
            tuple:
                The age bin edges (in years) and the mass in each bin.
        """
        ages = self.ages.to("yr").ndview

        # An SFH function is integrated exactly over fine bins of the axis
        if self.sf_hist_func is not None:
            edges = self._get_fine_edges(
                ages, self.sf_hist_func._get_breakpoints() - offset
            )
            return edges, self.sf_hist_func.get_bin_masses(edges + offset)

        # An instantaneous SFH is a single point
        if self._instant_sf is not None:
            age = self._instant_sf.to("yr").value + offset
            return np.array([age, age]), np.array([1.0])

        # An array is defined at the points of the axis
        if self.sf_hist is not None:
            return self._get_points_as_bins(ages, self.sf_hist)

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
        """
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
            return self._get_points_as_bins(metals, self.metal_dist)

        raise exceptions.InconsistentArguments(
            "A method for defining both the SFH and ZH must be provided!\n"
            "For each either an instantaneous"
            " value, a SFH/ZH object, or an array must be passed"
        )

    def _get_bins(self, age_offset=None):
        """Get the bins describing the population from the SFH and ZH.

        The SFH and metallicity distribution are binned separately and the
        population is their product.

        Args:
            age_offset (unyt_quantity):
                The offset to apply to the ages, e.g. to get the population
                at an earlier time.

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

        offset = 0.0 if age_offset is None else age_offset.to("yr").value
        age_edges, age_masses = self._get_age_bins(offset)
        metal_edges, metal_weights = self._get_metal_bins()
        edges = {"ages": age_edges, "metallicities": metal_edges}
        masses = (age_masses[:, None] * metal_weights[None, :])[None]
        return edges, masses

    def _get_bins_at_earlier_time(self, age_offset):
        """Get the bins describing this population at an earlier time.

        At a lookback time of age_offset every star was younger by
        age_offset and stars younger than it hadn't formed. So the bins are
        clipped to ages above age_offset (keeping only the mass in the part
        of each bin that had formed, see BinMask) and shifted down by it.

        Args:
            age_offset (unyt_quantity):
                The lookback time.

        Returns:
            tuple:
                The bin edges keyed by axis name and the bin masses.
        """
        offset = age_offset.to("yr").value
        mask = BinMask()
        mask.add_axis_condition("ages", ">=", offset)
        edges, masses = mask.get_masked_bins(self)
        edges["ages"] = edges["ages"] - offset
        return edges, masses

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
        return self._from_bins(*self._get_bins_at_earlier_time(age_offset))

    def _set_bins(self, edges, masses):
        """Set the bins describing this population.

        Args:
            edges (dict):
                The bin edges along each axis in bin_axes, in linear units
                (ages in yr), keyed by axis name.
            masses (np.ndarray):
                The mass in each bin (Msun) with shape (npop, nbins_0, ...,
                nbins_N) in the order of bin_axes.
        """
        self._bin_edges = edges
        self._bin_masses = masses

        # Anything derived from the bins is now stale
        self._sfzh_view = None
        self._summed_masses = None
        self._grid_weights = {}

        # Keep the SFH and metallicity distribution on the axes current
        self.sf_hist = np.sum(self.sfzh, axis=1)
        self.metal_dist = np.sum(self.sfzh, axis=0)

    def _get_point_bins(self, sfzh):
        """Get bins from a SFZH defined at the points of the axes.

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
        age_edges, _ = self._get_points_as_bins(
            self.ages.to("yr").ndview, np.zeros(self.ages.size)
        )
        metal_edges, _ = self._get_points_as_bins(
            self.metallicities, np.zeros(self.metallicities.size)
        )
        masses = np.zeros((1, age_edges.size - 1, metal_edges.size - 1))
        masses[0, ::2, ::2] = sfzh
        return {"ages": age_edges, "metallicities": metal_edges}, masses

    def _bins_to_axes(self, edges, masses):
        """Map bins onto this object's age and metallicity axes.

        Args:
            edges (dict):
                The bin edges keyed by axis name.
            masses (np.ndarray):
                The bin masses.

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
        return compute_parametric_weights(
            (
                np.ascontiguousarray(self.log10ages, dtype=np.float64),
                np.ascontiguousarray(metal_axis, dtype=np.float64),
            ),
            tuple(
                np.ascontiguousarray(edges[axis], dtype=np.float64)
                for axis in self.bin_axes
            ),
            np.ascontiguousarray(masses, dtype=np.float64),
            (True, log_metals),
            1,
            None,
        )

    def get_bins_for_grid(
        self,
        grid,
        mask=None,
        edges=None,
        masses=None,
        combine_populations=False,
    ):
        """Get bins ordered like a grid's axes and in the grid's units.

        Args:
            grid (Grid):
                The grid whose axes the bins are matched to by name (with any
                log10 prefix removed).
            mask (BinMask):
                A mask to apply to the bins, or None.
            edges (dict):
                The bin edges to use (in linear units, ages in yr), defaults
                to this population's.
            masses (np.ndarray):
                The bin masses to use, defaults to this population's.
            combine_populations (bool):
                Sum the default masses over the populations (exact for
                anything integrated over them, see summed_bin_masses).

        Returns:
            tuple:
                The bin edges along each grid axis (in the grid's units), the
                bin masses (with axes ordered like the grid's) and the log10
                flag for each grid axis.
        """
        if edges is None:
            edges = self._bin_edges
            masses = (
                self.summed_bin_masses
                if combine_populations
                else self.bin_masses
            )

        # Apply the mask, which clips the bins to the parts that pass and
        # scales their masses to match
        if mask is not None:
            edges, masses = mask.get_masked_bins(self, edges, masses)

        grid_edges = []
        log_flags = []
        bin_order = []
        for axis_name in grid._extract_axes:
            log = axis_name.startswith("log10")
            bin_axis = axis_name[5:] if log else axis_name
            if bin_axis not in edges:
                raise exceptions.MissingAttribute(
                    f"This parametric Stars has no bins along {bin_axis} "
                    f"(it is binned along {self.bin_axes})."
                )
            axis_edges = edges[bin_axis]

            # Convert to the grid's units if the edges have units
            if bin_axis == "ages":
                axis_edges = (
                    unyt_array(axis_edges, yr)
                    .to(grid._axes_units[bin_axis])
                    .ndview
                )
            grid_edges.append(
                np.ascontiguousarray(axis_edges, dtype=np.float64)
            )
            log_flags.append(log)
            bin_order.append(self.bin_axes.index(bin_axis))

        masses = np.transpose(masses, [0] + [i + 1 for i in bin_order])
        return (
            tuple(grid_edges),
            np.ascontiguousarray(masses, dtype=np.float64),
            tuple(log_flags),
        )

    def get_fraction_outside_grid(self, grid):
        """Get the fraction of the mass outside a grid's axes.

        Mass outside the grid is clamped onto the grid's edges when the
        emission is extracted (as it is for particles), so this is the
        fraction of the mass whose emission is only approximate. Mass is
        uniform within each bin, so a bin straddling a grid edge is only
        partly outside.

        Args:
            grid (Grid):
                The grid to compare to.

        Returns:
            float:
                The fraction of the mass outside the grid's axes.
        """
        edges, masses, log_flags = self.get_bins_for_grid(
            grid, combine_populations=True
        )
        total = np.sum(masses)
        if total == 0.0:
            return 0.0

        # Scale the masses by the fraction of each bin inside the grid
        # along each axis in turn
        inside = masses
        for iaxis, (axis, axis_edges, log) in enumerate(
            zip(grid._extract_axes, edges, log_flags)
        ):
            values = grid._extract_axes_values[axis]
            gmin, gmax = (
                (10 ** values[0], 10 ** values[-1])
                if log
                else (values[0], values[-1])
            )
            lo, hi = axis_edges[:-1], axis_edges[1:]
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
            shape[iaxis + 1] = -1
            inside = inside * frac.reshape(shape)

        return float(1.0 - np.sum(inside) / total)

    def _bins_to_grid(self, grid, edges, masses):
        """Map bins onto a grid's axes.

        Args:
            grid (Grid):
                The grid to map onto.
            edges (dict):
                The bin edges keyed by axis name.
            masses (np.ndarray):
                The bin masses.

        Returns:
            np.ndarray:
                The mass (Msun) at each grid point.
        """
        # Imported here to avoid importing the extension at module load
        from synthesizer.extensions.parametric_spectra import (
            compute_parametric_weights,
        )

        grid_edges, grid_masses, log_flags = self.get_bins_for_grid(
            grid, edges=edges, masses=masses
        )
        return compute_parametric_weights(
            tuple(grid._extract_axes_values[a] for a in grid._extract_axes),
            grid_edges,
            grid_masses,
            log_flags,
            1,
            None,
        )

    def _from_bins(
        self, edges, masses, log10ages=None, metallicities=None, **kwargs
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
        new = Stars.from_sfzh(
            log10ages,
            metallicities,
            np.zeros((len(log10ages), len(metallicities))),
            **kwargs,
        )
        new._set_bins(edges, masses)
        new.initial_mass = np.sum(masses) * Msun
        return new

    def get_particle_photo_lnu(self, *args, **kwargs):
        """Calculate the luminosity photometry of each population.

        This shares the Particles implementation, with one row per population
        in place of one per particle (see Particles.get_particle_photo_lnu).
        """
        from synthesizer.particle.particles import Particles

        return Particles.get_particle_photo_lnu(self, *args, **kwargs)

    def get_particle_photo_fnu(self, *args, **kwargs):
        """Calculate the flux photometry of each population.

        This shares the Particles implementation, with one row per population
        in place of one per particle (see Particles.get_particle_photo_fnu).
        """
        from synthesizer.particle.particles import Particles

        return Particles.get_particle_photo_fnu(self, *args, **kwargs)

    @property
    def bin_axes(self):
        """The names of the axes the population is binned along.

        Returns:
            tuple of str:
                The bin axis names in the order of the bin_masses axes.
        """
        return ("ages", "metallicities")

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

    @property
    def summed_bin_masses(self):
        """The mass in each bin summed over the populations (Msun).

        Every population shares the same edges, so summing over populations
        is exact for anything integrated over them (e.g. the integrated
        emission), and makes its cost independent of the number of
        populations.

        Returns:
            np.ndarray:
                The summed masses with shape (1, nbins_0, ..., nbins_N).
        """
        if self._summed_masses is None:
            self._summed_masses = np.sum(
                self._bin_masses, axis=0, keepdims=True
            )
        return self._summed_masses

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
        """
        if axis not in self._bin_edges:
            raise exceptions.MissingAttribute(
                f"This parametric Stars has no bins along {axis} (it is "
                f"binned along {self.bin_axes})."
            )
        edges = self._bin_edges[axis]
        if values_only or axis != "ages":
            return edges
        return edges * yr

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
        if self._sfzh_view is None:
            view = self._bins_to_axes(self._bin_edges, self.summed_bin_masses)
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
        can cut through a bin, so the mask records the allowed interval and
        converts it into the fraction of each bin passing (see BinMask).
        Conditions on a population level value (e.g. a fixed model parameter)
        include or exclude the whole population.

        Args:
            attr (str):
                The attribute to derive the mask from.
            thresh (float):
                The threshold value.
            op (str):
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=".
            mask (BinMask):
                Optionally, a mask to combine with the new mask.
            attr_override_obj (object):
                An alternative object to check from the attribute. This
                is specifically used when an EmissionModel may have a
                fixed parameter override, but can be used more generally.

        Returns:
            mask (BinMask):
                The combined mask.
        """
        # Start from a copy of the mask we're combining with
        if mask is None:
            new_mask = BinMask()
        elif isinstance(mask, BinMask):
            new_mask = mask.copy()
        else:
            raise exceptions.InconsistentArguments(
                "Parametric Stars masks must be BinMask objects, "
                f"got {type(mask)}."
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
            axis_units = yr if axis == "ages" else None
            new_mask.add_axis_condition(
                axis, op, to_axis_threshold(thresh, attr, axis_units)
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
        new_mask.add_population_condition(value, op, thresh)
        return new_mask

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
    def from_sfzh(cls, log10ages, metallicities, sfzh, **kwargs):
        """Create a Stars from the mass formed at each axis point.

        Each (age, metallicity) point of the axes holds the mass formed at
        exactly that age and metallicity (a zero width bin), e.g. the SFZH
        of a particle Stars binned onto a grid's axes. Use from_binned for
        bins that don't sit on the axes.

        Args:
            log10ages (np.ndarray of float):
                The log10 age axis.
            metallicities (np.ndarray of float):
                The metallicity axis.
            sfzh (np.ndarray of float):
                The mass (in Msun) formed at each (age, metallicity) point,
                with shape (len(log10ages), len(metallicities)).
            **kwargs (dict):
                Any other arguments for the Stars, e.g. initial_mass to
                normalise the SFZH, or fesc.

        Returns:
            Stars:
                The population.
        """
        return cls(log10ages, metallicities, _point_sfzh=sfzh, **kwargs)

    @classmethod
    def from_binned(
        cls,
        log10ages,
        metallicities,
        age_edges,
        metallicity_edges,
        masses,
        names=None,
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
                (n_age_bins, n_metallicity_bins) for one population or
                (npop, n_age_bins, n_metallicity_bins) for several.
            names (list of str):
                A name for each population.
            **kwargs (dict):
                Any other arguments for the Stars, e.g. per population
                parameters such as tau_v (one value per population).

        Returns:
            Stars:
                The binned population(s).
        """
        age_edges = (
            age_edges.to("yr").value
            if isinstance(age_edges, unyt_array)
            else np.asarray(age_edges, dtype=np.float64)
        )
        metallicity_edges = np.asarray(metallicity_edges, dtype=np.float64)
        masses = (
            masses.to("Msun").value
            if isinstance(masses, unyt_array)
            else np.array(masses, dtype=np.float64)
        )
        if masses.ndim == 2:
            masses = masses[None]

        # Accept decreasing (lookback) age edges
        if age_edges.size > 1 and age_edges[0] > age_edges[-1]:
            age_edges = age_edges[::-1]
            masses = masses[:, ::-1, :]

        if masses.shape[1:] != (
            age_edges.size - 1,
            metallicity_edges.size - 1,
        ):
            raise exceptions.InconsistentArguments(
                f"masses has shape {masses.shape} but the edges define "
                f"({age_edges.size - 1}, {metallicity_edges.size - 1}) bins."
            )
        if np.any(np.diff(age_edges) < 0) or np.any(
            np.diff(metallicity_edges) < 0
        ):
            raise exceptions.InconsistentArguments(
                "The bin edges must be monotonic."
            )

        stars = cls.from_sfzh(
            log10ages,
            metallicities,
            np.zeros((len(log10ages), len(metallicities))),
            **kwargs,
        )
        stars._set_bins(
            {"ages": age_edges, "metallicities": metallicity_edges},
            np.ascontiguousarray(masses),
        )
        stars.initial_mass = np.sum(masses) * Msun
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
        population parameters (one value per population) when they differ,
        and populations with different morphologies get a PerPopulation
        morphology.

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
            try:
                values = np.concatenate(
                    [
                        np.broadcast_to(
                            np.asarray(getattr(stars, name)), (stars.npop,)
                        )
                        for stars in populations
                    ]
                )
            except (ValueError, TypeError):
                warn(
                    f"Can't combine {name} into one value per population, "
                    "the combined Stars won't have it."
                )
                continue
            params[name] = values[0] if np.all(values == values[0]) else values

        # Combine the morphologies
        params["morphology"] = cls._combine_morphologies(populations)

        combined = first._from_bins(
            edges, np.concatenate(all_masses, axis=0), **params
        )

        # Name the populations
        if names is None and all(
            stars.population_names is not None for stars in populations
        ):
            names = [
                n for stars in populations for n in stars.population_names
            ]
        if names is not None and len(names) != combined.npop:
            raise exceptions.InconsistentArguments(
                f"Got {len(names)} names for {combined.npop} populations."
            )
        combined.population_names = None if names is None else list(names)
        return combined

    @staticmethod
    def _combine_morphologies(populations):
        """Combine the morphologies of a set of populations.

        Args:
            populations (list of parametric.Stars):
                The Stars being combined.

        Returns:
            MorphologyBase:
                The shared morphology if every population has the same one,
                otherwise a PerPopulation morphology (None if any population
                has no morphology).
        """
        morphologies = [stars.morphology for stars in populations]
        if all(morph is morphologies[0] for morph in morphologies):
            return morphologies[0]
        if any(morph is None for morph in morphologies):
            warn(
                "Some of the combined Stars have no morphology, the "
                "combined Stars will have no morphology."
            )
            return None

        # Each Stars covers its populations, a Stars with several
        # populations sharing one morphology repeats it for each
        entries = []
        for stars, morph in zip(populations, morphologies):
            if isinstance(morph, PopulationMorphology):
                entries.append(morph)
            else:
                entries.extend([morph] * stars.npop)
        return PerPopulation(entries)

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
        params = {}
        for name in self._parameter_names:
            value = getattr(self, name, None)
            if np.size(value) == self.npop and self.npop > 1:
                value = np.asarray(value)[index]
            params[name] = value

        # And its morphology
        morphology = self.morphology
        if isinstance(morphology, PopulationMorphology):
            morphology = morphology.get_population_morphology(index)
        params["morphology"] = morphology

        population = self._from_bins(
            {axis: edges.copy() for axis, edges in self._bin_edges.items()},
            self.bin_masses[index : index + 1].copy(),
            **params,
        )
        if self.population_names is not None:
            population.population_names = [self.population_names[index]]
        return population

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

        # Apply correction to the SFZH
        self._set_bins(
            self._bin_edges, self._bin_masses * np.asarray(conversion)
        )

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

        # Apply correction to the SFZH
        self._set_bins(
            self._bin_edges, self._bin_masses * np.asarray(conversion)
        )

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
        mask = BinMask()
        mask.add_axis_condition("ages", ">=", t_start)
        mask.add_axis_condition("ages", "<", t_end)
        mass = np.sum(self.bin_masses * mask.get_fractions(self)) * Msun

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
            self._bins_to_grid(grid, self._bin_edges, self.summed_bin_masses)
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

        return surviving_mass * Msun

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
        weights = self._bins_to_grid(
            grid, self._bin_edges, self.summed_bin_masses
        )
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
        return np.sum(masses) * Msun

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
        return np.sum(weights * grid.stellar_fraction) * Msun
