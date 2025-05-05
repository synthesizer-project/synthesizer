"""A module for working with arrays of stellar particles.

Contains the Stars class for use with particle based systems. This contains all
the data detailing collections of stellar particles. Each property is
stored in (N_star, ) shaped arrays for efficiency.

We also provide functions for creating "fake" stellar distributions, by
sampling a SFZH.

In both cases a myriad of extra optional properties can be set by providing
them as keyword arguments.

Example usages:

    stars = Stars(initial_masses, ages, metallicities,
                  redshift=redshift, current_masses=current_masses, ...)
    stars = sample_sfzh(sfzh, n, total_initial_mass,
                        smoothing_lengths=smoothing_lengths,
                        tau_v=tau_vs, coordinates=coordinates, ...)
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from unyt import Mpc, Msun, Myr, dimensionless, km, s, yr

from synthesizer import exceptions
from synthesizer.components.stellar import StarsComponent
from synthesizer.parametric import SFH
from synthesizer.parametric import Stars as Para_Stars
from synthesizer.particle.particles import Particles
from synthesizer.synth_warnings import deprecated, warn
from synthesizer.units import Quantity, accepts
from synthesizer.utils.ascii_table import TableFormatter
from synthesizer.utils.util_funcs import combine_arrays


class Stars(Particles, StarsComponent):
    """The base Stars class.

    This contains all data a collection of stars could contain. It inherits
    from the base Particles class holding attributes and
    methods common to all particle types.

    The Stars class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be passed a stars object for use with any of the Galaxy helper methods.

    Note that due to the many possible operations, this class has a large
    number of optional attributes which are set to None if not provided.

    Attributes:
        initial_masses (np.ndarray of float):
            The intial stellar mass of each particle in Msun.
        ages (np.ndarray of float):
            The age of each stellar particle in yrs.
        metallicities (np.ndarray of float):
            The metallicity of each stellar particle.
        tau_v (np.ndarray of float):
            V-band dust optical depth of each stellar particle.
        alpha_enhancement (np.ndarray of float):
            The alpha enhancement [alpha/Fe] of each stellar particle.
        resampled (bool):
            Flag for whether the young particles have been resampled.
        current_masses (np.ndarray of float):
            The current mass of each stellar particle in Msun.
        smoothing_lengths (np.ndarray of float):
            The smoothing lengths (describing the sph kernel) of each stellar
            particle in simulation length units.
        s_oxygen (np.ndarray of float):
            fractional oxygen abundance.
        s_hydrogen (np.ndarray of float):
            fractional hydrogen abundance.
        imf_hmass_slope (float):
            The slope of high mass end of the initial mass function (WIP).
        nstars (int):
            The number of stellar particles in the object.
        fesc (np.ndarray of float):
            The escape fractions of each stellar particle (i.e. the fraction of
            incident photons that escape the galaxy unreprocessed by the
            interstellar medium).
    """

    # Define the allowed attributes
    attrs = [
        "nparticles",
        "tau_v",
        "alpha_enhancement",
        "imf_hmass_slope",
        "log10ages",
        "log10metallicities",
        "resampled",
        "velocities",
        "s_oxygen",
        "s_hydrogen",
        "nstars",
        "tau_v",
        "_coordinates",
        "_smoothing_lengths",
        "_softening_lengths",
        "_masses",
        "_initial_masses",
        "_current_masses",
    ]

    # Define class level Quantity attributes
    initial_masses = Quantity("mass")
    current_masses = Quantity("mass")
    smoothing_lengths = Quantity("spatial")

    @accepts(
        initial_masses=Msun.in_base("galactic"),
        ages=Myr,
        coordinates=Mpc,
        velocities=km / s,
        current_masses=Msun.in_base("galactic"),
        smoothing_lengths=Mpc,
        softening_length=Mpc,
        centre=Mpc,
    )
    def __init__(
        self,
        initial_masses,
        ages,
        metallicities,
        redshift=None,
        tau_v=None,
        alpha_enhancement=None,
        coordinates=None,
        velocities=None,
        current_masses=None,
        smoothing_lengths=None,
        s_oxygen=None,
        s_hydrogen=None,
        softening_lengths=None,
        centre=None,
        metallicity_floor=1e-5,
        fesc=None,
        fesc_ly_alpha=None,
        **kwargs,
    ):
        """Intialise the Stars instance.

        The first 3 arguments are always required. All other arguments are
        optional attributes applicable in different situations.

        Args:
            initial_masses (np.ndarray of float):
                The intial stellar mass of each particle in Msun.
            ages (np.ndarray of float):
                The age of each stellar particle in yrs.
            metallicities (np.ndarray of float):
                The metallicity of each stellar particle.
            redshift (float):
                The redshift/s of the stellar particles.
            tau_v (np.ndarray of float):
                V-band dust optical depth of each stellar particle.
            alpha_enhancement (np.ndarray of float):
                The alpha enhancement [alpha/Fe] of each stellar particle.
            coordinates (np.ndarray of float):
                The 3D positions of the particles.
            velocities (np.ndarray of float):
                The 3D velocities of the particles.
            current_masses (np.ndarray of float):
                The current mass of each stellar particle in Msun.
            smoothing_lengths (np.ndarray of float):
                The smoothing lengths (describing the sph kernel) of each
                stellar particle in simulation length units.
            s_oxygen (np.ndarray of float):
                The fractional oxygen abundance.
            s_hydrogen (np.ndarray of float):
                The fractional hydrogen abundance.
            softening_lengths (float):
                The gravitational softening lengths of each stellar
                particle in simulation units
            centre (np.ndarray of float):
                The centre of the star particle. Can be defined in
                a number of way (e.g. centre of mass)
            metallicity_floor (float):
                The minimum metallicity allowed in the simulation.
            fesc (np.ndarray of float):
                The escape fraction of each stellar particle, i.e. the
                fraction of incident photons that escape the galaxy
                unreprocessed by the interstellar medium.
            fesc_ly_alpha (np.ndarray of float):
                The escape fraction of Ly-alpha photons for each stellar
                particle, i.e. the fraction of incident Ly-alpha photons
                that escape the galaxy unreprocessed by the interstellar
                medium.
            **kwargs (dict):
                Additional keyword arguments to be set as attributes.
        """
        # Instantiate parents
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=current_masses,
            redshift=redshift,
            softening_lengths=softening_lengths,
            nparticles=initial_masses.size,
            centre=centre,
            metallicity_floor=metallicity_floor,
            tau_v=tau_v,
            name="Stars",
        )
        StarsComponent.__init__(
            self,
            ages,
            metallicities,
            _star_type="particle",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )

        # Ensure we don't have negative ages
        if len(ages) > 0:
            if ages.min() < 0.0:
                raise exceptions.InconsistentArguments(
                    "Ages cannot be negative."
                )

        # Check for nan and inf on input
        if np.sum(~np.isfinite(initial_masses)) > 0:
            raise ValueError(
                (
                    "NaN or inf on `initial_masses` input, "
                    f"indices: {np.where(~np.isfinite(initial_masses))[0]}"
                )
            )

        if np.sum(~np.isfinite(ages)) > 0:
            raise ValueError(
                (
                    "NaN or inf on `ages` input, "
                    f"indices: {np.where(~np.isfinite(ages))[0]}"
                )
            )

        if np.sum(~np.isfinite(metallicities)) > 0:
            raise ValueError(
                (
                    "NaN or inf on `metallicities` input, "
                    f"indices: {np.where(~np.isfinite(metallicities))[0]}"
                )
            )

        # Set always required stellar particle properties
        self.initial_masses = initial_masses
        self.ages = ages
        self.metallicities = metallicities

        # Set the optional keyword arguments

        # Set the SPH kernel smoothing lengths
        self.smoothing_lengths = smoothing_lengths

        # Stellar particles also have a current mass, set it
        self.current_masses = self.masses

        # Set the alpha enhancement [alpha/Fe] (only used for >2 dimensional
        # SPS grids)
        self.alpha_enhancement = alpha_enhancement

        # Set the fractional abundance of elements
        self.s_oxygen = s_oxygen
        self.s_hydrogen = s_hydrogen

        # Set up IMF properties (updated later)
        self.imf_hmass_slope = None  # slope of the imf

        # Intialise the flag for resampling
        self.resampled = False

        # Initialise the flag for parametric young stars
        self.young_stars_parametrisation = False

        # Set a frontfacing clone of the number of particles
        # with clearer naming
        self.nstars = self.nparticles

        # Check the arguments we've been given
        self._check_star_args()

        # Particle stars can calculate and attach a SFZH (and SFH) analogous
        # to a parametric galaxy
        self.sfzh = None
        self.sfh = None

    def get_sfr(self, timescale=10 * Myr):
        """Return the star formation rate of the stellar particles.

        Args:
            timescale (float):
                The timescale over which to calculate the star formation rate.

        Returns:
            sfr (float):
                The star formation rate of the stellar particles.
        """
        age_mask = self.ages < timescale
        sfr = np.sum(self.initial_masses[age_mask]) / timescale  # Msun / Myr
        return sfr.to("Msun / yr")

    @property
    def total_mass(self):
        """Return the total mass of the stellar particles.

        Returns:
            total_mass (float):
                The total mass of the stellar particles.
        """
        total_mass = 0.0

        # Check if we're using parametric young stars
        if self.young_stars_parametrisation is not False:
            # Grab the old particle masses and sum
            total_mass += np.sum(self._old_stars.masses)

        # Get current masses of particles (if parametric young
        # stars are used, then the new star particles *should*
        # have zero current mass)
        total_mass += np.sum(self.masses)

        return total_mass

    @property
    def log10ages(self):
        """Return stellar particle ages in log (base 10).

        Returns:
            log10ages (np.ndarray):
                log10 stellar ages
        """
        return np.log10(self.ages, dtype=np.float64) * dimensionless

    def _check_star_args(self):
        """Sanitizes inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """
        # Ensure all arrays are the expected length
        for key in self.attrs:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent stellar array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

    def __str__(self):
        """Return a string representation of the stars object.

        Returns:
            table (str): A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Stars")

    def _concatenate_stars_arrays(self, other):
        """Create a dictionary of attributes from two stars objects combined.

        Args:
            other (Stars):
                The other Stars object to add to this one.

        Returns:
            dict: A dictionary of all the attributes of the combined Stars
                objects.
        """
        # Check the other object is the same type
        if not isinstance(other, Stars):
            raise exceptions.InconsistentAddition(
                "Cannot add Stars object to %s object" % type(other)
            )

        # Concatenate all the named arguments which need it (Nones are handled
        # inside the combine_arrays function)
        initial_masses = combine_arrays(
            self.initial_masses, other.initial_masses
        )
        ages = combine_arrays(self.ages, other.ages)
        metallicities = combine_arrays(self.metallicities, other.metallicities)
        alpha_enhancement = combine_arrays(
            self.alpha_enhancement, other.alpha_enhancement
        )
        coordinates = combine_arrays(self.coordinates, other.coordinates)
        velocities = combine_arrays(self.velocities, other.velocities)
        current_masses = combine_arrays(
            self.current_masses, other.current_masses
        )
        smoothing_lengths = combine_arrays(
            self.smoothing_lengths, other.smoothing_lengths
        )
        s_oxygen = combine_arrays(self.s_oxygen, other.s_oxygen)
        s_hydrogen = combine_arrays(self.s_hydrogen, other.s_hydrogen)

        # Handle tau_v which can either be arrays or single values that need
        # to be converted to arrays
        if self.tau_v is None and other.tau_v is None:
            tau_v = None
        elif self.tau_v is None:
            tau_v = None
        elif other.tau_v is None:
            tau_v = None
        elif isinstance(self.tau_v, np.ndarray) and isinstance(
            other.tau_v, np.ndarray
        ):
            tau_v = np.concatenate([self.tau_v, other.tau_v])
        elif isinstance(self.tau_v, np.ndarray):
            tau_v = np.concatenate(
                [self.tau_v, np.full(other.nparticles, other.tau_v)]
            )
        elif isinstance(other.tau_v, np.ndarray):
            tau_v = np.concatenate(
                [np.full(self.nparticles, self.tau_v), other.tau_v]
            )
        else:
            self_tau_v = np.full(self.nparticles, self.tau_v)
            other_tau_v = np.full(other.nparticles, other.tau_v)
            tau_v = np.concatenate([self_tau_v, other_tau_v])

        # Handle softening lengths which can be arrays or single values that
        # need to be converted to arrays
        if self.softening_lengths is None and other.softening_lengths is None:
            softening_lengths = None
        elif self.softening_lengths is None:
            softening_lengths = None
        elif other.softening_lengths is None:
            softening_lengths = None
        elif isinstance(self.softening_lengths, np.ndarray) and isinstance(
            other.softening_lengths, np.ndarray
        ):
            softening_lengths = np.concatenate(
                [self.softening_lengths, other.softening_lengths]
            )
        elif isinstance(self.softening_lengths, np.ndarray):
            softening_lengths = np.concatenate(
                [
                    self.softening_lengths,
                    np.full(other.nparticles, other.softening_lengths),
                ]
            )
        elif isinstance(other.softening_lengths, np.ndarray):
            softening_lengths = np.concatenate(
                [
                    np.full(self.nparticles, self.softening_lengths),
                    other.softening_lengths,
                ]
            )
        else:
            self_softening_lengths = np.full(
                self.nparticles, self.softening_lengths
            )
            other_softening_lengths = np.full(
                other.nparticles, other.softening_lengths
            )
            softening_lengths = np.concatenate(
                [self_softening_lengths, other_softening_lengths]
            )

        # Handle the redshifts which must be the same
        if self.redshift != other.redshift:
            raise exceptions.InconsistentAddition(
                "Cannot add Stars objects with different redshifts"
            )
        else:
            redshift = self.redshift

        # Handle the metallicity floors where we take the minimum
        metallicity_floor = min(
            self.metallicity_floor, other.metallicity_floor
        )

        # Handle the centre of the particles, this will be taken from the
        # first object but warn if they differ (and are not None)
        if self.centre is not None and other.centre is not None:
            if not np.allclose(self.centre, other.centre):
                warn(
                    "Centres of the Stars objects differ. "
                    "Using the centre of the first object."
                )
        centre = self.centre

        # Store everything we've done in a dictionary
        kwargs = {
            "initial_masses": initial_masses,
            "ages": ages,
            "metallicities": metallicities,
            "redshift": redshift,
            "tau_v": tau_v,
            "alpha_enhancement": alpha_enhancement,
            "coordinates": coordinates,
            "velocities": velocities,
            "current_masses": current_masses,
            "smoothing_lengths": smoothing_lengths,
            "s_oxygen": s_oxygen,
            "s_hydrogen": s_hydrogen,
            "softening_lengths": softening_lengths,
            "centre": centre,
            "metallicity_floor": metallicity_floor,
        }

        # Handle the extra keyword arguments
        for key in self.__dict__.keys():
            # Skip methods
            if callable(getattr(self, key)):
                continue

            # Skip any attributes which aren't on both objects
            if key not in other.__dict__:
                continue

            if key not in kwargs:
                # Combine the attributes, concatenate if arrays, copied if
                # scalars and the same for both objects or added if different
                # on each. If the attribute is None for one object and not the
                # other we'll assume None overall because the combination is
                # undefined.
                if getattr(self, key) is None or getattr(other, key) is None:
                    kwargs[key] = None
                elif isinstance(getattr(self, key), np.ndarray) and isinstance(
                    getattr(other, key), np.ndarray
                ):
                    kwargs[key] = np.concatenate(
                        [getattr(self, key), getattr(other, key)]
                    )
                elif (
                    isinstance(getattr(self, key), (int, float))
                    and isinstance(getattr(other, key), (int, float))
                    and getattr(self, key) == getattr(other, key)
                ):
                    kwargs[key] = getattr(self, key)
                elif isinstance(
                    getattr(self, key), (int, float)
                ) and isinstance(getattr(other, key), (int, float)):
                    kwargs[key] = getattr(self, key) + getattr(other, key)

        return kwargs

    def __add__(self, other):
        """Add two Stars objects together.

        This will correctly combine named arguments and create a new Stars
        object with the combined particles. Any extra keyword arguments will
        be either concatenated for arrays, summed for differing scalars or
        copied if the same for both objects.

        If either object carries None for an attribute the new instance will
        also have None for that attribute.

        Args:
            other (Stars):
                The other Stars object to add to this one.

        Returns:
            Stars
                A new Stars object containing the combined particles.
        """
        kwargs = self._concatenate_stars_arrays(other)

        return Stars(**kwargs)

    def _remove_stars(self, pmask):
        """Update stars attribute arrays based on a mask, `pmask`.

        Args:
            pmask (array-like, bool):
                A boolean mask to remove stars from the object.
        """
        # Remove the masked stars from this object
        self.initial_masses = self.initial_masses[~pmask]
        self.ages = self.ages[~pmask]
        self.metallicities = self.metallicities[~pmask]
        if self.masses is not None:
            self.masses = self.masses[~pmask]
        if self.coordinates is not None:
            self.coordinates = self.coordinates[~pmask]
        if self.tau_v is not None:
            self.tau_v = self.tau_v[~pmask]
        if self.alpha_enhancement is not None:
            self.alpha_enhancement = self.alpha_enhancement[~pmask]
        if self.velocities is not None:
            self.velocities = self.velocities[~pmask]
        if self.current_masses is not None:
            self.current_masses = self.current_masses[~pmask]
        if self.s_oxygen is not None:
            self.s_oxygen = self.s_oxygen[~pmask]
        if self.s_hydrogen is not None:
            self.s_hydrogen = self.s_hydrogen[~pmask]

        if self.redshift is not None:
            if isinstance(self.redshift, np.ndarray):
                self.redshift = self.redshift[~pmask]
        if self.smoothing_lengths is not None:
            if isinstance(self.smoothing_lengths, np.ndarray):
                self.smoothing_lengths = self.smoothing_lengths[~pmask]

        self.nparticles = len(self.initial_masses)
        self.nstars = self.nparticles

        # Check the arguments we've been given
        self._check_star_args()

    def parametric_young_stars(
        self,
        age,
        parametric_sfh,
        grid,
        **kwargs,
    ):
        """Replace young stars with individual parametric SFH's.

        Can be either a constant or truncated exponential, selected with the
        `parametric_sfh` argument. The metallicity is set to the metallicity
        of the parent star particle.

        Args:
            age (float):
                Age in Myr below which we replace Star particles.
                Used to set the duration of parametric SFH
            parametric_sfh (string):
                Form of the parametric SFH to use for young stars.
                Currently two are supported, `Constant` and
                `TruncatedExponential`, selected using the keyword
                arguments `constant` and `exponential`.
            grid (Grid):
                The spectral grid object.
            **kwargs (dict):
                Additional keyword arguments to be passed to the SFH
                object.
        """
        if self.young_stars_parametrisation is not False:
            warn(
                (
                    "This Stars object has already replaced young stars."
                    "\nParametrisation:"
                    f" {self.young_stars_parametrisation['parametrisation']}, "
                    f"\nAge: {self.young_stars_parametrisation['age']}. \n"
                    "Undoing before applying new parametric form..."
                )
            )

            pmask = self._get_masks(
                self.young_stars_parametrisation["age"], None
            )

            # Remove the 'parametric' stars from the object
            self._remove_stars(pmask)

            # Add old stars back on to object
            concat_arrays = self._concatenate_stars_arrays(self._old_stars)

            for key, value in concat_arrays.items():
                setattr(self, key, value)

            self.nparticles = len(self.initial_masses)
            self.nstars = self.nparticles

            # Check the arguments we've been given
            self._check_star_args()

        # Mask for particles below age
        pmask = self._get_masks(age, None)

        if np.sum(pmask) == 0:
            return None

        # initialise SFH object
        if parametric_sfh == "constant":
            sfh = SFH.Constant(max_age=age, **kwargs)
        elif parametric_sfh == "exponential":
            sfh = SFH.TruncatedExponential(
                tau=age / 2,
                max_age=age,
                min_age=0.0 * Myr,
                **kwargs,
            )
        else:
            raise ValueError(
                (
                    "Value of `parametric_sfh` provided, "
                    f"`{parametric_sfh}`, is not supported."
                    "Please use 'constant' or 'exponential'."
                )
            )

        stars = [None] * np.sum(pmask)

        # Loop through particles to be replaced
        for i, _pmask in enumerate(np.where(pmask)[0]):
            # Create a parametric Stars object
            stars[i] = Para_Stars(
                grid.log10age,
                grid.metallicity,
                sf_hist=sfh,
                metal_dist=self.metallicities[_pmask],
                initial_mass=self.initial_masses[_pmask],
            )

        if len(stars) > 1:
            # Combine the individual parametric forms for each particle
            stars = sum(stars[1:], stars[0])
        else:
            stars = stars[0]

        self._parametric_young_stars = stars

        # Create index pairs for the SFZH
        index_pairs = np.asarray(
            [
                [[j, i] for i in np.arange(len(grid.metallicity))]
                for j in np.arange(len(grid.log10age))
            ]
        )

        # Find the grid indexes on the parametric grid
        grid_indexes = index_pairs[stars.sfzh > 0]

        # Create new particle stars object from non-empty SFZH entries
        new_stars = self.__class__(
            stars.sfzh[stars.sfzh > 0] * Msun,
            10 ** grid.log10ages[grid_indexes[:, 0]] * yr,
            grid.metallicity[grid_indexes[:, 1]],
            redshift=self.redshift,
            masses=np.zeros(np.sum(stars.sfzh > 0)) * Msun,
        )

        # Save the old stars privately
        self._old_stars = self.__class__(
            self.initial_masses[pmask],
            self.ages[pmask],
            self.metallicities[pmask],
            redshift=self.redshift,
            current_masses=(
                self.masses[pmask] if self.masses is not None else None
            ),
        )

        self._remove_stars(pmask)

        # Add to current stars object
        concat_arrays = self._concatenate_stars_arrays(new_stars)

        for key, value in concat_arrays.items():
            setattr(self, key, value)

        self.nparticles = len(self.initial_masses)
        self.nstars = self.nparticles

        # Check the arguments we've been given
        self._check_star_args()

        self.young_stars_parametrisation = {
            "parametrisation": parametric_sfh,
            "age": age,
        }

    @accepts(young=yr, old=yr)
    def _get_masks(self, young=None, old=None):
        """Get masks for which components we are handling.

        If a sub-component has not been requested it's necessarily
        all particles.

        TODO: this is no longer really needed, we should swap it for the
        get_masks method in the base Particles class.

        Args:
            young (float):
                Age in Myr at which to filter for young star particles.
            old (float):
                Age in Myr at which to filter for old star particles.

        Raises:
            InconsistentParameter
                Can't select for both young and old components
                simultaneously

        """
        # We can't have both young and old set
        if young and old:
            raise exceptions.InconsistentParameter(
                "Galaxy sub-component can not be simultaneously young and old"
            )

        # Get the appropriate mask
        if young:
            # Mask out old stars
            s = self.log10ages <= np.log10(young)
        elif old:
            # Mask out young stars
            s = self.log10ages > np.log10(old)
        else:
            # Nothing to mask out
            s = np.ones(self.nparticles, dtype=bool)

        return s

    @accepts(stellar_mass=Msun.in_base("galactic"))
    def renormalise_mass(self, stellar_mass):
        """Renormalise the initial masses of the stars.

        Renormalises and overwrites the initial masses inplace based on the
        input total stellar mass.

        Args:
            stellar_mass (np.ndarray of float):
                The stellar mass array to be renormalised.
        """
        self.initial_masses *= stellar_mass / np.sum(self.initial_masses)

    def _power_law_sample(self, low_lim, upp_lim, g, size=1):
        """Sample from a power law over an interval not containing zero.

        Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b

        Args:
            low_lim (float):
                The lower bound of the interval over which to calulcate the
                power law.
            upp_lim (float):
                The upper bound of the interval over which to calulcate the
                power law.
            g (float):
                The power law index.
            size (int):
                The number of samples in the interval.

        Returns:
            array-like (float):
                The samples derived from the power law.
        """
        # Get a random sample
        rand = np.random.random(size=size)

        # Compute the value of the power law at the lower and upper bounds
        low_lim_g, upp_lim_g = low_lim**g, low_lim**g

        return (low_lim_g + (upp_lim_g - low_lim_g) * rand) ** (1 / g)

    @accepts(
        min_age=yr,
        min_mass=Msun.in_base("galactic"),
        max_mass=Msun.in_base("galactic"),
    )
    def resample_young_stars(
        self,
        min_age=1e8,
        min_mass=700,
        max_mass=1e6,
        power_law_index=-1.3,
        n_samples=1e3,
        force_resample=False,
        verbose=False,
    ):
        """Resample stars below a given age.

        Resample young stellar particles into individual HII regions, with a
        power law distribution of masses. A young stellar particle is a
        stellar particle with an age < min_age (defined in Myr?).

        This function overwrites the properties stored in attributes with the
        resampled properties.

        Note: Resampling and imaging are not supported. If attempted an error
              is thrown.

        Args:
            min_age (float):
                The age below which stars will be resampled, in yrs.
            min_mass (float):
                The lower bound of the mass interval used in the power law
                sampling, in Msun.
            max_mass (float):
                The upper bound of the mass interval used in the power law
                sampling, in Msun.
            power_law_index (float):
                The index of the power law from which to sample stellar
            n_samples (int):
                The number of samples to generate for each stellar particles
                younger than min_age.
            force_resample (bool):
                A flag for whether resampling should be forced. Only applicable
                if trying to resample and already resampled Stars object.
            verbose (bool):
                Are we talking?
        """
        # Warn the user we are resampling a resampled population
        if self.resampled and not force_resample:
            warn(
                "Galaxy stars already resampled. "
                "To force resample, set force_resample=True. Returning..."
            )
            return None

        if verbose:
            print("Masking resample stars")

        # Get the indices of young stars for resampling
        resample_idxs = np.where(self.ages < min_age)[0]

        # No work to do here, stars are too old
        if len(resample_idxs) == 0:
            return None

        # Set up container for the resample stellar particles
        new_ages = {}
        new_masses = {}

        if verbose:
            print("Loop through resample stars")

        # Loop over the young stars we need to resample
        for _idx in resample_idxs:
            # Sample the power law
            rvs = self._power_law_sample(
                min_mass, max_mass, power_law_index, int(n_samples)
            )

            # If not enough mass has been sampled, repeat
            while np.sum(rvs) < self.masses[_idx]:
                n_samples *= 2
                rvs = self._power_law_sample(
                    min_mass, max_mass, power_law_index, int(n_samples)
                )

            # Sum masses up to the total mass limit
            _mask = np.cumsum(rvs) < self.masses[_idx]
            _masses = rvs[_mask]

            # Scale up to the original mass
            _masses *= self.masses[_idx] / np.sum(_masses)

            # Sample uniform distribution of ages
            _ages = np.random.rand(len(_masses)) * min_age

            # Store our resampled properties
            new_ages[_idx] = _ages
            new_masses[_idx] = _masses

        # Unpack the resample properties and make note of how many particles
        # were produced
        new_lens = [len(new_ages[_idx]) for _idx in resample_idxs]
        new_ages = np.hstack([new_ages[_idx] for _idx in resample_idxs])
        new_masses = np.hstack([new_masses[_idx] for _idx in resample_idxs])

        if verbose:
            print("Concatenate new arrays to existing")

        # Include the resampled particles in the attributes
        for attr, new_arr in zip(["masses", "ages"], [new_masses, new_ages]):
            attr_array = getattr(self, attr)
            setattr(self, attr, np.append(attr_array, new_arr))

        if verbose:
            print("Duplicate existing attributes")

        # Handle the other propertys that need duplicating
        for attr in Stars.attrs:
            # Skip unset attributes
            if getattr(self, attr) is None:
                continue

            # Include resampled stellar particles in this attribute
            attr_array = getattr(self, attr)[resample_idxs]
            setattr(
                self,
                attr,
                np.append(
                    getattr(self, attr),
                    np.repeat(attr_array, new_lens, axis=0),
                ),
            )

        if verbose:
            print("Delete old particles")

        # Loop over attributes
        for attr in Stars.attrs:
            # Skip unset attributes
            if getattr(self, attr) is None:
                continue

            # Delete the original stellar particles that have been resampled
            attr_array = getattr(self, attr)
            attr_array = np.delete(attr_array, resample_idxs)
            setattr(self, attr, attr_array)

        # Recalculate log attributes
        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)

        # Set resampled flag
        self.resampled = True

    def _prepare_sfzh_args(
        self,
        log10ages,
        log10metallicities,
        grid_assignment_method,
        nthreads,
    ):
        """Prepare the arguments for SFZH computation with the C functions.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFZH.
            log10metallicities (np.ndarray of float):
                The logged metallicities of the desired SFZH.
            grid_assignment_method (string):
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """
        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(log10ages, dtype=np.float64),
            np.ascontiguousarray(log10metallicities, dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.log10ages, dtype=np.float64),
            np.ascontiguousarray(self.log10metallicities, dtype=np.float64),
        ]
        part_mass = np.ascontiguousarray(
            self._initial_masses, dtype=np.float64
        )

        # Make sure we set the number of particles to the size of the mask
        npart = np.int32(len(part_mass))

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props), dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

        # If nthreads = -1 we will use all available
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_props,
            part_props,
            part_mass,
            grid_dims,
            len(grid_props),
            npart,
            grid_assignment_method,
            nthreads,
            None,
        )

    def get_sfzh(
        self,
        log10ages,
        log10metallicities,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """Generate the binned SFZH history of these stars.

        The binned SFZH is calculated by binning the particles onto the
        desired grid defined by the input log10ages and metallicities.

        The binned SFZH produced by this method is equivalent to the weights
        used to extract spectra from the grid.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFZH.
            log10metallicities (np.ndarray of float):
                The logged metallicities of the desired SFZH.
            grid_assignment_method (string):
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or their uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used.

        Returns:
            numpy.ndarray:
                Numpy array of containing the SFZH.
        """
        # Import parametric stars here to avoid circular imports
        from synthesizer.extensions.sfzh import compute_sfzh
        from synthesizer.parametric import Stars as ParametricStars

        # Prepare the arguments for the C function.
        args = self._prepare_sfzh_args(
            log10ages,
            log10metallicities,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )

        # Get the SFZH and create the ParametricStars object
        self.sfzh = ParametricStars(
            log10ages,
            10**log10metallicities,
            sfzh=compute_sfzh(*args),
        )

        return self.sfzh

    def plot_sfzh(
        self,
        show=True,
    ):
        """Plot the binned SZFH.

        Args:
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig: The Figure object contain the plot axes.
            ax: The Axes object containing the plotted data.
        """
        # Ensure we have the SFZH
        if self.sfzh is None:
            raise exceptions.MissingAttribute(
                "The SFZH has not been calculated. Run get_sfzh() first."
            )
        return self.sfzh.plot_sfzh(show=show)

    def _prepare_sfh_args(
        self,
        log10ages,
        grid_assignment_method,
        nthreads,
    ):
        """Prepare the arguments for SFH computation with the C functions.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFH.
            grid_assignment_method (string):
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """
        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(log10ages, dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.log10ages, dtype=np.float64),
        ]
        part_mass = np.ascontiguousarray(
            self._initial_masses, dtype=np.float64
        )

        # Make sure we set the number of particles to the size of the mask
        npart = np.int32(len(part_mass))

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props), dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

        # If nthreads = -1 we will use all available
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_props,
            part_props,
            part_mass,
            grid_dims,
            len(grid_props),
            npart,
            grid_assignment_method,
            nthreads,
            None,
        )

    def get_sfh(self, log10ages, grid_assignment_method="cic", nthreads=0):
        """Generate the SFH of these stars in terms of mass.

        The SFH is calculated by summing the mass of the particles in each age
        bin defined by the input log10ages.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFH.
            grid_assignment_method (string):
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or their uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used. Defaults to 0.

        Returns:
            numpy.ndarray:
                Numpy array of containing the SFH.
        """
        # Import parametric stars here to avoid circular imports
        from synthesizer.extensions.sfzh import compute_sfzh

        # Prepare the arguments for the C function.
        args = self._prepare_sfh_args(
            log10ages,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )

        self.sfh = compute_sfzh(*args)

        return self.sfh

    def plot_sfh(
        self,
        log10ages,
        nthreads=0,
        xlimits=(),
        ylimits=(),
        show=True,
    ):
        """Plot the SFH in terms of mass.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFH.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used. Defaults to 0.
            xlimits (tuple):
                The limits of the x-axis. If not set, the limits are set to the
                minimum and maximum of the log10ages.
            ylimits (tuple):
                The limits of the y-axis. If not set, the limits are set to the
                minimum and maximum of the SFH.
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """
        # Compute the SFH
        sfh = self.get_sfh(log10ages, nthreads=nthreads)

        # Plot the SFH as a step function
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.step(log10ages, sfh, where="mid", color="blue")

        ax.fill_between(log10ages, sfh, step="mid", alpha=0.5, color="blue")
        ax.set_xlabel(r"$\log_{10}(\mathrm{age}/\mathrm{yr})$")
        ax.set_ylabel(r"SFH / M$_\odot$")

        # Apply any limits we have
        if len(xlimits) > 0:
            ax.set_xlim(xlimits)
        if len(ylimits) > 0:
            ax.set_ylim(ylimits)

        if show:
            plt.show()

        return fig, ax

    def _prepare_metal_dist_args(
        self,
        metallicities,
        grid_assignment_method,
        nthreads,
    ):
        """Prepare arguments for metalicity computation with the C functions.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFZH.
            metallicities (np.ndarray of float):
                The metallicities of the desired SFZH.
            grid_assignment_method (string):
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used. Defaults to 0.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """
        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(metallicities, dtype=np.float64),
        ]
        part_props = [
            np.ascontiguousarray(self.metallicities, dtype=np.float64),
        ]
        part_mass = np.ascontiguousarray(
            self._initial_masses, dtype=np.float64
        )

        # Make sure we set the number of particles to the size of the mask
        npart = np.int32(len(part_mass))

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props), dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

        # If nthreads = -1 we will use all available
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_props,
            part_props,
            part_mass,
            grid_dims,
            len(grid_props),
            npart,
            grid_assignment_method,
            nthreads,
            None,
        )

    def get_metal_dist(
        self,
        metallicities,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """Generate the metallicity distribution in terms of mass.

        Args:
            metallicities (np.ndarray of float):
                The metallicity bins of the desired metallicity distribution.
            grid_assignment_method (string):
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or their uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used. Defaults to 0.

        Returns:
            numpy.ndarray:
                Numpy array of containing the SFH.
        """
        # Import parametric stars here to avoid circular imports
        from synthesizer.extensions.sfzh import compute_sfzh

        # Prepare the arguments for the C function.
        args = self._prepare_metal_dist_args(
            metallicities,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )

        return compute_sfzh(*args)

    def plot_metal_dist(
        self,
        metallicities,
        nthreads=0,
        xlimits=(),
        ylimits=(),
        show=True,
    ):
        """Plot the metallicity distribution in terms of mass.

        Args:
            metallicities (np.ndarray of float):
                The metallicity bins of the desired metallicity distribution.
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used. Defaults to 0.
            xlimits (tuple):
                The limits of the x-axis. If not set, the limits are set to the
                minimum and maximum of the log10ages.
            ylimits (tuple):
                The limits of the y-axis. If not set, the limits are set to the
                minimum and maximum of the SFH.
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """
        # Compute the SFH
        metal_dist = self.get_metal_dist(metallicities, nthreads=nthreads)

        # Plot the SFH as a step function
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.step(metallicities, metal_dist, where="mid", color="red")

        ax.fill_between(
            metallicities, metal_dist, step="mid", alpha=0.5, color="red"
        )
        ax.set_xlabel(r"$Z$")
        ax.set_ylabel(r"Z_D / M$_\odot$")

        # Apply any limits we have
        if len(xlimits) > 0:
            ax.set_xlim(xlimits)
        if len(ylimits) > 0:
            ax.set_ylim(ylimits)

        if show:
            plt.show()

        return fig, ax

    @deprecated(
        message="is now just a wrapper "
        "around get_spectra. It will be removed by v1.0.0."
    )
    def get_particle_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """Generate stellar spectra as described by the emission model.

        Note: Now deprecated in favour of get_spectra and emission models
        knowing which spectra should be per particle.

        Args:
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool):
                Are we talking?
            kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict: A dictionary of spectra which can be attached to the
            appropriate spectra attribute of the component
            (spectra/particle_spectra).
        """
        previous_per_part = emission_model.per_particle
        emission_model.set_per_particle(True)
        spectra = self.get_spectra(
            emission_model,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )
        emission_model.set_per_particle(previous_per_part)
        return spectra

    @deprecated(
        message="is now just a wrapper "
        "around get_lines. It will be removed by v1.0.0."
    )
    def get_particle_lines(
        self,
        line_ids,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """Generate stellar lines as described by the emission model.

        Note: Now deprecated in favour of get_lines and emission models
        knowing which lines should be per particle.

        Args:
            line_ids (list):
                A list of line_ids. Doublets can be specified as a nested
                list or using a comma (e.g., 'OIII4363,OIII4959').
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool):
                Are we talking?
            kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            LineCollection:
                A LineCollection object containing the lines defined by the
                root model.
        """
        previous_per_part = emission_model.per_particle
        emission_model.set_per_particle(True)
        lines = self.get_lines(
            line_ids,
            emission_model,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )
        emission_model.set_per_particle(previous_per_part)
        return lines


@accepts(initial_mass=Msun.in_base("galactic"))
def sample_sfzh(
    sfzh,
    log10ages,
    log10metallicities,
    nstar,
    initial_mass=1 * Msun,
    **kwargs,
):
    """Create "fake" stellar particles by sampling a SFZH.

    Args:
        sfzh (np.ndarray of float):
            The Star Formation Metallicity History grid
            (from parametric.Stars).
        log10ages (np.ndarray of float):
            The log of the SFZH age axis.
        log10metallicities (np.ndarray of float):
            The log of the SFZH metallicities axis.
        nstar (int):
            The number of stellar particles to produce.
        initial_mass (int):
            The intial mass of the fake stellar particles.
        **kwargs:
            Any additional keyword arguments to pass to the Stars
            constructor.

    Returns:
        stars (Stars)
            An instance of Stars containing the fake stellar particles.
    """
    # Normalise the sfhz to produce a histogram (binned in time) between 0
    # and 1.
    hist = sfzh / np.sum(sfzh)

    # Compute the cumaltive distribution function
    cdf = np.cumsum(hist.flatten())
    cdf = cdf / cdf[-1]

    # Get a random sample from the cdf
    values = np.random.rand(nstar)
    value_bins = np.searchsorted(cdf, values)

    # Convert 1D random indices to 2D indices
    x_idx, y_idx = np.unravel_index(
        value_bins, (len(log10ages), len(log10metallicities))
    )

    # Extract the sampled ages and metallicites and create an array
    random_from_cdf = np.column_stack(
        (log10ages[x_idx], log10metallicities[y_idx])
    )

    # Extract the individual logged quantities
    log10ages, log10metallicities = random_from_cdf.T

    # Instantiate Stars object with extra keyword arguments
    stars = Stars(
        initial_mass * np.ones(nstar),
        10**log10ages * yr,
        10**log10metallicities,
        **kwargs,
    )

    return stars
