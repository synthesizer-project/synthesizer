"""A module for working with arrays of stellar particles.

Contains the Stars class for use with particle based systems. This houses all
the data detailing collections of stellar particles. Each property is
stored in (N_star, ) shaped arrays for efficiency.

We also provide functions for creating "fake" stellar distributions by
sampling a SFZH.

In both cases a myriad of extra optional properties can be set by providing
them as keyword arguments.

Example usages:

    stars = Stars(initial_masses, ages, metallicities,
                  redshift=redshift, current_masses=current_masses, ...)
    stars = sample_sfhz(sfzh, n, total_initial_mass, 
                        smoothing_lengths=smoothing_lengths,
                        tau_v=tau_vs, coordinates=coordinates, ...)
"""
import warnings

import numpy as np

from synthesizer.components import StarsComponent
from synthesizer.particle.particles import Particles
from synthesizer.units import Quantity
from synthesizer import exceptions


class Stars(Particles, StarsComponent):
    """
    The base Stars class. This contains all data a collection of stars could
    contain. It inherits from the base Particles class holding attributes and
    methods common to all particle types.

    The Stars class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be passed a stars object for use with any of the Galaxy helper methods.

    Note that due to the many possible operations, this class has a large number
    of optional attributes which are set to None if not provided.

    Attributes:
        initial_masses (array-like, float)
            The intial stellar mass of each particle in Msun.
        ages (array-like, float)
            The age of each stellar particle in Myrs.
        metallicities (array-like, float)
            The metallicity of each stellar particle.
        tau_v (array-like, float)
            V-band dust optical depth of each stellar particle.
        alpha_enhancement (array-like, float)
            The alpha enhancement [alpha/Fe] of each stellar particle.
        log10ages (array-like, float)
            Convnience attribute containing log10(age).
        log10metallicities (array-like, float)
            Convnience attribute containing log10(metallicity).
        resampled (bool)
            Flag for whether the young particles have been resampled.
        current_masses (array-like, float)
            The current mass of each stellar particle in Msun.
        smoothing_lengths (array-like, float)
            The smoothing lengths (describing the sph kernel) of each stellar
            particle in simulation length units.
        s_oxygen (array-like, float)
            fractional oxygen abundance.
        s_hydrogen (array-like, float)
            fractional hydrogen abundance.
        imf_hmass_slope (float)
            The slope of high mass end of the initial mass function (WIP).
        nstars (int)
            The number of stellar particles in the object.
    """

    # Define the allowed attributes
    __slots__ = [
        "metallicities",
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
        "_ages",
        "_current_masses",
    ]

    # Define class level Quantity attributes
    initial_masses = Quantity()
    ages = Quantity()
    current_masses = Quantity()
    smoothing_lengths = Quantity()

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
        imf_hmass_slope=None,
        softening_length=None,
    ):
        """
        Intialise the Stars instance. The first 3 arguments are always required.
        All other arguments are optional attributes applicable in different
        situations.

        Args:
            initial_masses (array-like, float)
                The intial stellar mass of each particle in Msun.
            ages (array-like, float)
                The age of each stellar particle in yrs.
            metallicities (array-like, float)
                The metallicity of each stellar particle.
            redshift (float)
                The redshift/s of the stellar particles.
            tau_v (array-like, float)
                V-band dust optical depth of each stellar particle.
            alpha_enhancement (array-like, float)
                The alpha enhancement [alpha/Fe] of each stellar particle.
            coordinates (array-like, float)
                The 3D positions of the particles.
            velocities (array-like, float)
                The 3D velocities of the particles.
            current_masses (array-like, float)
                The current mass of each stellar particle in Msun.
            smoothing_lengths (array-like, float)
                The smoothing lengths (describing the sph kernel) of each
                stellar particle in simulation length units.
            s_oxygen (array-like, float)
                The fractional oxygen abundance.
            s_hydrogen (array-like, float)
                The fractional hydrogen abundance.
            imf_hmass_slope (float)
                The slope of high mass end of the initial mass function (WIP)
        """

        # Instantiate parent
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=current_masses,
            redshift=redshift,
            softening_length=softening_length,
            nparticles=len(initial_masses),
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

        # Set the V band optical depths
        self.tau_v = tau_v

        # Set the alpha enhancement [alpha/Fe] (only used for >2 dimensional
        # SPS grids)
        self.alpha_enhancement = alpha_enhancement

        # Set the fractional abundance of elements
        self.s_oxygen = s_oxygen
        self.s_hydrogen = s_hydrogen

        # Compute useful logged quantities
        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)

        # Set up IMF properties (updated later)
        self.imf_hmass_slope = None  # slope of the imf

        # Intialise the flag for resampling
        self.resampled = False

        # Set a frontfacing clone of the number of particles with clearer naming
        self.nstars = self.nparticles

        # Check the arguments we've been given
        self._check_star_args()

    def _check_star_args(self):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """

        # Ensure all arrays are the expected length
        for key in self.__slots__:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent stellar array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

    def __str__(self):
        """
        Overloads the __str__ operator, enabling the printing of a summary of
        the Stars with print(stars) syntax, where stars is an instance of Stars.

        Returns:
            pstr (str)
                The summary string to be printed.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF STAR PARTICLES" + "\n"
        pstr += f"N_stars: {self.nparticles}" + "\n"
        pstr += "log10(total mass formed/Msol): "
        pstr += f"{np.log10(np.sum(self.initial_masses)): .2f}" + "\n"
        pstr += f"median(age/Myr): {np.median(self.ages)/1E6:.1f}" + "\n"
        pstr += "-" * 10

        return pstr

    def renormalise_mass(self, stellar_mass):
        """
        Renormalises and overwrites the initial masses. Useful when rescaling
        the mass of the system of stellar particles.

        Args:
            stellar_mass (array-like, float)
                The stellar mass array to be renormalised.
        """

        self.initial_masses *= stellar_mass / np.sum(self.initial_masses)

    def _power_law_sample(self, low_lim, upp_lim, g, size=1):
        """
        Sample from a power law over an interval not containing zero.

        Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b

        Args:
            low_lim (float)
                The lower bound of the interval over which to calulcate the
                power law.
            upp_lim (float)
                The upper bound of the interval over which to calulcate the
                power law.
            g (float)
                The power law index.
            size (int)
                The number of samples in the interval.

        Returns:
            array-like (float)
                The samples derived from the power law.
        """

        # Get a random sample
        rand = np.random.random(size=size)

        # Compute the value of the power law at the lower and upper bounds
        low_lim_g, upp_lim_g = low_lim**g, low_lim**g

        return (low_lim_g + (upp_lim_g - low_lim_g) * rand) ** (1 / g)

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
        """
        Resample young stellar particles into individual HII regions, with a
        power law distribution of masses. A young stellar particle is a
        stellar particle with an age < min_age (defined in Myr?).

        This function overwrites the properties stored in attributes with the
        resampled properties.

        Note: Resampling and imaging are not supported. If attempted an error
              is thrown.

        Args:
            min_age (float)
                The age below which stars will be resampled, in yrs.
            min_mass (float)
                The lower bound of the mass interval used in the power law
                sampling, in Msun.
            max_mass (float)
                The upper bound of the mass interval used in the power law
                sampling, in Msun.
            power_law_index (float)
                The index of the power law from which to sample stellar
            n_samples (int)
                The number of samples to generate for each stellar particles
                younger than min_age.
            force_resample (bool)
                A flag for whether resampling should be forced. Only applicable
                if trying to resample and already resampled Stars object.
            verbose (bool)
                Are we talking?
        """

        # Warn the user we are resampling a resampled population
        if self.resampled and not force_resample:
            warnings.warn(
                "Warning, galaxy stars already resampled. \
                    To force resample, set force_resample=True. Returning..."
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
        for attr in Stars.__slots__:
            # Skip unset attributes
            if getattr(self, attr) is None:
                continue

            # Include resampled stellar particles in this attribute
            attr_array = getattr(self, attr)[resample_idxs]
            setattr(
                self,
                attr,
                np.append(getattr(self, attr), np.repeat(attr_array, new_lens, axis=0)),
            )

        if verbose:
            print("Delete old particles")

        # Loop over attributes
        for attr in Stars.__slots__:
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


def sample_sfhz(sfzh, nstar, initial_mass=1, **kwargs):
    """
    Create "fake" stellar particles by sampling a SFZH.

    Args:
        sfhz (BinnedSFZH)
            The Star Formation Z (Metallicity) History object.
        nstar (int)
            The number of stellar particles to produce.
        intial_mass (int)
            The intial mass of the fake stellar particles.

    Returns:
        stars (Stars)
            An instance of Stars containing the fake stellar particles.
    """

    # Normalise the sfhz to produce a histogram (binned in time) between 0
    # and 1.
    hist = sfzh.sfzh / np.sum(sfzh.sfzh)

    # Compute the cumaltive distribution function
    cdf = np.cumsum(hist.flatten())
    cdf = cdf / cdf[-1]

    # Get a random sample from the cdf
    values = np.random.rand(nstar)
    value_bins = np.searchsorted(cdf, values)

    # Convert 1D random indices to 2D indices
    x_idx, y_idx = np.unravel_index(
        value_bins, (len(sfzh.log10ages), len(sfzh.log10metallicities))
    )

    # Extract the sampled ages and metallicites and create an array
    random_from_cdf = np.column_stack(
        (sfzh.log10ages[x_idx], sfzh.log10metallicities[y_idx])
    )

    # Extract the individual logged quantities
    log10ages, log10metallicities = random_from_cdf.T

    # Instantiate Stars object with extra keyword arguments
    stars = Stars(
        initial_mass * np.ones(nstar),
        10**log10ages,
        10**log10metallicities,
        **kwargs,
    )

    return stars
