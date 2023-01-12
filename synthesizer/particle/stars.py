# -*- coding: utf-8 -*-
"""
Contains the Stars class for use with particle based systems. This houses all
the data detailing collections of stellar particles, where each property is
stored as arrays for efficiency.

There are also functions for sampling "fake" stellar distributions by
sampling a SFHZ.

Notes
-----
    Some reformating neceesary with TODOs flagging needed commenting.
"""
import warnings
import numpy as np
from .particles import Particles


class Stars(Particles):
    """
    The base Stars class. This contains all data a collection of stars could
    contain. It inherits from the base Particles class holding attributes and
    methods common to all particle types.

    The Stars class can be handed to methods elsewhere to pass ingormation need
    about the stars needed in other computations. A galaxy object should have a
    link to the stars object containing its stars for example.

    Note that due to the many possible operations this class has a large number
    of optional attributes which are set to None if not provided.

    Attributes
    ----------
    initial_masses : array-like (float)
        The intial stellar mass of each particle in Msun.
    ages : array-like (float)
        The age of each stellar particle in Myrs.
    metallicities : array-like (float)
        The metallicity of each stellar particle.
    nparticle : int
        How many stars are there?
    tauV : float
        V-band dust optical depth.
    alpha : float
        The alpha enhancement [alpha/Fe].
    imf_hmass_slope : float
        The slope of high mass end of the initial mass function (WIP)
    log10ages : float
        Convnience attribute containing log10(age).
    log10metallicities : float
        Convnience attribute containing log10(metallicity).
    resampled : bool
        Flag for whether the young particles have been resampled.
    coordinates : array-like (float)
        The coordinates of each stellar particle in simulation length
        units.
    velocities : array-like (float)
        The velocity of each stellar particle in km/s.
    current_masses : array-like (float)
        The current mass of each stellar particle in Msun.
    smoothing_lengths : array-like (float)
        The smoothing lengths (describing the sph kernel) of each stellar
        particle in simulation length units.
    s_oxygen : array-like (float)
        ???
    s_hydrogen : array-like (float)
        ???
    """

    # Define the allowed attributes
    __slots__ = ["initial_masses", "ages", "metallicities", "nparticles",
                 "tauV", "alpha", "imf_hmass_slope", "log10ages",
                 "log10metallicities", "resampled", "coordinates",
                 "velocities", "current_masses", "smoothing_lengths",
                 "s_oxygen", "s_hydrogen"]

    def __init__(self, initial_masses, ages, metallicities, **kwargs):
        """
        Intialise the Stars instance. The first 3 arguments are always required
        with all other attributes optional based on what funcitonality is
        currently being utilised.
        """

        # Set always required stellar particle properties
        self.initial_masses = initial_masses
        self.ages = ages
        self.metallicities = metallicities

        # TODO: need to add check for particle array length

        # How many particles are there?
        self.nparticles = len(self.initial_masses)

        # Intialise stellar emission quantities (updated later)
        self.tauV = None  # V-band dust optical depth
        self.alpha = None  # alpha-enhancement [alpha/Fe]

        # IMF properties
        self.imf_hmass_slope = None  # slope of the imf

        # Useful logged quantities
        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)

        # Intialise the flag for resampling
        self.resampled = False

        # Ensure all attributes are intialised to None
        # NOTE (Will): I don't like this but can't think of something cleaner
        for attr in Stars.__slots__:
            try:
                getattr(self, attr)
            except AttributeError:
                setattr(self, attr, None)

        # Handle kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def renormalise_mass(self, stellar_mass):
        """
        Renormalises the masses and stores them in the initial_mass attribute.

        TODO: But why? What's the usage? Is it not dangerous to overwrite
        the intial mass?

        Parameters
        ----------
        stellar_mass : array-like (float)
            The stellar mass array to be renormalised.
        """

        self.initial_masses *= stellar_mass/np.sum(self.initial_masses)

    def __str__(self):
        """
        Overloads the __str__ operator. A summary can be achieved by
        print(stars) where stars is an instance of Stars.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-"*10 + "\n"
        pstr += "SUMMARY OF STAR PARTICLES" + "\n"
        pstr += "log10(total mass formed/Msol): "
        pstr += f"{np.log10(np.sum(self.initial_masses)): .2f}" + "\n"
        pstr += f"median(age/Myr): {np.median(self.ages)/1E6:.1f}" + "\n"
        pstr += "-"*10

        return pstr

    def _power_law_sample(self, a, b, g, size=1):
        """
        Power-law gen for pdf(x) propto x^{g-1} for a<=x<=b

        TODO: What is going on here?

        Parameters
        ----------
        a : float
            ???
        b : float
            ???
        g : float
            ???
        size : int
            Size of desired output array?

        Returns
        -------
        float
             What is it?
        """

        # Get a random sample
        rand = np.random.random(size=size)

        a_g, b_g = a ** g, b ** g

        return (a_g + (b_g - a_g) * rand) ** (1 / g)

    def resample_young_stars(self, min_age=1e8, min_mass=700, max_mass=1e6,
                             power_law_index=-1.3, n_samples=1e3,
                             force_resample=False, verbose=False):
        """
        Resample young stellar particles into HII regions, with a
        power law distribution of masses. A young stellar particle is a
        stellar particle with an age < min_age (defined in Myr?).

        This function overwrites the propertys stored in attributes with the
        resampled properties.

        Currently resampling and imaging are not supported. An error is thrown.

        Why do you resample?

        Parameters
        ----------
        min_age : float
            ???
        min_mass : float
            ???
        max_mass : float
            ???
        power_law_index : float
            ???
        n_samples : int
            ???
        force_resample : bool
            ???
        verbose : bool
            Are we talking?
        """

        # Warn the user we are resampling a resampled population
        if self.resampled & (~force_resample):
            warnings.warn("Warning, galaxy stars already resampled. \
                    To force resample, set force_resample=True. Returning...")
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
            rvs = self._power_law_sample(min_mass, max_mass,
                                         power_law_index, int(n_samples))

            # If not enough mass has been sampled, repeat
            while np.sum(rvs) < self.masses[_idx]:
                n_samples *= 2
                rvs = self._power_law_sample(min_mass, max_mass,
                                             power_law_index, int(n_samples))

            # Sum masses up to the total mass limit
            _mask = np.cumsum(rvs) < self.masses[_idx]
            _masses = rvs[_mask]

            # Scale up to the original mass
            _masses *= (self.masses[_idx] / np.sum(_masses))

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
        for attr, new_arr in zip(["masses", "ages"],
                                 [new_masses, new_ages]):
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
            setattr(self, attr, np.append(getattr(self, attr),
                                          np.repeat(attr_array, new_lens,
                                                    axis=0)))

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


def sample_sfhz(sfzh, n, initial_mass=1):
    """
    Create "fake" stellar particles by sampling a SFHZ.

    Parameters
    ----------
    sfhz : ???
        ???
    N : int
        Number of samples?
    intial_mass : int
        The intial mass of the fake stellar particles.

    Returns
    -------
    stars : obj (Stars)
        An instance of Stars containing the fake stellar particles.
    """

    # Normalise the sfhz to produce a histogram (binned in time)
    hist = sfzh.sfzh/np.sum(sfzh.sfzh)

    # Get the midpoints of x and y to...
    x_bin_midpoints = sfzh.log10ages
    y_bin_midpoints = sfzh.log10metallicities

    # Define the cumaltive distribution function
    cdf = np.cumsum(hist.flatten())
    cdf = cdf / cdf[-1]

    # Get a random sample from the cdf
    values = np.random.rand(n)
    value_bins = np.searchsorted(cdf, values)

    # Convert indices to correct shape and extract the ages (x) and
    # metallicites (y) from the random sample
    x_idx, y_idx = np.unravel_index(value_bins,
                                    (len(x_bin_midpoints),
                                     len(y_bin_midpoints)))
    random_from_cdf = np.column_stack((x_bin_midpoints[x_idx],
                                       y_bin_midpoints[y_idx]))
    log10ages, log10metallicities = random_from_cdf.T

    # Instantiate Stars object
    stars = Stars(initial_mass * np.ones(N), 10 ** log10ages,
                  10 ** log10metallicities)

    return stars
