"""A module for working with arrays of gas particles.

Contains the Gas class for use with particle based systems. This houses all
the data detailing collections of gas particles. Each property is
stored in (N_gas, ) shaped arrays for efficiency.

Extra optional properties can be set by providing
them as keyword arguments.

Example usages:

    gas = Gas(masses, metallicities,
              redshift=redshift, coordinates=coordinates, ...)
"""

import numpy as np
from unyt import Mpc, Msun, km, s

from synthesizer import exceptions
from synthesizer.particle.particles import Particles
from synthesizer.synth_warnings import warn
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter
from synthesizer.utils.util_funcs import combine_arrays


class Gas(Particles):
    """The particle Gas class.

    This contains all data a collection of gas particles
    could contain. It inherits from the base Particles class holding
    attributes and methods common to all particle types.

    The Gas class can be handed to methods elsewhere to pass information
    about the gas particles needed in other computations. A galaxy object
    should have a link to the Gas object containing its gas particles,
    for example.

    Note that due to the wide range of possible properties and operations,
    this class has a large number of optional attributes which are set to
    None if not provided.

    Attributes:
        metallicities (np.ndarray of float):
            The gas phase metallicity of each particle (integrated)
        star_forming (np.ndarray of bool):
            Flag for whether each gas particle is star forming or not.
        log10metallicities (np.ndarray of float):
            Convenience attribute containing log10(metallicity).
        smoothing_lengths (np.ndarray of float):
            The smoothing lengths (describing the sph kernel) of each gas
            particle in simulation length units.
        dust_to_metal_ratio (np.ndarray of float):
            The ratio between dust and total metal content in a gas
            particle. This can either be a single float or an array of
            values for each gas particle.
        dust_masses (unyt_quantity):
            The mass of dust in each particle in Msun.
        star_forming (np.ndarray of bool):
            A boolean array indicating whether each gas particle is
            star forming or not.
    """

    # Define the allowed attributes
    attrs = [
        "metallicities",
        "star_forming",
        "log10metallicities",
        "dust_to_metal_ratio",
        "_dust_masses",
        "_coordinates",
        "_velocities",
        "_smoothing_lengths",
        "_softening_lengths",
        "_masses",
    ]

    # Define class level Quantity attributes
    smoothing_lengths = Quantity("spatial")
    dust_masses = Quantity("mass")

    @accepts(
        masses=Msun.in_base("galactic"),
        coordinates=Mpc,
        velocities=km / s,
        smoothing_lengths=Mpc,
        softening_length=Mpc,
        dust_masses=Msun.in_base("galactic"),
        centre=Mpc,
    )
    def __init__(
        self,
        masses,
        metallicities,
        star_forming=None,
        redshift=None,
        coordinates=None,
        velocities=None,
        smoothing_lengths=None,
        softening_lengths=None,
        dust_to_metal_ratio=None,
        dust_masses=None,
        verbose=False,
        centre=None,
        metallicity_floor=1e-5,
        tau_v=None,
        **kwargs,
    ):
        """Initialise the gas object.

        Args:
            masses (np.ndarray of float):
                The mass of each particle in Msun.
            metallicities (np.ndarray of float):
                The metallicity of each gas particle.
            star_forming (array-like, bool):
                Flag for whether each gas particle is star forming or not.
            redshift (float):
                The redshift/s of the stellar particles.
            coordinates (np.ndarray of float):
                The 3D positions of the particles.
            velocities (np.ndarray of float):
                The 3D velocities of the particles.
            smoothing_lengths (np.ndarray of float):
                The smoothing lengths (describing the sph kernel) of each
                gas particle in simulation length units.
            softening_lengths (np.ndarray of float):
                The softening lengths of each gas particle in simulation
                length units.
            dust_to_metal_ratio (array_like, float or float):
                The ratio between dust and total metal content in a gas
                particle. This can either be a single float or an array of
                values for each gas particle.
            dust_masses (array_like, float):
                Mass of dust in each particle in Msun.
            verbose (bool):
                Whether to print extra information to the console.
            centre (np.ndarray of float):
                The centre of the galaxy in simulation length units.
            metallicity_floor (float):
                The metallicity floor when using log properties (only matters
                for baryons). This is used to avoid log(0) errors.
            tau_v (float):
                The dust optical depth in the V band.
            **kwargs (dict):
                Extra optional properties to set on the gas object.
        """
        # Instantiate parent
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=masses,
            redshift=redshift,
            softening_lengths=softening_lengths,
            nparticles=masses.size,
            centre=centre,
            metallicity_floor=metallicity_floor,
            tau_v=tau_v,
            name="Gas",
        )

        # Set the metallicites and log10 equivalent
        self.metallicities = metallicities

        # Set the star forming boolean mask array
        self.star_forming = star_forming

        # Set the smoothing lengths for these gas particles
        self.smoothing_lengths = smoothing_lengths

        # None metallicity warning already captured when loading gas
        if (
            (self.metallicities is not None)
            & (dust_to_metal_ratio is None)
            & (dust_masses is None)
        ):
            warn(
                "Neither dust mass nor dust to metal ratio "
                "provided. Assuming dust to metal ratio = 0.3"
            )
            self.dust_to_metal_ratio = 0.3
            self.calculate_dust_mass()
        elif dust_to_metal_ratio is not None:
            # The dust to metal ratio for gas particles. Either a scalar
            # or an array of values for each gas particle
            self.dust_to_metal_ratio = dust_to_metal_ratio
            self.calculate_dust_mass()
        else:  # if dust_masses is not None:
            self.dust_masses = dust_masses

            # Calculate the dust to metal ratio from the dust mass and
            # metallicity
            self.dust_to_metal_ratio = self.dust_masses / (
                self.masses * self.metallicities
            )

            self.dust_to_metal_ratio[self.dust_masses == 0.0] = 0.0
            self.dust_to_metal_ratio[self.metallicities == 0.0] = 0.0

        # Check the arguments we've been given
        self._check_gas_args()

        # Set any extra properties
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _check_gas_args(self):
        """Sanitize the inputs ensuring all arguments agree and are compatible.

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
                        "Inconsistent gas array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

    def calculate_dust_mass(self):
        """Calculate dust mass from a given dust-to-metals ratio."""
        self.dust_masses = (
            self.masses * self.metallicities * self.dust_to_metal_ratio
        )

    def __str__(self):
        """Return a string representation of the gas object.

        Returns:
            table (str):
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Gas")

    def __add__(self, other):
        """Add two gas objects together.

        Args:
            other (Gas):
                The gas object to add to this one.

        Returns:
            new_gas (Gas):
                A new gas object containing the combined gas particles.
        """
        # Ensure we have a gas object
        if not isinstance(other, Gas):
            raise exceptions.InconsistentAddition(
                "Can only add two Gas objects together!"
            )

        # Concatenate all the named arguments which need it (Nones are handled
        # inside the combine_arrays function)
        masses = combine_arrays(self.masses, other.masses)
        metallicities = combine_arrays(self.metallicities, other.metallicities)
        star_forming = combine_arrays(self.star_forming, other.star_forming)
        coordinates = combine_arrays(self.coordinates, other.coordinates)
        velocities = combine_arrays(self.velocities, other.velocities)
        smoothing_lengths = combine_arrays(
            self.smoothing_lengths, other.smoothing_lengths
        )
        dust_masses = combine_arrays(self.dust_masses, other.dust_masses)

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
            "masses": masses,
            "metallicities": metallicities,
            "star_forming": star_forming,
            "redshift": redshift,
            "coordinates": coordinates,
            "velocities": velocities,
            "smoothing_lengths": smoothing_lengths,
            "softening_lengths": softening_lengths,
            "dust_masses": dust_masses,
            "tau_v": tau_v,
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

        return Gas(**kwargs)
