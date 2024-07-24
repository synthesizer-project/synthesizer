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

from synthesizer import exceptions
from synthesizer.particle.particles import Particles
from synthesizer.units import Quantity
from synthesizer.utils import TableFormatter
from synthesizer.warnings import warn


class Gas(Particles):
    """
    The base Gas class. This contains all data a collection of gas particles
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
        metallicities (array-like, float)
            The gas phase metallicity of each particle (integrated)
        star_forming (array-like, bool)
            Flag for whether each gas particle is star forming or not.
        log10metallicities (float)
            Convnience attribute containing log10(metallicity).
        smoothing_lengths (array-like, float)
            The smoothing lengths (describing the sph kernel) of each gas
            particle in simulation length units.
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
    smoothing_lengths = Quantity()
    dust_masses = Quantity()

    def __init__(
        self,
        masses,
        metallicities,
        star_forming=None,
        redshift=None,
        coordinates=None,
        velocities=None,
        smoothing_lengths=None,
        softening_length=None,
        dust_to_metal_ratio=None,
        dust_masses=None,
        verbose=False,
        centre=None,
        metallicity_floor=1e-5,
        tau_v=None,
        **kwargs,
    ):
        """
        Initialise the gas object.

        Args:
            masses (array-like, float)
                The mass of each particle in Msun.
            metallicities (array-like, float)
                The metallicity of each gas particle.
            star_forming (array-like, bool)
                Flag for whether each gas particle is star forming or not.
            redshift (float)
                The redshift/s of the stellar particles.
            coordinates (array-like, float)
                The 3D positions of the particles.
            velocities (array-like, float)
                The 3D velocities of the particles.
            smoothing_lengths (array-like, float)
                The smoothing lengths (describing the sph kernel) of each
                gas particle in simulation length units.
            dust_to_metal_ratio (array_like, float or float)
                The ratio between dust and total metal content in a gas
                particle. This can either be a single float or an array of
                values for each gas particle.
            dust_masses (array_like, float)
                Mass of dust in each particle in Msun.
            verbose (bool)
                Whether to print extra information to the console.
            centre (array-like, float)
                The centre of the galaxy in simulation length units.
            metallicity_floor (float)
                The metallicity floor when using log properties (only matters
                for baryons). This is used to avoid log(0) errors.
            tau_v (float)
                The dust optical depth in the V band.
            **kwargs
                Extra optional properties to set on the gas object.
        """

        # Instantiate parent
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=masses,
            redshift=redshift,
            softening_length=softening_length,
            nparticles=len(masses),
            centre=centre,
            metallicity_floor=metallicity_floor,
            tau_v=tau_v,
            name="Gas",
        )

        # Set the metallicites and log10 equivalent
        self.metallicities = metallicities
        self.log10metallicities = np.log10(self.metallicities)

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

            # TODO: this should be removed when dust masses are
            # properly propagated to LOS calculation
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
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

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
        """
        Calculate dust mass from a given dust-to-metals ratio
        and gas particle properties (mass and metallicity)
        """
        self.dust_masses = (
            self.masses * self.metallicities * self.dust_to_metal_ratio
        )

    def __str__(self):
        """
        Return a string representation of the stars object.

        Returns:
            table (str)
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Gas")
