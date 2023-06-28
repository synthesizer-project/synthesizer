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

from synthesizer.particle.particles import Particles
from synthesizer.units import Quantity
from synthesizer import exceptions

class Gas(Particles):
    """
    The base Gas class. This contains all data a collection of gas particles
    could contain. It inherits from the base Particles class holding 
    attributes and methods common to all particle types.

    The Gas class can be handed to methods elsewhere to pass information
    about the gas particles needed in other computations. A galaxy object should 
    have a link to the Gas object containing its gas particles, for example.

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

    # # Define the allowed attributes
    # __slots__ = ["metallicities", "star_forming", 
    #              "log10metallicities", "smoothing_lengths"]

    # Define class level Quantity attributes
    smoothing_lengths = Quantity()

    def __init__(self, masses, metallicities, star_forming=None, redshift=None,
                 coordinates=None, velocities=None, smoothing_lengths=None,
                 softening_length=None):
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
        """

        # Instantiate parent
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=masses,
            redshift=redshift,
            softening_length=softening_length,
            nparticles=len(self.masses)
        )

        # Set the metallicites and log10 equivalent
        self.metallicities = metallicities
        self.log10metallicities = np.log10(self.metallicities)

        # Set the star forming boolean mask array
        self.star_forming = star_forming

        # Set the smoothing lengths for these gas particles
        self.smoothing_lengths = smoothing_lengths

        # Check the arguments we've been given
        self._check_gas_args()

    def _check_gas_args(self):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """

        # Ensure all arrays are the expected length
        for key in self.__dict__:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent gas array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

