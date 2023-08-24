"""A module for working with arrays of black holes.

Contains the BlackHoles class for use with particle based systems. This houses
all the data detailing collections of black hole particles. Each property is
stored in (N_bh, ) shaped arrays for efficiency.

When instantiate a BlackHoles object a myriad of extra optional properties can
be set by providing them as keyword arguments.

Example usages:

    bhs = BlackHoles(masses, metallicities,
                     redshift=redshift, accretion_rate=accretion_rate, ...)
"""
import numpy as np
from unyt import c

from synthesizer.particle.particles import Particles
from synthesizer.units import Quantity
from synthesizer import exceptions


class BlackHoles(Particles):
    """
    The base BlackHoles class. This contains all data a collection of black
    holes could contain. It inherits from the base Particles class holding
    attributes and methods common to all particle types.

    The BlackHoles class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be initialised with a BlackHoles object for use with any of the Galaxy
    helper methods.

    Note that due to the many possible operations, this class has a large number
    of optional attributes which are set to None if not provided.

    Attributes:
        accretion_rate (array-like, float)
            The accretion rate of the/each black hole in Msun/yr.
        metallicities (array-like, float)
            The metallicity of the region surrounding the/each black hole.
        nbh (int)
            The number of black hole particles in the object.
        bol_luminosity (array_like, float)
            The bolometric luminosity of the/each black hole in erg/s/Hz. Only
            populated when calculate_bolometric_luminosity is called.
        bb_temperature (array_like, float)
            The "Big Bump" temperature of the/each black hole.
    """

    # Define the allowed attributes
    __slots__ = ["masses", "metallicities", "nparticles",
                 "redshift", "accretion_rate", "bb_temperature",
                 "bol_luminosity", "nbh"]

    # Define class level Quantity attributes
    accretion_rate = Quantity()
    bol_luminosity = Quantity()
    bb_temperture = Quantity()

    def __init__(self, masses, metallicities, redshift=None,
                 accretion_rate=None, coordinates=None,
                 velocities=None, softening_length=None):
        """
        Intialise the Stars instance. The first 3 arguments are always required.
        All other arguments are optional attributes applicable in different
        situations.

        Args:
            masses (array-like, float)
                The mass of each particle in Msun.
            metallicities (array-like, float)
                The metallicity of the region surrounding the/each black hole.
            redshift (float)
                The redshift/s of the black hole particles.
            accretion_rate (array-like, float)
                The accretion rate of the/each black hole in Msun/yr.
            coordinates (array-like, float)
                The 3D positions of the particles.
            velocities (array-like, float)
                The 3D velocities of the particles.
            softening_length (float)
                The physical gravitational softening length.

        """

        #  TODO: handle when individual values are passed instead of arrays,
        # i.e. when there is only a single black hole.

        # Instantiate parent
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=masses,
            redshift=redshift,
            softening_length=softening_length,
            nparticles=len(masses)
        )

        # Accretion rate in Msun / yr
        self.accretion_rate = accretion_rate

        # Bolometric luminosity
        self.bol_luminosity = None

        # Calculate the big bump temperature
        self.bb_temperature = 2.24E9 * \
            self.accretion_rate ** (1 / 4) * self.masses * -0.5

        # The metallicity of the region surrounding the black hole.
        self.metallicities = metallicities

        # Set a frontfacing clone of the number of particles with clearer
        # naming
        self.nbh = self.nparticles

        # Check the arguments we've been given
        self._check_bh_args()

    def _check_bh_args(self):
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
                        "Inconsistent black hole array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

    def calculate_bolometric_luminosity(self, epsilon=0.1):
        """
        Create the black hole bolometric luminosity

        Parameters
        ----------
        mdot: unyt_array or float
            The black hole accretion rate. If not unyt_array assume Msun/yr

        epsilon: float
            The radiative efficiency, by default 0.1.

        Returns
        ----------
        unyt_array
            The black hole bolometric luminosity

        """

        self.bol_luminosity = epsilon * self.accretion_rate * c**2

        return self.bol_luminosity


# class Cloudy:

#     """
#     A class to hold routines for employing the Cloudy AGN model.
#     """

#     def __init__(self):
#         return None


# class Feltre16:

#     """
#     A class to hold routines for employing the Feltre16 AGN model.
#     """

#     def __init__(self):
#         return None

#     def incident(self, lam, alpha, luminosity=1):
#         """
#         Create intrinsic narrow-line AGN spectra as utilised by Feltre et al. (2016). This is utilised to build the cloudy grid.

#         Parameters
#         ----------
#         lam : array
#             Wavelength grid (array) in angstrom or unyt

#         alpha: float
#             UV/optical power-law index. Expected to be -2.0<alpha<-1.2

#         luminosity: float
#             Bolometric luminosity. Set to unity.


#         Returns
#         -------
#         lnu
#             Spectral luminosity density.
#         """

#         # create empty luminosity array
#         lnu = np.zeros(lam.shape)

#         # calculate frequency
#         nu = c/lam

#         # define edges
#         edges = [10., 2500., 100000., 1000000.] * Angstrom  # Angstrom

#         # define indices
#         indices = [alpha, -0.5, 2.]

#         # define normalisations
#         norms = [1.]

#         # calcualte remaining normalisations
#         for i, (edge_lam, ind1, ind2) in enumerate(zip(edges[1:], indices, indices[1:])):

#             edge_nu = c/edge_lam

#             norm = norms[i]*(edge_nu**ind1)/(edge_nu**ind2)
#             norms.append(norm)

#         # now construct spectra
#         for e1, e2, ind, norm in zip(edges[0:], edges[1:], indices, norms):

#             # identify indices within the wavelength range
#             s = (lam >= e1) & (lam < e2)

#             lnu[s] = norm * nu[s]**ind

#         # normalise -- not yet implemented

#         return lnu
