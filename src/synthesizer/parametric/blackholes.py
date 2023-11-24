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
from unyt import c, deg
from synthesizer.parametric.morphology import PointSource
from synthesizer.components import BlackholesComponent
from synthesizer.units import Quantity
from synthesizer import exceptions


class BlackHoles(BlackholesComponent):
    """
    The base parametric BlackHoles class. 
    """

    def __init__(
        self,
        bolometric_luminosity=None,
        mass=None,
        accretion_rate=None,
        epsilon=0.1,
        inclination=None,
        spin=None,
        metallicity=None,
        offset=None,
    ):
        """
        Intialise the Stars instance. The first two arguments are always 
        required. All other arguments are optional attributes applicable 
        in different situations.

        Args:
            bolometric_luminosity (float)
                The bolometric luminosity
            mass (float)
                The mass of each particle in Msun.
            accretion_rate (float)
                The accretion rate of the/each black hole in Msun/yr.
            metallicity (float)
                The metallicity of the region surrounding the/each black hole.
            epsilon (float)
                The radiative efficiency. By default set to 0.1.
            inclination (float)
                The inclination of the blackhole. Necessary for some disc 
                models.
            spin (float)
                The spin of the blackhole. Necessary for some disc models.
            offset (unyt_array, float)
                The (x,y) offsets of the blackhole relative to the centre of 
                the image. Units can be length or angle but should be 
                consistent with the scene.

        """

        # Initialise base class
        BlackholesComponent.__init__(
            self,
            bolometric_luminosity=bolometric_luminosity,
            mass=mass,
            accretion_rate=accretion_rate,
            epsilon=epsilon,
            inclination=inclination,
            spin=spin,
            metallicity=metallicity,
        )

        # Initialise morphology using the in-built point-source class
        self.morphology = PointSource(offset=offset)




