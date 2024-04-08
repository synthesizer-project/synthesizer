"""A module for working with a parametric black holes.

Contains the BlackHole class for use with parametric systems. This houses
all the attributes and functionality related to parametric black holes.

Example usages:

    bhs = BlackHole(
        bolometric_luminosity,
        mass,
        accretion_rate,
        epsilon,
        inclination,
        spin,
        metallicity,
        offset,
)
"""

import numpy as np
from unyt import kpc, unyt_array

from synthesizer import exceptions
from synthesizer.components import BlackholesComponent
from synthesizer.parametric.morphology import PointSource
from synthesizer.utils import has_units


class BlackHole(BlackholesComponent):
    """
    The base parametric BlackHole class.

    Attributes:
        morphology (PointSource)
            An instance of the PointSource morphology that describes the
            location of this blackhole
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
        offset=np.array([0.0, 0.0]) * kpc,
        **kwargs,
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
            offset (unyt_array)
                The (x,y) offsets of the blackhole relative to the centre of
                the image. Units can be length or angle but should be
                consistent with the scene.
            kwargs (dict)
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
        """

        if bolometric_luminosity is not None:
            bolometric_luminosity = unyt_array([bolometric_luminosity])
        if mass is not None:
            mass = unyt_array([mass])
        if accretion_rate is not None:
            accretion_rate = unyt_array([accretion_rate])
        if epsilon is not None:
            epsilon = unyt_array([epsilon])
        if inclination is not None:
            inclination = unyt_array([inclination])
        if spin is not None:
            spin = unyt_array([spin])
        if metallicity is not None:
            metallicity = unyt_array([metallicity])

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
            **kwargs,
        )

        # Ensure the offset has units
        if not has_units(offset):
            raise exceptions.MissingUnits(
                "The offset must be provided with units"
            )

        # Initialise morphology using the in-built point-source class
        self.morphology = PointSource(offset=offset)

        # Set a frontfacing clone of the number of particles with clearer
        # naming
        self.nbh = 1
