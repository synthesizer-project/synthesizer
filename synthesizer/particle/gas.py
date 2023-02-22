# -*- coding: utf-8 -*-
"""
Contains the Gas class for use with particle based systems. This houses all
the data detailing collections of gas particles, where each property is
stored as arrays for efficiency.

Notes
-----
"""

from .particles import Particles


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

    Attributes
    ----------
    metallicities : array-like (float)
        The gas phase metallicity of each particle (integrated)
    star_forming : array-like (bool)
        Flag for whether a gas particle is star forming or not
    nparticle : int
        How many gas particles are there?
    log10metallicities : float
        Convnience attribute containing log10(metallicity).
    coordinates : array-like (float)
        The coordinates of each gas particle in simulation length
        units.
    velocities : array-like (float)
        The velocity of each gas particle (km/s.)
    masses : array-like (float)
        The mass of each gas particle in Msun.
    smoothing_lengths : array-like (float)
        The smoothing lengths (describing the sph kernel) of each gas
        particle in simulation length units.
    s_oxygen : array-like (float)
        fractional oxygen abundance
    s_hydrogen : array-like (float)
        fractional hydrogen abundance
    """

        # Define the allowed attributes
    __slots__ = ["metallicities", "star_forming", "nparticles",
                 "log10metallicities", "coordinates",
                 "velocities", "masses", "smoothing_lengths",
                 "s_oxygen", "s_hydrogen"]

    def __init__(self, masses, metallicities, **kwargs):
        self.masses = masses
        self.metallicities = metallicities

        # Ensure all attributes are intialised to None
        # NOTE (Will): I don't like this but can't think of something cleaner
        for attr in Gas.__slots__:
            try:
                getattr(self, attr)
            except AttributeError:
                setattr(self, attr, None)

        # Handle kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

