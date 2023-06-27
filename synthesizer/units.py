"""A module for dynamically returning attributes with and without units.

The Units dataclass below acts as a container of unit definitions for various
attributes spread throughout Synthesizer.

The Quantity is the object that defines all attributes with attached units. Its
a helper class which enables the optional return of units.

Example defintion:

    class Foo:

        bar = Quantity()

        def __init__(self, bar):
            self.bar = bar

Example usage:

    foo = Foo(bar)

    bar_with_units = foo.bar
    bar_no_units = foo._bar
    
"""
from dataclasses import dataclass
from unyt import nJy, erg, s, Hz, Angstrom, cm, km, Msun, yr


@dataclass
class Units:

    """
    Holds the definition of the internal unit system using unyt.
    """

    # Wavelengths
    lam = Angstrom
    obslam = Angstrom
    wavelength = Angstrom

    # Frequencies
    nu = Hz
    nuz = Hz

    # Luminosities
    luminosity = erg / s  # luminosity
    lnu = erg / s / Hz  # spectral luminosity density
    llam = erg / s / Angstrom  # spectral luminosity density
    continuum = erg / s / Hz  # the continuum level of an emission line

    # Fluxes
    flux = erg / s / cm**2
    fnu = nJy

    # Equivalent width
    ew = Angstrom

    # Spatial quantities
    coordinates = Mpc
    smoothing_lengths = Mpc
    softening_length = Mpc

    # Velocities
    velocities = km / s

    # Masses
    masses = Msun
    initial_masses = Msun
    current_masses = Msun

    # Time quantities
    ages = yr


class Quantity:

    """
    Provides the ability to associate values with the unyt definition of the
    internal unit system to output quantities.

    Attributes:
        


    Methods
    -------
    __init__
        Initialise and set Units
    __set_name__
        Define public and private names
    __get__
        Return the quantity, a combination of the (unit-less) value and the unit
    __set__
        Set the value.

    """

    def __init__(self):
        # I suppose this could be a parameter allowing you to change the
        # implementation of units, e.g. swapping to astropy
        self.units = Units()  

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = '_' + name

    def __get__(self, obj, type=None):
        value = getattr(obj, self.private_name)
        unit = getattr(self.units, self.public_name)
        return value * unit

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)
