from dataclasses import dataclass
from unyt import nJy, erg, s, Hz, Angstrom, cm


@dataclass
class Units:

    """
    Holds the definition of the internal unit system using unyt.
    """

    lam = Angstrom
    lamz = Angstrom
    wavelength = Angstrom
    nu = Hz
    nuz = Hz

    luminosity = erg / s  # luminosity
    lnu = erg / s / Hz  # spectral luminosity density
    llam = erg / s / Angstrom  # spectral luminosity density
    continuum = erg / s / Hz  # the continuum level of an emission line

    flux = erg / s / cm**2
    fnu = nJy

    ew = Angstrom  # equivalent width


class Quantity:

    """
    Provides the ability to associate values with the unyt definition of the internal unit system to output quantities.


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
        self.units = (
            Units()
        )  # I suppose this could be a parameter allowing you to change the implementation of units, e.g. swapping to astropy

    def __set_name__(self, owner, name):
        self.public_name = name
        self.private_name = "_" + name

    def __get__(self, obj, type=None):
        value = getattr(obj, self.private_name)
        unit = getattr(self.units, self.public_name)
        return value * unit

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)
