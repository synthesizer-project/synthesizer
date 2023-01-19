

import unyt
from unyt import nJy, erg, s, Hz, Angstrom, cm


class Units:
    wavelength = Angstrom
    lnu = erg/s/Hz
    luminosity = erg/s
    flux = erg/s/cm**2
    continuum = erg/s/Hz  # the continuum level of an emission line
    fnu = nJy
    ew = Angstrom


class Quantity:

    def __init__(self):
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


# class Property:
#
#     def __init__(self):
#         self.units = {
#             'wavelength': Angstrom,
#             'lnu': erg/s,
#             'luminosity': erg/s,
#             'flux': erg/s/cm**2,
#             'lnu': erg/s/Hz,
#             'continuum': erg/s/Hz,  # the continuum level of an emission line
#             'fnu': nJy,
#             'ew': Angstrom,
#         }
#
#     def __set_name__(self, owner, name):
#         self.public_name = name
#         self.private_name = '_' + name
#
#     def __get__(self, obj, type=None):
#         value = getattr(obj, self.private_name)
#         return value * self.units[self.public_name]
#
#     def __set__(self, obj, value):
#         setattr(obj, self.private_name, value)
