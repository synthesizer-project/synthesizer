



import numpy as np

import unyt
from unyt import c, h, nJy, erg, s, Hz, pc

from .sed import convert_fnu_to_flam

class Line:

    """
    A class representing a spectral line or set of lines (e.g. a doublet)

    Attributes
    ----------
    lam : wavelength of the line

    Methods
    -------

    """

    def __init__(self, id_, wavelength_, luminosity_, continuum_):

        self.id_ = id_
        self.wavelength_ = wavelength_
        self.luminosity_ = luminosity_
        self.continuum_ = continuum_

        self.id = ','.join(id_)
        self.continuum = np.mean(continuum_) # mean continuum value
        self.wavelength = np.mean(wavelength_) # mean wavelength of the line
        self.luminosity = np.sum(luminosity_) # total luminosity of the line

        continuum_lam = convert_fnu_to_flam(self.wavelength, self.continuum)  # continuum at line wavelength, erg/s/AA
        self.ew = self.luminosity / continuum_lam  # AA
        print(self.ew)

    def summary(self):

        print('-'*5, self.id)
        print(f'log10(luminosity/erg/s): {np.log10(self.luminosity):.2f}')
        print(f'EW/AA: {self.ew:.0f}')
