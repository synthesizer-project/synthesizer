

import numpy as np

import unyt
from unyt import c, h, nJy, erg, s, Hz, pc

from .sed.sed import convert_fnu_to_flam


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
        self.continuum = np.mean(continuum_)  #  mean continuum value
        self.wavelength = np.mean(wavelength_)  # mean wavelength of the line
        self.luminosity = np.sum(luminosity_)  # total luminosity of the line

        # continuum at line wavelength, erg/s/AA
        self.continuum_lam = convert_fnu_to_flam(self.wavelength, self.continuum)
        self.ew = self.luminosity / self.continuum_lam  # AA

    def summary(self):

        print('-'*5, self.id)
        print(f'log10(continuum/erg/s/\AA): {np.log10(self.continuum_lam):.2f}')
        print(f'log10(continuum/erg/s/Hz): {np.log10(self.continuum):.2f}')
        print(f'log10(luminosity/erg/s): {np.log10(self.luminosity):.2f}')
        print(f'EW/\AA: {self.ew:.0f}')
