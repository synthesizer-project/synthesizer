import numpy as np
from astropy.modeling.models import Sersic2D as Sersic2D_
from unyt import kpc, mas
from unyt.dimensions import length, angle
import matplotlib.pyplot as plt


class MorphologyBase:
    """
    A base class holding common methods for parametric morphology descriptions


    Methods
    -------
    plot
        shows a plot of the model for a given resolution and npix
    """

    def plot(self, resolution, npix=None, cosmo=None, z=None):
        """
        Produce a plot of the current morphology

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        bins = resolution * np.arange(-npix/2, npix/2)

        xx, yy = np.meshgrid(bins, bins)

        img = self.img(xx, yy)

        plt.figure()
        plt.imshow(np.log10(img), origin='lower', interpolation='nearest',
                   vmin=-1, vmax=2)
        plt.show()


class Sersic2D(MorphologyBase):

    """
    A class holding the Sersic2D profile. This is a wrapper around the astropy.models.Sersic2D class.


    Methods
    -------
    img
        returns an image
    """

    def __init__(self, p, cosmo=None, z=None):

        self.update(p, cosmo=cosmo, z=z)

    def update(self, p, cosmo=None, z=None):

        self.parameters = {
            'r_eff_kpc': None,
            'r_eff_mas': None,
            'n': 1,
            'ellip': 0,
            'theta': 0}

        for key, value in list(p.items()):
            self.parameters[key] = value

        if p['r_eff'].units.dimensions == angle:
            self.parameters['r_eff_mas'] = p['r_eff'].to('mas').value
        elif p['r_eff'].units.dimensions == length:
            self.parameters['r_eff_kpc'] = p['r_eff'].to('kpc').value

        # if cosmology and redshift provided calculate the conversion of pkpc to mas
        if cosmo and z:
            self.kpc_proper_per_mas = cosmo.kpc_proper_per_arcmin(z).to('kpc/mas').value

            if self.parameters['r_eff_kpc']:
                self.parameters['r_eff_mas'] = self.parameters['r_eff_kpc'] / \
                    self.kpc_proper_per_mas
            else:
                self.parameters['r_eff_kpc'] = self.parameters['r_eff_mas'] * \
                    self.kpc_proper_per_mas

        if self.parameters['r_eff_kpc']:
            self.model_kpc = Sersic2D_(amplitude=1, r_eff=self.parameters['r_eff_kpc'],
                                       n=self.parameters['n'], ellip=self.parameters['ellip'], theta=self.parameters['theta'])
        else:
            self.model_kpc = None

        if self.parameters['r_eff_mas']:
            self.model_mas = Sersic2D_(amplitude=1, r_eff=self.parameters['r_eff_mas'],
                                       n=self.parameters['n'], ellip=self.parameters['ellip'], theta=self.parameters['theta'])
        else:
            self.model_mas = None

    def img(self, xx, yy, units=kpc):
        """
        Produce a plot of the current morphology

        Parameters
        ----------
        xx: float array
            x values on 2D grid
        yy: float array
            y values on 2D grid


        Returns
        ----------
        np.ndarray
            image

        Example
        ----------

        >>> bins = resolution * np.arange(-npix/2, npix/2)
        >>> xx, yy = np.meshgrid(bins, bins)
        >>> img = self.img(xx, yy)

        """

        if units == kpc:
            return self.model_kpc(xx, yy)
        if units == mas:
            return self.model_mas(xx, yy)
