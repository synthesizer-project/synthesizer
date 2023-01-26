import numpy as np
from astropy.modeling.models import Sersic2D as Sersic2D_
import matplotlib.pyplot as plt



class MorphologyBase:
    """
    A base class holding common methods for parametric morphology descriptions


    Methods
    -------
    plot
        shows a plot of the model for a given resolution and npix
    """

    def plot(self, resolution, npix=None):
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

    def __init__(self, p):

        parameters = {
            'r_eff': 1,
            'n': 1,
            'ellip': 0,
            'theta': 0}

        for key, value in list(p.items()):
            parameters[key] = value

        self.model = Sersic2D_(amplitude=1, r_eff=parameters['r_eff'],
                               n=parameters['n'], ellip=parameters['ellip'], theta=parameters['theta'])

    def img(self, xx, yy):
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

        return self.model(xx, yy)

