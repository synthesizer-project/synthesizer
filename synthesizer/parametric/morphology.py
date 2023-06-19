import numpy as np
from astropy.modeling.models import Sersic2D as Sersic2D_
from unyt import kpc, mas
from unyt.dimensions import length, angle
import matplotlib.pyplot as plt
import synthesizer.exceptions


class MorphologyBase:
    """
    A base class holding common methods for parametric morphology descriptions


    Methods
    -------
    plot_density_grid
        shows a plot of the model for a given resolution and npix
    """

    def plot_density_grid(self, resolution, npix=None, cosmo=None, z=None):
        """
        Produce a plot of the current morphology

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        bins = resolution * np.arange(-npix/2, npix/2)

        xx, yy = np.meshgrid(bins, bins)

        img = self.compute_density_grid(xx, yy)

        plt.figure()
        plt.imshow(np.log10(img), origin='lower', interpolation='nearest',
                   vmin=-1, vmax=2)
        plt.show()


class Sersic2D(MorphologyBase):

    """
    A class holding the Sersic2D profile. This is a wrapper around the
    astropy.models.Sersic2D class.

    Methods
    -------
    compute_density_grid
        Calculates and returns the density grid defined by this Morphology
    """

    def __init__(self, params, cosmo=None, z=None):

        # Define the initial parameter set
        self.parameters = {
            'r_eff_kpc': None,
            'r_eff_mas': None,
            'n': 1,
            'ellip': 0,
            'theta': 0,
        }

        # Initialise the Morphology object based on the 
        self.update(params, cosmo=cosmo, z=z)

    def update(self, p, cosmo=None, z=None):

        for key, value in list(p.items()):
            self.parameters[key] = value

        if p['r_eff'].units.dimensions == angle:
            self.parameters['r_eff_mas'] = p['r_eff'].to('mas').value
        elif p['r_eff'].units.dimensions == length:
            self.parameters['r_eff_kpc'] = p['r_eff'].to('kpc').value

        # If cosmology and redshift provided calculate the conversion of pkpc to mas
        if cosmo and z:
            self.kpc_proper_per_mas = cosmo.kpc_proper_per_arcmin(z).to('kpc/mas').value

            if self.parameters['r_eff_kpc']:
                self.parameters['r_eff_mas'] = self.parameters['r_eff_kpc'] / \
                    self.kpc_proper_per_mas
            else:
                self.parameters['r_eff_kpc'] = self.parameters['r_eff_mas'] * \
                    self.kpc_proper_per_mas

        if self.parameters['r_eff_kpc']:
            self.model_kpc = Sersic2D_(
                amplitude=1, r_eff=self.parameters['r_eff_kpc'],
                n=self.parameters['n'], ellip=self.parameters['ellip'],
                theta=self.parameters['theta']
            )
        else:
            self.model_kpc = None

        if self.parameters['r_eff_mas']:
            self.model_mas = Sersic2D_(
                amplitude=1, r_eff=self.parameters['r_eff_mas'],
                n=self.parameters['n'], ellip=self.parameters['ellip'],
                theta=self.parameters['theta']
            )
        else:
            self.model_mas = None

    def compute_density_grid(self, xx, yy, units=kpc):
        """
        Compute the density grid defined by this morphology as a function of
        the input coordinate grids.

        This acts as a wrapper to astropy functionality (defined above) which
        only work in units of kpc or milliarcseconds (mas)

        Parameters
        ----------
        xx: array-like (float)
            x values on a 2D grid.
        yy: array-like (float)
            y values on a 2D grid.
        units : unyt.unit
            The units in which the coordinate grids are defined.

        Returns
        ----------
        density_grid : np.ndarray
            The density grid produced
        """

        # Call the appropriate model function
        if units == kpc:
            return self.model_kpc(xx, yy)
        elif units == mas:
            return self.model_mas(xx, yy)
        else:
            raise exceptions.InconsistentArguments(
                "Only kpc and milliarcsecond (mas) units are supported "
                "for morphologies."
            )
