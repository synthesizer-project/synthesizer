import numpy as np
from astropy.modeling.models import Sersic2D as Sersic2D_
from unyt import kpc, mas
from unyt.dimensions import length, angle
import matplotlib.pyplot as plt

from synthesizer import exceptions


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

        bins = resolution * np.arange(-npix / 2, npix / 2)

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

    def __init__(self, r_eff_kpc=None, r_eff_mas=None, n=1, ellip=0, theta=0.0,
                 cosmo=None, redshift=None):
        """
        """

        # Define the parameter set
        self.r_eff_kpc = r_eff_kpc
        self.r_eff_mas = r_eff_mas
        self.n = n
        self.ellip = ellip
        self.theta = theta

        # Associate the cosmology and redshift to this object
        self.cosmo = cosmo
        self.redshift = redshift

        # Check inputs
        self._check_args()

        # If cosmology and redshift have been provided we can calculate both
        # models
        if cosmo is not None and redshift is not None:

            # Compute conversion
            kpc_proper_per_mas = self.cosmo.kpc_proper_per_arcmin(z).to(
                'kpc/mas'
            ).value

            # Calculate one effective radius from the other depending on what
            # we've been given.
            if self.r_eff_kpc is not None:
                self.r_eff_mas = self.r_eff_kpc / self.kpc_proper_per_mas
            else:
                self.r_eff_kpc = self.r_eff_mas * self.kpc_proper_per_mas

        # Intialise the kpc model 
        if self.r_eff_kpc is not None:
            self.model_kpc = Sersic2D_(
                amplitude=1, r_eff=self.r_eff_kpc,
                n=self.n, ellip=self.ellip,
                theta=self.theta
            )
        else:
            self.model_kpc = None

        # Intialise the miliarcsecond model 
        if self.r_eff_mas is not None:
            self.model_mas = Sersic2D_(
                amplitude=1, r_eff=self.r_eff_mas,
                n=self.n, ellip=self.ellip,
                theta=self.theta
            )
        else:
            self.model_mas = None

    def _check_args(self):
        """
        Tests the inputs to ensure they are a valid combination.
        """

        # Ensure at least one effective radius has been passed
        if self.r_eff_kpc is None and self.r_eff_mas is None:
            raise exceptions.InconsistentArguments(
                "An effective radius must be defined in either kpc (r_eff_kpc) "
                "or milliarcseconds (mas)"
            )

        # Ensure cosmo has been provided if redshift has been passed
        if self.redshift is not None and self.cosmo is None:
            raise exceptions.InconsistentArguments(
                "Astropy.cosmology object is missing, cannot perform "
                "comoslogical calculations."
            )
            

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

        # Ensure we have the model corresponding to the requested units
        if units == kpc and self.model_kpc is None:
            raise exceptions.InconsistentArguments(
                "Morphology has not been initialised with a kpc method. "
                "Reinitialise the model or use milliarcseconds."
            )
        elif units == mas and self.model_mas is None:
            raise exceptions.InconsistentArguments(
                "Morphology has not been initialised with a milliarcsecond "
                "method. Reinitialise the model or use kpc."
            )

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
