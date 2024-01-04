import numpy as np
from astropy.modeling.models import Sersic2D as Sersic2D_
from unyt import kpc, mas, unyt_array
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

    def plot_density_grid(self, resolution, npix):
        """
        A simple method to produce a quick density plot.

        Arguments
            resolution (float)
                The resolution (in the same units provded to the child class).
            npix (int)
                The number of pixels.
        """

        bins = resolution * np.arange(-npix / 2, npix / 2)

        xx, yy = np.meshgrid(bins, bins)

        img = self.compute_density_grid(xx, yy)

        plt.figure()
        plt.imshow(
            np.log10(img), origin="lower", interpolation="nearest", vmin=-1, vmax=2
        )
        plt.show()


class Sersic2D(MorphologyBase):

    """
    A class holding the Sersic2D profile. This is a wrapper around the
    astropy.models.Sersic2D class.
    """

    def __init__(
        self,
        r_eff=None,
        sersic_index=1,
        ellipticity=0,
        theta=0.0,
        cosmo=None,
        redshift=None,
    ):
        """
        Initialise the morphology.

        Arguments
            r_eff (unyt)
                Effective radius. This is converted as required.
            sersic_index (float)
                Sersic index.
            ellipticity (float)
                Ellipticity.
            theta (float)
                Theta, the rotation angle.
            cosmo (astro.cosmology)
                astropy cosmology object.
            redshift (float)
                Redshift.

        """
        self.r_eff_mas = None
        self.r_eff_kpc = None

        # Check units of r_eff and convert if necessary.
        if isinstance(r_eff, unyt_array):
            if r_eff.units.dimensions == length:
                self.r_eff_kpc = r_eff.to("kpc").value
            elif r_eff.units.dimensions == angle:
                self.r_eff_mas = r_eff.to("mas").value
            else:
                raise exceptions.IncorrectUnits(
                    "The units of r_eff must have length or angle dimensions"
                )
            self.r_eff = r_eff
        else:
            raise exceptions.MissingAttribute(
                """
            The effective radius must be provided"""
            )

        # Define the parameter set
        self.sersic_index = sersic_index
        self.ellipticity = ellipticity
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
            kpc_proper_per_mas = (
                self.cosmo.kpc_proper_per_arcmin(redshift).to("kpc/mas").value
            )

            # Calculate one effective radius from the other depending on what
            # we've been given.
            if self.r_eff_kpc is not None:
                self.r_eff_mas = self.r_eff_kpc / kpc_proper_per_mas
            else:
                self.r_eff_kpc = self.r_eff_mas * kpc_proper_per_mas

        # Intialise the kpc model
        if self.r_eff_kpc is not None:
            self.model_kpc = Sersic2D_(
                amplitude=1,
                r_eff=self.r_eff_kpc,
                n=self.sersic_index,
                ellip=self.ellipticity,
                theta=self.theta,
            )
        else:
            self.model_kpc = None

        # Intialise the miliarcsecond model
        if self.r_eff_mas is not None:
            self.model_mas = Sersic2D_(
                amplitude=1,
                r_eff=self.r_eff_mas,
                n=self.sersic_index,
                ellip=self.ellipticity,
                theta=self.theta,
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
                "An effective radius must be defined in either kpc (r_eff_kpc)"
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

        Arguments
            xx: array-like (float)
                x values on a 2D grid.
            yy: array-like (float)
                y values on a 2D grid.
            units : unyt.unit
                The units in which the coordinate grids are defined.

        Returns
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


class PointSource(MorphologyBase):

    """
    A class holding the Sersic2D profile. This is a wrapper around the
    astropy.models.Sersic2D class.
    """

    def __init__(
        self,
        offset=np.array([0.0, 0.0]) * kpc,
        cosmo=None,
        redshift=None,
    ):
        """
        Initialise the morphology.

        Arguments
            offset (unyt_array/float)
                The [x,y] offset in angular or physical units from the centre
                of the image. The default (0,0) places the source in the centre
                of the image.
            cosmo (astropy.cosmology)
                astropy cosmology object.
            redshift (float)
                Redshift.

        """
        # Check units of r_eff and convert if necessary
        if isinstance(offset, unyt_array):
            if offset.units.dimensions == length:
                self.offset_kpc = offset.to("kpc").value
            elif offset.units.dimensions == angle:
                self.offset_mas = offset.to("mas").value
            else:
                raise exceptions.IncorrectUnits(
                    "The units of offset must have length or angle dimensions"
                )
        else:
            raise exceptions.MissingUnits("The offset must be provided with units")

        # Associate the cosmology and redshift to this object
        self.cosmo = cosmo
        self.redshift = redshift

        # If cosmology and redshift have been provided we can calculate both
        # models
        if cosmo is not None and redshift is not None:
            # Compute conversion
            kpc_proper_per_mas = (
                self.cosmo.kpc_proper_per_arcmin(redshift).to("kpc/mas").value
            )

            # Calculate one offset from the other depending on what
            # we've been given.
            if self.offset_kpc is not None:
                self.offset_mas = self.offset_kpc / kpc_proper_per_mas
            else:
                self.offset_kpc = self.offset_mas * kpc_proper_per_mas

    def compute_density_grid(self, xx, yy, units=kpc):
        """
        Compute the density grid defined by this morphology as a function of
        the input coordinate grids.

        This acts as a wrapper to astropy functionality (defined above) which
        only work in units of kpc or milliarcseconds (mas)

        Arguments
            xx: array-like (float)
                x values on a 2D grid.
            yy: array-like (float)
                y values on a 2D grid.
            units : unyt.unit
                The units in which the coordinate grids are defined.

        Returns
            density_grid : np.ndarray
                The density grid produced
        """

        # Create empty density grid
        image = np.zeros((len(xx), len(yy)))

        if units == kpc:
            # find the pixel corresponding to the supplied offset
            i = np.argmin(np.fabs(xx[0] - self.offset_kpc[0]))
            j = np.argmin(np.fabs(yy[:, 0] - self.offset_kpc[1]))
            # set the pixel value to 1.0
            image[i, j] = 1.0
            return image

        elif units == mas:
            # find the pixel corresponding to the supplied offset
            i = np.argmin(np.fabs(xx[0] - self.offset_mas[0]))
            j = np.argmin(np.fabs(yy[:, 0] - self.offset_mas[1]))
            # set the pixel value to 1.0
            image[i, j] = 1.0
            return image
        else:
            raise exceptions.InconsistentArguments(
                "Only kpc and milliarcsecond (mas) units are supported "
                "for morphologies."
            )
