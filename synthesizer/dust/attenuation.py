import os
import numpy as np
from scipy import interpolate
from unyt import um

from dust_extinction.grain_models import WD01

from synthesizer import exceptions

this_dir, this_filename = os.path.split(__file__)

__all__ = ["PowerLaw", "MW_N18", "Calzetti2000", "GrainsWD01"]

def N09_tau(lam, slope, cent_lam, ampl, gamma):
    """
    Attenuation curve using a modified version of the Calzetti
    attenuation (Charlot+2000) law allowing for a varying UV slope
    and the presence of a UV bump; from Noll+2009

    Args:
        lam (array-like, float)
            The input wavelength array.

        slope (float)
            The slope of the attenuation curve.

        cent_lam (float)
            The central wavelength of the UV bump, expected in microns.

        ampl (float)
            The amplitude of the UV-bump.

        gamma (float)
            The width (FWHM) of the UV bump, in microns.

    Returns:
        (array-like, float)
            V-band normalised optical depth for given wavelength
    """

    # Wavelength in microns
    tmp_lam = np.arange(0.12, 2.2, 0.001)
    lam_v = 0.55
    k_lam = np.zeros_like(tmp_lam)

    ok = (tmp_lam < 0.63)
    k_lam[ok] = -2.156 + (1.509 / tmp_lam[ok]) \
        - (0.198 / tmp_lam[ok] ** 2) \
        + (0.011 / tmp_lam[ok] ** 3)
    k_lam[~ok] = -1.857 + (1.040 / tmp_lam[~ok])
    k_lam = 4.05 + 2.659 * k_lam

    D_lam = ampl * ((tmp_lam * gamma) ** 2) \
        / ((tmp_lam ** 2 - cent_lam ** 2) ** 2 + (tmp_lam * gamma) ** 2)

    tau_x_v = (1 / 4.05) * (k_lam + D_lam) * ((tmp_lam / lam_v) ** slope)

    func = interpolate.interp1d(tmp_lam, tau_x_v,
                                fill_value="extrapolate")

    return func(lam.to(um))


class AttenuationLaw:
    """
    A generic parent class for dust attenuation laws to hold common attributes
    and methods

    Attributes:
        description (str)
            A description of the type of model. Defined on children classes.
    """

    def __init__(self, description):
        """
        Initialise the parent and set common attributes.
        """

        # Store the description of the model.
        self.description = description

    def get_tau(self, *args):
        """
        A prototype method to be overloaded by the children defining various
        models.
        """
        raise exceptions.UnimplementedFunctionality(
            "AttenuationLaw should not be instantiated directly!"
            " Instead use one to child models ("
            + ", ".join(__all__) + ")"
        )

    def get_transmission(self, tau_v, lam):
        """
        Returns the transmitted flux/luminosity fraction based on an optical
        depth at a range of wavelengths.

        Args:
            tau_v (float/array-like, float)
                Optical depth in the V-band. Can either be a single float or
                array.

            lam (array-like, float)
                The wavelengths (with units) at which to calculate transmission.

        Returns:
            array-like
                The transmission at each wavelength. Either (lam.size,) in shape
                for singular tau_v values or (tau_v.size, lam.size) tau_v
                is an array.
        """

        # Get the optical depth at each wavelength
        tau_x_v = self.get_tau(lam)

        # Include the V band optical depth in the exponent
        exponent = tau_v * tau_x_v

        return np.exp(-exponent)


class PowerLaw(AttenuationLaw):
    """
    Custom power law dust curve.

    Attributes:
        slope (float)
            The slope of the power law.
    """

    def __init__(self, slope=-1.0):
        """
        Initialise the power law slope of the dust curve.

        Args:
            params (dict)
                A dictionary containing the parameters for the model.
        """

        description = "simple power law dust curve"
        AttenuationLaw.__init__(self, description)
        self.slope = slope

    def get_tau_at_lam(self, lam):
        """
        Calculate optical depth at a wavelength.

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/array-like, float
                The optical depth.
        """

        return (lam / 5500.0) ** self.slope

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth.

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/array-like, float
                The optical depth.
        """

        return self.get_tau_at_lam(lam) / self.get_tau_at_lam(5500.0)


class MW_N18(AttenuationLaw):
    """
    Milky Way attenuation curve used in Narayanan+2018.

    Attributes:
        data (array-like, float)
            The data describing the dust curve, loaded from MW_N18.npz.
        tau_lam_v (float)
            The V band optical depth.
    """

    def __init__(self):
        """
        Initialise the dust curve by loading the data and get the V band
        optical depth by interpolation.
        """

        description = "MW extinction curve from Desika"
        AttenuationLaw.__init__(self, description)
        self.data = np.load(f"{this_dir}/data/MW_N18.npz")
        self.tau_lam_v = np.interp(
            5500.0, self.data.f.mw_df_lam[::-1], self.data.f.mw_df_chi[::-1]
        )

    def get_tau(self, lam, interp="cubic"):
        """
        Calculate V-band normalised optical depth.

        Args:
            lam (float/array, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.
            interp (str)
                The type of interpolation to use. Can be ‘linear’, ‘nearest’,
                ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’,
                ‘previous’, or ‘next’. ‘zero’, ‘slinear’, ‘quadratic’ and
                ‘cubic’ refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array, float
                The optical depth.
        """

        func = interpolate.interp1d(
            self.data.f.mw_df_lam[::-1],
            self.data.f.mw_df_chi[::-1],
            kind=interp,
            fill_value="extrapolate",
        )

        return func(lam.to("Angstrom").v) / self.tau_lam_v


class Calzetti2000(AttenuationLaw):
    """
    Calzetti attenuation curve; with option for the slope and UV-bump
    implemented in Noll et al. 2009.

    Attributes:
        slope (float)
            The slope of the attenuation curve.

        cent_lam (float)
            The central wavelength of the UV bump, expected in microns.

        ampl (float)
            The amplitude of the UV-bump.

        gamma (float)
            The width (FWHM) of the UV bump, in microns.

    """

    def __init__(self, slope=0, cent_lam=0.2175, ampl=1, gamma=0.035):
        """
        Initialise the dust curve.

        Args:
            slope (float)
                The slope of the attenuation curve.

            cent_lam (float)
                The central wavelength of the UV bump, expected in microns.

            ampl (float)
                The amplitude of the UV-bump.

            gamma (float)
                The width (FWHM) of the UV bump, in microns.
        """
        description = (
            "Calzetti attenuation curve; with option"
            "for the slope and UV-bump implemented"
            "in Noll et al. 2009"
        )
        AttenuationLaw.__init__(self, description)

        # Define the parameters of the model.
        self.slope = slope
        self.cent_lam = cent_lam
        self.ampl = ampl
        self.gamma = gamma

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth. (Uses the N09_tau function
        defined above.)

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/array-like, float
                The optical depth.
        """
        return N09_tau(
            lam=lam,
            slope=self.slope,
            cent_lam=self.cent_lam,
            ampl=self.ampl,
            gamma=self.gamma,
        )


class GrainsWD01:
    """
    Weingarter and Draine 2001 dust grain extinction model
    for MW, SMC and LMC or any available in WD01.

    NOTE: this model does not inherit from AttenuationLaw because it is
          distinctly different.

    Attributes:
        model (str)
            The dust grain model used.
        emodel (function)
            The function that describes the model from WD01 imported above.
    """

    def __init__(self, model="SMCBar"):
        """
        Initialise the dust curve

        Args:
            model (str)
                The dust grain model to use.

        """

        self.description = (
            "Weingarter and Draine 2001 dust grain extinction"
            " model for MW, SMC and LMC"
        )

        # Get the correct model string
        if "MW" in model:
            self.model = "MWRV31"
        elif "LMC" in model:
            self.model = "LMCAvg"
        elif "SMC" in model:
            self.model = "SMCBar"
        else:
            self.model = model
        self.emodel = WD01(self.model)

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth.

        Args:
            lam (float/array-like, float)
                An array of wavelengths or a single wavlength at which to
                calculate optical depths. Must have unyts attached.

        Returns:
            float/array-like, float
                The optical depth.
        """
        return self.emodel(lam.to_astropy())

    def get_transmission(self, tau_v, lam):
        """
        Returns the transmitted flux/luminosity fraction based on an optical
        depth at a range of wavelengths.

        Args:
            tau_v (float/array-like, float)
                Optical depth in the V-band. Can either be a single float or
                array.

            lam (array-like, float)
                The wavelengths (with units) at which to calculate transmission.

        Returns:
            array-like
                The transmission at each wavelength. Either (lam.size,) in shape
                for singular tau_v values or (tau_v.size, lam.size) tau_v
                is an array.
        """
        return self.emodel.extinguish(x=lam.to_astropy(), Av=1.086 * tau_v)
