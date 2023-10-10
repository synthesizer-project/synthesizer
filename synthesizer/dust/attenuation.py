import os
import numpy as np
from scipy import interpolate
from unyt import um

from dust_extinction.grain_models import WD01

from synthesizer import exceptions

this_dir, this_filename = os.path.split(__file__)

__all__ = ["PowerLaw", "MW_N18", "Calzetti2000", "GrainsWD01"]


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
        Provide the transmitted flux/luminosity fraction based on an optical
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
    Custom power law dust curve

    Attributes
    ----------
    slope: float
        power law slope

    Methods
    -------
    tau
        calculates V-band normalised optical depth
    attenuate
        applies the attenuation curve for given V-band optical
        depth, returns the transmitted fraction
    """

    def __init__(self, params={"slope": -1.0}):
        """
        Initialise the power law slope of the dust curve

        Parameters
        ----------
        slope: float
            power law slope
        """

        description = "simple power law dust curve"
        AttenuationLaw.__init__(self, description)
        self.params = params

    def get_tau_at_lam(self, lam):
        """
        Calculate optical depth at lam

        Parameters
        ----------
        lam: float array
            wavelength, in Angstroms


        Returns
        ----------
        float array
            optical depth
        """

        return (lam / 5500.0) ** self.params["slope"]

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        # tau_x = (lam.to('Angstrom')/(5500.*Angstrom))**self.params['slope']
        # tau_V = np.interp(5500., lam.to('Angstrom').v, tau_x)

        return self.get_tau_at_lam(lam) / self.get_tau_at_lam(5500.0)


class MW_N18(AttenuationLaw):
    """
    Milky Way attenuation curve used in Narayanan+2018

    Attributes
    ----------
    lam: float
        wavlength, expected with units

    Methods
    -------
    tau
        calculates V-band normalised optical depth
    attenuate
        applies the attenuation curve for given V-band optical
        depth, returns the transmitted fraction

    """

    def __init__(self):
        """
        Initialise the dust curve

        Parameters
        ----------
        None
        """

        description = "MW extinction curve from Desika"
        AttenuationLaw.__init__(self, description)
        self.d = np.load(f"{this_dir}/data/MW_N18.npz")
        self.tau_lam_V = np.interp(
            5500.0, self.d.f.mw_df_lam[::-1], self.d.f.mw_df_chi[::-1]
        )

    def get_tau(self, lam, interp="cubic"):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        f = interpolate.interp1d(
            self.d.f.mw_df_lam[::-1],
            self.d.f.mw_df_chi[::-1],
            kind=interp,
            fill_value="extrapolate",
        )

        return f(lam.to("Angstrom").v) / self.tau_lam_V


class Calzetti2000(AttenuationLaw):
    """
    Calzetti attenuation curve; with option for the slope and UV-bump
    implemented in Noll et al. 2009.

    Parameters
    ----------
    slope: float
        slope of the attenuation curve

    x0: float
        central wavelength of the UV bump, expected in microns

    ampl: float
        amplitude of the UV-bump

    Methods
    -------
    tau
        calculates V-band normalised optical depth
    attenuate
        applies the attenuation curve for given V-band optical
        depth, returns the transmitted fraction

    """

    def __init__(self, params={'slope': 0., 'x0': 0.2175, 'ampl': 1., 'gamma': 0.035}):
        """
        Initialise the dust curve

        Parameters
        ----------
        slope: float
            slope of the attenuation curve

        x0: float
            central wavelength of the UV bump, expected in microns

        ampl: float
            amplitude of the UV-bump

        gamma: float
            Width (FWHM) of the UV bump, in microns

        """
        description = (
            "Calzetti attenuation curve; with option"
            "for the slope and UV-bump implemented"
            "in Noll et al. 2009"
        )
        AttenuationLaw.__init__(self, description)
        self.params = params

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected with units
        """
        return N09_tau(lam=lam,
                   slope=self.params['slope'],
                   x0=self.params['x0'],
                   ampl=self.params['ampl'],
                   gamma=self.params['gamma'])


class GrainsWD01:
    """
    Weingarter and Draine 2001 dust grain extinction model
    for MW, SMC and LMC or any available in WD01

    Parameters
    ----------
    model: string
        dust grain model to use

    Methods
    -------
    tau
        calculates V-band normalised optical depth
    attenuate
        applies the extinction curve for given V-band optical
        depth, returns the transmitted fraction
    """

    def __init__(self, params={"model": "SMCBar"}):
        """
        Initialise the dust curve

        Parameters
        ----------
        model: string
            dust grain model to use

        """

        self.description = (
            "Weingarter and Draine 2001 dust grain extinction"
            " model for MW, SMC and LMC"
        )
        self.params = {}
        if "MW" in params["model"]:
            self.params["model"] = "MWRV31"
        elif "LMC" in params["model"]:
            self.params["model"] = "LMCAvg"
        elif "SMC" in params["model"]:
            self.params["model"] = "SMCBar"
        else:
            self.params["model"] = params["model"]

        self.emodel = WD01(self.params["model"])

    def get_tau(self, lam):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        return self.emodel(lam.to_astropy())

    def get_transmission(self, tau_V, lam):
        """
        Get the transmission at different wavelength for the curve

        Parameters
        ----------
        tau_V: float
            optical depth in the V-band

        lam: float
            wavelength, expected with units
        """

        return self.emodel.extinguish(x=lam.to_astropy(), Av=1.086 * tau_V)


def N09_tau(lam, slope, x0, ampl, gamma):
    """
    Attenuation curve using a modified version of the Calzetti
    attenuation (Charlot+2000) law allowing for a varying UV slope 
    and the presence of a UV bump; from Noll+2009

    Args:
        lam (array-like, float)
            The input wavelength array

        slope: float
            slope of the attenuation curve

        x0: float
            central wavelength of the UV bump, expected in microns

        ampl: float
            amplitude of the UV-bump

        gamma: float
            Width (FWHM) of the UV bump, in microns

    Returns:
        (array-like, float)
        V-band normalised optical depth for given wavelength
    """

    # Wavelength in microns
    tmp_lam = np.arange(0.12, 2.2, 0.001) 
    lam_v = 0.55
    k_lam = np.zeros_like(tmp_lam)
    
    ok = (tmp_lam < 0.63)
    k_lam[ok] = -2.156 + (1.509/tmp_lam[ok]) \
        - (0.198/tmp_lam[ok]**2) \
        + (0.011/tmp_lam[ok]**3)
    k_lam[~ok] = -1.857 + (1.040/tmp_lam[~ok])
    k_lam = 4.05 + 2.659 * k_lam

    D_lam = ampl * ((tmp_lam*gamma)**2) \
        / ((tmp_lam**2 - x0**2)**2 + (tmp_lam*gamma)**2)
    
    tau_x_v = (1/4.05) * (k_lam + D_lam) * ((tmp_lam/lam_v)**slope)

    f = interpolate.interp1d(tmp_lam, tau_x_v,
                             fill_value="extrapolate")

    return f(lam.to(um))
