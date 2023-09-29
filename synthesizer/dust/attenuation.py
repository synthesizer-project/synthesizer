import os
import numpy as np
from scipy import interpolate
from unyt import um

# from dust_attenuation.shapes import N09
from dust_extinction.grain_models import WD01

this_dir, this_filename = os.path.split(__file__)

__all__ = ["PowerLaw", "MW_N18", "Calzetti2000", "GrainsWD01"]


class PowerLaw:
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

        self.description = "simple power law dust curve"
        self.params = params

    def tau_x(self, lam):
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

    def tau(self, lam):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        # tau_x = (lam.to('Angstrom')/(5500.*Angstrom))**self.params['slope']
        # tau_V = np.interp(5500., lam.to('Angstrom').v, tau_x)

        return self.tau_x(lam) / self.tau_x(5500.0)

    def attenuate(self, tau_V, lam):
        """
        Provide the transmitted flux/luminosity fraction

        Parameters
        ----------
        tau_V: float
            optical depth in the V-band

        lam: float
            wavelength, expected with units
        """

        tau_x_v = self.tau(lam)

        return np.exp(-(tau_V * tau_x_v))


class MW_N18:
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

        self.description = "MW extinction curve from Desika"
        self.d = np.load(f"{this_dir}/data/MW_N18.npz")
        self.tau_lam_V = np.interp(
            5500.0, self.d.f.mw_df_lam[::-1], self.d.f.mw_df_chi[::-1]
        )

    def tau(self, lam, interp="cubic"):
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

    def attenuate(self, tau_V, lam):
        """
        Provide the transmitted flux/luminosity fraction

        Parameters
        ----------
        tau_V: float
            optical depth in the V-band

        lam: float
            wavelength, expected with units
        """

        tau_x_v = self.tau(lam)

        return np.exp(-(tau_V * tau_x_v))


class Calzetti2000():
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
        self.description = """ Calzetti attenuation curve; with option 
                               for the slope and UV-bump implemented
                               in Noll et al. 2009"""
        self.params = params

    def tau(self, lam):
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

    def attenuate(self, tau_V, lam):
        """
        Get the transmission at different wavelength for the curve

        Parameters
        ----------
        tau_V: float
            optical depth in the V-band

        lam: float
            wavelength, expected with units
        """
        tau_x_v = self.tau(lam) 
    
        return np.exp(-(tau_V * tau_x_v))


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

    def tau(self, lam):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        return self.emodel(lam.to_astropy())

    def attenuate(self, tau_V, lam):
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
