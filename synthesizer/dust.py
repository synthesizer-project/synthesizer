
import os
import numpy as np
from unyt import Angstrom
from scipy import interpolate
from dust_attenuation.shapes import N09
from dust_extinction.grain_models import WD01
from . import exceptions
this_dir, this_filename = os.path.split(__file__)

# --- dust curves commonly used in literature

__all__ = ["power_law", "MW_N18", "Calzetti2000", "GrainsWD01"]


class power_law():
    """
    Custom power law dust curve

    Parameters
    ----------
    slope: float
        power law slope
    """
    def __init__(self, params={'slope': -1.}):
        self.description = 'simple power law dust curve'
        self.params = params

    def attenuate(self, tau_V, lam):

        tau_x_v = (lam.to('Angstrom')/5500.*Angstrom)**self.params['slope']
        return np.exp(-(tau_V * tau_x_v))


class MW_N18():
    """
    Milky Way attenuation curve used in Narayanan+2018
    """
    def __init__(self, params={}):
        self.description = 'MW extinction curve from Desika'
        self.d = np.load(f'{this_dir}/data/MW_N18.npz')
        self.tau_lam_V = np.interp(5500.,
                                   self.d.f.mw_df_lam[::-1],
                                   self.d.f.mw_df_chi[::-1])

    def tau(self, lam, interp='cubic'):

        f = interpolate.interp1d(self.d.f.mw_df_lam[::-1],
                                 self.d.f.mw_df_chi[::-1],
                                 kind=interp,
                                 fill_value='extrapolate')
        return f(lam)/self.tau_lam_V


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
    """
    def __init__(self, params={'slope': 0., 'x0': 0.2175, 'ampl': 0.}):
        self.description = 'Calzetti attenuation curve; with option for the slope and UV-bump implemented in Noll et al. 2009'

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
        return N09(Av=1.086*tau_V,
                   ampl=self.params['ampl'],
                   slope=self.params['slope'],
                   x0=self.params['x0']).attenuate(lam.to_astropy())


class GrainsWD01():
    """
    Weingarter and Draine 2001 dust grain extinction model
    for MW, SMC and LMC

    Accepted models are MW, SMC and LMC
    """
    def __init__(self, params={'model': 'SMCBar'}):
        self.description = 'Weingarter and Draine 2001 dust grain extinction model for MW, SMC and LMC'
        if 'MW' in params['model']:
            params['model'] = 'MWRV31'
        elif 'LMC' in params['model']:
            params['model'] = 'LMCAvg'
        elif 'SMC' in params['model']:
            params['model'] = 'SMCBar'
        else:
            exceptions.InconsistentParameter('Grain model not available')

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
            emodel = WD01(params['model'])
            return emodel.extinguish(x=lam.to_astropy(),
                                     Av=1.086*tau_V)
