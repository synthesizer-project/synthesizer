
import os
import numpy as np
from unyt import Angstrom
from scipy import interpolate
from functools import partial
from dust_attenuation.shapes import N09
from dust_extinction.grain_models import WD01
from . import exceptions
this_dir, this_filename = os.path.split(__file__)

# --- dust curves commonly used in literature

__all__ = ["power_law", "MW_N18", "Calzetti2000", "GrainsWD01"]


class power_law():
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

    def __init__(self, params={'slope': -1.}):
        """
        Initialise the power law slope of the dust curve

        Parameters
        ----------
        slope: float
            power law slope
        """

        self.description = 'simple power law dust curve'
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

        return (lam/5500.)**self.params['slope']

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

        return self.tau_x(lam)/self.tau_x(5500.)

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


class MW_N18():
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

        self.description = 'MW extinction curve from Desika'
        self.d = np.load(f'{this_dir}/data/MW_N18.npz')
        self.tau_lam_V = np.interp(5500.,
                                   self.d.f.mw_df_lam[::-1],
                                   self.d.f.mw_df_chi[::-1])

    def tau(self, lam, interp='cubic'):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        f = interpolate.interp1d(self.d.f.mw_df_lam[::-1],
                                 self.d.f.mw_df_chi[::-1],
                                 kind=interp,
                                 fill_value='extrapolate')

        return f(lam.to('Angstrom').v)/self.tau_lam_V

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

    def __init__(self, params={'slope': 0., 'x0': 0.2175, 'ampl': 0.}):
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

        """
        self.description = 'Calzetti attenuation curve; with option for the slope and UV-bump implemented in Noll et al. 2009'
        self.params = params

    def tau(self, lam):
        """
        Calculate V-band normalised optical depth

        Parameters
        ----------
        lam: float array
            wavelength, expected mwith units
        """

        return N09(Av=1.,
                   ampl=self.params['ampl'],
                   slope=self.params['slope'],
                   x0=self.params['x0'])(lam.to_astropy())

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

    def __init__(self, params={'model': 'SMCBar'}):
        """
        Initialise the dust curve

        Parameters
        ----------
        model: string
            dust grain model to use

        """

        self.description = 'Weingarter and Draine 2001 dust grain extinction model for MW, SMC and LMC'
        self.params = {}
        if 'MW' in params['model']:
            self.params['model'] = 'MWRV31'
        elif 'LMC' in params['model']:
            self.params['model'] = 'LMCAvg'
        elif 'SMC' in params['model']:
            self.params['model'] = 'SMCBar'
        else:
            self.params['model'] = params['model']

        self.emodel = WD01(self.params['model'])

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

        return self.emodel.extinguish(x=lam.to_astropy(),
                                      Av=1.086*tau_V)





from scipy import integrate
from unyt import h, c, kb, um, erg, s, Hz
from unyt import accepts, returns
from unyt.dimensions import length, time, temperature


def planck(nu, T):

    """
    Planck's law 
    """

    return (2.*h*(nu**3)*(c**-2))*(1./(np.exp(h*nu/(kb*T))-1.))



class EmissionBase():

    """
    Dust emission base class for holding common methods.
    """

    def normalise(self):

        """
        Provide normalisation of lnu_ by integrating the function from 8->1000 um
        """ 
        return integrate.quad(self.lnu_, c/(1000*um), c/(8*um), full_output=False, limit = 100)[0]

    # @accepts(lam=length)
    def lnu(self, lam):

        """
        Returns the normalised lnu for the provided wavelength grid

        Parameters
        ----------
        lam: unyt_array
            Wavelength grid

        """

        return (erg/s/Hz)*self.lnu_(c/lam).value/self.normalise()




class Blackbody(EmissionBase):
    
    """
    A class to generate a blackbody emission spectrum.
    """

    @accepts(T=temperature) # check T has dimensions of temperature
    def __init__(self, T):

        """
        A function to generate a simple blackbody spectrum.
        
        Parameters
        ----------
        T: unyt_array
            Temperature

        """

        self.T = T

    # @accepts(nu=1/time)
    def lnu_(self, nu):
        
        """
        Generate unnormalised spectrum for given frequency (nu) grid.
        
        Parameters
        ----------
        nu: unyt_array
            frequency

        Returns
        ----------
        lnu: unyt_array
            spectral luminosity density

        """

        return planck(nu, self.T)

    


class Greybody(EmissionBase):

    """
    A class to generate a greybody emission spectrum.
    """

    @accepts(T=temperature) # check T has dimensions of temperature
    def __init__(self, T, emissivity):

        """
        Initialise class
        
        Parameters
        ----------
        T: unyt_array
            Temperature

        emissivity: float
            Emissivity (dimensionless)

        """

        self.T = T
        self.emissivity = emissivity


    # @accepts(nu=1/time)
    def lnu_(self, nu):
        
        """
        Generate unnormalised spectrum for given frequency (nu) grid.
        
        Parameters
        ----------
        nu: unyt_array
            frequency

        Returns
        ----------
        lnu: unyt_array
            spectral luminosity density

        """

        return nu**self.emissivity * planck(nu, self.T)


class Casey12(EmissionBase):
    
    """
    A class to generate a dust emission spectrum using the Casey (2012) model.
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.425.3094C/abstract
    """

    @accepts(T=temperature) # check T has dimensions of temperature
    def __init__(self, T, emissivity, alpha, N_bb = 1.0, lam_0 = 200.*um):

        """
        Parameters
        ----------
        lam: unyt_array
            wavelength

        T: unyt_array
            Temperature

        emissivity: float
            Emissivity (dimensionless) [good value = 1.6]

        alpha: float
            Power-law slope (dimensionless)  [good value = 2.0]

        N_Bb: float
            Normalisation of the blackbody component [default 1.0]

        lam_0: float
            Wavelength at where the dust optical depth is unity
        """

        self.T = T
        self.emissivity = emissivity
        self.alpha = alpha
        self.N_bb = N_bb 
        self.lam_0 = lam_0

        # calculate the powerlaw turnover wavelength

        b1 = 26.68
        b2 = 6.246
        b3 = 0.0001905
        b4 = 0.00007243
        L = ( (b1+b2*alpha)**-2 + (b3+b4*alpha)*T.to('K').value)**-1

        self.lam_c = (3./4.)*L * um

        # calculate normalisation of the power-law term

        # See Casey+2012, Table 1 for the expression
        # Missing factors of lam_c and c in some places

        self.N_pl = self.N_bb * (1 - np.exp(-(self.lam_0/self.lam_c)**emissivity)) * (c/self.lam_c)**3 / (np.exp(h*c/(self.lam_c*kb*T))-1) 

    # @accepts(nu=1/time)
    def lnu_(self, nu):

        """
        Generate unnormalised spectrum for given frequency (nu) grid.
        
        Parameters
        ----------
        nu: unyt_array
            frequency

        Returns
        ----------
        lnu: unyt_array
            spectral luminosity density

        """
        
        if np.isscalar(nu): nu*=Hz
        
        PL = lambda x: self.N_pl * ((x/self.lam_c)**(self.alpha)) * np.exp(-(x/self.lam_c)**2) # x is wavelength NOT frequency

        BB = lambda x: self.N_bb * (1-np.exp(-(self.lam_0/x)**self.emissivity)) * (c/x)**3 / (np.exp((h*c)/(x*kb*self.T)) - 1.0) # x is wavelength NOT frequency
        
        # NOTE: THE ABOVE DOESN'T WORK WITH DIMENSIONS, I.E. BOTH PARTS ARE NOT DIMENSIONALLY CONSISTENT HENCE BELOW.
        return  PL(c/nu) + BB(c/nu) 
    