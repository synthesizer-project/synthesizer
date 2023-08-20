
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
from unyt import h, c, kb, um 


def planck(nu, T):

    """
    Planck's law 
    """

    return (2.*h*(nu**3)*(c**-2))*(1./(np.exp(h*nu/(kb*T))-1.))

def normalise(lnu):
        # normalise by integrating the function from 8->1000 um
        return integrate.quad(lnu, c/(1000*um), c/(8*um), full_output=False, limit = 100)[0]



class EmissionSpectrum():

    """
    A class for holding dust emission models.
    """

    def blackbody(lam, T):

        """
        A function to generate a simple blackbody spectrum.
        
        Parameters
        ----------
        lam: unyt_array
            wavelength

        T: unyt_array
            Temperature

        Returns 
        ----------
        spectral flux density: unyt_array

        """

        # need to add exception for lam and T not having units

        # define frequency array
        nu = c/lam
        
        # define lnu
        lnu = lambda x: planck(x, T)

        # normalise integral to unity. 
        # It might be better seperate the function and normalise it using the integral of the function
        # lnu /= -np.trapz(nu[::-1], lnu[::-1])

        return lnu(nu)/normalise(lnu)


    def greybody(lam, T, emissivity):

        """
        A function to generate a greybody spectrum.
        
        Parameters
        ----------
        lam: unyt_array
            wavelength

        T: unyt_array
            Temperature

        emissivity: float
            Emissivity (dimensionless)

        Returns 
        ----------
        spectral flux density: unyt_array

        """


        # need to add exception for lam and T not having units

        kappa = lambda x: x**emissivity
        lnu = lambda x: kappa(x) * planck(x, T)
                
        return lnu(c/lam)/normalise(lnu)


    def casey12(lam, T, emissivity, alpha, N_bb = 1.0, lam_0 = 200.*um):

        """
        A function to generate an IR spectrum using the Casey (2012) model.
        
        Parameters
        ----------
        lam: unyt_array
            wavelength

        T: unyt_array
            Temperature

        emissivity: float
            Emissivity (dimensionless)

        alpha: float

        N_Bb: float

        lam_0: float

        Returns 
        ----------
        spectral flux density: unyt_array

        """


        # need to add exception for lam and T not having units

        # define frequency array
        nu = c/lam


        b1 = 26.68
        b2 = 6.246
        b3 = 0.0001905
        b4 = 0.00007243
        L = ( (b1+b2*alpha)**-2 + (b3+b4*alpha)*T.to('K').value)**-1
        lam_c = (3./4.)*L * um
        
        A1 = (c/lam_c)**(emissivity+3.)
        A2 = np.exp((h*c)/(lam_c*kb*T)) - 1.0
        B = lam_c**alpha
        
        N_pl = A1/(A2*B)
        
        PL = lambda x: N_pl*(x**alpha)*np.exp(-(x/lam_c)**2) # x is wavelength NOT frequency

        BB = lambda x: N_bb * (c/x)**(emissivity+3) / (np.exp((h*c)/(x*kb*T)) - 1.0) # x is wavelength NOT frequency
        
        lnu = lambda x: BB(c/x) + PL(c/x) # x is frequency

        return lnu(c/lam)/normalise(lnu)
        