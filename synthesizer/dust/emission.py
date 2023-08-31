import numpy as np
from scipy import integrate
from unyt import h, c, kb, um, erg, s, Hz
from unyt import accepts
from unyt.dimensions import temperature

from synthesizer.utils import planck


class EmissionBase():

    """
    Dust emission base class for holding common methods.
    """

    def normalise(self):

        """
        Provide normalisation of lnu_ by integrating the function from 8->1000
        um
        """ 
        return integrate.quad(self.lnu_, c / (1000 * um), c / (8 * um),
                              full_output=False, limit=100)[0]

    # @accepts(lam=length)
    def lnu(self, lam):

        """
        Returns the normalised lnu for the provided wavelength grid

        Parameters
        ----------
        lam: unyt_array
            Wavelength grid

        """

        return (erg / s / Hz) * self.lnu_(c / lam).value / self.normalise()


class Blackbody(EmissionBase):
    """
    A class to generate a blackbody emission spectrum.
    """

    @accepts(T=temperature)  # check T has dimensions of temperature
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

    @accepts(T=temperature)  # check T has dimensions of temperature
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

    @accepts(T=temperature)  # check T has dimensions of temperature
    def __init__(self, T, emissivity, alpha, N_bb=1.0, lam_0=200.*um):

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
        L = ((b1+b2*alpha)**-2 + (b3+b4*alpha)*T.to('K').value)**-1

        self.lam_c = (3./4.)*L * um

        # calculate normalisation of the power-law term

        # See Casey+2012, Table 1 for the expression
        # Missing factors of lam_c and c in some places

        self.N_pl = self.N_bb \
            * (1 - np.exp(-(self.lam_0/self.lam_c)**emissivity)) \
            * (c/self.lam_c)**3 / (np.exp(h*c/(self.lam_c*kb*T))-1)

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
        
        # Essential, when using scipy.integrate, since
        # the integration limits are passed unitless 
        if np.isscalar(nu):
            nu *= Hz
        
        def PL(lam):
            """
            Calcualate the power-law component.
            """
            return self.N_pl * ((lam/self.lam_c)**(self.alpha)) \
                * np.exp(-(lam/self.lam_c)**2) 

        def BB(lam):
            """
            Calcualate the blackbody component.
            """
            return self.N_bb * (1-np.exp(-(self.lam_0/lam)**self.emissivity)) \
                * (c/lam)**3 / (np.exp((h*c)/(lam*kb*self.T)) - 1.0) 
        
        return PL(c/nu) + BB(c/nu)
