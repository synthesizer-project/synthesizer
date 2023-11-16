""" A module for holding blackhole emission models. 
"""
import numpy as np
import matplotlib.pyplot as plt
from unyt import Myr, deg, c, erg, s, Msun, unyt_quantity

from synthesizer import exceptions
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.line import Line, LineCollection
from synthesizer.sed import Sed
from synthesizer.units import Quantity


class BlackholesComponent:
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly.

    """

    # Define the allowed attributes
    # __slots__ = [
    #     "_mass",
    #     "_accretion_rate",
    #     "_bb_temperature",
    #     "_bolometric_luminosity",
    # ]

    # Define class level Quantity attributes
    accretion_rate = Quantity()
    bolometric_luminosity = Quantity()
    bb_temperature = Quantity()
    mass = Quantity()

    def __init__(
        self,
        mass=None,
        accretion_rate=None,
        epsilon=0.1,
        accretion_rate_eddington=None,
        inclination=None,
        spin=None,
        bolometric_luminosity=None,
        metallicity=None,
    ):
        """
        Initialise the BlackholeComponent. Where they're not provided missing 
        quantities are automatically calcualted. Only some quantities are 
        needed for each emission model.

        Args:
            mass (array-like, float)
                The mass of each blackhole.
            accretion_rate (array-like, float)
                The accretion rate of each blackhole.
            epsilon (array-like, float)
                The radiative efficiency of the blackhole.
            accretion_rate_eddington (array-like, float)
                The accretion rate expressed as a fraction of the Eddington
                accretion rate.
            inclination (array-like, float)
                The inclination of the blackhole disc.
            spin (array-like, float)
                The dimensionless spin of the blackhole.
            bolometric_luminosity (array-like, float)
                The bolometric luminosity of the blackhole.
            metallicity (array-like, float)
                The metallicity of the blackhole which is assumed for the line
                emitting regions.
            
             
        """

        # Save the arguments as attributes
        self.mass = mass
        self.accretion_rate = accretion_rate
        self.epsilon = epsilon
        self.accretion_rate_eddington = accretion_rate_eddington
        self.inclination = inclination
        self.spin = spin
        self.bolometric_luminosity = bolometric_luminosity
        self.metallicity = metallicity

        # Define the spectra dictionary to hold the stellar spectra
        self.spectra = {}

        # Define the line dictionary to hold the stellar emission lines
        self.lines = {}

        # If mass, accretion_rate, and epsilon provided calculate the
        # bolometric luminosity.
        if ((self.mass is not None) and (self.accretion_rate is not None) and (self.epsilon is not None)):
            self.calculate_bolometric_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # big bump temperature.
        if ((self.mass is not None) and (self.accretion_rate is not None) and (self.epsilon is not None)):
            self.calculate_bb_temperature()

        # If mass calculate the Eddington luminosity.
        if self.mass is not None:
            self.calculate_eddington_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # Eddington ratio.
        if ((self.mass is not None) and (self.accretion_rate is not None) and (self.epsilon is not None)):
            self.calculate_eddington_ratio()

        # If mass, accretion_rate, and epsilon provided calculate the
        # accretion rate in units of the Eddington accretion rate. This is the
        # bolometric_luminosity / eddington_luminosity.
        if ((self.mass is not None) and (self.accretion_rate is not None) and (self.epsilon is not None)):
            self.calculate_accretion_rate_eddington()

        # I will be hated for this. But left in for now to provide access to
        # both and not break the EmissionModel.
        for singular, plural in [
            ('mass', 'masses'),
            ('accretion_rate', 'accretion_rates'),
            ('metallicity', 'metallicities'),
            ('spin', 'spins'),
            ('inclination', 'inclinations'),
            ('epsilon', 'epsilons'),
            ('bb_temperature', 'bb_temperatures'),
            ('bolometric_luminosity', 'bolometric_luminosities'),
            ('accretion_rate_eddington', 'accretion_rates_eddington'),
            ('epsilon', 'epsilons'),
            ('eddington_ratio', 'eddington_ratios')]:    

            setattr(self, plural, getattr(self, singular))

    def calculate_bolometric_luminosity(self):
        """
        Calculate the black hole bolometric luminosity. This is by itself
        useful but also used for some emission models.
        
        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        self.bolometric_luminosity = self.epsilon * self.accretion_rate * c**2

        return self.bolometric_luminosity

    def calculate_eddington_luminosity(self):
        """
        Calculate the eddington luminosity of the black hole.
        
        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        # Note: the factor 1.257E38 comes from:
        # 4*pi*G*mp*c*Msun/sigma_thompson
        self.eddington_luminosity = (1.257E38 * erg / s / Msun) * self.mass 
    
        return self.eddington_luminosity
    
    def calculate_eddington_ratio(self):
        """
        Calculate the eddington ratio of the black hole.
        
        Returns
            unyt_array
                The black hole eddington ratio
        """

        self.eddington_ratio = (self.bolometric_luminosity
                                / self.eddington_luminosity)
    
        return self.eddington_ratio
    
    def calculate_bb_temperature(self):
        """
        Calculate the black hole big bump temperature. This is used for the
        cloudy disc model.
        
        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        # Calculate the big bump temperature
        self.bb_temperature = (
            2.24e9 * self._accretion_rate ** (1 / 4) * self._mass**-0.5
        )

        return self.bb_temperature

    def calculate_accretion_rate_eddington(self):
        """
        Calculate the black hole accretion in units of the Eddington rate.
        
        Returns
            unyt_array
                The black hole accretion rate in units of the Eddington rate.
        """

        self.accretion_rate_eddington = (self.bolometric_luminosity /
                                         self.eddington_luminosity)

        return self.accretion_rate_eddington

    def get_spectra(self, emission_model):
        """
        Generate blackhole spectra for a given emission_model.
        
        Args
            synthesizer.blackholes.BlackHoleEmissionModel
                A synthesizer BlackHoleEmissionModel instance.
        
        """

        # get the parameters that this particular emission model requires
        # TODO: this is horrible and will break when __slots__ exists.

        parameters = {}
        for parameter in emission_model.parameters:
            if parameter in self.__dict:
                parameters[parameter] = getattr(self, parameter)

        print(parameters)
       


        # self.spectra = emission_model(emission_model_parameters)