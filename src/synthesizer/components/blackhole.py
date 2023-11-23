""" A module for holding blackhole emission models. 
"""
import numpy as np
import matplotlib.pyplot as plt
from unyt import Myr, deg, c, erg, s, Msun, unyt_quantity
import inflect
from copy import deepcopy
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
    # TODO: reinstate this.
    # Define the allowed attributes
    # __slots__ = [
    #     "_mass",
    #     "_accretion_rate",
    #     "_bb_temperature",
    #     "_bolometric_luminosity",
    # ]

    # Define class level Quantity attributes
    accretion_rate = Quantity()
    inclination = Quantity()
    bolometric_luminosity = Quantity()
    eddington_luminosity = Quantity()
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

        # Initialise spectra
        self.spectra = None

        # Save the arguments as attributes
        self.mass = mass
        self.accretion_rate = accretion_rate
        self.epsilon = epsilon
        self.accretion_rate_eddington = accretion_rate_eddington
        self.inclination = inclination
        self.spin = spin
        self.bolometric_luminosity = bolometric_luminosity
        self.metallicity = metallicity

        # Check to make sure that both accretion rate and bolometric luminosity
        # haven't been provided because that could be confusing.
        if ((self.accretion_rate is not None)
                and (self.bolometric_luminosity is not None)):
            raise exceptions.InconsistentArguments(
                """Both accretion rate and bolometric luminosity provided but
                that is confusing. Provide one or the other!""")

        if ((self.accretion_rate_eddington is not None)
                and (self.bolometric_luminosity is not None)):
            raise exceptions.InconsistentArguments(
                """Both accretion rate (in terms of Eddington) and bolometric
                luminosity provided but that is confusing. Provide one or
                the other!""")

        # If mass, accretion_rate, and epsilon provided calculate the
        # bolometric luminosity.
        if ((self.mass is not None) and (self.accretion_rate is not None)
                and (self.epsilon is not None)):
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

        # If inclination, calculate the cosine of the inclination, required by 
        # some models (e.g. AGNSED).
        if (self.inclination is not None):
            self.consine_inclination = np.cos(
                self.inclination.to('radian').value)

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
        self.eddington_luminosity = 1.257E38 * self._mass 
    
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

        self.accretion_rate_eddington = (self._bolometric_luminosity /
                                         self._eddington_luminosity)

        return self.accretion_rate_eddington

    def get_spectra(self, emission_model, spectra_ids=None, verbose=True):
        """
        Generate blackhole spectra for a given emission_model.
        
        Args
            synthesizer.blackholes.BlackHoleEmissionModel
                A synthesizer BlackHoleEmissionModel instance.
        
        """

        # Get the parameters that this particular emission model requires
        emission_model_parameters = {}
        for parameter in emission_model.variable_parameters:
            attr = getattr(self, parameter, None)
            priv_attr = getattr(self, "_" + parameter, None)
            if attr is not None:
                emission_model_parameters[parameter] = attr
            elif priv_attr is not None:
                emission_model_parameters[parameter] = priv_attr
    
        # Loop over the blackholes associated to the model
        for i, values in enumerate(zip(*[x for x in emission_model_parameters.values()])):

            # Create a dictionary of the parameters to be passed to the
            # emission model.
            parameters = dict(zip(emission_model_parameters.keys(), values))

            # Get the parameters and spectra 
            parameter_dict, spectra = emission_model.get_spectra(
                spectra_ids=spectra_ids, **parameters
                )
            
            if self.spectra is None:
                # Necessary so not a pointer
                self.spectra = deepcopy(spectra)
            else:       
                for spectra_id, spectra_ in spectra.items():
                    self.spectra[spectra_id] = self.spectra[spectra_id].concat(
                        spectra_)
                    
        return self.spectra
        