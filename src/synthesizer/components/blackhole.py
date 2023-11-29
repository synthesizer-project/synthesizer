""" A module for holding blackhole emission models.

The class defined here should never be instantiated directly, there are only
ever instantiated by the parametric/particle child classes.
"""
from copy import deepcopy
import numpy as np
from unyt import c

from synthesizer import exceptions
from synthesizer.units import Quantity


class BlackholesComponent:
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly, instead it provides the common
    functionality and attributes used by the child parametric and particle
    BlackHole/s classes.

    Attributes:
        spectra (dict, Sed)
            A dictionary containing black hole spectra.
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

        # Check to make sure that both accretion rate and bolometric luminosity
        # haven't been provided because that could be confusing.
        if (self.accretion_rate is not None) and (
            self.bolometric_luminosity is not None
        ):
            raise exceptions.InconsistentArguments(
                """Both accretion rate and bolometric luminosity provided but
                that is confusing. Provide one or the other!"""
            )

        if (self.accretion_rate_eddington is not None) and (
            self.bolometric_luminosity is not None
        ):
            raise exceptions.InconsistentArguments(
                """Both accretion rate (in terms of Eddington) and bolometric
                luminosity provided but that is confusing. Provide one or
                the other!"""
            )

        # If mass, accretion_rate, and epsilon provided calculate the
        # bolometric luminosity.
        if (
            (self.mass is not None)
            and (self.accretion_rate is not None)
            and (self.epsilon is not None)
        ):
            self.calculate_bolometric_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # big bump temperature.
        if (
            (self.mass is not None)
            and (self.accretion_rate is not None)
            and (self.epsilon is not None)
        ):
            self.calculate_bb_temperature()

        # If mass calculate the Eddington luminosity.
        if self.mass is not None:
            self.calculate_eddington_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # Eddington ratio.
        if (
            (self.mass is not None)
            and (self.accretion_rate is not None)
            and (self.epsilon is not None)
        ):
            self.calculate_eddington_ratio()

        # If mass, accretion_rate, and epsilon provided calculate the
        # accretion rate in units of the Eddington accretion rate. This is the
        # bolometric_luminosity / eddington_luminosity.
        if (
            (self.mass is not None)
            and (self.accretion_rate is not None)
            and (self.epsilon is not None)
        ):
            self.calculate_accretion_rate_eddington()

        # If inclination, calculate the cosine of the inclination, required by
        # some models (e.g. AGNSED).
        if self.inclination is not None:
            self.cosine_inclination = np.cos(self.inclination.to("radian").value)

    def __str__(self):
        """Function to print a basic summary of the BlackHoles object.

        Returns a string containing the total mass formed and lists of the
        available SEDs, lines, and images.

        Returns
            str
                Summary string containing the total mass formed and lists of the
                available SEDs, lines, and images.
        """

        # Define the width to print within
        width = 80
        pstr = ""
        pstr += "-" * width + "\n"
        pstr += "SUMMARY OF BLACKHOLE".center(width + 4) + "\n"
        # pstr += get_centred_art(Art.blackhole, width) + "\n"

        pstr += f"Number of blackholes: {self.mass.size} \n"

        for attribute_id in [
            "mass",
            "accretion_rate",
            "accretion_rate_eddington",
            "bolometric_luminosity",
            "eddington_ratio",
            "bb_temperature",
            "eddington_luminosity",
            "spin",
            "epsilon",
            "inclination",
            "cosine_inclination",
        ]:
            attr = getattr(self, attribute_id, None)
            if attr is not None:
                attr = np.round(attr, 3)
                pstr += f"{attribute_id}: {attr} \n"

        return pstr

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
        self.eddington_luminosity = 1.257e38 * self._mass

        return self.eddington_luminosity

    def calculate_eddington_ratio(self):
        """
        Calculate the eddington ratio of the black hole.

        Returns
            unyt_array
                The black hole eddington ratio
        """

        self.eddington_ratio = self.bolometric_luminosity / self.eddington_luminosity

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

        self.accretion_rate_eddington = (
            self._bolometric_luminosity / self._eddington_luminosity
        )

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

        if self.mass.size > 1:
            # Loop over the blackholes associated to the model
            for i, values in enumerate(
                zip(*[x for x in emission_model_parameters.values()])
            ):
                # Create a dictionary of the parameters to be passed to the
                # emission model.
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
                            spectra_
                        )
        else:
            parameter_dict, self.spectra = emission_model.get_spectra(
                spectra_ids=spectra_ids, **emission_model_parameters
            )

        return self.spectra
