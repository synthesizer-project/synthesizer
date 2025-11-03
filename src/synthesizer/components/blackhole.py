"""A module for holding blackhole emission models.

The class defined here should never be instantiated directly, there are only
ever instantiated by the parametric/particle child classes.
BlackholesComponent is a child class of Component.
"""

import numpy as np
from unyt import Msun, c, cm, deg, erg, km, s, yr

from synthesizer import exceptions
from synthesizer.components.component import Component
from synthesizer.units import Quantity, accepts
from synthesizer.utils import (
    TableFormatter,
    array_to_scalar,
    scalar_to_array,
)


class BlackholesComponent(Component):
    """The parent class for black hole components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly, instead it provides the common
    functionality and attributes used by the child parametric and particle
    BlackHole/s classes.

    Attributes:
        spectra (dict, Sed)
            A dictionary containing black hole spectra.
        mass (np.ndarray of float):
            The mass of each blackhole.
        accretion_rate (np.ndarray of float):
            The accretion rate of each blackhole.
        epsilon (np.ndarray of float):
            The radiative efficiency of the blackhole.
        accretion_rate_eddington (np.ndarray of float):
            The accretion rate expressed as a fraction of the Eddington
            accretion rate.
        inclination (np.ndarray of float):
            The inclination of the blackhole disc.
        spin (np.ndarray of float):
            The dimensionless spin of the blackhole.
        bolometric_luminosity (np.ndarray of float):
            The bolometric luminosity of the blackhole.
        metallicity (np.ndarray of float):
            The metallicity of the blackhole which is assumed for the line
            emitting regions.

    Attributes (For EmissionModels):
        ionisation_parameter_blr (np.ndarray of float):
            The ionisation parameter of the broad line region.
        hydrogen_density_blr (np.ndarray of float):
            The hydrogen density of the broad line region.
        covering_fraction_blr (np.ndarray of float):
            The covering fraction of the broad line region (effectively
            the escape fraction).
        velocity_dispersion_blr (np.ndarray of float):
            The velocity dispersion of the broad line region.
        ionisation_parameter_nlr (np.ndarray of float):
            The ionisation parameter of the narrow line region.
        hydrogen_density_nlr (np.ndarray of float):
            The hydrogen density of the narrow line region.
        covering_fraction_nlr (np.ndarray of float):
            The covering fraction of the narrow line region (effectively
            the escape fraction).
        velocity_dispersion_nlr (np.ndarray of float):
            The velocity dispersion of the narrow line region.
        theta_torus (np.ndarray of float):
            The angle of the torus.
        torus_fraction (np.ndarray of float):
            The fraction of the torus angle to 90 degrees.
        disc_transmission (np.ndarray of str):
            The disc transmission scenario used by UnifiedAGN. Can either be
            "none", "blr", or "nlr".
        escape_transmission_fraction (np.ndarray of float):
            The fraction of the disc emission that escapes. Either 0.0 or 1.0
            depending on the scenario.
        nlr_transmission_fraction (np.ndarray of float):
            The fraction of the disc emission that is transmitted through the
            NLR. Either 0.0 or 1.0 depending on the scenario.
        blr_transmission_fraction (np.ndarray of float):
            The fraction of the disc emission that is transmitted through the
            NLR. Either 0.0 or 1.0 depending on the scenario.
    """

    # Define class level Quantity attributes
    accretion_rate = Quantity("mass_rate")
    inclination = Quantity("angle")
    bolometric_luminosity = Quantity("luminosity")
    eddington_luminosity = Quantity("luminosity")
    bb_temperature = Quantity("temperature")
    mass = Quantity("mass")

    @accepts(
        mass=Msun.in_base("galactic"),
        accretion_rate=Msun.in_base("galactic") / yr,
        inclination=deg,
        bolometric_luminosity=erg / s,
        hydrogen_density_blr=cm**-3,
        hydrogen_density_nlr=cm**-3,
        velocity_dispersion_blr=km / s,
        velocity_dispersion_nlr=km / s,
        theta_torus=deg,
    )
    def __init__(
        self,
        fesc,
        mass=None,
        accretion_rate=None,
        epsilon=0.1,
        accretion_rate_eddington=None,
        inclination=0.0 * deg,
        spin=None,
        bolometric_luminosity=None,
        metallicity=None,
        ionisation_parameter_blr=0.1,
        hydrogen_density_blr=1e9 / cm**3,
        covering_fraction_blr=0.1,
        velocity_dispersion_blr=2000 * km / s,
        ionisation_parameter_nlr=0.01,
        hydrogen_density_nlr=1e4 / cm**3,
        covering_fraction_nlr=0.1,
        velocity_dispersion_nlr=500 * km / s,
        theta_torus=10 * deg,
        **kwargs,
    ):
        """Initialise the BlackholeComponent.

        Where they're not provided missing quantities are automatically
        calculated. Not all parameters need to be set for every emission model.

        Args:
            fesc (float):
                The escape fraction of the blackhole.
            mass (np.ndarray of float):
                The mass of each blackhole.
            accretion_rate (np.ndarray of float):
                The accretion rate of each blackhole.
            epsilon (np.ndarray of float):
                The radiative efficiency of the blackhole.
            accretion_rate_eddington (np.ndarray of float):
                The accretion rate expressed as a fraction of the Eddington
                accretion rate.
            inclination (np.ndarray of float):
                The inclination of the blackhole disc.
            spin (np.ndarray of float):
                The dimensionless spin of the blackhole.
            bolometric_luminosity (np.ndarray of float):
                The bolometric luminosity of the blackhole.
            metallicity (np.ndarray of float):
                The metallicity of the blackhole which is assumed for the line
                emitting regions.
            ionisation_parameter_blr (np.ndarray of float):
                The ionisation parameter of the broad line region.
            hydrogen_density_blr (np.ndarray of float):
                The hydrogen density of the broad line region.
            covering_fraction_blr (np.ndarray of float):
                The covering fraction of the broad line region (effectively
                the escape fraction).
            velocity_dispersion_blr (np.ndarray of float):
                The velocity dispersion of the broad line region.
            ionisation_parameter_nlr (np.ndarray of float):
                The ionisation parameter of the narrow line region.
            hydrogen_density_nlr (np.ndarray of float):
                The hydrogen density of the narrow line region.
            covering_fraction_nlr (np.ndarray of float):
                The covering fraction of the narrow line region (effectively
                the escape fraction).
            velocity_dispersion_nlr (np.ndarray of float):
                The velocity dispersion of the narrow line region.
            theta_torus (np.ndarray of float):
                The angle of the torus.
            **kwargs (dict):
                Any other parameter for the emission models can be provided as
                kwargs.
        """
        # Initialise the parent class
        Component.__init__(self, "BlackHoles", fesc, **kwargs)

        # Save the black hole properties
        self.mass = mass
        self.accretion_rate = accretion_rate
        self.epsilon = epsilon
        self.accretion_rate_eddington = accretion_rate_eddington
        self.spin = spin
        self.bolometric_luminosity = bolometric_luminosity
        self.metallicity = metallicity

        # Below we attach all the possible attributes that could be needed by
        # the emission models.

        # Set BLR attributes
        self.ionisation_parameter_blr = ionisation_parameter_blr
        self.hydrogen_density_blr = hydrogen_density_blr
        self.covering_fraction_blr = covering_fraction_blr
        self.velocity_dispersion_blr = velocity_dispersion_blr

        # Set NLR attributes
        self.ionisation_parameter_nlr = ionisation_parameter_nlr
        self.hydrogen_density_nlr = hydrogen_density_nlr
        self.covering_fraction_nlr = covering_fraction_nlr
        self.velocity_dispersion_nlr = velocity_dispersion_nlr

        # If a covering_fraction_blr is set then randomly allocate a scenario
        # for the disc transmission. This can either be that emission entirely
        #  escapes (none), or is transmitted through the BLR (blr), or NLR
        # (nlr). These are allocated based on the relative covering fractions
        # of the BLR and NLR. These are used by the UnifiedAGN emission model.
        if self.covering_fraction_blr is not None:
            # Define transmission scenario choices
            transmission_scenario_choices = ["blr", "nlr", "none"]

            # Convert the blr_covering_fraction and nlr_covering_fraction to
            # arrays to allow us to use the same logic for both parametric and
            # particle blackholes.
            blr_covering_fraction = scalar_to_array(covering_fraction_blr)
            nlr_covering_fraction = scalar_to_array(covering_fraction_nlr)
            # Define number of blackholes.
            N = len(blr_covering_fraction)

            # Loop over blackholes and decide whether the disc emission
            # escapes (none), or is transmitted through the BLR (blr) or NLR
            # (nlr).
            disc_transmission_ = np.empty(N, dtype="U10")
            for i, (
                blr_covering_fraction_,
                nlr_covering_fraction_,
            ) in enumerate(zip(blr_covering_fraction, nlr_covering_fraction)):
                # Define the probiliaities for each option based on the
                # covering fractions.
                probabilities = [
                    blr_covering_fraction_,
                    nlr_covering_fraction_,
                    1.0 - blr_covering_fraction_ - nlr_covering_fraction_,
                ]

                # Randomly choose the scenario using these probabilities.
                disc_transmission_[i] = np.random.choice(
                    transmission_scenario_choices, p=probabilities
                )

            # Initialise transmission fraction arrays. These determine the
            # fraction of the disc emission that entirely escapes or is
            # transmitted through the BLR and NLR. Since, in this context,
            # the emission is only propagated through one component then one
            # of these must be unity with the other two zero. It is however
            # possible to use the average, but this is more clearly
            # implemented at the emission model level.
            escape_transmission_fraction = np.zeros(N)
            nlr_transmission_fraction = np.zeros(N)
            blr_transmission_fraction = np.zeros(N)

            # For each corresponding scenario set the transmission fraction to
            # unity.
            escape_transmission_fraction[disc_transmission_ == "none"] = 1.0
            nlr_transmission_fraction[disc_transmission_ == "nlr"] = 1.0
            blr_transmission_fraction[disc_transmission_ == "blr"] = 1.0

            # convert to scalars if only one value
            self.escape_transmission_fraction = array_to_scalar(
                escape_transmission_fraction
            )
            self.nlr_transmission_fraction = array_to_scalar(
                nlr_transmission_fraction
            )
            self.blr_transmission_fraction = array_to_scalar(
                blr_transmission_fraction
            )
            self.disc_transmission = array_to_scalar(disc_transmission_)

        # The inclination of the black hole disc
        self.inclination = (
            inclination if inclination is not None else 0.0 * deg
        )

        # The angle of the torus
        self.theta_torus = theta_torus
        self.torus_fraction = (self.theta_torus / (90 * deg)).value
        self._torus_edgeon_cond = self.inclination + self.theta_torus

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

        # Ensure that both accretion_rate and accretion_rate_eddington are not
        # provided, this is also confusing.
        if (self.accretion_rate is not None) and (
            self.accretion_rate_eddington is not None
        ):
            raise exceptions.InconsistentArguments(
                """Both accretion rate and accretion rate in terms of
                Eddington provided but that is confusing. Provide one or the
                other!"""
            )

        # If mass calculate the Eddington luminosity.
        if self.mass is not None:
            self.calculate_eddington_luminosity()

        # If accretion_rate_eddington provided calculate the accretion rate.
        if (
            self.accretion_rate_eddington is not None
            and self.eddington_luminosity is not None
        ):
            self.calculate_accretion_rate()

        # If mass, accretion_rate, and epsilon provided calculate the
        # bolometric luminosity.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_bolometric_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # big bump temperature.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_bb_temperature()

        # If mass, accretion_rate, and epsilon provided calculate the
        # Eddington ratio.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_eddington_ratio()

        # If mass, accretion_rate, and epsilon provided calculate the
        # accretion rate in units of the Eddington accretion rate. This is the
        # bolometric_luminosity / eddington_luminosity.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_accretion_rate_eddington()

        # If inclination, calculate the cosine of the inclination, required by
        # some models (e.g. AGNSED).
        if self.inclination is not None:
            self.cosine_inclination = np.cos(
                self.inclination.to("radian").value
            )

    def calculate_accretion_rate(self):
        """Calculate the black hole accretion rate from the eddington ratio.

        This is used when accretion rate is provided in terms of the Eddington
        ratio and not the explicit accretion rate.


        Returns:
            unyt_array:
                The black hole accretion rate
        """
        self.accretion_rate = (
            self.accretion_rate_eddington
            * self.eddington_luminosity
            / (self.epsilon * c**2)
        )

        return self.accretion_rate

    def calculate_bolometric_luminosity(self):
        """Calculate the black hole bolometric luminosity.

        Returns:
            unyt_array:
                The black hole bolometric luminosity
        """
        self.bolometric_luminosity = self.epsilon * self.accretion_rate * c**2

        return self.bolometric_luminosity

    def calculate_eddington_luminosity(self):
        """Calculate the eddington luminosity of the black hole.

        Returns:
            unyt_array
                The black hole eddington luminosity
        """
        # Note: the factor 1.257E38 comes from:
        # 4*pi*G*mp*c*Msun/sigma_thompson
        self.eddington_luminosity = 1.257e38 * self._mass

        return self.eddington_luminosity

    def calculate_eddington_ratio(self):
        """Calculate the eddington ratio of the black hole.

        Returns:
            unyt_array:
                The black hole eddington ratio
        """
        self.eddington_ratio = (
            self._bolometric_luminosity / self._eddington_luminosity
        )

        return self.eddington_ratio

    def calculate_bb_temperature(self):
        """Calculate the black hole big bump temperature.

        This is used in the cloudy disc model.

        Returns:
            unyt_array:
                The black hole big bump temperature
        """
        # Calculate the big bump temperature
        self.bb_temperature = (
            2.24e9 * self._accretion_rate ** (1 / 4) * self._mass**-0.5
        )

        return self.bb_temperature

    def calculate_accretion_rate_eddington(self):
        """Calculate the black hole accretion in units of the Eddington rate.

        Returns:
            unyt_array
                The black hole accretion rate in units of the Eddington rate.
        """
        self.accretion_rate_eddington = (
            self._bolometric_luminosity / self._eddington_luminosity
        )

        return self.accretion_rate_eddington

    def __str__(self):
        """Return a string representation of the particle object.

        Returns:
            table (str):
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Black Holes")
