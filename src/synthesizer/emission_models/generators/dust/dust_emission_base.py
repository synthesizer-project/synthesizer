"""A submodule containing base classes for dust emission generators.

This includes generic base classes for energy-balance and scaled dust emission
emissions, as well as a utility function for calculating CMB heating effects.

These base classes are not intended to be used directly, but rather to be
inherited by specific dust emission models that implement the wrapper methods
that generate the actual dust emission spectra and then scale them
appropriately.
"""

from typing import Tuple, Union

from unyt import K, unyt_quantity

from synthesizer import exceptions
from synthesizer.components.component import Component
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.generator import Generator
from synthesizer.units import accepts


@accepts(temperature=K)
def get_cmb_heating_factor(
    temperature: unyt_quantity,
    emissivity: float,
    redshift: float,
) -> tuple[float, unyt_quantity]:
    """Return the factor by which the CMB boosts the infrared luminosity.

    This will also update the temperature of the dust at redshift z.

    (See implementation in da Cunha+2013)

    Args:
        temperature (unyt_array):
            The temperature of the dust.
        emissivity (float):
            The emissivity index in the FIR (no unit)
        redshift (float):
            The redshift of the galaxy
    """
    # Temperature of CMB at redshift=0
    _T_cmb_0 = 2.73 * K
    _T_cmb_redshift = _T_cmb_0 * (1 + redshift)
    _exp_factor = 4.0 + emissivity

    # Updated temperature accounting for CMB heating
    _temperature = (
        temperature**_exp_factor
        + _T_cmb_redshift**_exp_factor
        - _T_cmb_0**_exp_factor
    ) ** (1 / _exp_factor)

    # Factor by which the CMB boosts the infrared luminosity
    cmb_factor: float = (_temperature / temperature) ** (4 + emissivity)

    return cmb_factor, _temperature


class DustEmission(Generator):
    """Dust emission base class.

    This is a base class for dust emission models and defines common
    methods and attributes.

    Dust emission comes in two main flavours: energy-balance and scaled
    dust emission. The former of these requires an intrinsic and attenuated
    emission to derive the energy absorbed by dust, while the latter
    requires a scaler emission to scale the dust emission by. These can also
    be used in isolation without defining the intrinsic, attenuated or scaler
    emissions, in which case normalised emissions are returned.
    """

    @accepts(temperature=K)
    def __init__(
        self,
        intrinsic: EmissionModel = None,
        attenuated: EmissionModel = None,
        scaler: EmissionModel = None,
        do_cmb_heating: bool = False,
        required_params: tuple[str, ...] = (),
    ) -> None:
        """Initialise the base class for dust emission models.

        Args:
            intrinsic (EmissionModel):
                The name of the intrinsic emission defining the unattenuated
                emission for energy balance.
            attenuated (EmissionModel):
                The name of the attenuated emission defining the attenuated
                emission for energy balance.
            scaler (EmissionModel):
                The name of the emission to use to scale the dust emission by.
            do_cmb_heating (bool):
                Are we applying CMB heating?
            required_params (tuple[str, ...]):
                Any extra required parameters needed by child classes.
        """
        # Construct the required emissions tuple
        required_emissions = ()
        if scaler is not None:
            required_emissions = (scaler,)
        if intrinsic is not None and attenuated is not None:
            required_emissions = (intrinsic, attenuated)

        # Call the parent init
        Generator.__init__(
            self,
            required_params=required_params,
            required_emissions=required_emissions,
        )

        # Define flags for the type of dust emission
        self.is_energy_balance = (
            intrinsic is not None and attenuated is not None
        )
        self.is_scaled = scaler is not None

        # Attach friendly pointers to the various scalers
        self._intrinsic = intrinsic
        self._attenuated = attenuated
        self._scaler = scaler

        # Ensure only one type of dust emission is being used
        if self.is_energy_balance and self.is_scaled:
            raise exceptions.InconsistentArguments(
                "Cannot use both energy-balance and scaled dust "
                "emission simultaneously state either intrinsic and "
                "attenuated models or a scaler model."
            )

        # Are we doing CMB heating?
        self.do_cmb_heating = do_cmb_heating

        # Store the last computed temperature for convenience
        self.last_effective_temperature = None

        # Store the last used cmb factor for convenience
        self.last_cmb_factor = None

    @property
    def temperature_z(self):
        """The last processed effective temperature."""
        return self.last_effective_temperature

    @property
    def cmb_factor(self):
        """The last processed cmb heating factor."""
        return self.last_cmb_factor

    def set_energy_balance(self, intrinsic, attenuated) -> None:
        """Set the dust emission to be energy-balance.

        Args:
            intrinsic (EmissionModel):
                The name of the intrinsic emission defining the unattenuated
                emission for energy balance.
            attenuated (EmissionModel):
                The name of the attenuated emission defining the attenuated
                emission for energy balance.
        """
        self.is_energy_balance = True
        self.is_scaled = False
        self.required_emissions = (intrinsic, attenuated)
        self._intrinsic = intrinsic
        self._attenuated = attenuated
        self._scaler = None

    def set_scaler(self, scaler) -> None:
        """Set the dust emission to be scaled.

        Args:
            scaler (EmissionModel):
                The name of the emission to use to scale the dust emission by.
        """
        self.is_energy_balance = False
        self.is_scaled = True
        self.required_emissions = (scaler,)
        self._intrinsic = None
        self._attenuated = None
        self._scaler = scaler

    def _get_energy_balance_luminosity(
        self,
        emissions: dict,
    ) -> unyt_quantity:
        """Calculate the energy absorbed by dust.

        Args:
            emissions (dict):
                Dictionary containing all emissions generated so far.

        Returns:
            unyt_quantity:
                The bolometric luminosity absorbed by dust.
        """
        # For ease, unpack the intrinsic and attenuated emissions
        # Handle both string labels and EmissionModel objects
        intrinsic_key = (
            self._intrinsic.label
            if hasattr(self._intrinsic, "label")
            else self._intrinsic
        )
        attenuated_key = (
            self._attenuated.label
            if hasattr(self._attenuated, "label")
            else self._attenuated
        )

        intrinsic = emissions[intrinsic_key]
        attenuated = emissions[attenuated_key]

        # Calculate the bolometric luminosity absorbed by dust
        ldust = (
            intrinsic.bolometric_luminosity - attenuated.bolometric_luminosity
        )

        return ldust

    def _get_scaling_luminosity(
        self,
        emissions: dict,
    ) -> unyt_quantity:
        """Extract the bolometric luminosity to scale the dust emission by.

        Args:
            emissions (dict):
                Dictionary containing all emissions generated so far.

        Returns:
            unyt_quantity:
                The bolometric luminosity to scale the dust emission by.
        """
        # For ease, unpack the scaler emission
        # Handle both string labels and EmissionModel objects
        scaler_key = (
            self._scaler.label
            if hasattr(self._scaler, "label")
            else self._scaler
        )
        scaler = emissions[scaler_key]

        # Get the bolometric luminosity to scale by
        lscale = scaler.bolometric_luminosity

        return lscale

    def get_scaling(
        self,
        emitter: Union[Component],
        model: EmissionModel,
        emissions: dict,
    ) -> unyt_quantity:
        """Get the bolometric luminosity to scale the dust emission by.

        Args:
            emitter (Stars/Gas/BlackHole):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.
            emissions (dict):
                Dictionary containing all emissions generated so far.

        Returns:
            unyt_quantity:
                The bolometric luminosity to scale the dust emission by.
        """
        # Get the right scaling luminosity
        if self.is_energy_balance:
            return self._get_energy_balance_luminosity(
                emissions=emissions,
            )
        elif self.is_scaled:
            return self._get_scaling_luminosity(
                emissions=emissions,
            )
        else:
            return 1.0

    def apply_cmb_heating(self, temperature, emissivity, redshift) -> Tuple:
        """Compute the cmb heating factor and modify the temperature.

        This stores the effective temperature in last_effective_temperature
        which can be returned by temperature_z for labelling. This feature is
        mostly only useful when using the generators in isolation.
        """
        # Return immediately if we aren't applying cmb heating
        if not self.do_cmb_heating:
            self.last_cmb_factor = 1.0
            self.last_effective_temperature = temperature
            return 1.0, temperature

        # Otherwise, do the calculation
        cmb_factor, _temperature = get_cmb_heating_factor(
            temperature,
            emissivity,
            redshift,
        )

        # Store the last results
        self.last_cmb_factor = cmb_factor
        self.last_effective_temperature = _temperature

        return cmb_factor, _temperature
