"""A submodule defining blackbody dust emission generators."""

from typing import Optional, Union

import numpy as np
from unyt import (
    Hz,
    K,
    angstrom,
    c,
    erg,
    s,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust import (
    EnergyBalanceDustEmission,
    ScaledDustEmission,
    get_cmb_heating_factor,
)
from synthesizer.emissions import LineCollection, Sed
from synthesizer.units import accepts
from synthesizer.utils import planck


class BlackbodyBase:
    """A base class for blackbody emission spectrum generators.

    This class defines common attributes and methods for blackbody emission
    which gets specialised by other classes.
    """

    temperature: unyt_quantity
    cmb_factor: float

    def __init__(
        self, temperature: unyt_quantity, cmb_factor: float = 1
    ) -> None:
        """Initialise the blackbody base class.

        Args:
            temperature (unyt_array):
                The temperature of the blackbody.
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
        """
        self.temperature = temperature
        self.cmb_factor = cmb_factor

    @accepts(lams=angstrom)
    def get_spectra(self, lams: unyt_array, redshift: float = 0) -> Sed:
        """Generate dust emission spectra for given wavelength grid.

        This will return the scaling free spectra for a given wavelength grid.
        It will not consider any emitter or model, so the temperature must
        have been provided directly to the class.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            redshift (float):
                The redshift at which to calculate the CMB heating.

        Returns:
            Sed:
                The generated dust emission SED.
        """
        # Ensure we have a temperature
        if self.temperature is None:
            raise exceptions.MissingAttribute(
                "Temperature must be provided to the BlackbodyEnergyBalance "
                "class when generating the normalised spectra directly."
            )

        # Account for CMB heating
        temperature = self.temperature
        if self.cmb_factor != 1:
            # Update the cmb_factor and temperature_z
            cmb_factor, temperature = get_cmb_heating_factor(
                temperature=temperature,
                emissivity=1.0,  # Blackbody emissivity=1
                redshift=redshift,
            )

        # Define frequencies
        nu = (c / lams).to(Hz)

        # Compute the planck function
        lnu = planck(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Apply CMB factor
        sed._lnu *= self.cmb_factor

        return sed


class BlackbodyEnergyBalance(EnergyBalanceDustEmission, BlackbodyBase):
    """A generator for blackbody emission spectrum based on energy balance.

    This class will generate a normalised blackbody spectrum based on the
    temperature of the dust. This temperature can either be provided directly
    to the class or extracted from the component or model (with the latter
    taking precedence). It will then scale the spectrum based on the energy
    absorbed by dust, calculated as the difference between the intrinsic and
    attenuated emissions. The "energy balance" method.
    """

    @accepts(temperature=K)
    def __init__(
        self,
        temperature: Optional[Union[unyt_quantity, float]] = None,
        cmb_factor: float = 1,
        intrinsic: Union[str, EmissionModel] = "intrinsic",
        attenuated: Union[str, EmissionModel] = "attenuated",
    ) -> None:
        """Generate a simple blackbody spectrum.

        Args:
            temperature (unyt_array, optional):
                The temperature of the dust. This will be extracted from
                the component or model by default but if not given there
                the value here will be used.
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            intrinsic (str):
                The name of the intrinsic spectrum defining the unattenuated
                emission for energy balance.
            attenuated (str):
                The name of the attenuated spectrum defining the attenuated
                emission for energy balance.
        """
        BlackbodyBase.__init__(self, temperature, cmb_factor)
        EnergyBalanceDustEmission.__init__(self, intrinsic, attenuated)

    @accepts(lams=angstrom)
    def _generate_spectra(self, lams, emitter, model) -> Sed:
        """Generate the dust emission spectra.

        Note that the temperature here will be extracted using this priority
        order:
            1. From the EmissionModels fixed parameters
            2. From the emitter object
            3. From the Blackbody object itself

        This is handle internally using the Generator's _extract_params method.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.

        Returns:
            Sed:
                The generated dust emission SED.
        """
        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack the temperature
        temperature = params["temperature"]

        # Define frequencies
        nu = (c / lams).to(Hz)

        # Account for CMB heating
        if self.cmb_factor != 1:
            # Update the cmb_factor and temperature_z
            cmb_factor, temperature = get_cmb_heating_factor(
                temperature=temperature,
                emissivity=1.0,  # Blackbody emissivity=1
                redshift=emitter.redshift,
            )

        # Compute the planck function
        lnu = planck(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Normalise the spectrum
        sed._lnu /= np.expand_dims(sed._bolometric_luminosity, axis=-1)

        # Apply CMB factor
        sed._lnu *= self.cmb_factor

        # Apply the energy balance scaling
        sed._lnu *= self._get_energy_balance_luminosity(emitter, model)

        return sed

    @accepts(line_lams=angstrom)
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
    ) -> LineCollection:
        """Generate line emission spectra.

        Blackbody emission does not produce line emission, so this
        method will simply return an empty Sed object.

        Args:
            line_ids (list):
                The IDs of the lines to generate.
            line_lams (unyt_array):
                The wavelength grid on which to generate the line spectra.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.

        Returns:
            LineCollection:
                The generated line collection.
        """
        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack the temperature
        temperature = params["temperature"]

        # Define frequencies
        nu = (c / line_lams).to(Hz)

        # Account for CMB heating
        if self.cmb_factor != 1:
            # Update the cmb_factor and temperature_z
            cmb_factor, temperature = get_cmb_heating_factor(
                temperature=temperature,
                emissivity=1.0,  # Blackbody emissivity=1
                redshift=emitter.redshift,
            )

        # Compute the planck function
        lnu = planck(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=line_lams, lnu=lnu)

        # Normalise the spectrum
        sed._lnu /= np.expand_dims(sed._bolometric_luminosity, axis=-1)

        # Apply CMB factor
        sed._lnu *= self.cmb_factor

        # Apply the energy balance scaling
        sed._lnu *= self._get_energy_balance_luminosity(emitter, model)

        # OK, now we have used the Sed magic lets return the LineCollection
        # the outside world expects. Note that the line luminosities are
        # by definition zero for generated lines and the contribution only
        # goes to the continuum.
        lines = LineCollection(
            line_ids,
            line_lams,
            np.zeros(sed._lnu.shape) * erg / s,
            cont=sed.lnu,
        )

        return lines


class BlackbodyScaler(ScaledDustEmission, BlackbodyBase):
    """A generator for blackbody emission spectrum scaled by another spectra.

    This class will generate a normalised blackbody spectrum based on the
    temperature of the dust. This temperature can either be provided directly
    to the class or extracted from the component or model (with the latter
    taking precedence). It will then scale the spectrum based on the bolometric
    luminosity of another emission.
    """

    @accepts(temperature=K)
    def __init__(
        self,
        temperature: Optional[Union[unyt_quantity, float]] = None,
        cmb_factor: float = 1,
        scaler: Union[str, EmissionModel] = "scaler",
    ) -> None:
        """Generate a simple blackbody spectrum.

        Args:
            temperature (unyt_array, optional):
                The temperature of the dust. This will be extracted from
                the component or model by default but if not given there
                the value here will be used.
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            scaler (str/EmissionModel):
                The name of the emission to scale the dust emission by.
        """
        BlackbodyBase.__init__(self, temperature, cmb_factor)
        ScaledDustEmission.__init__(self, temperature, cmb_factor, scaler)

    @accepts(lams=angstrom)
    def _generate_spectra(self, lams, emitter, model) -> Sed:
        """Generate the dust emission spectra.

        Note that the temperature here will be extracted using this priority
        order:
            1. From the EmissionModels fixed parameters
            2. From the emitter object
            3. From the Blackbody object itself

        This is handle internally using the Generator's _extract_params method.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.

        Returns:
            Sed:
                The generated dust emission SED.
        """
        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack the temperature
        temperature = params["temperature"]

        # Define frequencies
        nu = (c / lams).to(Hz)

        # Account for CMB heating
        if self.cmb_factor != 1:
            # Update the cmb_factor and temperature_z
            cmb_factor, temperature = get_cmb_heating_factor(
                temperature=temperature,
                emissivity=1.0,  # Blackbody emissivity=1
                redshift=emitter.redshift,
            )

        # Compute the planck function
        lnu = planck(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Normalise the spectrum
        sed._lnu /= np.expand_dims(sed._bolometric_luminosity, axis=-1)

        # Apply CMB factor
        sed._lnu *= self.cmb_factor

        # Apply the scaling luminosity
        sed._lnu *= self._get_scaling_luminosity(emitter, model)

        return sed

    @accepts(line_lams=angstrom)
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
    ) -> LineCollection:
        """Generate line emission spectra.

        Blackbody emission does not produce line emission, so this
        method will simply return an empty Sed object.

        Args:
            line_ids (list):
                The IDs of the lines to generate.
            line_lams (unyt_array):
                The wavelength grid on which to generate the line spectra.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.

        Returns:
            LineCollection:
                The generated line collection.
        """
        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack the temperature
        temperature = params["temperature"]

        # Define frequencies
        nu = (c / line_lams).to(Hz)

        # Account for CMB heating
        if self.cmb_factor != 1:
            # Update the cmb_factor and temperature_z
            cmb_factor, temperature = get_cmb_heating_factor(
                temperature=temperature,
                emissivity=1.0,  # Blackbody emissivity=1
                redshift=emitter.redshift,
            )

        # Compute the planck function
        lnu = planck(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=line_lams, lnu=lnu)

        # Normalise the spectrum
        sed._lnu /= np.expand_dims(sed._bolometric_luminosity, axis=-1)

        # Apply CMB factor
        sed._lnu *= self.cmb_factor

        # Apply the scaling luminosity
        sed._lnu *= self._get_scaling_luminosity(emitter, model)

        # OK, now we have used the Sed magic lets return the LineCollection
        # the outside world expects. Note that the line luminosities are
        # by definition zero for generated lines and the contribution only
        # goes to the continuum.
        lines = LineCollection(
            line_ids,
            line_lams,
            np.zeros(sed._lnu.shape) * erg / s,
            cont=sed.lnu,
        )

        return lines


class Blackbody:
    """A class to generate a blackbody emission spectrum.

    This class is a factory which will return the correct flavour of
    blackbody generator based on the input arguments.
    """

    def __new__(
        cls,
        temperature: unyt_quantity,
        cmb_heating: float = 1,
        intrinsic: Union[str, EmissionModel] = "intrinsic",
        attenuated: Union[str, EmissionModel] = "attenuated",
        scaler: Union[str, EmissionModel] = "scaler",
    ) -> Union[BlackbodyEnergyBalance, BlackbodyScaler]:
        """Initialise the blackbody emission model.

        Args:
            temperature (unyt_array):
                The temperature of the dust. This is only used if the
                temperature is not defined on the emitter and or model.
            cmb_heating (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            intrinsic (str/EmissionModel):
                The name of the intrinsic spectrum defining the unattenuated
                emission for energy balance.
            attenuated (str/EmissionModel):
                The name of the attenuated spectrum defining the attenuated
                emission for energy balance.
            scaler (str/EmissionModel):
                The name of the emission to scale the dust emission by.

        Returns:
            BlackbodyEnergyBalance or BlackbodyScaler:
                The appropriate blackbody generator based on the input args.
        """
        if (intrinsic is not None) and (attenuated is not None):
            return BlackbodyEnergyBalance(
                temperature=temperature,
                cmb_factor=cmb_heating,
                intrinsic=intrinsic,
                attenuated=attenuated,
            )
        elif scaler is not None:
            return BlackbodyScaler(
                temperature=temperature,
                cmb_factor=cmb_heating,
                scaler=scaler,
            )
        else:
            raise exceptions.InvalidInput(
                "Either intrinsic and attenuated emissions "
                "or a scaler emission must be provided to "
                "instantiate a Blackbody dust emission model."
            )
