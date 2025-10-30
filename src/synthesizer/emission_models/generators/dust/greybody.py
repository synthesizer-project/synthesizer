"""A submodule defining greybody dust emission generators."""

from typing import Optional, Union

import numpy as np
from unyt import (
    Hz,
    K,
    angstrom,
    c,
    erg,
    s,
    um,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emissions import LineCollection, Sed
from synthesizer.units import accepts
from synthesizer.utils import planck

from .dust_emission_base import (
    EnergyBalanceDustEmission,
    ScaledDustEmission,
    get_cmb_heating_factor,
)


class GreybodyBase:
    """A base class for greybody emission spectrum generators.

    This class defines common attributes and methods for greybody emission
    which gets specialised by other classes.

    Attributes:
        temperature (unyt_quantity):
            The temperature of the greybody.
        emissivity (float):
            The emissivity of the dust (dimensionless).
        cmb_factor (float):
            The multiplicative factor to account for CMB heating at
            high-redshift.
        optically_thin (bool):
            If dust is optically thin?
        lam_0 (unyt_quantity):
            Wavelength (in um) where the dust optical depth is unity.
    """

    temperature: unyt_quantity
    emissivity: float
    cmb_factor: float
    optically_thin: bool
    lam_0: unyt_quantity

    def __init__(
        self,
        temperature: unyt_quantity,
        emissivity: float,
        cmb_factor: float = 1,
        optically_thin: bool = True,
        lam_0: unyt_quantity = 100.0 * um,
    ) -> None:
        """Initialise the greybody base class.

        Args:
            temperature (unyt_array):
                The temperature of the greybody.
            emissivity (float):
                The emissivity of the dust (dimensionless).
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            optically_thin (bool):
                If dust is optically thin
            lam_0 (unyt_quantity):
                Wavelength (in um) where the dust optical depth is unity
        """
        self.temperature = temperature
        self.emissivity = emissivity
        self.cmb_factor = cmb_factor
        self.optically_thin = optically_thin
        self.lam_0 = lam_0

    @accepts(nu=Hz)
    def _lnu(self, nu: unyt_array, temperature: unyt_quantity) -> unyt_array:
        """Generate unnormalised spectrum for given frequency (nu) grid.

        Args:
            nu (unyt_array):
                The frequencies at which to calculate the spectral luminosity
                density.
            temperature (unyt_quantity):
                The temperature to use for the greybody.

        Returns:
            lnu (unyt_array):
                The unnormalised spectral luminosity density.
        """
        if self.optically_thin:
            return (nu / Hz) ** self.emissivity * planck(nu, temperature)
        else:
            _nu_0 = c / self.lam_0
            optically_thick_factor = 1 - np.exp(
                -((nu / _nu_0) ** self.emissivity)
            )
            return optically_thick_factor * planck(nu, temperature)

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
                "Temperature must be provided to the GreybodyEnergyBalance "
                "class when generating the normalised spectra directly."
            )

        # Account for CMB heating
        temperature = self.temperature
        if self.cmb_factor != 1:
            # Update the cmb_factor and temperature_z
            cmb_factor, temperature = get_cmb_heating_factor(
                temperature=temperature,
                emissivity=self.emissivity,
                redshift=redshift,
            )

        # Define frequencies
        nu = (c / lams).to(Hz)

        # Compute the greybody function
        lnu = self._lnu(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Apply CMB factor
        sed._lnu *= self.cmb_factor

        return sed


class GreybodyEnergyBalance(EnergyBalanceDustEmission, GreybodyBase):
    """A generator for greybody emission spectrum based on energy balance.

    This class will generate a normalised greybody spectrum based on the
    temperature and emissivity of the dust. This temperature can either be
    provided directly to the class or extracted from the component or model
    (with the latter taking precedence). It will then scale the spectrum based
    on the energy absorbed by dust, calculated as the difference between the
    intrinsic and attenuated emissions. The "energy balance" method.
    """

    @accepts(temperature=K, lam_0=um)
    def __init__(
        self,
        temperature: Optional[Union[unyt_quantity, float]] = None,
        emissivity: float = 1.5,
        cmb_factor: float = 1,
        optically_thin: bool = True,
        lam_0: unyt_quantity = 100.0 * um,
        intrinsic: Union[str, EmissionModel] = "intrinsic",
        attenuated: Union[str, EmissionModel] = "attenuated",
    ) -> None:
        """Generate a greybody spectrum.

        Args:
            temperature (unyt_array, optional):
                The temperature of the dust. This will be extracted from
                the component or model by default but if not given there
                the value here will be used.
            emissivity (float):
                The emissivity of the dust (dimensionless).
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            optically_thin (bool):
                If dust is optically thin
            lam_0 (unyt_quantity):
                Wavelength (in um) where the dust optical depth is unity
            intrinsic (str):
                The name of the intrinsic spectrum defining the unattenuated
                emission for energy balance.
            attenuated (str):
                The name of the attenuated spectrum defining the attenuated
                emission for energy balance.
        """
        GreybodyBase.__init__(
            self,
            temperature,
            emissivity,
            cmb_factor,
            optically_thin,
            lam_0,
        )
        EnergyBalanceDustEmission.__init__(self, intrinsic, attenuated)

    @accepts(lams=angstrom)
    def _generate_spectra(self, lams, emitter, model) -> Sed:
        """Generate the dust emission spectra.

        Note that the temperature here will be extracted using this priority
        order:
            1. From the EmissionModels fixed parameters
            2. From the emitter object
            3. From the Greybody object itself

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
                emissivity=self.emissivity,
                redshift=emitter.redshift,
            )

        # Compute the greybody function
        lnu = self._lnu(nu, temperature)

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

        Greybody emission does not produce line emission, so this
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
                emissivity=self.emissivity,
                redshift=emitter.redshift,
            )

        # Compute the greybody function
        lnu = self._lnu(nu, temperature)

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
            lum=np.zeros(sed._lnu.shape) * erg / s,
            cont=sed.lnu,
        )

        return lines


class GreybodyScaler(ScaledDustEmission, GreybodyBase):
    """A generator for greybody emission spectrum scaled by another spectra.

    This class will generate a normalised greybody spectrum based on the
    temperature and emissivity of the dust. This temperature can either be
    provided directly to the class or extracted from the component or model
    (with the latter taking precedence). It will then scale the spectrum based
    on the bolometric luminosity of another emission.
    """

    @accepts(temperature=K, lam_0=um)
    def __init__(
        self,
        temperature: Optional[Union[unyt_quantity, float]] = None,
        emissivity: float = 1.5,
        cmb_factor: float = 1,
        optically_thin: bool = True,
        lam_0: unyt_quantity = 100.0 * um,
        scaler: Union[str, EmissionModel] = "scaler",
    ) -> None:
        """Generate a greybody spectrum.

        Args:
            temperature (unyt_array, optional):
                The temperature of the dust. This will be extracted from
                the component or model by default but if not given there
                the value here will be used.
            emissivity (float):
                The emissivity of the dust (dimensionless).
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            optically_thin (bool):
                If dust is optically thin
            lam_0 (unyt_quantity):
                Wavelength (in um) where the dust optical depth is unity
            scaler (str/EmissionModel):
                The name of the emission to scale the dust emission by.
        """
        GreybodyBase.__init__(
            self,
            temperature,
            emissivity,
            cmb_factor,
            optically_thin,
            lam_0,
        )
        ScaledDustEmission.__init__(self, scaler)

    @accepts(lams=angstrom)
    def _generate_spectra(self, lams, emitter, model) -> Sed:
        """Generate the dust emission spectra.

        Note that the temperature here will be extracted using this priority
        order:
            1. From the EmissionModels fixed parameters
            2. From the emitter object
            3. From the Greybody object itself

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
                emissivity=self.emissivity,
                redshift=emitter.redshift,
            )

        # Compute the greybody function
        lnu = self._lnu(nu, temperature)

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

        Greybody emission does not produce line emission, so this
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
                emissivity=self.emissivity,
                redshift=emitter.redshift,
            )

        # Compute the greybody function
        lnu = self._lnu(nu, temperature)

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
            lum=np.zeros(sed._lnu.shape) * erg / s,
            cont=sed.lnu,
        )

        return lines


class Greybody:
    """A class to generate a greybody emission spectrum.

    This class is a factory which will return the correct flavour of
    greybody generator based on the input arguments.
    """

    def __new__(
        cls,
        temperature: unyt_quantity,
        emissivity: float = 1.5,
        cmb_heating: float = 1,
        optically_thin: bool = True,
        lam_0: unyt_quantity = 100.0 * um,
        intrinsic: Union[str, EmissionModel] = "intrinsic",
        attenuated: Union[str, EmissionModel] = "attenuated",
        scaler: Union[str, EmissionModel] = "scaler",
    ) -> Union[GreybodyEnergyBalance, GreybodyScaler]:
        """Initialise the greybody emission model.

        Args:
            temperature (unyt_array):
                The temperature of the dust. This is only used if the
                temperature is not defined on the emitter and or model.
            emissivity (float):
                The emissivity of the dust (dimensionless).
            cmb_heating (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            optically_thin (bool):
                If dust is optically thin
            lam_0 (unyt_quantity):
                Wavelength (in um) where the dust optical depth is unity
            intrinsic (str/EmissionModel):
                The name of the intrinsic spectrum defining the unattenuated
                emission for energy balance.
            attenuated (str/EmissionModel):
                The name of the attenuated spectrum defining the attenuated
                emission for energy balance.
            scaler (str/EmissionModel):
                The name of the emission to scale the dust emission by.

        Returns:
            GreybodyEnergyBalance or GreybodyScaler:
                The appropriate greybody generator based on the input args.
        """
        if (intrinsic is not None) and (attenuated is not None):
            return GreybodyEnergyBalance(
                temperature=temperature,
                emissivity=emissivity,
                cmb_factor=cmb_heating,
                optically_thin=optically_thin,
                lam_0=lam_0,
                intrinsic=intrinsic,
                attenuated=attenuated,
            )
        elif scaler is not None:
            return GreybodyScaler(
                temperature=temperature,
                emissivity=emissivity,
                cmb_factor=cmb_heating,
                optically_thin=optically_thin,
                lam_0=lam_0,
                scaler=scaler,
            )
        else:
            raise exceptions.InvalidInput(
                "Either intrinsic and attenuated emissions "
                "or a scaler emission must be provided to "
                "instantiate a Greybody dust emission model."
            )
