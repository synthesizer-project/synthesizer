"""A submodule defining Casey12 dust emission generators."""

from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from unyt import (
    Hz,
    K,
    angstrom,
    c,
    erg,
    h,
    kb,
    s,
    um,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emissions import LineCollection, Sed
from synthesizer.units import accepts

from .dust_emission_base import (
    EnergyBalanceDustEmission,
    ScaledDustEmission,
    get_cmb_heating_factor,
)


class Casey12Base:
    """A base class for Casey12 emission spectrum generators.

    This class defines common attributes and methods for Casey12 dust emission
    which gets specialised by other classes.

    Attributes:
        temperature (unyt_quantity):
            The temperature of the dust.
        emissivity (float):
            The emissivity of the dust (dimensionless).
        alpha (float):
            The power-law slope (dimensionless) [good value = 2.0].
        n_bb (float):
            Normalisation of the blackbody component [default 1.0].
        lam_0 (unyt_quantity):
            Wavelength where the dust optical depth is unity.
        lam_c (unyt_quantity):
            The power law turnover wavelength.
        n_pl (float):
            The power law normalisation.
        cmb_factor (float):
            The multiplicative factor to account for CMB heating at
            high-redshift.
    """

    temperature: unyt_quantity
    emissivity: float
    alpha: float
    n_bb: float
    lam_0: unyt_quantity
    lam_c: unyt_quantity
    n_pl: float
    cmb_factor: float

    @accepts(temperature=K, lam_0=angstrom)
    def __init__(
        self,
        temperature: unyt_quantity,
        emissivity: float,
        alpha: float,
        n_bb: float = 1.0,
        lam_0: unyt_quantity = 200.0 * um,
        cmb_factor: float = 1,
    ) -> None:
        """Initialise the Casey12 base class.

        Args:
            temperature (unyt_array):
                The temperature of the dust.
            emissivity (float):
                The emissivity (dimensionless) [good value = 2.0].
            alpha (float):
                The power-law slope (dimensionless) [good value = 2.0].
            n_bb (float):
                Normalisation of the blackbody component [default 1.0].
            lam_0 (unyt_quantity):
                Wavelength where the dust optical depth is unity.
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
        """
        self.temperature = temperature
        self.emissivity = emissivity
        self.alpha = alpha
        self.n_bb = n_bb
        self.lam_0 = lam_0
        self.cmb_factor = cmb_factor

        # Calculate the power law turnover wavelength
        b1 = 26.68
        b2 = 6.246
        b3 = 0.0001905
        b4 = 0.00007243
        lum = (
            (b1 + b2 * alpha) ** -2
            + (b3 + b4 * alpha) * self.temperature.to("K").value
        ) ** -1

        self.lam_c = (3.0 / 4.0) * lum * um

        # Calculate normalisation of the power-law term
        # See Casey+2012, Table 1 for the expression
        # Missing factors of lam_c and c in some places
        self.n_pl = (
            self.n_bb
            * (1 - np.exp(-((self.lam_0 / self.lam_c) ** emissivity)))
            * (c / self.lam_c) ** 3
            / (np.exp(h * c / (self.lam_c * kb * self.temperature)) - 1)
        )

    def _lnu_power_law(self, lam: unyt_array) -> unyt_array:
        """Calculate the power-law component.

        Args:
            lam (unyt_array):
                The wavelengths at which to calculate lnu.

        Returns:
            unyt_array:
                The power-law component of lnu.
        """
        return (
            (
                self.n_pl
                * ((lam / self.lam_c) ** (self.alpha))
                * np.exp(-((lam / self.lam_c) ** 2))
            ).value
            * erg
            / s
            / Hz
        )

    def _lnu_blackbody(
        self,
        lam: unyt_array,
        temperature: unyt_quantity,
    ) -> unyt_array:
        """Calculate the blackbody component.

        Args:
            lam (unyt_array):
                The wavelengths at which to calculate lnu.
            temperature (unyt_quantity):
                The temperature to use for the blackbody component.

        Returns:
            unyt_array:
                The blackbody component of lnu.
        """
        return (
            (
                self.n_bb
                * (1 - np.exp(-((self.lam_0 / lam) ** self.emissivity)))
                * (c / lam) ** 3
                / (np.exp((h * c) / (lam * kb * temperature)) - 1.0)
            ).value
            * erg
            / s
            / Hz
        )

    @accepts(nu=Hz)
    def _lnu(
        self,
        nu: unyt_array,
        temperature: unyt_quantity,
    ) -> Union[NDArray[np.float64], unyt_array]:
        """Generate unnormalised spectrum for given frequency (nu) grid.

        Args:
            nu (unyt_array):
                The frequencies at which to calculate the spectral luminosity
                density.
            temperature (unyt_quantity):
                The temperature to use for the Casey12 spectrum.

        Returns:
            lnu (unyt_array):
                The unnormalised spectral luminosity density.
        """
        # Convert frequencies to wavelengths
        lam = c / nu

        # Calculate the power-law and blackbody components
        power_law_component = self._lnu_power_law(lam)
        blackbody_component = self._lnu_blackbody(lam, temperature)

        return power_law_component + blackbody_component

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
                "Temperature must be provided to the Casey12EnergyBalance "
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

        # Compute the Casey12 function
        lnu = self._lnu(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Apply CMB factor
        sed._lnu *= self.cmb_factor

        return sed


class Casey12EnergyBalance(EnergyBalanceDustEmission, Casey12Base):
    """A generator for Casey12 emission spectrum based on energy balance.

    This class will generate a normalised Casey12 spectrum based on the
    temperature and other parameters of the dust. This temperature can either
    be provided directly to the class or extracted from the component or model
    (with the latter taking precedence). It will then scale the spectrum based
    on the energy absorbed by dust, calculated as the difference between the
    intrinsic and attenuated emissions. The "energy balance" method.
    """

    @accepts(temperature=K, lam_0=angstrom)
    def __init__(
        self,
        temperature: Optional[Union[unyt_quantity, float]] = None,
        emissivity: float = 2.0,
        alpha: float = 2.0,
        n_bb: float = 1.0,
        lam_0: unyt_quantity = 200.0 * um,
        cmb_factor: float = 1,
        intrinsic: Union[str, EmissionModel] = "intrinsic",
        attenuated: Union[str, EmissionModel] = "attenuated",
    ) -> None:
        """Generate a Casey12 spectrum.

        Args:
            temperature (unyt_array, optional):
                The temperature of the dust. This will be extracted from
                the component or model by default but if not given there
                the value here will be used.
            emissivity (float):
                The emissivity (dimensionless) [good value = 2.0].
            alpha (float):
                The power-law slope (dimensionless) [good value = 2.0].
            n_bb (float):
                Normalisation of the blackbody component [default 1.0].
            lam_0 (unyt_quantity):
                Wavelength where the dust optical depth is unity.
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
        Casey12Base.__init__(
            self,
            temperature,
            emissivity,
            alpha,
            n_bb,
            lam_0,
            cmb_factor,
        )
        EnergyBalanceDustEmission.__init__(self, intrinsic, attenuated)

    @accepts(lams=angstrom)
    def _generate_spectra(self, lams, emitter, model) -> Sed:
        """Generate the dust emission spectra.

        Note that the temperature here will be extracted using this priority
        order:
            1. From the EmissionModels fixed parameters
            2. From the emitter object
            3. From the Casey12 object itself

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

        # Compute the Casey12 function
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

        Casey12 emission does not produce line emission, so this
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

        # Compute the Casey12 function
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


class Casey12Scaler(ScaledDustEmission, Casey12Base):
    """A generator for Casey12 emission spectrum scaled by another spectra.

    This class will generate a normalised Casey12 spectrum based on the
    temperature and other parameters of the dust. This temperature can either
    be provided directly to the class or extracted from the component or model
    (with the latter taking precedence). It will then scale the spectrum based
    on the bolometric luminosity of another emission.
    """

    @accepts(temperature=K, lam_0=angstrom)
    def __init__(
        self,
        temperature: Optional[Union[unyt_quantity, float]] = None,
        emissivity: float = 2.0,
        alpha: float = 2.0,
        n_bb: float = 1.0,
        lam_0: unyt_quantity = 200.0 * um,
        cmb_factor: float = 1,
        scaler: Union[str, EmissionModel] = "scaler",
    ) -> None:
        """Generate a Casey12 spectrum.

        Args:
            temperature (unyt_array, optional):
                The temperature of the dust. This will be extracted from
                the component or model by default but if not given there
                the value here will be used.
            emissivity (float):
                The emissivity (dimensionless) [good value = 2.0].
            alpha (float):
                The power-law slope (dimensionless) [good value = 2.0].
            n_bb (float):
                Normalisation of the blackbody component [default 1.0].
            lam_0 (unyt_quantity):
                Wavelength where the dust optical depth is unity.
            cmb_factor (float):
                The multiplicative factor to account for CMB heating at
                high-redshift.
            scaler (str/EmissionModel):
                The name of the emission to scale the dust emission by.
        """
        Casey12Base.__init__(
            self,
            temperature,
            emissivity,
            alpha,
            n_bb,
            lam_0,
            cmb_factor,
        )
        ScaledDustEmission.__init__(self, scaler)

    @accepts(lams=angstrom)
    def _generate_spectra(self, lams, emitter, model) -> Sed:
        """Generate the dust emission spectra.

        Note that the temperature here will be extracted using this priority
        order:
            1. From the EmissionModels fixed parameters
            2. From the emitter object
            3. From the Casey12 object itself

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

        # Compute the Casey12 function
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

        Casey12 emission does not produce line emission, so this
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

        # Compute the Casey12 function
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


class Casey12:
    """A class to generate a Casey12 emission spectrum.

    This class is a factory which will return the correct flavour of
    Casey12 generator based on the input arguments.

    This is based on the work of Casey (2012):
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.425.3094C/abstract
    """

    def __new__(
        cls,
        temperature: unyt_quantity,
        emissivity: float = 2.0,
        alpha: float = 2.0,
        n_bb: float = 1.0,
        lam_0: unyt_quantity = 200.0 * um,
        cmb_heating: float = 1,
        intrinsic: Union[str, EmissionModel] = "intrinsic",
        attenuated: Union[str, EmissionModel] = "attenuated",
        scaler: Union[str, EmissionModel] = "scaler",
    ) -> Union[Casey12EnergyBalance, Casey12Scaler]:
        """Initialise the Casey12 emission model.

        Args:
            temperature (unyt_array):
                The temperature of the dust. This is only used if the
                temperature is not defined on the emitter and or model.
            emissivity (float):
                The emissivity (dimensionless) [good value = 2.0].
            alpha (float):
                The power-law slope (dimensionless) [good value = 2.0].
            n_bb (float):
                Normalisation of the blackbody component [default 1.0].
            lam_0 (unyt_quantity):
                Wavelength where the dust optical depth is unity.
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
            Casey12EnergyBalance or Casey12Scaler:
                The appropriate Casey12 generator based on the input args.
        """
        if (intrinsic is not None) and (attenuated is not None):
            return Casey12EnergyBalance(
                temperature=temperature,
                emissivity=emissivity,
                alpha=alpha,
                n_bb=n_bb,
                lam_0=lam_0,
                cmb_factor=cmb_heating,
                intrinsic=intrinsic,
                attenuated=attenuated,
            )
        elif scaler is not None:
            return Casey12Scaler(
                temperature=temperature,
                emissivity=emissivity,
                alpha=alpha,
                n_bb=n_bb,
                lam_0=lam_0,
                cmb_factor=cmb_heating,
                scaler=scaler,
            )
        else:
            raise exceptions.InvalidInput(
                "Either intrinsic and attenuated emissions "
                "or a scaler emission must be provided to "
                "instantiate a Casey12 dust emission model."
            )
