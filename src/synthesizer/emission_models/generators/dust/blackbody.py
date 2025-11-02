"""A submodule defining blackbody dust emission generators."""

from typing import Optional

import numpy as np
from unyt import Hz, angstrom, c, erg, s, unyt_array, unyt_quantity

from synthesizer.components.component import Component
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust.dust_emission_base import (
    DustEmission,
)
from synthesizer.emissions import LineCollection, Sed
from synthesizer.units import accepts
from synthesizer.utils import planck


class Blackbody(DustEmission):
    """A class to generate a blackbody emission spectrum.

    This can be used to generate standalone normalised blackbody spectra, or
    it can be used to generate scaled blackbody spectra by providing either
    a scaler EmissionModel or an intrinsic/attenuated EmissionModel pair to
    scale the blackbody spectrum accordingly. For more details see the base
    DustEmission parent class.

    Attributes:
        temperature (unyt_quantity):
            The temperature of the blackbody.
        temperature_z (unyt_quantity):
            The temperature of the greybody at redshift z, accounting for
            CMB heating. Stores the last used temperature (important when
            used with emitter temperatures).
        cmb_factor (float):
            The multiplicative factor to account for CMB heating at
            high-redshift.
    """

    temperature: unyt_quantity

    def __init__(
        self,
        temperature: unyt_quantity,
        do_cmb_heating: bool = False,
        intrinsic: Optional[EmissionModel] = None,
        attenuated: Optional[EmissionModel] = None,
        scaler: Optional[EmissionModel] = None,
    ) -> None:
        """Initialise the blackbody base class.

        Args:
            temperature (unyt_array):
                The temperature of the blackbody.
            do_cmb_heating (bool):
                Should we apply cmb heating?
            intrinsic (EmissionModel):
                The name of the intrinsic emission defining the unattenuated
                emission for energy balance.
            attenuated (EmissionModel):
                The name of the attenuated emission defining the attenuated
                emission for energy balance.
            scaler (EmissionModel):
                The name of the emission to use to scale the dust emission by.
        """
        self.temperature = temperature

        DustEmission.__init__(
            self,
            intrinsic,
            attenuated,
            scaler,
            do_cmb_heating,
            required_params=("temperature",),
        )

    @accepts(lams=angstrom)
    def _generate_spectra(
        self,
        lams: unyt_array,
        emitter: Component,
        model: EmissionModel,
        emissions: dict,
        redshift: float = 0,
    ) -> Sed:
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
            emissions (dict):
                Dictionary containing all emissions generated so far.
            redshift (float):
                The redshift at which to calculate the CMB heating. (Ignored
                if not applying CMB heating).

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
        cmb_factor, temperature = self.apply_cmb_heating(
            temperature=temperature,
            emissivity=1.0,  # Blackbody emissivity=1
            redshift=redshift,
        )

        # Compute the planck function
        lnu = planck(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Normalise the spectrum and apply scaling with proper unit handling
        # Get the bolometric luminosity with proper units
        bol_lum = sed.bolometric_luminosity
        scaling = self.get_scaling(emitter, model, emissions)

        # Properly handle units: normalize then scale
        sed._lnu = (sed.lnu / bol_lum * scaling * cmb_factor).value

        return sed

    @accepts(line_lams=angstrom)
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
        emissions,
        redshift=0,
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
            emissions (dict):
                Dictionary containing all emissions generated so far.
            redshift (float):
                The redshift at which to calculate the CMB heating. (Ignored
                if not applying CMB heating).

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
        cmb_factor, temperature = self.apply_cmb_heating(
            temperature=temperature,
            emissivity=1.0,  # Blackbody emissivity=1
            redshift=redshift,
        )

        # Compute the planck function
        lnu = planck(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=line_lams, lnu=lnu)

        # Normalise the spectrum and apply scaling with proper unit handling
        # Get the bolometric luminosity with proper units
        bol_lum = sed.bolometric_luminosity
        scaling = self.get_scaling(emitter, model, emissions)

        # Properly handle units: normalize then scale
        sed._lnu = (sed.lnu / bol_lum * scaling * cmb_factor).value

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
        return self._generate_spectra(lams, None, None, {}, redshift)
