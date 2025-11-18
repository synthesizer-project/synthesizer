"""A submodule defining greybody dust emission generators."""

from typing import Optional

import numpy as np
from unyt import (
    Hz,
    angstrom,
    c,
    erg,
    s,
    um,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.components.component import Component
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust.dust_emission_base import (
    DustEmission,
)
from synthesizer.emissions import LineCollection, Sed
from synthesizer.units import accepts
from synthesizer.utils import planck


class Greybody(DustEmission):
    """A class to generate a greybody emission spectrum.

    This can be used to generate standalone normalised greybody spectra, or
    it can be used to generate scaled greybody spectra by providing either
    a scaler EmissionModel or an intrinsic/attenuated EmissionModel pair to
    scale the greybody spectrum accordingly. For more details see the base
    DustEmission parent class.

    Attributes:
        temperature (unyt_quantity):
            The temperature of the greybody.
        temperature_z (unyt_quantity):
            The temperature of the greybody at redshift z, accounting for
            CMB heating. Stores the last used temperature (important when
            used with emitter temperatures).
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
    optically_thin: bool
    lam_0: unyt_quantity

    def __init__(
        self,
        temperature: unyt_quantity,
        emissivity: float = 1.5,
        optically_thin: bool = True,
        lam_0: unyt_quantity = 100.0 * um,
        do_cmb_heating: bool = False,
        intrinsic: Optional[EmissionModel] = None,
        attenuated: Optional[EmissionModel] = None,
        scaler: Optional[EmissionModel] = None,
    ) -> None:
        """Initialise the greybody base class.

        Args:
            temperature (unyt_array):
                The temperature of the greybody.
            emissivity (float):
                The emissivity of the dust (dimensionless).
            optically_thin (bool):
                If dust is optically thin
            lam_0 (unyt_quantity):
                Wavelength (in um) where the dust optical depth is unity
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
        self.emissivity = emissivity
        self.optically_thin = optically_thin
        self.lam_0 = lam_0

        DustEmission.__init__(
            self,
            intrinsic,
            attenuated,
            scaler,
            do_cmb_heating,
            required_params=("temperature", "emissivity"),
        )

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
            3. From the Greybody object itself

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
            emissivity=self.emissivity,
            redshift=redshift,
        )

        # Compute the greybody function
        lnu = self._lnu(nu, temperature)

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Get the bolometric luminosity with proper units
        bol_lum = sed._bolometric_luminosity

        # Normalise the greybody spectrum
        sed._lnu /= bol_lum
        lnu = sed._lnu

        # Get the scaling we will need
        scaling = self.get_scaling(emitter, model, emissions)

        # Handle per particle scaling (we need to expand the scaling shape)
        if model is not None and model.per_particle:
            if not hasattr(scaling, "shape"):
                # scaling is a float, need to convert to array
                scaling = np.full(emitter.nparticles, scaling)
            scaling = scaling[:, np.newaxis]

        # Properly handle units: normalize then scale
        result = lnu * scaling * cmb_factor
        sed._lnu = result.value if hasattr(result, "value") else result

        return sed

    @accepts(line_lams=angstrom)
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
        emissions,
        spectra,
        redshift=0,
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
            emissions (dict):
                Dictionary containing all emissions generated so far.
            spectra (dict):
                Dictionary containing all spectra generated so far
                (used for scaling).
            redshift (float):
                The redshift at which to calculate the CMB heating. (Ignored
                if not applying CMB heating).

        Returns:
            LineCollection:
                The generated line collection.
        """
        # If we are missing this spectra then we cannot generate lines
        if model.label not in spectra:
            raise exceptions.MissingSpectraType(
                f"Cannot generate lines for {model.label} as spectra are "
                "missing. Please generate spectra first or remove the "
                "generation of lines."
            )
        else:
            sed = spectra[model.label]

        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack the temperature
        temperature = params["temperature"]

        # Define frequencies
        nu = (c / line_lams).to(Hz)
        sed_nu = sed.nu

        # Account for CMB heating
        cmb_factor, temperature = self.apply_cmb_heating(
            temperature=temperature,
            emissivity=self.emissivity,
            redshift=redshift,
        )

        # Compute the greybody function
        lnu = self._lnu(nu, temperature)
        norm_lnu = self._lnu(sed_nu, temperature)

        # Create an SED object for convenience
        norm_sed = Sed(lam=sed.lam, lnu=norm_lnu)

        # Get the bolometric luminosity with proper units
        bol_lum = norm_sed._bolometric_luminosity

        # Normalise the greybody spectrum
        lnu /= bol_lum

        # Normalise the spectrum and apply scaling with proper unit handling
        scaling = self.get_scaling(emitter, model, spectra)

        # Handle per particle scaling (we need to expand the scaling shape)
        if model is not None and model.per_particle:
            if not hasattr(scaling, "shape"):
                # scaling is a float, need to convert to array
                scaling = np.full(emitter.nparticles, scaling)
            scaling = scaling[:, np.newaxis]

        # Properly handle units: normalize then scale
        result = lnu * scaling * cmb_factor
        lnu = result.value if hasattr(result, "value") else result

        # Return as LineCollection with continuum only
        lines = LineCollection(
            line_ids,
            line_lams,
            lum=np.zeros(lnu.shape) * erg / s,
            cont=lnu * erg / s / Hz,
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
