"""A submodule defining Casey12 dust emission generators."""

from typing import Optional

import numpy as np
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
from synthesizer.components.component import Component
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust.dust_emission_base import (
    DustEmission,
)
from synthesizer.emissions import LineCollection, Sed
from synthesizer.units import accepts


class Casey12(DustEmission):
    """A class to generate dust emission spectra using the Casey (2012) model.

    This follows the model from Casey (2012) which combines a mid-infrared
    power law with a far-infrared greybody component.
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.425.3094C/abstract

    This can be used to generate standalone normalised Casey12 spectra, or
    it can be used to generate scaled Casey12 spectra by providing either
    a scaler EmissionModel or an intrinsic/attenuated EmissionModel pair to
    scale the Casey12 spectrum accordingly. For more details see the base
    DustEmission parent class.

    Attributes:
        temperature (unyt_quantity):
            The temperature of the dust.
        temperature_z (unyt_quantity):
            The temperature of the dust at redshift z, accounting for
            CMB heating. Stores the last used temperature (important when
            used with emitter temperatures).
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
        n_pl (unyt_quantity):
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
    n_pl: unyt_quantity

    @accepts(temperature=K, lam_0=um)
    def __init__(
        self,
        temperature: unyt_quantity,
        emissivity: float = 2.0,
        alpha: float = 2.0,
        n_bb: float = 1.0,
        lam_0: unyt_quantity = 200.0 * um,
        do_cmb_heating: bool = False,
        intrinsic: Optional[EmissionModel] = None,
        attenuated: Optional[EmissionModel] = None,
        scaler: Optional[EmissionModel] = None,
    ) -> None:
        """Initialise the Casey12 dust emission model.

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
        self.alpha = alpha
        self.n_bb = n_bb
        self.lam_0 = lam_0

        # Calculate the power law turnover wavelength
        b1 = 26.68
        b2 = 6.246
        b3 = 0.0001905
        b4 = 0.00007243
        lum = (
            (b1 + b2 * alpha) ** -2
            + (b3 + b4 * alpha) * temperature.to("K").value
        ) ** -1

        self.lam_c = (3.0 / 4.0) * lum * um

        # Calculate normalisation of the power-law term
        # See Casey+2012, Table 1 for the expression
        # Missing factors of lam_c and c in some places
        self.n_pl = (
            self.n_bb
            * (1 - np.exp(-((self.lam_0 / self.lam_c) ** emissivity)))
            * (c / self.lam_c) ** 3
            / (np.exp(h * c / (self.lam_c * kb * temperature)) - 1)
        )

        DustEmission.__init__(
            self,
            intrinsic,
            attenuated,
            scaler,
            do_cmb_heating,
            required_params=("temperature",),
        )

    @accepts(nu=Hz)
    def _lnu(self, nu: unyt_array, temperature: unyt_quantity) -> unyt_array:
        """Generate unnormalised spectrum for given frequency (nu) grid.

        Args:
            nu (unyt_array):
                The frequencies at which to calculate the spectral luminosity
                density.
            temperature (unyt_quantity):
                The temperature to use for the Casey12 model.

        Returns:
            lnu (unyt_array):
                The unnormalised spectral luminosity density.
        """
        # Convert to wavelength
        lam = c / nu

        # Define a function to calculate the power-law component.
        def _power_law(lam: unyt_array) -> unyt_array:
            """Calculate the power-law component.

            Args:
                lam (unyt_array):
                    The wavelengths at which to calculate lnu.
            """
            return (
                (
                    self.n_pl
                    * ((lam / self.lam_c) ** self.alpha)
                    * np.exp(-((lam / self.lam_c) ** 2))
                ).value
                * erg
                / s
                / Hz
            )

        def _blackbody(lam: unyt_array) -> unyt_array:
            """Calculate the blackbody component.

            Args:
                lam (unyt_array):
                    The wavelengths at which to calculate lnu.
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

        return _power_law(lam) + _blackbody(lam)

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
            3. From the Casey12 object itself

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

        # Recompute the Casey12 parameters if temperature changed due to CMB
        # heating
        b1 = 26.68
        b2 = 6.246
        b3 = 0.0001905
        b4 = 0.00007243
        lum = (
            (b1 + b2 * self.alpha) ** -2
            + (b3 + b4 * self.alpha) * temperature.to("K").value
        ) ** -1

        lam_c = (3.0 / 4.0) * lum * um

        n_pl = (
            self.n_bb
            * (1 - np.exp(-((self.lam_0 / lam_c) ** self.emissivity)))
            * (c / lam_c) ** 3
            / (np.exp(h * c / (lam_c * kb * temperature)) - 1)
        )

        # Store temporary values for _lnu calculation
        original_lam_c = self.lam_c
        original_n_pl = self.n_pl
        self.lam_c = lam_c
        self.n_pl = n_pl

        # Compute the Casey12 function
        lnu = self._lnu(nu, temperature)

        # Restore original values
        self.lam_c = original_lam_c
        self.n_pl = original_n_pl

        # Create an SED object for convenience
        sed = Sed(lam=lams, lnu=lnu)

        # Get the bolometric luminosity with proper units
        bol_lum = sed._bolometric_luminosity

        # Normalise the Casey12 spectrum
        sed._lnu /= bol_lum
        lnu = sed._lnu

        # Get the scaling we will need
        scaling = self.get_scaling(emitter, model, emissions)

        # Handle per particle scaling (we need to expand the scaling shape)
        if model.per_particle:
            scaling = scaling[:, np.newaxis]

        # Properly handle units: normalize then scale
        sed._lnu = (lnu * scaling * cmb_factor).value

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

        # Recompute the Casey12 parameters if temperature changed due to CMB
        # heating
        b1 = 26.68
        b2 = 6.246
        b3 = 0.0001905
        b4 = 0.00007243
        lum = (
            (b1 + b2 * self.alpha) ** -2
            + (b3 + b4 * self.alpha) * temperature.to("K").value
        ) ** -1

        lam_c = (3.0 / 4.0) * lum * um

        n_pl = (
            self.n_bb
            * (1 - np.exp(-((self.lam_0 / lam_c) ** self.emissivity)))
            * (c / lam_c) ** 3
            / (np.exp(h * c / (lam_c * kb * temperature)) - 1)
        )

        # Store temporary values for _lnu calculation
        original_lam_c = self.lam_c
        original_n_pl = self.n_pl
        self.lam_c = lam_c
        self.n_pl = n_pl

        # Compute the Casey12 function
        lnu = self._lnu(nu, temperature)
        norm_lnu = self._lnu(sed_nu, temperature)

        # Restore original values
        self.lam_c = original_lam_c
        self.n_pl = original_n_pl

        # Create an SED object for convenience
        norm_sed = Sed(lam=sed.lam, lnu=norm_lnu)

        # Get the bolometric luminosity with proper units
        bol_lum = norm_sed._bolometric_luminosity

        # Normalise the Casey12 spectrum
        lnu /= bol_lum

        # Normalise the spectrum and apply scaling with proper unit handling
        scaling = self.get_scaling(emitter, model, spectra)

        # Handle per particle scaling (we need to expand the scaling shape)
        if model.per_particle:
            scaling = scaling[:, np.newaxis]

        # Properly handle units: normalize then scale
        lnu = (lnu * scaling * cmb_factor).value

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
