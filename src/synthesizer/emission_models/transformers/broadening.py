"""A module containing transformers for applying broadening to spectra.

This module contains classes for applying doppler broadening and thermal
broadening to spectra.
"""

from unyt import K, amu, km, s

from synthesizer import exceptions
from synthesizer.emission_models.transformers.transformer import Transformer
from synthesizer.emissions.sed import Sed


class DopplerBroadening(Transformer):
    """A transformer that applies Doppler broadening to a Sed."""

    def __init__(
        self,
        sigma_v_attr="sigma_v",
        apply_units=True,
        sigma_v_units=km / s,
    ):
        """Initialise the Doppler broadening transformer.

        Args:
            sigma_v_attr (str):
                The attribute to extract from the model, emission, or emitter
                to use as the velocity dispersion.
            apply_units (bool):
                Whether to apply ``sigma_v_units`` to extracted unitless
                values. Defaults to True.
            sigma_v_units (unyt.Unit):
                The units to apply when the extracted velocity dispersion is
                unitless.
        """
        self.sigma_v_attr = sigma_v_attr
        self.apply_units = apply_units
        self.sigma_v_units = sigma_v_units
        Transformer.__init__(self, required_params=(sigma_v_attr,))

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            "DopplerBroadening("
            f"sigma_v_attr={self.sigma_v_attr}, "
            f"apply_units={self.apply_units}, "
            f"sigma_v_units={self.sigma_v_units})"
        )

    def _transform(self, emission, emitter, model, mask, lam_mask):
        """Apply Doppler broadening to the emission.

        Args:
            emission (Sed):
                The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.
            mask (np.ndarray):
                The mask to apply to the emission.
            lam_mask (np.ndarray):
                The wavelength mask to apply to the emission.

        Returns:
            Sed: The broadened emission.
        """
        if not isinstance(emission, Sed):
            raise exceptions.InconsistentArguments(
                "DopplerBroadening can only be applied to Sed objects."
            )
        if lam_mask is not None:
            raise exceptions.InconsistentArguments(
                "Wavelength masks are not supported for Doppler broadening."
            )

        params = self._extract_params(model, emission, emitter)

        sigma_v = params[self.sigma_v_attr]
        # Parameter extraction normalises fixed parameters for the C backends,
        # which strips unyt units. Reattach the transformer's default units so
        # the unit-checked Sed API still receives a physical velocity.
        if self.apply_units and not hasattr(sigma_v, "units"):
            sigma_v = sigma_v * self.sigma_v_units

        return emission.doppler_broaden(sigma_v, mask=mask)


class ThermalBroadening(Transformer):
    """A transformer that applies thermal broadening to a Sed."""

    def __init__(
        self,
        temperature_attr="temperature",
        mu_attr=None,
        mu=1.0 * amu,
        apply_units=True,
        temperature_units=K,
        mu_units=amu,
    ):
        """Initialise the thermal broadening transformer.

        Args:
            temperature_attr (str):
                The attribute to extract from the model, emission, or emitter
                to use as the temperature.
            mu_attr (str):
                The optional attribute to extract from the model, emission, or
                emitter to use as the mean molecular weight.
            mu (unyt_quantity):
                The mean molecular weight to use when ``mu_attr`` is not set.
            apply_units (bool):
                Whether to apply ``temperature_units`` and ``mu_units`` to
                extracted unitless values. Defaults to True.
            temperature_units (unyt.Unit):
                The units to apply when the extracted temperature is unitless.
            mu_units (unyt.Unit):
                The units to apply when the extracted mean molecular weight is
                unitless.
        """
        required_params = (temperature_attr,)
        if mu_attr is not None:
            required_params = (*required_params, mu_attr)

        self.temperature_attr = temperature_attr
        self.mu_attr = mu_attr
        self.mu = mu
        self.apply_units = apply_units
        self.temperature_units = temperature_units
        self.mu_units = mu_units
        Transformer.__init__(self, required_params=required_params)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            "ThermalBroadening("
            f"temperature_attr={self.temperature_attr}, "
            f"mu_attr={self.mu_attr}, "
            f"mu={self.mu}, "
            f"apply_units={self.apply_units}, "
            f"temperature_units={self.temperature_units}, "
            f"mu_units={self.mu_units})"
        )

    def _transform(self, emission, emitter, model, mask, lam_mask):
        """Apply thermal broadening to the emission.

        Args:
            emission (Sed):
                The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.
            mask (np.ndarray):
                The mask to apply to the emission.
            lam_mask (np.ndarray):
                The wavelength mask to apply to the emission.

        Returns:
            Sed: The broadened emission.
        """
        if not isinstance(emission, Sed):
            raise exceptions.InconsistentArguments(
                "ThermalBroadening can only be applied to Sed objects."
            )
        if lam_mask is not None:
            raise exceptions.InconsistentArguments(
                "Wavelength masks are not supported for thermal broadening."
            )

        params = self._extract_params(model, emission, emitter)
        mu = self.mu if self.mu_attr is None else params[self.mu_attr]
        temperature = params[self.temperature_attr]
        # As above, extracted fixed parameters may be plain ndarrays. Keep the
        # transformer responsible for restoring the physical units expected by
        # Sed.thermally_broaden, while allowing callers to disable this for
        # already-normalised custom inputs.
        if self.apply_units and not hasattr(temperature, "units"):
            temperature = temperature * self.temperature_units
        if self.apply_units and not hasattr(mu, "units"):
            mu = mu * self.mu_units

        return emission.thermally_broaden(
            temperature,
            mu=mu,
            mask=mask,
        )
