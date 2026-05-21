"""A module containing transformers for applying broadening to spectra.

This module contains classes for applying doppler broadening and thermal
broadening to spectra.
"""

from unyt import amu

from synthesizer import exceptions
from synthesizer.emission_models.transformers.transformer import Transformer
from synthesizer.emissions.sed import Sed


class DopplerBroadening(Transformer):
    """A transformer that applies Doppler broadening to a Sed."""

    def __init__(self, sigma_v_attr="sigma_v"):
        """Initialise the Doppler broadening transformer.

        Args:
            sigma_v_attr (str):
                The attribute to extract from the model, emission, or emitter
                to use as the velocity dispersion.
        """
        self.sigma_v_attr = sigma_v_attr
        Transformer.__init__(self, required_params=(sigma_v_attr,))

    def __repr__(self):
        """Return a string representation of the object."""
        return f"DopplerBroadening(sigma_v_attr={self.sigma_v_attr})"

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

        return emission.doppler_broaden(
            params[self.sigma_v_attr],
            mask=mask,
        )


class ThermalBroadening(Transformer):
    """A transformer that applies thermal broadening to a Sed."""

    def __init__(
        self,
        temperature_attr="temperature",
        mu_attr=None,
        mu=1.0 * amu,
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
        """
        required_params = (temperature_attr,)
        if mu_attr is not None:
            required_params = (*required_params, mu_attr)

        self.temperature_attr = temperature_attr
        self.mu_attr = mu_attr
        self.mu = mu
        Transformer.__init__(self, required_params=required_params)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            "ThermalBroadening("
            f"temperature_attr={self.temperature_attr}, "
            f"mu_attr={self.mu_attr}, "
            f"mu={self.mu})"
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

        return emission.thermally_broaden(
            params[self.temperature_attr],
            mu=mu,
            mask=mask,
        )
