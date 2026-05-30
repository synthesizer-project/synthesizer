"""A module containing transformers for applying broadening to spectra.

This module contains classes for applying doppler broadening and thermal
broadening to spectra.
"""

from unyt import amu

from synthesizer import exceptions
from synthesizer.emission_models.transformers.transformer import Transformer
from synthesizer.emissions.sed import Sed
from synthesizer.utils.operation_timers import timed


class DopplerBroadening(Transformer):
    """A transformer that applies Doppler broadening to a Sed."""

    def __init__(self, sigma_v_attr="sigma_v"):
        """Initialise the Doppler broadening transformer.

        Args:
            sigma_v_attr (str):
                The attribute to extract from the model, emission, or emitter
                to use as the velocity dispersion.
        """
        # Attach the required parameter
        self.sigma_v_attr = sigma_v_attr

        # Initialize the base class with the required parameters
        Transformer.__init__(self, required_params=(sigma_v_attr,))

    def __repr__(self):
        """Return a string representation of the object."""
        return f"DopplerBroadening(sigma_v_attr={self.sigma_v_attr})"

    @timed("DopplerBroadening._transform")
    def _transform(
        self,
        emission,
        emitter,
        model,
        mask,
        lam_mask,
        nthreads=1,
    ):
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
            nthreads (int):
                Unused thread-count placeholder passed through the generic
                transformation interface.

        Returns:
            Sed: The broadened emission.
        """
        # Ensure we have an Sed to work on (this transformation is not defined
        # for other emission types)
        if not isinstance(emission, Sed):
            raise exceptions.InconsistentArguments(
                "DopplerBroadening can only be applied to Sed objects."
            )

        # Ensure no wavelength mask is provided, this is unsupported for
        # broadening transformations as it stands
        if lam_mask is not None:
            raise exceptions.UnimplementedFunctionality(
                "Wavelength masks are not supported for Doppler broadening."
            )

        # Get the velocity dispersion we need
        params = self._extract_params(
            model,
            emission,
            emitter,
            preserve_units=True,
        )

        # Unpack the velocity dispersion
        sigma_v = params[self.sigma_v_attr]

        # Apply the Doppler broadening to the emission
        return emission.doppler_broaden(sigma_v, mask=mask)


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
        # Resolve the required parameters, this will always include the
        # temperature, but may also include the mean molecular weight if
        # mu_attr is set
        required_params = (temperature_attr,)
        if mu_attr is not None:
            required_params = (*required_params, mu_attr)

        # Attach the required parameters
        self.temperature_attr = temperature_attr
        self.mu_attr = mu_attr
        self.mu = mu

        # Initialize the base class with the required parameters
        Transformer.__init__(self, required_params=required_params)

    def __repr__(self):
        """Return a string representation of the object."""
        return (
            "ThermalBroadening("
            f"temperature_attr={self.temperature_attr}, "
            f"mu_attr={self.mu_attr}, "
            f"mu={self.mu})"
        )

    @timed("ThermalBroadening._transform")
    def _transform(
        self,
        emission,
        emitter,
        model,
        mask,
        lam_mask,
        nthreads=1,
    ):
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
            nthreads (int):
                Unused thread-count placeholder passed through the generic
                transformation interface.

        Returns:
            Sed: The broadened emission.
        """
        # Ensure we have an Sed to work on (this transformation is not defined
        # for other emission types)
        if not isinstance(emission, Sed):
            raise exceptions.InconsistentArguments(
                "ThermalBroadening can only be applied to Sed objects."
            )

        # Ensure no wavelength mask is provided, this is unsupported for
        # broadening transformations as it stands
        if lam_mask is not None:
            raise exceptions.UnimplementedFunctionality(
                "Wavelength masks are not supported for thermal broadening."
            )

        # Get the temperature and mean molecular weight we need
        params = self._extract_params(
            model,
            emission,
            emitter,
            preserve_units=True,
        )
        # Unpack the temperature and mean molecular weight (if required)
        mu = self.mu if self.mu_attr is None else params[self.mu_attr]
        temperature = params[self.temperature_attr]

        # Apply the thermal broadening to the emission
        return emission.thermally_broaden(
            temperature,
            mu=mu,
            mask=mask,
        )
