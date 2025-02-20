"""A submodule contatining the escape fraction transformers."""

from synthesizer import exceptions
from synthesizer.emission_models.transformers.transformer import Transformer


class EscapeFraction(Transformer):
    """
    A transformer that applies an escape fraction to an emission.

    This can be thought of as an attenuation law with a single value, reducing
    the emission by a constant factor.

    Note that the inverse (the escaped emission) can be achieved using the
    EscapedFraction transformer. This is effectively identical to this
    transformer but will apply fesc instead of (1 - fesc), as the scaling.

    If fesc is an array then the shape of the emission must match along all
    axes accept the final one for multi-dimensional emissions, 1D emissions
    must match the shape of 1D fesc arrays.
    """

    def __init__(self):
        """Initialise the escape fraction transformer."""
        # Call the parent class constructor and declare we need fesc for this
        # transformer.
        Transformer.__init__(self, required_params=("fesc",))

    def _transform(self, emission, emitter, model, mask, lam_mask):
        """
        Apply the escape fraction to the emission.

        Args:
            emission (Line/Sed):
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
            Line/Sed: The transformed emission.
        """
        # Extract the required parameters
        params = self._extract_params(model, emission, emitter)

        # Ensure the mask matches the shape of the emission
        if mask is not None and mask.shape != emission.shape:
            raise exceptions.InconsistentMultiplication(
                "Mask shape must match emission shape."
            )

        return emission.scale(
            1 - params["fesc"],
            mask=mask,
            lam_mask=lam_mask,
        )


class EscapedFraction(Transformer):
    """
    A transformer that applies an escaped fraction to an emission.

    This can be thought of as an attenuation law with a single value, reducing
    the emission by a constant factor.

    Note that the inverse (the escaped emission) can be achieved using the
    EscapedFraction transformer. This is effectively identical to this
    transformer but will apply fesc instead of (1 - fesc), as the scaling.

    If fesc is an array then the shape of the emission must match along all
    axes accept the final one for multi-dimensional emissions, 1D emissions
    must match the shape of 1D fesc arrays.
    """

    def __init__(self):
        """Initialise the escape fraction transformer."""
        # Call the parent class constructor and declare we need fesc for this
        # transformer.
        Transformer.__init__(self, required_params=("fesc",))

    def _transform(self, emission, emitter, model, mask, lam_mask):
        """
        Apply the escape fraction to the emission.

        Args:
            emission (Line/Sed):
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
            Line/Sed: The transformed emission.
        """
        # Extract the required parameters
        params = self._extract_params(model, emission, emitter)

        # Ensure the mask matches the shape of the emission
        if mask is not None and mask.shape != emission.shape:
            raise exceptions.InconsistentMultiplication(
                "Mask shape must match emission shape."
            )

        return emission.scale(
            params["fesc"],
            mask=mask,
            lam_mask=lam_mask,
        )
