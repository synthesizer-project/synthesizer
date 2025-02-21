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

    def __init__(self, fesc_attrs=("fesc",)):
        """
        Initialise the escape fraction transformer.

        Args:
            fesc_attrs (tuple, optional):
                The attributes to extract from the model to use as the escape
                fraction. If multiple are passed these will be added
                toegether. Default is ("fesc",).
        """
        # Call the parent class constructor and declare we need fesc for this
        # transformer.
        Transformer.__init__(self, required_params=fesc_attrs)

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

        # Ensure the mask is compatible with the emission
        if (
            mask is not None
            and mask.shape != emission.shape[: len(mask.shape)]
        ):
            raise exceptions.InconsistentMultiplication(
                "Mask shape must match emission shape "
                f"(mask.shape: {mask.shape}, "
                f"emission.shape: {emission.shape})."
            )

        # Combine the escape fractions
        fesc = sum([params[attr] for attr in self.required_params])

        return emission.scale(
            1 - fesc,
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

    def __init__(self, fesc_attrs=("fesc",)):
        """
        Initialise the escaped fraction transformer.

        Args:
            fesc_attrs (tuple, optional):
                The attributes to extract from the model to use as the escape
                fraction. If multiple are passed these will be added
                together. Default is ("fesc",).
        """
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

        # Ensure the mask is compatible with the emission
        if (
            mask is not None
            and mask.shape != emission.shape[: len(mask.shape)]
        ):
            raise exceptions.InconsistentMultiplication(
                "Mask shape must match emission shape "
                f"(mask.shape: {mask.shape}, "
                f"emission.shape: {emission.shape})."
            )

        # Combine the escape fractions
        fesc = sum([params[attr] for attr in self.required_params])

        return emission.scale(
            fesc,
            mask=mask,
            lam_mask=lam_mask,
        )


# Define aliases for the AGN covering fraction transformers. These are just
# the same as the escape fraction transformers but with the default attributes
# set to the covering fraction attributes.


class CoveringFraction(EscapedFraction):
    """
    A transformer that applies a covering fraction to an emission.

    This is an alias for the AGN covering fraction which is effectively the
    EscapedFraction transformer (i.e. it transforms disc emission to get the
    emisison covered and trasmitted through a line region).

    This can be thought of as an attenuation law with a single value, reducing
    the emission by a constant factor.

    Note that the inverse (the escaped emission) can be achieved using the
    EscapingFraction transformer. This is effectively identical to this
    transformer but will apply fesc instead of (1 - fesc), as the scaling.

    If fesc is an array then the shape of the emission must match along all
    axes accept the final one for multi-dimensional emissions, 1D emissions
    must match the shape of 1D fesc arrays.
    """

    def __init__(self, fesc_attrs):
        """
        Initialise the covering fraction transformer.

        Args:
            fesc_attrs (tuple, optional):
                The attributes to extract from the model to use as the escape
                fraction. If multiple are passed these will be added
                together. We pass no default here since there is ambiguity.
        """
        # Call the parent class constructor and declare we need fesc for this
        # transformer.
        EscapeFraction.__init__(self, fesc_attrs=fesc_attrs)


class EscapingFraction(EscapeFraction):
    """
    A transformer that applies a covering fraction to an emission.

    This is an alias for the AGN covering fraction which is effectively the
    EscapeFraction transformer (i.e. it transforms disc emission to get the
    emisison not covered by a line region).

    This can be thought of as an attenuation law with a single value, reducing
    the emission by a constant factor.

    Note that the inverse (the escaped emission) can be achieved using the
    EscapingFraction transformer. This is effectively identical to this
    transformer but will apply fesc instead of (1 - fesc), as the scaling.

    If fesc is an array then the shape of the emission must match along all
    axes accept the final one for multi-dimensional emissions, 1D emissions
    must match the shape of 1D fesc arrays.
    """

    def __init__(self, fesc_attrs):
        """
        Initialise the covering fraction transformer.

        Args:
            fesc_attrs (tuple, optional):
                The attributes to extract from the model to use as the escape
                fraction. If multiple are passed these will be added
                together. We pass no default here since there is ambiguity.
        """
        # Call the parent class constructor and declare we need fesc for this
        # transformer.
        EscapeFraction.__init__(self, fesc_attrs=fesc_attrs)
