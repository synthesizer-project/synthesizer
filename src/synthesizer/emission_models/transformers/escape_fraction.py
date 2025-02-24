"""A submodule contatining the escape fraction transformers."""

from synthesizer import exceptions
from synthesizer.emission_models.transformers.transformer import Transformer
from synthesizer.synth_warnings import warn


class ProcessedFraction(Transformer):
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
        # Ensure we have been given a list of escape attributes
        if not isinstance(fesc_attrs, (list, tuple)):
            fesc_attrs = (fesc_attrs,)
            warn(
                "The escape attributes must be a list or tuple. Wrapping "
                "the given value in a tuple."
            )

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
        fesc = sum([params[attr] for attr in self._required_params])

        # Ensure the escape fraction is between 0 and 1
        if not 0 <= fesc <= 1:
            raise exceptions.InvalidProcessedFraction(
                f"Escape fraction must be between 0 and 1 (got {fesc})."
            )

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
        # Ensure we have been given a list of escape attributes
        if not isinstance(fesc_attrs, (list, tuple)):
            fesc_attrs = (fesc_attrs,)
            warn(
                "The escape attributes must be a list or tuple. Wrapping "
                "the given value in a tuple."
            )

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
        fesc = sum([params[attr] for attr in self._required_params])

        # Ensure the escape fraction is between 0 and 1
        if not 0 <= fesc <= 1:
            raise exceptions.InvalidProcessedFraction(
                f"Escape fraction must be between 0 and 1 (got {fesc})."
            )

        return emission.scale(
            fesc,
            mask=mask,
            lam_mask=lam_mask,
        )


class CoveringFraction(Transformer):
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

    def __init__(self, covering_attrs):
        """
        Initialise the covering fraction transformer.

        Args:
            covering_attrs (tuple, optional):
                The attributes to extract from the model to use as the escape
                fraction. If multiple are passed these will be added
                together. We pass no default here since there is ambiguity.
        """
        # Ensure we have been given a list of covering attributes
        if not isinstance(covering_attrs, (list, tuple)):
            covering_attrs = (covering_attrs,)
            warn(
                "The covering attributes must be a list or tuple. Wrapping "
                "the given value in a tuple."
            )

        # Call the parent class constructor and declare we need fesc for this
        # transformer.
        Transformer.__init__(self, required_params=covering_attrs)

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
        fcov = sum([params[attr] for attr in self._required_params])

        # Ensure the escape fraction is between 0 and 1
        if not 0 <= fcov <= 1:
            raise exceptions.InvalidProcessedFraction(
                f"Covering fraction must be between 0 and 1 (got {fcov})."
            )

        return emission.scale(
            fcov,
            mask=mask,
            lam_mask=lam_mask,
        )


class EscapingFraction(Transformer):
    """
    A transformer that applies a covering fraction to an emission.

    This is an alias for the AGN covering fraction which is effectively the
    ProcessedFraction transformer (i.e. it transforms disc emission to get the
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

    def __init__(self, covering_attrs):
        """
        Initialise the covering fraction transformer.

        Args:
            covering_attrs (tuple, optional):
                The attributes to extract from the model to use as the escape
                fraction. If multiple are passed these will be added
                together. We pass no default here since there is ambiguity.
        """
        # Ensure we have been given a list of covering attributes
        if not isinstance(covering_attrs, (list, tuple)):
            covering_attrs = (covering_attrs,)
            warn(
                "The covering attributes must be a list or tuple. Wrapping "
                "the given value in a tuple."
            )

        # Call the parent class constructor and declare we need fesc for this
        # transformer.
        Transformer.__init__(self, required_params=covering_attrs)

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
        fcov = sum([params[attr] for attr in self._required_params])

        # Ensure the escape fraction is between 0 and 1
        if not 0 <= fcov <= 1:
            raise exceptions.InvalidProcessedFraction(
                f"Covering fraction must be between 0 and 1 (got {fcov})."
            )

        return emission.scale(
            1 - fcov,
            mask=mask,
            lam_mask=lam_mask,
        )
