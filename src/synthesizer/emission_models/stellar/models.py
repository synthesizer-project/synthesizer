"""A submodule containing the definitions of common stellar emission models.

This module contains the definitions of commoon stellar emission models that
can be used "out of the box" to generate spectra from components or as a
foundation to work from when creating more complex models.

Example usage::

    # Create a simple emission model
    model = TotalEmission(
        grid=grid,
        dust_curve=dust_curve,
        dust_emission_model=dust_emission_model,
        fesc=0.0,
    )

    # Generate the spectra
    spectra = stars.get_spectra(model)

"""

import numpy as np
from unyt import Angstrom, km, s
from unyt.exceptions import UnitConversionError

from synthesizer import exceptions
from synthesizer.emission_models.base_model import (
    EmissionModel,
    StellarEmissionModel,
)
from synthesizer.emission_models.models import AttenuatedEmission, DustEmission
from synthesizer.emission_models.transformers import (
    DopplerBroadening,
    EscapedFraction,
    ProcessedFraction,
)
from synthesizer.synth_warnings import warn


def _validate_velocity_dispersion(velocity_dispersion, kwargs):
    """Validate an optional velocity dispersion.

    Args:
        velocity_dispersion (unyt.unyt_quantity):
            Scalar velocity dispersion. A value of zero disables broadening.
        kwargs (dict):
            Additional emission model arguments. These are inspected to reject
            simultaneous particle velocity shifting.

    Returns:
        unyt.unyt_quantity or None:
            The validated velocity dispersion, or ``None`` when broadening is
            disabled.

    Raises:
        InconsistentArguments:
            If the dispersion is not a finite, non-negative scalar with
            velocity units, or if ``vel_shift=True`` is also requested.
    """
    if velocity_dispersion is None:
        return None
    if kwargs.get("vel_shift", False):
        raise exceptions.InconsistentArguments(
            "velocity_dispersion and vel_shift=True cannot be used together."
        )
    try:
        value = velocity_dispersion.to(km / s).value
    except (AttributeError, UnitConversionError) as exc:
        raise exceptions.InconsistentArguments(
            "velocity_dispersion must be a scalar quantity with velocity "
            "units."
        ) from exc
    if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
        raise exceptions.InconsistentArguments(
            "velocity_dispersion must be a finite, non-negative scalar."
        )
    return None if value == 0 else velocity_dispersion


def _broaden_model(model, label, velocity_dispersion, kwargs):
    """Wrap an existing model in a scalar Doppler broadening transformation.

    Args:
        model (StellarEmissionModel):
            The model whose output spectrum will be broadened.
        label (str):
            The label for the broadened output model.
        velocity_dispersion (unyt.unyt_quantity):
            Scalar velocity dispersion. ``None`` or zero disables broadening.
        kwargs (dict):
            Additional keyword arguments for the broadened model.

    Returns:
        EmissionModel:
            The original model when broadening is disabled, otherwise a model
            applying ``DopplerBroadening`` to the original model.

    Raises:
        InconsistentArguments:
            If ``velocity_dispersion`` is invalid or particle velocity shifting
            is also enabled.
    """
    velocity_dispersion = _validate_velocity_dispersion(
        velocity_dispersion, kwargs
    )
    if velocity_dispersion is None:
        return model

    # Keep the requested label on the public output. The unbroadened spectrum
    # remains in the graph for evaluation but is not saved separately.
    predispersion_label = f"{label}_predispersion"
    model._relabel_models({model.label: predispersion_label})
    model.set_save(False)
    # Preserve galaxy-level models when broadening a total containing dust
    # emission; component-only models retain their stellar emitter type.
    model_class = (
        StellarEmissionModel if model.emitter == "stellar" else EmissionModel
    )
    broadened = model_class(
        label=label,
        apply_to=model,
        transformer=DopplerBroadening(
            sigma_v_attr="velocity_dispersion",
        ),
        velocity_dispersion=velocity_dispersion,
        **kwargs,
    )
    # Mark the graph so later recursive velocity-shift configuration can reject
    # this incompatible combination before mutating any models.
    broadened._has_velocity_dispersion = True
    return broadened


def _init_broadened_model(
    instance,
    label,
    velocity_dispersion,
    kwargs,
    model_class=StellarEmissionModel,
    **model_kwargs,
):
    """Initialize an emission model with optional velocity broadening.

    This helper initializes ``instance`` directly so concrete model types are
    preserved. When broadening is enabled, ``model_kwargs`` define a hidden
    unbroadened model and ``instance`` becomes its broadening transformation.

    Args:
        instance (EmissionModel):
            Concrete model instance being initialized.
        label (str):
            Label for the final output model.
        velocity_dispersion (unyt.unyt_quantity):
            Scalar velocity dispersion. ``None`` or zero disables broadening.
        kwargs (dict):
            Additional keyword arguments for the emission model.
        model_class (type):
            Base emission model class used to initialize the operation.
        **model_kwargs:
            Arguments defining the extraction, combination, or transformation
            performed before broadening.

    Raises:
        InconsistentArguments:
            If ``velocity_dispersion`` is invalid or particle velocity shifting
            is also enabled.
    """
    velocity_dispersion = _validate_velocity_dispersion(
        velocity_dispersion, kwargs
    )

    # Without broadening, initialize the concrete class as the requested
    # operation. This retains the original graph shape and model identity.
    if velocity_dispersion is None:
        model_class.__init__(
            instance,
            label=label,
            **model_kwargs,
            **kwargs,
        )
        return

    # Broadening needs the original operation as an input node. Suppress that
    # intermediate spectrum while retaining the requested public output label.
    predispersion_label = f"{label}_predispersion"
    predispersion_kwargs = {**kwargs, **model_kwargs, "save": False}
    predispersion = model_class(
        label=predispersion_label,
        **predispersion_kwargs,
    )
    output_kwargs = dict(kwargs)
    if "emitter" in model_kwargs:
        output_kwargs["emitter"] = model_kwargs["emitter"]
    model_class.__init__(
        instance,
        label=label,
        apply_to=predispersion,
        transformer=DopplerBroadening(
            sigma_v_attr="velocity_dispersion",
        ),
        velocity_dispersion=velocity_dispersion,
        **output_kwargs,
    )
    # Mark the graph so later recursive velocity-shift configuration can reject
    # this incompatible combination before mutating any models.
    instance._has_velocity_dispersion = True


class IncidentEmission(StellarEmissionModel):
    """An emission model that extracts the incident radiation field.

    This defines an extraction of key "incident" from SPS grid. An optional
    scalar ``velocity_dispersion`` broadens the final output spectrum.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(
        self, grid, label="incident", velocity_dispersion=None, **kwargs
    ):
        """Initialise the IncidentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the output spectrum. ``None`` or zero
                disables broadening.
            **kwargs: Additional keyword arguments.
        """
        _init_broadened_model(
            self,
            label,
            velocity_dispersion,
            kwargs,
            grid=grid,
            extract="incident",
        )


class NebularLineEmission(StellarEmissionModel):
    """An emission model for the nebular line emission.

    This defines the luminosity contribution of the lines to the total
    nebular output. An optional scalar ``velocity_dispersion`` broadens the
    final output spectrum.

    This is a child of the EmissionModel class; for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(
        self,
        grid,
        label="nebular_line",
        fesc_ly_alpha="fesc_ly_alpha",
        fesc="fesc",
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the NebularLineEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the output spectrum. ``None`` or zero
                disables broadening.
            **kwargs: Additional keyword arguments.
        """
        # Get the lyman alpha wavelength elements and create a mask for
        # the line
        lyman_alpha_ind = np.argmin(np.abs(grid.lam - 1216.0 * Angstrom))
        lyman_alpha_mask = np.zeros(len(grid.lam), dtype=bool)
        lyman_alpha_mask[lyman_alpha_ind] = True

        # Since the spectra may have been resampled, we may have split the
        # lyman-alpha line into two. Therefore, we need to mask the
        # surrounding elements as well.
        if lyman_alpha_ind > 0:
            lyman_alpha_mask[lyman_alpha_ind - 1] = True
        if lyman_alpha_ind < len(grid.lam) - 1:
            lyman_alpha_mask[lyman_alpha_ind + 1] = True

        # For lyman-alpha, we reduce the overall luminosity by fesc,
        # then the remaining luminosity by fesc_ly_alpha
        lyman_alpha_no_fesc = StellarEmissionModel(
            label="_" + label + "_no_fesc",
            extract="linecont",
            grid=grid,
            save=False,
            **kwargs,
        )

        # We can't define fesc and fesc_ly_alpha in the same model if
        # the user tried tell them they can't
        if "fesc" in kwargs:
            raise exceptions.InconsistentArguments(
                "Cannot define fesc and fesc_ly_alpha in the same model. "
                "Please use another Transformation on NebularLineEmission "
                "to apply your fesc."
            )

        # Apply broadening after the Lyman-alpha escape fraction, ensuring all
        # nebular lines receive the same kinematic treatment.
        _init_broadened_model(
            self,
            label,
            velocity_dispersion,
            kwargs,
            apply_to=lyman_alpha_no_fesc,
            transformer=EscapedFraction(fesc_attrs=("fesc_ly_alpha",)),
            fesc_ly_alpha=fesc_ly_alpha,
            lam_mask=lyman_alpha_mask,
        )


class TransmittedEmissionNoEscaped(StellarEmissionModel):
    """An emission model that extracts the transmitted radiation field.

    This defines an extraction of the key "transmitted" from. SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(self, grid, label="transmitted", **kwargs):
        """Initialise the TransmittedEmissionNoEscaped object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            **kwargs,
        )


class TransmittedEmissionWithEscaped(StellarEmissionModel):
    """An emission model that extracts the transmitted radiation field.

    This defines 3 models:
      - An extraction of the key "transmitted" from SPS grid.
      - A transformed emission, for the transmitted radiation field accounting
        for the escape fraction.
      - A transformed emission, for the escaped radiation field accounting for
        the escape fraction.

    If fesc = 0.0 then there will only be the extraction model.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(
        self,
        grid,
        label="transmitted",
        fesc="fesc",
        related_models=(),
        incident=None,
        escaped_label="escaped",
        **kwargs,
    ):
        """Initialise the TransmittedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            related_models (list): A list of related models to combine with.
                This is used to combine the escaped and transmitted emission
                models.
            incident (EmissionModel): An incident emission model to use, if
                None then one will be created. This is only matters if
                fesc > 0.0, otherwise the incident contribution is 0.0.
            escaped_label (str): The label for the escaped emission model.
            **kwargs: Additional keyword arguments.
        """
        # Define the transmitted extraction model
        full_transmitted = StellarEmissionModel(
            grid=grid,
            label="full_" + label,
            extract="transmitted",
            **kwargs,
        )

        # We need an incident emission model to calculate the escaped if one
        # has not been passed warn the user we will make one
        if incident is None:
            warn(
                "TransmittedEmission requires an incident emission model. "
                f"We'll create one with the label '_{label}_incident'."
                " If you want to use a different incident model, please "
                "pass your own to the incident argument.",
            )
            incident = IncidentEmission(
                grid=grid,
                label=f"_{label}_incident",
                **kwargs,
            )

        # Get the escaped emission
        escaped = StellarEmissionModel(
            label=escaped_label,
            grid=grid,
            apply_to=incident,
            transformer=EscapedFraction(),
            fesc=fesc,
            **kwargs,
        )

        # Combine any extra related_models
        related_models = (escaped,) + tuple(related_models)

        # Get the transmitted emission (accounting for fesc)
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            apply_to=full_transmitted,
            transformer=ProcessedFraction(),
            fesc=fesc,
            related_models=related_models,
            **kwargs,
        )


class TransmittedEmission:
    """An emission model that extracts the transmitted radiation field.

    This is a wrapper around the TransmittedEmissionWithEscaped model
    and the TransmittedEmissionNoEscaped model. It will choose the
    appropriate model based on the inputs. An optional scalar
    ``velocity_dispersion`` broadens the final output spectrum.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.
    """

    def __new__(
        cls,
        grid,
        label="transmitted",
        fesc="fesc",
        incident=None,
        related_models=(),
        escaped_label="escaped",
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise and return the correct TransmittedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission, for no escaped
                emission this can either be set to 0.0 or None.
            incident (EmissionModel): An incident emission model to use, if
                None then one will be created. This is only matters if
                fesc > 0.0, otherwise the incident contribution is 0.0.
            related_models (list): A list of related models to combine with.
                This is used to combine the escaped and transmitted emission
                models.
            escaped_label (str): The label for the escaped emission model
                created if fesc > 0.0.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the output spectrum. ``None`` or zero
                disables broadening.
            **kwargs: Additional keyword arguments.
        """
        # If fesc is None or 0.0 then we only need the transmitted
        # emission without the escaped component.
        if fesc is None or fesc == 0.0:
            model = TransmittedEmissionNoEscaped(
                grid=grid,
                label=label,
                related_models=related_models,
                **kwargs,
            )

        # Otherwise we need the transmitted emission with the escaped emission
        else:
            model = TransmittedEmissionWithEscaped(
                grid=grid,
                label=label,
                fesc=fesc,
                incident=incident,
                related_models=related_models,
                escaped_label=escaped_label,
                **kwargs,
            )
        return _broaden_model(model, label, velocity_dispersion, kwargs)


class NebularContinuumEmission(StellarEmissionModel):
    """An emission model that extracts the nebular continuum emission.

    This defines an extraction of key "nebular_continuum" from. SPS grid. An
    optional scalar ``velocity_dispersion`` broadens the final output spectrum.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        label="nebular_continuum",
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the NebularContinuumEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the output spectrum. ``None`` or zero
                disables broadening.
            **kwargs: Additional keyword arguments.
        """
        _init_broadened_model(
            self,
            label,
            velocity_dispersion,
            kwargs,
            grid=grid,
            extract="nebular_continuum",
        )


class NebularEmission(StellarEmissionModel):
    """An emission model that combines the nebular emissions.

    This defines a combination of the nebular continuum and line emission
    components. An optional scalar ``velocity_dispersion`` broadens the final
    output spectrum.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        label="nebular",
        fesc_ly_alpha="fesc_ly_alpha",
        nebular_line=None,
        nebular_continuum=None,
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the NebularEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            nebular_line (EmissionModel): The nebular line model to use, if
                None then one will be created.
            nebular_continuum (EmissionModel): The nebular continuum model to
                use, if None then one will be created.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the combined nebular spectrum. ``None``
                or zero disables broadening.
            **kwargs: Additional keyword arguments.
        """
        # If we have a Lyman-alpha escape fraction then calculate the
        # updated line emission and combine with the nebular continuum.
        # Make a nebular line model if we need one
        if nebular_line is None:
            warn(
                "NebularEmission requires a nebular line model. "
                f"We'll create one for you with the label '_{label}_line'. "
                "If you want to use a different nebular line model, please "
                "pass your own to the nebular_line argument.",
            )
            nebular_line = NebularLineEmission(
                grid=grid,
                fesc_ly_alpha=fesc_ly_alpha,
                label="_" + label + "_line",
                **kwargs,
            )

        # Make a nebular continuum model if we need one
        if nebular_continuum is None:
            warn(
                "NebularEmission requires a nebular continuum model. "
                "We'll create one for you with the label "
                f"'_{label}_continuum'. If you want to use a "
                "different nebular continuum model, please "
                "pass your own to the nebular_continuum argument.",
            )
            nebular_continuum = NebularContinuumEmission(
                grid=grid,
                label="_" + label + "_continuum",
                **kwargs,
            )

        # Broaden the combined line and continuum spectrum so this operation
        # remains distinct from any component-level broadening requested above.
        _init_broadened_model(
            self,
            label,
            velocity_dispersion,
            kwargs,
            combine=(nebular_line, nebular_continuum),
        )


class ReprocessedEmission(StellarEmissionModel):
    """An emission model that combines the reprocessed emission.

    This defines a combination of the nebular and transmitted components. An
    optional scalar ``velocity_dispersion`` broadens the final output spectrum.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        label="reprocessed",
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        nebular=None,
        transmitted=None,
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the ReprocessedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            nebular (EmissionModel): The nebular model to use, if None then one
                will be created.
            transmitted (EmissionModel): The transmitted model to use, if None
                then one will be created.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the combined reprocessed spectrum.
                ``None`` or zero disables broadening.
            **kwargs: Additional keyword arguments.
        """
        # Make a nebular model if we need one
        if nebular is None:
            warn(
                "ReprocessedEmission requires a nebular model. "
                "We'll create one for you with the "
                f"label '_{label}_nebular'. If you want to use a "
                "different nebular model, please pass your own to the "
                "nebular argument.",
            )
            nebular = NebularEmission(
                grid=grid,
                label="_" + label + "_nebular",
                fesc_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )

        # Make a transmitted model if we need one
        if transmitted is None:
            warn(
                "ReprocessedEmission requires a transmitted model. "
                "We'll create one for you with the label"
                f" '_{label}_transmitted'. If you want to use a "
                "different transmitted model, please pass your own to the "
                "transmitted argument.",
            )
            transmitted = TransmittedEmission(
                grid=grid,
                label="_" + label + "_transmitted",
                fesc=fesc,
                **kwargs,
            )

        # Apply a single dispersion to the complete reprocessed spectrum rather
        # than broadening its transmitted and nebular components independently.
        _init_broadened_model(
            self,
            label,
            velocity_dispersion,
            kwargs,
            grid=grid,
            combine=(nebular, transmitted),
        )


class IntrinsicEmission:
    """An emission model that defines the intrinsic emission.

    This defines a combination of the reprocessed and escaped emission as
    long as we have an escape fraction greater than 0.0. Otherwise, it
    is identical to the reprocessed emission. An optional scalar
    ``velocity_dispersion`` broadens the final output spectrum.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        label="intrinsic",
        fesc_ly_alpha="fesc_ly_alpha",
        fesc="fesc",
        reprocessed=None,
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the IntrinsicEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            reprocessed (EmissionModel): The reprocessed model to use, if None
                then one will be created.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the intrinsic spectrum. ``None`` or zero
                disables broadening.
            **kwargs: Additional keyword arguments.
        """
        # Make a reprocessed model if we need one
        if reprocessed is None:
            warn(
                "IntrinsicEmission requires a reprocessed model. "
                "We'll create one for you with the label"
                f" '_{label}_reprocessed'. If you want to use a "
                "different reprocessed model, please pass your own to the "
                "reprocessed argument.",
            )
            nebular_line = NebularLineEmission(
                grid=grid,
                fesc_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )
            nebular_continuum = NebularContinuumEmission(
                grid=grid,
                **kwargs,
            )
            nebular = NebularEmission(
                grid=grid,
                nebular_line=nebular_line,
                nebular_continuum=nebular_continuum,
                **kwargs,
            )
            incident = IncidentEmission(
                grid=grid,
                **kwargs,
            )
            transmitted = TransmittedEmission(
                grid=grid,
                fesc=fesc,
                incident=incident,
                **kwargs,
            )
            reprocessed = ReprocessedEmission(
                grid=grid,
                label="_" + label + "_reprocessed",
                fesc_ly_alpha=fesc_ly_alpha,
                fesc=fesc,
                nebular=nebular,
                transmitted=transmitted,
                **kwargs,
            )

        # If we have no escaped emission and no fesc then
        # intrinsic = reprocessed
        if fesc == 0.0 or fesc is None:
            warn(
                "IntrinsicEmission is identical to ReprocessedEmission when "
                "fesc is 0.0 or None. We'll return the reprocessed model "
                "instead of creating a new model.",
            )
            return _broaden_model(
                reprocessed, label, velocity_dispersion, kwargs
            )

        # Unpack the escaped emission from the reprocessed model
        escaped = reprocessed["escaped"]

        model = StellarEmissionModel(
            grid=grid,
            label=label,
            combine=(escaped, reprocessed),
            **kwargs,
        )
        return _broaden_model(model, label, velocity_dispersion, kwargs)


class EmergentEmission(StellarEmissionModel):
    """An emission model that defines the emergent emission.

    This defines combination of the attenuated and escaped emission components
    to produce the emergent emission. An optional scalar
    ``velocity_dispersion`` broadens the final output spectrum.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        dust_curve=None,
        apply_to=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label="emergent",
        attenuated=None,
        escaped=None,
        velocity_dispersion=None,
        **kwargs,
    ):
        """Initialise the EmergentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            apply_to (EmissionModel): The model to apply the dust to.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            attenuated (EmissionModel): The attenuated model to use, if None
                then one will be created.
            escaped (EmissionModel): The escaped model to use, if None then one
                will be created.
            velocity_dispersion (unyt.unyt_quantity): Scalar velocity
                dispersion applied to the emergent spectrum. ``None`` or zero
                disables broadening.
            **kwargs: Additional keyword arguments.
        """
        # If apply_to is None then we need to make a model to apply to
        if apply_to is None and attenuated is None:
            warn(
                "EmergentEmission requires an apply_to model. "
                "We'll create one for you with the label "
                f"'_{label}_reprocessed'. If you want to apply dust to a "
                "different model, please pass your own to the "
                "apply_to argument.",
            )
            nebular_line = NebularLineEmission(
                grid=grid,
                fesc_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )
            nebular_continuum = NebularContinuumEmission(
                grid=grid,
                **kwargs,
            )
            nebular = NebularEmission(
                grid=grid,
                nebular_line=nebular_line,
                nebular_continuum=nebular_continuum,
                **kwargs,
            )
            incident = IncidentEmission(
                grid=grid,
                **kwargs,
            )
            transmitted = TransmittedEmission(
                grid=grid,
                fesc=fesc,
                incident=incident,
                **kwargs,
            )
            apply_to = ReprocessedEmission(
                grid=grid,
                label="_" + label + "_reprocessed",
                fesc=fesc,
                fesc_ly_alpha=fesc_ly_alpha,
                nebular=nebular,
                transmitted=transmitted,
                **kwargs,
            )

        # Make an attenuated model if we need one
        if attenuated is None:
            warn(
                "EmergentEmission requires an attenuated model. "
                "We'll create one for you with the label "
                f"'_{label}_attenuated'. If you want to use a "
                "different attenuated model, please pass your own to the "
                "attenuated argument.",
            )
            attenuated = AttenuatedEmission(
                grid=grid,
                label="_" + label + "_attenuated",
                dust_curve=dust_curve,
                apply_to=apply_to,
                emitter="stellar",
                **kwargs,
            )

        # Do we have an escaped model?
        if escaped is None and "escaped" not in attenuated._models:
            raise exceptions.InconsistentArguments(
                "EmergentEmission requires an escaped model. "
                "Please pass your own to the escaped argument."
            )
        elif escaped is None:
            warn(
                "EmergentEmission requires an escaped model. "
                "We'll try to extract one from the attenuated model. "
                "If you want to use a different escaped model, please "
                "pass your own to the escaped argument.",
            )
            escaped = attenuated["escaped"]

        # Combine attenuation and escape before applying any requested bulk
        # broadening to the emergent stellar spectrum.
        _init_broadened_model(
            self,
            label,
            velocity_dispersion,
            kwargs,
            grid=grid,
            combine=(attenuated, escaped),
            fesc=fesc,
        )


class TotalEmissionWithEscapedWithDust(StellarEmissionModel):
    """An emission model that defines total emission with an escape fraction.

    This defines the combination of the emergent and dust emission components
    to produce the total emission.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        dust_curve,
        dust_emission_model,
        label="total",
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        velocity_dispersion_starpop=None,
        velocity_dispersion_total=None,
        **kwargs,
    ):
        """Initialise the TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            velocity_dispersion_starpop (unyt.unyt_quantity): Optional scalar
                internal stellar-population dispersion applied before dust
                attenuation.
            velocity_dispersion_total (unyt.unyt_quantity): Optional scalar
                bulk dispersion applied to the final stellar and thermal-dust
                emission.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            fesc=fesc,
            incident=incident,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            fesc=fesc,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        escaped = transmitted["escaped"]
        emergent = EmergentEmission(
            grid=grid,
            attenuated=attenuated,
            escaped=escaped,
            **kwargs,
        )
        dust_emission_model.set_energy_balance(reprocessed, attenuated)
        dust_emission = DustEmission(
            dust_emission_model=dust_emission_model,
            emitter="stellar",
            **kwargs,
        )

        # Bulk motion applies to the complete system, including its emitting
        # dust. Relative source-dust motion belongs in attenuation instead.
        _init_broadened_model(
            self,
            label,
            velocity_dispersion_total,
            kwargs,
            grid=grid,
            combine=(
                emergent,
                dust_emission,
            ),
        )


class TotalEmissionNoEscapedWithDust(StellarEmissionModel):
    """An emission model that defines total emission.

    This defines the combination of the emergent and dust emission components
    to produce the total emission.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __init__(
        self,
        grid,
        dust_curve,
        dust_emission_model,
        fesc_ly_alpha="fesc_ly_alpha",
        label="total",
        velocity_dispersion_starpop=None,
        velocity_dispersion_total=None,
        **kwargs,
    ):
        """Initialise the TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            velocity_dispersion_starpop (unyt_quantity): Optional scalar
                internal stellar-population dispersion, applied before dust.
            velocity_dispersion_total (unyt_quantity): Optional scalar bulk
                dispersion applied to final stellar and thermal-dust emission.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            incident=incident,
            fesc=0.0,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        dust_emission_model.set_energy_balance(reprocessed, attenuated)
        dust_emission = DustEmission(
            dust_emission_model=dust_emission_model,
            emitter="stellar",
            **kwargs,
        )

        # Bulk motion applies to the complete system, including its emitting
        # dust. Relative source-dust motion belongs in attenuation instead.
        _init_broadened_model(
            self,
            label,
            velocity_dispersion_total,
            kwargs,
            grid=grid,
            combine=(
                attenuated,
                dust_emission,
            ),
        )


class TotalEmissionNoEscapedNoDust:
    """An emission model that defines total emission without dust emission.

    When no escape fraction is applied and no dust emission is included
    the total emission is simply the attenuated emission. This is just a
    helpful wrapper around that case.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        dust_curve,
        label="attenuated",
        fesc_ly_alpha="fesc_ly_alpha",
        velocity_dispersion_starpop=None,
        **kwargs,
    ):
        """Initialise the TotalEmissionNoEscapeNoDust object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            velocity_dispersion_starpop (unyt.unyt_quantity): Optional scalar
                internal stellar-population dispersion applied before dust
                attenuation.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            incident=incident,
            fesc=0.0,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        return attenuated


class TotalEmissionWithEscapedNoDust:
    """An emission model that defines total emission with an escape fraction.

    When there is an escape fraction applied but no dust emission is included
    the total emission is simply the emergent emission. This is just a
    helpful wrapper around that case.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        dust_curve,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label="total",
        velocity_dispersion_starpop=None,
        **kwargs,
    ):
        """Initialise the TotalEmissionWithEscapeNoDust object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            velocity_dispersion_starpop (unyt.unyt_quantity): Optional scalar
                internal stellar-population dispersion applied before dust
                attenuation.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            fesc=fesc,
            incident=incident,
            velocity_dispersion=velocity_dispersion_starpop,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            fesc=fesc,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            **kwargs,
        )
        escaped = transmitted["escaped"]
        emergent = EmergentEmission(
            grid=grid,
            attenuated=attenuated,
            escaped=escaped,
            **kwargs,
        )
        return emergent


class TotalEmission:
    """An emission model that defines the total emission.

    This is a wrapper around the TotalEmissionWithEscape and
    TotalEmissionNoEscape models. It will choose the appropriate model based on
    the inputs.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class .
    """

    def __new__(
        cls,
        grid,
        dust_curve,
        dust_emission_model=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label="total",
        velocity_dispersion_starpop=None,
        velocity_dispersion_total=None,
        **kwargs,
    ):
        """Initialise and return the correct TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            velocity_dispersion_starpop (unyt.unyt_quantity): Optional scalar
                internal stellar-population dispersion applied before dust
                attenuation.
            velocity_dispersion_total (unyt.unyt_quantity): Optional scalar
                bulk dispersion applied to final stellar and thermal-dust
                emission.
            **kwargs: Additional keyword arguments.
        """
        velocity_dispersion_starpop = _validate_velocity_dispersion(
            velocity_dispersion_starpop, kwargs
        )
        velocity_dispersion_total = _validate_velocity_dispersion(
            velocity_dispersion_total, kwargs
        )

        # If fesc is None or 0.0 then we only need the total emission without
        # the escaped component.
        if fesc is None or fesc == 0.0:
            # If we have no dust emission then we can just return the
            # attenuated emission
            if dust_emission_model is None:
                model = TotalEmissionNoEscapedNoDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    label=label,
                    fesc_ly_alpha=fesc_ly_alpha,
                    velocity_dispersion_starpop=velocity_dispersion_starpop,
                    **kwargs,
                )
                return _broaden_model(
                    model, model.label, velocity_dispersion_total, kwargs
                )
            else:
                return TotalEmissionNoEscapedWithDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    dust_emission_model=dust_emission_model,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    velocity_dispersion_starpop=velocity_dispersion_starpop,
                    velocity_dispersion_total=velocity_dispersion_total,
                    **kwargs,
                )

        # Otherwise we need the total emission with the escaped component
        else:
            # If we have no dust emission then we can just return the
            # emergent emission
            if dust_emission_model is None:
                model = TotalEmissionWithEscapedNoDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    velocity_dispersion_starpop=velocity_dispersion_starpop,
                    **kwargs,
                )
                return _broaden_model(
                    model, model.label, velocity_dispersion_total, kwargs
                )
            else:
                # Otherwise we return the total emission with the escaped
                # component
                return TotalEmissionWithEscapedWithDust(
                    grid=grid,
                    dust_curve=dust_curve,
                    dust_emission_model=dust_emission_model,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    velocity_dispersion_starpop=velocity_dispersion_starpop,
                    velocity_dispersion_total=velocity_dispersion_total,
                    **kwargs,
                )
