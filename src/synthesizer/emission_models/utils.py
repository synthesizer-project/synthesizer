"""A submodule containing utility functions for the emission models."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from synthesizer.components.component import Component
    from synthesizer.emission_models.base_model import EmissionModel

import numpy as np

from synthesizer import exceptions
from synthesizer.utils import (
    depluralize,
    ensure_array_c_compatible_double,
    get_attr_c_compatible_double,
    pluralize,
)

# A sentinel object for detecting if a default value was provided
_NO_DEFAULT = object()


def cache_param(
    param: str,
    emitter: Component,
    model_label: str,
    value: Any,
) -> None:
    """Cache a parameter value on an emitter for a given model.

    This function stores a computed parameter value in the emitter's
    model_param_cache under the specified model label.

    NOTE: This does not duplicate the value, it simply stores a reference to
    it. Under the hood this just increases the reference count by one.

    Args:
        param (str):
            The name of the parameter to cache.
        emitter (Stars/Gas/BlackHoles/Galaxy):
            The emitter object where the parameter will be cached.
        model_label (str):
            The label of the model associated with the parameter.
        value:
            The value to cache for the parameter.
    """
    # Cache the parameter value on the emitter
    emitter.model_param_cache.setdefault(model_label, {})[param] = value


def cache_model_params(
    model: EmissionModel,
    emitter: Component,
) -> None:
    """Cache all model specific parameters on to the emitter.

    This function stores all predefined parameters from the model including:
        - extract: The key that will extracted from the Grid.
        - combine: The models that will be combined to create the emission.
        - apply_to: The label of the model the transformation applies to.
        - transformer: The transformer's repr.
        - generator: The generator's repr.
        - mask parameters: Any parameters used in masking.

    Note that all fixed parameters that are used will automatically be
    associated at the point of use via calls to get_param.

    Args:
        model (EmissionModel):
            The model object containing the parameters to cache.
        emitter (Stars/Gas/BlackHoles/Galaxy):
            The emitter object where the parameters will be cached.
    """
    # Cache the predefined parameters on the emitter based on the type of model
    if model._is_extracting:
        cache_param(
            param="extract",
            emitter=emitter,
            model_label=model.label,
            value=model.extract,
        )
    elif model._is_combining:
        cache_param(
            param="combine",
            emitter=emitter,
            model_label=model.label,
            value=[
                m if isinstance(m, str) else m.label for m in model.combine
            ],
        )
    elif model._is_transforming:
        cache_param(
            param="apply_to",
            emitter=emitter,
            model_label=model.label,
            value=model.apply_to
            if isinstance(model.apply_to, str)
            else model.apply_to.label,
        )
        cache_param(
            param="transformer",
            emitter=emitter,
            model_label=model.label,
            value=repr(model.transformer),
        )
    elif model._is_generating:
        cache_param(
            param="generator",
            emitter=emitter,
            model_label=model.label,
            value=repr(model.generator),
        )

    # Cache any mask parameters in the form of <attr> <op> <thresh> strings
    masks = []
    for mask in model.masks:
        masks.append(f"{mask['attr']} {mask['op']} {mask['thresh']}")
    if len(masks) > 0:
        cache_param(
            param="masks",
            emitter=emitter,
            model_label=model.label,
            value="\n".join(masks),
        )


def get_param(
    param,
    model,
    emission,
    emitter,
    obj=None,
    default=_NO_DEFAULT,
    _singular_attempted=False,
    _plural_attempted=False,
    _visited=None,
):
    """Extract a parameter from a model, emission, emitter, or object.

    The priority of extraction is:
        1. Model (EmissionModel)
        2. Emission (Sed/LineCollection)
        3. Emitter (Stars/Gas/Galaxy)
        4. Object (any object)

    If we find a string value this should mean the parameter points to another
    attribute, so we will recursively look for that attribute.

    Args:
        param (str):
            The parameter to extract.
        model (EmissionModel):
            The model object.
        emission (Sed/LineCollection):
            The emission object.
        emitter (Stars/Gas/Galaxy):
            The emitter object.
        obj (object, optional):
            An optional additional object to look for the parameter on last.
        default (object, optional):
            The default value to return if the parameter is not found.
        _singular_attempted (bool, optional):
            Internal flag to track if singular version has been attempted.
        _plural_attempted (bool, optional):
            Internal flag to track if plural version has been attempted.
        _visited (set, optional):
            Internal set to track visited parameters and detect cycles.

    Returns:
        value
            The value of the parameter extracted from the appropriate object.

    Raises:
        MissingAttribute
            If the parameter is not found in the model, emission, or emitter.
            This is only raised if no default is passed.
    """
    # Initialize the visited set to detect cycles in alias resolution
    if _visited is None:
        _visited = set()

    # Initialize the value to None
    value = None

    # Are we looking for a logged parameter?
    logged = "log10" in param

    # Check the model's fixed parameters first
    if model is not None and param in model.fixed_parameters:
        value = (
            ensure_array_c_compatible_double(model.fixed_parameters[param])
            if (
                not isinstance(
                    model.fixed_parameters[param],
                    str,
                )
            )
            else model.fixed_parameters[param]
        )

    # Check the emission next
    elif emission is not None and hasattr(emission, param):
        value = get_attr_c_compatible_double(emission, param)

    # Check the emitter
    elif emitter is not None and hasattr(emitter, param):
        value = get_attr_c_compatible_double(emitter, param)

    # Finally, if we have an additional object, check that
    elif obj is not None and hasattr(obj, param):
        value = get_attr_c_compatible_double(obj, param)

    # Do we need to recursively look for the parameter? (We know we're only
    # looking on the emitter at this point)
    if value is not None and isinstance(value, str):
        # Check for cycles before following this alias
        if value in _visited:
            # We've seen this alias before - there's a cycle
            if default is not _NO_DEFAULT:
                return default
            raise exceptions.MissingAttribute(
                f"Cyclic alias detected while resolving parameter '{param}'. "
                f"Alias chain: {' -> '.join(_visited)} -> {value}"
            )
        # Add current alias to visited set and follow it
        new_visited = _visited | {param}
        value = get_param(
            value,
            None,
            None,
            emitter,
            default=default,
            _visited=new_visited,
        )

    # If we found a value, return it (early exit chance to avoid extra logic)
    if value is not None:
        # Only cache if we are in a cacheable context (have a model
        # and emitter)
        if model is not None and emitter is not None:
            cache_param(
                param=param,
                emitter=emitter,
                model_label=model.label,
                value=value,
            )
        return value

    # If we were finding a logged parameter but failed, try the non-logged
    # version and log it
    if logged and value is None:
        logless_param = param.replace("log10", "")
        value = get_param(
            logless_param,
            model,
            emission,
            emitter,
            obj,
            default=default,
            _visited=_visited,
        )
        if value is not None:
            # Attach this to the emitter so we don't have to do this again
            value = np.log10(value)
            setattr(emitter, param, value)

    # Lets just check we don't have the singular/plural version of the
    # parameter if we still have nothing
    if value is None and not _singular_attempted:
        singular_param = depluralize(param)
        if singular_param != param:
            value = get_param(
                singular_param,
                model,
                emission,
                emitter,
                obj,
                default=None,
                _singular_attempted=True,
                _plural_attempted=_plural_attempted,
                _visited=_visited,
            )
    if value is None and not _plural_attempted:
        plural_param = pluralize(param)
        if plural_param != param:
            value = get_param(
                plural_param,
                model,
                emission,
                emitter,
                obj,
                default=None,
                _singular_attempted=_singular_attempted,
                _plural_attempted=True,
                _visited=_visited,
            )

    # OK, if we found nothing an have a default, now is the time to use it
    # NOTE: we can't use None as a default value because None could be a valid
    # parameter value, so we use a sentinel object instead.
    if value is None and default is not _NO_DEFAULT:
        value = default

    # If we found a value, return it
    if value is not None:
        # Only cache if we are in a cacheable context (have a model
        # and emitter)
        if model is not None and emitter is not None:
            cache_param(
                param=param,
                emitter=emitter,
                model_label=model.label,
                value=value,
            )
        return value

    # Otherwise raise an exception
    else:
        raise exceptions.MissingAttribute(
            f"{param} can't be found on the model "
            f"({model.label if model is not None else None}),"
            " emission ("
            f"{emission.__class__.__name__ if emission is not None else None}"
            "), or emitter ("
            f"{emitter.__class__.__name__ if emitter is not None else None})."
        )


def get_params(params, model, emission, emitter, obj=None):
    """Extract a list of parameters from a model, emission, emitter, or object.

    The priority of extraction is:
        1. Model (EmissionModel)
        2. Emission (Sed/LineCollection)
        3. Emitter (Stars/Gas/Galaxy)
        4. Object (any object)

    Args:
        params (list):
            The parameters to extract.
        model (EmissionModel):
            The model object.
        emission (Sed/LineCollection):
            The emission object.
        emitter (Stars/BlackHoles/Gas/Galaxy):
            The emitter object.
        obj (object, optional):
            An optional additional object to look for parameters on last.

    Returns:
        values (dict):
            A dictionary of the values of the parameters extracted from the
            appropriate object.
    """
    values = {}
    for param in params:
        values[param] = get_param(
            param,
            model,
            emission,
            emitter,
            obj,
        )

    return values
