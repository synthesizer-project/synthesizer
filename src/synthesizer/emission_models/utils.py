"""A submodule containing utility functions for the emission models."""

import numpy as np

from synthesizer import exceptions
from synthesizer.utils import (
    depluralize,
    ensure_array_c_compatible_double,
    get_attr_c_compatible_double,
    pluralize,
)

_NO_DEFAULT = object()


def get_param(param, model, emission, emitter, default=_NO_DEFAULT):
    """Extract a parameter from a model, emission, and emitter.

    The priority of extraction is:
        1. Model (EmissionModel)
        2. Emission (Sed/LineCollection)
        3. Emitter (Stars/Gas/Galaxy)

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
        default (object, optional):
            The default value to return if the parameter is not found.

    Returns:
        value
            The value of the parameter extracted from the appropriate object.

    Raises:
        MissingAttribute
            If the parameter is not found in the model, emission, or emitter.
            This is only raised if no default is passed.
    """
    # Initialize the value to None
    value = None

    # Are we looking for a logged parameter?
    logged = "log10" in param

    # Check the model's fixed parameters first
    if model is not None and param in model.fixed_parameters:
        value = (
            ensure_array_c_compatible_double(model.fixed_parameters[param])
            if not isinstance(model.fixed_parameters[param], str)
            else model.fixed_parameters[param]
        )

    # Check the emission next
    elif emission is not None and hasattr(emission, param):
        value = get_attr_c_compatible_double(emission, param)

    # Finally check the emitter
    elif emitter is not None and hasattr(emitter, param):
        value = get_attr_c_compatible_double(emitter, param)

    # Do we need to recursively look for the parameter? (We know we're only
    # looking on the emitter at this point)
    if value is not None and isinstance(value, str):
        return get_param(value, None, None, emitter, default=default)
    elif value is not None:
        return value

    # If we were finding a logged parameter but failed, try the non-logged
    # version and log it
    if logged:
        logless_param = param.replace("log10", "")
        value = get_param(
            logless_param,
            model,
            emission,
            emitter,
            default=default,
        )
        if value is not None:
            return np.log10(value)

    # If we got here the parameter is missing, raise an exception or return
    # the default
    if default is not _NO_DEFAULT:
        return default
    else:
        # Before we raise an exception, lets just check we don't have the
        # singular/plural version of the parameter
        singular_param = depluralize(param)
        plural_param = pluralize(param)
        value = get_param(
            singular_param,
            model,
            emission,
            emitter,
            default=None,
        )
        if value is None:
            value = get_param(
                plural_param,
                model,
                emission,
                emitter,
                default=None,
            )
        if value is not None:
            return value

        raise exceptions.MissingAttribute(
            f"{param} can't be found on the model "
            f"({model.label if model is not None else None}),"
            " emission ("
            f"{emission.__class__.__name__ if emission is not None else None}"
            "), or emitter ("
            f"{emitter.__class__.__name__ if emitter is not None else None})."
        )


def get_params(params, model, emission, emitter):
    """Extract a list of parameters from a model, emission, and emitter.

    Missing parameters will return None.

    The priority of extraction is:
        1. Model (EmissionModel)
        2. Emission (Sed/LineCollection)
        3. Emitter (Stars/Gas/Galaxy)

    Args:
        params (list):
            The parameters to extract.
        model (EmissionModel):
            The model object.
        emission (Sed/LineCollection):
            The emission object.
        emitter (Stars/BlackHoles/Gas/Galaxy):
            The emitter object.

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
        )

    return values
