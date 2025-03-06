"""A submodule containing utility functions for the emission models."""

import numpy as np

from synthesizer import exceptions
from synthesizer.utils import depluralize, pluralize

_NO_DEFAULT = object()


def get_param(param, model, emission, emitter, default=_NO_DEFAULT):
    """
    Extracts a parameter from the model, emission, and emitter objects.
    
    The function searches for the specified parameter in the model’s fixed parameters,
    then in the emission object's attributes, and finally in the emitter object's
    attributes. If the retrieved value is a string, it is treated as a reference to another
    parameter and is resolved recursively. When the requested parameter name includes
    "log10", the function attempts to fetch the corresponding non-logged parameter and,
    if found, returns its base-10 logarithm. If the parameter remains unfound, the function
    also checks singular and plural forms before either returning a given default or
    raising a MissingAttribute exception.
    
    Args:
        param (str): The name of the parameter to extract.
        model (EmissionModel): The model object with fixed parameters.
        emission (Sed or LineCollection): The emission object possibly containing the parameter.
        emitter (Stars, Gas, or Galaxy): The emitter object possibly containing the parameter.
        default (object, optional): A default value to return if the parameter is not found.
    
    Returns:
        The extracted parameter value, or the default if provided.
    
    Raises:
        MissingAttribute: If the parameter is not found in any object and no default is given.
    """
    # Initialize the value to None
    value = None

    # Are we looking for a logged parameter?
    logged = "log10" in param

    # Check the model's fixed parameters first
    if model is not None and param in model.fixed_parameters:
        value = model.fixed_parameters[param]

    # Check the emission next
    elif emission is not None and hasattr(emission, param):
        value = getattr(emission, param)

    # Finally check the emitter
    elif emitter is not None and hasattr(emitter, param):
        value = getattr(emitter, param)

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
            f"{param} can't be found on the model, emission, or emitter"
        )


def get_params(params, model, emission, emitter):
    """
    Extract a list of parameters from a model, emission, and emitter.

    Missing parameters will return None.

    The priority of extraction is:
        1. Model (EmissionModel)
        2. Emission (Sed/LineCollection)
        3. Emitter (Stars/Gas/Galaxy)

    Args:
        params (list)
            The parameters to extract.
        model (EmissionModel)
            The model object.
        emission (Sed/LineCollection)
            The emission object.
        emitter (Stars/BlackHoles/Gas/Galaxy)
            The emitter object.

    Returns:
        values (dict)
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
            default=None,
        )

    return values
