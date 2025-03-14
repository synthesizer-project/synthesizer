"""A submodule containing utility functions for the emission models."""

from synthesizer import exceptions

_NO_DEFAULT = object()


def get_param(param, model, emission, emitter, default=_NO_DEFAULT):
    """
    Extract a parameter from a model, emission, and emitter.

    The priority of extraction is:
        1. Model (EmissionModel)
        2. Emission (Sed/LineCollection)
        3. Emitter (Stars/Gas/Galaxy)

    If we find a string value this should mean the parameter points to another
    attribute, so we will recursively look for that attribute.

    Args:
        param (str)
            The parameter to extract.
        model (EmissionModel)
            The model object.
        emission (Sed/LineCollection)
            The emission object.
        emitter (Stars/Gas/Galaxy)
            The emitter object.
        default (object, optional)
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

    # If we got here the parameter is missing, raise an exception or return
    # the default
    if default is not _NO_DEFAULT:
        return default
    else:
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
        )

    return values
