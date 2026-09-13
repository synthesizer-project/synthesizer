"""A module for controlling the default output precision.

Synthesizer computes spectra, lines, photometry and other outputs at a
user-controllable floating-point precision. Every function that produces a
floating-point output takes an ``out_dtype`` argument; when this is left
unset (``None``) the global default defined here is used.

The global default is float64. To halve the memory footprint of outputs
across the board, set the default to float32 once at the top of a script:

    import numpy as np
    from synthesizer import set_default_out_dtype

    set_default_out_dtype(np.float32)

Individual calls can always override the global default by passing
``out_dtype`` explicitly.

Note that ``out_dtype`` only controls output arrays. Input arrays (particle
data and grids) are used at whatever precision they are provided at and are
never cast or copied behind the scenes; see the precision documentation for
the rules on mixing input precisions.
"""

import numpy as np

# The allowed floating point dtypes for outputs.
_ALLOWED_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

# The global default output dtype. Module level state, modified only via
# set_default_out_dtype.
_default_out_dtype = np.dtype(np.float64)


def _validate_dtype(dtype):
    """Validate and normalise a requested output dtype.

    Args:
        dtype (np.dtype/type):
            The requested dtype.

    Returns:
        np.dtype:
            The normalised dtype.

    Raises:
        ValueError:
            If the dtype is not float32 or float64.
    """
    resolved = np.dtype(dtype)
    if resolved not in _ALLOWED_DTYPES:
        raise ValueError(
            f"Unsupported output dtype ({resolved}). Synthesizer outputs "
            "must be float32 or float64."
        )
    return resolved


def set_default_out_dtype(dtype):
    """Set the global default output dtype.

    Args:
        dtype (np.dtype/type):
            The dtype to use for all outputs when out_dtype is not passed
            explicitly. Must be float32 or float64.
    """
    global _default_out_dtype
    _default_out_dtype = _validate_dtype(dtype)


def get_default_out_dtype():
    """Return the global default output dtype.

    Returns:
        np.dtype:
            The current global default output dtype.
    """
    return _default_out_dtype


def resolve_out_dtype(out_dtype):
    """Resolve a function's out_dtype argument to a concrete dtype.

    Args:
        out_dtype (np.dtype/type/None):
            The out_dtype argument as passed by the caller. None means
            "use the global default".

    Returns:
        np.dtype:
            The dtype outputs should be allocated with.
    """
    if out_dtype is None:
        return _default_out_dtype
    return _validate_dtype(out_dtype)
