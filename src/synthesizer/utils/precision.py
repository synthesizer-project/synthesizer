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

import functools
import inspect
import warnings

import numpy as np
from unyt import unyt_array

from synthesizer import exceptions

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


def scalar_like(value, ref):
    """Return a scalar typed to match a reference array's dtype.

    NumPy's weak scalar rule normally keeps a bare Python float from widening
    a reduced precision array, but unyt does not honour it: arithmetic between
    a float32 unyt_array and a Python float still produces float64. Typing the
    scalar against the array keeps such an expression at the precision that
    was asked for.

    Non-floating reference dtypes are left alone, since typing a float against
    an integer array would truncate it.

    Args:
        value (float/int):
            The scalar to type.
        ref (np.ndarray/unyt_array):
            The array whose dtype the scalar should take.

    Returns:
        np.floating/float/int:
            ``value`` as a scalar of ``ref``'s dtype, or unchanged when that
            dtype is not a floating one.

    Raises:
        AttributeError:
            If ``ref`` has no dtype, i.e. is not an array.
    """
    if ref.dtype.kind != "f":
        return value
    return ref.dtype.type(value)


def convert_array_dtype(array, dtype, overflow="raise", name=None):
    """Convert a numeric array-like object to a floating-point dtype.

    This works for NumPy arrays, unyt_arrays (preserving their units), lists
    and scalars, and ensures the result has contiguous storage. If the input
    already has the requested dtype and contiguous storage it is returned
    untouched, otherwise a copy is made.

    Converting to a lower precision can overflow: values beyond the range of
    the target dtype would silently become inf. This is always checked, and
    ``overflow`` controls what happens:

    - ``"raise"`` raises a PrecisionOverflow error.
    - ``"keep"`` returns the input unchanged, at its original precision. Use
      this where a value simply can't be stored at the target precision and
      should stay as it is (e.g. a luminosity in erg/s at float32).

    Only floating-point targets are converted; for any other target dtype
    the input is returned unchanged.

    Args:
        array (array-like/float):
            The input array-like object or scalar.
        dtype (np.dtype/type):
            The target dtype to convert to.
        overflow (str):
            What to do if the values don't fit in the target dtype, either
            "raise" (the default) or "keep".
        name (str, optional):
            The name of the array, for the error message.

    Returns:
        array-like/float:
            The converted array (or scalar), or the input unchanged.

    Raises:
        PrecisionOverflow:
            If the values don't fit in the target dtype and
            ``overflow="raise"``.
        ValueError:
            If the input isn't numeric or ``overflow`` is invalid.
    """
    # Nothing to do if input is None
    if array is None:
        return None

    # Only floating-point targets are converted
    dtype = np.dtype(dtype)
    if dtype.kind != "f":
        return array

    if overflow not in ("raise", "keep"):
        raise ValueError(
            f"overflow must be 'raise' or 'keep' (got {overflow!r})."
        )

    # If the array already has the requested dtype and contiguous storage
    # there is nothing to do; return it untouched to avoid a needless copy.
    if (
        isinstance(array, np.ndarray)
        and array.dtype == dtype
        and array.flags["C_CONTIGUOUS"]
    ):
        return array

    # Work on the raw values (without units)
    values = (
        array.ndview if isinstance(array, unyt_array) else np.asarray(array)
    )

    # Validate it's numeric (integers convert to floats safely)
    if values.dtype.kind not in "iuf":
        raise ValueError(
            f"Unsupported array type or dtype for conversion: "
            f"type(array)={type(array)}, dtype={values.dtype}"
        )

    # Check the (finite) values fit if we are reducing the precision
    if values.dtype.itemsize > dtype.itemsize and values.size > 0:
        largest = float(
            np.max(np.abs(values), initial=0.0, where=np.isfinite(values))
        )
        if largest > float(np.finfo(dtype).max):
            if overflow == "keep":
                return array
            raise exceptions.PrecisionOverflow(
                f"{name or 'An array'} holds values up to {largest:.3g}, "
                f"beyond the {dtype} range (up to {np.finfo(dtype).max:.3g}), "
                f"so converting it to {dtype} would overflow to inf. Keep it "
                f"at {values.dtype}, or use units that bring its values into "
                "range."
            )

    # Convert, reattaching units where we had them (ascontiguousarray would
    # promote a scalar to a 1D array, so scalars are converted directly)
    if values.ndim == 0:
        converted = values.astype(dtype)
    else:
        converted = np.ascontiguousarray(values, dtype=dtype)
    if isinstance(array, unyt_array):
        return unyt_array(converted, array.units, bypass_validation=True)
    if np.ndim(array) == 0 and not isinstance(array, np.ndarray):
        return dtype.type(converted)
    return converted


class InternalPrecisionWarning(RuntimeWarning):
    """Warning for outputs that didn't respect the requested out_dtype.

    This always indicates a bug in Synthesizer rather than a problem with the
    user's inputs. The test suite turns it into an error.
    """


# The private array attributes of Synthesizer's output objects (Sed,
# LineCollection, PhotometryCollection) that verify_out_precision checks
_OUTPUT_ARRAY_ATTRS = (
    "_lnu",
    "_fnu",
    "_luminosity",
    "_continuum",
    "_flux",
    "_continuum_flux",
    "_photometry_data",
    "_photo_lnu",
    "_photo_fnu",
)


def _convert_output_dtype(value, dtype, where):
    """Convert an output to the out_dtype if needed, checking for overflow.

    Args:
        value (object):
            The output: an array, a Synthesizer output object, or a
            dict/list/tuple of these.
        dtype (np.dtype):
            The requested output dtype.
        where (str):
            The name of the function that produced the output.

    Returns:
        object:
            The output, converted where needed.
    """
    # Containers: verify each entry
    if isinstance(value, dict):
        return {
            key: _convert_output_dtype(v, dtype, where)
            for key, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return type(value)(
            _convert_output_dtype(v, dtype, where) for v in value
        )

    # Synthesizer output objects: verify their arrays in place
    if not isinstance(value, np.ndarray):
        for attr in _OUTPUT_ARRAY_ATTRS:
            array = getattr(value, attr, None)
            if isinstance(array, np.ndarray):
                setattr(
                    value,
                    attr,
                    _convert_output_dtype(array, dtype, f"{where} ({attr})"),
                )
        return value

    # Only floating-point arrays carry a precision
    if value.dtype.kind != "f":
        return value

    if value.dtype != dtype:
        warnings.warn(
            f"{where} returned {value.dtype} results although "
            f"out_dtype={dtype} was requested. The results have been "
            "converted, but this is an internal inconsistency in Synthesizer: "
            "please report it at "
            "https://github.com/synthesizer-project/synthesizer/issues.",
            InternalPrecisionWarning,
            stacklevel=4,
        )
        value = convert_array_dtype(value, dtype, name=f"{where} output")

    # Reduced precision outputs can overflow to inf when the values don't fit
    if dtype.itemsize < 8 and np.isinf(value).any():
        raise exceptions.PrecisionOverflow(
            f"{where} produced values too large to be stored at {dtype}, "
            "which overflowed to inf. Use float64 outputs "
            "(out_dtype=np.float64) or units that bring these values into "
            "range."
        )

    return value


def verify_out_precision(*checks):
    """Verify a function's outputs respect its out_dtype argument.

    Decorates any function or method taking an ``out_dtype`` argument. The
    requested dtype is resolved as the function would (None meaning the
    global default), and each checked output is verified:

    - An output at a different precision is an internal inconsistency: it
      is converted, with an InternalPrecisionWarning asking for it to be
      reported (converting raises PrecisionOverflow if the values don't fit).
    - A reduced precision output containing inf has overflowed, which raises
      PrecisionOverflow.

    Outputs can be arrays, Synthesizer output objects (Sed, LineCollection,
    PhotometryCollection) or dicts, lists and tuples of these.

    Args:
        *checks (bool):
            For functions returning a tuple, whether to check each returned
            value (e.g. ``verify_out_precision(True, False)`` checks only the
            first). With no arguments every output is checked.

    Returns:
        callable:
            The decorator.
    """

    def decorator(func):
        signature = inspect.signature(func)
        if "out_dtype" not in signature.parameters:
            raise TypeError(
                f"verify_out_precision can only decorate functions taking an "
                f"out_dtype argument ({func.__qualname__} doesn't)."
            )

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)
            bound = signature.bind(*args, **kwargs)
            dtype = resolve_out_dtype(bound.arguments.get("out_dtype"))
            where = func.__qualname__
            if not checks:
                return _convert_output_dtype(result, dtype, where)
            if not isinstance(result, tuple):
                return (
                    _convert_output_dtype(result, dtype, where)
                    if checks[0]
                    else result
                )
            return (
                tuple(
                    _convert_output_dtype(value, dtype, where)
                    if check
                    else value
                    for value, check in zip(result, checks)
                )
                + result[len(checks) :]
            )

        return wrapped

    return decorator
