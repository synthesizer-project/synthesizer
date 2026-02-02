"""Precision utilities for synthesizer.

This module provides utilities to query and work with the compiled
floating-point precision of synthesizer's C extensions.

The precision is determined at compile time via the SINGLE_PRECISION
environment variable:
    - Default (no flag or 0): double precision (float64)
    - SINGLE_PRECISION=1: single precision (float32)

Example usage:
    >>> from synthesizer.utils import precision
    >>> print(precision.get_precision())
    'float64'
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> converted = precision.array_to_precision(arr, copy=True)
"""

from functools import wraps
from typing import Any, Optional

import numpy as np
from unyt import unyt_array, unyt_quantity

from synthesizer.extensions.precision_info import (
    get_float_bytes as _get_float_bytes,
)
from synthesizer.extensions.precision_info import (
    get_precision as _get_precision,
)
from synthesizer.synth_warnings import warn


def get_precision() -> str:
    """Return the compiled floating-point precision.

    Returns:
        str: Either 'float32' or 'float64' depending on how the
            C extensions were compiled.
    """
    # Call the C++ extension function to get the precision
    return _get_precision()


def get_float_bytes() -> int:
    """Return the number of bytes per floating-point value.

    Returns:
        int: Either 4 (single precision) or 8 (double precision).
    """
    # Call the C++ extension function to get the byte size
    return _get_float_bytes()


def get_numpy_dtype() -> np.dtype:
    """Return the numpy dtype matching the compiled floating-point precision.

    This function returns the numpy dtype for the compiled floating-point
    precision. The precision is determined at compile time via the
    SINGLE_PRECISION environment variable:
    - Default (no flag or 0): float64
    - SINGLE_PRECISION=1: float32

    Returns:
        np.dtype: Either np.float32 or np.float64 depending on how the
            C extensions were compiled.

    Examples:
        >>> get_numpy_dtype()  # Returns dtype based on compilation
        dtype('float32')  # if compiled with SINGLE_PRECISION=1
        dtype('float64')  # if compiled without SINGLE_PRECISION
    """
    # What precision are we using?
    precision = get_precision()

    # Return the corresponding numpy dtype
    if precision == "float32":
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def get_integer_dtype() -> np.dtype:
    """Return the integer dtype matching the compiled floating-point precision.

    This function infers the integer precision from the compiled floating-point
    precision, ensuring consistent bit-width across the codebase:
    - If compiled with float32: returns int32
    - If compiled with float64: returns int64

    This approach assumes that integer operations should match the precision
    of floating-point operations. When specific integer precision is needed,
    the dtype should be passed explicitly to decorators.

    Returns:
        np.dtype: Either np.int32 (if float precision is float32) or np.int64
            (if float precision is float64).

    Examples:
        >>> get_integer_dtype()  # Returns dtype based on float
        dtype('int32')   # if compiled with SINGLE_PRECISION=1
        dtype('int64')   # if compiled without SINGLE_PRECISION
    """
    # Get the floating-point precision
    precision = get_precision()

    # Infer integer precision from float precision
    if precision == "float32":
        return np.dtype(np.int32)
    return np.dtype(np.int64)


def get_boolean_dtype() -> np.dtype:
    """Return the boolean dtype (always np.bool_).

    This function is provided for consistency with get_numpy_dtype() and
    get_integer_dtype(). Boolean values are always represented with np.bool_
    regardless of the compiled floating-point precision.

    Returns:
        np.dtype: Always np.bool_.

    Examples:
        >>> get_boolean_dtype()
        dtype('bool')
    """
    return np.dtype(np.bool_)


def ensure_compatible_precision(
    value: Any,
    target_dtype: np.dtype,
) -> None:
    """Check that a value can be safely converted to the target precision.

    This function checks whether converting a value to the target precision
    would cause an overflow. It supports scalars, numpy arrays, unyt_array,
    and unyt_quantity objects, with the latter treating the wrapped numpy
    array.

    Args:
        value: The value to check. Can be a scalar (int, float), numpy array,
            unyt_array, or unyt_quantity.
        target_dtype: Numpy dtype to check compatibility with. If not provided,
            uses the compiled precision.

    Raises:
        OverflowError: If any value would overflow when converted to the
            target precision.
    """
    # Get the info about the target dtype to check overflow limits
    if np.issubdtype(target_dtype, np.floating):
        finfo = np.finfo(target_dtype)
        max_val = finfo.max
        min_val = finfo.min
    elif np.issubdtype(target_dtype, np.integer):
        iinfo = np.iinfo(target_dtype)
        max_val = iinfo.max
        min_val = iinfo.min
    else:
        raise TypeError(
            f"Target dtype {target_dtype} is not a numeric type. "
            "ensure_compatible_precision only supports floating-point and "
            "integer dtypes."
        )

    # Extract the actual data to check
    if isinstance(value, unyt_quantity):
        data_to_check = value.value
    elif isinstance(value, unyt_array):
        data_to_check = value.ndview
    elif isinstance(value, np.ndarray):
        data_to_check = value
    elif isinstance(value, (int, float, np.number)):
        data_to_check = np.asarray(value)
    else:
        # For unsupported types, skip the check
        print(
            f"Skipping precision check for unsupported type {type(value)}. "
            "If you are seeing this message for a numeric type, please report "
            "an issue to the Synthesizer developers.",
        )
        return

    # For floating-point dtypes, only check finite values
    if np.issubdtype(target_dtype, np.floating):
        if np.any(np.isfinite(data_to_check)):
            finite_data = data_to_check[np.isfinite(data_to_check)]
            if len(finite_data) > 0:
                max_input = np.max(np.abs(finite_data))
                if max_input > max_val:
                    raise OverflowError(
                        f"Value {max_input} exceeds the maximum representable "
                        f"value {max_val} for dtype {target_dtype}."
                    )
        else:
            warn(
                "All values are non-finite; skipping overflow check for "
                f"dtype {target_dtype}.",
            )

    # For integer dtypes, check all values
    else:
        max_input = np.max(data_to_check)
        min_input = np.min(data_to_check)
        if max_input > max_val or min_input < min_val:
            raise OverflowError(
                f"Value range [{min_input}, {max_input}] exceeds the "
                f"representable range [{min_val}, {max_val}] for dtype "
                f"{target_dtype}."
            )


def array_to_precision(
    arr: np.ndarray,
    copy: bool = True,
    target_dtype: Optional[np.dtype] = None,
) -> np.ndarray:
    """Convert an array to the compiled or passed dtype.

    When no conversion is needed, the original array is returned.

    Args:
        arr: Input numpy array.
        copy: If True, always return a copy of the array in the correct
            precision. If False, raise a TypeError if the input array does not
            already match the compiled precision.
        target_dtype: Optional numpy dtype to convert to instead of the
            compiled precision. If provided and input array needs conversion,
            a copy must be made (copy=False will raise a TypeError).

    Returns:
        np.ndarray: Array with the compiled precision dtype.

    Raises:
        TypeError: If copy=False and the input dtype doesn't match
            the compiled precision.
    """
    # Ensure input is a numpy array
    arr = np.asanyarray(arr)
    target_dtype = get_numpy_dtype() if target_dtype is None else target_dtype

    # Ensure compatibility before conversion
    ensure_compatible_precision(arr, target_dtype)

    # Ensure the array is C-contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        if not copy and arr.dtype != target_dtype:
            raise TypeError(
                f"Array has dtype {arr.dtype} but expected {target_dtype}."
            )
        return np.ascontiguousarray(arr, dtype=target_dtype)

    # Check if the array already has the correct dtype, if so return it
    if arr.dtype == target_dtype:
        return arr

    # OK, convert the array to the target dtype making a copy and return it
    if not copy:
        raise TypeError(
            f"Array has dtype {arr.dtype} but expected {target_dtype}."
        )
    return arr.astype(target_dtype)


def scalar_to_precision(
    scalar: Any,
    copy: bool = True,
    target_dtype: Optional[np.dtype] = None,
) -> Any:
    """Convert a scalar to the compiled of passed dtype.

    When no conversion is needed, the original scalar is returned.

    Args:
        scalar: Input scalar value (int, float, or numpy scalar).
        copy: If True, always return a value in the correct precision.
            If False, raise a TypeError if the input scalar does not already
            match the compiled precision.
        target_dtype: Optional numpy dtype to convert to instead of the
            compiled precision.

    Returns:
        Scalar value with the compiled precision dtype.

    Raises:
        TypeError: If copy=False and the input dtype doesn't match
            the compiled precision.
        OverflowError: If the scalar would overflow when converted to the
            target precision.
    """
    # Get the target dtype if not provided
    target_dtype = get_numpy_dtype() if target_dtype is None else target_dtype

    # Ensure compatibility before conversion
    ensure_compatible_precision(scalar, target_dtype)

    # Convert to numpy scalar for dtype checking
    scalar_arr = np.asarray(scalar)
    scalar_dtype = scalar_arr.dtype

    # If the scalar already has the correct dtype, return it
    if scalar_dtype == target_dtype:
        return scalar if isinstance(scalar, np.generic) else scalar_arr.item()

    # If copy is False and dtypes don't match, raise an error
    if not copy:
        raise TypeError(
            f"Scalar has dtype {scalar_dtype} but expected {target_dtype}."
        )

    # Convert the scalar to the target dtype
    converted = np.asarray(scalar, dtype=target_dtype)
    return converted.item()


def unyt_array_to_precision(
    arr: unyt_array,
    copy: bool = True,
    target_dtype: Optional[np.dtype] = None,
) -> unyt_array:
    """Convert a unyt_array to the compiled precision dtype.

    Args:
        arr: Input unyt_array.
        copy: If True, always return a copy of the array in the correct
            precision. If False, raise a TypeError if the input array does not
            already match the compiled precision.
        target_dtype: Optional numpy dtype to convert to instead of the
            compiled precision. If provided and input array needs conversion,
            a copy must be made (copy=False will raise a TypeError).

    Returns:
        unyt.unyt_array: unyt_array with the compiled precision dtype.
    """
    # Ensure input is a unyt_array
    if not isinstance(arr, unyt_array):
        raise TypeError("Input must be a unyt.unyt_array")

    # Get the target dtype
    target_dtype = get_numpy_dtype() if target_dtype is None else target_dtype

    # Ensure the array is C-contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        return unyt_array(
            array_to_precision(
                arr.ndview,
                copy=copy,
                target_dtype=target_dtype,
            ),
            arr.units,
        )

    # If the array already has the correct dtype, return it
    if arr.dtype == target_dtype:
        return arr

    # OK, convert the underlying data to the target dtype making a copy
    return unyt_array(
        array_to_precision(
            arr.ndview,
            copy=copy,
            target_dtype=target_dtype,
        ),
        arr.units,
    )


def ensure_arg_precision(
    arg: Any,
    copy: bool = True,
    target_dtype: Optional[np.dtype] = None,
) -> Any:
    """Ensure the argument is in the compiled or passed precision.

    This function checks the type of arg, and if it is a numpy array,
    unyt_array, unyt_quantity, float,

    Args:
        arg: Input argument, if this is an array or float that demands a
            specific precision, it will be converted. Otherwise, it is returned
            unchanged.
        copy: If True, always return a copy of arrays in the correct precision.
            If False, raise a TypeError if the input array does not already
            match the compiled precision.
        target_dtype: Optional numpy dtype to use instead of the compiled
            precision. If provided, for a float argument this will override the
            compiled precision. For any other data type, a dtype must be
            provided to this function.

    Returns:
        Any: Argument converted to the compiled precision if applicable.
    """
    # If the argument is a unyt_array or unyt_quantity, convert it
    if isinstance(arg, (unyt_array, unyt_quantity)):
        return unyt_array_to_precision(
            arg,
            copy=copy,
            target_dtype=target_dtype,
        )

    # If the argument is a numpy float array, convert numeric arrays only
    if isinstance(arg, np.ndarray):
        if np.issubdtype(arg.dtype, np.number):
            return array_to_precision(
                arg,
                copy=copy,
                target_dtype=target_dtype,
            )
        return arg

    # If the argument is a numeric scalar or numpy scalar, convert it
    if isinstance(arg, (int, float, np.number)):
        return scalar_to_precision(arg, copy=copy, target_dtype=target_dtype)

    # Otherwise, return the argument unchanged
    return arg


def accept_precisions(**precisions):
    """Ensure wrapped function arguments have the correct precision.

    This decorator will check the precision of any of the arguments passed to
    the wrapped function with the precisions defined in this decorators kwargs.
    If the precision does not match the compiled precision of the C extensions,
    e.g. float64 (double) by default or float32 (single) if installed with the
    SINGLE_PRECISION flag, then the argument will be converted to match.

    When specifying integer precision, you can use get_integer_dtype() from
    synthesizer.utils.precision to automatically match the compiled float
    precision:

        from synthesizer.utils.precision import (
            get_numpy_dtype,
            get_integer_dtype,
        )

        @accept_precisions(
            values=get_numpy_dtype(),  # Float argument
            count=get_integer_dtype(),  # Integer (auto-matched)
        )

    Args:
        **precisions (dict):
            The keyword arguments defined with this decorator. Each takes the
            form of argument=precision_for_argument. In reality this is a
            dictionary of the form {"variable": np.dtype}.

    Returns:
        function
            The wrapped function.
    """

    def check_precisions(func):
        """Check arguments have correct precision.

        This will check the arguments passed to the wrapped function have
        the correct precision. If the precision does not match the compiled
        precision of the C extensions, e.g. float64 (double) by default or
        float32 (single) if installed with the SINGLE_PRECISION flag, then
        the argument will be converted to match.

        Args:
            func (function): The function to be wrapped.

        Returns:
            function: The wrapped function.
        """
        arg_names = func.__code__.co_varnames

        @wraps(func)
        def wrapped(*args, **kwargs):
            """Handle all the arguments passed to the wrapped function.

            Args:
                *args:
                    The arguments passed to the wrapped function.
                **kwargs:
                    The keyword arguments passed to the wrapped function.

            Returns:
                The result of the wrapped function.
            """
            # Convert the positional arguments to a list (it must be mutable
            # for what comes next)
            args = list(args)

            # Check the positional arguments
            for i, (name, value) in enumerate(zip(arg_names, args)):
                # Get the target dtype for this argument, either the stated
                # precision or None if not specified. In the None case we will
                # use the compiled precision of the C extensions.
                if name not in precisions:
                    target_dtype = None
                else:
                    target_dtype = precisions[name]

                # Ensure the precision of the argument matches the
                # compiled or stated precision of the C extensions. Note that
                # arguments that don't require a conversion are returned
                # unchanged.
                args[i] = ensure_arg_precision(
                    args[i],
                    copy=True,
                    target_dtype=target_dtype,
                )

            # Check the keyword arguments
            for name, value in kwargs.items():
                # Get the target dtype for this argument, either the stated
                # precision or None if not specified. In the None case we will
                # use the compiled precision of the C extensions.
                if name not in precisions:
                    target_dtype = None
                else:
                    target_dtype = precisions[name]

                # Ensure the precision of the argument matches the compiled
                # precision of the C extensions. Note that arguments that don't
                # require a conversion are just returned unchanged.
                kwargs[name] = ensure_arg_precision(
                    kwargs[name],
                    copy=True,
                    target_dtype=target_dtype,
                )

            return func(*args, **kwargs)

        return wrapped

    return check_precisions
