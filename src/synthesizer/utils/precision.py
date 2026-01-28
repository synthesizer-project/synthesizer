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

from typing import Any

import numpy as np
from unyt import unyt_array, unyt_quantity

from synthesizer.extensions.precision_info import (
    get_float_bytes as _get_float_bytes,
)
from synthesizer.extensions.precision_info import (
    get_precision as _get_precision,
)


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
    """Return the numpy dtype matching the compiled precision.

    Returns:
        np.dtype: Either np.float32 or np.float64.
    """
    # What precision are we using?
    precision = get_precision()

    # Return the corresponding numpy dtype
    if precision == "float32":
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def array_to_precision(arr: np.ndarray) -> np.ndarray:
    """Convert an array to the compiled precision dtype.

    Args:
        arr: Input numpy array.

    Returns:
        np.ndarray: Array with the compiled precision dtype.

    Raises:
        TypeError: If copy=False and the input dtype doesn't match
            the compiled precision.
    """
    # Ensure input is a numpy array
    arr = np.asanyarray(arr)
    target_dtype = get_numpy_dtype()

    # Ensure the array is C-contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(arr, dtype=target_dtype)

    # Check if the array already has the correct dtype, if so return it
    if arr.dtype == target_dtype:
        return arr

    # OK, convert the array to the target dtype making a copy and return it
    return arr.astype(target_dtype)


def unyt_array_to_precision(arr: unyt_array) -> unyt_array:
    """Convert a unyt_array to the compiled precision dtype.

    Args:
        arr: Input unyt_array.

    Returns:
        unyt.unyt_array: unyt_array with the compiled precision dtype.
    """
    # Ensure input is a unyt_array
    if not isinstance(arr, unyt_array):
        raise TypeError("Input must be a unyt.unyt_array")

    # Get the target dtype
    target_dtype = get_numpy_dtype()

    # Ensure the array is C-contiguous
    if not arr.flags["C_CONTIGUOUS"]:
        return unyt_array(
            np.ascontiguousarray(arr.value, dtype=target_dtype),
            arr.units,
        )

    # If the array already has the correct dtype, return it
    if arr.dtype == target_dtype:
        return arr

    # OK, convert the underlying data to the target dtype making a copy
    return unyt_array(
        arr.value.astype(target_dtype),
        arr.units,
    )


def ensure_arg_precision(arg: Any) -> Any:
    """Ensure the argument is in the compiled precision.

    Args:
        arg: Input argument, if this is an array or float that demands a
            specific precision, it will be converted. Otherwise, it is returned
            unchanged.

    Returns:
        Any: Argument converted to the compiled precision if applicable.
    """
    # If the argument is a numpy array, convert it
    if isinstance(arg, np.ndarray):
        return array_to_precision(arg)

    # If the argument is a unyt_array or unyt_quantity, convert it
    if isinstance(arg, (unyt_array, unyt_quantity)):
        return unyt_array_to_precision(arg)

    # If the argument is a float, convert it
    if isinstance(arg, float):
        target_dtype = get_numpy_dtype()
        return target_dtype.type(arg)

    # Otherwise, return the argument unchanged
    return arg


def ensure_extra_attr_precision_and_attach(obj: Any, **kwargs: Any) -> None:
    """Ensure extra attributes are in the compiled precision and attach them.

    This will attach every key/value pair in kwargs as an attribute to obj,
    converting any arrays or floats to the compiled precision.

    Every class which takes optional extra attributes via **kwargs should call
    this function to ensure the attributes are in the correct precision.

    Args:
        obj: The object to which the attributes will be attached.
        **kwargs: Key/value pairs to attach as attributes to obj.

    Returns:
        None: The function modifies obj in place.
    """
    # Iterate over each key/value pair in kwargs
    for key, value in kwargs.items():
        # Ensure the value is in the compiled precision
        converted_value = ensure_arg_precision(value)

        # Attach the converted value as an attribute to obj
        setattr(obj, key, converted_value)
