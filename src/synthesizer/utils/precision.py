"""Precision utilities for synthesizer.

This module provides utilities to query and work with the compiled
floating-point precision of synthesizer's C extensions.

The precision is determined at compile time via the SINGLE_PRECISION
environment variable:
    - Default (no flag or 0): double precision (float64)
    - SINGLE_PRECISION=1: single precision (float32)

Example usage:
    >>> from synthesizer import precision
    >>> print(precision.get_precision())
    'float64'
    >>> arr = np.array([1.0, 2.0, 3.0])
    >>> converted = precision.array_to_precision(arr, copy=True)
"""

import numpy as np

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
    return _get_precision()


def get_float_bytes() -> int:
    """Return the number of bytes per floating-point value.

    Returns:
        int: Either 4 (single precision) or 8 (double precision).
    """
    return _get_float_bytes()


def get_numpy_dtype() -> np.dtype:
    """Return the numpy dtype matching the compiled precision.

    Returns:
        np.dtype: Either np.float32 or np.float64.
    """
    precision = get_precision()
    if precision == "float32":
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def array_to_precision(arr: np.ndarray, *, copy: bool = False) -> np.ndarray:
    """Convert an array to the compiled precision dtype.

    Args:
        arr: Input numpy array.
        copy: If False (default) and the dtype already matches,
            returns the original array (no copy). If False and the
            dtype differs, raises TypeError.
            If True, always returns a converted copy.

    Returns:
        np.ndarray: Array with the compiled precision dtype.

    Raises:
        TypeError: If copy=False and the input dtype doesn't match
            the compiled precision.
    """
    arr = np.asanyarray(arr)
    target_dtype = get_numpy_dtype()

    if arr.dtype == target_dtype:
        if copy:
            return arr.copy()
        return arr

    if copy:
        return arr.astype(target_dtype)

    raise TypeError(
        f"Array has dtype {arr.dtype} but compiled precision is "
        f"{target_dtype}. Use copy=True to convert, or pass an array "
        f"with the correct dtype."
    )
