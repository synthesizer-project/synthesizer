""" A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
"""
import numpy as np
from unyt import c, h, kb, unyt_array, unyt_quantity


from synthesizer import exceptions


def planck(nu, temperature):
    """
    Planck's law.

    Args:
        nu (unyt_array/array-like, float)
            The frequencies at which to calculate the distribution.
        temperature  (float/array-like, float)
            The dust temperature. Either a single value or the same size
            as nu.

    Returns:
        array-like, float
            The values of the distribution at nu.
    """

    return (2.0 * h * (nu**3) * (c**-2)) * (
        1.0 / (np.exp(h * nu / (kb * temperature)) - 1.0)
    )


def has_units(x):
    """
    Check whether the passed variable has units, i.e. is a unyt_quanity or
    unyt_array.

    Args:
        x (generic variable)
            The variables to check.

    Returns:
        bool
            True if the variable has units, False otherwise.
    """

    # Do the check
    if isinstance(x, (unyt_array, unyt_quantity)):
        return True

    return False


def rebin_1d(arr, resample_factor, func=np.sum):
    """
    A simple function for rebinning a 1D array using a specificed
    function (e.g. sum or mean).

    Args:
        arr (array-like)
            The input 1D array.
        resample_factor (int)
            The integer rebinning factor, i.e. how many bins to rebin by.
        func (func)
            The function to use (e.g. mean or sum).

    Returns:
        array-like
            The input array resampled by i.
    """

    # Ensure the array is 1D
    if arr.ndim != 1:
        raise exceptions.InconsistentArguments(
            f"Input array must be 1D (input was {arr.ndim}D)"
        )

    # Safely handle no integer resamples
    if not isinstance(resample_factor, int):
        print(
            f"resample factor ({resample_factor}) is not an"
            " integer, converting it to ",
            end="\r",
        )
        resample_factor = int(resample_factor)
        print(resample_factor)

    # How many elements in the input?
    n = len(arr)

    # If array is not the right size truncate it
    if n % resample_factor != 0:
        arr = arr[: int(resample_factor * np.floor(n / resample_factor))]

    # Set up the 2D array ready to have func applied
    rows = len(arr) // resample_factor
    brr = arr.reshape(rows, resample_factor)

    return func(brr, axis=1)


def value_to_array(value):
    """
    A helper functions for converting a single value to an array holding
    a single value.

    Args:
        value (float/unyt_quantity)
            The value to wrapped into an array.

    Returns:
        array-like/unyt_array
            An array containing the single value

    Raises:
        InconsistentArguments
            If the argument is not a float or unyt_quantity.
    """

    # Just return it if we have been handed an array already or None
    if isinstance(value, (np.ndarray, unyt_array)) or value is None:
        return value

    if isinstance(value, float):
        arr = np.array(
            [
                value,
            ]
        )
    elif isinstance(value, unyt_quantity):
        arr = (
            np.array(
                [
                    value.value,
                ]
            )
            * value.units
        )
    else:
        raise exceptions.InconsistentArguments(
            "Value to convert to an array wasn't a float or a unyt_quantity:"
            f"type(value) = {type(value)}"
        )

    return arr
