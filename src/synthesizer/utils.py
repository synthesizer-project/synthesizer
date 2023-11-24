""" A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
"""
import numpy as np
from unyt import c, h, kb, unyt_array, unyt_quantity


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
