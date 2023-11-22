""" A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
"""
import numpy as np
from unyt import c, h, kb


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


def rebin_1d(a, i, func=np.sum):
    """
    A simple function for rebinning a 1D array using a specificed
    function (e.g. sum or mean).

    TODO: add exeption to make sure a is an 1D array and that i is an integer.

    Args:
        a (ndarray)
            the input 1D array
        i (int)
            integer rebinning factor, i.e. how many bins to rebin by
        func (func)
            the function to use (e.g. mean or sum)


    """

    n = len(a)

    # if array is not the right size truncate it
    if n % i != 0:
        a = a[: int(i * np.floor(n / i))]

    x = len(a) // i
    b = a.reshape(x, i)

    return func(b, axis=1)
