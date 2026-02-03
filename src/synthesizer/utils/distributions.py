"""A module containing statistical/physical distribution functions.

This module is separated from util_funcs to avoid cyclic imports with
the units module (which imports from precision, creating an import cycle
when util_funcs imports accepts from units).

Example usage:

    planck(frequency, temperature=10000 * K)
    sigmoid(x, A, a, c, center)
"""

import numpy as np
import unyt.physical_constants as const
from unyt import Hz, K, erg, pc, s

from synthesizer.units import accepts
from synthesizer.utils.precision import get_numpy_dtype


def sigmoid(x, A, a, c, center):
    """Sigmoid function.

    Args:
        x (float): Input value.
        A (float): Amplitude parameter.
        a (float): Slope parameter.
        c (float): Offset parameter.
        center (float): Center of the sigmoid function.

    Returns:
        float: Sigmoid function value.
    """
    return A / (1 + np.exp(-a * (x - center))) + c


@accepts(frequency=Hz, temperature=K)
def planck(frequency, temperature):
    """Compute the planck distribution for a given frequency and temperature.

    This function computes the spectral radiance of a black body at a given
    frequency and temperature using Planck's law. The spectral radiance is
    then converted to spectral luminosity density assuming a luminosity
    distance of 10 pc.

    Parameters:
        frequency (float or unyt_quantity): Frequency of the radiation in Hz.
        temperature (float or unyt_quantity): Temperature in Kelvin.

    Returns:
        unyt_quantity: Spectral luminosity density in erg/s/Hz.
    """
    # Planck's law: B(ν, T) = (2*h*ν^3) / (c^2 * (exp(hν / kT) - 1))
    dtype = np.float64
    frequency = frequency.astype(dtype)
    temperature = temperature.astype(dtype)
    exponent = (const.h * frequency) / (const.kb * temperature)
    exponent_value = exponent.to_value("")
    max_log = np.log(np.finfo(dtype).max)
    exponent_value = np.clip(exponent_value, None, max_log)
    spectral_radiance = (2 * const.h * frequency**3) / (
        const.c**2 * np.expm1(exponent_value)
    )

    # Convert from spectral radiance density to spectral luminosity density,
    # here we'll assume a luminosity distance of 10 pc
    lnu = spectral_radiance * 4 * np.pi * (10 * pc) ** 2

    # Convert the result to erg/s/Hz and return
    lnu = lnu.to(erg / s / Hz)
    return lnu.astype(get_numpy_dtype())
