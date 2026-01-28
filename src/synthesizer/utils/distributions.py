"""A module containing statistical/physical distribution functions.

This module is separated from util_funcs to avoid cyclic imports with
the units module (which imports from precision, creating an import cycle
when util_funcs imports accepts from units).

Example usage:

    planck(frequency, temperature=10000 * K)
"""

import numpy as np
import unyt.physical_constants as const
from unyt import Hz, K, erg, pc, s

from synthesizer.units import accepts


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
    exponent = (const.h * frequency) / (const.kb * temperature)
    spectral_radiance = (2 * const.h * frequency**3) / (
        const.c**2 * (np.exp(exponent) - 1)
    )

    # Convert from spectral radiance density to spectral luminosity density,
    # here we'll assume a luminosity distance of 10 pc
    lnu = spectral_radiance * 4 * np.pi * (10 * pc) ** 2

    # Convert the result to erg/s/Hz and return
    return lnu.to(erg / s / Hz)
