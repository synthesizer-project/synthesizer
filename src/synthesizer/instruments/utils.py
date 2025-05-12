"""A submodule for instrument related utility functions.

Example usage:

    # Generate a wavelength array with a constant resolving power of 10000
    lams = generate_wavelength_array(400, 700, 10000)

    # Generate a wavelength array with a variable resolving power
    func = lambda wav: 10000 + (wav - 400) / 300 * 5000
    lams = generate_wavelength_array(400, 700, func)
"""

from typing import Callable, Union

import numpy as np
from unyt import angstrom, unyt_array, unyt_quantity

from synthesizer.units import accepts


@accepts(lam_min=angstrom, lam_max=angstrom)
def get_lams_from_resolving_power(
    lam_min: unyt_quantity,
    lam_max: unyt_quantity,
    resolving_power: Union[float, Callable[[unyt_quantity], float]],
) -> unyt_array:
    """Generate a wavelength array with variable resolving power.

    This function creates an array of wavelengths between `lam_min` and
    `lam_max`, where the spacing between wavelengths is determined by the
    resolving power. The resolving power can be specified as a constant
    or as a function that varies with wavelength.

    Args:
        lam_min (unyt_quantity): Minimum wavelength in nanometers.
        lam_max (unyt_quantity): Maximum wavelength in nanometers.
        resolving_power (float or callable): Resolving power (R = λ / Δλ).
            Can be a constant value or a function that takes a wavelength
            with units and returns a resolving power.

    Returns:
        unyt_array: Array of wavelengths in nanometers.
    """
    wavelengths = [lam_min]
    current_wav = lam_min

    while current_wav < lam_max:
        # Determine resolving power at current wavelength
        _resolving_power = (
            resolving_power(current_wav)
            if callable(resolving_power)
            else resolving_power
        )

        # Calculate wavelength step
        delta_lambda = current_wav / _resolving_power

        # Update current wavelength
        current_wav += delta_lambda

        # Include the current wavelength in the array if it's within bounds
        if current_wav <= lam_max:
            wavelengths.append(current_wav)

    return np.array(wavelengths) * angstrom
