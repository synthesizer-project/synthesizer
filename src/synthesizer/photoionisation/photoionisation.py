"""A submodule containing photoinisation utilities."""

import numpy as np
from unyt import c, cm, dimensionless, s

from synthesizer.units import accepts


@accepts(n_h=cm**-3)
def calculate_Q_from_U(U_avg, n_h):
    """Calcualte Q for a given U assuming a n_h.

    Args:
        U_avg (float/np.ndarray of float):
            Ionisation parameter
        n_h (unyt_quantity/unyt_array):
            Hyodrogen density

    Returns:
        unyt_quantity/unyt_array:
            Ionising photon luminosity (in s^-1)
    """
    alpha_B = 2.59e-13 * cm**3 / s
    epsilon = 1.0

    return (
        ((U_avg * c) ** 3 / alpha_B**2)
        * ((4 * np.pi) / (3 * epsilon**2 * n_h))
    ).to(1 / s)


@accepts(Q_avg=1 / s, n_h=cm**-3)
def calculate_U_from_Q(Q_avg, n_h=100 * cm**-3):
    """Calcualte the ionisation parameter for given Q assuming a n_h.

    Args:
        Q_avg (unyt_quantity/unyt_array):
            Ionising photon luminosity
        n_h (unyt_quantity/unyt_array):
            Hyodrogen density (defaults to 100 cm^-3)

    Returns:
        unyt_quantity/unyt_array:
            Ionisation parameter (dimensionless)
    """
    alpha_B = 2.59e-13 * cm**3 / s
    epsilon = 1.0

    return (
        ((alpha_B ** (2.0 / 3)) / c)
        * ((3 * Q_avg * (epsilon**2) * n_h) / (4 * np.pi)) ** (1.0 / 3)
    ).to(dimensionless)
