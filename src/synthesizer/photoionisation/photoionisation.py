"""A submodule containing photoinisation utilities."""

import numpy as np


def calculate_Q_from_U(U_avg, n_h):
    """Calcualte Q for a given U assuming a n_h.

    Args:
        U_avg (float):
            Ionisation parameter
        n_h (float):
            Hyodrogen density (units: cm^-3)

    Returns:
        float:
            Ionising photon luminosity (units: s^-1)
    """
    alpha_B = 2.59e-13  # cm^3 s^-1
    c_cm = 2.99e8 * 100  # cm s^-1
    epsilon = 1.0

    return ((U_avg * c_cm) ** 3 / alpha_B**2) * (
        (4 * np.pi) / (3 * epsilon**2 * n_h)
    )


def calculate_U_from_Q(Q_avg, n_h=100):
    """Calcualte the ionisation parameter for given Q assuming a n_h.

    Args:
        Q_avg (float):
            Ionising photon luminosity (units: s^-1)
        n_h (float):
            Hyodrogen density (units: cm^-3)

    Returns:
        float:
            Ionisation parameter
    """
    alpha_B = 2.59e-13  # cm^3 s^-1
    c_cm = 2.99e8 * 100  # cm s^-1
    epsilon = 1.0

    return ((alpha_B ** (2.0 / 3)) / c_cm) * (
        (3 * Q_avg * (epsilon**2) * n_h) / (4 * np.pi)
    ) ** (1.0 / 3)
