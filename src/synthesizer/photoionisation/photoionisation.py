"""A submodule containing photoinisation utilities."""

from dataclasses import dataclass

import numpy as np
from unyt import eV


@dataclass
class Ions:
    """A dataclass holding the ionisation energy of various ions.

    Used for calculating ionising photon luminosities (Q).

    Values taken from: \
        https://en.wikipedia.org/wiki/Ionization_energies_of_the_elements_(data_page)
    """

    energy = {
        "HI": 13.6 * eV,
        "HeI": 24.6 * eV,
        "HeII": 54.4 * eV,
        "CII": 24.4 * eV,
        "CIII": 47.9 * eV,
        "CIV": 64.5 * eV,
        "NI": 14.5 * eV,
        "NII": 29.6 * eV,
        "NIII": 47.4 * eV,
        "OI": 13.6 * eV,
        "OII": 35.1 * eV,
        "OIII": 54.9 * eV,
        "NeI": 21.6 * eV,
        "NeII": 41.0 * eV,
        "NeIII": 63.45 * eV,
    }


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
