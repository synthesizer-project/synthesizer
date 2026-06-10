"""A submodule containing photoinisation utilities."""

from dataclasses import dataclass

import numpy as np
from unyt import eV


@dataclass
class Ions:
    """A dataclass holding the ionisation energy of various ions.

    Used for calculating ionising photon production rates (Q)
    for different species.

    Values taken from: \
        https://en.wikipedia.org/wiki/Ionization_energies_of_the_elements_(data_page)
    """

    energy = {
        "H+": 13.6 * eV,
        "He+": 24.6 * eV,
        "He2+": 54.4 * eV,
        "C2+": 24.4 * eV,
        "C3+": 47.9 * eV,
        "C4+": 64.5 * eV,
        "N+": 14.5 * eV,
        "N2+": 29.6 * eV,
        "N3+": 47.4 * eV,
        "O+": 13.6 * eV,
        "O2+": 35.1 * eV,
        "O3+": 54.9 * eV,
        "Ne+": 21.6 * eV,
        "Ne2+": 41.0 * eV,
        "Ne3+": 63.45 * eV,
    }

    label = {
        "H+": r"H$^{+}$",
        "He+": r"He$^{+}$",
        "He2+": r"He$^{2+}$",
        "C2+": r"C$^{2+}$",
        "C3+": r"C$^{3+}$",
        "C4+": r"C$^{4+}$",
        "N+": r"N$^{+}$",
        "N2+": r"N$^{2+}$",
        "N3+": r"N$^{3+}$",
        "O+": r"O$^{+}$",
        "O2+": r"O$^{2+}$",
        "O3+": r"O$^{3+}$",
        "Ne+": r"Ne$^{+}$",
        "Ne2+": r"Ne$^{2+}$",
        "Ne3+": r"Ne$^{3+}$",
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
            Ionising photon production rate (units: s^-1)
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
            Ionising photon production rate (units: s^-1)
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
