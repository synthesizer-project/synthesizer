"""A submodule containing photoinisation utilities."""

from dataclasses import dataclass

import numpy as np
from unyt import angstrom, c, eV, h


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


def add_ionisation_edges(
    lam,
    lnu,
    ions=("HI", "HeI", "HeII"),
    rel_offset=1e-4,
):
    """Insert points either side of ionisation edges in a tabulated SED.

    Photoionisation codes interpolate a tabulated input SED between its
    points. If the wavelength grid has no points close to an ionisation
    edge (e.g. a grid with points at 905 A and 915 A around the 912 A
    Lyman edge), the interpolation smears the edge: points just blueward of
    it get a flux far closer to the much brighter redward side. The ionising
    photon rate the code sees is then too high, and anything normalised to
    it (such as the grid spectra in a Cloudy grid normalised by Q) ends up
    wrong.

    This inserts a pair of points at ``lam_edge * (1 - rel_offset)`` and
    ``lam_edge * (1 + rel_offset)`` around each edge. The blueward point
    takes the value of the nearest point blueward of the edge, and the
    redward point the value of the nearest point redward of it, so the
    break stays sharp at the edge. The blueward value follows the same
    convention as ``Sed.calculate_ionising_photon_production_rate``, which
    extends the last point below the edge up to the edge. Original points
    are left unchanged.

    Args:
        lam (np.ndarray or unyt_array):
            Wavelengths (Angstrom if no units), any order.
        lnu (np.ndarray or unyt_array):
            Spectral luminosity density at ``lam``. Units are kept.
        ions (tuple of str):
            Keys of ``Ions.energy`` whose edges to sharpen.
        rel_offset (float):
            Fractional offset of the inserted points from each edge.

    Returns:
        tuple:
            (lam, lnu) sorted by increasing wavelength, with the extra
            points included, in the same units as the input. New arrays are
            returned; the inputs are not modified.
    """
    lam_units = getattr(lam, "units", None)
    lnu_units = getattr(lnu, "units", None)
    lam_a = (
        np.asarray(lam.to("angstrom").value, dtype=float)
        if lam_units is not None
        else np.asarray(lam, dtype=float)
    )
    lnu_a = np.asarray(
        lnu.value if lnu_units is not None else lnu, dtype=float
    )

    order = np.argsort(lam_a)
    lam_a, lnu_a = lam_a[order], lnu_a[order]

    new_lam, new_lnu = [], []
    for ion in ions:
        edge = float((h * c / Ions.energy[ion]).to("angstrom").value)
        blue, red = edge * (1 - rel_offset), edge * (1 + rel_offset)

        # Only edges strictly inside the wavelength range can be sharpened
        if not (lam_a[0] < blue and red < lam_a[-1]):
            continue

        # Nearest points blueward and redward of the edge
        i_blue = np.searchsorted(lam_a, edge, side="left") - 1
        i_red = np.searchsorted(lam_a, edge, side="right")

        # Skip edges that already have points within the offset window
        if lam_a[i_blue] >= blue and lam_a[i_red] <= red:
            continue

        new_lam += [blue, red]
        new_lnu += [lnu_a[i_blue], lnu_a[i_red]]

    if new_lam:
        lam_a = np.concatenate([lam_a, new_lam])
        lnu_a = np.concatenate([lnu_a, new_lnu])
        order = np.argsort(lam_a)
        lam_a, lnu_a = lam_a[order], lnu_a[order]

    lam_out = lam_a * angstrom if lam_units is not None else lam_a
    if lam_units is not None:
        lam_out = lam_out.to(lam_units)
    lnu_out = lnu_a * lnu_units if lnu_units is not None else lnu_a
    return lam_out, lnu_out


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
