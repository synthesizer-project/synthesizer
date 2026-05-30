"""A test suite for testing the Sed class."""

import numpy as np
from unyt import Hz, angstrom, erg, s

from synthesizer.emissions.sed import Sed


def test_sed_empty(empty_sed):
    """Test the empty SED object."""
    all_zeros = not np.any(empty_sed.lnu)
    assert all_zeros


def test_scale_threaded_row_broadcast_matches_numpy():
    """Threaded row scaling should match NumPy broadcasting."""
    lam = np.linspace(1000, 2000, 4) * angstrom
    lnu = (np.arange(12, dtype=float).reshape(3, 4) + 1.0) * erg / s / Hz
    scaling = np.array([2.0, 3.0, 4.0])

    scaled = Sed(lam=lam, lnu=lnu).scale(scaling, nthreads=2)

    np.testing.assert_allclose(
        scaled.lnu.value,
        lnu.value * scaling[:, None],
    )


def test_scale_threaded_row_broadcast_respects_masks():
    """Threaded row scaling should respect row and wavelength masks."""
    lam = np.linspace(1000, 2000, 4) * angstrom
    lnu = (np.arange(12, dtype=float).reshape(3, 4) + 1.0) * erg / s / Hz
    scaling = np.array([2.0, 3.0, 4.0])
    mask = np.array([True, False, True])
    lam_mask = np.array([False, True, True, False])

    scaled = Sed(lam=lam, lnu=lnu).scale(
        scaling,
        mask=mask,
        lam_mask=lam_mask,
        nthreads=2,
    )

    expected = lnu.value.copy()
    expected[np.ix_(mask, lam_mask)] *= scaling[mask][:, None]
    np.testing.assert_allclose(scaled.lnu.value, expected)
