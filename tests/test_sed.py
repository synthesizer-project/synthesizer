"""A test suite for testing the Sed class."""

import numpy as np
from astropy.cosmology import Planck18
from unyt import Hz, angstrom, erg, nJy, s

from synthesizer.emissions import Sed


def test_sed_empty(empty_sed):
    """Test the empty SED object."""
    all_zeros = not np.any(empty_sed.lnu)
    assert all_zeros


def test_get_fnu_matches_expected_observer_frame_conversion():
    """Observed-frame flux conversion should preserve values and units."""
    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = np.linspace(1.0, 8.0, 8) * erg / s / Hz

    sed = Sed(lam=lam, lnu=lnu)
    z = 2.0

    fnu = sed.get_fnu(Planck18, z)

    assert sed._fnu.shape == sed._lnu.shape
    np.testing.assert_allclose(sed._obslam, sed._lam * (1.0 + z))
    np.testing.assert_allclose(sed._obsnu, sed._nu / (1.0 + z))
    assert fnu.units.same_dimensions_as(nJy)
    np.testing.assert_allclose(fnu.to("nJy").value, sed._fnu)


def test_get_fnu_handles_multidimensional_spectra():
    """Observer-frame conversion should work on multidimensional spectra."""
    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = (np.arange(24, dtype=float).reshape(3, 8) + 1.0) * erg / s / Hz

    sed = Sed(lam=lam, lnu=lnu)
    fnu = sed.get_fnu(Planck18, 1.5)

    assert sed._fnu.shape == (3, 8)
    assert fnu.shape == (3, 8)
    assert fnu.units.same_dimensions_as(nJy)


def test_get_fnu0_handles_multidimensional_spectra():
    """Rest-frame flux-at-10pc conversion should preserve multidim shape."""
    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = (np.arange(16, dtype=float).reshape(2, 8) + 1.0) * erg / s / Hz

    sed = Sed(lam=lam, lnu=lnu)
    fnu = sed.get_fnu0()

    assert sed._fnu.shape == (2, 8)
    assert fnu.shape == (2, 8)
    assert fnu.units.same_dimensions_as(nJy)
    np.testing.assert_allclose(sed._obslam, sed._lam)
    np.testing.assert_allclose(sed._obsnu, sed._nu)
