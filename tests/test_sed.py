"""A test suite for testing the Sed class."""

import numpy as np
from astropy.cosmology import Planck18
from unyt import Hz, angstrom, cm, erg, nJy, pc, s

from synthesizer.cosmology import get_luminosity_distance
from synthesizer.emissions import Sed


def test_sed_empty(empty_sed):
    """Test the empty SED object."""
    all_zeros = not np.any(empty_sed.lnu)
    assert all_zeros


def test_get_fnu_matches_expected_observer_frame_conversion():
    """Observed-frame flux conversion should match the old formula."""
    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = np.linspace(1.0, 8.0, 8) * erg / s / Hz

    sed = Sed(lam=lam, lnu=lnu)
    z = 2.0
    old_luminosity_distance = get_luminosity_distance(Planck18, z).to(cm)
    expected_fnu = lnu * (1.0 + z) / (4 * np.pi * old_luminosity_distance**2)

    fnu = sed.get_fnu(Planck18, z)

    assert sed._fnu.shape == sed._lnu.shape
    np.testing.assert_allclose(sed._obslam, sed._lam * (1.0 + z))
    np.testing.assert_allclose(sed._obsnu, sed._nu / (1.0 + z))
    assert fnu.units.same_dimensions_as(nJy)
    np.testing.assert_allclose(fnu.to("nJy").value, sed._fnu)
    np.testing.assert_allclose(
        fnu.to("nJy").value,
        expected_fnu.to("nJy").value,
    )


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
    """Rest-frame flux-at-10pc should match the old formula."""
    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = (np.arange(16, dtype=float).reshape(2, 8) + 1.0) * erg / s / Hz
    expected_fnu = lnu / (4 * np.pi * (10 * pc) ** 2)

    sed = Sed(lam=lam, lnu=lnu)
    fnu = sed.get_fnu0()

    assert sed._fnu.shape == (2, 8)
    assert fnu.shape == (2, 8)
    assert fnu.units.same_dimensions_as(nJy)
    np.testing.assert_allclose(sed._obslam, sed._lam)
    np.testing.assert_allclose(sed._obsnu, sed._nu)
    np.testing.assert_allclose(
        fnu.to("nJy").value,
        expected_fnu.to("nJy").value,
    )


def test_get_fnu0_reuses_final_contiguous_wavelength_buffers():
    """Rest-frame flux conversion should alias the final wavelength buffers."""
    lam = np.linspace(1000, 2000, 16)[::2] * angstrom
    lnu = (np.arange(16, dtype=float).reshape(2, 8) + 1.0) * erg / s / Hz

    sed = Sed(lam=lam, lnu=lnu)
    sed.get_fnu0()

    assert sed._obslam is sed._lam
    assert sed._obsnu is sed._nu
    assert sed._lam.flags.c_contiguous
    assert sed._nu.flags.c_contiguous


def test_get_fnu_applies_igm_with_observer_frame_wavelengths():
    """IGM attenuation should use observer-frame wavelengths once."""

    class DummyIGM:
        def __init__(self):
            self.last_z = None
            self.last_lam_obs = None

        def get_transmission(self, redshift, lam_obs):
            self.last_z = redshift
            self.last_lam_obs = lam_obs
            return np.full(lam_obs.shape, 0.5)

    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = np.linspace(1.0, 8.0, 8) * erg / s / Hz
    z = 2.0

    sed = Sed(lam=lam, lnu=lnu)
    baseline = Sed(lam=lam, lnu=lnu).get_fnu(Planck18, z)
    igm = DummyIGM()

    fnu = sed.get_fnu(Planck18, z, igm=igm)

    assert igm.last_z == z
    np.testing.assert_allclose(igm.last_lam_obs.value, sed._obslam)
    np.testing.assert_allclose(fnu.value, 0.5 * baseline.value)
