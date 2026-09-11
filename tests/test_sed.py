"""A test suite for testing the Sed class."""

import numpy as np
from astropy.cosmology import Planck18
from unyt import Hz, angstrom, c, cm, erg, km, nJy, pc, s

from synthesizer.cosmology import get_luminosity_distance
from synthesizer.emission_models.attenuation import PowerLaw
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


def test_apply_attenuation_uses_separable_row_kernel():
    """Row-wise attenuation should match direct transmission broadcasting."""
    lam = np.linspace(1000, 2000, 4) * angstrom
    lnu = (np.arange(12, dtype=float).reshape(3, 4) + 1.0) * erg / s / Hz
    tau_v = np.array([0.1, 0.2, 0.3])
    mask = np.array([True, False, True])
    dust_curve = PowerLaw(slope=-0.7)

    attenuated = Sed(lam=lam, lnu=lnu).apply_attenuation(
        tau_v=tau_v,
        dust_curve=dust_curve,
        mask=mask,
        nthreads=2,
    )

    transmission = dust_curve.get_transmission(tau_v, lam)
    expected = lnu.value.copy()
    expected[mask] *= transmission[mask]
    np.testing.assert_allclose(attenuated.lnu.value, expected)


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


def test_get_fnu_peculiar_velocity_zero_matches_default():
    """peculiar_velocity of None or 0 reproduces the cosmological get_fnu."""
    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = np.linspace(1.0, 8.0, 8) * erg / s / Hz
    z = 1.5

    base = Sed(lam=lam, lnu=lnu).get_fnu(Planck18, z)
    none = Sed(lam=lam, lnu=lnu).get_fnu(Planck18, z, peculiar_velocity=None)
    zero = Sed(lam=lam, lnu=lnu).get_fnu(Planck18, z, peculiar_velocity=0.0)

    np.testing.assert_array_equal(none.value, base.value)
    np.testing.assert_allclose(zero.value, base.value)


def test_get_fnu_peculiar_velocity_shifts_to_observed_redshift():
    """A peculiar velocity shifts to z_obs while the luminosity distance and
    (1+z) factor stay tied to the cosmological z."""
    lam = np.linspace(1000, 2000, 8) * angstrom
    lnu = np.linspace(1.0, 8.0, 8) * erg / s / Hz
    z = 1.0
    v = 600.0  # km/s, receding

    z_obs = (1.0 + z) * (1.0 + v / c.to_value("km/s")) - 1.0
    d_l = get_luminosity_distance(Planck18, z).to(cm)
    d_l_eff = d_l * (1.0 + z_obs) / (1.0 + z)
    expected = lnu * (1.0 + z_obs) / (4 * np.pi * d_l_eff**2)

    sed = Sed(lam=lam, lnu=lnu)
    fnu = sed.get_fnu(Planck18, z, peculiar_velocity=v)

    np.testing.assert_allclose(sed._obslam, sed._lam * (1.0 + z_obs))
    np.testing.assert_allclose(sed._obsnu, sed._nu / (1.0 + z_obs))
    np.testing.assert_allclose(fnu.to("nJy").value, expected.to("nJy").value)

    # A unyt velocity matches the float (km/s) shorthand.
    fnu_unyt = Sed(lam=lam, lnu=lnu).get_fnu(
        Planck18, z, peculiar_velocity=v * km / s
    )
    np.testing.assert_allclose(fnu_unyt.to("nJy").value, fnu.to("nJy").value)
