"""Test suite for the IGM classes.

TODO: This needs to actually be implemented with proper useful tests.
"""

import numpy as np


def test_I14_name(i14):
    """Test the I14 model name."""
    assert isinstance(i14.name, str)


def test_M96_name(m96):
    """Test the M96 model name."""
    assert isinstance(m96.name, str)


def test_I14_transmission(i14, lam):
    """Test the I14 model transmission."""
    z = 2.0
    assert isinstance(i14.get_transmission(z, lam), np.ndarray)


def test_M96_transmission(m96, lam):
    """Test the M96 model transmission."""
    z = 2.0
    assert isinstance(m96.get_transmission(z, lam), np.ndarray)


def test_A24_name(a24):
    """Test the A24 model name."""
    assert isinstance(a24.name, str)


def test_A24_transmission(a24, lam):
    """Test the A24 model transmission."""
    z = 2.0
    assert isinstance(a24.get_transmission(z, lam), np.ndarray)


def test_A24_transmission_high_z(a24, lam):
    """Test the A24 model transmission at high redshift (z>6)."""
    z = 7.0
    transmission = a24.get_transmission(z, lam)
    assert isinstance(transmission, np.ndarray)
    # At high redshift, transmission should be less than 1
    assert np.all(transmission <= 1.0)
    assert np.all(transmission >= 0.0)


def test_A24_without_cgm(lam):
    """Test A24 model with CGM turned off."""
    from synthesizer.emission_models.attenuation import Asada25

    a24_no_cgm = Asada25(add_cgm=False)
    z = 7.0
    transmission = a24_no_cgm.get_transmission(z, lam)
    assert isinstance(transmission, np.ndarray)
    assert np.all(transmission <= 1.0)
    assert np.all(transmission >= 0.0)


def test_A24_sigmoid_params(lam):
    """Test A24 model with custom sigmoid parameters."""
    from synthesizer.emission_models.attenuation import Asada25

    a24_custom = Asada25(sigmoid_params=(4.0, 2.0, 19.0))
    z = 7.0
    transmission = a24_custom.get_transmission(z, lam)
    assert isinstance(transmission, np.ndarray)
    assert np.all(transmission <= 1.0)
    assert np.all(transmission >= 0.0)


def test_igm_params_from_emitter(i14, lam):
    """Parameters on the emitter take precedence over the IGM's own."""
    from types import SimpleNamespace

    from synthesizer.emission_models.attenuation import Inoue14

    # The emitter's value should be used instead of the default
    emitter = SimpleNamespace(scale_tau=2.0)
    np.testing.assert_allclose(
        i14.get_transmission(3.0, lam, emitter=emitter),
        Inoue14(scale_tau=2.0).get_transmission(3.0, lam),
    )

    # A string value names the attribute to use on the emitter
    alias = Inoue14(scale_tau="igm_scale_tau")
    emitter = SimpleNamespace(igm_scale_tau=2.0)
    np.testing.assert_allclose(
        alias.get_transmission(3.0, lam, emitter=emitter),
        Inoue14(scale_tau=2.0).get_transmission(3.0, lam),
    )


def test_A24_params_from_emitter(a24, lam):
    """Asada25 reads all of its parameters from the emitter."""
    from types import SimpleNamespace

    from synthesizer.emission_models.attenuation import Asada25

    params = {
        "sigmoid_params": (4.0, 2.0, 19.0),
        "scale_tau": 0.5,
        "add_cgm": False,
    }
    np.testing.assert_allclose(
        a24.get_transmission(7.0, lam, emitter=SimpleNamespace(**params)),
        Asada25(**params).get_transmission(7.0, lam),
    )


def test_igm_transform_uses_emitter_params(lam):
    """Applying an IGM via _transform uses the emitter's parameters."""
    from types import SimpleNamespace

    from astropy.cosmology import Planck18
    from unyt import Hz, erg, s

    from synthesizer.emission_models.attenuation import Inoue14
    from synthesizer.emissions import Sed

    # The IGM needs the observed wavelengths and redshift on the Sed
    sed = Sed(lam=lam, lnu=np.ones(len(lam)) * erg / s / Hz)
    sed.get_fnu(Planck18, 3.0)

    emitter = SimpleNamespace(scale_tau=2.0, model_param_cache={})
    model = SimpleNamespace(fixed_parameters={}, label="igm")
    out = Inoue14()._transform(sed, emitter, model)

    np.testing.assert_allclose(
        out.lnu.value,
        Inoue14(scale_tau=2.0).get_transmission(3.0, sed.obslam),
    )
