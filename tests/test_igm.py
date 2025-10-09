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
    from synthesizer.emission_models.attenuation import Asada24

    a24_no_cgm = Asada24(add_cgm=False)
    z = 7.0
    transmission = a24_no_cgm.get_transmission(z, lam)
    assert isinstance(transmission, np.ndarray)
    assert np.all(transmission <= 1.0)
    assert np.all(transmission >= 0.0)


def test_A24_sigmoid_params(lam):
    """Test A24 model with custom sigmoid parameters."""
    from synthesizer.emission_models.attenuation import Asada24

    a24_custom = Asada24(sigmoid_params=(4.0, 2.0, 19.0))
    z = 7.0
    transmission = a24_custom.get_transmission(z, lam)
    assert isinstance(transmission, np.ndarray)
    assert np.all(transmission <= 1.0)
    assert np.all(transmission >= 0.0)
