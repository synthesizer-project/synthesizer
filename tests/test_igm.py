import numpy as np


def test_I14_name(i14):
    assert isinstance(i14.name, str)


def test_M96_name(m96):
    assert isinstance(m96.name, str)


def test_I14_transmission(i14, lam):
    z = 2.0
    assert isinstance(i14.get_transmission(z, lam), np.ndarray)


def test_M96_transmission(m96, lam):
    z = 2.0
    assert isinstance(m96.get_transmission(z, lam), np.ndarray)
