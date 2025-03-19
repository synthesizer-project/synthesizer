import numpy as np
import pytest

from synthesizer.emission_models.attenuation import Inoue14, Madau96


@pytest.fixture
def i14():
    return Inoue14()


@pytest.fixture
def m96():
    return Madau96()


def test_I14_name(i14):
    assert isinstance(i14.name, str)


def test_M96_name(m96):
    assert isinstance(m96.name, str)


def test_I14_transmission(i14, lam):
    z = 2.0
    assert isinstance(i14.get_transmission(z, lam.value), np.ndarray)


def test_M96_transmission(m96, lam):
    z = 2.0
    assert isinstance(m96.get_transmission(z, lam.value), np.ndarray)
