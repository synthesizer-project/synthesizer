import pytest
import numpy as np

from synthesizer.igm import Inoue14, Madau96


@pytest.fixture
def i14():
    return Inoue14()


@pytest.fixture
def m96():
    return Madau96()


def test_I14_name(i14):
    assert type(i14.name) is str


def test_M96_name(i14):
    assert type(m96.name) is str


def test_I14_transmission(i14):
    lam = np.loadtxt('tests/test_sed/lam.txt')
    z = 2.
    assert isinstance(i14.T(z, lam), np.ndarray)


def test_M96_transmission(i14):
    lam = np.loadtxt('tests/test_sed/lam.txt')
    z = 2.
    assert isinstance(m96.T(z, lam), np.ndarray)
