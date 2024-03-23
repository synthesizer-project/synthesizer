import numpy as np
import pytest
from numpy.typing import NDArray
from synthesizer.igm import Inoue14, Madau96


@pytest.fixture
def i14() -> Inoue14:
    return Inoue14()


@pytest.fixture
def m96() -> Madau96:
    return Madau96()


def test_I14_name(i14: Inoue14) -> None:
    assert isinstance(i14.name, str)


def test_M96_name(m96: Madau96) -> None:
    assert isinstance(m96.name, str)


def test_I14_transmission(i14: Inoue14) -> None:
    lam: NDArray[np.float64] = np.loadtxt("tests/test_sed/lam.txt")
    z: float = 2.0
    assert isinstance(i14.T(z, lam), np.ndarray)


def test_M96_transmission(m96: Madau96) -> None:
    lam: NDArray[np.float64] = np.loadtxt("tests/test_sed/lam.txt")
    z: float = 2.0
    assert isinstance(m96.T(z, lam), np.ndarray)
