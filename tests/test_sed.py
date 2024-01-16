import pytest
import numpy as np

from synthesizer.sed import Sed


@pytest.fixture
def empty_sed():
    """returns an Sed instance"""
    lam = np.loadtxt("tests/test_sed/lam.txt")

    return Sed(lam=lam)


def test_sed_empty(empty_sed):
    all_zeros = not np.any(empty_sed.lnu)
    assert all_zeros
