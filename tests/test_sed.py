import numpy as np
import pytest
from numpy.typing import NDArray
from synthesizer.sed import Sed


@pytest.fixture
def empty_sed() -> Sed:
    """returns an Sed instance"""
    lam: NDArray[np.float64] = np.loadtxt("tests/test_sed/lam.txt")

    return Sed(lam=lam)


def test_sed_empty(empty_sed: Sed) -> None:
    all_zeros: np.bool_ = np.any(empty_sed.lnu != 0.0)
    assert all_zeros
