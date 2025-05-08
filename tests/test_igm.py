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
