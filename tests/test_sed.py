"""A test suite for testing the Sed class."""

import numpy as np


def test_sed_empty(empty_sed):
    all_zeros = not np.any(empty_sed.lnu)
    assert all_zeros
