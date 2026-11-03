"""Tests for the load_data binning helpers."""

import numpy as np

from synthesizer.load_data.utils import split_age_bins

GRID = 10 ** np.arange(6.0, 8.1, 0.5)  # 1e6, 3.16e6, 1e7, 3.16e7, 1e8 yr


def test_split_narrow_bin_unchanged():
    """A bin between two grid ages stays one segment at its midpoint."""
    b, mid, frac = split_age_bins(np.array([1.2e6]), np.array([2e6]), GRID)
    assert b.tolist() == [0]
    assert mid.tolist() == [1.6e6]
    assert frac.tolist() == [1.0]


def test_split_wide_bin():
    """A bin spanning grid ages is cut at each one, fractions by width."""
    lo, hi = np.array([0.0, 5e7]), np.array([2e7, 6e7])
    b, mid, frac = split_age_bins(lo, hi, GRID)
    # First bin contains 1e6, 3.16e6 and 1e7; second contains no grid age
    np.testing.assert_array_equal(b, [0, 0, 0, 0, 1])
    np.testing.assert_allclose(np.bincount(b, weights=frac), 1.0)
    cuts = np.array([0.0, GRID[0], GRID[1], GRID[2], 2e7])
    np.testing.assert_allclose(frac[:4], np.diff(cuts) / 2e7)
    np.testing.assert_allclose(mid[:4], 0.5 * (cuts[1:] + cuts[:-1]))


def test_zero_width_bin_is_burst():
    """A zero-width bin puts all its mass at its age."""
    lo = hi = np.array([5e6])
    b, mid, frac = split_age_bins(lo, hi, GRID)
    assert b.tolist() == [0]
    assert mid.tolist() == [5e6]
    assert frac.tolist() == [1.0]
