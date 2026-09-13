"""Tests for the load_data binning helpers."""

import numpy as np

from synthesizer.load_data.utils import (
    bin_overlap_matrix,
    cic_matrix,
    split_age_bins,
)

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


def test_overlap_matrix():
    """Overlap fractions sum to one and follow the parametric cell edges."""
    W = bin_overlap_matrix(np.array([0.0, 5e8]), np.array([2e7, 6e8]), GRID)
    np.testing.assert_allclose(W.sum(axis=1), 1.0)
    # A bin older than the grid lands entirely in the last cell
    np.testing.assert_array_equal(W[1], [0, 0, 0, 0, 1])
    # The first cell owns [0, midpoint(1e6, 3.16e6)]
    np.testing.assert_allclose(W[0, 0], 0.5 * (GRID[0] + GRID[1]) / 2e7)


def test_overlap_matches_parametric_stars(test_grid):
    """The overlap rule matches a Constant SFH binned by parametric Stars."""
    from unyt import Msun, yr

    from synthesizer.parametric import Stars
    from synthesizer.parametric.metal_dist import DeltaConstant
    from synthesizer.parametric.sf_hist import Constant

    ages = 10**test_grid.log10ages
    lo, hi = 0.0, 1e8
    slow = Stars(
        test_grid.log10ages,
        test_grid.metallicities,
        sf_hist=Constant(min_age=lo * yr, max_age=hi * yr),
        metal_dist=DeltaConstant(metallicity=0.01),
        initial_mass=1 * Msun,
    ).sfzh.sum(axis=1)
    fast = bin_overlap_matrix(np.array([lo]), np.array([hi]), ages)[0]
    np.testing.assert_allclose(fast, slow, atol=1e-8)


def test_cic_matrix():
    """Linear weights between neighbours, clamped beyond the grid."""
    grid = np.array([-3.0, -2.0, -1.0])
    w = cic_matrix(np.array([-2.25, -5.0, 0.0]), grid)
    np.testing.assert_allclose(w, [[0.25, 0.75, 0.0], [1, 0, 0], [0, 0, 1]])
