"""Tests for the parametric spectra C extension (bin in cell weights)."""

import numpy as np
import pytest
from synthesizer.extensions.integrated_spectra import compute_integrated_sed
from synthesizer.extensions.parametric_spectra import (
    compute_integrated_parametric_sed,
)

# An irregular 3D grid: a log10 age axis, a log10 metallicity axis and a
# linear axis, with deliberately uneven spacing on each
GRID_AXES = (
    np.array([6.0, 6.3, 7.0, 7.4, 8.5, 9.0, 10.1]),
    np.array([-4.0, -3.1, -2.5, -2.2, -1.4]),
    np.array([0.0, 0.5, 2.0, 2.5]),
)

# Which of the grid axes above hold log10 coordinates
LOG_FLAGS = (True, True, False)

# The number of wavelength elements in the test grid spectra
NLAM = 7

# The largest finite float, used for open mask bounds since the extension is
# compiled with -ffast-math and can't handle infinities
BIG = np.finfo(np.float64).max


def _grid_spectra(dtype=np.float64):
    """Return deterministic positive grid spectra for GRID_AXES.

    Args:
        dtype (np.dtype):
            The floating-point dtype of the returned spectra.

    Returns:
        np.ndarray:
            The C contiguous spectra with shape (*grid axis sizes, NLAM).
    """
    # Draw from a fixed seed so every call returns the same spectra
    shape = tuple(axis.size for axis in GRID_AXES) + (NLAM,)
    rng = np.random.default_rng(0)
    return np.ascontiguousarray(rng.uniform(0.5, 2.0, shape).astype(dtype))


def _cic_weights(points, masses):
    """Get CIC grid weights for points (linear units) via the particle path.

    This is the oracle the BIC weights are compared against, since a zero
    width bin must reduce exactly to CIC.

    Args:
        points (tuple of np.ndarray):
            The particle coordinates along each grid axis, in linear units.
        masses (np.ndarray):
            The mass of each particle.

    Returns:
        np.ndarray:
            The grid weights with the grid's shape.
    """
    # Convert the points into the grid's coordinates (log10 where flagged)
    props = tuple(
        np.ascontiguousarray(np.log10(p) if is_log else p, dtype=np.float64)
        for p, is_log in zip(points, LOG_FLAGS)
    )

    # Run the particle integrated spectra extension and keep only the weights
    _, weights = compute_integrated_sed(
        _grid_spectra(),
        GRID_AXES,
        props,
        np.ascontiguousarray(masses, dtype=np.float64),
        np.array([a.size for a in GRID_AXES], dtype=np.int32),
        len(GRID_AXES),
        masses.size,
        NLAM,
        "cic",
        1,
        None,
        None,
        None,
        None,
    )
    return weights


def _bic(edges, masses, nthreads=1, mask=None, lam_mask=None, weights=None):
    """Call compute_integrated_parametric_sed with the test grid.

    Args:
        edges (tuple of np.ndarray):
            The bin edges along each grid axis, in linear units.
        masses (np.ndarray):
            The masses with shape (npop, nbins_0, nbins_1, nbins_2).
        nthreads (int):
            The number of threads to use.
        mask (np.ndarray of bool):
            A mask with the shape of the masses, or None.
        lam_mask (np.ndarray of bool):
            A wavelength mask, or None.
        weights (np.ndarray):
            Precomputed grid weights to reuse, or None.

    Returns:
        tuple:
            The integrated spectrum and the grid weights.
    """
    return compute_integrated_parametric_sed(
        _grid_spectra(),
        GRID_AXES,
        edges,
        masses,
        LOG_FLAGS,
        nthreads,
        weights,
        mask,
        lam_mask,
        None,
    )


def _wide_edges():
    """Return edges with wide bins, zero lower edges and out of range bins.

    Returns:
        tuple of np.ndarray:
            The bin edges along each grid axis, in linear units.
    """
    return (
        np.array([0.0, 1e6, 5e6, 3e7, 2e8, 5e9, 3e10]),
        np.array([0.0, 1e-4, 1e-3, 4e-3, 2e-2, 0.1]),
        np.array([-1.0, 0.3, 1.0, 2.2, 4.0]),
    )


def _random_masses(edges, npop=3, seed=1):
    """Return random masses for npop populations on the given edges.

    Args:
        edges (tuple of np.ndarray):
            The bin edges along each grid axis.
        npop (int):
            The number of populations.
        seed (int):
            The random seed.

    Returns:
        np.ndarray:
            The masses with shape (npop, nbins_0, nbins_1, nbins_2).
    """
    # There is one fewer bin than edges along each axis
    shape = (npop,) + tuple(e.size - 1 for e in edges)
    return np.random.default_rng(seed).uniform(0.0, 1.0, shape)


def test_zero_width_bins_match_cic():
    """Zero width bins must reproduce particle CIC exactly."""
    # Place a single random point, which may lie off the grid, along each axis
    rng = np.random.default_rng(2)
    points = (
        10 ** rng.uniform(5.5, 10.5, 1),
        10 ** rng.uniform(-4.5, -1.0, 1),
        rng.uniform(-0.5, 3.0, 1),
    )

    # Turn the point into a zero width bin and spread it onto the grid
    edges = tuple(np.array([p[0], p[0]]) for p in points)
    masses = np.array([[[[2.5]]]])
    _, weights = _bic(edges, masses)

    # The weights must match a particle of the same mass at the point
    np.testing.assert_allclose(weights, _cic_weights(points, np.array([2.5])))


# A single bin that is wide along every axis (in linear units)
WIDE_BIN = (
    np.array([2e6, 4e8]),
    np.array([2e-4, 3e-2]),
    np.array([0.2, 2.3]),
)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_wide_bin_matches_averaged_cic(axis):
    """A bin's weights must equal CIC averaged uniformly over the bin.

    The oracle samples the bin along one axis at many evenly spaced points in
    linear units and averages their CIC weights, the other axes are held at
    a single point.

    Args:
        axis (int):
            The axis along which the bin is wide.
    """
    # Make the bin wide along one axis and a point along the others
    point = (3e7, 5e-3, 1.0)
    edges = tuple(
        WIDE_BIN[i] if i == axis else np.array([point[i], point[i]])
        for i in range(3)
    )
    _, weights = _bic(edges, np.array([[[[1.0]]]]))

    # Sample the wide axis at evenly spaced midpoints, sharing the mass
    # equally, and get the CIC weights of those samples
    nsamp = 20000
    lo, hi = WIDE_BIN[axis]
    points = [np.full(nsamp, p) for p in point]
    points[axis] = lo + (hi - lo) * (np.arange(nsamp) + 0.5) / nsamp
    expected = _cic_weights(tuple(points), np.full(nsamp, 1.0 / nsamp))

    # The BIC weights must match the averaged CIC weights
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-7)


def test_wide_bin_is_separable():
    """An N dimensional bin's weights are the product of its axis weights."""
    # Spread a bin that is wide along every axis
    _, weights = _bic(WIDE_BIN, np.array([[[[1.0]]]]))

    # Get the marginal weights along each axis by summing over the others
    marginals = [
        weights.sum(axis=tuple(j for j in range(3) if j != i))
        for i in range(3)
    ]

    # The full weights must be the outer product of the marginals
    np.testing.assert_allclose(
        weights, np.einsum("i,j,k->ijk", *marginals), atol=1e-15
    )


def test_mass_is_conserved_with_clamping():
    """All mass, including mass outside the grid, must land on the grid."""
    # Spread bins which extend beyond the grid along every axis
    edges = _wide_edges()
    masses = _random_masses(edges)
    _, weights = _bic(edges, masses)

    # The total weight must equal the total mass
    np.testing.assert_allclose(weights.sum(), masses.sum())


def test_out_of_range_mass_is_clamped_to_edge_points():
    """Bins entirely outside the grid must land on the edge grid points."""
    # A bin below the age axis and above the other two axes
    edges = (
        np.array([0.0, 5e5]),
        np.array([0.5, 1.0]),
        np.array([3.0, 4.0]),
    )
    masses = np.array([[[[1.0]]]])
    _, weights = _bic(edges, masses)

    # All of the mass must land on the corresponding corner grid point
    expected = np.zeros_like(weights)
    expected[0, -1, -1] = 1.0
    np.testing.assert_allclose(weights, expected)


@pytest.mark.parametrize("nthreads", [2, 4])
def test_threads_match_serial(nthreads):
    """Threaded results must match the serial results.

    Args:
        nthreads (int):
            The number of threads to compare against the serial result.
    """
    # Compute the spectra and weights serially and threaded
    edges = _wide_edges()
    masses = _random_masses(edges, npop=5)
    spec, weights = _bic(edges, masses)
    spec_t, weights_t = _bic(edges, masses, nthreads=nthreads)

    # Both must agree
    np.testing.assert_allclose(weights_t, weights)
    np.testing.assert_allclose(spec_t, spec)


def test_mask_clips_bins():
    """A range mask must equal clipping the bins to it by hand.

    Mass is uniform within a bin, so masking must be the same as clipping
    the edges to the allowed range and scaling each bin's mass by the
    fraction of it inside the range.
    """
    # Mask the first two axes to ranges that cut through bins
    edges = _wide_edges()
    masses = _random_masses(edges)
    mask = np.array([[2e6, 1e8], [-BIG, 5e-3], [-BIG, BIG]])
    spec, weights = _bic(edges, masses, mask=mask)

    # Clip the edges by hand and scale the masses by the fraction of each
    # bin inside the range
    clipped = []
    scaled = masses.copy()
    for iaxis, (axis_edges, (lo, hi)) in enumerate(zip(edges, mask)):
        clip = np.clip(axis_edges, lo, hi)
        frac = np.diff(clip) / np.diff(axis_edges)
        shape = [1] * masses.ndim
        shape[iaxis + 1] = frac.size
        scaled *= frac.reshape(shape)
        clipped.append(clip)
    spec_clipped, weights_clipped = _bic(tuple(clipped), scaled)

    # Both must agree
    np.testing.assert_allclose(weights, weights_clipped, atol=1e-14)
    np.testing.assert_allclose(spec, spec_clipped)


def test_mask_on_points_is_inclusive():
    """Zero width bins pass only when inside the inclusive range."""
    # A single point along every axis
    point = (3e7, 5e-3, 1.0)
    edges = tuple(np.array([p, p]) for p in point)
    masses = np.array([[[[1.0]]]])

    # A range ending exactly on the point keeps it
    mask = np.array([[-BIG, 3e7], [-BIG, BIG], [-BIG, BIG]])
    _, weights = _bic(edges, masses, mask=mask)
    assert np.isclose(weights.sum(), 1.0)

    # A range ending one step below the point excludes it
    mask[0, 1] = np.nextafter(3e7, 0.0)
    _, weights = _bic(edges, masses, mask=mask)
    assert weights.sum() == 0.0


def test_lam_mask_zeroes_masked_wavelengths():
    """Masked wavelengths must be zero and the rest unchanged."""
    # Compute the spectra with and without a wavelength mask
    edges = _wide_edges()
    masses = _random_masses(edges)
    lam_mask = np.array([True, False, True, True, False, True, True])
    spec, _ = _bic(edges, masses)
    spec_masked, _ = _bic(edges, masses, lam_mask=lam_mask)

    # Unmasked wavelengths must match and masked wavelengths must be zero
    np.testing.assert_allclose(spec_masked[lam_mask], spec[lam_mask])
    assert np.all(spec_masked[~lam_mask] == 0)


def test_precomputed_weights_are_reused():
    """Passing weights back in must skip the weight loop."""
    # Compute the spectra and weights from the masses
    edges = _wide_edges()
    masses = _random_masses(edges)
    spec, weights = _bic(edges, masses)

    # Pass the weights back in with zero masses, which would give a zero
    # spectrum if the weights were recomputed
    spec_again, _ = _bic(edges, np.zeros_like(masses), weights=weights)

    # The spectrum must be unchanged
    np.testing.assert_allclose(spec_again, spec)


def test_precomputed_weights_with_mask_raise():
    """A mask can't be applied to precomputed weights, so must raise."""
    # Compute the weights to pass back in
    edges = _wide_edges()
    masses = _random_masses(edges)
    _, weights = _bic(edges, masses)

    # Passing them back in alongside a mask must be rejected
    with pytest.raises(ValueError):
        _bic(
            edges,
            masses,
            mask=np.array([[-BIG, BIG]] * 3),
            weights=weights,
        )


def test_float32_inputs():
    """float32 populations and outputs must agree with float64."""
    # Compute the float64 reference spectrum
    edges = _wide_edges()
    masses = _random_masses(edges)
    spec, _ = _bic(edges, masses)

    # Compute the spectrum from float32 edges and masses at float32
    spec32, _ = compute_integrated_parametric_sed(
        _grid_spectra(),
        GRID_AXES,
        tuple(e.astype(np.float32) for e in edges),
        masses.astype(np.float32),
        LOG_FLAGS,
        1,
        None,
        None,
        None,
        np.float32,
    )

    # The output must be float32 and agree to float32 precision
    assert spec32.dtype == np.float32
    np.testing.assert_allclose(spec32, spec, rtol=1e-5)


@pytest.mark.parametrize(
    ("edges", "masses", "mask"),
    [
        # Too few axes
        (_wide_edges()[:2], np.ones((1, 6, 5)), None),
        # Edges that decrease
        (
            (np.array([1e7, 1e6]),) + _wide_edges()[1:],
            np.ones((1, 1, 5, 4)),
            None,
        ),
        # Masses not matching the edges
        (_wide_edges(), np.ones((1, 5, 5, 4)), None),
        # A mask with the wrong shape
        (_wide_edges(), np.ones((1, 6, 5, 4)), np.zeros((2, 2))),
        # Mismatched dtypes
        (
            tuple(e.astype(np.float32) for e in _wide_edges()),
            np.ones((1, 6, 5, 4)),
            None,
        ),
    ],
)
def test_invalid_inputs_raise(edges, masses, mask):
    """Invalid population inputs must raise rather than misbehave.

    Args:
        edges (tuple of np.ndarray):
            The (possibly invalid) bin edges along each axis.
        masses (np.ndarray):
            The (possibly invalid) masses.
        mask (np.ndarray of bool):
            The (possibly invalid) mask, or None.
    """
    # Each case must be rejected with a Python error
    with pytest.raises((ValueError, TypeError)):
        _bic(edges, masses, mask=mask)
