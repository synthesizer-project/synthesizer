"""Tests for the parametric spectra C extension (bin in cell weights)."""

import numpy as np
import pytest
from synthesizer.extensions.integrated_spectra import compute_integrated_sed
from synthesizer.extensions.parametric_spectra import (
    compute_integrated_parametric_sed,
    compute_population_seds,
)
from threadpoolctl import threadpool_info

from synthesizer.utils.blas import limit_blas_threads

# An irregular 3D grid: a log10 age axis, a log10 metallicity axis and a
# linear axis, with deliberately uneven spacing on each
GRID_AXES = (
    np.array([6.0, 6.3, 7.0, 7.4, 8.5, 9.0, 10.1]),
    np.array([-4.0, -3.1, -2.5, -2.2, -1.4]),
    np.array([0.0, 0.5, 2.0, 2.5]),
)
LOG_FLAGS = (True, True, False)
NLAM = 7


def _grid_spectra(dtype=np.float64):
    """Return deterministic positive grid spectra for GRID_AXES."""
    shape = tuple(axis.size for axis in GRID_AXES) + (NLAM,)
    rng = np.random.default_rng(0)
    return np.ascontiguousarray(rng.uniform(0.5, 2.0, shape).astype(dtype))


def _to_linear(values, is_log):
    """Convert grid coordinates into linear units."""
    return 10**values if is_log else values


def _cic_weights(points, masses):
    """Get CIC grid weights for points (linear units) via the particle path."""
    props = tuple(
        np.ascontiguousarray(np.log10(p) if is_log else p, dtype=np.float64)
        for p, is_log in zip(points, LOG_FLAGS)
    )
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
    """Call compute_integrated_parametric_sed with the test grid."""
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
    """Return edges with wide bins, zero lower edges and out of range bins."""
    return (
        np.array([0.0, 1e6, 5e6, 3e7, 2e8, 5e9, 3e10]),
        np.array([0.0, 1e-4, 1e-3, 4e-3, 2e-2, 0.1]),
        np.array([-1.0, 0.3, 1.0, 2.2, 4.0]),
    )


def _random_masses(edges, npop=3, seed=1):
    """Return random masses for npop populations on the given edges."""
    shape = (npop,) + tuple(e.size - 1 for e in edges)
    return np.random.default_rng(seed).uniform(0.0, 1.0, shape)


def test_zero_width_bins_match_cic():
    """Zero width bins must reproduce particle CIC exactly."""
    rng = np.random.default_rng(2)
    points = (
        10 ** rng.uniform(5.5, 10.5, 1),
        10 ** rng.uniform(-4.5, -1.0, 1),
        rng.uniform(-0.5, 3.0, 1),
    )
    edges = tuple(np.array([p[0], p[0]]) for p in points)
    masses = np.array([[[[2.5]]]])
    _, weights = _bic(edges, masses)
    np.testing.assert_allclose(weights, _cic_weights(points, np.array([2.5])))


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
    """
    point = (3e7, 5e-3, 1.0)
    edges = tuple(
        WIDE_BIN[i] if i == axis else np.array([point[i], point[i]])
        for i in range(3)
    )
    _, weights = _bic(edges, np.array([[[[1.0]]]]))

    nsamp = 20000
    lo, hi = WIDE_BIN[axis]
    points = [np.full(nsamp, p) for p in point]
    points[axis] = lo + (hi - lo) * (np.arange(nsamp) + 0.5) / nsamp
    expected = _cic_weights(tuple(points), np.full(nsamp, 1.0 / nsamp))
    np.testing.assert_allclose(weights, expected, rtol=0.0, atol=1e-7)


def test_wide_bin_is_separable():
    """An N dimensional bin's weights are the product of its axis weights."""
    _, weights = _bic(WIDE_BIN, np.array([[[[1.0]]]]))
    marginals = [
        weights.sum(axis=tuple(j for j in range(3) if j != i))
        for i in range(3)
    ]
    np.testing.assert_allclose(
        weights, np.einsum("i,j,k->ijk", *marginals), atol=1e-15
    )


def test_mass_is_conserved_with_clamping():
    """All mass, including mass outside the grid, must land on the grid."""
    edges = _wide_edges()
    masses = _random_masses(edges)
    _, weights = _bic(edges, masses)
    np.testing.assert_allclose(weights.sum(), masses.sum())


def test_out_of_range_mass_is_clamped_to_edge_points():
    """Bins entirely outside the grid must land on the edge grid points."""
    edges = (
        np.array([0.0, 5e5]),
        np.array([0.5, 1.0]),
        np.array([3.0, 4.0]),
    )
    masses = np.array([[[[1.0]]]])
    _, weights = _bic(edges, masses)
    expected = np.zeros_like(weights)
    expected[0, -1, -1] = 1.0
    np.testing.assert_allclose(weights, expected)


def test_population_seds_sum_to_integrated():
    """Per population spectra must sum to the integrated spectrum."""
    edges = _wide_edges()
    masses = _random_masses(edges, npop=4)
    spec, _ = _bic(edges, masses)
    pop_specs, pop_weights = compute_population_seds(
        _grid_spectra(),
        GRID_AXES,
        edges,
        masses,
        LOG_FLAGS,
        1,
        None,
        None,
        None,
        None,
    )
    assert pop_specs.shape == (4, NLAM)
    np.testing.assert_allclose(pop_specs.sum(axis=0), spec)

    # The per population weights sum to the integrated weights and can be
    # passed back in to skip computing them
    np.testing.assert_allclose(pop_weights.sum(axis=0), _bic(edges, masses)[1])
    again, _ = compute_population_seds(
        _grid_spectra(),
        GRID_AXES,
        edges,
        np.zeros_like(masses),
        LOG_FLAGS,
        1,
        pop_weights,
        None,
        None,
        None,
    )
    np.testing.assert_allclose(again, pop_specs)

    # Each population on its own must match the integrated function
    for ipop in range(masses.shape[0]):
        single, _ = _bic(edges, masses[ipop : ipop + 1])
        np.testing.assert_allclose(pop_specs[ipop], single)


@pytest.mark.parametrize("nthreads", [2, 4])
def test_threads_match_serial(nthreads):
    """Threaded results must match the serial results."""
    edges = _wide_edges()
    masses = _random_masses(edges, npop=5)
    spec, weights = _bic(edges, masses)
    spec_t, weights_t = _bic(edges, masses, nthreads=nthreads)
    np.testing.assert_allclose(weights_t, weights)
    np.testing.assert_allclose(spec_t, spec)
    args = (_grid_spectra(), GRID_AXES, edges, masses, LOG_FLAGS)
    np.testing.assert_allclose(
        compute_population_seds(*args, nthreads, None, None, None, None)[0],
        compute_population_seds(*args, 1, None, None, None, None)[0],
    )


def test_mask_excludes_bins():
    """Masked bins must contribute nothing, like zeroing their mass."""
    edges = _wide_edges()
    masses = _random_masses(edges)
    mask = np.random.default_rng(3).random(masses.shape) > 0.5
    spec, weights = _bic(edges, masses, mask=mask)
    spec_zeroed, weights_zeroed = _bic(edges, np.where(mask, masses, 0.0))
    np.testing.assert_allclose(weights, weights_zeroed)
    np.testing.assert_allclose(spec, spec_zeroed)


def test_lam_mask_zeroes_masked_wavelengths():
    """Masked wavelengths must be zero and the rest unchanged."""
    edges = _wide_edges()
    masses = _random_masses(edges)
    lam_mask = np.array([True, False, True, True, False, True, True])
    spec, _ = _bic(edges, masses)
    spec_masked, _ = _bic(edges, masses, lam_mask=lam_mask)
    np.testing.assert_allclose(spec_masked[lam_mask], spec[lam_mask])
    assert np.all(spec_masked[~lam_mask] == 0)
    pop_specs, _ = compute_population_seds(
        _grid_spectra(),
        GRID_AXES,
        edges,
        masses,
        LOG_FLAGS,
        1,
        None,
        None,
        lam_mask,
        None,
    )
    assert np.all(pop_specs[:, ~lam_mask] == 0)


def test_precomputed_weights_are_reused():
    """Passing weights back in must skip the weight loop."""
    edges = _wide_edges()
    masses = _random_masses(edges)
    spec, weights = _bic(edges, masses)
    spec_again, _ = _bic(edges, np.zeros_like(masses), weights=weights)
    np.testing.assert_allclose(spec_again, spec)


def test_float32_inputs():
    """float32 populations and outputs must agree with float64."""
    edges = _wide_edges()
    masses = _random_masses(edges)
    spec, _ = _bic(edges, masses)
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
        (_wide_edges(), np.ones((1, 6, 5, 4)), np.ones((6, 5, 4), bool)),
        # Mismatched dtypes
        (
            tuple(e.astype(np.float32) for e in _wide_edges()),
            np.ones((1, 6, 5, 4)),
            None,
        ),
    ],
)
def test_invalid_inputs_raise(edges, masses, mask):
    """Invalid population inputs must raise rather than misbehave."""
    with pytest.raises((ValueError, TypeError)):
        _bic(edges, masses, mask=mask)


@pytest.mark.parametrize("nthreads", [-1, 1, 2])
def test_blas_thread_limit(nthreads):
    """Limiting BLAS threads must apply the limit and not change results.

    Where the BLAS can be limited (e.g. OpenBLAS on Linux) the limit must be
    in place inside the block. Apple's Accelerate can't be limited, in which
    case the limit is a no-op.
    """
    edges = _wide_edges()
    masses = _random_masses(edges, npop=6)
    args = (_grid_spectra(), GRID_AXES, edges, masses, LOG_FLAGS, 1)
    expected = compute_population_seds(*args, None, None, None, None)[0]
    with limit_blas_threads(nthreads):
        if nthreads > 0:
            for info in threadpool_info():
                if info["user_api"] == "blas":
                    assert info["num_threads"] == nthreads
        result = compute_population_seds(*args, None, None, None, None)[0]
    np.testing.assert_allclose(result, expected)
