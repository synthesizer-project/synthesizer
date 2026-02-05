"""Tests for the integration module."""

import numpy as np
import pytest
from scipy.integrate import simpson, trapezoid

from synthesizer.utils import integrate, precision


@pytest.fixture
def example_data_1d():
    """Fixture for 1D example data."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)  # 1D x array
    ys = np.sin(xs).astype(dtype)  # Corresponding y values
    return xs, ys


@pytest.fixture
def example_data_2d():
    """Fixture for 2D example data."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)  # 1D x array
    ys = np.sin(xs).astype(dtype)  # Corresponding y values
    ys = np.tile(ys, (100, 1))  # Reshape ys to be 2D
    return xs, ys


@pytest.fixture
def example_data_3d():
    """Fixture for 3D example data."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)  # 1D x array
    ys = np.sin(xs).astype(dtype)  # Corresponding y values
    ys = np.tile(ys, (10, 10, 1))  # Reshape ys to be 3D, last axis matches xs
    ys = np.sin(xs * ys).astype(dtype)  # Repeat ys along the last axis
    return xs, ys


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_trapz_integration(example_data, threads, request):
    """Test the trapezoidal integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    expected = trapezoid(y=ys, x=xs, axis=-1)
    result = integrate.integrate_last_axis(
        xs, ys, nthreads=threads, method="trapz"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_simpson_integration(example_data, threads, request):
    """Test the Simpson's rule integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    expected = simpson(y=ys, x=xs, axis=-1)
    result = integrate.integrate_last_axis(
        xs, ys, nthreads=threads, method="simps"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_extreme_integration():
    """Test integration with extreme values to check for overflow issues."""
    dtype = precision.get_numpy_dtype()

    # Use large values that would overflow float32 if not handled
    # Width = 1e15, Height = 1e30 -> Integral = 1e45
    xs = np.linspace(1e14, 1e15, 1000).astype(dtype)
    ys = np.full_like(xs, 1e30).astype(dtype)
    expected = 9e44

    # Use the wrapper which should handle scaling
    result = integrate.integrate_last_axis(xs, ys, nthreads=1, method="trapz")

    assert np.isfinite(result), f"Integration result is not finite: {result}"
    np.testing.assert_allclose(result, expected, rtol=1e-4)


@pytest.fixture
def example_data_with_weights_1d():
    """Fixture for 1D example data with weights."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)
    ys = np.sin(xs).astype(dtype)
    weights = np.exp(-xs / 5).astype(dtype)  # Exponential decay weights
    return xs, ys, weights


@pytest.fixture
def example_data_with_weights_2d():
    """Fixture for 2D example data with weights."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)
    ys = np.sin(xs).astype(dtype)
    ys = np.tile(ys, (100, 1))
    weights = np.exp(-xs / 5).astype(dtype)
    return xs, ys, weights


@pytest.fixture
def example_data_with_weights_3d():
    """Fixture for 3D example data with weights."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)
    ys = np.sin(xs).astype(dtype)
    ys = np.tile(ys, (10, 10, 1))
    ys = np.sin(xs * ys).astype(dtype)
    weights = np.exp(-xs / 5).astype(dtype)
    return xs, ys, weights


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data",
    [
        example_data_with_weights_1d,
        example_data_with_weights_2d,
        example_data_with_weights_3d,
    ],
)
def test_trapz_weighted_integration(example_data, threads, request):
    """Test weighted trapezoidal integration against scipy."""
    xs, ys, weights = request.getfixturevalue(example_data.__name__)

    # Compute expected result using scipy (manually applying weights)
    ys_weighted = ys * weights
    expected = trapezoid(y=ys_weighted, x=xs, axis=-1)

    # Test our weighted integration
    result = integrate.integrate_last_axis(
        xs, ys, weights=weights, nthreads=threads, method="trapz"
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data",
    [
        example_data_with_weights_1d,
        example_data_with_weights_2d,
        example_data_with_weights_3d,
    ],
)
def test_simpson_weighted_integration(example_data, threads, request):
    """Test weighted Simpson's integration against scipy."""
    xs, ys, weights = request.getfixturevalue(example_data.__name__)

    # Compute expected result using scipy (manually applying weights)
    ys_weighted = ys * weights
    expected = simpson(y=ys_weighted, x=xs, axis=-1)

    # Test our weighted integration
    result = integrate.integrate_last_axis(
        xs, ys, weights=weights, nthreads=threads, method="simps"
    )

    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_weighted_vs_unweighted():
    """Test that weighted integration with unit weights equals unweighted."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)
    ys = np.sin(xs).astype(dtype)
    ys = np.tile(ys, (50, 1))
    weights = np.ones_like(xs).astype(dtype)

    # Both should give same result
    result_unweighted = integrate.integrate_last_axis(
        xs, ys, nthreads=1, method="trapz"
    )
    result_weighted = integrate.integrate_last_axis(
        xs, ys, weights=weights, nthreads=1, method="trapz"
    )

    np.testing.assert_allclose(
        result_weighted, result_unweighted, rtol=1e-10, atol=1e-12
    )


def test_extreme_weighted_integration():
    """Test weighted integration with extreme values for overflow handling."""
    dtype = precision.get_numpy_dtype()

    # Large values that would overflow if not scaled properly
    xs = np.linspace(1e14, 1e15, 1000).astype(dtype)
    ys = np.full_like(xs, 1e30).astype(dtype)
    weights = np.full_like(xs, 1e10).astype(dtype)

    # Expected: integral of (1e30 * 1e10) over width 9e14 = 9e54
    expected = 9e54

    result = integrate.integrate_last_axis(
        xs, ys, weights=weights, nthreads=1, method="trapz"
    )

    assert np.isfinite(result), (
        f"Weighted integration result is not finite: {result}"
    )
    np.testing.assert_allclose(result, expected, rtol=1e-4)


def test_zero_weights():
    """Test that zero weights produce zero integral."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 1000).astype(dtype)
    ys = np.sin(xs).astype(dtype)
    ys = np.tile(ys, (20, 1))
    weights = np.zeros_like(xs).astype(dtype)

    result = integrate.integrate_last_axis(
        xs, ys, weights=weights, nthreads=1, method="trapz"
    )

    np.testing.assert_allclose(result, 0.0, atol=1e-15)


def test_integration_consistency_across_methods():
    """Test that trapz and simps give similar results for smooth functions."""
    dtype = precision.get_numpy_dtype()
    # Use a polynomial that doesn't integrate to ~0 to avoid rtol issues
    xs = np.linspace(0, np.pi, 10000).astype(dtype)
    ys = (xs**2 + 1).astype(dtype)  # Integral from 0 to pi is pi^3/3 + pi

    result_trapz = integrate.integrate_last_axis(
        xs, ys, nthreads=1, method="trapz"
    )
    result_simps = integrate.integrate_last_axis(
        xs, ys, nthreads=1, method="simps"
    )

    # For smooth functions with fine sampling, both should be close
    np.testing.assert_allclose(result_trapz, result_simps, rtol=1e-3)


def test_invalid_method():
    """Test that invalid integration method raises error."""
    dtype = precision.get_numpy_dtype()
    xs = np.linspace(0, 10, 100).astype(dtype)
    ys = np.sin(xs).astype(dtype)

    with pytest.raises(Exception):  # Should raise InconsistentArguments
        integrate.integrate_last_axis(xs, ys, method="invalid_method")
