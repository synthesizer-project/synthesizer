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
    result = integrate.integrate_last_axis(xs, ys, threads, method="trapz")
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_simpson_integration(example_data, threads, request):
    """Test the Simpson's rule integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    expected = simpson(y=ys, x=xs, axis=-1)
    result = integrate.integrate_last_axis(xs, ys, threads, method="simps")
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
    result = integrate.integrate_last_axis(xs, ys, 1, method="trapz")

    assert np.isfinite(result), f"Integration result is not finite: {result}"
    np.testing.assert_allclose(result, expected, rtol=1e-4)
