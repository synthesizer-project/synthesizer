"""Tests for the integration module."""

import numpy as np
import pytest
from scipy.integrate import simpson, trapezoid

from synthesizer.extensions.integration import (
    simps_last_axis,
    trapz_last_axis,
)


@pytest.fixture
def example_data_1d():
    """Fixture for 1D example data."""
    xs = np.linspace(0, 10, 1000)  # 1D x array
    ys = np.sin(xs)  # Corresponding y values
    return xs, ys


@pytest.fixture
def example_data_2d():
    """Fixture for 2D example data."""
    xs = np.linspace(0, 10, 1000)  # 1D x array
    ys = np.sin(xs)  # Corresponding y values
    ys = np.tile(ys, (100, 1))  # Reshape ys to be 2D
    return xs, ys


@pytest.fixture
def example_data_3d():
    """Fixture for 3D example data."""
    xs = np.linspace(0, 10, 1000)  # 1D x array
    ys = np.sin(xs)  # Corresponding y values
    ys = np.tile(ys, (10, 10, 1))  # Reshape ys to be 3D, last axis matches xs
    ys = np.sin(xs * ys)  # Repeat ys along the last axis
    return xs, ys


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_trapz_integration(example_data, threads, request):
    """Test the trapezoidal integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    expected = trapezoid(y=ys, x=xs, axis=-1)
    result = trapz_last_axis(xs, ys, threads)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_simpson_integration(example_data, threads, request):
    """Test the Simpson's rule integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    expected = simpson(y=ys, x=xs, axis=-1)
    result = simps_last_axis(xs, ys, threads)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main(["-k", "test_integration.py"])
