"""Tests for the integration module."""

import numpy as np
import pytest
from scipy.integrate import simpson, trapezoid

from synthesizer.extensions.integration import (
    simps_last_axis,
    trapz_last_axis,
    weighted_simps_last_axis,
    weighted_trapz_last_axis,
    weighted_trapz_last_axis_many,
)
from synthesizer.utils.integrate import (
    integrate_weighted_last_axis,
    integrate_weighted_many_last_axis,
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


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_weighted_trapz_integration(example_data, threads, request):
    """Test weighted trapezoidal integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    weights = np.exp(-0.5 * ((xs - xs.mean()) / xs.std()) ** 2)

    expected_num = trapezoid(y=ys * weights, x=xs, axis=-1)
    expected_den = trapezoid(y=weights, x=xs)
    expected = expected_num / expected_den

    result = weighted_trapz_last_axis(xs, ys, weights, threads)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_weighted_simpson_integration(example_data, threads, request):
    """Test weighted Simpson integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    weights = np.exp(-0.5 * ((xs - xs.mean()) / xs.std()) ** 2)

    expected_num = simpson(y=ys * weights, x=xs, axis=-1)
    expected_den = simpson(y=weights, x=xs)
    expected = expected_num / expected_den

    result = weighted_simps_last_axis(xs, ys, weights, threads)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("method", ["trapz", "simps"])
def test_integrate_weighted_last_axis_wrapper(method):
    """Test Python weighted wrapper against SciPy reference."""
    xs = np.linspace(0, 10, 200)
    ys = np.tile(np.sin(xs), (5, 1))
    weights = 1.0 / (xs + 1.0)

    if method == "trapz":
        expected_num = trapezoid(y=ys * weights, x=xs, axis=-1)
        expected_den = trapezoid(y=weights, x=xs)
    else:
        expected_num = simpson(y=ys * weights, x=xs, axis=-1)
        expected_den = simpson(y=weights, x=xs)

    expected = expected_num / expected_den
    result = integrate_weighted_last_axis(xs, ys, weights, method=method)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
def test_weighted_trapz_many_integration(threads):
    """Test batched weighted trapezoid integration for many filters."""
    xs = np.linspace(0, 10, 500)
    ys = np.tile(np.sin(xs), (7, 1))
    weights = np.vstack(
        [
            np.exp(-0.5 * ((xs - 2.0) / 1.0) ** 2),
            np.exp(-0.5 * ((xs - 5.0) / 1.2) ** 2),
            np.exp(-0.5 * ((xs - 8.0) / 0.8) ** 2),
        ]
    )
    denoms = trapezoid(y=weights, x=xs, axis=-1)
    starts = np.array([0, 0, 0], dtype=np.int64)
    ends = np.array([xs.size, xs.size, xs.size], dtype=np.int64)

    expected = np.stack(
        [
            trapezoid(y=ys * w, x=xs, axis=-1) / d
            for w, d in zip(weights, denoms)
        ],
        axis=-1,
    )

    result = weighted_trapz_last_axis_many(
        xs,
        ys,
        weights,
        denoms,
        starts,
        ends,
        threads,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


def test_integrate_weighted_many_last_axis_wrapper():
    """Test batched Python wrapper against SciPy reference."""
    xs = np.linspace(0, 10, 300)
    ys = np.tile(np.sin(xs), (4, 1))
    weights = np.vstack([1.0 / (xs + 1.0), 1.0 / (xs + 2.0)])
    denoms = trapezoid(y=weights, x=xs, axis=-1)
    starts = np.array([0, 0], dtype=np.int64)
    ends = np.array([xs.size, xs.size], dtype=np.int64)

    expected = np.stack(
        [
            trapezoid(y=ys * w, x=xs, axis=-1) / d
            for w, d in zip(weights, denoms)
        ],
        axis=-1,
    )

    result = integrate_weighted_many_last_axis(
        xs,
        ys,
        weights,
        denoms,
        starts,
        ends,
        method="trapz",
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main(["-k", "test_integration.py"])
