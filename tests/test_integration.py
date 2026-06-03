"""Tests for the integration module."""

import numpy as np
import pytest
from scipy.integrate import simpson, trapezoid

from synthesizer.extensions.integration import (
    simps_last_axis,
    trapz_last_axis,
    weighted_simps_last_axis,
    weighted_trapz_last_axis,
)
from synthesizer.utils.integrate import (
    integrate_last_axis,
    integrate_weighted_last_axis,
)

PRECISIONS = {
    "float32": np.float32,
    "float64": np.float64,
}


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


@pytest.fixture
def example_data_nonuniform_2d():
    """Fixture for non-uniform 2D example data."""
    xs = np.geomspace(1e-2, 10.0, 1001)
    ys = np.tile(np.sin(xs), (8, 1))
    return xs, ys


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_trapz_integration(example_data, threads, request):
    """Test the trapezoidal integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    expected = trapezoid(y=ys, x=xs, axis=-1)
    result = trapz_last_axis(xs, ys, threads, np.float64)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
@pytest.mark.parametrize(
    "example_data", [example_data_1d, example_data_2d, example_data_3d]
)
def test_simpson_integration(example_data, threads, request):
    """Test the Simpson's rule integration."""
    xs, ys = request.getfixturevalue(example_data.__name__)
    expected = simpson(y=ys, x=xs, axis=-1)
    result = simps_last_axis(xs, ys, threads, np.float64)
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

    result = weighted_trapz_last_axis(xs, ys, weights, threads, np.float64)
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

    result = weighted_simps_last_axis(xs, ys, weights, threads, np.float64)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize("threads", [1, 2])
def test_weighted_simpson_integration_nonuniform_grid(
    threads, example_data_nonuniform_2d
):
    """Weighted Simpson integration should support non-uniform x grids."""
    xs, ys = example_data_nonuniform_2d
    weights = 1.0 / (xs + 1.0)

    expected_num = simpson(y=ys * weights, x=xs, axis=-1)
    expected_den = simpson(y=weights, x=xs)
    expected = expected_num / expected_den

    result = weighted_simps_last_axis(xs, ys, weights, threads, np.float64)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.parametrize(
    "method, extension_function",
    [
        ("trapz", trapz_last_axis),
        ("simps", simps_last_axis),
    ],
)
@pytest.mark.parametrize("input_name", ["float32", "float64"])
@pytest.mark.parametrize("output_name", ["float32", "float64"])
def test_integration_precision_combinations_agree_with_float64_reference(
    method, extension_function, input_name, output_name
):
    """Precision combinations should agree with a float64 reference."""
    xs64 = np.linspace(0.0, 10.0, 1024, dtype=np.float64)
    ys64 = np.vstack(
        [
            np.sin(xs64),
            np.cos(0.5 * xs64),
            np.exp(-0.2 * xs64),
        ]
    )

    expected = (
        trapezoid(y=ys64, x=xs64, axis=-1)
        if method == "trapz"
        else simpson(y=ys64, x=xs64, axis=-1)
    )

    input_dtype = PRECISIONS[input_name]
    output_dtype = PRECISIONS[output_name]
    xs = np.array(xs64, dtype=input_dtype, order="C", copy=True)
    ys = np.array(ys64, dtype=input_dtype, order="C", copy=True)

    result = extension_function(xs, ys, 1, output_dtype)

    assert result.dtype == output_dtype
    np.testing.assert_allclose(
        result,
        expected.astype(output_dtype),
        rtol=5e-4 if input_dtype == np.float32 else 1e-6,
        atol=5e-5 if output_dtype == np.float32 else 1e-8,
    )


@pytest.mark.parametrize(
    "extension_function", [trapz_last_axis, simps_last_axis]
)
def test_integration_accepts_mismatched_precision_families(extension_function):
    """Integration should handle mixed float32/float64 inputs."""
    xs = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    ys = np.ones((4, 32), dtype=np.float64)

    result = extension_function(xs, ys, 1, np.float32)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.ones(4, dtype=np.float32))


@pytest.mark.parametrize("method", ["trapz", "simps"])
def test_integrate_last_axis_wrapper_honors_out_dtype(method):
    """The public wrapper should preserve the requested output dtype."""
    xs = np.linspace(0.0, 5.0, 128, dtype=np.float32)
    ys = np.tile(np.sin(xs), (3, 1)).astype(np.float32)

    result = integrate_last_axis(xs, ys, method=method, out_dtype=np.float64)

    assert result.dtype == np.float64


@pytest.mark.parametrize("method", ["trapz", "simps"])
def test_integrate_weighted_last_axis_wrapper_honors_out_dtype(method):
    """Weighted wrapper should preserve the requested output dtype."""
    xs = np.linspace(0.0, 10.0, 200, dtype=np.float32)
    ys = np.tile(np.sin(xs), (5, 1)).astype(np.float32)
    weights = (1.0 / (xs + 1.0)).astype(np.float32)

    result = integrate_weighted_last_axis(
        xs,
        ys,
        weights,
        method=method,
        out_dtype=np.float64,
    )

    assert result.dtype == np.float64


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
    result = integrate_weighted_last_axis(
        xs,
        ys,
        weights,
        method=method,
        out_dtype=np.float64,
    )
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=1e-8)


if __name__ == "__main__":
    pytest.main(["-k", "test_integration.py"])
