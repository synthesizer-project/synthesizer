"""A test suite for the utils module."""

import numpy as np
import unyt

from synthesizer.emission_models.utils import (
    ensure_array_c_compatible_double,
)


class TestEnsureCompatibles:
    """A test suite for functions helping ensure compatibility."""

    def test_c_compatible_array_scalar(self):
        """Test the ensure_array_c_compatible_double function."""
        # Test with a single value
        value = 1.0
        result = ensure_array_c_compatible_double(value)
        assert result == np.float64(1.0), (
            "Expected a single float value to be returned as np.float64 "
            f"but got {result}"
        )

    def test_c_compatible_array_list(self):
        """Test the ensure_array_c_compatible_double function."""
        # Test with a list of values
        value = [1.0, 2.0, 3.0]
        result = ensure_array_c_compatible_double(value)
        assert isinstance(result, np.ndarray), (
            "Expected a list to be converted to a numpy array "
            f"but got {result}"
        )
        assert result.dtype == np.float64, (
            f"Expected a numpy array of type float64 but got {result.dtype}"
        )

    def test_c_compatible_array_numpy32(self):
        """Test the ensure_array_c_compatible_double function."""
        # Test with a numpy array of type float32
        value = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = ensure_array_c_compatible_double(value)
        assert isinstance(result, np.ndarray), (
            f"Expected a numpy array to be returned but got {result}"
        )
        assert result.dtype == np.float64, (
            f"Expected a numpy array of type float64 but got {result.dtype}"
        )

    def test_c_compatible_array_numpy64(self):
        """Test the ensure_array_c_compatible_double function."""
        # Test with a numpy array of type float64
        value = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        result = ensure_array_c_compatible_double(value)
        assert isinstance(result, np.ndarray), (
            f"Expected a numpy array to be returned but got {result}"
        )
        assert result.dtype == np.float64, (
            f"Expected a numpy array of type float64 but got {result.dtype}"
        )

    def test_c_compatible_array_unyt32(self):
        """Test the ensure_array_c_compatible_double function."""
        value = unyt.unyt_array(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "cm",
        )
        result = ensure_array_c_compatible_double(value)
        assert isinstance(result, unyt.unyt_array), (
            f"Expected a numpy array to be returned but got {result}"
        )
        assert result.dtype == np.float64, (
            f"Expected a numpy array of type float64 but got {result.dtype}"
        )
        assert result.units == value.units, (
            f"Expected a numpy array of type {value.units} but "
            f"got {result.units}"
        )

    def test_c_compatible_array_unyt64(self):
        """Test the ensure_array_c_compatible_double function."""
        value = unyt.unyt_array(
            np.array([1.0, 2.0, 3.0], dtype=np.float32),
            "cm",
        )
        result = ensure_array_c_compatible_double(value)
        assert isinstance(result, unyt.unyt_array), (
            f"Expected a numpy array to be returned but got {result}"
        )
        assert result.dtype == np.float64, (
            f"Expected a numpy array of type float64 but got {result.dtype}"
        )
        assert result.units == value.units, (
            f"Expected a numpy array of type {value.units} but "
            f"got {result.units}"
        )
