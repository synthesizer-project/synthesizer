"""Test suite for particle based black holes."""

import numpy as np
from unyt import s, unyt_array

from synthesizer.utils import scalar_to_array


class TestBlackHolesInit:
    """Test suite for initialising BlackHoles instances."""

    def test_scalar_to_array(self):
        """Test that scalar_to_array works in various situations."""
        # Scalar with no units
        arr = scalar_to_array(1)
        assert isinstance(arr, np.ndarray), (
            f"Scalar with no units failed: 1->{arr}"
        )

        # Scalar with units
        arr = scalar_to_array(1 * s)
        assert isinstance(arr, unyt_array), (
            f"Scalar with units failed: 1 * s->{arr}"
        )
        assert arr.units == s, f"Scalar with units failed: 1 * s->{arr}"
        assert arr.shape == (1,), (
            f"Scalar with units shape is wrong: {arr.shape} "
            f"(value: {arr}, type: {type(arr)})"
        )

        # Check that an array without units is returned as is
        arr = scalar_to_array(np.arange(10))
        assert isinstance(arr, np.ndarray), (
            f"Array without units failed: {np.arange(10)}->{arr}"
        )

        # Check that an array with units is returned as is
        arr = scalar_to_array(np.arange(10) * s)
        assert isinstance(arr, unyt_array), (
            f"Array with units failed: {np.arange(10) * s}->{arr}"
        )
        assert arr.units == s, (
            f"Array with units failed: {np.arange(10) * s}->{arr}"
        )

        # Check that a ndim = 2 array without units is returned as is
        arr = scalar_to_array(np.arange(10).reshape(2, 5))
        assert isinstance(arr, np.ndarray), (
            "2D array without units failed: "
            f"{np.arange(10).reshape(2, 5)}->{arr}"
        )

        # Check that a ndim = 2 array with units is returned as is
        arr = scalar_to_array(np.arange(10).reshape(2, 5) * s)
        assert isinstance(arr, unyt_array), (
            "2D array with units failed:"
            f" {np.arange(10).reshape(2, 5) * s}->{arr}"
        )

        # Check that a 1 element aray without units is returned as is
        arr = scalar_to_array(np.array([1]))
        assert isinstance(arr, np.ndarray), (
            f"1 element array without units failed: {np.array([1])}->{arr}"
        )
        assert arr.shape == (1,), (
            f"1 element array without units failed: {np.array([1])}->{arr}"
        )
        assert arr.ndim == 1, (
            f"1 element array without units failed: {np.array([1])}->{arr}"
        )
