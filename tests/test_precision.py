"""Tests for precision utilities."""

import numpy as np
import pytest

from synthesizer import precision


def test_precision_info():
    """Test that precision info functions return consistent values."""
    prec = precision.get_precision()
    assert prec in ["float32", "float64"]

    dtype = precision.get_numpy_dtype()
    if prec == "float32":
        assert dtype == np.float32
    else:
        assert dtype == np.float64


def test_array_to_precision():
    """Test the array_to_precision utility function."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32

    # Test matching dtype (no copy)
    arr = np.array([1.0, 2.0], dtype=target_dtype)
    res = precision.array_to_precision(arr, copy=False)
    assert res is arr
    assert res.dtype == target_dtype

    # Test mismatching dtype with copy=False (should raise error)
    arr_wrong = np.array([1.0, 2.0], dtype=other_dtype)
    with pytest.raises(TypeError, match="Array has dtype"):
        precision.array_to_precision(arr_wrong, copy=False)

    # Test mismatching dtype with copy=True (should convert)
    res_conv = precision.array_to_precision(arr_wrong, copy=True)
    assert res_conv.dtype == target_dtype
    assert np.allclose(res_conv, arr_wrong)

    # Test list input (should convert)
    res_list = precision.array_to_precision([1.0, 2.0], copy=True)
    assert res_list.dtype == target_dtype
    assert np.allclose(res_list, [1.0, 2.0])
