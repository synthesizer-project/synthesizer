"""Tests for precision utilities."""

import numpy as np
import pytest
from unyt import unyt_array, unyt_quantity

from synthesizer import precision

# Test precision info functions


def test_get_precision():
    """Test that get_precision returns valid precision string."""
    prec = precision.get_precision()
    assert prec in ["float32", "float64"]


def test_get_float_bytes():
    """Test that get_float_bytes returns correct byte size."""
    num_bytes = precision.get_float_bytes()
    assert num_bytes in [4, 8]

    # Check consistency with get_precision
    prec = precision.get_precision()
    if prec == "float32":
        assert num_bytes == 4
    else:
        assert num_bytes == 8


def test_get_numpy_dtype():
    """Test that get_numpy_dtype returns correct numpy dtype."""
    dtype = precision.get_numpy_dtype()
    assert dtype in [np.float32, np.float64]

    # Check consistency with get_precision
    prec = precision.get_precision()
    if prec == "float32":
        assert dtype == np.float32
    else:
        assert dtype == np.float64


def test_get_integer_dtype():
    """Test that get_integer_dtype returns correct integer dtype."""
    int_dtype = precision.get_integer_dtype()
    assert int_dtype in [np.int32, np.int64]

    # Check consistency with float precision
    float_dtype = precision.get_numpy_dtype()
    if float_dtype == np.float32:
        assert int_dtype == np.int32
    else:
        assert int_dtype == np.int64


def test_get_boolean_dtype():
    """Test that get_boolean_dtype returns boolean dtype."""
    bool_dtype = precision.get_boolean_dtype()
    assert bool_dtype == np.bool_


# Test array conversion functions


def test_array_to_precision_matching_dtype():
    """Test array_to_precision with matching dtype (no copy needed)."""
    target_dtype = precision.get_numpy_dtype()
    arr = np.array([1.0, 2.0, 3.0], dtype=target_dtype)

    # With copy=False, should return same object if dtype matches
    res = precision.array_to_precision(arr, copy=False)
    assert res is arr
    assert res.dtype == target_dtype


def test_array_to_precision_mismatching_dtype_no_copy():
    """Test array_to_precision with mismatched dtype and copy=False."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    arr = np.array([1.0, 2.0, 3.0], dtype=other_dtype)

    # With copy=False and mismatched dtype, should raise error
    with pytest.raises(TypeError, match="Array has dtype"):
        precision.array_to_precision(arr, copy=False)


def test_array_to_precision_mismatching_dtype_with_copy():
    """Test array_to_precision with mismatched dtype and copy=True."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    arr = np.array([1.0, 2.0, 3.0], dtype=other_dtype)

    # With copy=True and mismatched dtype, should convert
    res = precision.array_to_precision(arr, copy=True)
    assert res.dtype == target_dtype
    assert np.allclose(res, arr)
    assert res is not arr


def test_array_to_precision_from_list():
    """Test array_to_precision with list input."""
    target_dtype = precision.get_numpy_dtype()
    data = [1.0, 2.0, 3.0]

    res = precision.array_to_precision(data, copy=True)
    assert res.dtype == target_dtype
    assert np.allclose(res, data)


def test_array_to_precision_explicit_target_dtype():
    """Test array_to_precision with explicit target dtype."""
    # Force target dtype different from compiled precision
    compiled_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if compiled_dtype == np.float32 else np.float32

    arr = np.array([1.0, 2.0, 3.0], dtype=compiled_dtype)
    res = precision.array_to_precision(
        arr, copy=True, target_dtype=other_dtype
    )
    assert res.dtype == other_dtype
    assert np.allclose(res, arr)


def test_array_to_precision_non_contiguous():
    """Test array_to_precision with non-contiguous array."""
    target_dtype = precision.get_numpy_dtype()
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=target_dtype)
    # Get a non-contiguous slice
    arr_nc = arr[::1, ::2]
    assert not arr_nc.flags["C_CONTIGUOUS"]

    res = precision.array_to_precision(arr_nc, copy=True)
    assert res.dtype == target_dtype
    assert res.flags["C_CONTIGUOUS"]
    assert np.allclose(res, arr_nc)


# Test scalar conversion functions


def test_scalar_to_precision_matching_dtype():
    """Test scalar_to_precision with matching dtype."""
    target_dtype = precision.get_numpy_dtype()
    # Keep as numpy scalar to preserve dtype
    scalar = np.asarray(3.14, dtype=target_dtype)

    # With copy=False, should return same value if dtype matches
    res = precision.scalar_to_precision(scalar, copy=False)
    assert res == scalar.item()


def test_scalar_to_precision_mismatching_dtype_no_copy():
    """Test scalar_to_precision with mismatched dtype and copy=False."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    scalar = np.asarray(3.14, dtype=other_dtype).item()

    # With copy=False and mismatched dtype, should raise error
    with pytest.raises(TypeError, match="Scalar has dtype"):
        precision.scalar_to_precision(scalar, copy=False)


def test_scalar_to_precision_mismatching_dtype_with_copy():
    """Test scalar_to_precision with mismatched dtype and copy=True."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    scalar = np.asarray(3.14, dtype=other_dtype).item()

    # With copy=True and mismatched dtype, should convert
    res = precision.scalar_to_precision(scalar, copy=True)
    assert np.isclose(res, scalar)


def test_scalar_to_precision_python_float():
    """Test scalar_to_precision with Python float."""
    scalar = 3.14

    res = precision.scalar_to_precision(scalar, copy=True)
    assert isinstance(res, (float, np.floating))


def test_scalar_to_precision_python_int():
    """Test scalar_to_precision with Python int."""
    scalar = 42

    res = precision.scalar_to_precision(scalar, copy=True)
    # Should be converted to float since target is float
    assert isinstance(res, (float, np.floating))


def test_scalar_to_precision_explicit_target_dtype():
    """Test scalar_to_precision with explicit target dtype."""
    compiled_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if compiled_dtype == np.float32 else np.float32

    scalar = np.asarray(3.14, dtype=compiled_dtype)
    res = precision.scalar_to_precision(
        scalar, copy=True, target_dtype=other_dtype
    )
    assert np.isclose(res, scalar.item())


# Test unyt array conversion functions


def test_unyt_array_to_precision_matching_dtype():
    """Test unyt_array_to_precision with matching dtype."""
    target_dtype = precision.get_numpy_dtype()
    arr = unyt_array([1.0, 2.0, 3.0], "m", dtype=target_dtype)

    # With copy=False, should return same object if dtype matches
    res = precision.unyt_array_to_precision(arr, copy=False)
    assert res is arr
    assert res.dtype == target_dtype


def test_unyt_array_to_precision_mismatching_dtype():
    """Test unyt_array_to_precision with mismatched dtype."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    arr = unyt_array([1.0, 2.0, 3.0], "m", dtype=other_dtype)

    res = precision.unyt_array_to_precision(arr, copy=True)
    assert res.dtype == target_dtype
    assert np.allclose(res, arr)
    assert res.units == arr.units


def test_unyt_array_to_precision_explicit_target_dtype():
    """Test unyt_array_to_precision with explicit target dtype."""
    compiled_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if compiled_dtype == np.float32 else np.float32

    arr = unyt_array([1.0, 2.0, 3.0], "m", dtype=compiled_dtype)
    res = precision.unyt_array_to_precision(
        arr, copy=True, target_dtype=other_dtype
    )
    assert res.dtype == other_dtype
    assert np.allclose(res, arr)


def test_unyt_array_to_precision_invalid_input():
    """Test unyt_array_to_precision with invalid input type."""
    with pytest.raises(TypeError, match="Input must be a unyt.unyt_array"):
        precision.unyt_array_to_precision(np.array([1.0, 2.0]))


def test_unyt_array_to_precision_non_contiguous():
    """Test unyt_array_to_precision with non-contiguous array."""
    target_dtype = precision.get_numpy_dtype()
    arr = unyt_array([[1.0, 2.0], [3.0, 4.0]], "m", dtype=target_dtype)
    # Get a non-contiguous slice
    arr_nc = arr[::1, ::2]
    assert not arr_nc.flags["C_CONTIGUOUS"]

    res = precision.unyt_array_to_precision(arr_nc, copy=True)
    assert res.dtype == target_dtype
    assert res.flags["C_CONTIGUOUS"]


# Test ensure_arg_precision function


def test_ensure_arg_precision_numpy_array():
    """Test ensure_arg_precision with numpy array."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    arr = np.array([1.0, 2.0, 3.0], dtype=other_dtype)

    res = precision.ensure_arg_precision(arr, copy=True)
    assert res.dtype == target_dtype
    assert np.allclose(res, arr)


def test_ensure_arg_precision_unyt_array():
    """Test ensure_arg_precision with unyt_array."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    arr = unyt_array([1.0, 2.0, 3.0], "m", dtype=other_dtype)

    res = precision.ensure_arg_precision(arr, copy=True)
    assert res.dtype == target_dtype
    assert np.allclose(res, arr)


def test_ensure_arg_precision_unyt_quantity():
    """Test ensure_arg_precision with unyt_quantity."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32
    qty = unyt_quantity(3.14, "m", dtype=other_dtype)

    res = precision.ensure_arg_precision(qty, copy=True)
    assert res.dtype == target_dtype
    assert np.isclose(res, qty)


def test_ensure_arg_precision_scalar():
    """Test ensure_arg_precision with scalar."""
    scalar = 3.14

    res = precision.ensure_arg_precision(scalar, copy=True)
    assert isinstance(res, (float, np.floating))


def test_ensure_arg_precision_int_default_dtype():
    """Test ensure_arg_precision uses integer dtype by default."""
    int_value = 42

    res = precision.ensure_arg_precision(int_value, copy=True)
    assert isinstance(res, (int, np.integer))
    assert np.asarray(res).dtype == precision.get_integer_dtype()


def test_ensure_arg_precision_bool_default_dtype():
    """Test ensure_arg_precision uses boolean dtype by default."""
    bool_value = True

    res = precision.ensure_arg_precision(bool_value, copy=True)
    assert isinstance(res, (bool, np.bool_))
    assert np.asarray(res).dtype == np.dtype(np.bool_)


def test_ensure_arg_precision_other_type():
    """Test ensure_arg_precision with unsupported type (returns unchanged)."""
    # String should be returned unchanged
    string = "test"
    res = precision.ensure_arg_precision(string, copy=True)
    assert res == string


def test_ensure_arg_precision_non_numeric_array():
    """Test ensure_arg_precision with non-numeric array."""
    # String array should be returned unchanged
    arr = np.array(["a", "b", "c"], dtype=str)
    res = precision.ensure_arg_precision(arr, copy=True)
    assert res is arr
    assert res.dtype == arr.dtype


# Test ensure_compatible_precision function


def test_ensure_compatible_precision_float_array():
    """Test ensure_compatible_precision with float array."""
    target_dtype = precision.get_numpy_dtype()
    arr = np.array([1.0, 2.0, 3.0], dtype=target_dtype)

    # Should not raise for valid values
    precision.ensure_compatible_precision(arr, target_dtype)


def test_ensure_compatible_precision_integer_array():
    """Test ensure_compatible_precision with integer array."""
    int_dtype = precision.get_integer_dtype()
    arr = np.array([1, 2, 3], dtype=int_dtype)

    # Should not raise for valid values
    precision.ensure_compatible_precision(arr, int_dtype)


def test_ensure_compatible_precision_overflow_int32():
    """Test ensure_compatible_precision detects integer overflow for int32."""
    arr = np.array([2**31], dtype=np.int64)

    # Should raise for overflow to int32
    with pytest.raises(OverflowError):
        precision.ensure_compatible_precision(arr, np.int32)


def test_ensure_compatible_precision_overflow_float32():
    """Test ensure_compatible_precision detects float overflow for float32."""
    # Use a value that's large but not infinite when represented as float64
    large_value = float(np.finfo(np.float32).max) * 1.5
    arr = np.array([large_value], dtype=np.float64)

    # Should raise for overflow to float32
    with pytest.raises(OverflowError):
        precision.ensure_compatible_precision(arr, np.float32)


def test_ensure_compatible_precision_unyt_array():
    """Test ensure_compatible_precision with unyt_array."""
    target_dtype = precision.get_numpy_dtype()
    arr = unyt_array([1.0, 2.0, 3.0], "m", dtype=target_dtype)

    # Should not raise for valid values
    precision.ensure_compatible_precision(arr, target_dtype)


def test_ensure_compatible_precision_unyt_quantity():
    """Test ensure_compatible_precision with unyt_quantity."""
    target_dtype = precision.get_numpy_dtype()
    qty = unyt_quantity(3.14, "m", dtype=target_dtype)

    # Should not raise for valid values
    precision.ensure_compatible_precision(qty, target_dtype)


def test_ensure_compatible_precision_scalar():
    """Test ensure_compatible_precision with scalar."""
    target_dtype = precision.get_numpy_dtype()

    # Should not raise for valid values
    precision.ensure_compatible_precision(3.14, target_dtype)


def test_ensure_compatible_precision_invalid_dtype():
    """Test ensure_compatible_precision with invalid dtype."""
    arr = np.array([1.0, 2.0, 3.0])

    # Should raise for non-numeric dtype
    with pytest.raises(TypeError, match="not a numeric type"):
        precision.ensure_compatible_precision(arr, np.dtype("U10"))


def test_ensure_compatible_precision_finite_values():
    """Test ensure_compatible_precision with inf and nan."""
    target_dtype = precision.get_numpy_dtype()
    arr = np.array([1.0, np.inf, np.nan, 2.0], dtype=target_dtype)

    # Should not raise - inf and nan are allowed
    precision.ensure_compatible_precision(arr, target_dtype)


def test_ensure_compatible_precision_all_non_finite():
    """Test ensure_compatible_precision with all non-finite values."""
    target_dtype = precision.get_numpy_dtype()
    arr = np.array([np.inf, np.nan, np.inf], dtype=target_dtype)

    # Should warn but not raise when all values are non-finite
    with pytest.warns(match="All values are non-finite"):
        precision.ensure_compatible_precision(arr, target_dtype)


def test_array_to_precision_non_contiguous_with_copy():
    """Test array_to_precision with non-contiguous array and copy=True."""
    target_dtype = precision.get_numpy_dtype()
    # Create a non-contiguous array by transposing
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=target_dtype)
    arr_nc = arr[:, ::2]  # Column slice creates non-contiguous array

    if not arr_nc.flags["C_CONTIGUOUS"]:
        res = precision.array_to_precision(
            arr_nc, copy=True, target_dtype=target_dtype
        )
        assert res.dtype == target_dtype
        assert res.flags["C_CONTIGUOUS"]
        assert np.allclose(res, arr_nc)
    else:
        pytest.skip("Could not create non-contiguous array on this platform")


def test_array_to_precision_non_contiguous_no_copy_error():
    """Test array_to_precision raises error for non-contiguous no-copy."""
    target_dtype = precision.get_numpy_dtype()
    other_dtype = np.float64 if target_dtype == np.float32 else np.float32

    # Create a non-contiguous array
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=other_dtype)
    arr_nc = arr[:, ::2]  # Column slice creates non-contiguous array

    if not arr_nc.flags["C_CONTIGUOUS"]:
        # Should raise because we have dtype mismatch, non-contiguous,
        # and copy=False
        with pytest.raises(TypeError, match="Array has dtype"):
            precision.array_to_precision(
                arr_nc, copy=False, target_dtype=target_dtype
            )
    else:
        msg = "Could not create non-contiguous array on this platform"
        pytest.skip(msg)


# Parametrized tests for consistency across dtypes


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_ensure_compatible_precision_integer_dtypes(dtype):
    """Test ensure_compatible_precision works for all integer dtypes."""
    arr = np.array([1, 2, 3], dtype=dtype)
    precision.ensure_compatible_precision(arr, dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_array_to_precision_all_float_dtypes(dtype):
    """Test array_to_precision works for all float dtypes."""
    arr = np.array([1.0, 2.0, 3.0], dtype=dtype)
    res = precision.array_to_precision(arr, copy=True, target_dtype=dtype)
    assert res.dtype == dtype


# Tests for accept_precisions decorator


def test_accept_precisions_decorator_basic():
    """Test accept_precisions decorator with basic float argument."""
    target_dtype = precision.get_numpy_dtype()

    @precision.accept_precisions(value=target_dtype)
    def process_value(value):
        return value

    # Test with matching dtype
    arr = np.array([1.0, 2.0], dtype=target_dtype)
    result = process_value(arr)
    assert result.dtype == target_dtype
    assert np.allclose(result, arr)


def test_accept_precisions_decorator_mixed_args():
    """Test accept_precisions with positional and keyword arguments."""
    float_dtype = precision.get_numpy_dtype()
    int_dtype = precision.get_integer_dtype()

    @precision.accept_precisions(
        data=float_dtype,
        count=int_dtype,
    )
    def process_data(data, count):
        return data, count

    data_arr = np.array([1.0, 2.0], dtype=float_dtype)
    count_arr = np.array([1, 2], dtype=int_dtype)

    result_data, result_count = process_data(data_arr, count_arr)
    assert result_data.dtype == float_dtype
    assert result_count.dtype == int_dtype


def test_accept_precisions_decorator_no_conversion_needed():
    """Test accept_precisions when no conversion is needed."""
    target_dtype = precision.get_numpy_dtype()

    @precision.accept_precisions(value=target_dtype)
    def process_value(value):
        return value

    # Test with matching dtype - should return same object
    arr = np.array([1.0, 2.0], dtype=target_dtype)
    result = process_value(arr)
    assert result.dtype == target_dtype


def test_accept_precisions_decorator_with_kwargs():
    """Test accept_precisions decorator with keyword arguments."""
    target_dtype = precision.get_numpy_dtype()

    @precision.accept_precisions(value=target_dtype)
    def process_value(value):
        return value

    arr = np.array([1.0, 2.0], dtype=target_dtype)
    result = process_value(value=arr)
    assert result.dtype == target_dtype


def test_accept_precisions_decorator_unyt_array():
    """Test accept_precisions with unyt arrays."""
    target_dtype = precision.get_numpy_dtype()

    @precision.accept_precisions(wavelength=target_dtype)
    def process_wavelength(wavelength):
        return wavelength

    arr = unyt_array([100, 200], "angstrom", dtype=target_dtype)
    result = process_wavelength(arr)
    assert result.dtype == target_dtype
    assert result.units == arr.units


def test_accept_precisions_decorator_unyt_quantity():
    """Test accept_precisions with unyt quantities."""
    target_dtype = precision.get_numpy_dtype()

    @precision.accept_precisions(wavelength=target_dtype)
    def process_wavelength(wavelength):
        return wavelength

    qty = unyt_quantity(100.0, "angstrom", dtype=target_dtype)
    result = process_wavelength(qty)
    assert result.dtype == target_dtype
    assert result.units == qty.units


def test_accept_precisions_decorator_partial_args():
    """Test accept_precisions with partial argument specification."""
    target_dtype = precision.get_numpy_dtype()

    @precision.accept_precisions(value=target_dtype)
    def process_data(value, other):
        return value, other

    arr = np.array([1.0, 2.0], dtype=target_dtype)
    other = "unchanged"
    result_arr, result_other = process_data(arr, other)
    assert result_arr.dtype == target_dtype
    assert result_other == other


def test_accept_precisions_decorator_none_argument():
    """Test accept_precisions skips None arguments."""
    target_dtype = precision.get_numpy_dtype()

    @precision.accept_precisions(value=target_dtype)
    def process_data(value):
        return value

    # Should handle None gracefully
    result = process_data(None)
    assert result is None


def test_accept_precisions_decorator_multiple_args():
    """Test accept_precisions with multiple precision targets."""
    float_dtype = precision.get_numpy_dtype()
    int_dtype = precision.get_integer_dtype()

    @precision.accept_precisions(
        floats=float_dtype,
        ints=int_dtype,
    )
    def process_arrays(floats, ints):
        return floats, ints

    float_arr = np.array([1.5, 2.5], dtype=float_dtype)
    int_arr = np.array([1, 2], dtype=int_dtype)

    result_float, result_int = process_arrays(floats=float_arr, ints=int_arr)
    assert result_float.dtype == float_dtype
    assert result_int.dtype == int_dtype
