"""A test suite for the utils module."""

import numpy as np
import pytest
import unyt

from synthesizer import exceptions
from synthesizer.utils import check_array_c_compatible_float
from synthesizer.utils.precision import get_numpy_dtype


class TestCheckArrayCCompatibleFloat:
    """A test suite for check_array_c_compatible_float."""

    def test_valid_scalar_passes(self):
        """Test that a valid scalar passes through unchanged."""
        value = 1.0
        result = check_array_c_compatible_float(value)
        assert result == value

    def test_none_passes(self):
        """Test that None passes through unchanged."""
        result = check_array_c_compatible_float(None)
        assert result is None

    def test_valid_array_passes(self):
        """Test that a valid C-contiguous array with correct dtype passes."""
        dtype = get_numpy_dtype()
        value = np.ascontiguousarray([1.0, 2.0, 3.0], dtype=dtype)
        result = check_array_c_compatible_float(value)
        assert result is value  # Should be the same object

    def test_valid_unyt_array_passes(self):
        """Test that a valid unyt array passes through."""
        dtype = get_numpy_dtype()
        value = unyt.unyt_array(
            np.ascontiguousarray([1.0, 2.0, 3.0], dtype=dtype),
            "cm",
        )
        result = check_array_c_compatible_float(value)
        assert result is value

    def test_list_raises_type_error(self):
        """Test that a list raises TypeError."""
        value = [1.0, 2.0, 3.0]
        with pytest.raises(TypeError, match="Expected a numpy array"):
            check_array_c_compatible_float(value)

    def test_wrong_dtype_raises_error(self):
        """Test that wrong dtype raises InconsistentArguments."""
        dtype = get_numpy_dtype()
        # Use the opposite dtype
        wrong_dtype = np.float32 if dtype == np.float64 else np.float64
        value = np.ascontiguousarray([1.0, 2.0, 3.0], dtype=wrong_dtype)
        with pytest.raises(
            exceptions.InconsistentArguments, match="incorrect dtype"
        ):
            check_array_c_compatible_float(value)

    def test_non_contiguous_raises_error(self):
        """Test that non-C-contiguous array raises InconsistentArguments."""
        dtype = get_numpy_dtype()
        # Create a non-contiguous array via slicing
        arr = np.ascontiguousarray(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=dtype
        )
        value = arr[:, ::2]  # Non-contiguous slice
        assert not value.flags["C_CONTIGUOUS"]
        with pytest.raises(
            exceptions.InconsistentArguments, match="not C contiguous"
        ):
            check_array_c_compatible_float(value)


class TestPluralization:
    """Test suite for pluralize and depluralize functions."""

    def test_pluralize_gas(self):
        """Test pluralize with 'gas' (ends in s but is singular)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("gas") == "gases"

    def test_pluralize_blackhole(self):
        """Test pluralize with 'blackhole' (codebase component)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("blackhole") == "blackholes"

    def test_depluralize_blackholes(self):
        """Test depluralize with 'blackholes' (codebase component)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("blackholes") == "blackhole"

    def test_pluralize_star(self):
        """Test pluralize with 'star' (codebase component)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("star") == "stars"

    def test_depluralize_stars(self):
        """Test depluralize with 'stars' (codebase component)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("stars") == "star"

    def test_depluralize_ages(self):
        """Test depluralize with 'ages'."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("ages") == "age"

    def test_depluralize_mass(self):
        """Test depluralize with 'mass' (should not depluralize)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("mass") == "mass"

    def test_depluralize_gas(self):
        """Test depluralize with 'gas' (should not depluralize)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("gas") == "gas"

    def test_pluralize_mass(self):
        """Test pluralize with 'mass' (common codebase attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("mass") == "masses"

    def test_depluralize_masses(self):
        """Test depluralize with 'masses' (common codebase attribute)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("masses") == "mass"

    def test_pluralize_axis(self):
        """Test pluralize with 'axis' (common grid attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("axis") == "axes"

    def test_depluralize_axes(self):
        """Test depluralize with 'axes' (common grid attribute)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("axes") == "axis"

    def test_pluralize_age(self):
        """Test pluralize with 'age' (common attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("age") == "ages"

    def test_depluralize_ages_real(self):
        """Test depluralize ages (common attribute check)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("ages") == "age"

    def test_pluralize_metallicity(self):
        """Test pluralize with 'metallicity' (common codebase attribute)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("metallicity") == "metallicities"

    def test_depluralize_metallicities(self):
        """Test depluralize with 'metallicities'."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("metallicities") == "metallicity"
