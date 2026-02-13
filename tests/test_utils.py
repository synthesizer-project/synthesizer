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
