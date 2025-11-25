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

    def test_pluralize_bias(self):
        """Test pluralize with 'bias' (ends in s but is singular)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("bias") == "biases"

    def test_pluralize_lens(self):
        """Test pluralize with 'lens' (ends in s but is singular)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("lens") == "lenses"

    def test_pluralize_box(self):
        """Test pluralize with 'box' (x -> xes)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("box") == "boxes"

    def test_pluralize_church(self):
        """Test pluralize with 'church' (ch -> ches)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("church") == "churches"

    def test_pluralize_baby(self):
        """Test pluralize with 'baby' (y -> ies)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("baby") == "babies"

    def test_pluralize_leaf(self):
        """Test pluralize with 'leaf' (f -> ves)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("leaf") == "leaves"

    def test_pluralize_knife(self):
        """Test pluralize with 'knife' (fe -> ves)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("knife") == "knives"

    def test_pluralize_hero(self):
        """Test pluralize with 'hero' (o -> oes)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("hero") == "heroes"

    def test_pluralize_cat(self):
        """Test pluralize with 'cat' (simple s)."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("cat") == "cats"

    def test_pluralize_already_plural_boxes(self):
        """Test pluralize with already plural 'boxes'."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("boxes") == "boxes"

    def test_pluralize_already_plural_leaves(self):
        """Test pluralize with already plural 'leaves'."""
        from synthesizer.utils.util_funcs import pluralize

        assert pluralize("leaves") == "leaves"

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

    def test_depluralize_buses(self):
        """Test depluralize with 'buses'."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("buses") == "bus"

    def test_depluralize_bonuses(self):
        """Test depluralize with 'bonuses'."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("bonuses") == "bonus"

    def test_depluralize_radios(self):
        """Test depluralize with 'radios'."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("radios") == "radio"

    def test_depluralize_leaves(self):
        """Test depluralize with 'leaves' (ves -> f)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("leaves") == "leaf"

    def test_depluralize_knives(self):
        """Test depluralize with 'knives' (ves -> fe)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("knives") == "knife"

    def test_depluralize_wives(self):
        """Test depluralize with 'wives' (ves -> fe)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("wives") == "wife"

    def test_depluralize_babies(self):
        """Test depluralize with 'babies' (ies -> y)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("babies") == "baby"

    def test_depluralize_heroes(self):
        """Test depluralize with 'heroes' (oes -> o)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("heroes") == "hero"

    def test_depluralize_boxes(self):
        """Test depluralize with 'boxes' (xes -> x)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("boxes") == "box"

    def test_depluralize_churches(self):
        """Test depluralize with 'churches' (ches -> ch)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("churches") == "church"

    def test_depluralize_cats(self):
        """Test depluralize with 'cats' (simple s removal)."""
        from synthesizer.utils.util_funcs import depluralize

        assert depluralize("cats") == "cat"

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
