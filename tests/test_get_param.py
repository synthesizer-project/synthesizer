"""Comprehensive test suite for get_param."""

import numpy as np
import pytest
import unyt

from synthesizer import exceptions
from synthesizer.emission_models.utils import get_param


# Mock classes for testing
class MockModel:
    """Mock emission model for testing."""

    def __init__(self, label="test_model"):
        """Initialize mock model with label and fixed_parameters."""
        self.label = label
        self.fixed_parameters = {}


class MockEmission:
    """Mock emission object for testing."""

    def __init__(self):
        """Initialize mock emission with test attribute."""
        self.test_emission_attr = 42.0


class MockEmitter:
    """Mock emitter object for testing."""

    def __init__(self):
        """Initialize mock emitter with test attributes."""
        self.test_emitter_attr = 100.0
        self.ages = np.array([1.0, 2.0, 3.0])
        self.initial_masses = np.array([1e6, 2e6, 3e6])
        self.model_param_cache = {}


class MockObject:
    """Mock additional object for testing."""

    def __init__(self):
        """Initialize mock object with test attribute."""
        self.test_obj_attr = 200.0


class TestGetParamBasics:
    """Test basic functionality of get_param."""

    def test_get_param_from_model_fixed_parameters(self):
        """Test getting parameter from model's fixed_parameters."""
        model = MockModel()
        model.fixed_parameters["test_param"] = 10.0

        result = get_param("test_param", model, None, None)
        assert result == 10.0

    def test_get_param_from_emission(self):
        """Test getting parameter from emission object."""
        model = MockModel()
        emission = MockEmission()

        result = get_param("test_emission_attr", model, emission, None)
        assert result == 42.0

    def test_get_param_from_emitter(self):
        """Test getting parameter from emitter object."""
        model = MockModel()
        emitter = MockEmitter()

        result = get_param("test_emitter_attr", model, None, emitter)
        assert result == 100.0

    def test_get_param_from_obj(self):
        """Test getting parameter from additional object."""
        model = MockModel()
        obj = MockObject()

        result = get_param("test_obj_attr", model, None, None, obj=obj)
        assert result == 200.0

    def test_get_param_priority_model_first(self):
        """Test that model fixed_parameters have highest priority."""
        model = MockModel()
        model.fixed_parameters["test_param"] = 10.0
        emission = MockEmission()
        emission.test_param = 20.0
        emitter = MockEmitter()
        emitter.test_param = 30.0

        result = get_param("test_param", model, emission, emitter)
        assert result == 10.0

    def test_get_param_priority_emission_second(self):
        """Test that emission has second priority."""
        model = MockModel()
        emission = MockEmission()
        emission.test_param = 20.0
        emitter = MockEmitter()
        emitter.test_param = 30.0

        result = get_param("test_param", model, emission, emitter)
        assert result == 20.0

    def test_get_param_priority_emitter_third(self):
        """Test that emitter has third priority."""
        model = MockModel()
        emitter = MockEmitter()
        emitter.test_param = 30.0
        obj = MockObject()
        obj.test_param = 40.0

        result = get_param("test_param", model, None, emitter, obj=obj)
        assert result == 30.0

    def test_get_param_priority_obj_last(self):
        """Test that obj has lowest priority."""
        model = MockModel()
        obj = MockObject()
        obj.test_param = 40.0

        result = get_param("test_param", model, None, None, obj=obj)
        assert result == 40.0


class TestGetParamDefaults:
    """Test default value handling in get_param."""

    def test_get_param_no_default_raises(self):
        """Test that MissingAttribute is raised when no default."""
        model = MockModel()
        with pytest.raises(exceptions.MissingAttribute):
            get_param("nonexistent_param", model, None, None)

    def test_get_param_with_direct_param(self):
        """Test that parameters found directly don't use default."""
        model = MockModel()
        model.fixed_parameters["test_param"] = 10.0

        result = get_param("test_param", model, None, None, default=999.0)
        assert result == 10.0


class TestGetParamStringIndirection:
    """Test string value indirection in get_param."""

    def test_get_param_string_indirection(self):
        """Test that string values are looked up recursively."""
        model = MockModel()
        model.fixed_parameters["alias"] = "test_emitter_attr"
        emitter = MockEmitter()

        result = get_param("alias", model, None, emitter)
        assert result == 100.0

    def test_get_param_string_indirection_not_found(self):
        """Test string indirection with nonexistent target."""
        model = MockModel()
        model.fixed_parameters["alias"] = "nonexistent"
        emitter = MockEmitter()

        with pytest.raises(exceptions.MissingAttribute):
            get_param("alias", model, None, emitter)


class TestGetParamLogged:
    """Test log10 parameter handling in get_param."""

    def test_get_param_log10_from_nonlogged(self):
        """Test that log10 parameter is computed from non-logged version."""
        model = MockModel()
        emitter = MockEmitter()

        result = get_param("log10test_emitter_attr", model, None, emitter)
        assert np.isclose(result, np.log10(100.0))

    def test_get_param_log10_caches_on_emitter(self):
        """Test that log10 value is cached on emitter."""
        model = MockModel()
        emitter = MockEmitter()

        get_param("log10test_emitter_attr", model, None, emitter)
        assert hasattr(emitter, "log10test_emitter_attr")
        assert np.isclose(emitter.log10test_emitter_attr, np.log10(100.0))

    def test_get_param_log10_array(self):
        """Test log10 with array values."""
        model = MockModel()
        emitter = MockEmitter()

        result = get_param("log10ages", model, None, emitter)
        expected = np.log10(np.array([1.0, 2.0, 3.0]))
        assert np.allclose(result, expected)


class TestGetParamPluralSingular:
    """Test plural/singular parameter name handling."""

    def test_get_param_pluralize(self):
        """Test that singular is pluralized to find parameter."""
        model = MockModel()
        emitter = MockEmitter()

        # Looking for 'age' should find 'ages'
        result = get_param("age", model, None, emitter)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))

    def test_get_param_depluralize(self):
        """Test that plural is depluralized to find parameter."""
        model = MockModel()
        emitter = MockEmitter()
        emitter.mass = 1e6

        # Looking for 'masses' should find 'mass'
        result = get_param("masses", model, None, emitter)
        assert result == 1e6

    def test_get_param_pluralize_with_initial_mass(self):
        """Test pluralization with 'initial_mass' -> 'initial_masses'."""
        model = MockModel()
        emitter = MockEmitter()

        # Looking for 'initial_mass' should find 'initial_masses'
        result = get_param("initial_mass", model, None, emitter)
        assert np.array_equal(result, np.array([1e6, 2e6, 3e6]))

    def test_get_param_no_infinite_pluralize_loop(self):
        """Test that pluralization doesn't create infinite loops."""
        model = MockModel()
        emitter = MockEmitter()

        # This should not cause infinite recursion
        with pytest.raises(exceptions.MissingAttribute):
            get_param("nonexistent", model, None, emitter)


class TestGetParamCaching:
    """Test parameter caching behavior."""

    def test_get_param_caches_value(self):
        """Test that values are cached on emitter."""
        model = MockModel()
        emitter = MockEmitter()
        emitter.test_param = 42.0

        get_param("test_param", model, None, emitter)

        # Check cache was created
        assert "test_model" in emitter.model_param_cache
        assert "test_param" in emitter.model_param_cache["test_model"]
        assert emitter.model_param_cache["test_model"]["test_param"] == 42.0

    def test_get_param_cache_is_passive(self):
        """Test that cache is stored but not used for lookups.

        NOTE: The cache in get_param is passive - it stores values for
        inspection but doesn't use them to speed up lookups.
        """
        model = MockModel()
        emitter = MockEmitter()
        emitter.test_param = 42.0

        # First call caches the value
        result1 = get_param("test_param", model, None, emitter)
        assert result1 == 42.0

        # Modify the attribute
        emitter.test_param = 999.0

        # Second call gets the NEW value (cache not used for lookups)
        result2 = get_param("test_param", model, None, emitter)
        assert result2 == 999.0

    def test_get_param_no_cache_without_model(self):
        """Test that values aren't cached without model."""
        emitter = MockEmitter()
        emitter.test_param = 42.0

        get_param("test_param", None, None, emitter)

        # Cache should not be created
        assert len(emitter.model_param_cache) == 0

    def test_get_param_no_cache_without_emitter(self):
        """Test that values aren't cached without emitter."""
        model = MockModel()
        model.fixed_parameters["test_param"] = 42.0

        result = get_param("test_param", model, None, None)
        assert result == 42.0


class TestGetParamArrayConversion:
    """Test C-compatible array conversion in get_param."""

    def test_get_param_converts_list_to_array(self):
        """Test that lists are converted to numpy arrays."""
        model = MockModel()
        model.fixed_parameters["test_list"] = [1.0, 2.0, 3.0]

        result = get_param("test_list", model, None, None)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64

    def test_get_param_converts_float32_to_float64(self):
        """Test that float32 arrays are converted to float64."""
        model = MockModel()
        model.fixed_parameters["test_array"] = np.array(
            [1.0, 2.0, 3.0], dtype=np.float32
        )

        result = get_param("test_array", model, None, None)
        assert result.dtype == np.float64

    def test_get_param_preserves_unyt_units(self):
        """Test that unyt units are stripped from fixed_parameters."""
        model = MockModel()
        test_val = unyt.unyt_array([1.0, 2.0, 3.0], "Myr")
        model.fixed_parameters["test_unyt"] = test_val

        result = get_param("test_unyt", model, None, None)
        # Units are stripped when converting from fixed_parameters
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64


class TestGetParamEdgeCases:
    """Test edge cases and error conditions."""

    def test_get_param_all_none_objects(self):
        """Test get_param with all None objects."""
        with pytest.raises(exceptions.MissingAttribute):
            get_param("test_param", None, None, None)

    def test_get_param_empty_string_parameter(self):
        """Test get_param with empty string."""
        model = MockModel()
        with pytest.raises(exceptions.MissingAttribute):
            get_param("", model, None, None)

    def test_get_param_with_none_value_on_emitter(self):
        """Test get_param raises when attribute is None (not found)."""
        model = MockModel()
        emitter = MockEmitter()
        emitter.none_attr = None

        with pytest.raises(exceptions.MissingAttribute):
            get_param("none_attr", model, None, emitter)

    def test_get_param_preserves_zero_values(self):
        """Test that zero values are preserved, not treated as missing."""
        model = MockModel()
        model.fixed_parameters["zero_val"] = 0.0

        result = get_param("zero_val", model, None, None)
        assert result == 0.0

    def test_get_param_preserves_negative_values(self):
        """Test that negative values work correctly."""
        model = MockModel()
        model.fixed_parameters["neg_val"] = -10.0

        result = get_param("neg_val", model, None, None)
        assert result == -10.0

    def test_get_param_with_special_characters_in_name(self):
        """Test parameter names with underscores and numbers."""
        model = MockModel()
        model.fixed_parameters["test_param_123"] = 42.0

        result = get_param("test_param_123", model, None, None)
        assert result == 42.0


class TestGetParams:
    """Test suite for the get_params helper function."""

    def test_get_params_multiple_parameters(self):
        """Test get_params retrieves multiple parameters at once."""
        from synthesizer.emission_models.utils import get_params

        model = MockModel()
        model.fixed_parameters["param1"] = 10.0
        model.fixed_parameters["param2"] = 20.0
        model.fixed_parameters["param3"] = 30.0

        result = get_params(["param1", "param2", "param3"], model, None, None)

        assert isinstance(result, dict)
        assert result["param1"] == 10.0
        assert result["param2"] == 20.0
        assert result["param3"] == 30.0

    def test_get_params_from_different_sources(self):
        """Test get_params retrieves parameters from different sources."""
        from synthesizer.emission_models.utils import get_params

        model = MockModel()
        model.fixed_parameters["param1"] = 100.0
        emission = MockEmission()
        emitter = MockEmitter()

        result = get_params(
            ["param1", "test_emission_attr", "test_emitter_attr"],
            model,
            emission,
            emitter,
        )

        assert result["param1"] == 100.0
        assert result["test_emission_attr"] == 42.0
        assert result["test_emitter_attr"] == 100.0

    def test_get_params_empty_list(self):
        """Test get_params with empty parameter list."""
        from synthesizer.emission_models.utils import get_params

        model = MockModel()
        result = get_params([], model, None, None)

        assert isinstance(result, dict)
        assert len(result) == 0
