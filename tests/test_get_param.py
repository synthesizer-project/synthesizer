"""Comprehensive test suite for get_param and ParameterFunction."""

import numpy as np
import pytest
import unyt

from synthesizer import exceptions
from synthesizer.emission_models.utils import ParameterFunction, get_param


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


class TestGetParamParameterFunction:
    """Test ParameterFunction handling in get_param."""

    def test_get_param_with_parameter_function(self):
        """Test that ParameterFunction is called correctly."""

        def compute_value(test_emitter_attr):
            return test_emitter_attr * 2

        model = MockModel()
        func = ParameterFunction(
            compute_value, "computed", ["test_emitter_attr"]
        )
        model.fixed_parameters["computed"] = func
        emitter = MockEmitter()

        result = get_param("computed", model, None, emitter)
        assert result == 200.0

    def test_get_param_parameter_function_with_multiple_args(self):
        """Test ParameterFunction with multiple arguments."""

        def compute_sum(test_emitter_attr, test_emission_attr):
            return test_emitter_attr + test_emission_attr

        model = MockModel()
        func = ParameterFunction(
            compute_sum, "sum", ["test_emitter_attr", "test_emission_attr"]
        )
        model.fixed_parameters["sum"] = func
        emission = MockEmission()
        emitter = MockEmitter()

        result = get_param("sum", model, emission, emitter)
        assert result == 142.0

    def test_get_param_parameter_function_with_fixed_params(self):
        """Test ParameterFunction accessing other fixed_parameters."""

        def compute_scaled(test_emitter_attr, scale):
            return test_emitter_attr * scale

        model = MockModel()
        model.fixed_parameters["scale"] = 3.0
        func = ParameterFunction(
            compute_scaled, "scaled", ["test_emitter_attr", "scale"]
        )
        model.fixed_parameters["scaled"] = func
        emitter = MockEmitter()

        result = get_param("scaled", model, None, emitter)
        assert result == 300.0


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


class TestParameterFunction:
    """Test ParameterFunction class."""

    def test_parameter_function_init(self):
        """Test ParameterFunction initialization."""

        def test_func(arg1, arg2):
            return arg1 + arg2

        param_func = ParameterFunction(test_func, "result", ["arg1", "arg2"])
        assert param_func.func == test_func
        assert param_func.func_args == ["arg1", "arg2"]
        assert param_func.sets == "result"

    def test_parameter_function_init_validates_signature(self):
        """Test that ParameterFunction validates function signature."""

        def test_func(arg1):
            return arg1

        # Should raise because func_args doesn't match function signature
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterFunction(test_func, "test", ["wrong_arg"])

    def test_parameter_function_init_validates_missing_args(self):
        """Test that ParameterFunction validates all args are in func_args."""

        def test_func(arg1, arg2):
            return arg1 + arg2

        # Should raise because arg2 is not in func_args
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterFunction(test_func, "test", ["arg1"])

    def test_parameter_function_call_basic(self):
        """Test calling ParameterFunction with basic arguments."""

        def test_func(arg1, arg2):
            return arg1 * arg2

        param_func = ParameterFunction(test_func, "result", ["arg1", "arg2"])
        model = MockModel()
        model.fixed_parameters["arg1"] = 5.0
        model.fixed_parameters["arg2"] = 3.0
        emitter = MockEmitter()

        result = param_func(model, None, emitter, None)
        assert result == 15.0

    def test_parameter_function_call_from_emitter(self):
        """Test ParameterFunction accessing emitter attributes."""

        def test_func(test_emitter_attr):
            return test_emitter_attr * 2

        param_func = ParameterFunction(
            test_func, "doubled", ["test_emitter_attr"]
        )
        model = MockModel()
        emitter = MockEmitter()

        result = param_func(model, None, emitter, None)
        assert result == 200.0

    def test_parameter_function_call_from_emission(self):
        """Test ParameterFunction accessing emission attributes."""

        def test_func(test_emission_attr):
            return test_emission_attr + 10

        param_func = ParameterFunction(
            test_func, "modified", ["test_emission_attr"]
        )
        model = MockModel()
        emission = MockEmission()
        emitter = MockEmitter()

        result = param_func(model, emission, emitter, None)
        assert result == 52.0

    def test_parameter_function_call_mixed_sources(self):
        """Test ParameterFunction with arguments from multiple sources."""

        def test_func(fixed_val, emitter_val, emission_val):
            return fixed_val + emitter_val + emission_val

        param_func = ParameterFunction(
            test_func, "sum", ["fixed_val", "emitter_val", "emission_val"]
        )
        model = MockModel()
        model.fixed_parameters["fixed_val"] = 10.0
        emission = MockEmission()
        emission.emitter_val = 20.0
        emitter = MockEmitter()
        emitter.emission_val = 30.0

        result = param_func(model, emission, emitter, None)
        assert result == 60.0

    def test_parameter_function_with_obj(self):
        """Test ParameterFunction with obj argument."""

        def test_func(test_obj_attr):
            return test_obj_attr / 2

        param_func = ParameterFunction(test_func, "halved", ["test_obj_attr"])
        model = MockModel()
        # Put the value in model fixed_parameters instead
        model.fixed_parameters["test_obj_attr"] = 200.0
        obj = MockObject()
        emitter = (
            MockEmitter()
        )  # ParameterFunction requires an emitter for caching

        result = param_func(model, None, emitter, obj)
        assert result == 100.0

    def test_parameter_function_missing_argument(self):
        """Test ParameterFunction with missing argument raises error."""

        def test_func(nonexistent_arg):
            return nonexistent_arg

        param_func = ParameterFunction(test_func, "fail", ["nonexistent_arg"])
        model = MockModel()
        emitter = MockEmitter()

        with pytest.raises(exceptions.MissingAttribute):
            param_func(model, None, emitter, None)

    def test_parameter_function_with_arrays(self):
        """Test ParameterFunction with array operations."""

        def test_func(ages):
            return np.mean(ages)

        param_func = ParameterFunction(test_func, "mean_age", ["ages"])
        model = MockModel()
        emitter = MockEmitter()

        result = param_func(model, None, emitter, None)
        assert result == 2.0

    def test_parameter_function_complex_computation(self):
        """Test ParameterFunction with complex computation."""

        def test_func(ages, initial_masses):
            # Compute mass-weighted age
            return np.sum(ages * initial_masses) / np.sum(initial_masses)

        param_func = ParameterFunction(
            test_func, "mass_weighted_age", ["ages", "initial_masses"]
        )
        model = MockModel()
        emitter = MockEmitter()

        result = param_func(model, None, emitter, None)
        expected = np.sum(emitter.ages * emitter.initial_masses) / np.sum(
            emitter.initial_masses
        )
        assert np.isclose(result, expected)


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


class TestParameterFunctionErrorHandling:
    """Test suite for ParameterFunction error handling."""

    def test_parameter_function_non_callable_raises(self):
        """Test ParameterFunction raises ValueError for non-callable."""
        not_a_function = "this is not a function"

        with pytest.raises(ValueError, match="func must be a callable"):
            ParameterFunction(not_a_function, "test", [])

    def test_parameter_function_call_error_propagates(self):
        """Test that errors in wrapped function are caught, re-raised."""

        def failing_func(some_param):
            raise RuntimeError("Something went wrong in the function")

        param_func = ParameterFunction(failing_func, "result", ["some_param"])
        model = MockModel()
        model.fixed_parameters["some_param"] = 42.0
        emitter = MockEmitter()

        with pytest.raises(
            exceptions.ParameterFunctionError,
            match="Error calling ParameterFunction 'failing_func'",
        ):
            param_func(model, None, emitter, None)

    def test_parameter_function_call_error_preserves_original(self):
        """Test that the original exception is chained properly."""

        def failing_func(some_param):
            raise ValueError("Original error message")

        param_func = ParameterFunction(failing_func, "result", ["some_param"])
        model = MockModel()
        model.fixed_parameters["some_param"] = 42.0
        emitter = MockEmitter()

        try:
            param_func(model, None, emitter, None)
        except exceptions.ParameterFunctionError as e:
            # Check that the original exception is chained
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, ValueError)
            assert "Original error message" in str(e.__cause__)
