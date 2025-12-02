"""Comprehensive test suite for get_param and cache_model_params."""

import numpy as np
import pytest
import unyt

from synthesizer import exceptions
from synthesizer.emission_models.utils import cache_model_params, get_param


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


class TestCacheModelParams:
    """Test suite for the cache_model_params function."""

    def test_cache_extracting_model(self, test_grid):
        """Test caching parameters for an extracting emission model."""
        from synthesizer.emission_models import IncidentEmission

        # Create an extraction model
        model = IncidentEmission(grid=test_grid, label="incident_extract")
        emitter = MockEmitter()

        # Cache the parameters
        cache_model_params(model, emitter)

        # Verify cache was created
        assert model.label in emitter.model_param_cache
        assert "extract" in emitter.model_param_cache[model.label]
        assert emitter.model_param_cache[model.label]["extract"] == (
            "incident"
        )

    def test_cache_combining_model(self, test_grid):
        """Test caching parameters for a combining emission model."""
        from synthesizer.emission_models import (
            IncidentEmission,
            StellarEmissionModel,
        )

        # Create sub-models
        model1 = IncidentEmission(grid=test_grid, label="incident_a")
        model2 = IncidentEmission(grid=test_grid, label="incident_b")

        # Create a combination model
        model = StellarEmissionModel(
            label="test_combine",
            combine=(model1, model2),
        )
        emitter = MockEmitter()

        # Cache the parameters
        cache_model_params(model, emitter)

        # Verify cache was created
        assert "test_combine" in emitter.model_param_cache
        assert "combine" in emitter.model_param_cache["test_combine"]
        assert emitter.model_param_cache["test_combine"]["combine"] == [
            "incident_a",
            "incident_b",
        ]

    def test_cache_transforming_model_with_repr(self):
        """Test caching transformer repr for a transforming emission model."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import PowerLaw

        # Create a dust curve
        dust_curve = PowerLaw(slope=-1.0)

        # Create a transformation model
        model = StellarEmissionModel(
            label="test_transform",
            apply_to="intrinsic",
            transformer=dust_curve,
        )
        emitter = MockEmitter()

        # Cache the parameters
        cache_model_params(model, emitter)

        # Verify cache was created
        assert "test_transform" in emitter.model_param_cache
        assert "apply_to" in emitter.model_param_cache["test_transform"]
        assert "transformer" in emitter.model_param_cache["test_transform"]
        assert emitter.model_param_cache["test_transform"]["apply_to"] == (
            "intrinsic"
        )
        # Verify the transformer repr was cached
        assert emitter.model_param_cache["test_transform"]["transformer"] == (
            "PowerLaw(slope=-1.0)"
        )

    def test_cache_generating_model_with_repr(self):
        """Test caching generator repr for a generating emission model."""
        from unyt import K

        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.generators.dust import Blackbody

        # Create a generator
        generator = Blackbody(temperature=100 * K)

        # Create a generation model
        model = StellarEmissionModel(
            label="test_generate",
            generator=generator,
        )
        emitter = MockEmitter()

        # Cache the parameters
        cache_model_params(model, emitter)

        # Verify cache was created
        assert "test_generate" in emitter.model_param_cache
        assert "generator" in emitter.model_param_cache["test_generate"]
        # Verify the generator repr was cached
        assert emitter.model_param_cache["test_generate"]["generator"] == (
            "Blackbody(temperature=100 K)"
        )

    def test_cache_model_with_masks(self, test_grid):
        """Test caching mask parameters for a masked emission model."""
        from synthesizer.emission_models import IncidentEmission

        # Create a model with masks
        model = IncidentEmission(grid=test_grid, label="incident_masked")
        model.add_mask(attr="ages", op=">", thresh=1.0 * unyt.Myr)
        emitter = MockEmitter()

        # Cache the parameters
        cache_model_params(model, emitter)

        # Verify cache was created
        assert model.label in emitter.model_param_cache
        assert "masks" in emitter.model_param_cache[model.label]
        # Verify mask format is correct
        mask_str = emitter.model_param_cache[model.label]["masks"]
        assert "ages" in mask_str
        assert ">" in mask_str
        assert "Myr" in mask_str

    def test_cache_model_with_multiple_masks(self, test_grid):
        """Test caching multiple mask parameters."""
        from synthesizer.emission_models import IncidentEmission

        # Create a model with multiple masks
        model = IncidentEmission(
            grid=test_grid,
            label="incident_multiple_masks",
        )
        model.add_mask(attr="ages", op=">", thresh=1.0 * unyt.Myr)
        model.add_mask(attr="ages", op="<", thresh=5.0 * unyt.Myr)
        emitter = MockEmitter()

        # Cache the parameters
        cache_model_params(model, emitter)

        # Verify cache was created
        assert model.label in emitter.model_param_cache
        assert "masks" in emitter.model_param_cache[model.label]
        # Verify masks are newline-separated
        masks = emitter.model_param_cache[model.label]["masks"]
        assert "\n" in masks
        lines = masks.split("\n")
        assert len(lines) == 2
        assert lines[0].startswith("ages > ")
        assert "Myr" in lines[0]
        assert lines[1].startswith("ages < ")
        assert "Myr" in lines[1]

    def test_cache_no_duplication_of_values(self, test_grid):
        """Test that cache stores references, not copies."""
        from synthesizer.emission_models import IncidentEmission

        # Create an extracting model with a known attribute
        model = IncidentEmission(
            grid=test_grid,
            label="incident_ref_cache",
        )
        emitter = MockEmitter()

        # Cache the parameters
        cache_model_params(model, emitter)

        # Cached value should be the exact same object (no duplication)
        cached_extract = emitter.model_param_cache[model.label]["extract"]
        assert cached_extract is model.extract

    def test_cache_multiple_models_on_same_emitter(self, test_grid):
        """Test caching parameters from multiple models on same emitter."""
        from synthesizer.emission_models import (
            IncidentEmission,
            IntrinsicEmission,
        )

        # Create multiple models
        model1 = IntrinsicEmission(grid=test_grid)
        model2 = IncidentEmission(grid=test_grid)
        emitter = MockEmitter()

        # Cache both models
        cache_model_params(model1, emitter)
        cache_model_params(model2, emitter)

        # Verify both caches exist
        assert model1.label in emitter.model_param_cache
        assert model2.label in emitter.model_param_cache
        assert emitter.model_param_cache[model1.label]["combine"] == [
            "escaped",
            "_intrinsic_reprocessed",
        ]
        assert emitter.model_param_cache[model2.label]["extract"] == "incident"


class TestCacheModelParamsWithDifferentTransformers:
    """Test caching with various transformer types."""

    def test_cache_attenuation_law_transformers(self):
        """Test caching different attenuation law transformer reprs."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import (
            MWN18,
            Calzetti2000,
            PowerLaw,
        )

        emitter = MockEmitter()

        # Test PowerLaw
        model1 = StellarEmissionModel(
            label="powerlaw",
            apply_to="test",
            transformer=PowerLaw(slope=-0.7),
        )
        cache_model_params(model1, emitter)
        assert emitter.model_param_cache["powerlaw"]["transformer"] == (
            "PowerLaw(slope=-0.7)"
        )

        # Test Calzetti2000
        model2 = StellarEmissionModel(
            label="calzetti",
            apply_to="test",
            transformer=Calzetti2000(),
        )
        cache_model_params(model2, emitter)
        assert (
            "Calzetti2000"
            in emitter.model_param_cache["calzetti"]["transformer"]
        )

        # Test MWN18
        model3 = StellarEmissionModel(
            label="mwn18",
            apply_to="test",
            transformer=MWN18(),
        )
        cache_model_params(model3, emitter)
        assert emitter.model_param_cache["mwn18"]["transformer"] == "MWN18()"

    def test_cache_escape_fraction_transformers(self):
        """Test caching escape fraction transformer reprs."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import (
            EscapedFraction,
            ProcessedFraction,
        )

        emitter = MockEmitter()

        # Test ProcessedFraction
        model1 = StellarEmissionModel(
            label="processed",
            apply_to="test",
            transformer=ProcessedFraction(fesc_attrs=("fesc",)),
        )
        cache_model_params(model1, emitter)
        assert (
            "ProcessedFraction"
            in emitter.model_param_cache["processed"]["transformer"]
        )

        # Test EscapedFraction
        model2 = StellarEmissionModel(
            label="escaped",
            apply_to="test",
            transformer=EscapedFraction(fesc_attrs=("fesc_ly_alpha",)),
        )
        cache_model_params(model2, emitter)
        assert (
            "EscapedFraction"
            in emitter.model_param_cache["escaped"]["transformer"]
        )

    def test_cache_igm_transformers(self):
        """Test caching IGM transformer reprs."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import (
            Asada25,
            Inoue14,
            Madau96,
        )

        emitter = MockEmitter()

        # Test Inoue14
        model1 = StellarEmissionModel(
            label="inoue14",
            apply_to="test",
            transformer=Inoue14(scale_tau=1.0),
        )
        cache_model_params(model1, emitter)
        assert emitter.model_param_cache["inoue14"]["transformer"] == (
            "Inoue14(scale_tau=1.0)"
        )

        # Test Madau96
        model2 = StellarEmissionModel(
            label="madau96",
            apply_to="test",
            transformer=Madau96(),
        )
        cache_model_params(model2, emitter)
        assert emitter.model_param_cache["madau96"]["transformer"] == (
            "Madau96()"
        )

        # Test Asada25
        model3 = StellarEmissionModel(
            label="asada25",
            apply_to="test",
            transformer=Asada25(scale_tau=1.0, add_cgm=True),
        )
        cache_model_params(model3, emitter)
        assert "Asada25" in emitter.model_param_cache["asada25"]["transformer"]


class TestCacheModelParamsWithDifferentGenerators:
    """Test caching with various generator types."""

    def test_cache_dust_emission_generators(self):
        """Test caching different dust emission generator reprs."""
        from unyt import K, um

        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.generators.dust import (
            Blackbody,
            Greybody,
        )

        emitter = MockEmitter()

        # Test Blackbody
        model1 = StellarEmissionModel(
            label="blackbody",
            generator=Blackbody(temperature=100 * K),
        )
        cache_model_params(model1, emitter)
        assert emitter.model_param_cache["blackbody"]["generator"] == (
            "Blackbody(temperature=100 K)"
        )

        # Test Greybody
        model2 = StellarEmissionModel(
            label="greybody",
            generator=Greybody(
                temperature=50 * K,
                emissivity=1.5,
                optically_thin=True,
                lam_0=100.0 * um,
            ),
        )
        cache_model_params(model2, emitter)
        assert "Greybody" in emitter.model_param_cache["greybody"]["generator"]

    def test_cache_casey12_generator(self):
        """Test caching Casey12 generator repr."""
        from unyt import K, um

        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.generators.dust import Casey12

        emitter = MockEmitter()

        model = StellarEmissionModel(
            label="casey12",
            generator=Casey12(
                temperature=50 * K,
                emissivity=2.0,
                alpha=2.0,
                n_bb=1.0,
                lam_0=200.0 * um,
            ),
        )
        cache_model_params(model, emitter)
        assert "Casey12" in emitter.model_param_cache["casey12"]["generator"]
