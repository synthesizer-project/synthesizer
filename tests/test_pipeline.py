"""Tests for the pipeline module."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from unyt import unyt_array

from synthesizer import exceptions
from synthesizer.emissions import Sed
from synthesizer.pipeline.pipeline_utils import (
    cached_split,
    combine_list_of_dicts,
    count_and_check_dict_recursive,
    discover_attr_paths_recursive,
    discover_dict_recursive,
    discover_dict_structure,
    get_dataset_properties,
    get_full_memory,
    unify_dict_structure_across_ranks,
)
from synthesizer.units import Quantity


@pytest.fixture
def simple_class():
    """Create a simple class instance for testing."""

    class SimpleClass:
        quantity = Quantity("spatial")

        def __init__(self):
            self.public_attr = unyt_array([1, 2, 3], "erg/s")
            self._private_attr = "private"
            self.quantity = unyt_array([4, 5, 6], "Mpc")

        @property
        def property_attr(self):
            return unyt_array([7, 8, 9], "Msun")

    return SimpleClass()


@pytest.fixture
def nested_dict():
    """Create a nested dictionary for testing."""
    return {
        "level1": {
            "level2a": unyt_array([1, 2, 3], "erg/s"),
            "level2b": {
                "level3": unyt_array([4, 5, 6], "Msun"),
            },
        },
        "another_branch": unyt_array([7, 8, 9], "K"),
    }


class TestDiscoveryUtils:
    """Tests for the discovery utility functions."""

    def test_discover_attr_paths_recursive_dict(self, nested_dict):
        """Test discovering outputs recursively in a dictionary."""
        output_set = set()
        result = discover_attr_paths_recursive(
            nested_dict, prefix="", output_set=output_set
        )

        # Check that all paths are discovered
        assert "/level1/level2a" in result
        assert "/level1/level2b/level3" in result
        assert "/another_branch" in result
        assert len(result) == 3

    def test_discover_attr_paths_recursive_class(self, simple_class):
        """Test discovering outputs recursively in a class."""
        output_set = set()
        result = discover_attr_paths_recursive(
            simple_class, prefix="", output_set=output_set
        )

        # Check that public attributes are discovered but private ones aren't
        assert "/public_attr" in result
        assert "/property_attr" in result
        assert "/quantity" in result
        assert "/_private_attr" not in result

    def test_discover_attr_paths_recursive_none(self):
        """Test discovering outputs with None input."""
        output_set = set()
        result = discover_attr_paths_recursive(
            None, prefix="", output_set=output_set
        )

        # None should be skipped
        assert len(result) == 0

    def test_discover_attr_paths_recursive_simple_types(self):
        """Test discovering outputs with simple types that have no depth."""
        output_set = set()
        for value in ["string", True, False]:
            result = discover_attr_paths_recursive(
                value, prefix="", output_set=output_set
            )
            # Simple types should be skipped
            assert len(result) == 0

    def test_discover_attr_paths_recursive_array(self):
        """Test discovering outputs with arrays."""
        array = unyt_array([1, 2, 3], "erg/s")
        output_set = set()
        result = discover_attr_paths_recursive(
            array, prefix="/path", output_set=output_set
        )

        # Array should be added to the output set
        assert "/path" in result
        assert len(result) == 1

    def test_discover_dict_recursive(self, nested_dict):
        """Test discovering the structure of a nested dictionary."""
        output_set = set()
        result = discover_dict_recursive(
            nested_dict, prefix="", output_set=output_set
        )

        # Check paths in the result
        assert "level1/level2a" in result
        assert "level1/level2b/level3" in result
        assert "another_branch" in result
        assert len(result) == 3

    def test_discover_dict_structure(self, nested_dict):
        """Test discovering the overall structure of a dictionary."""
        result = discover_dict_structure(nested_dict)

        # Check paths in the result
        assert "level1/level2a" in result
        assert "level1/level2b/level3" in result
        assert "another_branch" in result
        assert len(result) == 3


class TestCountAndCheck:
    """
    Tests for the count_and_check_dict_recursive function.

    This is used liberally for reporting progress to the user, so it's
    important that it works as expected.
    """

    def test_count_and_check_dict_recursive_valid(self, nested_dict):
        """Test counting and checking a valid nested dictionary."""
        # Add shape to array-like values to simulate proper data
        for path, value in [
            (["level1", "level2a"], nested_dict["level1"]["level2a"]),
            (
                ["level1", "level2b", "level3"],
                nested_dict["level1"]["level2b"]["level3"],
            ),
            (["another_branch"], nested_dict["another_branch"]),
        ]:
            if hasattr(value, "shape") and not hasattr(value, "shape"):
                value.shape = (3,)

        count = count_and_check_dict_recursive(nested_dict)

        # Count should be the sum of array lengths
        assert count == 9  # Three arrays of length 3

    def test_count_and_check_dict_recursive_none(self):
        """Test that BadResult is raised when None is encountered."""
        test_dict = {"key": None}

        with pytest.raises(exceptions.BadResult) as excinfo:
            count_and_check_dict_recursive(test_dict)

        # Check error message
        assert "found a nonetype object" in str(excinfo.value).lower()
        assert (
            "all results should be numeric with associated units"
            in str(excinfo.value).lower()
        )

    def test_count_and_check_dict_recursive_no_units(self):
        """Test that BadResult is raised for an array without units."""
        test_dict = {"key": np.array([1, 2, 3])}

        with pytest.raises(exceptions.BadResult) as excinfo:
            count_and_check_dict_recursive(test_dict)

        # Check error message
        assert "without units" in str(excinfo.value).lower()
        assert (
            "all results should be numeric with associated units"
            in str(excinfo.value).lower()
        )

    def test_count_and_check_dict_recursive_non_array(self):
        """Test that BadResult is raised for a non-array value."""
        test_dict = {"key": 123}  # Not an array

        with pytest.raises(exceptions.BadResult) as excinfo:
            count_and_check_dict_recursive(test_dict)

        # Check error message
        assert "non-array object" in str(excinfo.value).lower()
        assert (
            "all results should be numeric with associated units"
            in str(excinfo.value).lower()
        )

    def test_count_and_check_dict_recursive_sed(self):
        """Test that Sed objects are counted as 1."""
        test_dict = {
            "line": Sed(
                lam=unyt_array([6563], "angstrom"),
                lnu=unyt_array([10], "erg/s/Hz"),
            )
        }

        count = count_and_check_dict_recursive(test_dict)
        assert count == 1

    def test_cached_split(self):
        """Test the cached split function."""
        result1 = cached_split("a/b/c")
        assert result1 == ["a", "b", "c"]

        # Get the result again, should use cache
        result2 = cached_split("a/b/c")
        assert result2 == ["a", "b", "c"]
        assert result1 is result2  # Same object due to caching


class TestCombineDicts:
    """Tests for the combine_list_of_dicts function."""

    def test_combine_list_of_dicts_simple(self):
        """Test combining a list of simple dictionaries."""
        dict1 = {"a": unyt_array([1, 2], "erg/s")}
        dict2 = {"a": unyt_array([2, 1], "erg/s")}

        result = combine_list_of_dicts([dict1, dict2])

        assert "a" in result
        np.testing.assert_array_equal(
            result["a"].value,
            [[1, 2], [2, 1]],
            err_msg="Dictionary list combination failed"
            f" {dict1} + {dict2} + {result['a']}",
        )
        assert str(result["a"].units) == "erg/s"

    def test_combine_list_of_dicts_nested(self):
        """Test combining a list of nested dictionaries."""
        dict1 = {"level1": {"level2": unyt_array([1, 2], "Msun")}}
        dict2 = {"level1": {"level2": unyt_array([3, 4], "Msun")}}

        result = combine_list_of_dicts([dict1, dict2])

        assert "level1" in result
        assert "level2" in result["level1"]
        np.testing.assert_array_equal(
            result["level1"]["level2"].value, [[1, 2], [3, 4]]
        )
        assert str(result["level1"]["level2"].units) == "Msun"

    def test_combine_list_of_dicts_missing_key(self):
        """Test that an error is raised when keys are missing."""
        dict1 = {"a": unyt_array([1], "erg/s")}
        dict2 = {"b": unyt_array([2], "erg/s")}

        with pytest.raises(ValueError) as excinfo:
            combine_list_of_dicts([dict1, dict2])

        assert "key" in str(excinfo.value).lower()
        assert "missing" in str(excinfo.value).lower()

    def test_unify_dict_structure_across_ranks(self):
        """Test unifying dictionary structure across MPI ranks."""
        # Create a mock MPI communicator
        comm = MagicMock()
        comm.rank = 0
        comm.gather.return_value = [{"path1", "path2"}, {"path1", "path3"}]
        comm.bcast.return_value = {"path1", "path2", "path3"}

        # Test data with only path1
        data = {"path1": unyt_array([1], "erg/s")}

        result = unify_dict_structure_across_ranks(data, comm)

        # Check that path2 and path3 were added
        assert "path1" in result
        assert "path2" in result
        assert "path3" in result

    def test_unify_dict_structure_across_ranks_non_dict(self):
        """Test that non-dict inputs are returned as-is."""
        comm = MagicMock()
        data = unyt_array([1, 2, 3], "erg/s")

        result = unify_dict_structure_across_ranks(data, comm)

        # Should return the input unchanged
        assert result is data


class TestGetDatasetProperties:
    """Tests for the get_dataset_properties function."""

    def test_get_dataset_properties(self):
        """Test getting dataset properties from a dictionary."""
        data = {
            "a": unyt_array([1, 2, 3], "erg/s"),
            "b": {
                "c": unyt_array([4, 5], "Msun"),
            },
        }

        # Create a mock MPI communicator
        comm = MagicMock()
        comm.rank = 0
        comm.gather.return_value = [{"a", "b/c"}]
        comm.bcast.return_value = {"a", "b/c"}

        shapes, dtypes, units, paths = get_dataset_properties(data, comm)

        # Check results
        assert shapes["a"] == (3,)
        assert shapes["b/c"] == (2,)
        assert str(dtypes["a"]) == str(np.array([1]).dtype)
        assert str(dtypes["b/c"]) == str(np.array([1]).dtype)
        assert units["a"] == "erg/s"
        assert units["b/c"] == "Msun"
        assert paths == {"a", "b/c"}

    def test_get_dataset_properties_non_dict(self):
        """Test get_dataset_properties with non-dict input."""
        data = unyt_array([1, 2, 3], "erg/s")
        comm = MagicMock()

        shapes, dtypes = get_dataset_properties(data, comm)

        assert shapes[""] == (3,)
        assert str(dtypes[""]) == str(np.array([1]).dtype)


class TestMemoryProfiling:
    """Tests for memory profiling utilities."""

    def test_get_full_memory_simple(self):
        """Test memory estimation for simple objects."""
        # Test with a simple list
        data = [1, 2, 3]
        mem = get_full_memory(data)
        assert mem > 0  # Should have some memory usage

        # Test with a dictionary
        data = {"a": 1, "b": 2}
        mem = get_full_memory(data)
        assert mem > 0

    def test_get_full_memory_numpy(self):
        """Test memory estimation for NumPy arrays."""
        # Small array
        data = np.array([1, 2, 3])
        mem_small = get_full_memory(data)

        # Larger array
        data_large = np.ones((100, 100))
        mem_large = get_full_memory(data_large)

        # Larger array should use more memory
        assert mem_large > mem_small

    def test_get_full_memory_nested(self):
        """Test memory estimation for nested structures."""
        data = {
            "arr": np.ones((10, 10)),
            "list": [1, 2, 3, {"nested": np.ones(5)}],
        }

        mem = get_full_memory(data)
        assert mem > 0

        # Test with a class
        class TestClass:
            def __init__(self):
                self.arr = np.ones(10)
                self.value = 42

        obj = TestClass()
        mem_obj = get_full_memory(obj)
        assert mem_obj > 0

    def test_get_full_memory_circular(self):
        """Test memory estimation with circular references."""
        data = {"self_ref": None}
        data["self_ref"] = data  # Create circular reference

        mem = get_full_memory(data)
        assert mem > 0  # Should still give a result without infinite recursion
