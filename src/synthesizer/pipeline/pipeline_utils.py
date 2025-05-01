"""A submodule with helpers for writing out Synthesizer pipeline results."""

import inspect
import sys
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.emissions import Sed
from synthesizer.synth_warnings import warn


def discover_attr_paths_recursive(obj, prefix="", output_set=None):
    """Recursively discover all outputs attached to an object.

    This function will collate all paths to attributes at any level within
    the input object.

    If the object is a dictionary, we will loop over all keys and values
    recursing where appropriate.

    If the object is a class instance (e.g. Galaxy, Stars,
    ImageCollection, etc.), we will loop over all attributes and
    recurse where appropriate.

    If the object is a "value" (i.e. an array or a scalar), we will append
    the full path to the output list.

    NOTE: this function is currently unused but is kept for debugging purposes
    since it is extremely useful to see the nesting of attributes on objects.

    Args:
        obj (dict):
            The dictionary to search.
        prefix (str):
            A prefix to add to the keys of the arrays.
        output_set (set):
            A set to store the output paths in.

    Returns:
        dict:
            A dictionary of all the numpy arrays in the input dictionary.
    """
    # If the obj is a dictionary, loop over the keys and values and recurse
    if isinstance(obj, dict):
        for k, v in obj.items():
            output_set = discover_attr_paths_recursive(
                v,
                prefix=f"{prefix}/{k}",
                output_set=output_set,
            )

    # If it's a class instance and not a leaf type
    elif hasattr(obj, "__class__") and not isinstance(
        obj, (unyt_array, unyt_quantity, np.ndarray, str, bool, int, float)
    ):
        members = inspect.getmembers(
            obj.__class__, lambda a: isinstance(a, property)
        )
        prop_names = {name for name, _ in members}

        # Collect public instance attributes if the object has a __dict__
        # attribute
        if hasattr(obj, "__dict__"):
            keys = set(vars(obj).keys())
        else:
            # Otherwise, just collect the property names
            keys = set()
        keys.update(prop_names)

        for k in keys:
            # Handle Quantity objects
            if hasattr(obj, k[1:]):
                k = k[1:]

            # Skip private attributes
            if k.startswith("_"):
                continue

            try:
                v = getattr(obj, k)
            except Exception:
                continue  # Skip properties that raise errors

            # Skip if None
            if v is None:
                continue

            discover_attr_paths_recursive(
                v,
                prefix=f"{prefix}/{k}",
                output_set=output_set,
            )

    # Nothing to do if its an unset optional value
    elif obj is None:
        return output_set

    # Skip undesirable types
    elif isinstance(obj, (str, bool)):
        return output_set

    # Otherwise, we have something we need to write out so add the path to
    # the set
    else:
        output_set.add(prefix.replace(" ", "_"))

    return output_set


def discover_dict_recursive(data, prefix="", output_set=None):
    """Recursively discover all leaves in a dictionary.

    Args:
        data (dict): The dictionary to search.
        prefix (str): A prefix to add to the keys of the arrays.
        output_set (set): A set to store the output paths in.

    Returns:
        dict:
            A dictionary of all the numpy arrays in the input dictionary.
    """
    # If the obj is a dictionary, loop over the keys and values and recurse
    if isinstance(data, dict):
        for k, v in data.items():
            output_set = discover_dict_recursive(
                v,
                prefix=f"{prefix}/{k}",
                output_set=output_set,
            )

    # Otherwise, we have something we need to write out so add the path to
    # the set
    else:
        output_set.add(prefix[1:].replace(" ", "_"))

    return output_set


def discover_dict_structure(data):
    """Recursively discover the structure of a dictionary.

    Args:
        data (dict):
            The dictionary to search.

    Returns:
        dict:
            A dictionary of all the paths in the input dictionary.
    """
    # Set up the set to hold the global output paths
    output_set = set()

    # Loop over the galaxies and recursively discover the outputs
    output_set = discover_dict_recursive(data, output_set=output_set)

    return output_set


def count_and_check_dict_recursive(data, prefix=""):
    """Recursively count the number of leaves in a dictionary.

    Args:
        data (dict): The dictionary to search.
        prefix (str): A prefix to add to the keys of the arrays.

    Returns:
        dict:
            A dictionary of all the numpy arrays in the input dictionary.
    """
    count = 0

    # If the obj is a dictionary, loop over the keys and values and recurse
    if isinstance(data, dict):
        for k, v in data.items():
            count += count_and_check_dict_recursive(
                v,
                prefix=f"{prefix}/{k}",
            )
        return count

    # Otherwise, we are at a leaf with some data to account for. Check the
    # result makes sense.The count is always the first element of the
    # shape tuple
    if data is None:
        raise exceptions.BadResult(
            f"Found a NoneType object at {prefix}. "
            "All results should be numeric with associated units."
        )

    if not hasattr(data, "units") and isinstance(data, np.ndarray):
        raise exceptions.BadResult(
            f"Found an array object without units at {prefix}. "
            "All results should be numeric with associated units. "
            f"Data: {data}"
        )

    if not hasattr(data, "shape"):
        raise exceptions.BadResult(
            f"Found a non-array object at {prefix}. "
            "All results should be numeric with associated units."
        )

    # If we have a Sed then we have a count of 1
    if isinstance(data, Sed):
        return 1
    return data.shape[0]


@lru_cache(maxsize=500)
def cached_split(split_key):
    """Split a key into a list of keys.

    This is a cached version of the split function to avoid repeated
    splitting of the same key.

    Args:
        split_key (str):
            The key to split in "key1/key2/.../keyN" format.

    Returns:
        list:
            A list of the split keys.
    """
    return split_key.split("/")


def combine_list_of_dicts(dicts):
    """Combine a list of dictionaries into a single dictionary.

    Args:
        dicts (list):
            A list of dictionaries to combine.

    Returns:
        dict:
            The combined dictionary.
    """

    def combine_values(values):
        # Combine values into a unyt_array
        return unyt_array(values)

    def recursive_merge(dict_list):
        if len(dict_list) == 0:
            return {}
        if not isinstance(dict_list[0], dict):
            # Base case: combine non-dict leaves
            return combine_values(dict_list)

        # Recursive case: merge dictionaries
        merged = {}
        keys = dict_list[0].keys()
        for key in keys:
            # Ensure all dictionaries have the same keys
            if not all(key in d for d in dict_list):
                raise ValueError(
                    f"Key '{key}' is missing in some dictionaries."
                )
            # Recurse for each key
            merged[key] = recursive_merge([d[key] for d in dict_list])
        return merged

    return recursive_merge(dicts)


def unify_dict_structure_across_ranks(data, comm, root=0):
    """Recursively unify the structure of a dictionary across all ranks.

    This function will ensure that all ranks have the same structure in their
    dictionaries. This is necessary for writing out the data in parallel.

    Args:
        data (dict): The data to unify.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.
    """
    # If we don't have a dict, just return the data straight away, theres no
    # need to check the structure
    if not isinstance(data, dict):
        return data

    # Ok, we have a dict. Before we get to the meat, lets make sure we have
    # the same structure on all ranks
    my_out_paths = discover_dict_structure(data)
    gathered_out_paths = comm.gather(my_out_paths, root=root)
    if comm.rank == root:
        unique_out_paths = set.union(*gathered_out_paths)
    else:
        unique_out_paths = None
    out_paths = comm.bcast(unique_out_paths, root=root)

    # Warn the user if the structure is different
    if len(out_paths) != len(my_out_paths):
        warn(
            "The structure of the data is different on different ranks. "
            "We'll unify the structure but something has gone awry."
        )

        # Ensure all ranks have the same structure
        for path in out_paths:
            d = data
            for k in path.split("/")[:-1]:
                d = d.setdefault(k, {})
            d.setdefault(path.split("/")[-1], unyt_array([], "dimensionless"))

    return data


def get_dataset_properties(data, comm, root=0):
    """Return the shapes, dtypes and units of all data arrays in a dictionary.

    Args:
        data (dict): The data to get the shapes of.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.

    Returns:
        dict: A dictionary of the shapes of all data arrays.
        dict: A dictionary of the dtypes of all data arrays.
        dict: A dictionary of the units of all data arrays.
    """
    # If we don't have a dict, just return the data straight away, theres no
    # need to check the structure
    if not isinstance(data, dict):
        return {"": data.shape}, {"": data.dtype}

    # Ok, we have a dict. Before we get to the meat, lets make sure we have
    # the same structure on all ranks
    my_out_paths = discover_dict_structure(data)
    gathered_out_paths = comm.gather(my_out_paths, root=root)
    if comm.rank == root:
        unique_out_paths = set.union(*gathered_out_paths)
    else:
        unique_out_paths = None
    out_paths = comm.bcast(unique_out_paths, root=root)

    # Create a dictionary to store the shapes and dtypes
    shapes = {}
    dtypes = {}
    units = {}

    # Loop over the paths and get the shapes
    for path in out_paths:
        d = data
        for k in path.split("/"):
            d = d[k]
        shapes[path] = d.shape
        dtypes[path] = d.dtype
        units[path] = str(d.units)

    return shapes, dtypes, units, out_paths


def get_full_memory(obj, seen=None):
    """Estimate memory usage of a Python object, including NumPy arrays.

    Args:
        obj: The object to inspect.
        seen: Set of seen object ids to avoid double-counting.

    Returns:
        int: Approximate size in bytes.
    """
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)

    size = 0

    # NumPy arrays — very important to check early
    if isinstance(obj, np.ndarray):
        return obj.nbytes + sys.getsizeof(obj)

    # Built-in container types
    elif isinstance(obj, Mapping):
        size += sys.getsizeof(obj)
        for k, v in obj.items():
            size += get_full_memory(k, seen)
            size += get_full_memory(v, seen)

    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sys.getsizeof(obj)
        for item in obj:
            size += get_full_memory(item, seen)

    # Objects with __dict__
    elif hasattr(obj, "__dict__"):
        size += sys.getsizeof(obj)
        size += get_full_memory(vars(obj), seen)

    # Objects with __slots__
    elif hasattr(obj, "__slots__"):
        size += sys.getsizeof(obj)
        for slot in obj.__slots__:
            if hasattr(obj, slot):
                size += get_full_memory(getattr(obj, slot), seen)

    else:
        # Fallback: include basic object size
        size += sys.getsizeof(obj)

    return size
