"""A submodule with helpers for writing out Synthesizer pipeline results."""

import inspect
import sys
from collections import defaultdict
from collections.abc import Mapping
from functools import lru_cache

import numpy as np
from unyt import Unit, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.emissions import Sed
from synthesizer.synth_warnings import warn
from synthesizer.units import unit_is_compatible

# Special model label for operations that are not tied to a specific model.
# This can be used for operations such as SFZH / SFH with no relation to
# an emission model.
NO_MODEL_LABEL = "no_model_label"


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


def validate_noise_unit_compatibility(instruments, expected_unit):
    """Validate that noise attributes have compatible units.

    This function checks that instruments with noise capabilities have
    depth and noise_maps attributes with units compatible with the expected
    unit for the image type (luminosity or flux).

    Note: depth can be specified as:
        - Plain float/dict of floats: apparent magnitudes (dimensionless,
          valid for both luminosity and flux images)
        - unyt_quantity/dict of unyt_quantity: flux/luminosity with units
          (must match image type)

    Args:
        instruments (list):
            A list of Instrument objects to validate.
        expected_unit (unyt.Unit):
            The expected unit for the image type (e.g., "erg/s/Hz" for
            luminosity images or "nJy" for flux images).

    Raises:
        InconsistentArguments:
            If an instrument has depth or noise_maps with incompatible units.
    """
    # Ensure expected_unit is a Unit object
    if not isinstance(expected_unit, Unit):
        expected_unit = Unit(expected_unit)

    for inst in instruments:
        if inst.can_do_noisy_imaging:
            # Check depth units if using SNR-based noise
            if inst.depth is not None:
                if isinstance(inst.depth, dict):
                    for filt, depth_val in inst.depth.items():
                        # Skip plain floats/ints (apparent magnitudes)
                        if isinstance(depth_val, (float, int)):
                            continue
                        # Validate unyt quantities
                        if isinstance(depth_val, unyt_quantity):
                            if not unit_is_compatible(
                                depth_val, expected_unit
                            ):
                                raise exceptions.InconsistentArguments(
                                    f"Depth units must be compatible with "
                                    f"{expected_unit}. Got {depth_val.units} "
                                    f"for filter {filt} in instrument "
                                    f"{inst.label}. Are you using a "
                                    "rest-frame or observed-frame instrument "
                                    "with the wrong image type?"
                                )
                        else:
                            raise exceptions.InconsistentArguments(
                                f"Depth must be a float (apparent magnitude) "
                                f"or unyt_quantity with units. Got "
                                f"{type(depth_val)} for filter {filt} in "
                                f"instrument {inst.label}."
                            )
                # Skip plain floats/ints (apparent magnitudes)
                elif isinstance(inst.depth, (float, int)):
                    pass  # Apparent magnitudes are valid for both types
                # Validate unyt quantities
                elif isinstance(inst.depth, unyt_quantity):
                    if not unit_is_compatible(inst.depth, expected_unit):
                        raise exceptions.InconsistentArguments(
                            f"Depth units must be compatible with "
                            f"{expected_unit}. Got {inst.depth.units} "
                            f"in instrument {inst.label}. Are you using a "
                            "rest-frame or observed-frame instrument with "
                            "the wrong image type?"
                        )
                else:
                    raise exceptions.InconsistentArguments(
                        f"Depth must be a float (apparent magnitude) or "
                        f"unyt_quantity with units. Got {type(inst.depth)} "
                        f"in instrument {inst.label}."
                    )

            # Check noise_maps units if using noise maps
            if inst.noise_maps is not None:
                if isinstance(inst.noise_maps, dict):
                    for filt, noise_map in inst.noise_maps.items():
                        if isinstance(noise_map, unyt_array):
                            if not unit_is_compatible(
                                noise_map, expected_unit
                            ):
                                raise exceptions.InconsistentArguments(
                                    f"Noise map units must be compatible "
                                    f"with {expected_unit}. Got "
                                    f"{noise_map.units} for filter {filt} "
                                    f"in instrument {inst.label}. Are you "
                                    "using a rest-frame or observed-frame "
                                    "instrument with the wrong image type?"
                                )
                        else:
                            raise exceptions.InconsistentArguments(
                                f"Noise map must be a unyt_array with units. "
                                f"Got {type(noise_map)} for filter {filt} in "
                                f"instrument {inst.label}."
                            )
                elif isinstance(inst.noise_maps, unyt_array):
                    if not unit_is_compatible(inst.noise_maps, expected_unit):
                        raise exceptions.InconsistentArguments(
                            f"Noise map units must be compatible with "
                            f"{expected_unit}. Got "
                            f"{inst.noise_maps.units} in instrument "
                            f"{inst.label}. Are you using a rest-frame or "
                            "observed-frame instrument with the wrong image "
                            "type?"
                        )
                else:
                    raise exceptions.InconsistentArguments(
                        f"Noise map must be a unyt_array with units. Got "
                        f"{type(inst.noise_maps)} in instrument {inst.label}."
                    )


class OperationKwargsHandler:
    """Container for Pipeline operation kwargs.

    This handler enables the running of pipeline operation multiple times
    with different parameters for different models in a clean, expandable and
    organized manner.

    This helper abstracts away the internal representation of operation kwargs
    from the Pipeline. It stores kwargs dictionaries keyed by
    (model_label, func_name) and provides a clean interface to loop over
    each operations kwargs.

    Internally it uses a nested defaultdict:

        self._queues[model_label][func_name] -> list[dict]

    where each list behaves as a FIFO queue of kwargs dicts.

    Example usage:

        handler = OperationKwargsHandler(
            model_labels=list(emission_model._models.keys()) + [NO_MODEL_LABEL]
        )

        # During "Pipeline.get_*" signalling:
        handler.add(
            NO_MODEL_LABEL,
            "get_images_flux",
            fov=fov,
            img_type="smoothed",
            ...
        )

        # During "Pipeline._get*" operation execution:
        for model_label, op_kwargs in handler["get_images_flux"]:
            ... use op_kwargs ...

    Attributes:
        _allowed_models (set):
            The models associated with the EmissionModel the pipeline is
            using.
        _queues (defaultdict):
            Nested mapping of model_label -> func_name -> list[kwargs dict].
            Each list is a FIFO (First In, First Out) queue of kwargs for that
            (model, func) pair.
    """

    def __init__(self, model_labels):
        """Initialise the OperationKwargsHandler.

        Args:
            model_labels (list or sea of str):
                All the labels associated with the EmissionModel we
                are working on.
        """
        # Convert the input model_labels to a set for efficient lookup.
        self._allowed_models = set(model_labels)

        # We can always use the special NO_MODEL_LABEL.
        self._allowed_models.add(NO_MODEL_LABEL)

        # Nested mapping: model_label -> func_name -> list[kwargs dict].
        self._queues = defaultdict(lambda: defaultdict(list))

    def _check_model_label(self, model_label):
        """Validate that the provided model_label is allowed.

        Args:
            model_label (str):
                Emission model label to validate.

        Raises:
            exceptions.InconsistentArguments:
                If validation is enabled and model_label is not allowed.
        """
        if model_label in self._allowed_models:
            return

        raise exceptions.InconsistentArguments(
            f"Model label {model_label} not found in the Pipeline's "
            "EmissionModel."
        )

    def __getitem__(self, func_name):
        """Return an iterator over (model_label, kwargs) for an operation.

        This allows syntax like:

            for model_label, op_kwargs in handler["get_images_flux"]:
                ...

        This is a non-consuming iterator: the internal queues are not
        modified by iteration. To consume the queue, use iter_all with
        consume=True.

        Args:
            func_name (str):
                Operation / method name.
        """
        return self.iter_all(func_name, consume=False)

    def __contains__(self, func_name):
        """Return True if any kwargs are queued for this operation.

        This allows syntax like:

            if "get_images_flux" in handler:
                ...

        Args:
            func_name (str):
                Operation / method name.
        """
        return self.has(func_name)

    def _get_from_queue(self, model_label, func_name):
        """Internal helper to get the queue for a (model_label, func_name).

        Args:
            model_label (str):
                Model label to get the queue for.
            func_name (str):
                Operation / method name.

        Returns:
            list:
                The list of kwargs dicts for this (model_label, func_name).
        """
        self._check_model_label(model_label)
        return self._queues[model_label][func_name]

    def add(self, model_label, func_name, **kwargs):
        """Add a kwargs dict for a given (model_label, func_name) pair.

        Args:
            model_label (str):
                Emission model label or a special label such as NO_MODEL_LABEL
                for operations that are not tied to a specific model.
            func_name (str):
                Operation / method name, e.g. "get_images_luminosity".
            **kwargs:
                Arbitrary keyword arguments to store for this (model, func).
        """
        self._get_from_queue(model_label, func_name).append(kwargs)

    def has(self, func_name, model_label=None):
        """Return True if any kwargs are queued for the given operation.

        Args:
            func_name (str):
                Operation / method name.
            model_label (str, optional):
                If provided, restrict the check to this model. If omitted,
                all models are searched.

        Returns:
            bool: True if at least one kwargs dict is present, False otherwise.
        """
        if model_label is not None:
            return bool(self._get_from_queue(model_label, func_name))

        # No model_label specified: search across all models.
        for model_queues in self._queues.values():
            if model_queues.get(func_name):
                return True
        return False

    def pop_next(self, model_label, func_name):
        """Pop and return the next kwargs dict for (model_label, func_name).

        This provides single-step consumption of the queue associated with
        a given (model, func) pair.

        Args:
            model_label (str):
                Model label to pop from.
            func_name (str):
                Operation / method name.

        Returns:
            dict or None:
                The next kwargs dict from the queue if available, otherwise
                None.
        """
        queue = self._get_from_queue(model_label, func_name)
        if len(queue) > 0:
            return queue.pop(0)
        return None

    def iter_for(self, model_label, func_name, consume=False):
        """Iterate over kwargs dicts for a single (model_label, func_name).

        Args:
            model_label (str):
                Model label to iterate for.
            func_name (str):
                Operation / method name.
            consume (bool, optional):
                If True, entries are popped from the internal queue as they
                are yielded. If False (default), the internal queue is left
                unchanged.

        Yields:
            dict: Each kwargs dict stored for (model_label, func_name).
        """
        queue = self._get_from_queue(model_label, func_name)

        if consume:
            # FIFO consumption: pop until the queue is empty.
            while queue:
                yield queue.pop(0)
        else:
            # Non-consuming: iterate over a shallow copy.
            for kw in list(queue):
                yield kw

    def iter_all(self, func_name, consume=False):
        """Iterate over (model_label, kwargs) pairs for a given operation.

        This is the main entry point for Pipeline methods that want to
        process all configs for a given operation, regardless of model.

        Args:
            func_name (str):
                Operation / method name.
            consume (bool, optional):
                If True, entries are popped from each queue as they are
                yielded (consuming the queues). If False (default), the
                internal queues are left unchanged.

        Yields:
            (model_label, dict):
                Tuples of model label and kwargs dict.
        """
        if consume:
            # Iterate over a snapshot of keys to avoid modifying while
            # iterating.
            for model_label in list(self._queues.keys()):
                queue = self._queues[model_label].get(func_name, [])
                while queue:
                    yield model_label, queue.pop(0)
        else:
            # Non-consuming: iterate over shallow copies of the queues.
            for model_label, model_queues in list(self._queues.items()):
                queue = model_queues.get(func_name, [])
                for kw in list(queue):
                    yield model_label, kw

    def clear(self, func_name=None, model_label=None):
        """Clear queued kwargs for a given operation and/or model.

        Args:
            func_name (str, optional):
                If provided, only clear this operation.
            model_label (str, optional):
                If provided, only clear for this model.

        Behaviour:
            - func_name is None and model_label is None:
                Clear all queues.
            - func_name is None and model_label is not None:
                Clear all operations for that model.
            - func_name is not None and model_label is None:
                Clear that operation across all models.
            - func_name is not None and model_label is not None:
                Clear that single (model_label, func_name) queue.
        """
        # Clear everything.
        if func_name is None and model_label is None:
            self._queues.clear()
            return

        # Clear all operations for a specific model.
        if func_name is None and model_label is not None:
            self._check_model_label(model_label)
            self._queues[model_label].clear()
            return

        # Clear a specific operation across all models.
        if func_name is not None and model_label is None:
            for m_lab in list(self._queues.keys()):
                self._queues[m_lab].pop(func_name, None)
            return

        # Clear a specific (model_label, func_name) queue.
        self._get_from_queue(model_label, func_name).clear()
