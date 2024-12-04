"""A submodule with helpers for writing out Synthesizer pipeline results."""

from functools import lru_cache

from unyt import unyt_array


def discover_outputs_recursive(obj, prefix="", output_set=None):
    """
    Recursively discover all outputs attached to a galaxy.

    This function will collate all paths to attributes at any level within
    the input object.

    If the object is a dictionary, we will loop over all keys and values
    recursing where appropriate.

    If the object is a class instance (e.g. Galaxy, Stars,
    ImageCollection, etc.), we will loop over all attributes and
    recurse where appropriate.

    If the object is a "value" (i.e. an array or a scalar), we will append
    the full path to the output list.

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
            output_set = discover_outputs_recursive(
                v,
                prefix=f"{prefix}/{k}",
                output_set=output_set,
            )

    # If the obj is a class instance, loop over the attributes and recurse
    elif hasattr(obj, "__dict__"):
        for k, v in obj.__dict__.items():
            # # Skip callables as long as not a property
            # if callable(v) and not isinstance(v, property):
            #     continue

            # Handle Quantity objects
            if hasattr(obj, k[1:]):
                k = k[1:]

            # Skip private attributes
            if k.startswith("_"):
                continue

            # Recurse
            output_set = discover_outputs_recursive(
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


def discover_outputs(galaxies):
    """
    Recursively discover all outputs attached to a galaxy.

    This function will collate all paths to attributes at any level within
    the input object.

    If the object is a dictionary, we will loop over all keys and values
    recursing where appropriate.

    If the object is a class instance (e.g. Galaxy, Stars,
    ImageCollection, etc.), we will loop over all attributes and
    recurse where appropriate.

    If the object is a "value" (i.e. an array or a scalar), we will append
    the full path to the output list.

    Args:
        galaxy (dict):
            The dictionary to search.
        prefix (str):
            A prefix to add to the keys of the arrays.
        output (dict):
            A dictionary to store the output paths in.
    """
    # Set up the set to hold the global output paths
    output_set = set()

    # Loop over the galaxies and recursively discover the outputs
    for galaxy in galaxies:
        output_set = discover_outputs_recursive(galaxy, output_set=output_set)

    return output_set


@lru_cache(maxsize=500)
def cached_split(split_key):
    """
    Split a key into a list of keys.

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
    """
    Combine a list of dictionaries into a single dictionary.

    Args:
        dicts (list):
            A list of dictionaries to combine.

    Returns:
        dict:
            The combined dictionary.
    """

    def combine_values(*values):
        # Combine values into a unyt_array
        return unyt_array(values)

    def recursive_merge(dict_list):
        if not isinstance(dict_list[0], dict):
            # Base case: combine non-dict leaves
            return combine_values(*dict_list)

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

    if not isinstance(dicts[0], dict):
        TypeError(
            f"Input must be a list of dictionaries, not {type(dicts[0])}"
        )

    return recursive_merge(dicts)


def unify_dict_structure_across_ranks(data, comm, root=0):
    """
    Recursively unify the structure of a dictionary across all ranks.

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
    gathered_out_paths = comm.gather(discover_outputs(data), root=root)
    if comm.rank == root:
        unique_out_paths = set.union(*gathered_out_paths)
    else:
        unique_out_paths = None
    out_paths = comm.bcast(unique_out_paths, root=root)

    # Ensure all ranks have the same structure
    for path in out_paths:
        d = data
        for k in path.split("/")[:-1]:
            d = d.setdefault(k, {})
        d[path.split("/")[-1]] = unyt_array([], "dimensionsless")

    return data
