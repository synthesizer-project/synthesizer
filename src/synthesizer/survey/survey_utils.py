"""A submodule with helpers for writing out Synthesizer pipeline results."""

from functools import lru_cache

import numpy as np
from unyt import unyt_array, unyt_quantity


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

    # Skip undesiable types
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


def pack_data(d, data, output_key):
    """
    Pack data into a dictionary recursively.

    Args:
        d (dict):
            The dictionary to pack the data into.
        data (any):
            The data to store at output_key.
        output_key (str):
            The key to pack the data into in "key1/key2/.../keyN" format.
    """
    # Split the keys
    keys = cached_split(output_key)

    # Loop until we reach the "leaf" key, we should have the dictionary
    # structure in place by the time we reach the leaf key, if we don't we
    # need an error to highlight a mismatch (all ranks must agree on the
    # structure in MPI land)
    for key in keys[:-1]:
        d = d[key]

    # Store the data at the leaf key
    d[keys[-1]] = data


def unpack_data(obj, output_path):
    """
    Unpack data from an object recursively.

    This is a helper function for traversing complex attribute paths. These
    can include attributes which are dictionaries or objects with their own
    attributes. A "/" defines the string to the right of it as the key to
    a dictionary, while a "." defines the string to the right of it as an
    attribute of an object.

    Args:
        obj (dict):
            The dictionary to search.
        output_path (tuple):
            The path to the desired attribute of the form
            ".attr1/key2.attr2/.../keyN".
    """
    # Split the output path
    keys = cached_split(output_path)

    # Recurse until we reach the desired attribute
    for key in keys:
        if getattr(obj, key, None) is not None:
            obj = getattr(obj, key)
        elif getattr(obj, "_" + key, None) is not None:
            obj = getattr(obj, "_" + key)
        else:
            try:
                obj = obj[key]
            except (KeyError, ValueError, TypeError):
                raise KeyError(f"Key {key} not found in {type(obj)}")

    return obj


def sort_data_recursive(data, sinds):
    """
    Sort a dictionary recursively.

    Args:
        data (dict): The data to sort.
        sinds (dict): The sorted indices.
    """
    # If the data isn't a dictionary just return the sorted data
    if not isinstance(data, dict):
        # If there is no data we can't sort it, just return the empty array.
        # This can happen if there are no galaxies.
        if len(data) == 0:
            return data

        # Convert the list of data to an array (but we don't want to lose the
        # units)
        if isinstance(data[0], (unyt_quantity, unyt_array)):
            data = unyt_array(np.array([d.value for d in data]), data[0].units)
        else:
            data = np.array(data)

        return data[sinds]

    # Loop over the data
    sorted_data = {}
    for k, v in data.items():
        sorted_data[k] = sort_data_recursive(v, sinds)

    return sorted_data


def write_datasets_recursive(hdf, data, key):
    """
    Write a dictionary to an HDF5 file recursively.

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (dict): The data to write.
        key (str): The key to write the data to.
    """
    # If the data isn't a dictionary just write the dataset
    if not isinstance(data, dict):
        try:
            # Write the dataset with the appropriate units (if its not
            # dimensionless if will be a unyt object)
            if isinstance(data, (unyt_quantity, unyt_array)):
                dset = hdf.create_dataset(key, data=data.value)
                dset.attrs["Units"] = str(data.units)
            else:
                dset = hdf.create_dataset(key, data=data)
                dset.attrs["Units"] = "dimensionless"
        except TypeError as e:
            print(f"Failed to write dataset {key} (type={type(data)})")
            raise TypeError(e)
        return

    # Loop over the data
    for k, v in data.items():
        write_datasets_recursive(hdf, v, f"{key}/{k}")


def write_datasets_recursive_parallel(hdf, data, key, indexes, comm):
    """
    Write a dictionary to an HDF5 file recursively in parallel.

    This function requires that h5py has been built with parallel support.

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (dict): The data to write.
        key (str): The key to write the data to.
    """
    # If the data isn't a dictionary, just write the dataset
    if not isinstance(data, dict):
        try:
            # Gather the shape from everyone
            shape = comm.gather(data.shape, root=0)

            # Only rank 0 creates the dataset
            if comm.rank == 0:
                dtype = (
                    data.value.dtype
                    if isinstance(data, (unyt_quantity, unyt_array))
                    else data.dtype
                )
                dset = hdf.create_dataset(
                    key,
                    shape=np.sum(shape, axis=0),
                    dtype=dtype,
                )
                dset.attrs["Units"] = (
                    str(data.units)
                    if isinstance(data, (unyt_quantity, unyt_array))
                    else "dimensionless"
                )
            # Ensure all ranks see the dataset
            comm.barrier()

            # Get the dataset on all ranks
            dset = hdf[key]

            # Each rank writes its own part of the data based on 'indexes'
            for idx in indexes:
                dset[idx] = (
                    data.value[idx]
                    if isinstance(data, (unyt_quantity, unyt_array))
                    else data[idx]
                )
        except TypeError as e:
            print(f"Failed to write dataset {key}")
            raise TypeError(e)
        return

    # Loop over the data
    for k, v in data.items():
        write_datasets_recursive_parallel(hdf, v, f"{key}/{k}", indexes, comm)


def recursive_gather(data, comm, root=0):
    """
    Recursively collect data from all ranks onto the root.

    This function is effectively just comm.gather but will ensure the
    gather is performed at the root of any passed dictionaries to avoid
    overflows and any later recursive concatenation issues.

    Args:
        data (any): The data to gather.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.
        sinds (array): The sorting indices.
    """
    # If we don't have a dict, just gather the data straight away
    if not isinstance(data, dict):
        collected_data = comm.gather(data, root=root)
        if comm.rank == root:
            collected_data = [d for d in collected_data if len(d) > 0]
            if len(collected_data) > 0:
                return np.concatenate(collected_data)
            else:
                return []
        else:
            return []

    # Recurse through the whole dict communicating an lists or
    # arrays we hit along the way
    def _gather(d, comm, root):
        new_d = {}
        for k, v in d.items():
            if isinstance(v, (list, np.ndarray)):
                collected_data = comm.gather(v, root=root)
                if comm.rank == root:
                    collected_data = [d for d in collected_data if len(d) > 0]
                    if len(collected_data) > 0:
                        new_d[k] = np.concatenate(collected_data)
                    else:
                        new_d[k] = []
                else:
                    new_d[k] = []
            elif isinstance(v, dict):
                new_d[k] = _gather(v, comm, root)
            else:
                raise ValueError(f"Unexpected type {type(v)}")
        return new_d

    return _gather(data, comm, root)
