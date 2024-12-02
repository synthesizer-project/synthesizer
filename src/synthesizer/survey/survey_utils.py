"""A submodule with helpers for writing out Synthesizer pipeline results."""

from functools import lru_cache

import numpy as np
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
    # error automatically to highlight a mismatch (all ranks must agree on the
    # structure in MPI land)
    for key in keys[:-1]:
        d = d[key]

    # Store the data at the leaf key
    d[keys[-1]] = unyt_array(data)


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
                raise KeyError(
                    f"Key {'/'.join(keys)} not found in {type(obj)}"
                )

    return obj


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


def sort_data_recursive(data, sinds):
    """
    Sort a dictionary recursively.

    Args:
        data (dict): The data to sort.
        sinds (dict): The sorted indices.
    """
    # Early exit if data is None
    if data is None:
        return None

    # If the data isn't a dictionary just return the sorted data
    if not isinstance(data, dict):
        # If there is no data we can't sort it, just return the empty array.
        # This can happen if there are no galaxies.
        if len(data) == 0:
            return unyt_array(data)

        # Convert the list of data to an array (but we don't want to lose the
        # units)
        data = unyt_array([d.value for d in data])

        try:
            # Apply the sorting indices to the first axis
            return np.take_along_axis(data, sinds, axis=0)
        except (IndexError, ValueError, AttributeError) as e:
            print(data)
            print(sinds)
            print(len(sinds))
            print(data.shape)
            raise IndexError(f"Failed to sort data - {e}")

    # Loop over the data
    sorted_data = {}
    for k, v in data.items():
        sorted_data[k] = sort_data_recursive(v, sinds)

    return sorted_data


def write_dataset(hdf, data, key):
    """
    Write a dataset to an HDF5 file.

    We handle various different cases here:
    - If the data is a unyt object, we write the value and units.
    - If the data is a string we'll convert it to a h5py compatible string and
      write it with dimensionless units.
    - If the data is a numpy array, we write the data and set the units to
      "dimensionless".

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (any): The data to write.
        key (str): The key to write the data to.
    """
    # Strip the units off the data and convert to a numpy array
    if hasattr(data, "units"):
        units = str(data.units)
        data = data.value
    else:
        units = "dimensionless"

    # If we have an array of strings, convert to a h5py compatible string
    if data.dtype.kind == "U":
        data = np.array([d.encode("utf-8") for d in data])

    # Write the dataset with the appropriate units
    dset = hdf.create_dataset(key, data=data)
    dset.attrs["Units"] = units


def write_datasets_recursive(hdf, data, key):
    """
    Write a dictionary to an HDF5 file recursively.

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (dict): The data to write.
        key (str): The key to write the data to.
    """
    # Early exit if data is None
    if data is None:
        return

    # If the data isn't a dictionary just write the dataset
    if not isinstance(data, dict):
        try:
            write_dataset(hdf, data, key)
        except TypeError as e:
            raise TypeError(
                f"Failed to write dataset {key} (type={type(data)}) - {e}"
            )
        return

    # Loop over the data
    for k, v in data.items():
        write_datasets_recursive(hdf, v, f"{key}/{k}")


def write_datasets_recursive_parallel(
    hdf,
    data,
    key,
    indexes,
    comm,
    num_galaxies,
):
    """
    Write a dictionary to an HDF5 file recursively in parallel.

    This function requires that h5py has been built with parallel support.

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (dict): The data to write.
        key (str): The key to write the data to.
        indexes (array): The sorting indices.
        comm (mpi.Comm): The MPI communicator.
        num_galaxies (int): The total number of galaxies (first axis count).
    """
    rank = comm.Get_rank()

    # If the data isn't a dictionary, write the dataset
    if not isinstance(data, dict):
        try:
            local_shape = data.shape

            # Only rank 0 creates the dataset
            if rank == 0:
                # Determine dtype
                if hasattr(data, "value"):
                    dtype = data.value.dtype
                else:
                    dtype = data.dtype

                # Calculate global shape
                global_shape = [num_galaxies]
                for i in range(1, len(local_shape)):
                    global_shape.append(local_shape[i])

                # Create the dataset
                dset = hdf.create_dataset(
                    key,
                    shape=tuple(global_shape),
                    dtype=dtype,
                )

                # Handle units if present
                if hasattr(data, "units"):
                    dset.attrs["Units"] = str(data.units)

            # Synchronize all ranks before writing
            comm.barrier()

            # Get the dataset on all ranks
            dset = hdf[key]

            # Set collective I/O property for the write operation
            with dset.collective:
                # Write the data using the appropriate slice
                start = sum(comm.allgather(local_shape[0])[:rank])
                end = start + local_shape[0]
                dset[start:end, ...] = data

        except Exception as e:
            raise RuntimeError(
                f"Failed to write dataset '{key}' on rank {rank}: {e}"
            )

        return

    # Recursively handle dictionary data
    for k, v in data.items():
        write_datasets_recursive_parallel(
            hdf, v, f"{key}/{k}", indexes, comm, num_galaxies
        )


def _gather_and_write_recursive(hdf, data, key, comm, root):
    """
    Recursively gather data and write it out.

    We will recurse through the dictionary and gather all arrays or lists
    at the leaves and write them out to the HDF5 file. Doing so limits the
    size of communications and minimises the amount of data we have collected
    on the root rank at any one time.

    Args:
        hdf (h5py.File): The HDF5 file to write to.
        data (dict): The data to write.
        key (str): The key to write the data to.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.
    """
    # If the data is a dictionary we need to recurse
    if isinstance(data, dict):
        for k, v in data.items():
            _gather_and_write_recursive(hdf, v, f"{key}/{k}", comm, root)
        return

    # First gather the data
    collected_data = comm.gather(data, root=root)

    # If we aren't the root we're done
    if collected_data is None:
        return

    # Remove any empty datasets
    collected_data = [d for d in collected_data if len(d) > 0]

    # If there's nothing to write we're done
    if len(collected_data) == 0:
        return

    # Right, we know we have data so lets combine the list of arrays into a
    # single unyt_array
    try:
        combined_data = unyt_array(np.concatenate(collected_data))
    except ValueError as e:
        raise ValueError(f"Failed to concatenate {key} - {e}")

    # Write the dataset
    try:
        write_dataset(hdf, combined_data, key)
    except TypeError as e:
        print(type(data), type(combined_data))
        raise TypeError(f"Failed to write dataset {key} - {e}")

    # Now that data has been written we can clear the list of collected data
    # The garbage collector would probably do this for us but we're being
    # explicit here
    del collected_data


def gather_and_write_datasets(hdf, data, key, comm, root=0):
    """
    Recursively collect data from all ranks onto the root and write it out.

    We will recurse through the dictionary and gather all arrays or lists
    at the leaves and write them out to the HDF5 file. Doing so limits the
    size of communications and minimises the amount of data we have collected
    on the root rank at any one time.

    Args:
        data (any): The data to gather.
        comm (mpi.Comm): The MPI communicator.
        root (int): The root rank to gather data to.
        sinds (array): The sorting indices.
    """
    # If we don't have a dict, just gather and write the data straight away,
    # theres no need to check the structure
    if not isinstance(data, dict):
        _gather_and_write_recursive(hdf, data, key, comm, root)
        return

    # Ok, we have a dict. Before we get to the meat, lets make sure we have
    # the same structure on all ranks
    gathered_out_paths = comm.gather(discover_outputs(data), root=root)
    out_paths = comm.bcast(
        set.union(*gathered_out_paths) if comm.rank == 0 else None, root=root
    )

    # Ensure all ranks have the same structure
    for path in out_paths:
        d = data
        for k in path.split("/")[:-1]:
            d = d.setdefault(k, {})
        d[path.split("/")[-1]] = unyt_array([], "dimensionsless")

    # Recurse through the whole dict communicating and writing all lists or
    # arrays we hit along the way
    _gather_and_write_recursive(hdf, data, key, comm, root)
