"""A module for handling I/O operations in the pipeline.

This module contains classes and functions for reading and writing data
in the pipeline. This includes reading and writing HDF5 files, as well as
handling the MPI communications for parallel I/O operations.

Example usage:

    # Write data to an HDF5 file
    writer = PipelineIO("output.hdf5")
    writer.write_data(data, key)


"""

import os
import time

import h5py
import numpy as np
from unyt import unyt_array

from synthesizer._version import __version__
from synthesizer.instruments import InstrumentCollection
from synthesizer.pipeline.pipeline_utils import (
    get_dataset_properties,
    unify_dict_structure_across_ranks,
)
from synthesizer.synth_warnings import warn


class PipelineIO:
    """A class for writing data to an HDF5 file.

    This class provides methods for writing data to an HDF5 file. It can
    handle writing data in parallel using MPI if the h5py library has been
    built with parallel support.

    Attributes:
        hdf (h5py.File): The HDF5 file to write to.
        comm (mpi.Comm): The MPI communicator.
        num_galaxies (int): The total number of galaxies.
        rank (int): The rank of the MPI process.
        is_parallel (bool): Whether the writer is running in parallel.
        is_root (bool): Whether the writer is running on the root process.
        is_collective (bool): Whether the writer is running in collective mode.
        verbose (bool): Whether to print verbose output.
        _start_time (float): The start time of the pipeline.
    """

    # Is our h5py build parallel?
    if hasattr(h5py, "get_config"):
        PARALLEL = h5py.get_config().mpi
    else:
        PARALLEL = False

    def __init__(
        self,
        filepath,
        comm=None,
        ngalaxies_local=None,
        start_time=None,
        verbose=1,
        parallel_io=False,
    ):
        """Initialize the HDF5Writer class.

        Args:
            filepath (str): The path to the HDF5 file to write to.
            comm (mpi.Comm, optional): The MPI communicator.
            ngalaxies_local (int, optional): The local number of galaxies.
            start_time (float, optional): The start time of the pipeline, used
                for timing information.
            verbose (int, optional): How verbose the output should be. 1 will
                only print on rank 0, 2 will print on all ranks, 0 will be
                silent. Defaults to 1.
            parallel_io (bool, optional): Whether to use parallel I/O. This
                requires h5py to be built with parallel support. Defaults to
                False.
        """
        # Store the file path
        self.filepath = filepath

        # Store the communicator and its properties
        self.comm = comm
        self.size = comm.Get_size() if comm is not None else 1
        self.rank = comm.Get_rank() if comm is not None else 0

        # Make sure we can write in parallel if requested
        if parallel_io and not self.PARALLEL:
            raise RuntimeError(
                "h5py not built with parallel support. "
                "Cannot use parallel_io option."
            )

        # Flags for behavior
        self.is_parallel = comm is not None
        self.is_root = self.rank == 0
        self.is_collective = self.is_parallel and self.PARALLEL and parallel_io

        # TODO: fix collective I/O
        if self.is_collective:
            self.is_collective = False
            warn(
                "Collective I/O currently locks up. "
                "Writing a file per rank instead. "
                "Feel free to contribute a fix!"
            )

        # Store the start time
        if start_time is None:
            self._start_time = time.perf_counter()
        else:
            self._start_time = start_time

        # Are we talking?
        self.verbose = verbose

        # Time how long we have to wait for everyone to get here
        start = time.perf_counter()
        if self.is_parallel:
            self.comm.Barrier()
            self._took(start, "Waiting for all ranks to get to I/O")

        # If we are writing in parallel but not using collective I/O we need
        # write a file per rank. Modify the file path to include the rank.
        ext = filepath.split(".")[-1]
        self.filepath = filepath.replace(f".{ext}", f"_{self.rank}.{ext}")

        # Report some useful information
        if self.is_collective:
            self._print(
                f"Writing in parallel to {filepath} "
                f"with {comm.Get_size()} ranks, and collective I/O."
            )
        elif self.is_parallel:
            self._print(
                f"Writing to {filepath} "
                f"with {comm.Get_size()} ranks, and a file per rank."
            )
        else:
            self._print(f"Writing to {filepath}.")

        # For collective I/O we need the counts on each rank so we know where
        # to write the data
        if self.is_collective or self.is_parallel:
            rank_gal_counts = self.comm.allgather(ngalaxies_local)
            self._ngalaxies_local = ngalaxies_local
            self.num_galaxies = sum(rank_gal_counts)
            self.start = sum(rank_gal_counts[: self.rank])
            self.end = self.start + ngalaxies_local
        else:
            self.num_galaxies = ngalaxies_local
            self._ngalaxies_local = ngalaxies_local
            self.start = 0
            self.end = ngalaxies_local

    def _print(self, *args, **kwargs):
        """Print a message to the screen with extra information.

        The prints behave differently depending on whether we are using MPI or
        not. We can also set the verbosity level at the Pipeline level which
        will control the verbosity of the print statements.

        Verbosity:
            0: No output beyond hello and goodbye.
            1: Outputs with timings but only on rank 0 (when using MPI).
            2: Outputs with timings on all ranks (when using MPI).

        Args:
            *args: The message to print.
            **kwargs: Additional arguments to pass to the print function.
        """
        # At verbosity 0 we are silent
        if self.verbose == 0:
            return

        # Get the current time code in seconds with 0 padding and 2
        # decimal places
        now = time.perf_counter() - self._start_time
        int_now = str(int(now)).zfill(
            len(str(int(now))) + 1 if now > 9999 else 5
        )
        decimal = str(now).split(".")[-1][:2]
        now_str = f"{int_now}.{decimal}"

        # Create the prefix for the print, theres extra info to output if
        # we are using MPI
        if self.is_parallel:
            # Only print on rank 0 if we are using MPI and have verbosity 1
            if self.verbose == 1 and self.rank != 0:
                return

            prefix = (
                f"[{str(self.rank).zfill(len(str(self.size)) + 1)}]"
                f"[{now_str}]:"
            )

        else:
            prefix = f"[{now_str}]:"

        print(prefix, *args, **kwargs)

    def _took(self, start, message):
        """Print a message with the time taken since the start time.

        Args:
            start (float): The start time of the process.
            message (str): The message to print.
        """
        elapsed = time.perf_counter() - start

        # Report in sensible units
        if elapsed < 1:
            elapsed *= 1000
            units = "ms"
        elif elapsed < 60:
            units = "s"
        else:
            elapsed /= 60
            units = "mins"

        # Report how blazingly fast we are
        self._print(f"{message} took {elapsed:.3f} {units}.")

    def create_file_with_metadata(self, instruments, emission_model):
        """Write metadata to the HDF5 file.

        This writes useful metadata to the root group of the HDF5 file and
        outputs the instruments and emission model to the appropriate groups.

        Args:
            instruments (dict): A dictionary of instrument objects.
            emission_model (dict): A dictionary of emission model objects.
        """
        start = time.perf_counter()

        # Unpack the instruments into a single instrument collection containing
        # all unique instruments
        unique_instruments = InstrumentCollection()
        for insts in instruments.values():
            for inst in insts:
                if inst.label in unique_instruments:
                    continue
                unique_instruments.add_instruments(inst)
        instruments = unique_instruments

        # Only write this metadata once
        if self.is_root:
            with h5py.File(self.filepath, "w") as hdf:
                # Write out some top level metadata
                hdf.attrs["synthesizer_version"] = __version__

                # Create groups for the instruments, emission model, and
                # galaxies
                inst_group = hdf.create_group("Instruments")
                model_group = hdf.create_group("EmissionModel")
                hdf.create_group("Galaxies")  # we'll use this in a mo

                # Write out the instruments
                inst_start = time.perf_counter()
                inst_group.attrs["ninstruments"] = instruments.ninstruments
                for label, instrument in instruments.items():
                    instrument.to_hdf5(inst_group.create_group(label))
                self._took(inst_start, "Writing instruments")

                # Write out the emission model
                model_start = time.perf_counter()
                for label, model in emission_model.items():
                    model.to_hdf5(model_group.create_group(label))
                self._took(model_start, "Writing emission model")

                self._took(start, "Writing metadata")

        if self.is_parallel:
            self.comm.Barrier()

    def write_dataset(self, data, key):
        """Write a dataset to an HDF5 file.

        We handle various different cases here:
        - If the data is a unyt object, we write the value and units.
        - If the data is a string we'll convert it to a h5py compatible string
          and write it with dimensionless units.
        - If the data is a numpy array, we write the data and set the units to
          "dimensionless".

        Args:
            data (any): The data to write.
            key (str): The key to write the data to.
        """
        # Ensure we are working with a unyt_array
        data = unyt_array(data)

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
        with h5py.File(self.filepath, "a") as hdf:
            dset = hdf.create_dataset(key, data=data, compression="gzip")
            dset.attrs["Units"] = units

    def write_dataset_parallel(self, data, key):
        """Write a dataset to an HDF5 file in parallel.

        This function requires that h5py has been built with parallel support.

        Args:
            data (any): The data to write.
            key (str): The key to write the data to.
        """
        # If we have an array of strings, convert to a h5py compatible string
        if data.dtype.kind == "U":
            data = np.array([d.encode("utf-8") for d in data])

        # Write the data for our slice
        with h5py.File(
            self.filepath,
            "a",
            driver="mpio",
            comm=self.comm,
            compression="gzip",
        ) as hdf:
            dset = hdf[key]
            dset[self.start : self.end, ...] = data

        self._print(f"Wrote dataset {key} with shape {data.shape}")

    def write_datasets_recursive(self, data, key):
        """Write a dictionary to an HDF5 file recursively.

        Args:
            data (dict): The data to write.
            key (str): The key to write the data to.
        """
        # Early exit if data is None
        if data is None:
            return

        # If the data isn't a dictionary just write the dataset
        if not isinstance(data, dict):
            try:
                self.write_dataset(data, key)
            except TypeError as e:
                raise TypeError(
                    f"Failed to write dataset {key} (type={type(data)}) - {e}"
                )
            return

        # Loop over the data
        for k, v in data.items():
            self.write_datasets_recursive(v, f"{key}/{k}")

    def write_datasets_parallel(self, data, key, paths):
        """Write a dictionary to an HDF5 file recursively in parallel.

        This function requires that h5py has been built with parallel support.

        Args:
            data (dict): The data to write.
            key (str): The key to write the data to.
            paths (list): The paths (key structure) of the datasets to write.
        """
        # Loop over each path and write the data
        for path in paths:
            d = data
            for k in path.split("/"):
                d = d[k]
            self.write_dataset_parallel(d, f"{key}/{path}")

    def create_datasets_parallel(self, data, key):
        """Create datasets ready to be populated in parallel.

        This is only needed for collective I/O operations. We will first make
        the datasets here in serial so they can be written to in any order on
        any rank.

        Args:
            data (dict): The data to write.
            key (str): The key to write the data to.
        """
        start = time.perf_counter()

        # Get the shapes and dtypes of the data
        shapes, dtypes, units, paths = get_dataset_properties(data, self.comm)

        # Create the datasets
        if self.is_root:
            with h5py.File(self.filepath, "a") as hdf:
                for k, shape in shapes.items():
                    dset = hdf.create_dataset(
                        f"{key}/{k}",
                        shape=shape,
                        dtype=dtypes[k],
                    )
                    dset.attrs["Units"] = units[k]

        self._took(start, f"Creating datasets for {key}")

        return paths

    def write_data(self, data, key, indexes=None, root=0):
        """Write data using the appropriate method based on the environment.

        Args:
            data (any): The data to write.
            key (str): The key to write the data to.
            indexes (list, optional): The indexes to write the data to.
            root (int, optional): The root rank for gathering and writing.
        """
        try:
            start = time.perf_counter()
            # In parallel land we need to make sure we're on the same page with
            # the structure we are writing
            if self.is_parallel:
                data = unify_dict_structure_across_ranks(data, self.comm)

            # Early exit if data is empty
            if data is None or len(data) == 0:
                return

            # Use the appropriate write method
            if self.is_collective:
                # For collective I/O we need to create the datasets first, then
                # write the data to them in parallel.
                paths = self.create_datasets_parallel(data, key)
                self.comm.barrier()
                self.write_datasets_parallel(data, key, paths)
            else:
                # Otherwise, we can just write everything recursively. Bear
                # in mind that when using MPI this will write a file per
                # rank ready for later combination into a virtual file.
                self.write_datasets_recursive(data, key)

            self._took(start, f"Writing {key} (and subgroups)")

        except Exception as e:
            self._print(f"Failed to write {key} - {e}")

    def combine_rank_files(self):
        """Combine the rank files into a single file.

        Args:
            output_file (str): The name of the output file.
        """
        start = time.perf_counter()

        def _recursive_copy(src, dest, slice):
            """Recursively copy the contents of an HDF5 group.

            Args:
                src (h5py.Group): The source group.
                dest (h5py.Group): The destination group.
                slice: The slice (along the first axis) the data belongs to.
            """
            # Copy over the attributes
            for attr in src.attrs:
                dest.attrs[attr] = src.attrs[attr]

            # Loop over the items in the source group
            for k, v in src.items():
                if k in ["Instruments", "EmissionModel"]:
                    continue

                # If we found a group we need to recurse and create the group
                # in the destination file if it doesn't exist. We also need to
                # copy the attributes.
                if isinstance(v, h5py.Group):
                    # Create the group if it doesn't exist
                    if k not in dest:
                        dest.create_group(k)

                    # Recurse
                    _recursive_copy(v, dest[k], slice)

                elif slice is None:
                    # Just copy the dataset directly
                    dset = dest.create_dataset(
                        k,
                        data=v[...],
                        dtype=v.dtype,
                    )

                    # Copy the attributes
                    for attr in v.attrs:
                        dset.attrs[attr] = v.attrs[attr]

                else:
                    # If the dataset doesn't exist we need to create it
                    if k not in dest:
                        dset = dest.create_dataset(
                            k,
                            shape=(self.num_galaxies, *v.shape[1:]),
                            dtype=v.dtype,
                        )
                    else:
                        dset = dest[k]

                    # Copy the data into the slice
                    dset[slice, ...] = v[...]

                    # Copy the attributes
                    for attr in v.attrs:
                        dset.attrs[attr] = v.attrs[attr]

        # Get the number of galaxies on each rank
        starts = self.comm.gather(self.start, root=0)
        ends = self.comm.gather(self.end, root=0)

        # Only the root rank needs to do this work
        if not self.is_root:
            return

        # Define the new file path
        ext = self.filepath.split(".")[-1]
        path_no_ext = ".".join(self.filepath.split(".")[:-1])
        new_path = "_".join(path_no_ext.split("_")[:-1]) + f".{ext}"
        temp_path = "_".join(path_no_ext.split("_")[:-1]) + "_<rank>.hdf5"

        # Open the output file
        with h5py.File(new_path, "w") as hdf:
            # Loop over each rank file
            for rank in range(self.size):
                # Open the rank file
                with h5py.File(
                    temp_path.replace("<rank>", str(rank)),
                    "r",
                ) as rank_hdf:
                    # We only the metadata groups once
                    if rank == 0:
                        # Copy the instruments over (no slice needed)
                        hdf.create_group("Instruments")
                        _recursive_copy(
                            rank_hdf["Instruments"],
                            hdf["Instruments"],
                            slice=None,
                        )

                        # Copy the emission model over (no slice needed)
                        hdf.create_group("EmissionModel")
                        _recursive_copy(
                            rank_hdf["EmissionModel"],
                            hdf["EmissionModel"],
                            slice=None,
                        )

                    # Copy the contents of the rank file to the output file
                    _recursive_copy(
                        rank_hdf,
                        hdf,
                        slice=slice(starts[rank], ends[rank]),
                    )

                # Delete the rank file
                os.remove(temp_path.replace("<rank>", str(rank)))

        self._took(start, "Combining files")

    def combine_rank_files_virtual(self):
        """Combine the rank files into a single virtual file.

        Note that the virtual file this produces requires the rank files to
        remain in the same location as when they were created.
        """
        start = time.perf_counter()

        # Gather start/end indices from all ranks
        starts = self.comm.gather(self.start, root=0)
        ends = self.comm.gather(self.end, root=0)

        # Only the root rank (rank == 0) needs to create the virtual file
        if not self.is_root:
            return

        # Define file paths
        ext = self.filepath.split(".")[-1]
        path_no_ext = ".".join(self.filepath.split(".")[:-1])
        temp_path = "_".join(path_no_ext.split("_")[:-1]) + "_<rank>.hdf5"
        new_path = "_".join(path_no_ext.split("_")[:-1]) + f".{ext}"

        # Remove the old combined file if it exists
        if os.path.exists(new_path):
            os.remove(new_path)

        # Open the first rank file (rank 0 file) to discover the structure
        first_file_path = temp_path.replace("<rank>", "0")
        with h5py.File(first_file_path, "r") as f0:
            # Gather dataset information (for virtual datasets)
            datasets_info = []

            def gather_datasets(group, group_path="/"):
                for k, v in group.items():
                    if k in ["Instruments", "EmissionModel"]:
                        continue
                    current_path = f"{group_path}{k}"
                    if isinstance(v, h5py.Group):
                        gather_datasets(v, current_path + "/")
                    else:
                        # Record dataset path, shape, dtype, and attributes
                        datasets_info.append(
                            (current_path, v.shape, v.dtype, dict(v.attrs))
                        )

            gather_datasets(f0)

            # Gather group structure (to replicate in the virtual file)
            groups_info = []

            def gather_groups(group, group_path="/"):
                for k, v in group.items():
                    if k in ["Instruments", "EmissionModel"]:
                        continue
                    current_path = f"{group_path}{k}"
                    if isinstance(v, h5py.Group):
                        groups_info.append((current_path, dict(v.attrs)))
                        gather_groups(v, current_path + "/")

            gather_groups(f0)

            # Create the virtual file
            with h5py.File(new_path, "w") as hdf:
                for meta_group in ["Instruments", "EmissionModel"]:
                    if meta_group in f0:
                        hdf.copy(f0[meta_group], meta_group)

                # Create empty group structure
                for gpath, gattrs in groups_info:
                    grp = hdf.create_group(gpath)
                    for attr_name, attr_val in gattrs.items():
                        grp.attrs[attr_name] = attr_val

                # Construct the virtual datasets for each dataset
                for dpath, shape, dtype, dattrs in datasets_info:
                    final_shape = (self.num_galaxies,) + shape[1:]
                    layout = h5py.VirtualLayout(shape=final_shape, dtype=dtype)

                    # Loop over each rank and map the data to the layout
                    for rank, (start_i, end_i) in enumerate(zip(starts, ends)):
                        src_file = temp_path.replace("<rank>", str(rank))
                        with h5py.File(src_file, "r") as rank_f:
                            src_dset = rank_f[dpath]
                            local_size = src_dset.shape[0]

                        # Ensure the local size is as expected
                        expected_size = end_i - start_i
                        if local_size != expected_size:
                            end_i = start_i + local_size

                        # Create a virtual source for this rank
                        vsource = h5py.VirtualSource(
                            src_file, dpath, shape=(local_size,) + shape[1:]
                        )

                        # Map the portion of the layout to this source slice
                        layout[start_i:end_i, ...] = vsource[...]

                    # Create the virtual dataset in the final file
                    vds = hdf.create_virtual_dataset(dpath, layout)

                    # Copy dataset attributes
                    for attr_name, attr_val in dattrs.items():
                        vds.attrs[attr_name] = attr_val

        self._took(start, "Creating virtual file")
