"""A module for handling I/O operations in the pipeline.

This module contains classes and functions for reading and writing data
in the pipeline. This includes reading and writing HDF5 files, as well as
handling the MPI communications for parallel I/O operations.

Example usage:

    # Write data to an HDF5 file
    writer = HDF5Writer(hdf)
    writer.write_data(data, key)


"""

import time

import h5py
import numpy as np
from unyt import unyt_array

from synthesizer._version import __version__


class PipelineIO:
    """
    A class for writing data to an HDF5 file.

    This class provides methods for writing data to an HDF5 file. It can
    handle writing data in parallel using MPI if the h5py library has been
    built with parallel support.

    Example usage:

            # Write data to an HDF5 file
            writer = HDF5Writer(hdf)
            writer.write_data(data, key)

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
        num_galaxies=None,
        start_time=None,
        verbose=1,
    ):
        """
        Initialize the HDF5Writer class.

        Args:
            hdf (h5py.File): The HDF5 file to write to.
            comm (mpi.Comm, optional): The MPI communicator.
            num_galaxies (int, optional): The total number of galaxies.
            pipeline (Pipeline): The pipeline object.
            start_time (float, optional): The start time of the pipeline, used
                for timing information.
            verbose (int, optional): How verbose the output should be. 1 will
                only print on rank 0, 2 will print on all ranks, 0 will be
                silent. Defaults to 1.
        """
        # Open the HDF5 file, either in MPI mode or not based on the h5py
        # build and the communicator provided
        if comm is not None and self.PARALLEL:
            self.hdf = h5py.File(filepath, "w", driver="mpio", comm=comm)
        else:
            self.hdf = h5py.File(filepath, "w")

        # Store the communicator and number of galaxies
        self.comm = comm
        self.num_galaxies = num_galaxies
        self.rank = comm.Get_rank() if comm is not None else None

        # Flags for behavior
        self.is_parallel = comm is not None
        self.is_root = self.rank == 0 if self.is_parallel else True
        self.is_collective = self.is_parallel and self.PARALLEL

        # Store the start time
        if start_time is None:
            self._start_time = time.perf_counter()
        else:
            self._start_time = start_time

        # Are we talking?
        self.verbose = verbose

        # Report some useful information
        self._print(f"Writing to {filepath}.")
        if self.is_parallel:
            self._print(f"Writing in parallel with {comm.Get_size()} ranks.")
        if self.is_collective:
            self._print("Using collective I/O.")

        # Time how long we have to wait for everyone to get here
        start = time.perf_counter()
        if self.is_parallel:
            self.comm.barrier()
            self._took(start, "Waiting for all ranks to get to I/O.")

    def __del__(self):
        """Close the HDF5 file when the object is deleted."""
        self.hdf.close()

    def _print(self, *args, **kwargs):
        """
        Print a message to the screen with extra information.

        The prints behave differently depending on whether we are using MPI or
        not. We can also set the verbosity level at the Survey level which will
        control the verbosity of the print statements.

        Verbosity:
            0: No output beyond hello and goodbye.
            1: Outputs with timings but only on rank 0 (when using MPI).
            2: Outputs with timings on all ranks (when using MPI).

        Args:
            message (str): The message to print.
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
        """
        Print a message with the time taken since the start time.

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

    def write_metadata(self, instruments, emission_model):
        """
        Write metadata to the HDF5 file.

        This writes useful metadata to the root group of the HDF5 file and
        outputs the instruments and emission model to the appropriate groups.

        Args:
            instruments (dict): A dictionary of instrument objects.
            emission_model (dict): A dictionary of emission model objects.
        """
        start = time.perf_counter()

        # Only write this metadata once
        if self.is_root:
            # Write out some top level metadata
            self.hdf.attrs["synthesizer_version"] = __version__

            # Create groups for the instruments, emission model, and
            # galaxies
            inst_group = self.hdf.create_group("Instruments")
            model_group = self.hdf.create_group("EmissionModel")
            self.hdf.create_group("Galaxies")  # we'll use this in a mo

            # Write out the instruments
            inst_group.attrs["ninstruments"] = instruments.ninstruments
            for label, instrument in instruments.items():
                instrument.to_hdf5(inst_group.create_group(label))

            # Write out the emission model
            for label, model in emission_model.items():
                model.to_hdf5(model_group.create_group(label))

            self._took(start, "Writing metadata")

    def write_dataset(self, data, key):
        """
        Write a dataset to an HDF5 file.

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
        start = time.perf_counter()

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
        dset = self.hdf.create_dataset(key, data=data)
        dset.attrs["Units"] = units

        self._took(start, f"Writing dataset {key}")

    def write_datasets_recursive(self, data, key):
        """
        Write a dictionary to an HDF5 file recursively.

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

    def write_datasets_recursive_parallel(self, data, key, indexes):
        """
        Write a dictionary to an HDF5 file recursively in parallel.

        This function requires that h5py has been built with parallel support.

        Args:
            data (dict): The data to write.
            key (str): The key to write the data to.
            indexes (array): The sorting indices.
        """
        if not self.is_parallel:
            raise RuntimeError(
                "Parallel write requested but no MPI communicator provided."
            )

        # If the data isn't a dictionary, write the dataset
        if not isinstance(data, dict):
            start = time.perf_counter()

            try:
                local_shape = data.shape

                # Only rank 0 creates the dataset
                if self.rank == 0:
                    # Determine dtype
                    if hasattr(data, "value"):
                        dtype = data.value.dtype
                    else:
                        dtype = data.dtype

                    # Calculate global shape
                    global_shape = [self.num_galaxies]
                    for i in range(1, len(local_shape)):
                        global_shape.append(local_shape[i])

                    # Create the dataset
                    dset = self.hdf.create_dataset(
                        key,
                        shape=tuple(global_shape),
                        dtype=dtype,
                    )

                    # Handle units if present
                    if hasattr(data, "units"):
                        dset.attrs["Units"] = str(data.units)

                # Synchronize all ranks before writing
                self.comm.barrier()

                # Get the dataset on all ranks
                dset = self.hdf[key]

                # Set collective I/O property for the write operation
                with dset.collective:
                    # Write the data using the appropriate slice
                    start = sum(
                        self.comm.allgather(local_shape[0])[: self.rank]
                    )
                    end = start + local_shape[0]
                    dset[start:end, ...] = data

            except Exception as e:
                raise RuntimeError(
                    f"Failed to write dataset '{key}' on rank {self.rank}: {e}"
                )

            self._took(start, f"Writing dataset {key}")

            return

        # Recursively handle dictionary data
        for k, v in data.items():
            self.write_datasets_recursive_parallel(v, f"{key}/{k}", indexes)

    def gather_and_write_datasets(self, data, key, root=0):
        """
        Recursively collect data from all ranks onto the root and write it out.

        We will recurse through the dictionary and gather all arrays or lists
        at the leaves and write them out to the HDF5 file. Doing so limits the
        size of communications and minimises the amount of data we have
        collected on the root rank at any one time.

        Args:
            data (any): The data to gather.
            key (str): The key to write the data to.
            root (int): The root rank to gather data to.
        """
        if not self.is_parallel:
            raise RuntimeError(
                "Gather and write requested but no MPI communicator provided."
            )

        # If the data is a dictionary we need to recurse
        if isinstance(data, dict):
            for k, v in data.items():
                self.gather_and_write_datasets(v, f"{key}/{k}", root)
            return

        start = time.perf_counter()

        # First gather the data
        collected_data = self.comm.gather(data, root=root)

        # If we aren't the root we're done
        if collected_data is None:
            return

        # Remove any empty datasets
        collected_data = [d for d in collected_data if len(d) > 0]

        # If there's nothing to write we're done
        if len(collected_data) == 0:
            return

        # Combine the list of arrays into a single unyt_array
        try:
            combined_data = unyt_array(np.concatenate(collected_data))
        except ValueError as e:
            raise ValueError(f"Failed to concatenate {key} - {e}")

        self._took(start, f"Gathering {key}")

        # Write the dataset
        try:
            self.write_dataset(combined_data, key)
        except TypeError as e:
            print(type(data), type(combined_data))
            raise TypeError(f"Failed to write dataset {key} - {e}")

        # Clear collected data explicitly
        del collected_data

    def write_data(self, data, key, indexes=None, root=0):
        """
        Write data using the appropriate method based on the environment.

        Args:
            data (any): The data to write.
            key (str): The key to write the data to.
            indexes (array, optional): The sorting indices for parallel writes.
            root (int, optional): The root rank for gathering and writing.
        """
        if self.is_parallel:
            if indexes is not None:
                self.write_datasets_recursive_parallel(data, key, indexes)
            else:
                self.gather_and_write_datasets(data, key, root)
        else:
            self.write_datasets_recursive(data, key)
