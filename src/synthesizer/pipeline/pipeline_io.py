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
from synthesizer.pipeline.pipeline_utils import (
    get_dataset_properties,
    unify_dict_structure_across_ranks,
)


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
        ngalaxies_local=None,
        start_time=None,
        verbose=1,
    ):
        """
        Initialize the HDF5Writer class.

        Args:
            hdf (h5py.File): The HDF5 file to write to.
            comm (mpi.Comm, optional): The MPI communicator.
            ngalaxies_local (int, optional): The local number of galaxies.
            pipeline (Pipeline): The pipeline object.
            start_time (float, optional): The start time of the pipeline, used
                for timing information.
            verbose (int, optional): How verbose the output should be. 1 will
                only print on rank 0, 2 will print on all ranks, 0 will be
                silent. Defaults to 1.
        """
        # Store the file path
        self.filepath = filepath

        # Create the private hdf attributes we'll use to store the file object
        # itself when its open
        self._hdf = None
        self._hdf_mpi = None

        # Create the file in memory ready to be appended to in the methods
        # below (this will overwrite any existing file)
        h5py.File(filepath, "w").close()

        # Store the communicator and its properties
        self.comm = comm
        self.size = comm.Get_size() if comm is not None else 1
        self.rank = comm.Get_rank() if comm is not None else 0

        # Flags for behavior
        self.is_parallel = comm is not None
        self.is_root = self.rank == 0
        self.is_collective = self.is_parallel and self.PARALLEL

        # Store the start time
        if start_time is None:
            self._start_time = time.perf_counter()
        else:
            self._start_time = start_time

        # Are we talking?
        self.verbose = verbose

        # Report some useful information
        if self.is_collective:
            self._print(
                f"Writing in parallel to {filepath} "
                f"with {comm.Get_size()} ranks, and collective I/O."
            )
        elif self.is_parallel:
            self._print(
                f"Writing in parallel to {filepath} "
                f"with {comm.Get_size()} ranks."
            )
        else:
            self._print(f"Writing to {filepath}.")

        # Time how long we have to wait for everyone to get here
        start = time.perf_counter()
        if self.is_parallel:
            self.comm.Barrier()
            self._took(start, "Waiting for all ranks to get to I/O")

        # For collective I/O we need the counts on each rank so we know where
        # to write the data
        if self.is_collective:
            rank_gal_counts = self.comm.allgather(ngalaxies_local)
            self.num_galaxies = sum(rank_gal_counts)
            self.start = sum(rank_gal_counts[: self.rank])
            self.end = self.start + ngalaxies_local
        else:
            self.num_galaxies = ngalaxies_local
            self.start = 0
            self.end = ngalaxies_local

    def __del__(self):
        """Close the HDF5 file when the object is deleted."""
        self.close()

    @property
    def hdf(self):
        """Return a reference to the HDF5 file for serial writes."""
        if self._hdf is None:
            self._print("Opening HDF5 file in serial mode.")
            self._hdf = h5py.File(self.filepath, "a")
        return self._hdf

    @property
    def hdf_mpi(self):
        """Return a reference to the HDF5 file for parallel writes."""
        if self._hdf_mpi is None:
            self._print("Opening HDF5 file in parallel mode.")
            self._hdf_mpi = h5py.File(
                self.filepath,
                "a",
                driver="mpio",
                comm=self.comm,
            )
        return self._hdf_mpi

    def close(self):
        """Close the HDF5 file."""
        self._print("Attempting to close HDF5 file.")
        if self._hdf is not None:
            self._hdf.close()
            self._hdf = None
        if self._hdf_mpi is not None:
            self._hdf_mpi.close()
            self._hdf_mpi = None

    def _print(self, *args, **kwargs):
        """
        Print a message to the screen with extra information.

        The prints behave differently depending on whether we are using MPI or
        not. We can also set the verbosity level at the Pipeline level which
        will control the verbosity of the print statements.

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

        # Close the open file
        self.close()

        if self.is_parallel:
            self.comm.Barrier()

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

    def write_dataset_parallel(self, data, key):
        """
        Write a dataset to an HDF5 file in parallel.

        This function requires that h5py has been built with parallel support.

        Args:
            data (any): The data to write.
            key (str): The key to write the data to.
        """
        if not self.is_parallel:
            raise RuntimeError(
                "Parallel write requested but no MPI communicator provided."
            )

        # If we have an array of strings, convert to a h5py compatible string
        if data.dtype.kind == "U":
            data = np.array([d.encode("utf-8") for d in data])

        # Write the data for our slice
        self.hdf_mpi[key][self.start : self.end, ...] = data

        self._print(f"Writing dataset {key} with shape {data.shape}")

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
            self.write_dataset_parallel(unyt_array(data), key)
            return

        # Recursively handle dictionary data
        for k, v in data.items():
            self._print(f"Recursing into {key}/{k}, {type(v)}")
            self.write_datasets_recursive_parallel(v, f"{key}/{k}", indexes)

    def create_datasets_parallel(self, data, key):
        """
        Create datasets ready to be populated in parallel.

        This is only needed for collective I/O operations. We will first make
        the datasets here in serial so they can be written to in any order on
        any rank.

        Args:
            shapes (dict): The shapes of the datasets to create.
            dtypes (dict): The data types of the datasets to create.
        """
        start = time.perf_counter()

        # Get the shapes and dtypes of the data
        shapes, dtypes, units = get_dataset_properties(data, self.comm)

        # Create the datasets
        if self.is_root:
            for k, shape in shapes.items():
                dset = self.hdf.create_dataset(
                    f"{key}/{k}",
                    shape=shape,
                    dtype=dtypes[k],
                )
                dset.attrs["Units"] = units[k]
            self.close()

        self.comm.Barrier()

        self._took(start, f"Creating datasets for {key}")

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
            self.create_datasets_parallel(data, key)
            self.write_datasets_recursive_parallel(data, key, indexes)
        elif self.is_parallel:
            self.gather_and_write_datasets(data, key, root)
        else:
            self.write_datasets_recursive(data, key)

        self._took(start, f"Writing {key} (and subgroups)")
