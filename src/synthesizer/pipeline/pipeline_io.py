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
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import ClassVar

import h5py
import numpy as np
from unyt import unyt_array

from synthesizer import exceptions
from synthesizer._version import __version__
from synthesizer.instruments import InstrumentCollection
from synthesizer.pipeline.pipeline_utils import (
    clear_lists_recursive,
    combine_list_of_dicts,
    get_buffered_nbytes,
    get_dataset_properties,
    get_dict_structure,
    get_row_template,
    iter_row_lists,
    make_empty_row,
    rows_to_arrays,
    unify_dict_structure_across_ranks,
)
from synthesizer.synth_warnings import warn
from synthesizer.utils.operation_timers import timed

# These live under Galaxies alongside the outputs, but describe the whole
# run rather than one row per galaxy. Rank 0 writes them once.
_SHARED_AXIS_GROUPS = (
    "/Galaxies/Stars/SFZHAxes",
    "/Galaxies/Stars/SFHAxes",
)


@contextmanager
def _hdf5_replaced_on_success(path):
    """Open an HDF5 file for writing which only replaces path on success.

    The file is written to a temporary path next to path, so a failure while
    writing leaves any existing file at path untouched, rather than an
    incomplete file which looks valid.

    Args:
        path (str): The path the file should end up at.

    Yields:
        h5py.File: The open file to write to.
    """
    temp_path = f"{path}.tmp"
    try:
        with h5py.File(temp_path, "w") as hdf:
            yield hdf

        # Everything was written (and the file closed), so swap it in
        os.replace(temp_path, path)
    except BaseException:
        # Clean up the incomplete file before re-raising
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def _per_component():
    """Return an empty container for each component's outputs."""
    return {"Galaxy": {}, "Stars": {}, "BlackHole": {}}


@dataclass(eq=False, repr=False)
class PipelineIOBuffers:
    """The outputs a Pipeline holds in memory until written.

    Each per-galaxy output is a (nested) container of lists with a row per
    galaxy held. The Pipeline appends a row for each galaxy it unpacks and
    the buffers are emptied once their rows have been written, either at
    the end of the run or, if flushing was requested, whenever a flush
    threshold is reached.

    Not every galaxy produces every output (e.g. a galaxy without black
    holes has no black hole spectra), so the buffers also pad the outputs a
    galaxy didn't produce, keeping every row in line with its galaxy.

    The outputs that aren't per galaxy (e.g. the cosmic SEDs) are also held
    here but are accumulated over the whole run, so they are only written
    once, at the end.

    Attributes:
        flush_ngalaxies (int):
            The number of galaxies at which to flush to file (None for no
            flushing based on the number of galaxies).
        flush_memory (float):
            The memory (in GB) at which to flush to file (None for no
            flushing based on memory).
        analysis_results (dict):
            The results of each extra analysis function, keyed by the
            function's result key.
        cosmic_lnus (dict):
            The cosmic SEDs (spectral luminosity densities) for each
            component, keyed by label and then emission model label.
        cosmic_fnus (dict):
            The observed cosmic SEDs (spectral flux densities) for each
            component, keyed by label and then emission model label.
        cosmic_lnu_counts (dict):
            The number of galaxies contributing to each cosmic_lnus entry.
        cosmic_fnu_counts (dict):
            The number of galaxies contributing to each cosmic_fnus entry.
        line_lams (unyt_array):
            The rest frame wavelengths of the emission lines (shared by
            all galaxies).
        line_ids (list):
            The IDs of the emission lines (shared by all galaxies).
        ngalaxies (int):
            The number of galaxies held since the last flush.
        nbytes (int):
            The bytes held by the outputs since the last flush (only
            counted when flushing based on memory).
    """

    # The per-galaxy outputs (excluding the extra analysis results), each
    # with the Pipeline flag requesting it and where it's written. Outputs
    # held per component are written beneath each component's group (see
    # COMPONENT_GROUPS), the others at the path given.
    GALAXY_OUTPUTS: ClassVar[dict] = {
        "sfzhs": ("_write_sfzh", "Galaxies/Stars/SFZH"),
        "sfhs": ("_write_sfh", "Galaxies/Stars/SFH"),
        "lnu_spectra": (
            "_write_lnu_spectra",
            "Spectra/SpectralLuminosityDensities",
        ),
        "fnu_spectra": (
            "_write_fnu_spectra",
            "Spectra/SpectralFluxDensities",
        ),
        "luminosities": ("_write_luminosities", "Photometry/Luminosities"),
        "fluxes": ("_write_fluxes", "Photometry/Fluxes"),
        "line_lums": ("_write_lines", "Lines/Luminosity"),
        "line_cont_lums": ("_write_lines", "Lines/Continuum"),
        "line_fluxes": ("_write_flux_lines", "Lines/Flux"),
        "line_cont_fluxes": ("_write_flux_lines", "Lines/ContinuumFlux"),
        "line_obs_lams": (
            "_write_flux_lines",
            "Galaxies/Lines/ObservedWavelengths",
        ),
        "images_lum": ("_write_images_lum", "Images/Luminosity"),
        "images_lum_psf": ("_write_images_lum_psf", "PSFImages/Luminosity"),
        "images_lum_noise": (
            "_write_images_lum_noise",
            "NoiseImages/Luminosity",
        ),
        "images_flux": ("_write_images_flux", "Images/Flux"),
        "images_flux_psf": ("_write_images_flux_psf", "PSFImages/Flux"),
        "images_flux_noise": ("_write_images_flux_noise", "NoiseImages/Flux"),
        "line_maps_lum": ("_write_line_maps_lum", "LineMaps/Luminosity"),
        "line_maps_lum_psf": (
            "_write_line_maps_lum_psf",
            "PSFLineMaps/Luminosity",
        ),
        "line_maps_lum_noise": (
            "_write_line_maps_lum_noise",
            "NoiseLineMaps/Luminosity",
        ),
        "line_maps_flux": ("_write_line_maps_flux", "LineMaps/Flux"),
        "line_maps_flux_psf": (
            "_write_line_maps_flux_psf",
            "PSFLineMaps/Flux",
        ),
        "line_maps_flux_noise": (
            "_write_line_maps_flux_noise",
            "NoiseLineMaps/Flux",
        ),
    }

    # The group each component's outputs are written beneath
    COMPONENT_GROUPS: ClassVar[dict] = {
        "Galaxy": "Galaxies",
        "Stars": "Galaxies/Stars",
        "BlackHole": "Galaxies/BlackHoles",
    }

    # The flushing thresholds
    flush_ngalaxies: int = None
    flush_memory: float = None

    # The outputs themselves
    sfzhs: list = field(init=False, default_factory=list)
    sfhs: list = field(init=False, default_factory=list)
    lnu_spectra: dict = field(init=False, default_factory=_per_component)
    fnu_spectra: dict = field(init=False, default_factory=_per_component)
    luminosities: dict = field(init=False, default_factory=_per_component)
    fluxes: dict = field(init=False, default_factory=_per_component)
    line_lums: dict = field(init=False, default_factory=_per_component)
    line_cont_lums: dict = field(init=False, default_factory=_per_component)
    line_fluxes: dict = field(init=False, default_factory=_per_component)
    line_cont_fluxes: dict = field(init=False, default_factory=_per_component)
    line_obs_lams: list = field(init=False, default_factory=list)
    images_lum: dict = field(init=False, default_factory=_per_component)
    images_lum_psf: dict = field(init=False, default_factory=_per_component)
    images_lum_noise: dict = field(init=False, default_factory=_per_component)
    images_flux: dict = field(init=False, default_factory=_per_component)
    images_flux_psf: dict = field(init=False, default_factory=_per_component)
    images_flux_noise: dict = field(init=False, default_factory=_per_component)
    line_maps_lum: dict = field(init=False, default_factory=_per_component)
    line_maps_lum_psf: dict = field(init=False, default_factory=_per_component)
    line_maps_lum_noise: dict = field(
        init=False, default_factory=_per_component
    )
    line_maps_flux: dict = field(init=False, default_factory=_per_component)
    line_maps_flux_psf: dict = field(
        init=False, default_factory=_per_component
    )
    line_maps_flux_noise: dict = field(
        init=False, default_factory=_per_component
    )
    analysis_results: dict = field(init=False, default_factory=dict)

    # The outputs that aren't per galaxy. These are accumulated over the
    # whole run (flushing leaves them alone) and written once at the end.
    cosmic_lnus: dict = field(init=False, default_factory=_per_component)
    cosmic_fnus: dict = field(init=False, default_factory=_per_component)
    line_lams: unyt_array = field(init=False, default=None)
    line_ids: list = field(init=False, default=None)
    lnu_data_cubes: dict = field(init=False, default_factory=_per_component)
    fnu_data_cubes: dict = field(init=False, default_factory=_per_component)
    spectroscopy: dict = field(init=False, default_factory=_per_component)

    # The number of galaxies contributing to each cosmic SED, so they can
    # be averaged once every rank's contributions have been summed
    cosmic_lnu_counts: dict = field(init=False, default_factory=_per_component)
    cosmic_fnu_counts: dict = field(init=False, default_factory=_per_component)

    # What we hold since the last flush
    ngalaxies: int = field(init=False, default=0)
    nbytes: int = field(init=False, default=0)

    # The shape, dtype and units of a row of each output, so we can pad
    # the outputs a galaxy doesn't produce. This outlives the rows
    # themselves since flushing empties the lists.
    _row_templates: dict = field(init=False, default_factory=dict)

    # The keys of the outputs the current galaxy has produced so far (see
    # append_output)
    _appended: set = field(init=False, default_factory=set)

    # The key structure of each analysis result from the first flush,
    # later flushes must match it to append consistently
    _analysis_result_structures: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        """Validate the flushing thresholds."""
        # Don't accept a zero or negative threshold: we'd otherwise flush
        # immediately after every galaxy, which is a surprising way to ask
        # for no buffering. Numpy integers are fine but booleans aren't
        if self.flush_ngalaxies is not None:
            if (
                not isinstance(self.flush_ngalaxies, (int, np.integer))
                or isinstance(self.flush_ngalaxies, bool)
                or self.flush_ngalaxies < 1
            ):
                raise exceptions.InconsistentArguments(
                    "flush_ngalaxies must be an int >= 1"
                )
            self.flush_ngalaxies = int(self.flush_ngalaxies)
        if self.flush_memory is not None and not (
            0 < self.flush_memory < float("inf")
        ):
            raise exceptions.InconsistentArguments("flush_memory must be > 0")

    @property
    def flushing(self):
        """Return whether outputs are flushed to file while running."""
        return (
            self.flush_ngalaxies is not None or self.flush_memory is not None
        )

    @property
    def flush_nbytes(self):
        """Return the memory threshold in bytes (None if not set)."""
        if self.flush_memory is None:
            return None
        return self.flush_memory * 1024**3

    def _iter_row_lists(self):
        """Yield (key, rows) for every per-galaxy output list.

        Yields:
            tuple:
                The output's key (its name followed by its path through
                the nested containers) and its list of rows.
        """
        for name in self.GALAXY_OUTPUTS:
            for path, rows in iter_row_lists(getattr(self, name)):
                yield (name, *path), rows

    def _hold(self, rows, new_rows):
        """Add rows to an output's list, counting their memory if needed.

        Args:
            rows (list):
                The output's list of rows.
            new_rows (list):
                The rows to add to the end of the list.
        """
        rows.extend(new_rows)

        # Track how much memory we're holding if we flush on memory
        if self.flush_memory is not None:
            self.nbytes += get_buffered_nbytes(new_rows)

    def append_output(self, value, name, *keys):
        """Append the current galaxy's row to a per-galaxy output.

        Any nested containers along the way are created on first use, so
        an output is only held once a galaxy produces it.

        Args:
            value (array-like):
                The galaxy's row.
            name (str):
                The name of the output (a key of GALAXY_OUTPUTS).
            *keys (str):
                The path to the row's list through the output's nested
                containers (e.g. the component, then the emission model
                label). Omitted for outputs which are a single list.
        """
        # Find the list for this row, creating it (and any containers above
        # it) if this is the first time it's been produced
        rows = getattr(self, name)
        if keys:
            for key in keys[:-1]:
                rows = rows.setdefault(key, {})
            rows = rows.setdefault(keys[-1], [])

        # If no galaxy has produced this output before, pad a row for every
        # galaxy we already hold. Galaxies already flushed are padded by the
        # IO helper when it creates the dataset.
        key = (name, *keys)
        if key not in self._row_templates:
            self._row_templates[key] = get_row_template(value)
            self._hold(
                rows,
                [
                    make_empty_row(self._row_templates[key])
                    for _ in range(self.ngalaxies)
                ],
            )

        # Store the galaxy's row and note the galaxy produced this output
        self._hold(rows, [value])
        self._appended.add(key)

    def add_analysis_output(self, key):
        """Add an output for the results of an extra analysis function.

        There is no equivalent to this method for the other outputs because
        their keys are known ahead of time, whereas the keys of analysis
        outputs are entirely arbitrary (they're chosen by the user).

        Args:
            key (str):
                The key of the analysis function's results.

        Raises:
            InconsistentArguments:
                If an analysis output with this key already exists.
        """
        if key in self.analysis_results:
            raise exceptions.InconsistentArguments(
                f"{key} already exists in the analysis results. "
                "Choose a different result_key"
            )
        self.analysis_results[key] = []

    def append_analysis_output(self, key, result):
        """Append the current galaxy's result of an extra analysis function.

        Args:
            key (str):
                The key of the analysis function's results.
            result (object):
                The galaxy's result.
        """
        self._hold(self.analysis_results[key], [result])

    def end_galaxy(self):
        """Pad the outputs a galaxy didn't produce and count the galaxy.

        Without a row for every galaxy, every later row would be written
        against the wrong galaxy index, so we pad the missing outputs with
        empty rows shaped like the rows we already hold.
        """
        # Pad any output this galaxy didn't append to. We can only do so if
        # we know what a row looks like, if we don't then no galaxy has
        # produced the output yet (e.g. the SFZH list exists from the start)
        for key, rows in self._iter_row_lists():
            if key not in self._appended and key in self._row_templates:
                self._hold(rows, [make_empty_row(self._row_templates[key])])

        # Count the galaxy and start afresh for the next one
        self._appended.clear()
        self.ngalaxies += 1

    def should_flush(self):
        """Return whether the outputs held should be flushed to file.

        Returns:
            bool: True if either flush threshold has been reached.
        """
        # A normal run still asks after each galaxy, so leave its in-memory
        # outputs alone unless flushing was requested
        if not self.flushing:
            return False

        # Have we held enough galaxies?
        if (
            self.flush_ngalaxies is not None
            and self.ngalaxies >= self.flush_ngalaxies
        ):
            return True

        # Have we held enough memory? (Galaxies can differ substantially in
        # size so this isn't necessarily a fixed number of galaxies.)
        if self.flush_memory is None:
            return False
        return self.nbytes >= self.flush_nbytes

    def to_arrays(self):
        """Convert each list of rows held into a single array.

        This is done once the run is complete, so the outputs left in memory
        are arrays (rather than lists of rows) when they are written.
        """
        for name in self.GALAXY_OUTPUTS:
            setattr(self, name, rows_to_arrays(getattr(self, name)))

    def combine_analysis_results(self):
        """Combine the lists of extra analysis results into unyt arrays.

        Raises:
            BadResult:
                If an analysis function's results differ in type or
                structure between galaxies (or from an earlier flush).
        """
        # Unlike the other outputs we need to do some checks here to ensure
        # that the data is in a consistent format. This is because the user
        # can return any type of data from their analysis functions.
        for key, res in self.analysis_results.items():
            # Ensure the data of all results is in the same format.
            types = set([type(r) for r in res])
            if len(types) > 1:
                raise exceptions.BadResult(
                    "All results from extra analysis functions must be "
                    f"of the same type. Got: {set(types)}"
                )

            # Ensure all results have the same keys (the merge below would
            # silently drop any extras)
            self._check_analysis_structure(key, res)

            # Combine the list of results into a dict of unyt arrays or a
            # single unyt array. This will simply return a unyt array if we
            # hand a list of values
            self.analysis_results[key] = combine_list_of_dicts(res)

    def _check_analysis_structure(self, key, res):
        """Ensure every result of an analysis function has the same keys.

        The structure is remembered between calls since each flush appends
        to the datasets the first flush created, so a result with a
        different structure would leave some datasets missing rows.

        Args:
            key (str):
                The key of the analysis function.
            res (list):
                The results of the analysis function for each galaxy.

        Raises:
            BadResult:
                If a result's structure differs from the expected one.
        """
        # Nothing to check
        if len(res) == 0:
            return

        # Compare against an earlier flush if there was one, otherwise
        # against the first result
        expected = self._analysis_result_structures.setdefault(
            key, get_dict_structure(res[0])
        )
        for result in res:
            structure = get_dict_structure(result)
            if structure != expected:
                raise exceptions.BadResult(
                    f"Extra analysis result {key} changed structure. "
                    f"Expected {expected}, got {structure}."
                )

    def write(self, io_helper, requested):
        """Write the rows held to file.

        On each call the rows are appended to anything written by earlier
        calls (i.e. earlier flushes).

        Args:
            io_helper (PipelineIO):
                The writer for the output file.
            requested (set):
                The names of the outputs the Pipeline was asked for. Only
                these are written, the others were never collected.
        """
        # An index marks these as per-galaxy data to the writer. In MPI
        # each rank writes its own file, so the index also avoids
        # collective calls during a flush (ranks need not flush at the same
        # galaxy).
        indexes = io_helper.galaxy_indices
        if indexes is None:
            indexes = range(self.ngalaxies)

        # Write each requested output, beneath each component's group if
        # it's held per component
        for name, (_, path) in self.GALAXY_OUTPUTS.items():
            if name not in requested:
                continue
            data = getattr(self, name)
            if not isinstance(data, dict):
                io_helper.write_data(data, path, indexes)
                continue
            for component, group in self.COMPONENT_GROUPS.items():
                io_helper.write_data(
                    data[component], f"{group}/{path}", indexes
                )

        # Write out the extra analysis results
        for key, res in self.analysis_results.items():
            io_helper.write_data(res, f"Galaxies/{key}", indexes)

        # Record the rows we've written so datasets first appearing in a
        # later flush can be padded to line up with the galaxies
        io_helper.rows_written += self.ngalaxies

    def clear(self):
        """Empty the buffers once their rows have been written.

        The containers' structure is left in place so the lists (and their
        ids) persist into the next batch.
        """
        for name in self.GALAXY_OUTPUTS:
            clear_lists_recursive(getattr(self, name))

        # The analysis results are combined into arrays before writing, so
        # go back to an empty list for each analysis function
        self.analysis_results = {key: [] for key in self.analysis_results}

        # Reset the counters for the next batch
        self.ngalaxies = 0
        self.nbytes = 0


class PipelineIO:
    """A class for writing data to an HDF5 file.

    This class provides methods for writing data to an HDF5 file. It can
    handle writing data in parallel using MPI if the h5py library has been
    built with parallel support.

    Attributes:
        hdf (h5py.File): The HDF5 file to write to.
        io_buffers (PipelineIOBuffers): The buffers holding the outputs to
            write.
        comm (mpi.Comm): The MPI communicator.
        num_galaxies (int): The total number of galaxies.
        rank (int): The rank of the MPI process.
        is_parallel (bool): Whether the writer is running in parallel.
        is_root (bool): Whether the writer is running on the root process.
        is_collective (bool): Whether the writer is running in collective mode.
        verbose (bool): Whether to print verbose output.
        outpath (str): The path to write to (see the outpath property).
        filepath (str): The path of the file this rank writes (the outpath
            with the rank added when writing a file per rank).
        _start_time (float): The start time of the pipeline.
    """

    # Is our h5py build parallel?
    if hasattr(h5py, "get_config"):
        PARALLEL = h5py.get_config().mpi
    else:
        PARALLEL = False

    @timed("PipelineIO.__init__")
    def __init__(
        self,
        io_buffers,
        outpath=None,
        comm=None,
        start_time=None,
        verbose=1,
        parallel_io=False,
    ):
        """Initialize the HDF5Writer class.

        Args:
            io_buffers (PipelineIOBuffers): The buffers holding the outputs
                to write.
            outpath (str, optional): The path to the HDF5 file to write to.
                This can be set later (see outpath) but must be set before
                the file is created. Required if the buffers are flushed to
                file while running.
            comm (mpi.Comm, optional): The MPI communicator.
            start_time (float, optional): The start time of the pipeline, used
                for timing information.
            verbose (int, optional): How verbose the output should be. 1 will
                only print on rank 0, 2 will print on all ranks, 0 will be
                silent. Defaults to 1.
            parallel_io (bool, optional): Whether to use parallel I/O. This
                requires h5py to be built with parallel support. Defaults to
                False.

        Raises:
            InconsistentArguments:
                If the buffers are flushed to file while running but no
                outpath was given.
        """
        # A path is only mandatory if we flush during run, otherwise it can
        # be set later (e.g. by passing it to Pipeline.write)
        if io_buffers.flushing and outpath is None:
            raise exceptions.InconsistentArguments(
                "An outpath must be given to flush outputs to file while "
                "running (flush_ngalaxies or flush_memory were set)."
            )

        # Attach the buffers holding the outputs we'll write
        self.io_buffers = io_buffers

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

        # Treat a communicator of size 1 as a serial run so single-rank jobs do
        # not create synthetic per-rank filenames such as ``_0``.
        self.is_parallel = comm is not None and self.size > 1
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

        # Flag for whether the output file has been created (we can't
        # change where we write once we've started writing)
        self._file_created = False

        # Flag for whether the file has been finalised (no more per-galaxy
        # data can be written)
        self.finalised = False

        # The number of galaxy rows already flushed to file, and whether
        # we're writing per-galaxy data. A dataset first appearing after
        # a flush is padded with rows_written empty rows so its rows line up
        # with the galaxies.
        self.rows_written = 0
        self._writing_galaxy_data = False

        # Attach the output path (if we have one yet)
        self._outpath = None
        self.filepath = None
        self.outpath = outpath

    @contextmanager
    def silenced(self):
        """Silence the writer's reporting within a with block."""
        # Flushing happens between progress rows, so suppress its per-dataset
        # chatter and restore the requested verbosity even if writing fails.
        verbose = self.verbose
        self.verbose = 0
        try:
            yield
        finally:
            self.verbose = verbose

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

    @property
    def outpath(self):
        """Return the path to write to (before any per rank suffix).

        Returns:
            str:
                The path to the HDF5 file (None if not yet set).
        """
        return self._outpath

    @outpath.setter
    def outpath(self, outpath):
        """Set the path to write to.

        The path can be changed until we start writing to the file, and again
        once we've finished. When writing a file per rank, the rank is
        added to the path of the file this rank writes (filepath).

        Args:
            outpath (str):
                The path to the HDF5 file to write to.

        Raises:
            InconsistentArguments:
                If a different path is given while the file is being written
                (e.g. after outputs were flushed to it).
        """
        # h5py needs a string, but we may have been given a Path
        if outpath is not None:
            outpath = str(outpath)

        # Don't move the file part way through writing it
        if (
            self._file_created
            and not self.finalised
            and outpath != self._outpath
        ):
            raise exceptions.InconsistentArguments(
                f"Cannot write to {outpath}, outputs have already been "
                f"written to {self._outpath}."
            )
        self._outpath = outpath

        # If we are writing in parallel but not using collective I/O we
        # need to write a file per rank. Modify the file path to include
        # the rank.
        self.filepath = outpath
        if outpath is not None and self.is_parallel and not self.is_collective:
            ext = outpath.split(".")[-1]
            self.filepath = outpath.replace(f".{ext}", f"_{self.rank}.{ext}")

    @timed("PipelineIO.create_file_with_metadata")
    def create_file_with_metadata(
        self,
        ngalaxies_local,
        instruments,
        emission_model,
        include_lines=False,
    ):
        """Create the HDF5 file and write metadata to it.

        This writes useful metadata to the root group of the HDF5 file and
        outputs the instruments and emission model to the appropriate groups.
        Any earlier file at the path is replaced.

        This is a collective operation.

        Args:
            ngalaxies_local (int): The number of galaxies on this rank.
            instruments (dict): A dictionary of instrument objects.
            emission_model (dict): A dictionary of emission model objects.
            include_lines (bool): Whether to create the root-level line
                metadata group.

        Raises:
            InconsistentArguments:
                If no output path has been set.
        """
        # We have to have somewhere to write
        if self.outpath is None:
            raise exceptions.InconsistentArguments(
                "No outpath has been set to write to."
            )

        # Time how long we have to wait for everyone to get here
        start = time.perf_counter()
        if self.is_parallel:
            self.comm.Barrier()
            self._took(start, "Waiting for all ranks to get to I/O")

        # Report some useful information
        if self.is_collective:
            self._print(
                f"Writing in parallel to {self.outpath} "
                f"with {self.size} ranks, and collective I/O."
            )
        elif self.is_parallel:
            self._print(
                f"Writing to {self.outpath} "
                f"with {self.size} ranks, and a file per rank."
            )
        else:
            self._print(f"Writing to {self.outpath}.")

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

        # In MPI land the per-galaxy data needs the indices of this rank's
        # galaxies in the full set of galaxies
        if self.is_parallel:
            self.galaxy_indices = np.arange(self.start, self.end)
        else:
            self.galaxy_indices = None

        # We're starting a new file, so nothing has been written to it yet
        self._file_created = True
        self.finalised = False
        self.rows_written = 0

        start = time.perf_counter()

        # Unpack the instruments into a single instrument collection containing
        # all unique instruments
        unique_instruments = InstrumentCollection()
        for inst in instruments:
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
                # galaxies.
                inst_group = hdf.create_group("Instruments")
                model_group = hdf.create_group("EmissionModel")
                if include_lines:
                    hdf.create_group("Lines")
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

        # The other ranks only hold galaxy outputs. Start each file afresh
        # here too, including the Galaxies group: a rank with no galaxies
        # will never create it through write_data, but combine_files needs it.
        elif self.is_parallel and not self.is_collective:
            with h5py.File(self.filepath, "w") as hdf:
                hdf.create_group("Galaxies")

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

        If the dataset already exists (i.e. the Pipeline has flushed earlier
        galaxies to this file) the data is appended along the first axis.

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

        # If we have an array of strings, convert to h5py's variable length
        # strings (so later flushes can append longer strings)
        if data.dtype.kind == "U":
            data = data.astype(h5py.string_dtype())

        # Write the dataset with the appropriate units
        with h5py.File(self.filepath, "a") as hdf:
            # A previous flush may already have created this dataset, in
            # which case we just add to it
            if self._append_to_dataset(hdf, data, key, units):
                return

            # A per-galaxy dataset first appearing after a flush needs empty
            # rows for the galaxies already flushed so its rows line up with
            # the galaxies. We leave those rows unwritten, HDF5 returns its
            # fill value (zero) for them without us allocating them
            nempty = (
                self.rows_written
                if self._writing_galaxy_data and data.ndim > 0
                else 0
            )

            # Create the dataset, leaving its galaxy axis unlimited so future
            # flushes can append to it
            if data.ndim > 0:
                dset = hdf.create_dataset(
                    key,
                    shape=(nempty + data.shape[0], *data.shape[1:]),
                    dtype=data.dtype,
                    maxshape=(None, *data.shape[1:]),
                    compression="gzip",
                )
                dset[nempty:, ...] = data
            else:
                dset = hdf.create_dataset(key, data=data, compression="gzip")
            dset.attrs["Units"] = units

    def _append_to_dataset(self, hdf, data, key, units):
        """Append data to a dataset created by an earlier flush.

        Args:
            hdf (h5py.File): The open file to write to.
            data (np.ndarray): The data to append (without units).
            key (str): The key of the dataset.
            units (str): The units of the data.

        Returns:
            bool:
                True if we've handled the data, False if the dataset still
                needs creating.
        """
        # Nothing to append to
        if key not in hdf:
            return False

        # An empty batch has nothing to add
        if data.ndim > 0 and data.shape[0] == 0:
            return True

        # An empty placeholder (written by a rank or batch without any
        # galaxies) may not have the right trailing shape, so replace it
        dset = hdf[key]
        if dset.shape[0] == 0:
            del hdf[key]
            return False

        # Every batch must agree on what a row looks like
        if (
            dset.shape[1:] != data.shape[1:]
            or dset.dtype != data.dtype
            or dset.attrs["Units"] != units
        ):
            raise ValueError(
                f"Cannot append to {key}: existing dataset has "
                f"shape {dset.shape}, dtype {dset.dtype}, units "
                f"{dset.attrs['Units']}; new data has shape "
                f"{data.shape}, dtype {data.dtype}, units {units}."
            )

        # Extend the galaxy axis and fill it
        nexisting = dset.shape[0]
        dset.resize(nexisting + data.shape[0], axis=0)
        dset[nexisting:, ...] = data
        return True

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

    @timed("PipelineIO.write_data")
    def write_data(self, data, key, indexes=None, root=0):
        """Write data using the appropriate method based on the environment.

        Args:
            data (any): The data to write.
            key (str): The key to write the data to.
            indexes (list, optional): The galaxy indexes of the data. These
                only mark the data as per galaxy (appended to the rows
                already written) rather than data written once by the root
                rank, the values themselves aren't used.
            root (int, optional): The root rank for gathering and writing.
        """
        try:
            start = time.perf_counter()

            # Only per-galaxy data (which has indexes) needs padding when a
            # dataset first appears after a flush
            self._writing_galaxy_data = indexes is not None

            # In parallel land we need to make sure we're on the same page with
            # the structure we are writing. Per-galaxy data (indexes given)
            # can be flushed at different times on different ranks, so don't
            # try to synchronise here. The file combiners handle missing
            # datasets when all ranks have finished writing.
            if self.is_parallel and indexes is None:
                data = unify_dict_structure_across_ranks(data, self.comm)

            # Early exit if data is empty. Zero-row per-galaxy arrays are
            # still written so empty ranks hold the dataset with the right
            # trailing shape.
            if data is None or (
                len(data) == 0 and (isinstance(data, dict) or indexes is None)
            ):
                return

            # Root-level metadata should only be written once, even when using
            # MPI or collective I/O.
            if self.is_parallel and indexes is None:
                if self.rank != root:
                    return

                self.write_datasets_recursive(data, key)
                self._took(start, f"Writing {key} (and subgroups)")
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
            raise OSError(f"Failed to write {key} - {e}") from e

    @timed("PipelineIO.combine_rank_files")
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
                # If we found a group we need to recurse and create the group
                # in the destination file if it doesn't exist. We also need to
                # copy the attributes.
                if isinstance(v, h5py.Group):
                    # Bin axes have no galaxy dimension. Copy their group
                    # once instead of slicing each dataset by rank.
                    if v.name in _SHARED_AXIS_GROUPS:
                        if k not in dest:
                            src.file.copy(v, dest, name=k)
                        continue

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
                    # Each rank must hold exactly one row per galaxy
                    if v.shape[0] != slice.stop - slice.start:
                        raise ValueError(
                            f"Cannot combine {v.name}: {v.shape[0]} rows "
                            f"for {slice.stop - slice.start} galaxies"
                        )
                    # Empty ranks only hold zero-row placeholders, which
                    # must not decide the combined dataset's shape
                    if v.shape[0] == 0:
                        continue
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

        # Open the output file (any existing file is only replaced once the
        # new one is complete)
        with _hdf5_replaced_on_success(new_path) as hdf:
            # Loop over each rank file
            for rank in range(self.size):
                # Open the rank file
                with h5py.File(
                    temp_path.replace("<rank>", str(rank)),
                    "r",
                ) as rank_hdf:
                    # Copy root metadata once and only combine Galaxies
                    if rank == 0:
                        for attr in rank_hdf.attrs:
                            hdf.attrs[attr] = rank_hdf.attrs[attr]

                        for key, value in rank_hdf.items():
                            if key == "Galaxies":
                                hdf.create_group("Galaxies")
                                continue
                            hdf.copy(value, key)

                    _recursive_copy(
                        rank_hdf["Galaxies"],
                        hdf["Galaxies"],
                        slice=slice(starts[rank], ends[rank]),
                    )

        for rank in range(self.size):
            os.remove(temp_path.replace("<rank>", str(rank)))

        self._took(start, "Combining files")

    @timed("PipelineIO.combine_rank_files_virtual")
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

        # Remove any old virtual file. Unlike a copied file it can't be kept
        # if this combine fails, since it points at the rank files this run
        # has already overwritten
        if os.path.exists(new_path):
            os.remove(new_path)

        # Open the first rank file (rank 0 file) to discover the structure
        first_file_path = temp_path.replace("<rank>", "0")
        with h5py.File(first_file_path, "r") as f0:
            # Gather dataset information (for virtual datasets). A rank
            # without galaxies may hold empty placeholders so we take each
            # dataset's information from the first rank with rows for it
            datasets_info = {}

            def gather_datasets(group, group_path="/"):
                for k, v in group.items():
                    current_path = f"{group_path}{k}"
                    # Axes are shared metadata, not virtual galaxy rows.
                    # We'll copy them directly from rank 0 below.
                    if current_path in _SHARED_AXIS_GROUPS:
                        continue
                    if isinstance(v, h5py.Group):
                        gather_datasets(v, current_path + "/")
                    elif current_path not in datasets_info or (
                        datasets_info[current_path][0][0] == 0
                        and v.shape[0] > 0
                    ):
                        # Record dataset shape, dtype, and attributes
                        datasets_info[current_path] = (
                            v.shape,
                            v.dtype,
                            dict(v.attrs),
                        )

            # Gather group structure (to replicate in the virtual file)
            groups_info = {}

            def gather_groups(group, group_path="/"):
                for k, v in group.items():
                    current_path = f"{group_path}{k}"
                    if current_path in _SHARED_AXIS_GROUPS:
                        continue
                    if isinstance(v, h5py.Group):
                        groups_info.setdefault(current_path, dict(v.attrs))
                        gather_groups(v, current_path + "/")

            gather_datasets(f0["Galaxies"], "/Galaxies/")
            gather_groups(f0["Galaxies"], "/Galaxies/")

            # A rank without galaxies may be missing datasets entirely, so
            # also gather the structure from the other rank files
            for rank in range(1, self.size):
                src_file = temp_path.replace("<rank>", str(rank))
                with h5py.File(src_file, "r") as rank_f:
                    gather_datasets(rank_f["Galaxies"], "/Galaxies/")
                    gather_groups(rank_f["Galaxies"], "/Galaxies/")

            # Create the virtual file (any existing file is only replaced
            # once the new one is complete)
            with _hdf5_replaced_on_success(new_path) as hdf:
                for attr in f0.attrs:
                    hdf.attrs[attr] = f0.attrs[attr]

                for key, value in f0.items():
                    if key == "Galaxies":
                        continue
                    hdf.copy(value, key)

                hdf.create_group("Galaxies")

                # Create empty group structure
                for gpath, gattrs in groups_info.items():
                    grp = hdf.create_group(gpath)
                    for attr_name, attr_val in gattrs.items():
                        grp.attrs[attr_name] = attr_val

                # Preserve the axes at their usual paths without interpreting
                # their lengths as a number of galaxies. Only rank 0 writes
                # them, so it is also the source of truth here.
                for path in _SHARED_AXIS_GROUPS:
                    if path in f0:
                        hdf.require_group("Galaxies/Stars")
                        f0.copy(path, hdf, name=path)

                # Construct the virtual datasets for each dataset
                for dpath, (shape, dtype, dattrs) in datasets_info.items():
                    final_shape = (self.num_galaxies,) + shape[1:]
                    layout = h5py.VirtualLayout(shape=final_shape, dtype=dtype)

                    # Loop over each rank and map the data to the layout,
                    # keeping track of whether any rank had rows to map
                    mapped = False
                    for rank, (start_i, end_i) in enumerate(zip(starts, ends)):
                        src_file = temp_path.replace("<rank>", str(rank))
                        with h5py.File(src_file, "r") as rank_f:
                            # Nothing to map from a rank without this dataset
                            if dpath not in rank_f:
                                continue

                            src_dset = rank_f[dpath]
                            src_shape = src_dset.shape
                            src_units = src_dset.attrs.get("Units")
                            local_size = src_dset.shape[0]

                        # Nothing to map from a rank without any rows
                        if local_size == 0:
                            continue

                        # Ensure the local size is as expected
                        expected_size = end_i - start_i
                        if local_size != expected_size:
                            raise ValueError(
                                f"Cannot map {dpath}: rank {rank} has "
                                f"{local_size} rows for {expected_size} "
                                "galaxies"
                            )

                        target_shape = (end_i - start_i,) + shape[1:]

                        if (
                            len(src_shape) != len(shape)
                            or src_shape[1:] != shape[1:]
                        ):
                            raise ValueError(
                                "Cannot create virtual dataset mapping for "
                                f"{dpath!r}: rank {rank} dataset in "
                                f"{src_file!r} "
                                f"has shape {src_shape}, but other ranks have "
                                f"shape {shape}. Non-leading dimensions must "
                                "match across rank files."
                            )

                        # The units must also match, we don't convert them
                        if src_units != dattrs.get("Units"):
                            raise ValueError(
                                "Cannot create virtual dataset mapping for "
                                f"{dpath!r}: rank {rank} dataset in "
                                f"{src_file!r} has units {src_units!r}, but "
                                f"other ranks have units "
                                f"{dattrs.get('Units')!r}."
                            )

                        # Create a virtual source for this rank
                        vsource = h5py.VirtualSource(
                            src_file, dpath, shape=(local_size,) + shape[1:]
                        )

                        # Map the portion of the layout to this source slice
                        try:
                            layout[start_i:end_i, ...] = vsource[...]
                        except ValueError as exc:
                            raise ValueError(
                                "Failed to map rank dataset into virtual "
                                f"dataset for {dpath!r}. Rank {rank}, source "
                                f"file {src_file!r}, source shape "
                                f"{src_shape}, virtual slice "
                                f"[{start_i}:{end_i}] with target "
                                f"shape {target_shape}, final virtual shape "
                                f"{final_shape}. Original error: {exc}"
                            ) from exc
                        mapped = True

                    # Skip datasets without rows on any rank, otherwise we'd
                    # make a dataset full of fill values
                    if not mapped:
                        continue

                    # Create the virtual dataset in the final file
                    vds = hdf.create_virtual_dataset(dpath, layout)

                    # Copy dataset attributes
                    for attr_name, attr_val in dattrs.items():
                        vds.attrs[attr_name] = attr_val

        self._took(start, "Creating virtual file")
