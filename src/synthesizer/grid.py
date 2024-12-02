"""The Grid class containing tabulated spectral and line data.

This object underpins all of Synthesizer function which generates synthetic
spectra and line luminosities from models or simulation outputs. The Grid
object contains attributes and methods for interfacing with spectral and line
grids.

The grids themselves use a standardised HDF5 format which can be generated
using the synthesizer-grids sister package.

Example usage:

    from synthesizer import Grid

    # Load a grid
    grid = Grid("bc03_salpeter")

    # Get the axes of the grid
    print(grid.axes)

    # Get the spectra grid
    print(grid.spectra)
"""

import os

import cmasher as cmr
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from spectres import spectres
from unyt import Hz, angstrom, erg, s, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.line import Line, LineCollection, flatten_linelist
from synthesizer.sed import Sed
from synthesizer.units import Quantity, accepts
from synthesizer.utils.ascii_table import TableFormatter
from synthesizer.warnings import warn

from . import __file__ as filepath


class Grid:
    """
    The Grid class containing tabulated spectral and line data.

    This object contains attributes and methods for reading and
    manipulating the spectral grids which underpin all spectra/line
    generation in synthesizer.

    Attributes:
        grid_dir (str)
            The directory containing the grid HDF5 file.
        grid_name (str)
            The name of the grid (as defined by the file name)
            with no extension.
        grid_ext (str)
            The grid extension. Either ".hdf5" or ".h5". If the passed
            grid_name has no extension then ".hdf5" is assumed.
        grid_filename (str)
            The full path to the grid file.
        available_lines (bool/list)
            A list of lines on the Grid.
        available_spectra (bool/list)
            A list of spectra on the Grid.
        lam (Quantity, float)
            The wavelengths at which the spectra are defined.
        spectra (dict, array-like, float)
            The spectra array from the grid. This is an N-dimensional
            grid where N is the number of axes of the SPS grid. The final
            dimension is always wavelength.
        line_lams (dict, dist, float)
            A dictionary of line wavelengths.
        line_lums (dict, dict, float)
            A dictionary of line luminosities.
        line_conts (dict, dict, float)
            A dictionary of line continuum luminosities.
        parameters (dict)
            A dictionary containing the grid's parameters used in its
            generation.
        axes (list, str)
            A list of the names of the spectral grid axes.
        naxes
            The number of axes the spectral grid has.
        logQ10 (dict)
            A dictionary of ionisation Q parameters. (DEPRECATED)
        log10_specific_ionising_luminosity (dict)
            A dictionary of log10 specific ionising luminosities.
        <grid_axis> (array-like, float)
            A Grid will always contain 1D arrays corresponding to the axes
            of the spectral grid. These are read dynamically from the HDF5
            file so can be anything but usually contain at least stellar ages
            and stellar metallicity.
    """

    # Define Quantities
    lam = Quantity()

    @accepts(new_lam=angstrom)
    def __init__(
        self,
        grid_name,
        grid_dir=None,
        spectra_to_read=None,
        read_lines=True,
        new_lam=None,
        lam_lims=(),
    ):
        """
        Initialise the grid object.

        This will open the grid file and extract the axes, spectra (if
        requested), and lines (if requested) and any other relevant data.

        Args:
            grid_name (str)
                The file name of the grid (if no extension is provided then
                hdf5 is assumed).
            grid_dir (str)
                The file path to the directory containing the grid file.
            spectra_to_read (list)
                A list of spectra to read in. If None then all available
                spectra will be read. Default is None.
            read_lines (bool)
                Should we read lines? If a list then a subset of lines will be
                read.
            new_lam (array-like, float)
                An optional user defined wavelength array the spectra will be
                interpolated onto, see Grid.interp_spectra.
            lam_lims (tuple, float)
                A tuple of the lower and upper wavelength limits to truncate
                the grid to (i.e. (lower_lam, upper_lam)). If new_lam is
                provided these limits will be ignored.
        """
        # Get the grid file path data
        self.grid_dir = ""
        self.grid_name = ""
        self.grid_ext = "hdf5"  # can be updated if grid_name has an extension
        self._parse_grid_path(grid_dir, grid_name)

        # Prepare lists of available lines and spectra
        self.available_lines = []
        self.available_spectra = []

        # Set up property flags. These will be set when their property methods
        # are first called to avoid reading the file too often.
        self._reprocessed = None
        self._lines_available = None

        # Set up spectra and lines dictionaries (if we don't read them they'll
        # just stay as empty dicts)
        self.spectra = {}
        self.line_lams = {}
        self.line_lums = {}
        self.line_conts = {}

        # Set up dictionary to hold parameters used in grid generation
        self.parameters = {}

        # Get the axes of the grid from the HDF5 file
        self._get_axes()

        # Get the ionising luminosity (if available)
        self._get_ionising_luminosity()

        # We always read spectra, but can read a subset if requested
        self.lam = None
        self.available_spectra = None
        self._get_spectra_grid(spectra_to_read)

        # Prepare lines attributes
        self.available_lines = []

        # Read in lines
        if read_lines or isinstance(read_lines, list):
            self._get_lines_grid(read_lines)

        # Prepare the wavelength axis (if new_lam and lam_lims are
        # all None, this will do nothing, leaving the grid's wavelength array
        # as it is in the HDF5 file)
        self._prepare_lam_axis(new_lam, lam_lims)

    def _parse_grid_path(self, grid_dir, grid_name):
        """Parse the grid path and set the grid directory and filename."""
        # If we haven't been given a grid directory, assume the grid is in
        # the package's "data/grids" directory.
        if grid_dir is None:
            grid_dir = os.path.join(os.path.dirname(filepath), "data/grids")

        # Store the grid directory
        self.grid_dir = grid_dir

        # Have we been passed an extension?
        grid_name_split = grid_name.split(".")[-1]
        ext = grid_name_split[-1]
        if ext == "hdf5" or ext == "h5":
            self.grid_ext = ext

        # Strip the extension off the name (harmless if no extension)
        self.grid_name = grid_name.replace(f".{self.grid_ext}", "")

        # Construct the full path
        self.grid_filename = (
            f"{self.grid_dir}/{self.grid_name}.{self.grid_ext}"
        )

    @property
    def log10metallicities(self):
        """Return the log10 metallicity axis."""
        return np.log10(self.metallicity)

    @property
    def log10ages(self):
        """
        Return the log10 age axis.

        This is an alias to provide a pluralised version of the log10age
        attribute.
        """
        return self.log10age

    def _get_axes(self):
        """Get the grid axes from the HDF5 file."""
        # Get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            self.parameters = {k: v for k, v in hf.attrs.items()}

            # Get list of axes
            self.axes = list(hf.attrs["axes"])

            # Set the values of each axis as an attribute
            # e.g. self.log10age == hdf["axes"]["log10age"]
            for axis in self.axes:
                setattr(self, axis, hf["axes"][axis][:])

            # Number of axes
            self.naxes = len(self.axes)

    def _get_ionising_luminosity(self):
        """Get the ionising luminosity from the HDF5 file."""
        # Get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            # Extract any ionising luminosities
            if "log10_specific_ionising_luminosity" in hf.keys():
                self.log10_specific_ionising_lum = {}
                for ion in hf["log10_specific_ionising_luminosity"].keys():
                    self.log10_specific_ionising_lum[ion] = hf[
                        "log10_specific_ionising_luminosity"
                    ][ion][:]

            # Old name for backwards compatibility (DEPRECATED)
            if "log10Q" in hf.keys():
                self.log10Q = {}
                for ion in hf["log10Q"].keys():
                    self.log10Q[ion] = hf["log10Q"][ion][:]

    def _get_spectra_grid(self, spectra_to_read):
        """
        Get the spectra grid from the HDF5 file.

        If using a cloudy reprocessed grid this method will automatically
        calculate 2 spectra not native to the grid file:
            total = transmitted + nebular
            nebular_continuum = nebular - linecont

        Args:
            spectra_to_read (list)
                A list of spectra to read in. If None then all available
                spectra will be read.
        """
        with h5py.File(self.grid_filename, "r") as hf:
            # Are we reading everything?
            if spectra_to_read is None:
                self.available_spectra = self._get_spectra_ids_from_file()
            elif isinstance(spectra_to_read, list):
                all_spectra = self._get_spectra_ids_from_file()
                self.available_spectra = spectra_to_read

                # Check the requested spectra are available
                missing_spectra = set(spectra_to_read) - set(all_spectra)
                if len(missing_spectra) > 0:
                    raise exceptions.MissingSpectraType(
                        f"The following requested spectra are not available"
                        "in the supplied grid file: "
                        f"{missing_spectra}"
                    )
            else:
                raise exceptions.InconsistentArguments(
                    "spectra_to_read must either be None or a list "
                    "containing a subset of spectra to read."
                )

            # Read the wavelengths
            self.lam = hf["spectra/wavelength"][:]

            # Get all our spectra
            for spectra_id in self.available_spectra:
                self.spectra[spectra_id] = hf["spectra"][spectra_id][:]

            # # THIS IS A HACK BECAUSE SW CHANGED THE AGN EMISSIONS WITHOUT
            # # UPDATING THE TEST GRIDS AND ONLY HE HAS ACCESS TO THE DROPBOX.
            # # THIS WILL BE REMOVED WHEN THE GRIDS ARE UPDATED.
            # if "nlr" in self.grid_filename or "blr" in self.grid_filename:
            #     for spectra_id in self.available_spectra:
            #         # Normalise spectra by bolometric luminosity
            #         sed = Sed(
            #             self.lam, lnu=self.spectra[spectra_id] * erg / s / Hz
            #         )
            #         self.spectra[spectra_id] = (
            #             self.spectra[spectra_id]
            #             / sed.bolometric_luminosity[..., None]
            #         )

        # If a full cloudy grid is available calculate some
        # other spectra for convenience.
        if self.reprocessed:
            # The total emission (ignoring any dust reprocessing) is just
            # the transmitted plus the nebular
            self.spectra["total"] = (
                self.spectra["transmitted"] + self.spectra["nebular"]
            )
            self.available_spectra.append("total")

            # The nebular continuum is the nebular emission with the line
            # contribution removed
            self.spectra["nebular_continuum"] = (
                self.spectra["nebular"] - self.spectra["linecont"]
            )
            self.available_spectra.append("nebular_continuum")

    def _get_lines_grid(self, read_lines):
        """
        Get the lines grid from the HDF5 file.

        Args:
            read_lines (bool/list)
                Flag for whether to read all available lines or subset of
                lines to read.
        """
        # Double check we actually have lines to read
        if not self.lines_available:
            if not self.reprocessed:
                raise exceptions.GridError(
                    "Grid hasn't been reprocessed with cloudy and has no "
                    "lines. Either pass `read_lines=False` or load a grid "
                    "which has been run through cloudy."
                )

            else:
                raise exceptions.GridError(
                    (
                        "No lines available on this grid object. "
                        "Either set `read_lines=False`, or load a grid "
                        "containing line information"
                    )
                )

        with h5py.File(self.grid_filename, "r") as hf:
            # Are we only reading a subset?
            if isinstance(read_lines, list):
                self.available_lines = flatten_linelist(read_lines)
            else:
                # If not, read all available lines
                self.available_lines, _ = self._get_line_ids_from_file()

            # Create an entry for each spectra type
            for spectra in self.available_spectra:
                self.line_lums[spectra] = {}
                self.line_conts[spectra] = {}

            # We read the lines themselves into "nebular" and "linecont"
            # so make sure this exists
            if (
                "nebular" not in self.line_lums
                or "linecont" not in self.line_lums
            ):
                raise exceptions.GridError(
                    "No nebular or linecont spectra found. "
                    "The nebular and linecont spectra is"
                    " required for storing lines. Either you have explictly"
                    " requested a subset of spectra without these, or "
                    "something is wrong with the grid file."
                )

            # Read the line wavelengths
            lines_outside_lam = []
            for line in self.available_lines:
                self.line_lams[line] = hf["lines"][line].attrs["wavelength"]

                # Ensure this wavelength is within the wavelength array of the
                # grid
                if (
                    self.line_lams[line] < self.lam[0]
                    or self.line_lams[line] > self.lam[-1]
                ):
                    lines_outside_lam.append(line)

            # If we have lines outside the wavelength range of the grid
            # warn the user
            if len(lines_outside_lam) > 0:
                warn(
                    "The following lines are outside the wavelength "
                    f"range of the grid: {lines_outside_lam}"
                )

            # Read the lines into the nebular and linecont entries. The
            # continuum for "linecont" is by definition 0 (we'll do all
            # other continua below)
            for line in self.available_lines:
                self.line_lums["nebular"][line] = hf["lines"][line][
                    "luminosity"
                ][:]
                self.line_lums["linecont"][line] = hf["lines"][line][
                    "luminosity"
                ][:]
                self.line_conts["linecont"][line] = np.zeros(
                    self.spectra["linecont"].shape[:-1]
                )

            # Now that we have read the line data itself we need to populate
            # the other spectra entries. the line luminosities for all entries
            # other than nebular are 0 but the continuums need to be sampled
            # from the spectra.
            for spectra in self.available_spectra:
                # Define the extraction key
                extract = spectra

                # For nebular the extraction key is "nebular_continuum",
                # otherwise we would extract the line luminosity along with
                # the continuum
                if spectra == "nebular":
                    extract = "nebular_continuum"

                # Create an interpolation function for this spectra
                interp_func = interp1d(
                    self._lam,
                    self.spectra[extract],
                    axis=-1,
                    kind="linear",
                    fill_value=0.0,
                    bounds_error=False,
                )

                # Get the continuum luminosities by interpolating the to get
                # the continuum at the line wavelengths (we use scipy here
                # because spectres can't handle single value interpolation).
                # The linecont continuum is explicitly 0 and set above.
                if spectra != "linecont":
                    self.line_conts[spectra] = {
                        line: interp_func(self.line_lams[line])
                        for line in self.available_lines
                    }

                # Set the line luminosities to 0 as long as they haven't
                # already been set
                if spectra != "nebular" and spectra != "linecont":
                    self.line_lums[spectra] = {
                        line: np.zeros(self.spectra[spectra].shape[:-1])
                        for line in self.available_lines
                    }

    def _prepare_lam_axis(
        self,
        new_lam,
        lam_lims,
    ):
        """
        Modify the grid wavelength axis to adhere to user defined wavelengths.

        This method will do nothing if the user has not provided new_lam
        or lam_lims.

        If the user has passed any of these the wavelength array will be
        limited and/or interpolated to match the user's input.

        - If new_lam is provided, the spectra will be interpolated onto this
          array.
        - If lam_lims are provided, the grid will be truncated to these
          limits.

        Args:
            new_lam (array-like, float)
                An optional user defined wavelength array the spectra will be
                interpolated onto.
            lam_lims (tuple, float)
                A tuple of the lower and upper wavelength limits to truncate
                the grid to (i.e. (lower_lam, upper_lam)).
        """
        # If we have both new_lam and wavelength limits
        # the limits become meaningless tell the user so.
        if len(lam_lims) > 0 and (new_lam is not None):
            warn(
                "Passing new_lam and lam_lims is contradictory, "
                "lam_lims will be ignored."
            )

        # Has a new wavelength grid been passed to interpolate
        # the spectra onto?
        if new_lam is not None:
            # Interpolate the spectra grid
            self.interp_spectra(new_lam)

        # If we have been given wavelength limtis truncate the grid
        if len(lam_lims) > 0 and new_lam is None:
            self.truncate_grid_lam(*lam_lims)

    @property
    def reprocessed(self):
        """
        Flag for whether grid has been reprocessed through cloudy.

        This will only access the file the first time this property is
        accessed.

        Returns:
            True if reprocessed, False otherwise.
        """
        if self._reprocessed is None:
            with h5py.File(self.grid_filename, "r") as hf:
                self._reprocessed = (
                    True if "cloudy_version" in hf.attrs.keys() else False
                )

        return self._reprocessed

    @property
    def lines_available(self):
        """
        Flag for whether line emission exists.

        This will only access the file the first time this property is
        accessed.

        Returns:
            bool:
                True if lines are available, False otherwise.
        """
        if self._lines_available is None:
            with h5py.File(self.grid_filename, "r") as hf:
                self._lines_available = True if "lines" in hf.keys() else False

        return self._lines_available

    @property
    def has_spectra(self):
        """Return whether the Grid has spectra."""
        return len(self.spectra) > 0

    @property
    def has_lines(self):
        """Return whether the Grid has lines."""
        return len(self.line_lums) > 0

    def _get_spectra_ids_from_file(self):
        """
        Get a list of the spectra available in a grid file.

        Returns:
            list:
                List of available spectra
        """
        with h5py.File(self.grid_filename, "r") as hf:
            spectra_keys = list(hf["spectra"].keys())

        # Clean up the available spectra list
        spectra_keys.remove("wavelength")

        # Remove normalisation dataset
        if "normalisation" in spectra_keys:
            spectra_keys.remove("normalisation")

        return spectra_keys

    def _get_line_ids_from_file(self):
        """
        Get a list of the lines available on a grid.

        Returns:
            list:
                List of available lines
                List of associated wavelengths.
        """
        with h5py.File(self.grid_filename, "r") as hf:
            lines = list(hf["lines"].keys())

            wavelengths = np.array(
                [hf["lines"][line].attrs["wavelength"] for line in lines]
            )
        return lines, wavelengths

    @accepts(new_lam=angstrom)
    def interp_spectra(self, new_lam, loop_grid=False):
        """
        Interpolates the spectra grid onto the provided wavelength grid.

        NOTE: this will overwrite self.lam and self.spectra, overwriting
        the attributes loaded from the grid file. To get these back a new grid
        will need to instantiated with no lam argument passed.

        Args:
            new_lam (unyt_array/array-like, float)
                The new wavelength array to interpolate the spectra onto.
            loop_grid (bool)
                flag for whether to do the interpolation over the whole
                grid, or loop over the first axes. The latter is less memory
                intensive, but slower. Defaults to False.
        """
        # Loop over spectra to interpolate
        for spectra_type in self.available_spectra:
            # Are we doing the look up in one go, or looping?
            if loop_grid:
                new_spectra = [None] * len(self.spectra[spectra_type])

                # Loop over first axis of spectra array
                for i, _spec in enumerate(self.spectra[spectra_type]):
                    new_spectra[i] = spectres(
                        new_lam.value,
                        self._lam,
                        _spec,
                        fill=0,
                    )

                del self.spectra[spectra_type]
                new_spectra = np.asarray(new_spectra)
            else:
                # Evaluate the function at the desired wavelengths
                new_spectra = spectres(
                    new_lam.value,
                    self._lam,
                    self.spectra[spectra_type],
                    fill=0,
                )

            # Update this spectra
            self.spectra[spectra_type] = new_spectra

        # Update wavelength array
        self.lam = new_lam

    def __str__(self):
        """
        Return a string representation of the particle object.

        Returns:
            table (str)
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Grid")

    @property
    def shape(self):
        """Return the shape of the grid."""
        return self.spectra[self.available_spectra[0]].shape

    @staticmethod
    def get_nearest_index(value, array):
        """
        Calculate the closest index in an array for a given value.

        TODO: What is this doing here!?

        Args:
            value (float/unyt_quantity)
                The target value.

            array (np.ndarray/unyt_array)
                The array to search.

        Returns:
            int
                The index of the closet point in the grid (array)
        """
        # Handle units on both value and array
        # First do we need a conversion?
        if isinstance(array, unyt_array) and isinstance(value, unyt_quantity):
            if array.units != value.units:
                value = value.to(array.units)

        # Get the values
        if isinstance(array, unyt_array):
            array = array.value
        if isinstance(value, unyt_quantity):
            value = value.value

        return (np.abs(array - value)).argmin()

    def get_grid_point(self, **kwargs):
        """
        Identify the nearest grid point for a tuple of values.

        Args:
            **kwargs (dict)
                Pairs of axis names and values for the desired grid point,
                e.g. log10ages=9.3, log10metallicities=-2.1.

        Returns:
            tuple
                A tuple of integers specifying the closest grid point.
        """
        return tuple(
            [
                self.get_nearest_index(value, getattr(self, axis))
                for axis, value in kwargs.items()
            ]
        )

    def get_spectra(self, grid_point, spectra_id="incident"):
        """
        Create an Sed object for a specific grid point.

        Args:
            grid_point (tuple)
                A tuple of integers specifying the closest grid point.
            spectra_id (str)
                The name of the spectra (in the grid) that is desired.

        Returns:
            synthesizer.sed.Sed
                A synthesizer Sed object
        """
        # Throw exception if the spectra_id not in list of available spectra
        if spectra_id not in self.available_spectra:
            raise exceptions.InconsistentParameter(
                "Provided spectra_id is not in the list of available spectra."
            )

        # Throw exception if the grid_point has a different shape from the grid
        if len(grid_point) != self.naxes:
            raise exceptions.InconsistentParameter(
                "The grid_point tuple provided"
                "as an argument should have the same shape as the grid."
            )

        # Throw an exception if grid point is outside grid bounds
        try:
            return Sed(
                self.lam,
                lnu=self.spectra[spectra_id][grid_point] * erg / s / Hz,
            )
        except IndexError:
            # Modify the error message for clarity
            raise IndexError(
                f"grid_point is outside of the grid (grid.shape={self.shape}, "
                f"grid_point={grid_point})"
            )

    def get_line(self, grid_point, line_id, spectra_type="nebular"):
        """
        Create a Line object for a given line_id and grid_point.

        Args:
            grid_point (tuple)
                A tuple of integers specifying the closest grid point.
            line_id (str)
                The id of the line.
            spectra_type (str)
                The spectra type to extract the line from. Default is
                "nebular", all other spectra will have line luminosities of 0
                by definition.

        Returns:
            line (synthesizer.line.Line)
                A synthesizer Line object.
        """
        # Throw exception if the grid_point has a different shape from the grid
        if len(grid_point) != self.naxes:
            raise exceptions.InconsistentParameter(
                "The grid_point tuple provided"
                "as an argument should have the same shape as the grid."
            )

        if isinstance(line_id, str):
            line_id = [line_id]

        # Set up a list to hold all lines
        lines = []

        for line_id_ in line_id:
            # Throw exception if tline_id not in list of available lines
            if line_id_ not in self.available_lines:
                raise exceptions.InconsistentParameter(
                    f"Provided line_id ({line_id_}) is not in the list "
                    "of available lines."
                )

            # Create the line
            # TODO: Need to use units from the grid file but they currently
            # aren't stored correctly.
            lam = self.line_lams[line_id_] * angstrom
            lum = self.line_lums[spectra_type][line_id_][grid_point] * erg / s
            cont = (
                self.line_conts[spectra_type][line_id_][grid_point]
                * erg
                / s
                / Hz
            )
            lines.append(
                Line(
                    line_id=line_id,
                    wavelength=lam,
                    luminosity=lum,
                    continuum=cont,
                )
            )

        return Line(combine_lines=lines)

    def get_lines(self, grid_point, line_ids=None):
        """
        Create a LineCollection for multiple lines.

        Args:
            grid_point (tuple)
                A tuple of the grid point indices.
            line_ids (list)
                A list of lines, if None use all available lines.

        Returns:
            lines (lines.LineCollection)
        """
        # If no line ids are provided calculate all lines
        if line_ids is None:
            line_ids = self.available_lines

        # Line dictionary
        lines = {}

        for line_id in line_ids:
            line = self.get_line(grid_point, line_id)

            # Add to dictionary
            lines[line.id] = line

        # Create and return collection
        return LineCollection(lines)

    def plot_specific_ionising_lum(
        self,
        ion="HI",
        hsize=3.5,
        vsize=None,
        cmap=cmr.sapphire,
        vmin=None,
        vmax=None,
        max_log10age=None,
    ):
        """
        Make a simple plot of the specific ionising photon luminosity.

        The resulting figure will show the (log) specific ionsing photon
        luminosity as a function of (log) age and metallicity for a given grid
        and ion.

        Args:
           ion (str)
                The desired ion. Most grids only have HI and HII calculated by
                default.

            hsize (float)
                The horizontal size of the figure

            vsize (float)
                The vertical size of the figure

            cmap (object/str)
                A colourmap object or string defining the matplotlib colormap.

            vmin (float)
                The minimum specific ionising luminosity used in the colourmap

            vmax (float)
                The maximum specific ionising luminosity used in the colourmap

            max_log10age (float)
                The maximum log10(age) to plot

        Returns:
            matplotlib.Figure
                The created figure containing the axes.
            matplotlib.Axis
                The axis on which to plot.
        """
        # Define the axis coordinates
        left = 0.2
        height = 0.65
        bottom = 0.15
        width = 0.75

        # Scale the plot height if necessary
        if vsize is None:
            vsize = hsize * width / height

        # Create the figure
        fig = plt.figure(figsize=(hsize, vsize))

        # Create the axes
        ax = fig.add_axes((left, bottom, width, height))
        cax = fig.add_axes([left, bottom + height + 0.01, width, 0.05])

        # Create an index array
        y = np.arange(len(self.metallicity))

        # Select grid for specific ion
        if hasattr(self, "log10_specific_ionising_lum"):
            log10_specific_ionising_lum = self.log10_specific_ionising_lum[ion]
        else:
            log10_specific_ionising_lum = self.log10Q[ion]

        # Truncate grid if max age provided
        if max_log10age is not None:
            ia_max = self.get_nearest_index(max_log10age, self.log10age)
            log10_specific_ionising_lum = log10_specific_ionising_lum[
                :ia_max, :
            ]
        else:
            ia_max = -1

        # If no limits are supplied set a sensible range for HI ion otherwise
        # use min max
        if ion == "HI":
            if vmin is None:
                vmin = 42.5
            if vmax is None:
                vmax = 47.5
        else:
            if vmin is None:
                vmin = np.min(log10_specific_ionising_lum)
            if vmax is None:
                vmax = np.max(log10_specific_ionising_lum)

        # Plot the grid of log10_specific_ionising_lum
        ax.imshow(
            log10_specific_ionising_lum.T,
            origin="lower",
            extent=[
                self.log10age[0],
                self.log10age[ia_max],
                y[0] - 0.5,
                y[-1] + 0.5,
            ],
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # Define the normalisation for the colorbar
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        cmapper.set_array([])

        # Add colourbar
        fig.colorbar(cmapper, cax=cax, orientation="horizontal")
        cax.xaxis.tick_top()
        cax.xaxis.set_label_position("top")
        cax.set_xlabel(
            r"$\rm log_{10}(\dot{n}_{" + ion + "}/s^{-1}\\ M_{\\odot}^{-1})$"
        )
        cax.set_yticks([])

        # Set custom tick marks
        ax.set_yticks(y, self.metallicity)
        ax.minorticks_off()

        # Set labels
        ax.set_xlabel("$\\log_{10}(\\mathrm{age}/\\mathrm{yr})$")
        ax.set_ylabel("$Z$")

        return fig, ax

    def get_delta_lambda(self, spectra_id="incident"):
        """
        Calculate the delta lambda for the given spectra.

        Args:
            spectra_id (str)
                Identifier for the spectra (default is "incident").

        Returns:
            tuple
                A tuple containing the list of wavelengths and delta lambda.
        """
        # Calculate delta lambda for each wavelength
        delta_lambda = np.log10(self.lam[1:]) - np.log10(self.lam[:-1])

        # Return tuple of wavelengths and delta lambda
        return self.lam, delta_lambda

    def get_sed(self, spectra_type):
        """
        Get the spectra grid as an Sed object.

        This enables grid wide use of Sed methods for flux, photometry,
        indices, ionising photons, etc.

        Args:
            spectra_type (string)
                The key of the spectra grid to extract as an Sed object.

        Returns:
            Sed
                The spectra grid as an Sed object.
        """
        return Sed(self.lam, self.spectra[spectra_type] * erg / s / Hz)

    @accepts(min_lam=angstrom, max_lam=angstrom)
    def truncate_grid_lam(self, min_lam, max_lam):
        """
        Truncate the grid to a specific wavelength range.

        If out of range wavlengths are requested, the grid will be
        truncated to the nearest wavelength within the grid.

        Args:
            min_lam (unyt_quantity)
                The minimum wavelength to truncate the grid to.

            max_lam (unyt_quantity)
                The maximum wavelength to truncate the grid to.
        """
        # Get the indices of the wavelengths to keep
        okinds = np.logical_and(self.lam >= min_lam, self.lam <= max_lam)

        # Apply the mask to the grid wavelengths
        self.lam = self.lam[okinds]

        # Apply the mask to the spectra
        for spectra_type in self.available_spectra:
            self.spectra[spectra_type] = self.spectra[spectra_type][
                ..., okinds
            ]

    def animate_grid(
        self,
        show=False,
        save_path=None,
        fps=30,
        spectra_type="incident",
    ):
        """
        Create an animation of the grid stepping through wavelength.

        Each frame of the animation is a wavelength bin.

        Args:
            show (bool):
                Should the animation be shown?
            save_path (str, optional):
                Path to save the animation. If not specified, the
                animation is not saved.
            fps (int, optional):
                the number of frames per second in the output animation.
                Default is 30 frames per second.

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        # Create the figure and axes
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))

        # Get the spectra grid
        spectra = self.spectra[spectra_type]

        # Get the normalisation
        vmin = 10**10
        vmax = np.percentile(spectra, 99.9)

        # Define the norm
        norm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

        # Create a placeholder image
        img = ax.imshow(
            spectra[:, :, 0],
            extent=[
                self.log10age.min(),
                self.log10age.max(),
                self.metallicity.min(),
                self.metallicity.max(),
            ],
            origin="lower",
            animated=True,
            norm=norm,
            aspect="auto",
        )

        cbar = fig.colorbar(img)
        cbar.set_label(
            r"$L_{\nu}/[\mathrm{erg s}^{-1}\mathrm{ Hz}^{-1} "
            r"\mathrm{ M_\odot}^{-1}]$"
        )

        ax.set_title(f"Wavelength: {self.lam[0]:.2f}{self.lam.units}")
        ax.set_xlabel("$\\log_{10}(\\mathrm{age}/\\mathrm{yr})$")
        ax.set_ylabel("$Z$")

        def update(i):
            # Update the image for the ith frame
            img.set_data(spectra[:, :, i])
            ax.set_title(f"Wavelength: {self.lam[i]:.2f}{self.lam.units}")
            return [
                img,
            ]

        # Calculate interval in milliseconds based on fps
        interval = 1000 / fps

        # Create the animation
        anim = FuncAnimation(
            fig, update, frames=self.lam.size, interval=interval, blit=False
        )

        # Save if a path is provided
        if save_path is not None:
            anim.save(save_path, writer="imagemagick")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return anim


class Template:
    """
    A simplified grid contain only a single template spectra.

    The model is different to all other emission models in that it scales a
    template by bolometric luminosity.

    Attributes:
        sed (Sed)
            The template spectra for the AGN.
        normalisation (unyt_quantity)
            The normalisation for the spectra. In reality this is the
            bolometric luminosity.
    """

    # Define Quantities
    lam = Quantity()
    lnu = Quantity()

    @accepts(lam=angstrom, lnu=erg / s / Hz)
    def __init__(
        self,
        lam,
        lnu,
        fesc=0.0,
        unify_with_grid=None,
        **kwargs,
    ):
        """
        Initialise the Template.

        Args:
            lam (array)
                Wavelength array.
            lnu (array)
                Luminosity array.
            fesc (float)
                The escape fraction of the AGN.
            unify_with_grid (Grid)
                A grid object to unify the template with. This will ensure
                the template has the same wavelength array as the grid.
            **kwargs

        """
        # It's convenient to have an sed object for the next steps
        sed = Sed(lam, lnu)

        # Before we do anything, do we have a grid we need to unify with?
        if unify_with_grid is not None:
            # Interpolate the template Sed onto the grid wavelength array
            sed = sed.get_resampled_sed(new_lam=unify_with_grid.lam)

        # Attach the template now we've done the interpolation (if needed)
        self.lnu = sed.lnu
        self.lam = sed.lam

        # Normalise, just in case
        self.normalisation = sed._bolometric_luminosity
        self.lnu /= self.normalisation

        # Set the escape fraction
        self.fesc = fesc

    @accepts(bolometric_luminosity=erg / s)
    def get_spectra(self, bolometric_luminosity):
        """
        Calculate the blackhole spectra by scaling the template.

        Args:
            bolometric_luminosity (float)
                The bolometric luminosity of the blackhole(s) for scaling.

        """
        # Ensure we have units for safety
        if bolometric_luminosity is not None and not isinstance(
            bolometric_luminosity, unyt_array
        ):
            raise exceptions.MissingUnits(
                "bolometric luminosity must be provided with units"
            )

        # Compute the scaling based on normalisation
        scaling = bolometric_luminosity.value

        # Handle the dimensions of the bolometric luminosity
        if bolometric_luminosity.shape[0] == 1:
            sed = Sed(
                self.lam,
                scaling * self.lnu * (1 - self.fesc),
            )
        else:
            sed = Sed(
                self.lam,
                scaling[:, None] * self.lnu * (1 - self.fesc),
            )

        return sed
