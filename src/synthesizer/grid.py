"""The Grid class containing tabulated spectral and line data.

This object underpins all of Synthesizer function which generates synthetic
spectra and line luminosities from models or simulation outputs. The Grid
object contains attributes and methods for interfacing with spectral and line
grids.

The grids themselves use a standardised HDF5 format which can be generated
using the grid-generation sister package.

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
from synthesizer.emissions import LineCollection, Sed
from synthesizer.synth_warnings import warn
from synthesizer.units import Quantity, accepts
from synthesizer.utils import depluralize, pluralize
from synthesizer.utils.ascii_table import TableFormatter

from . import __file__ as filepath


class Grid:
    """The Grid class containing tabulated spectral and line data.

    This object contains attributes and methods for reading and
    manipulating the spectral grids which underpin all spectra/line
    generation in synthesizer.

    Attributes:
        grid_dir (str):
            The directory containing the grid HDF5 file.
        grid_name (str):
            The name of the grid (as defined by the file name)
            with no extension.
        grid_ext (str):
            The grid extension. Either ".hdf5" or ".h5". If the passed
            grid_name has no extension then ".hdf5" is assumed.
        grid_filename (str):
            The full path to the grid file.
        available_lines (bool/list)
            A list of lines on the Grid.
        available_spectra (bool/list)
            A list of spectra on the Grid.
        lam (Quantity, float)
            The wavelengths at which the spectra are defined.
        spectra (dict, np.ndarray of float)
            The spectra array from the grid. This is an N-dimensional
            grid where N is the number of axes of the SPS grid. The final
            dimension is always wavelength.
        line_lams (dict, dist, float)
            A dictionary of line wavelengths.
        line_lums (dict, dict, float)
            A dictionary of line luminosities.
        line_conts (dict, dict, float)
            A dictionary of line continuum luminosities.
        parameters (dict):
            A dictionary containing the grid's parameters used in its
            generation.
        axes (list, str)
            A list of the names of the spectral grid axes.
        naxes
            The number of axes the spectral grid has.
        logQ10 (dict):
            A dictionary of ionisation Q parameters. (DEPRECATED)
        log10_specific_ionising_luminosity (dict):
            A dictionary of log10 specific ionising luminosities.
        <grid_axis> (np.ndarray of float):
            A Grid will always contain 1D arrays corresponding to the axes
            of the spectral grid. These are read dynamically from the HDF5
            file so can be anything but usually contain at least stellar ages
            and stellar metallicity.
    """

    # Define Quantities
    lam = Quantity("wavelength")
    line_lams = Quantity("wavelength")

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
        """Initialise the grid object.

        This will open the grid file and extract the axes, spectra (if
        requested), and lines (if requested) and any other relevant data.

        Args:
            grid_name (str):
                The file name of the grid (if no extension is provided then
                hdf5 is assumed).
            grid_dir (str):
                The file path to the directory containing the grid file.
            spectra_to_read (list):
                A list of spectra to read in. If None then all available
                spectra will be read. Default is None.
            read_lines (bool):
                Should we read lines? If a list then a subset of lines will be
                read.
            new_lam (np.ndarray of float):
                An optional user defined wavelength array the spectra will be
                interpolated onto, see Grid.interp_spectra.
            lam_lims (tuple, float):
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
        self.line_lams = None
        self.line_lums = {}
        self.line_conts = {}

        # Get the axes of the grid from the HDF5 file
        self.axes = []  # axes names
        self._axes_values = {}
        self._axes_units = {}
        self._extract_axes = []
        self._extract_axes_values = {}
        self._get_axes()

        # Read in the metadata
        self._weight_var = None
        self._model_metadata = {}
        self._get_grid_metadata()

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

    def _get_grid_metadata(self):
        """Unpack the grids metadata into the Grid."""
        # Open the file
        with h5py.File(self.grid_filename, "r") as hf:
            # What component variable do we need to weight by for the
            # emission in the grid?
            self._weight_var = hf.attrs.get("WeightVariable")

            # Loop over the Model metadata stored in the Model group
            # and store it in the Grid object
            if "Model" in hf:
                for key, value in hf["Model"].attrs.items():
                    self._model_metadata[key] = value

            # Attach all the root level attribtues to the grid object
            for k, v in hf.attrs.items():
                # Skip the axes attribute as we've already read that
                if k == "axes" or k == "WeightVariable":
                    continue
                setattr(self, k, v)

    def __getattr__(self, name):
        """Return an attribute handling arbitrary axis names.

        This method allows for the dynamic extraction of axes with units,
        either logged or not or using singular or plural axis names (to handle
        legacy naming conventions).
        """
        # First up, do we just have the attribute and it isn't an axis?
        if name in self.__dict__:
            return self.__dict__[name]

        # Now, do some silly pluralisation checks to handle old naming
        # conventions. We do this now so everything works, we can grumble
        # about it later
        plural_name = pluralize(name)
        singular_name = depluralize(name)

        # Another old convention was allowing for logged axes to be stored in
        # the grid file (this is no longer allowed)
        if name[0] != "_":
            log_name = f"log10{name}"
            log_plural_name = f"log10{plural_name}"
            log_singular_name = f"log10{singular_name}"
        else:
            log_name = f"_log10{name[1:]}"
            log_plural_name = f"_log10{plural_name[1:]}"
            log_singular_name = f"_log10{singular_name[1:]}"

        # If we have the axis name, return the axis with units (handling all
        # the silly pluralisation and logging conventions)
        if name in self.axes:
            return unyt_array(self._axes_values[name], self._axes_units[name])
        elif plural_name in self.axes:
            return unyt_array(
                self._axes_values[plural_name], self._axes_units[plural_name]
            )
        elif singular_name in self.axes:
            warn(
                "The use of singular axis names is deprecated. Update "
                "your grid file."
            )
            return unyt_array(
                self._axes_values[singular_name],
                self._axes_units[singular_name],
            )
        elif log_name in self.axes:
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return unyt_array(
                10 ** self._axes_values[log_name], self._axes_units[log_name]
            )
        elif log_plural_name in self.axes:
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return unyt_array(
                10 ** self._axes_values[log_plural_name],
                self._axes_units[log_plural_name],
            )
        elif log_singular_name in self.axes:
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return unyt_array(
                10 ** self._axes_values[log_singular_name],
                self._axes_units[log_singular_name],
            )

        # It might be a Quantity style unitless request? (handling all
        # the silly pluralisation and logging conventions)
        elif name[1:] in self.axes:
            return self._axes_values[name[1:]]
        elif plural_name[1:] in self.axes:
            return self._axes_values[plural_name[1:]]
        elif singular_name[1:] in self.axes:
            warn(
                "The use of singular axis names is deprecated. Update "
                "your grid file."
            )
            return self._axes_values[singular_name[1:]]
        elif log_name[1:] in self.axes:
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return 10 ** self._axes_values[log_name[1:]]
        elif log_plural_name[1:] in self.axes:
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return 10 ** self._axes_values[log_plural_name[1:]]
        elif log_singular_name[1:] in self.axes:
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return 10 ** self._axes_values[log_singular_name[1:]]

        # Are we doing a log10 request? (handling all the silly pluralisation)
        elif name[:5] == "log10" and name[5:] in self.axes:
            return np.log10(self._axes_values[name[5:]])
        elif plural_name[:5] == "log10" and plural_name[5:] in self.axes:
            return np.log10(self._axes_values[plural_name[5:]])
        elif singular_name[:5] == "log10" and singular_name[5:] in self.axes:
            warn(
                "The use of singular axis names is deprecated. Update "
                "your grid file."
            )
            return np.log10(self._axes_values[singular_name[5:]])
        elif log_name[:5] == "log10" and log_name[5:] in self.axes:
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return self._axes_values[log_name[5:]]
        elif (
            log_plural_name[:5] == "log10" and log_plural_name[5:] in self.axes
        ):
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return self._axes_values[log_plural_name[5:]]
        elif (
            log_singular_name[:5] == "log10"
            and log_singular_name[5:] in self.axes
        ):
            warn(
                "The use of logged axis names is deprecated. Update "
                "your grid file."
            )
            return self._axes_values[log_singular_name[5:]]

        # If we get here, we don't have the attribute
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{name}'"
        )

    def _get_axes(self):
        """Get the grid axes from the HDF5 file."""
        # Get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            # Get list of axes
            axes = list(hf.attrs["axes"])

            # Set the values of each axis as an attribute
            # e.g. self.log10age == hdf["axes"]["log10age"]
            for axis in axes:
                # What are the units of this axis?
                axis_units = hf["axes"][axis].attrs.get("Units")
                log_axis = hf["axes"][axis].attrs.get("log_on_read")

                if "log10" in axis:
                    raise exceptions.GridError(
                        "Logged axes are no longer supported because "
                        "of ambiguous units. Please update your grid file."
                    )

                # Get the values
                values = hf["axes"][axis][:]

                # Set all the axis attributes as is (without accounting
                # for any log10 conversions needed for extraction)
                self.axes.append(axis)
                self._axes_values[axis] = values
                self._axes_units[axis] = axis_units

                # Now we handle the extractions
                if log_axis:
                    self._extract_axes.append(f"log10{axis}")
                    self._extract_axes_values[f"log10{axis}"] = np.log10(
                        values
                    )
                else:
                    self._extract_axes.append(axis)
                    self._extract_axes_values[axis] = values

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
                self.log10_specific_ionising_lum = {}
                for ion in hf["log10Q"].keys():
                    self.log10_specific_ionising_lum[ion] = hf["log10Q"][ion][
                        :
                    ]

    def _get_spectra_grid(self, spectra_to_read):
        """Get the spectra grid from the HDF5 file.

        If using a cloudy reprocessed grid this method will automatically
        calculate 2 spectra not native to the grid file:
            total = transmitted + nebular
            nebular_continuum = nebular - linecont

        Args:
            spectra_to_read (list):
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
        """Get the lines grid from the HDF5 file.

        Args:
            read_lines (bool/list):
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
            self.available_lines = np.array(
                [id.decode("utf-8") for id in hf["lines"]["id"][:]]
            )

            # Read the line wavelengths
            lams = hf["lines"]["wavelength"][...]
            lam_units = hf["lines"]["wavelength"].attrs.get("Units")
            self.line_lams = unyt_array(lams, lam_units).to(angstrom)

            # Check the lines are within the wavelength range of the grid. Its
            # not a big issue if any are outside the range but we should warn
            # the user since it could lead to confusion.
            lines_outside_lam = []
            for ind, line in enumerate(self.available_lines):
                if (
                    self.line_lams[ind] < self.lam[0]
                    or self.line_lams[ind] > self.lam[-1]
                ):
                    lines_outside_lam.append(line)

            # If we have lines outside the wavelength range of the grid
            # warn the user
            if len(lines_outside_lam) > 0:
                warn(
                    "The following lines are outside the wavelength "
                    f"range of the grid: {lines_outside_lam}"
                )

            # Get the units, we only do this once since all the
            # luminosities and continuums will have the same units
            lum_units = hf["lines"]["luminosity"].attrs.get("Units")
            cont_units = hf["lines"]["nebular_continuum"].attrs.get("Units")

            # Read the nebular line luminosities and continuums
            self.line_lums["nebular"] = unyt_array(
                hf["lines"]["luminosity"][...],
                lum_units,
            )
            self.line_conts["nebular"] = unyt_array(
                hf["lines"]["nebular_continuum"][...],
                cont_units,
            )

            # Read the line contribution luminosities and continuums (as
            # called by cloudy, this is the same as nebular in our
            # nomenclature - the line emissions from the birth cloud)
            self.line_lums["linecont"] = unyt_array(
                hf["lines"]["luminosity"][...],
                lum_units,
            )
            self.line_conts["linecont"] = unyt_array(
                np.zeros(self.line_lums["nebular"].shape),
                cont_units,
            )

            # Read the nebular continuum luminosities and continuums
            self.line_lums["nebular_continuum"] = unyt_array(
                np.zeros(self.line_lums["nebular"].shape),
                lum_units,
            )
            self.line_conts["nebular_continuum"] = unyt_array(
                hf["lines"]["nebular_continuum"][...],
                cont_units,
            )

            # Read the transmitted line luminosities and continuums (the
            # emission transmitted through the birth cloud)
            self.line_lums["transmitted"] = unyt_array(
                np.zeros(self.line_lums["nebular"].shape),
                lum_units,
            )
            self.line_conts["transmitted"] = unyt_array(
                hf["lines"]["transmitted"][...],
                cont_units,
            )

            # Now that we have read the line data itself we need to populate
            # the other spectra entries. the line luminosities for all entries
            # other than nebular are 0 but the continuums need to be sampled
            # from the spectra.
            for spectra in self.available_spectra:
                # Define the extraction key
                extract = spectra

                # Skip those we have already read
                if extract in self.line_conts:
                    continue

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
                self.line_conts[spectra] = unyt_array(
                    interp_func(self.line_lams),
                    cont_units,
                )

                # Set the line luminosities to 0 as long as they haven't
                # already been set
                self.line_lums[spectra] = unyt_array(
                    np.zeros((*self.spectra[spectra].shape[:-1], self.nlines)),
                    lum_units,
                )

    def _prepare_lam_axis(
        self,
        new_lam,
        lam_lims,
    ):
        """Modify the grid wavelength axis to match user defined wavelengths.

        This method will do nothing if the user has not provided new_lam
        or lam_lims.

        If the user has passed any of these the wavelength array will be
        limited and/or interpolated to match the user's input.

        - If new_lam is provided, the spectra will be interpolated onto this
          array.
        - If lam_lims are provided, the grid will be truncated to these
          limits.

        Args:
            new_lam (np.ndarray of float):
                An optional user defined wavelength array the spectra will be
                interpolated onto.
            lam_lims (tuple, float):
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
        """Flag for whether grid has been reprocessed through cloudy.

        This will only access the file the first time this property is
        accessed.

        Returns:
            True if reprocessed, False otherwise.
        """
        if self._reprocessed is None:
            with h5py.File(self.grid_filename, "r") as hf:
                old_grids = (
                    True if "cloudy_version" in hf.attrs.keys() else False
                )
                new_grid = True if "CloudyParams" in hf.keys() else False
                self._reprocessed = old_grids or new_grid

        return self._reprocessed

    @property
    def lines_available(self):
        """Flag for whether line emission exists.

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
    def new_line_format(self):
        """Return whether the grid has the new line format."""
        if not self.lines_available:
            return False

        # If the lines group has a luminosity dataset then we have the new
        # format
        with h5py.File(self.grid_filename, "r") as hf:
            return "luminosity" in hf["lines"]

    @property
    def has_spectra(self):
        """Return whether the Grid has spectra."""
        return len(self.spectra) > 0

    @property
    def has_lines(self):
        """Return whether the Grid has lines."""
        return len(self.line_lums) > 0

    @property
    def line_ids(self):
        """Return the line IDs."""
        return self.available_lines

    def _get_spectra_ids_from_file(self):
        """Get a list of the spectra available in a grid file.

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
        """Get a list of the lines available on a grid.

        Returns:
            list:
                List of available lines
                List of associated wavelengths.
        """
        with h5py.File(self.grid_filename, "r") as hf:
            lines = np.array(
                [id.decode("utf-8") for id in hf["lines"]["id"][:]]
            )

            # Read the line wavelengths
            lams = hf["lines"]["wavelength"][...]
            lam_units = hf["lines"]["wavelength"].attrs.get("Units")
            lams = unyt_array(lams, lam_units).to(angstrom)

        return lines, lams

    @accepts(new_lam=angstrom)
    def interp_spectra(self, new_lam, loop_grid=False):
        """Interpolates the spectra grid onto the provided wavelength grid.

        NOTE: this will overwrite self.lam and self.spectra, overwriting
        the attributes loaded from the grid file. To get these back a new grid
        will need to instantiated with no lam argument passed.

        Args:
            new_lam (unyt_array/np.ndarray of float):
                The new wavelength array to interpolate the spectra onto.
            loop_grid (bool):
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
        """Return a string representation of the particle object.

        Returns:
            table (str):
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Grid")

    @property
    def shape(self):
        """Return the shape of the grid."""
        return self.spectra[self.available_spectra[0]].shape

    @property
    def ndim(self):
        """Return the number of dimensions in the grid."""
        return len(self.shape)

    @property
    def nlam(self):
        """Return the number of wavelengths in the grid."""
        return len(self.lam)

    @property
    def nlines(self):
        """Return the number of lines in the grid."""
        return len(self.available_lines)

    @staticmethod
    def get_nearest_index(value, array):
        """Calculate the closest index in an array for a given value.

        TODO: What is this doing here!?

        Args:
            value (float/unyt_quantity):
                The target value.
            array (np.ndarray/unyt_array):
                The array to search.

        Returns:
            int
                The index of the closet point in the grid (np.ndarray):
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
        """Identify the nearest grid point for a tuple of values.

        Any axes not specified will be returned as a full slice.

        Args:
            **kwargs (dict):
                Pairs of axis names and values for the desired grid point,
                e.g. log10ages=9.3, log10metallicities=-2.1.

        Returns:
            tuple
                A tuple of integers specifying the closest grid point.
        """
        # Create a list we will return
        indices = []

        # Loop over axes and get the nearest index for each
        for axis in self.axes:
            # Get plural, singular and log10 versions of the axis name
            plural_axis = pluralize(axis)
            singular_axis = depluralize(axis)
            log10_axis = f"log10{axis}"
            log10_plural_axis = f"log10{plural_axis}"
            log10_singular_axis = f"log10{singular_axis}"
            if axis in kwargs:
                indices.append(
                    self.get_nearest_index(
                        kwargs.pop(axis), getattr(self, axis)
                    )
                )
            elif plural_axis in kwargs:
                indices.append(
                    self.get_nearest_index(
                        kwargs.pop(plural_axis), getattr(self, plural_axis)
                    )
                )
            elif singular_axis in kwargs:
                indices.append(
                    self.get_nearest_index(
                        kwargs.pop(singular_axis), getattr(self, singular_axis)
                    )
                )
            elif log10_axis in kwargs:
                indices.append(
                    self.get_nearest_index(
                        kwargs.pop(log10_axis), getattr(self, axis)
                    )
                )
            elif log10_plural_axis in kwargs:
                indices.append(
                    self.get_nearest_index(
                        kwargs.pop(log10_plural_axis),
                        getattr(self, plural_axis),
                    )
                )
            elif log10_singular_axis in kwargs:
                indices.append(
                    self.get_nearest_index(
                        kwargs.pop(log10_singular_axis),
                        getattr(self, singular_axis),
                    )
                )
            else:
                indices.append(slice(None))

        # Warn the user is any kwargs weren't a grid axis
        if len(kwargs) > 0:
            warn(
                "The following axes are not on the grid:"
                f" {list(kwargs.keys())}"
            )

        return tuple(indices)

    def _collapse_grid_marginalize(self, axis, marginalize_function):
        """Collapse the grid by marginalizing over an entire axis.

        Args:
            axis (str):
                The name of the axis to collapse.
            marginalize_function (function):
                The function to use for marginalizing over the axis.
        """
        # Get the index of the axis to collapse
        axis_index = self.axes.index(axis)

        for spectra_id in self.available_spectra:
            # Marginalize over the entire axis
            self.spectra[spectra_id] = marginalize_function(
                self.spectra[spectra_id], axis=axis_index
            )

            # Marginalize the line luminosities and continua
            self.line_lums[spectra_id] = marginalize_function(
                self.line_lums[spectra_id], axis=axis_index
            )
            self.line_conts[spectra_id] = marginalize_function(
                self.line_conts[spectra_id], axis=axis_index
            )

    def _collapse_grid_interpolate(self, axis, value, pre_interp_function):
        """Collapse the grid by interpolating to the specified value.

        Args:
            axis (str):
                The name of the axis to collapse.
            value (float):
                The value to collapse the grid at.
            pre_interp_function (function):
                A function to apply to the axis values before interpolation.
        """
        # Get the index of the axis to collapse
        axis_index = self.axes.index(axis)
        axis_values = getattr(self, axis)

        # Check the value is provided
        if value is None:
            raise exceptions.InconsistentParameter(
                "Must provide kwarg `value` if `method` is "
                "`interpolate` or `nearest`"
            )

        # Check the value is within the bounds of the axis
        if value < np.min(axis_values) or value > np.max(axis_values):
            raise exceptions.InconsistentParameter(
                f"Value {value} is outside the bounds of the axis {axis}."
            )

        # Validate axis monotonicity
        dv = np.diff(axis_values)
        if not np.all(dv > 0):
            raise exceptions.UnimplementedFunctionality(
                "Axis values must be strictly increasing for interpolation."
            )

        # Adopt pre-interpolation function if provided
        pre_interp_function = pre_interp_function or (lambda x: x)

        for spectra_id in self.available_spectra:
            # Interpolate the spectra
            i0 = np.argmax(axis_values[axis_values <= value])
            i1 = i0 + 1
            value_0 = pre_interp_function(axis_values[i0])
            value_1 = pre_interp_function(axis_values[i1])
            value_i = pre_interp_function(value)
            c0 = (value_1 - value_i) / (value_1 - value_0)
            c1 = (value_i - value_0) / (value_1 - value_0)

            self.spectra[spectra_id] = np.sum(
                [
                    c0
                    * np.take(self.spectra[spectra_id], i0, axis=axis_index),
                    c1
                    * np.take(self.spectra[spectra_id], i1, axis=axis_index),
                ],
                axis=0,
            )

            # Interpolate the line luminosities and continua
            self.line_lums[spectra_id] = np.sum(
                [
                    c0
                    * np.take(
                        self.line_lums[spectra_id],
                        i0,
                        axis=axis_index,
                    ),
                    c1
                    * np.take(
                        self.line_lums[spectra_id],
                        i1,
                        axis=axis_index,
                    ),
                ],
                axis=0,
            )
            self.line_conts[spectra_id] = np.sum(
                [
                    c0
                    * np.take(
                        self.line_conts[spectra_id],
                        i0,
                        axis=axis_index,
                    ),
                    c1
                    * np.take(
                        self.line_conts[spectra_id],
                        i1,
                        axis=axis_index,
                    ),
                ],
                axis=0,
            )

    def _collapse_grid_nearest(self, axis, value):
        """Collapse the grid by extracting the nearest value of the axis.

        Args:
            axis (str):
                The name of the axis to collapse.
            value (float):
                The value to collapse the grid at.
        """
        # Get the index of the axis to collapse
        axis_index = self.axes.index(axis)
        axis_values = getattr(self, axis)

        # Check the value is provided
        if value is None:
            raise exceptions.InconsistentParameter(
                "Must provide kwarg `value` if `method` is "
                "`interpolate` or `nearest`"
            )
        # Check the value is within the bounds of the axis
        if value < np.min(axis_values) or value > np.max(axis_values):
            raise exceptions.InconsistentParameter(
                f"Value {value} is outside the bounds of the axis {axis}."
            )

        for spectra_id in self.available_spectra:
            # Extract the nearest value in the axis
            self.spectra[spectra_id] = np.take(
                self.spectra[spectra_id],
                np.argmin(np.abs(axis_values - value)),
                axis=axis_index,
            )

            # Extract the line luminosities and continua
            self.line_lums[spectra_id] = np.take(
                self.line_lums[spectra_id],
                np.argmin(np.abs(axis_values - value)),
                axis=axis_index,
            )
            self.line_conts[spectra_id] = np.take(
                self.line_conts[spectra_id],
                np.argmin(np.abs(axis_values - value)),
                axis=axis_index,
            )

    def collapse(
        self,
        axis,
        method="marginalize",
        value=None,
        marginalize_function=np.average,
        pre_interp_function=None,
    ):
        """Collapse the grid in place along a specified axis.

        Reduces the dimensionality of the grid by collapsing along the
        specified axis, using the specified method. The method can be
        "marginalize", "interpolate", or "nearest". Note that
        interpolation can yield unrealistic results if the grid is
        particularly coarse, and should be used with caution.

        Args:
            axis (str):
                The name of the axis to collapse.
            method (str):
                The method to use for collapsing the grid. Options are
                "marginalize", "interpolate", or "nearest". Defaults
                to "marginalize".
            value (float):
                The value to collapse the grid at. Required if method
                is "interpolate" or "nearest".
            marginalize_function (function):
                The function to use for marginalizing over the axis.
                Defaults to np.average.
            pre_interp_function (function):
                A function to apply to the axis values before interpolation.
                Can be used to interpolate in logarithmic space, for example.
                Defaults to None, i.e., interpolation in linear space.

        Returns:
            None
                Collapses the grid in-place over the specified axis.
        """
        # Check the axis is valid
        if axis not in self.axes:
            raise exceptions.InconsistentParameter(
                f"Axis {axis} is not a valid axis on the grid."
            )

        # Check the method is valid
        if method not in ["marginalize", "interpolate", "nearest"]:
            raise exceptions.InconsistentParameter(
                f"Method {method} is not a valid collapse method."
            )

        # Collapse the spectra based on the method
        if method == "marginalize":
            # Marginalize over the entire axis
            self._collapse_grid_marginalize(axis, marginalize_function)

        elif method == "interpolate":
            # Interpolate the grid at the given value
            self._collapse_grid_interpolate(axis, value, pre_interp_function)

        elif method == "nearest":
            # Extract the nearest value in the axis
            self._collapse_grid_nearest(axis, value)

        # Update the grid metadata
        self.axes.pop(self.axes.index(axis))
        self._axes_values.pop(axis)
        self._axes_units.pop(axis)
        self.naxes -= 1

    def get_sed_at_grid_point(self, grid_point, spectra_type="incident"):
        """Create an Sed object for a specific grid point.

        Args:
            grid_point (tuple):
                A tuple of integers specifying the closest grid point.
            spectra_type (str):
                The name of the spectra (in the grid) that is desired.

        Returns:
            synthesizer.emissions.Sed
                A synthesizer.emissions object
        """
        # Throw exception if the spectra_id not in list of available spectra
        if spectra_type not in self.available_spectra:
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
                lnu=self.spectra[spectra_type][grid_point] * erg / s / Hz,
            )
        except IndexError:
            # Modify the error message for clarity
            raise IndexError(
                f"grid_point is outside of the grid (grid.shape={self.shape}, "
                f"grid_point={grid_point})"
            )

    def get_sed_for_full_grid(self, spectra_type="incident"):
        """Create an Sed object for the full grid.

        Args:
            spectra_type (str):
                The name of the spectra (in the grid) that is desired.

        Returns:
            synthesizer.emissions.Sed:
                A synthesizer.emissions object
        """
        # Throw exception if the spectra_id not in list of available spectra
        if spectra_type not in self.available_spectra:
            raise exceptions.InconsistentParameter(
                "Provided spectra_id is not in the list of available spectra."
            )

        return Sed(self.lam, self.spectra[spectra_type] * erg / s / Hz)

    def get_sed(self, grid_point=None, spectra_type="incident"):
        """Create an Sed object from the grid.

        This can either get an Sed containing the entire grid (if
        grid_point=False) or for a specific grid point.

        This enables grid wide use of Sed methods for flux, photometry,
        indices, ionising photons, etc.

        Args:
            grid_point (tuple/bool):
                A tuple of integers specifying the closest grid point or a
                bool.
            spectra_type (str):
                The key of the spectra grid to extract as an Sed object.

        Returns:
            Sed:
                An Sed object.
        """
        # If a grid point is provided call the function above ...
        if grid_point is not None:
            return self.get_sed_at_grid_point(
                grid_point=grid_point,
                spectra_type=spectra_type,
            )

        # ... otherwise, return the entire Sed grid.
        else:
            return self.get_sed_for_full_grid(
                spectra_type=spectra_type,
            )

    def get_lines_at_grid_point(
        self, grid_point, line_id=None, spectra_type="nebular"
    ):
        """Create a LineCollection object for a specific grid point.

        Args:
            grid_point (tuple):
                A tuple of integers specifying the closest grid point.
            line_id (str/list):
                The id/s of the line. If a string contains a comma separated
                list of line_ids a composite line will be returned containing
                the sum of the luminosities and the mean of the wavelengths.
                If a list of line_ids is provided a subset of lines will be
                returned. If None then all available lines will be returned.
            spectra_type (str):
                The spectra type to extract the line from. Default is
                "nebular", all other spectra will have line luminosities of 0
                by definition.

        Returns:
            synthesizer.emissions.Sed
                A synthesizer.emissions object
        """
        # First create a LineCollection containing the grid point
        all_lines = LineCollection(
            line_ids=self.available_lines,
            lam=self.line_lams,
            lum=self.line_lums[spectra_type][grid_point],
            cont=self.line_conts[spectra_type][grid_point],
        )

        # If we have no line_id we are done and can return the full collection
        if line_id is None:
            return all_lines

        # Now we simply extract the line we want from the collection
        line = all_lines[line_id]

        # Formally delete the all_lines object, gargabe collection would
        # probably do this anyway but we do it here to be sure.
        del all_lines

        return line

    def get_lines_for_full_grid(self, line_id=None, spectra_type="nebular"):
        """Create a LineCollection object for a the full grid.

        Args:
            line_id (str/list):
                The id/s of the line. If a string contains a comma separated
                list of line_ids a composite line will be returned containing
                the sum of the luminosities and the mean of the wavelengths.
                If a list of line_ids is provided a subset of lines will be
                returned. If None then all available lines will be returned.
            spectra_type (str):
                The spectra type to extract the line from. Default is
                "nebular", all other spectra will have line luminosities of 0
                by definition.

        Returns:
            Sed:
                A synthesizer.emissions object
        """
        # First create a LineCollection containing the grid point
        all_lines = LineCollection(
            line_ids=self.available_lines,
            lam=self.line_lams,
            lum=self.line_lums[spectra_type],
            cont=self.line_conts[spectra_type],
        )

        # If we have no line_id we are done and can return the full collection
        if line_id is None:
            return all_lines

        # Now we simply extract the line we want from the collection
        line = all_lines[line_id]

        # Formally delete the all_lines object, gargabe collection would
        # probably do this anyway but we do it here to be sure.
        del all_lines

        return line

    def get_lines(self, grid_point=None, line_id=None, spectra_type="nebular"):
        """Create a Line object for a given line_id and grid_point.

        Args:
            grid_point (tuple/bool):
                A tuple of integers specifying the closest grid point.
            line_id (str/list):
                The id/s of the line. If a string contains a comma separated
                list of line_ids a composite line will be returned containing
                the sum of the luminosities and the mean of the wavelengths.
                If a list of line_ids is provided a subset of lines will be
                returned. If None then all available lines will be returned.
            spectra_type (str):
                The spectra type to extract the line from. Default is
                "nebular", all other spectra will have line luminosities of 0
                by definition.

        Returns:
            lines (LineCollection):
                A LineCollection object containing the line luminosities and
                continuums.
        """
        # If a grid point is provided call the function above ...
        if grid_point is not None:
            return self.get_lines_at_grid_point(
                grid_point=grid_point,
                line_id=line_id,
                spectra_type=spectra_type,
            )
        # ... otherwise return the full grid.
        else:
            return self.get_lines_for_full_grid(
                line_id=line_id, spectra_type=spectra_type
            )

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
        """Make a simple plot of the specific ionising photon luminosity.

        The resulting figure will show the (log) specific ionsing photon
        luminosity as a function of (log) age and metallicity for a given grid
        and ion.

        Args:
            ion (str):
                The desired ion. Most grids only have HI and HII calculated by
                default.
            hsize (float):
                The horizontal size of the figure
            vsize (float):
                The vertical size of the figure
            cmap (object/str):
                A colourmap object or string defining the matplotlib colormap.
            vmin (float):
                The minimum specific ionising luminosity used in the colourmap
            vmax (float):
                The maximum specific ionising luminosity used in the colourmap
            max_log10age (float):
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
        log10_specific_ionising_lum = self.log10_specific_ionising_lum[ion]

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
        """Calculate the delta lambda for the given spectra.

        Args:
            spectra_id (str):
                Identifier for the spectra (default is "incident").

        Returns:
            tuple
                A tuple containing the list of wavelengths and delta lambda.
        """
        # Calculate delta lambda for each wavelength
        delta_lambda = np.log10(self.lam[1:]) - np.log10(self.lam[:-1])

        # Return tuple of wavelengths and delta lambda
        return self.lam, delta_lambda

    def get_flattened_axes_values(self):
        """Get the flattened axis values for the grid.

        This will return a dictionary of the axes values that correspond to
        the grid once flattened.

        Returns:
            dict
                A dictionary containing the flattened versions of the axes.
        """
        # Create a tuple of the axes values
        axes_values_tuple = (self._axes_values[axis] for axis in self.axes)

        # Create a meshgrid of the axes_values
        axes_values_mesh_tuple = np.meshgrid(*axes_values_tuple, indexing="ij")

        # Create a dictionary with flattened versions of the axes values
        flattened_axes_values = {
            axis: unyt_array(
                axes_values_mesh_tuple[axis_index].flatten(),
                self._axes_units[axis],
            )
            for axis_index, axis in enumerate(self.axes)
        }

        return flattened_axes_values

    @accepts(min_lam=angstrom, max_lam=angstrom)
    def truncate_grid_lam(self, min_lam, max_lam):
        """Truncate the grid to a specific wavelength range.

        If out of range wavlengths are requested, the grid will be
        truncated to the nearest wavelength within the grid.

        Args:
            min_lam (unyt_quantity):
                The minimum wavelength to truncate the grid to.

            max_lam (unyt_quantity):
                The maximum wavelength to truncate the grid to.
        """
        # Get the indices of the wavelengths to keep
        okinds = np.logical_and(self.lam >= min_lam, self.lam <= max_lam)

        # Apply the mask to the grid wavelengths
        self.lam = self.lam[okinds]

        # Apply the mask to the spectra
        for spectra_type in self.available_spectra:
            self.spectra[spectra_type] = np.ascontiguousarray(
                self.spectra[spectra_type][..., okinds]
            )

    def animate_grid(
        self,
        show=False,
        save_path=None,
        fps=30,
        spectra_type="incident",
    ):
        """Create an animation of the grid stepping through wavelength.

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
            spectra_type (str):
                The spectra type to animate. Default is "incident".

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
    """A simplified grid contain only a single template spectra.

    The model is different to all other emission models in that it scales a
    template by bolometric luminosity.

    Attributes:
        sed (Sed)
            The template spectra for the AGN.
        normalisation (unyt_quantity):
            The normalisation for the spectra. In reality this is the
            bolometric luminosity.
    """

    # Define Quantities
    lam = Quantity("wavelength")
    lnu = Quantity("luminosity_density_frequency")

    @accepts(lam=angstrom, lnu=erg / s / Hz)
    def __init__(
        self,
        lam,
        lnu,
        unify_with_grid=None,
        **kwargs,
    ):
        """Initialise the Template.

        Args:
            lam (unyt_array):
                Wavelength array.
            lnu (unyt_array):
                Luminosity array.
            unify_with_grid (Grid):
                A grid object to unify the template with. This will ensure
                the template has the same wavelength array as the grid.
            **kwargs (dict):
                Additional keyword arguments to pass to the Sed object.

        """
        # It's convenient to have an sed object for the next steps
        sed = Sed(lam, lnu)

        # Before we do anything, do we have a grid we need to unify with?
        if unify_with_grid is not None:
            # Interpolate the template Sed onto the grid wavelength array
            sed = sed.get_resampled_sed(new_lam=unify_with_grid.lam)

        # Attach the template now we've done the interpolation (if needed)
        self._sed = sed
        self.lam = sed.lam

        # Normalise, just in case
        self.normalisation = sed.bolometric_luminosity
        self._sed._lnu /= self.normalisation.to(self._sed.lnu.units * Hz).value

    @accepts(bolometric_luminosity=erg / s)
    def get_spectra(self, bolometric_luminosity):
        """Calculate the blackhole spectra by scaling the template.

        Args:
            bolometric_luminosity (float):
                The bolometric luminosity of the blackhole(s) for scaling.

        """
        # Ensure we have units for safety
        if bolometric_luminosity is not None and not isinstance(
            bolometric_luminosity, unyt_array
        ):
            raise exceptions.MissingUnits(
                "bolometric luminosity must be provided with units"
            )

        # Scale the spectra and return
        return self._sed * (bolometric_luminosity / Hz)
