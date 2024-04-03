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
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize
from matplotlib.figure import Figure
from numpy.typing import NDArray
from spectres import spectres
from unyt import angstrom, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.filters import FilterCollection
from synthesizer.line import Line, LineCollection, flatten_linelist
from synthesizer.sed import Sed
from synthesizer.units import Quantity

from . import __file__ as filepath


class Grid:
    """
    The Grid class containing tabulated spectral and line data.

    This object contains attributes and methods for reading and
    manipulating the spectral grids which underpin all spectra/line
    generation in synthesizer.

    Attributes:
        grid_dir: The directory containing the grid HDF5 file.
        grid_name: The name of the grid (as defined by the file name)
                   with no extension.
        grid_ext: The grid extension. Either ".hdf5" or ".h5". If the passed
                  grid_name has no extension then ".hdf5" is assumed.
        grid_filename: The full path to the grid file.
        available_lines: A list of lines on the Grid.
        available_spectra: A list of spectra on the Grid.
        reprocessed: Flag for whether the grid has been reprocessed through
                     cloudy.
        lines_available: Flag for whether lines are available on this grid.
        lam: The wavelengths at which the spectra are defined.
        spectra: The spectra array from the grid. This is an N-dimensional
                 grid where N is the number of axes of the SPS grid. The final
                 dimension is always wavelength.
        lines: The lines array from the grid. This is an N-dimensional grid
               where N is the number of axes of the SPS grid. The final
               dimension is always wavelength.
        parameters: A dictionary containing the grid's parameters used in its
                    generation.
        axes: A list of the names of the spectral grid axes.
        naxe: The number of axes the spectral grid has.
        logQ10: A dictionary of ionisation Q parameters. (DEPRECATED)
        log10_specific_ionising_luminosity: A dictionary of log10 specific
                                            ionising luminosities.
        <grid_axis>: A Grid will always contain 1D arrays corresponding to the
                     axes of the spectral grid. These are read dynamically from
                     the HDF5 file so can be anything but usually contain at
                     least stellar ages and stellar metallicity.
        lam: The wavelengths at which the spectra are defined.
    """

    grid_dir: str
    grid_name: str
    grid_ext: str
    grid_filename: str
    available_lines: List[str]
    available_spectra: List[str]
    _reprocessed: Optional[bool] = None
    _lines_available: Optional[bool] = None
    spectra: Dict[str, NDArray[np.float64]]
    lines: Dict[str, Dict[str, NDArray[np.float64]]]
    parameters: Dict[str, Any]
    axes: List[str]
    naxes: int
    logQ10: Optional[Dict[str, unyt_array]] = None
    log10_specific_ionising_luminosity: Optional[Dict[str, unyt_array]] = None

    # Type hint some common grid axes (it would be good to have all commonly
    # used grid axes listed here, both for documentation purposes but also
    # to capture them in the typing should be ever apply mypyc)
    log10age: unyt_array
    metallicity: NDArray[np.float64]

    # Define Quantities
    _lam: NDArray[np.float64]
    lam: unyt_array
    lam = Quantity()

    def __init__(
        self,
        grid_name: str,
        grid_dir: Optional[str] = None,
        read_spectra: Union[bool, List[str]] = True,
        read_lines: Union[bool, List[str]] = True,
        new_lam: Optional[unyt_array] = None,
        filters: Optional[FilterCollection] = None,
        lam_lims: Union[Tuple[()], Tuple[float, float]] = (),
    ) -> None:
        """
        Initialise the grid object.

        This will open the grid file and extract the axes, spectra (if
        requested), and lines (if requested) and any other relevant data.

        Args:
            grid_name: The file name of the grid (if no extension is provided
                       then hdf5 is assumed).
            grid_dir: The file path to the directory containing the grid file.
            read_spectra: Should we read the spectra? If a list then a subset
                          of spectra will be read.
            read_lines: Should we read lines? If a list then a subset of lines
                        will be read.
            new_lam: An optional user defined wavelength array the spectra will
                     be interpolated onto, see Grid.interp_spectra.
            filters: An optional FilterCollection object to unify the grids
                     wavelength grid with. If provided, this will override
                     new_lam whether passed or not.
            lam_lims: A tuple of the lower and upper wavelength limits to
                      truncate the grid to (i.e. (lower_lam, upper_lam)). If
                      new_lam or filters are provided these limits will be
                      ignored.
        """
        # Get the grid file path data
        self.grid_dir = grid_dir if grid_dir is not None else ""
        self.grid_name = ""
        self.grid_ext = "hdf5"  # can be updated if grid_name has an extension
        self.grid_filename = ""
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
        self.lines = {}

        # Set up dictionary to hold parameters used in grid generation
        self.parameters = {}

        # Get the axes of the grid from the HDF5 file
        self.axes = []
        self.naxes = 0
        self._get_axes()

        # Get the ionising luminosity (if available)
        self._get_ionising_luminosity()

        # Read in spectra
        if read_spectra:  # also True if read_spectra is a list
            self._get_spectra_grid(read_spectra)

        # Read in lines
        if read_lines:  # also True if read_lines is a list
            self._get_lines_grid(read_lines)

        # Prepare the wavelength axis (if new_lam, lam_lims and filters are
        # all None, this will do nothing, leaving the grid's wavelength array
        # as it is in the HDF5 file)
        self._prepare_lam_axis(new_lam, filters, lam_lims)

    def _parse_grid_path(
        self,
        grid_dir: Optional[str],
        grid_name: str,
    ) -> None:
        """
        Parse the grid path and set the grid directory and filename.

        Args:
            grid_dir: The directory containing the grid file.
            grid_name: The name of the grid file.
        """
        # If we haven't been given a grid directory, assume the grid is in
        # the package's "data/grids" directory.
        if grid_dir is None:
            grid_dir = os.path.join(os.path.dirname(filepath), "data/grids")

        # Store the grid directory
        self.grid_dir = grid_dir

        # Have we been passed an extension?
        grid_name_split: List[str] = grid_name.split(".")
        ext: str = grid_name_split[-1]
        if ext == "hdf5" or ext == "h5":
            self.grid_ext = ext

        # Strip the extension off the name (harmless if no extension)
        self.grid_name = grid_name.replace(f".{self.grid_ext}", "")

        # Construct the full path
        self.grid_filename = (
            f"{self.grid_dir}/{self.grid_name}.{self.grid_ext}"
        )

    def _get_axes(self) -> None:
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

    def _get_ionising_luminosity(self) -> None:
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

    def _get_spectra_grid(self, read_spectra: Union[bool, List[str]]) -> None:
        """
        Get the spectra grid from the HDF5 file.

        If using a cloudy reprocessed grid this method will automatically
        calculate 2 spectra not native to the grid file:
            total = transmitted + nebular
            nebular_continuum = nebular + linecont

        Args:
            read_spectra: Flag for whether to read all available spectra or
                          subset of spectra to read.
        """
        with h5py.File(self.grid_filename, "r") as hf:
            # Are we only reading a subset?
            if isinstance(read_spectra, list):
                self.available_spectra = read_spectra
            else:
                # If not, read all available spectra
                self.available_spectra = self.get_grid_spectra_ids()

            # Remove wavelength dataset
            self.available_spectra.remove("wavelength")

            # Remove normalisation dataset
            if "normalisation" in self.available_spectra:
                self.available_spectra.remove("normalisation")

            # Get all our spectra
            for spectra_id in self.available_spectra:
                self.lam = hf["spectra/wavelength"][:]
                self.spectra[spectra_id] = hf["spectra"][spectra_id][:]

        # If a full cloudy grid is available calculate some
        # other spectra for convenience.
        if self.reprocessed:
            self.spectra["total"] = (
                self.spectra["transmitted"] + self.spectra["nebular"]
            )
            self.available_spectra.append("total")

            self.spectra["nebular_continuum"] = (
                self.spectra["nebular"] - self.spectra["linecont"]
            )
            self.available_spectra.append("nebular_continuum")

    def _get_lines_grid(self, read_lines: Union[bool, List[str]]) -> None:
        """
        Get the lines grid from the HDF5 file.

        Args:
            read_lines: Flag for whether to read all available lines or subset
                        of lines to read.
        """
        # Double check we actually have lines to read
        if not self.lines_available:
            if not self.reprocessed:
                raise exceptions.GridError(
                    "Grid hasn't been reprocessed with cloudy and has no"
                    "lines. Either pass `read_lines=False` or load a grid"
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
                self.available_lines, _ = self.get_grid_line_ids()

            # Read in the lines
            for line in self.available_lines:
                self.lines[line] = {}
                self.lines[line]["wavelength"] = hf["lines"][line].attrs[
                    "wavelength"
                ]
                self.lines[line]["luminosity"] = hf["lines"][line][
                    "luminosity"
                ][:]
                self.lines[line]["continuum"] = hf["lines"][line]["continuum"][
                    :
                ]

    def _prepare_lam_axis(
        self,
        new_lam: Optional[unyt_array],
        filters: Optional[FilterCollection],
        lam_lims: Union[Tuple[()], Tuple[float, float]] = (),
    ) -> None:
        """
        Modify the grid wavelength axis to adhere to user defined wavelengths.

        This method will do nothing if the user has not provided new_lam,
        filters, or lam_lims.

        If the user has passed any of these the wavelength array will be
        limited and/or interpolated to match the user's input.

        - If new_lam is provided, the spectra will be interpolated onto this
          array.
        - If filters are provided, the spectra will be limited wavelengths
          with non-zero transmission and the filters will be interpolated
          onto the grid's wavelength array in this non-zero range.
        - If lam_lims are provided, the grid will be truncated to these
          limits.

        Args:
            new_lam: An optional user defined wavelength array the spectra will
                     be interpolated onto.
            filters: An optional FilterCollection object to unify the grids
                     wavelength grid with.
            lam_lims: A tuple of the lower and upper wavelength limits to
                      truncate the grid to (i.e. (lower_lam, upper_lam)).
        """
        # If we have both new_lam (or filters) and wavelength limits
        # the limits become meaningless tell the user so.
        if len(lam_lims) > 0 and (new_lam is not None or filters is not None):
            print(
                "Passing new_lam or filters and lam_lims is contradictory, "
                "lam_lims will be ignored."
            )

        # Has a new wavelength grid been passed to interpolate
        # the spectra onto?
        if new_lam is not None and filters is None:
            # Double check we aren't being asked to do something impossible.
            if self.spectra is None:
                raise exceptions.InconsistentArguments(
                    "Can't interpolate spectra onto a new wavelength array if"
                    " no spectra have been read in! Set read_spectra=True."
                )

            # Interpolate the spectra grid
            self.interp_spectra(new_lam)

        # Are we unifying with a filter collection?
        if filters is not None:
            # Double check we aren't being asked to do something impossible.
            if self.spectra is None:
                raise exceptions.InconsistentArguments(
                    "Can't interpolate spectra onto a FilterCollection "
                    "wavelength array if no spectra have been read in! "
                    "Set read_spectra=True."
                )

            # Warn the user the new_lam will be ignored
            if new_lam is not None:
                print(
                    "If a FilterCollection is defined alongside new_lam "
                    "then FilterCollection.lam takes precedence and new_lam "
                    "is ignored"
                )

            self.unify_with_filters(filters)

        # If we have been given wavelength limtis truncate the grid
        if len(lam_lims) > 0 and filters is None and new_lam is None:
            self.truncate_grid_lam(*lam_lims)

    @property
    def reprocessed(self) -> bool:
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
    def lines_available(self) -> bool:
        """
        Flag for whether line emission information is available on this grid.

        This will only access the file the first time this property is
        accessed.

        Returns:
            True if lines are available, False otherwise.
        """
        if self._lines_available is None:
            with h5py.File(self.grid_filename, "r") as hf:
                self._lines_available = True if "lines" in hf.keys() else False

        return self._lines_available

    @property
    def has_spectra(self) -> bool:
        """
        Return whether the Grid has spectra.

        Returns:
            True if the Grid has spectra, False otherwise.
        """
        return len(self.spectra) > 0

    @property
    def has_lines(self) -> bool:
        """
        Return whether the Grid has lines.

        Returns:
            True if the Grid has lines, False otherwise.
        """
        return len(self.lines) > 0

    def get_grid_spectra_ids(self) -> List[str]:
        """
        Get a list of the spectra available on a grid.

        Returns:
            List of available spectra
        """
        with h5py.File(self.grid_filename, "r") as hf:
            return list(hf["spectra"].keys())

    def get_grid_line_ids(self) -> Tuple[List[str], NDArray[np.float64]]:
        """
        Get a list of the lines available on a grid.

        Returns:
            List of available lines
            List of associated wavelengths.
        """
        with h5py.File(self.grid_filename, "r") as hf:
            lines = list(hf["lines"].keys())

            wavelengths = np.array(
                [hf["lines"][line].attrs["wavelength"] for line in lines]
            )
        return lines, wavelengths

    def interp_spectra(
        self,
        new_lam: unyt_array,
        loop_grid: bool = False,
    ) -> None:
        """
        Interpolates the spectra grid onto the provided wavelength grid.

        NOTE: this will overwrite self.lam and self.spectra, overwriting
        the attributes loaded from the grid file. To get these back a new grid
        will need to instantiated with no lam argument passed.

        Args:
            new_lam: The new wavelength array to interpolate the spectra onto.
            loop_grid: Flag for whether to do the interpolation over the whole
                       grid, or loop over the first axes. The latter is less
                       memory intensive, but slower. Defaults to False.
        """
        # Ensure we've been passed wavelengths with units
        if not isinstance(new_lam, unyt_array):
            raise exceptions.InconsistentArguments(
                "Wavelengths must be passed with units!"
            )

        # Convert the passed wavelengths to the same units as the Grid
        if new_lam.units != self.lam.units:
            new_lam = new_lam.to(self.lam.units)

        # Strip off the units (interpolation methods tend not to like them)
        new_lam_value: NDArray[np.float64] = new_lam.to(self.lam.units).value

        # Loop over spectra to interpolate
        new_spectra_arr: NDArray[np.float64]
        for spectra_type in self.available_spectra:
            # Are we doing the look up in one go, or looping?
            if loop_grid:
                new_spectra: List[Optional[NDArray[np.float64]]] = [
                    None
                ] * len(self.spectra[spectra_type])

                # Loop over first axis of spectra array
                for i, _spec in enumerate(self.spectra[spectra_type]):
                    new_spectra[i] = spectres(new_lam_value, self._lam, _spec)

                # Convert to an array
                new_spectra_arr = np.asarray(new_spectra)
            else:
                # Evaluate the function at the desired wavelengths
                new_spectra_arr = spectres(
                    new_lam_value, self._lam, self.spectra[spectra_type]
                )

            # Update this spectra
            self.spectra[spectra_type] = new_spectra_arr

        # Update wavelength array
        self.lam = new_lam

    def __str__(self) -> str:
        """
        Return a basic summary of the Grid object.

        Returns:
            A summary of the Grid.
        """
        # Set up the string for printing
        pstr: str = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 30 + "\n"
        pstr += "SUMMARY OF GRID" + "\n"
        for axis in self.axes:
            pstr += f"{axis}: {getattr(self, axis)} \n"
        for k, v in self.parameters.items():
            pstr += f"{k}: {v} \n"
        if self.lines:
            pstr += f"available lines: {self.available_lines}\n"
        if self.spectra:
            pstr += f"available spectra: {self.available_spectra}\n"
        pstr += "-" * 30 + "\n"

        return pstr

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Return the shape of the grid.

        Returns:
            The shape of the spectra grids.
        """
        return self.spectra[self.available_spectra[0]].shape

    @staticmethod
    def get_nearest_index(
        value: Union[float, unyt_quantity],
        array: Union[NDArray[np.float64], unyt_array, unyt_quantity],
    ) -> int:
        """
        Calculate the closest index in an array for a given value.

        Args:
            value: The target value.

            array: The array to search.

        Returns:
            The index of the closet point in the grid (array)
        """
        array_value: NDArray[np.float64]
        value_value: float

        # Do we need a conversion?
        if isinstance(array, unyt_array) and isinstance(value, unyt_quantity):
            if array.units != value.units:
                value = value.to(array.units)
        else:
            array_value = array
            value_value = value

        # Strip off the units
        if isinstance(array, unyt_array):
            array_value = array.value
        if isinstance(value, unyt_quantity):
            value_value = value.value

        return (np.abs(array_value - value_value)).argmin()

    def get_grid_point(
        self,
        values: Tuple[float, unyt_quantity],
    ) -> Tuple[int, ...]:
        """
        Identify the nearest grid point for a tuple of values.

        Args:
            values: The values for which we want the grid point. These have to
                    be in the same order as the axes.

        Returns:
            A tuple of integers specifying the closest grid point.
        """
        return tuple(
            [
                self.get_nearest_index(value, getattr(self, axis))
                for axis, value in zip(self.axes, values)
            ]
        )

    def get_spectra(
        self,
        grid_point: Tuple[int],
        spectra_id: str = "incident",
    ) -> Sed:
        """
        Create an Sed object for a specific grid point.

        Args:
            grid_point: A tuple of integers specifying the closest grid point.
            spectra_id: The name of the spectra (in the grid) that is desired.

        Returns:
            A synthesizer Sed object containing the spectra at grid_point.
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
            return Sed(self.lam, lnu=self.spectra[spectra_id][grid_point])
        except IndexError:
            # Modify the error message for clarity
            raise IndexError(
                f"grid_point is outside of the grid (grid.shape={self.shape}, "
                f"grid_point={grid_point})"
            )

    def get_line(
        self, grid_point: Tuple[int], line_id: Union[str, List[str]]
    ) -> Line:
        """
        Create a Line object for a given line_id and grid_point.

        Args:
            grid_point: A tuple of integers specifying the closest grid point.
            line_id: The id of the line.

        Returns:
            A synthesizer Line object containing the line at grid_point.
        """
        # Throw exception if the grid_point has a different shape from the grid
        if len(grid_point) != self.naxes:
            raise exceptions.InconsistentParameter(
                "The grid_point tuple provided"
                "as an argument should have the same shape as the grid."
            )

        line_ids: List[str]
        if isinstance(line_id, str):
            line_ids = [line_id]
        else:
            line_ids = line_id

        wavelength: List[np.float64] = []
        luminosity: List[np.float64] = []
        continuum: List[np.float64] = []

        for line_id_ in line_ids:
            # Throw exception if tline_id not in list of available lines
            if line_id_ not in self.available_lines:
                raise exceptions.InconsistentParameter(
                    "Provided line_id is not in the list of available lines."
                )

            line_: Dict[str, Any] = self.lines[line_id_]
            wavelength.append(line_["wavelength"])
            luminosity.append(line_["luminosity"][grid_point])
            continuum.append(line_["continuum"][grid_point])

        return Line(line_id, wavelength, luminosity, continuum)

    def get_lines(
        self,
        grid_point: Tuple[int],
        line_ids: Optional[List[str]] = None,
    ) -> LineCollection:
        """
        Create a LineCollection for multiple lines.

        Args:
            grid_point: A tuple of the grid point indices.
            line_ids: A list of lines, if None use all available lines.

        Returns:
            A LineCollection containing all requested lines at grid_point (or
            the requested subset).
        """
        # If no line ids are provided calculate all lines
        if line_ids is None:
            line_ids = self.available_lines

        # Line dictionary
        lines: Dict[str, Line] = {}

        # Loop over the provided lines creating the indivdual Line objects
        for line_id in line_ids:
            line: Line = self.get_line(grid_point, line_id)

            # Add to dictionary
            lines[line.id] = line

        # Create and return collection
        return LineCollection(lines)

    def plot_specific_ionising_lum(
        self,
        ion: str = "HI",
        hsize: float = 3.5,
        vsize: Optional[float] = None,
        cmap: str = "plasma",
        vmin: float = -1.0,
        vmax: float = -1.0,
        max_log10age: Optional[float] = None,
    ) -> Tuple[Figure, Axes]:
        """
        Make a simple plot of the specific ionising photon luminosity.

        The resulting figure will show the (log) specific ionsing photon
        luminosity as a function of (log) age and metallicity for a given grid
        and ion.

        Args:
           ion: The desired ion. Most grids only have HI and HII calculated by
                default.
            hsize: The horizontal size of the figure
            vsize: The vertical size of the figure
            cmap: Colourmap object or string defining the matplotlib colormap.
            vmin: Minimum specific ionising luminosity used in the colourmap
            vmax: Maximum specific ionising luminosity used in the colourmap
            max_log10age: The maximum log10(age) to plot

        Returns:
            The created figure containing the axes.
            The axis on which to plot.
        """
        # Define the axis coordinates
        left: float = 0.2
        height: float = 0.65
        bottom: float = 0.15
        width: float = 0.75

        # Scale the plot height if necessary
        if vsize is None:
            vsize = hsize * width / height

        # Create the figure
        fig: Figure = plt.figure(figsize=(hsize, vsize))

        # Create the axes
        ax: Axes = fig.add_axes((left, bottom, width, height))
        cax: Axes = fig.add_axes((left, bottom + height + 0.01, width, 0.05))

        # Create an index array
        y: NDArray[np.int32] = np.arange(len(self.metallicity), dtype=np.int32)

        # Select grid for specific ion
        log10_specific_ionising_lum: NDArray[np.float64]
        if hasattr(self, "log10_specific_ionising_lum"):
            log10_specific_ionising_lum = self.log10_specific_ionising_lum[ion]
        else:
            log10_specific_ionising_lum = self.log10Q[ion]

        # Truncate grid if max age provided
        ia_max: int
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
            if vmin < 0:
                vmin = 42.5
            if vmax < 0:
                vmax = 47.5
        else:
            if vmin < 0:
                vmin = float(np.min(log10_specific_ionising_lum))
            if vmax < 0:
                vmax = float(np.max(log10_specific_ionising_lum))

        # Plot the grid of log10_specific_ionising_lum
        extent: Tuple[float, float, float, float] = (
            self.log10age[0],
            self.log10age[ia_max],
            y[0] - 0.5,
            y[-1] + 0.5,
        )
        ax.imshow(
            log10_specific_ionising_lum.T,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        # Define the normalisation for the colorbar
        norm: Normalize = Normalize(vmin=vmin, vmax=vmax)
        cmapper: ScalarMappable = ScalarMappable(norm=norm, cmap=cmap)
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

    def get_delta_lambda(self) -> Tuple[NDArray[np.float64], ...]:
        """
        Calculate delta lambda.

        Returns:
            A tuple containing the list of wavelengths and delta lambda.
        """
        # Calculate delta lambda for each wavelength
        delta_lambda: NDArray[np.float64] = np.log10(self.lam[1:]) - np.log10(
            self.lam[:-1]
        )

        # Return tuple of wavelengths and delta lambda
        return self.lam, delta_lambda

    def get_sed(self, spectra_type: str) -> Sed:
        """
        Get the spectra grid as an Sed object.

        This enables grid wide use of Sed methods for flux, photometry,
        indices, ionising photons, etc.

        Args:
            spectra_type: The key of the spectra grid to extract as an Sed
                          object.

        Returns:
            The spectra grid as an Sed object.
        """
        return Sed(self.lam, self.spectra[spectra_type])

    def truncate_grid_lam(
        self,
        min_lam: unyt_quantity,
        max_lam: unyt_quantity,
    ) -> None:
        """
        Truncate the grid to a specific wavelength range.

        If out of range wavlengths are requested, the grid will be
        truncated to the nearest wavelength within the grid.

        Args:
            min_lam: The minimum wavelength to truncate the grid to.

            max_lam: The maximum wavelength to truncate the grid to.
        """
        # Get the indices of the wavelengths to keep
        okinds: NDArray[np.bool_] = np.logical_and(
            self.lam >= min_lam, self.lam <= max_lam
        )

        # Apply the mask to the grid wavelengths
        self.lam = self.lam[okinds]

        # Apply the mask to the spectra
        for spectra_type in self.available_spectra:
            self.spectra[spectra_type] = self.spectra[spectra_type][
                ..., okinds
            ]

    def unify_with_filters(self, filters: FilterCollection) -> None:
        """
        Unify the grid with a FilterCollection object.

        This will:
        - Find the Grid wavelengths at which transmission is non-zero.
        - Limit the Grid's spectra and wavelength array to where transmision
          is non-zero.
        - Interpolate the filter collection onto the Grid's new wavelength
          array.

        Args:
            filters: The FilterCollection object to unify with this grid.
        """
        # Get the minimum and maximum wavelengths with non-zero transmission
        min_lam: unyt_quantity
        max_lam: unyt_quantity
        min_lam, max_lam = filters.get_non_zero_lam_lims()

        # Ensure we have at least 1 element with 0 transmission to solve
        # any issues at the boundaries
        min_lam -= 10 * angstrom
        max_lam += 10 * angstrom

        # Truncate the grid to these wavelength limits
        self.truncate_grid_lam(min_lam, max_lam)

        # Interpolate the filters onto this new wavelength range
        filters.resample_filters(new_lam=self.lam)

    def animate_grid(
        self,
        show: bool = False,
        save_path: Optional[str] = None,
        fps: int = 30,
        spectra_type: str = "incident",
    ) -> FuncAnimation:
        """
        Create an animation of the grid stepping through wavelength.

        Each frame of the animation is a wavelength bin.

        Args:
            show: Should the animation be shown?
            save_path: Path to save the animation. If not specified, the
                       animation is not saved.
            fps: The number of frames per second in the output animation.
                 Default is 30 frames per second.

        Returns:
            The animation object.
        """
        fig: Figure
        ax: Axes
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))

        spectra: np.ndarray = self.spectra[spectra_type]

        vmin: float = 10**10
        vmax: float = np.percentile(spectra, 99.9)

        norm: LogNorm = LogNorm(vmin=vmin, vmax=vmax, clip=True)

        img = ax.imshow(
            spectra[:, :, 0],
            extent=(
                self.log10age.min(),
                self.log10age.max(),
                self.metallicity.min(),
                self.metallicity.max(),
            ),
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

        def update(i: int) -> Tuple[Artist,]:
            img.set_data(spectra[:, :, i])
            ax.set_title(f"Wavelength: {self.lam[i]:.2f}{self.lam.units}")
            return (img,)

        interval: int = int(1000 / fps)

        anim: FuncAnimation = FuncAnimation(
            fig, update, frames=self.lam.size, interval=interval, blit=False
        )

        if save_path is not None:
            anim.save(save_path, writer="imagemagick")

        if show:
            plt.show()
        else:
            plt.close(fig)

        return anim
