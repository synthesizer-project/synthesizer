import os
import numpy as np
import h5py
from . import __file__ as filepath
from . import exceptions
from .sed import Sed
from .line import Line, LineCollection


def get_available_lines(grid_name, grid_dir, include_wavelengths=False):
    """Get a list of the lines available to a grid

    Arguments:
        grid_name (str)
            The name of the grid file.

        grid_dir (str)
            The directory to the grid file.

    Returns:
        (list)
            List of available lines
    """

    grid_filename = f"{grid_dir}/{grid_name}.hdf5"
    with h5py.File(grid_filename, "r") as hf:
        lines = list(hf["lines"].keys())

        if include_wavelengths:
            wavelengths = np.array(
                [hf["lines"][line].attrs["wavelength"] for line in lines]
            )
            return lines, wavelengths
        else:
            return lines


def flatten_linelist(list_to_flatten):
    """
    Flatten a mixed list of lists and strings and remove duplicates. Used when
    converting a desired line list which may contain single lines and doublets.

    Arguments:
        list_to_flatten (list)
            list containing lists and/or strings and integers

    Returns:
        (list)
            flattend list
    """

    flattend_list = []
    for lst in list_to_flatten:
        if isinstance(lst, list) or isinstance(lst, tuple):
            for ll in lst:
                flattend_list.append(ll)

        elif isinstance(lst, str):
            # if the line is a doublet resolve it and add each line
            # individually
            if len(lst.split(",")) > 1:
                flattend_list += lst.split(",")
            else:
                flattend_list.append(lst)

        else:
            # raise exception
            pass

    return list(set(flattend_list))


def parse_grid_id(grid_id):
    """
    This is used for parsing a grid ID to return the SPS model,
    version, and IMF
    """

    if len(grid_id.split("_")) == 2:
        sps_model_, imf_ = grid_id.split("_")
        cloudy = cloudy_model = ""

    if len(grid_id.split("_")) == 4:
        sps_model_, imf_, cloudy, cloudy_model = grid_id.split("_")

    if len(sps_model_.split("-")) == 1:
        sps_model = sps_model_.split("-")[0]
        sps_model_version = ""

    if len(sps_model_.split("-")) == 2:
        sps_model = sps_model_.split("-")[0]
        sps_model_version = sps_model_.split("-")[1]

    if len(sps_model_.split("-")) > 2:
        sps_model = sps_model_.split("-")[0]
        sps_model_version = "-".join(sps_model_.split("-")[1:])

    if len(imf_.split("-")) == 1:
        imf = imf_.split("-")[0]
        imf_hmc = ""

    if len(imf_.split("-")) == 2:
        imf = imf_.split("-")[0]
        imf_hmc = imf_.split("-")[1]

    if imf in ["chab", "chabrier03", "Chabrier03"]:
        imf = "Chabrier (2003)"
    if imf in ["kroupa"]:
        imf = "Kroupa (2003)"
    if imf in ["salpeter", "135all"]:
        imf = "Salpeter (1955)"
    if imf.isnumeric():
        imf = rf"$\alpha={float(imf)/100}$"

    return {
        "sps_model": sps_model,
        "sps_model_version": sps_model_version,
        "imf": imf,
        "imf_hmc": imf_hmc,
    }


class Grid:
    """
    The Grid class, containing attributes and methods for reading and
    manipulating spectral grids.

    Attributes:

        grid_name (str)
            The name of the grid file.

        grid_dir (str)
            The directory to the grid file.

        parameters (dict)
            A dictionary containing parameters used to create the grid.

        axes (list)
            A list of the axes.

        axes_values (dict)
            A dictionary containing the axes values. Note, these are also set
            as attributes, e.g. self.axes_values['log10age']==self.log10age.

        naxes (int)
            The number of axes.

        lam (numpy.ndarray)
            The wavelength grid.

        spectra (dict)
            A dictionary containing arrays for the different available spectra.
            Each array has shape (*axes.shape, len(lam))

        available_spectra (list)
            A list of the available spectra.

        lines (dict)
            A dictionary containing line quantities.

        available_lines (list)
            A list of the available lines.

        log10Q (dict)
            A dictionary containing log10Q arrays for each ions.

    Methods:

        get_nearest_index

        get_grid_point

        get_spectra

        get_line

        get_lines
    """

    def __init__(
        self,
        grid_name,
        grid_dir=None,
        read_spectra=True,
        read_lines=True,
    ):
        """
        Initialisation function.

        Arguments:
            grid_name (str)
                The name of the grid file.

            grid_dir (str)
                The directory to the grid file.

            read_spectra (bool)
                Flag specifying whether to read spectra.

            read_lines (bool)
                Flag specifying whether to read lines.

        """

        # if no grid_dir provided assume in <current path>/data/grids/
        # NOTE: not sure this is desired anymore.
        if not grid_dir:
            grid_dir = os.path.join(os.path.dirname(filepath), "data/grids")

        self.grid_dir = grid_dir
        self.grid_name = grid_name
        self.grid_filename = f"{self.grid_dir}/{self.grid_name}.hdf5"
        self.spectra = None
        self.lines = None
        self.available_spectra = []
        self.available_lines = []

        # get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            self.parameters = {k: v for k, v in hf.attrs.items()}

            # get list of axes
            self.axes = list(hf.attrs["axes"])

            # put the values of each axis in a dictionary
            self.axes_values = {axis: hf["axes"][axis][:] for axis
                                in self.axes}

            # set the values of each axis as an attribute
            # e.g. self.log10age == self.axes_values['log10age']
            for axis in self.axes:
                setattr(self, axis, self.axes_values[axis])

            # number of axes
            self.naxes = len(self.axes)

            # if log10Q is available set this as an attribute as well
            if "log10Q" in hf.keys():
                self.log10Q = {}
                for ion in hf["log10Q"].keys():
                    self.log10Q[ion] = hf["log10Q"][ion][:]

        # read in spectra
        if read_spectra:
            self.spectra = {}

            with h5py.File(f"{self.grid_dir}/{self.grid_name}.hdf5",
                           "r") as hf:

                # get list of available spectra
                spectra_ids = list(hf["spectra"].keys())

                # remove wavelength dataset
                spectra_ids.remove("wavelength")

                # remove normalisation dataset
                if "normalisation" in spectra_ids:
                    spectra_ids.remove("normalisation")

                for spectra_id in spectra_ids:
                    self.lam = hf["spectra/wavelength"][:]
                    self.spectra[spectra_id] = hf["spectra"][spectra_id][:]
            """
            If full cloudy grid available calculate some other spectra for
            convenience.
            """
            if "linecont" in self.spectra.keys():
                self.spectra["total"] = (
                    self.spectra["transmitted"] + self.spectra["nebular"]
                )

                self.spectra["nebular_continuum"] = (
                    self.spectra["nebular"] - self.spectra["linecont"]
                )

            # save list of available spectra
            self.available_spectra = list(self.spectra.keys())

        if read_lines is not False:
            """
            If read_lines is True read all available lines in the grid,
            otherwise if read_lines is a list just read the lines in the list.
            """

            self.lines = {}

            # if a list of lines is provided then only read lines in this list
            if isinstance(read_lines, list):
                read_lines = flatten_linelist(read_lines)
            # if a list isn't provided then use all available lines to the grid
            else:
                read_lines = get_available_lines(self.grid_name, self.grid_dir)

            with h5py.File(f"{self.grid_dir}/{self.grid_name}.hdf5",
                           "r") as hf:
                for line in read_lines:
                    self.lines[line] = {}
                    self.lines[line]["wavelength"] = hf["lines"][line].attrs[
                        "wavelength"]
                    self.lines[line]["luminosity"] = hf["lines"][line][
                        "luminosity"][:]
                    self.lines[line]["continuum"] = hf["lines"][line][
                        "continuum"][:]

            # save list of available lines
            self.available_lines = list(self.lines.keys())

    def __str__(self):
        """
        Function to print a basic summary of the Grid object.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 30 + "\n"
        pstr += "SUMMARY OF GRID" + "\n"
        for axis in self.axes:
            pstr += f"{axis}: {getattr(self, axis)} \n"
        for k, v in self.parameters.items():
            pstr += f"{k}: {v} \n"
        if self.spectra:
            pstr += f"available lines: {self.available_lines}\n"
        if self.lines:
            pstr += f"available spectra: {self.available_spectra}\n"
        pstr += "-" * 30 + "\n"

        return pstr

    def get_nearest_index(self, value, array):
        """
        Function for calculating the closest index in an array for a
        given value.

        TODO: This could be moved to utils?

        Arguments:
            value (float)
                The target value.

            array (np.ndarray)
                The array to search.

        Returns:
            int
                The index of the closet point in the grid (array)
        """

        return (np.abs(array - value)).argmin()

    def get_grid_point(self, values):

        """
        Function to identify the nearest grid point for a tuple of values.

        Arguments:
            values (tuple)
                The values for which we want the grid point. These have to be 
                in the same order as the axes.

        Returns:
            (tuple)
                A tuple of integers specifying the closest grid point.
        """

        return tuple(
            [
                self.get_nearest_index(value, getattr(self, axis))
                for axis, value in zip(self.axes, values)
            ]
        )

    def get_spectra(self, grid_point, spectra_id="incident"):
        """
        Function for creating an Sed object for a specific grid point.

        Arguments:
            grid_point (tuple)
                A tuple of integers specifying the closest grid point.
            spectra_id (str)
                The name of the spectra (in the grid) that is desired.

        Returns:
            sed (synthesizer.sed.Sed)
                A synthesizer Sed object
        """
        # throw exception if tline_id not in list of available lines
        if spectra_id not in self.available_spectra:
            raise exceptions.InconsistentParameter("""Provided spectra_id is not in
            list of available spectra.""")

        # throw exception if the grid_point has a different shape from the grid
        if len(grid_point) != self.naxes:
            raise exceptions.InconsistentParameter("""The grid_point tuple provided
            as an argument should have same shape as the grid.""")

        # TODO: throw an exception if grid point is outside grid bounds

        return Sed(self.lam, lnu=self.spectra[spectra_id][grid_point])

    def get_line(self, grid_point, line_id):
        """
        Function for creating a Line object for a given line_id and grid_point.

        Arguments:
            grid_point (tuple)
                A tuple of integers specifying the closest grid point.
            line_id (str)
                The id of the line.

        Returns:
            line (synthesizer.line.Line)
                A synthesizer Line object.
        """
        # throw exception if tline_id not in list of available lines
        if line_id not in self.available_lines:
            raise exceptions.InconsistentParameter("""Provided line_id is not in list
            of available lines.""")

        # throw exception if the grid_point has a different shape from the grid
        if len(grid_point) != self.naxes:
            raise exceptions.InconsistentParameter("""The grid_point tuple provided
            as an argument should have same shape as the grid.""")

        if type(line_id) is str:
            line_id = [line_id]

        wavelength = []
        luminosity = []
        continuum = []

        for line_id_ in line_id:
            line_ = self.lines[line_id_]
            wavelength.append(line_["wavelength"])
            luminosity.append(line_["luminosity"][grid_point])
            continuum.append(line_["continuum"][grid_point])

        return Line(line_id, wavelength, luminosity, continuum)

    def get_lines(self, grid_point, line_ids=None):
        """
        Function a LineCollection of multiple lines.

        Arguments:
            grid_point (tuple)
                A tuple of the grid point indices.
            line_ids (list)
                A list of lines, if None use all available lines.

        Returns:
            lines (lines.LineCollection)
        """

        # if no line ids provided calcualte all lines
        if line_ids is None:
            line_ids = self.available_lines

        # line dictionary
        lines = {}

        for line_id in line_ids:
            line = self.get_line(grid_point, line_id)

            # add to dictionary
            lines[line.id] = line

        # create and return collection
        return LineCollection(lines)
