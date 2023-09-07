"""
Create a Grid object
"""

import os
import numpy as np
import h5py

from . import __file__ as filepath
from .sed import Sed
from .line import Line, LineCollection


from collections.abc import Iterable


def get_available_lines(grid_name, grid_dir, include_wavelengths=False):
    """Get a list of the lines available to a grid

    Parameters
    ----------
    grid_name : str
        list containing lists and/or strings and integers

    grid_dir : str
        path to grid

    Returns
    -------
    list
        list of lines
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
    """Flatten a mixed list of lists and strings and remove duplicates

    Flattens a mixed list of lists and strings. Used when converting a desired line list which may contain single lines and doublets.

    Parameters
    ----------
    list : list
        list containing lists and/or strings and integers


    Returns
    -------
    list
        flattend list
    """

    flattend_list = []
    for l in list_to_flatten:
        if isinstance(l, list) or isinstance(l, tuple):
            for ll in l:
                flattend_list.append(ll)

        elif isinstance(l, str):
            # --- if the line is a doublet resolve it and add each line individually
            if len(l.split(",")) > 1:
                flattend_list += l.split(",")
            else:
                flattend_list.append(l)

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
    The Grid class, containing attributes and methods for reading and manipulating spectral grids

    Attributes
    ----------


    Methods
    -------

    """

    def __init__(
        self,
        grid_name,
        grid_dir=None,
        verbose=False,
        read_spectra=True,
        read_lines=False,
    ):
        if not grid_dir:
            grid_dir = os.path.join(os.path.dirname(filepath), "data/grids")

        self.grid_dir = grid_dir
        self.grid_name = grid_name
        self.grid_filename = f"{self.grid_dir}/{self.grid_name}.hdf5"

        self.read_lines = read_lines
        self.read_spectra = read_spectra  #  not used

        self.spectra = None
        self.lines = None

        # convert line list into flattend list and remove duplicates
        if isinstance(read_lines, list):
            read_lines = flatten_linelist(read_lines)

        # get basic info of the grid
        with h5py.File(self.grid_filename, "r") as hf:
            self.parameters = {k: v for k, v in hf.attrs.items()}

            # get list of axes
            self.axes = list(hf.attrs["axes"])

            for axis in self.axes:
                setattr(self, axis, hf["axes"][axis][:])

            # number of axes
            self.naxes = len(self.axes)

            if "log10Q" in hf.keys():
                self.log10Q = {}
                for ion in hf["log10Q"].keys():
                    self.log10Q[ion] = hf["log10Q"][ion][:]

        # read in spectra
        if read_spectra:
            self.get_spectra()

        # read in lines
        if read_lines:
            self.get_lines()

    def __str__(self):
        """
        Function to print a basic summary of the Grid object.

        Returns
        -------
        str

        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 30 + "\n"
        pstr += f"SUMMARY OF GRID" + "\n"
        for axis in self.axes:
            pstr += f"{axis}: {getattr(self, axis)} \n"
        for k, v in self.parameters.items():
            pstr += f"{k}: {v} \n"
        if self.spectra:
            pstr += f"spectra: {list(self.spectra.keys())}\n"
        if self.lines:
            pstr += f"lines: {list(self.lines.keys())}\n"
        pstr += "-" * 30 + "\n"

        return pstr

    def get_spectra(self):
        """
        Function to read in spectra from the HDF5 grid

        Returns
        -------
        str

        """

        self.spectra = {}

        with h5py.File(f"{self.grid_dir}/{self.grid_name}.hdf5", "r") as hf:
            # get list of available spectra
            self.spec_names = list(hf["spectra"].keys())
            self.spec_names.remove("wavelength")
            if "normalisation" in self.spec_names:
                self.spec_names.remove("normalisation")

            for spec_name in self.spec_names:
                self.lam = hf["spectra/wavelength"][:]
                self.nu = 3e8 / (self.lam * 1e-10)  # used?

                self.spectra[spec_name] = hf["spectra"][spec_name][:]

                # if incident is one of the spectra also add a spectra called "stellar"
                if spec_name == "incident":
                    self.spectra["stellar"] = self.spectra[spec_name]

        """ if full cloudy grid available calculate
        some other spectra for convenience """
        if "linecont" in self.spec_names:
            self.spectra["total"] = (
                self.spectra["transmitted"] + self.spectra["nebular"]
            )  #  assumes fesc = 0

            self.spectra["nebular_continuum"] = (
                self.spectra["nebular"] - self.spectra["linecont"]
            )

    def get_lines(self):
        """
        Function to read in lines from the HDF5 grid

        Returns
        -------
        str

        """

        self.lines = {}

        if isinstance(self.read_lines, list):
            self.line_list = flatten_linelist(self.read_lines)
        else:
            self.line_list = get_available_lines(self.grid_name, self.grid_dir)

        with h5py.File(f"{self.grid_dir}/{self.grid_name}.hdf5", "r") as hf:
            for line in self.line_list:
                self.lines[line] = {}
                self.lines[line]["wavelength"] = hf["lines"][line].attrs[
                    "wavelength"
                ]  # angstrom
                self.lines[line]["luminosity"] = hf["lines"][line]["luminosity"][:]
                self.lines[line]["continuum"] = hf["lines"][line]["continuum"][:]

    def get_nearest_index(self, value, array):
        """
        Simple function for calculating the closest index in an array for a given value

        Parameters
        ----------
        value : float
            The target value

        array : nn.ndarray
            The array to search

        Returns
        -------
        int
             The index of the closet point in the grid (array)
        """

        return (np.abs(array - value)).argmin()

    def get_grid_point(self, values):
        """
        Identify the nearest grid point for a tuple of values

        Parameters
        ----------
        value : float
            The target value

        Returns
        -------
        int
             The index of the closet point in the grid (array)
        """

        return tuple(
            [
                self.get_nearest_index(value, getattr(self, axis))
                for axis, value in zip(self.axes, values)
            ]
        )

    def get_nearest(self, value, array):
        """
        Simple function for calculating the closest index in an array for a given value

        Parameters
        ----------
        value : float
            The target value

        array : nn.ndarray
            The array to search

        Returns
        -------
        int
             The index of the closet point in the grid (array)
        """

        idx = self.get_nearest_index(value, array)

        return idx, array[idx]

    def get_sed(self, grid_point, spec_name="incident"):
        """
        Returns an Sed object for a given spectra type for a given grid point.

        Parameters
        ----------
        grid_point: tuple (int)
            A tuple of the grid point indices

        Returns
        -------
        obj (Sed)
             An Sed object at the defined grid point
        """

        if len(grid_point) != self.naxes:
            # throw exception
            print("grid point tuple should have same shape as grid")
            pass

        return Sed(self.lam, lnu=self.spectra[spec_name][grid_point])

    def get_line_info(self, line_id, grid_point):
        """
        Returns a Line object for a given line_id and grid_point

        Parameters
        ----------
        grid_point: tuple (int)
            A tuple of the grid point indices

        Returns
        -------
        obj (Line)
             A Line object at the defined grid point
        """

        if len(grid_point) != self.naxes:
            # throw exception
            print("grid point tuple should have same shape as grid")
            pass

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

    def get_lines_info(self, line_ids, grid_point):
        """
        Return a LineCollection object for a given line and metallicity/age index

        Parameters
        ----------
        grid_point: tuple (int)
            A tuple of the grid point indices

        Returns:
        ------------
        obj (Line)
        """

        # line dictionary
        lines = {}

        for line_id in line_ids:
            line = self.get_line_info(line_id, grid_point)

            # add to dictionary
            lines[line.id] = line

        # create collection
        line_collection = LineCollection(lines)

        # return collection
        return line_collection
