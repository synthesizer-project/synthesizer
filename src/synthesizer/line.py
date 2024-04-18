"""A module containing functionality for working with spectral lines.

The primary class is Line which holds information about an individual or
blended emission line, including its identification, wavelength, luminosity,
and the strength of the continuum. From these the equivalent width is
automatically calculated. Combined with a redshift and cosmology the flux can
also be calcualted.

A second class is LineCollection which holds a collection of Line objects and
provides additional functionality such as calcualting line ratios and diagrams
(e.g. BPT-NII, OHNO).

Several functions exist for obtaining line, ratio, and diagram labels for use
in plots etc.

"""

import numpy as np
from unyt import Angstrom, unyt_array

from synthesizer import exceptions, line_ratios
from synthesizer.conversions import lnu_to_llam
from synthesizer.units import Quantity


def vacuum_to_air(wavelength):
    """
    A function for converting a vacuum wavelength into an air wavelength.

    Arguments:
        wavelength (float or unyt_array)
            A wavelength in air.

    Returns:
        wavelength (unyt_array)
            A wavelength in vacuum.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # calculate wavelenegth squared for simplicty
    wave2 = wavelength.to("Angstrom").value ** 2.0

    # calcualte conversion factor
    conversion = (
        1.0 + 2.735182e-4 + 131.4182 / wave2 + 2.76249e8 / (wave2**2.0)
    )

    return wavelength / conversion


def air_to_vacuum(wavelength):
    """
    A function for converting an air wavelength into a vacuum wavelength.

    Arguments
        wavelength (float or unyt_array)
            A standard wavelength.

    Returns
        wavelength (unyt_array)
            A wavelength in vacuum.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # Convert to wavenumber squared
    sigma2 = (1.0e4 / wavelength.to("Angstrom").value) ** 2.0

    # Compute conversion factor
    conversion = (
        1.0
        + 6.4328e-5
        + 2.94981e-2 / (146.0 - sigma2)
        + 2.5540e-4 / (41.0 - sigma2)
    )

    return wavelength * conversion


def standard_to_vacuum(wavelength):
    """
    A function for converting a standard wavelength into a vacuum wavelength.

    Standard wavelengths are defined in vacuum at <2000A and air at >= 2000A.

    Arguments
        wavelength (float or unyt_array)
            A standard wavelength.

    Returns
        wavelength (unyt_array)
            A wavelength in vacuum.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # if wavelength is < 2000A simply return since no change required.
    if wavelength <= 2000.0 * Angstrom:
        return wavelength

    # otherwise conver to vacuum
    else:
        return air_to_vacuum(wavelength)


def vacuum_to_standard(wavelength):
    """
    A function for converting a vacuum wavelength into a standard wavelength.

    Standard wavelengths are defined in vacuum at <2000A and air at >= 2000A.

    Arguments
        wavelength (float or unyt_array)
            A vacuum wavelength.

    Returns
        wavelength (unyt_array)
            A standard wavelength.
    """

    # if wavelength is not a unyt_array conver to one assume unit is Anstrom.
    if not isinstance(wavelength, unyt_array):
        wavelength *= Angstrom

    # if wavelength is < 2000A simply return since no change required.
    if wavelength <= 2000.0 * Angstrom:
        return wavelength

    # otherwise conver to vacuum
    else:
        return vacuum_to_air(wavelength)


def get_line_id(id):
    """
    A function for converting a line id possibly represented as a list to
    a single string.

    Arguments
        id (str, list, tuple)
            a str, list, or tuple containing the id(s) of the lines

    Returns
        id (str)
            string representation of the id
    """

    if isinstance(id, list):
        return ",".join(id)
    else:
        return id


def get_line_label(line_id):
    """
    Get a line label for a given line_id, ratio, or diagram. Where the line_id
    is one of several predifined lines in line_ratios.line_labels this label
    is used, otherwise the label is constructed from the line_id.

    Argumnents
        line_id (str or list)
            The line_id either as a list of individual lines or a string. If
            provided as a list this is automatically converted to a single
            string so it can be used as a key.

    Returns
        line_label (str)
            A nicely formatted line label.
    """

    # if the line_id is a list (denoting a doublet or higher)
    if isinstance(line_id, list):
        line_id = ",".join(line_id)

    if line_id in line_ratios.line_labels.keys():
        line_label = line_ratios.line_labels[line_id]
    else:
        line_id = line_id.split(",")
        line_labels = []
        for line_id_ in line_id:
            # get the element, ion, and wavelength
            element, ion, wavelength = line_id_.split(" ")

            # extract unit and convert to latex str
            unit = wavelength[-1]

            if unit == "A":
                unit = r"\AA"
            if unit == "m":
                unit = r"\mu m"
            wavelength = wavelength[:-1] + unit

            line_labels.append(
                f"{element}{get_roman_numeral(int(ion))}{wavelength}"
            )

        line_label = "+".join(line_labels)

    return line_label


def flatten_linelist(list_to_flatten):
    """
    Flatten a mixed list of lists and strings and remove duplicates.

    Used when converting a line list which may contain single lines
    and doublets.

    Args:
        list_to_flatten (list)
            list containing lists and/or strings and integers

    Returns:
        (list)
            flattened list
    """
    flattened_list = []
    for lst in list_to_flatten:
        if isinstance(lst, list) or isinstance(lst, tuple):
            for ll in lst:
                flattened_list.append(ll)

        elif isinstance(lst, str):
            # If the line is a doublet, resolve it and add each line
            # individually
            if len(lst.split(",")) > 1:
                flattened_list += lst.split(",")
            else:
                flattened_list.append(lst)

        else:
            raise Exception(
                (
                    "Unrecognised type provided. Please provide"
                    "a list of lists and strings"
                )
            )

    return list(set(flattened_list))


def get_ratio_label(ratio_id):
    """
    Get a label for a given ratio_id.

    Arguments:
        ratio_id (str)
            The ratio identificantion, e.g. R23.

    Returns:
        label (str)
            A string representation of the label.
    """

    # get the list of lines for a given ratio_id

    # if the id is a string get the lines from the line_ratios sub-module
    if isinstance(ratio_id, str):
        ratio_line_ids = line_ratios.ratios[ratio_id]
    if isinstance(ratio_id, list):
        ratio_line_ids = ratio_id

    numerator = get_line_label(ratio_line_ids[0])
    denominator = get_line_label(ratio_line_ids[1])
    label = f"{numerator}/{denominator}"

    return label


def get_diagram_labels(diagram_id):
    """
    Get a x and y labels for a given diagram_id

    Arguments:
        diagram_id (str)
            The diagram identificantion, e.g. OHNO.

    Returns:
        xlabel (str)
            A string representation of the x-label.
        ylabel (str)
            A string representation of the y-label.
    """

    # get the list of lines for a given ratio_id
    diagram_line_ids = line_ratios.diagrams[diagram_id]
    xlabel = get_ratio_label(diagram_line_ids[0])
    ylabel = get_ratio_label(diagram_line_ids[1])

    return xlabel, ylabel


def get_roman_numeral(number):
    """
    Function to convert an integer into a roman numeral str.

    Used for renaming emission lines from the cloudy defaults.

    Arguments:
        number (int)
            The number to convert into a roman numeral.

    Returns:
        number_representation (str)
            String reprensentation of the roman numeral.
    """

    num = [1, 4, 5, 9, 10, 40, 50, 90, 100, 400, 500, 900, 1000]
    sym = [
        "I",
        "IV",
        "V",
        "IX",
        "X",
        "XL",
        "L",
        "XC",
        "C",
        "CD",
        "D",
        "CM",
        "M",
    ]
    i = 12

    roman = ""
    while number:
        div = number // num[i]
        number %= num[i]

        while div:
            roman += sym[i]
            div -= 1
        i -= 1
    return roman


def get_bpt_kewley01(logNII_Ha):
    """BPT-NII demarcations from Kewley+2001

    Kewley+03: https://arxiv.org/abs/astro-ph/0106324
    Demarcation defined by:
        log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.47) + 1.19

    Arguments:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line

    Returns:
        array
            Corresponding log([OIII]/Hb) ratio array
    """

    return 0.61 / (logNII_Ha - 0.47) + 1.19


def get_bpt_kauffman03(logNII_Ha):
    """BPT-NII demarcations from Kauffman+2003

    Kauffman+03: https://arxiv.org/abs/astro-ph/0304239
    Demarcation defined by:
        log([OIII]/Hb) = 0.61 / (log([NII]/Ha) - 0.05) + 1.3

    Args:
        logNII_Ha (array)
            Array of log([NII]/Halpha) values to give the
            SF-AGN demarcation line

    Returns:
        array
            Corresponding log([OIII]/Hb) ratio array
    """

    return 0.61 / (logNII_Ha - 0.05) + 1.3


class LineCollection:
    """
    A class holding a collection of emission lines. This enables additional
    functionality such as quickly calculating line ratios or line diagrams.

    Arguments
        lines (dictionary of Line objects)
            A dictionary of line objects.

    Methods

    """

    def __init__(self, lines):
        """
        Initialise LineCollection.

        Arguments:
            lines (dict)
                A dictionary of synthesizer.line.Line objects.
        """

        # dictionary of synthesizer.line.Line objects.
        self.lines = lines

        # create an array of line_ids
        self.line_ids = np.array(list(self.lines.keys()))

        # Atrributes to enable looping
        self._current_ind = 0
        self.nlines = len(self.line_ids)

        # create list of line wavelengths
        self.wavelengths = (
            np.array(
                [
                    line.wavelength.to("Angstrom").value
                    for line in self.lines.values()
                ]
            )
            * Angstrom
        )

        # get the arguments that would sort wavelength
        sorted_arguments = np.argsort(self.wavelengths)

        # sort the line_ids and wavelengths
        self.line_ids = self.line_ids[sorted_arguments]
        self.wavelengths = self.wavelengths[sorted_arguments]

        # include line ratio and diagram definitions dataclass
        self.line_ratios = line_ratios

        # create list of available line ratios
        self.available_ratios = []
        for ratio_id, ratio in self.line_ratios.ratios.items():
            # flatten line ratio list
            ratio_line_ids = [x for xs in ratio for x in xs]

            # check if line ratio is available
            if set(ratio_line_ids).issubset(self.line_ids):
                self.available_ratios.append(ratio_id)

        # create list of available line diagnostics
        self.available_diagrams = []
        for diagram_id, diagram in self.line_ratios.diagrams.items():
            # flatten line ratio list
            diagram_line_ids = [x for xs in diagram[0] for x in xs] + [
                x for xs in diagram[1] for x in xs
            ]

            # check if line ratio is available
            if set(diagram_line_ids).issubset(self.line_ids):
                self.available_diagrams.append(diagram_id)

    def __getitem__(self, line_id):
        """
        Simply returns one particular line from the collection.

        Returns:
            line (synthesizer.line.Line)
                A synthesizer.line.Line object.
        """

        return self.lines[line_id]

    def __str__(self):
        """
        Function to print a basic summary of the LineCollection object.

        Returns a string containing the id, wavelength, luminosity,
        equivalent width, and flux if generated.

        Returns:
            summary (str)
                Summary string containing the total mass formed and
                lists of the available SEDs, lines, and images.
        """

        # Set up string for printing
        summary = ""

        # Add the content of the summary to the string to be printed
        summary += "-" * 10 + "\n"
        summary += "LINE COLLECTION\n"
        summary += f"number of lines: {len(self.line_ids)}\n"
        summary += f"lines: {self.line_ids}\n"
        summary += f"available ratios: {self.available_ratios}\n"
        summary += f"available diagrams: {self.available_diagrams}\n"
        summary += "-" * 10

        return summary

    def __iter__(self):
        """
        Overload iteration to allow simple looping over Line objects,
        combined with __next__ this enables for l in LineCollection syntax
        """
        return self

    def __next__(self):
        """
        Overload iteration to allow simple looping over Line objects,
        combined with __iter__ this enables for l in LineCollection syntax
        """

        # Check we haven't finished
        if self._current_ind >= self.nlines:
            self._current_ind = 0
            raise StopIteration
        else:
            # Increment index
            self._current_ind += 1

            # Return the filter
            return self.lines[self.line_ids[self._current_ind - 1]]

    def get_ratio_(self, ab):
        """
        Measure (and return) a line ratio

        Arguments:
            ab
                a list of lists of lines, e.g. [[l1,l2], [l3]]

        Returns:
            float
                a line ratio
        """

        a, b = ab

        return np.sum([self.lines[_line].luminosity for _line in a]) / np.sum(
            [self.lines[_line].luminosity for _line in b]
        )

    def get_ratio(self, ratio_id):
        """
        Measure (and return) a line ratio

        Arguments:
            ratio_id
                a ratio_id where the ratio lines are defined in line_ratios

        Returns:
            float
                a line ratio
        """

        ab = self.line_ratios.ratios[ratio_id]

        return self.get_ratio_(ab)

    def get_diagram(self, diagram_id):
        """
        Return a pair of line ratios for a given diagram_id (E.g. BPT)

        Arguments:
            diagram_id
                a diagram_id where the pairs of ratio lines are
                defined in line_ratios

        Returns:
            tuple (float)
                a pair of line ratios
        """
        ab, cd = self.line_ratios.diagrams[diagram_id]

        return self.get_ratio_(ab), self.get_ratio_(cd)

    def get_ratio_label(self, ratio_id):
        """
        Wrapper around get_ratio_label
        """

        return get_ratio_label(ratio_id)

    def get_diagram_labels(self, diagram_id):
        """
        Wrapper around get_ratio_label
        """

        return get_diagram_labels(diagram_id)


class Line:
    """
    A class representing a spectral line or set of lines (e.g. a doublet)

    Attributes
        lam
            wavelength of the line

    Methods

    """

    wavelength = Quantity()
    vacuum_wavelength = Quantity()
    continuum = Quantity()
    luminosity = Quantity()
    flux = Quantity()
    ew = Quantity()

    def __init__(self, id_, wavelength_, luminosity_, continuum_):
        self.id_ = id_

        """
        Initialise the Line object.

        Arguments:
            id_ (str or list)
                The id of the line or collection of lines. For doublets this
                can be a list with an entry for each contributing line, e.g.
                "S 2 6730.82A", "S 2 6716.44A".
            wavelength_ (float or list)
                The standard (not vacuum) wavelength of the line. For doublets
                this is a list with an entry for each contributing line.
            luminosity_ (float or list)
                The luminosity the line. For doublets this is a list with an
                entry for each contributing line.
            continuum_ (float or list)
                The continuum at the line. For doublets this is a list with an
                entry for each contributing line.
        """

        """ these are maintained because we may want to hold on
        to the individual lines of a doublet"""
        self.wavelength_ = wavelength_
        self.luminosity_ = luminosity_
        self.continuum_ = continuum_

        # get line_id as a single string instead of e.g. a list
        self.id = get_line_id(id_)

        # mean continuum value in units of erg/s/Hz
        self.continuum = np.mean(continuum_)

        # mean wavelength of the line in units of AA
        self.wavelength = np.mean(wavelength_)

        # calculate the vacuum wavelength.
        self.vacuum_wavelength = air_to_vacuum(self.wavelength)

        # total luminosity of the line in units of erg/s/Hz
        self.luminosity = np.sum(luminosity_)

        # line flux in erg/s/cm2, generated by method
        self.flux = None

        # continuum at line wavelength, erg/s/AA
        self.continuum_lam = lnu_to_llam(self.wavelength, self.continuum)
        self.equivalent_width = self.luminosity / self.continuum_lam  # AA

        # element
        self.element = self.id.split(" ")[0]

    def __str__(self):
        """Function to print a basic summary of the Line object.

        Returns a string containing the id, wavelength, luminosity,
        equivalent width, and flux if generated.

        Returns:
            summary (str)
                Summary string containing the total mass formed and
                lists of the available SEDs, lines, and images.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 10 + "\n"
        pstr += f"SUMMARY OF {self.id}" + "\n"
        pstr += f"wavelength: {self.wavelength:.1f}" + "\n"
        pstr += (
            f"log10(luminosity/{self.luminosity.units}): "
            f"{np.log10(self.luminosity):.2f}\n"
        )
        pstr += f"equivalent width: {self.equivalent_width:.0f}" + "\n"
        if self._flux:
            pstr += f"log10(flux/{self.flux.units}): {np.log10(self.flux):.2f}"
        pstr += "-" * 10

        return pstr

    def __add__(self, second_line):
        """
        Function allowing adding of two Line objects together. This should
        NOT be used to add different lines together.

        Returns
            (synthesizer.line.Line)
                New instance of synthesizer.line.Line
        """

        if second_line.id == self.id:
            return Line(
                self.id,
                self._wavelength,
                self._luminosity + second_line._luminosity,
                self._continuum + second_line._continuum,
            )

        else:
            raise exceptions.InconsistentAddition(
                "Wavelength grids must be identical"
            )

    def get_flux(self, cosmo, z):
        """
        Calculate the line flux in units of erg/s/cm2

        Returns the line flux and (optionally) updates the line object.

        Arguments:
            cosmo (astropy.cosmology.)
                Astropy cosmology object.

            z (float)
                The redshift.

        Returns:
            flux (float)
                Flux of the line in units of erg/s/cm2 by default.
        """

        luminosity_distance = (
            cosmo.luminosity_distance(z).to("cm").value
        )  # the luminosity distance in cm

        self.flux = self._luminosity / (4 * np.pi * luminosity_distance**2)

        return self.flux
