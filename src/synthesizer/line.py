"""A module containing functionality for working with spectral lines.

TODO: Complete doc string.

Example usage:
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from unyt import Angstrom

from synthesizer import exceptions
from synthesizer.conversions import lnu_to_llam
from synthesizer.units import Quantity


def get_line_id(id):
    """
    A class representing a spectral line or set of lines (e.g. a doublet)

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
    Get a line label for a given line_id, ratio, or diagram.
    """

    # dictionary of common line labels to use by default
    special_line_labels = {
        "O 2 3726.03A,O 2 3728.81A": "[OII]3726,3729",
        "H 1 4862.69A": r"H\beta",
        "O 3 4958.91A,O 3 5006.84A": "[OIII]4959,5007",
        "H 1 6564.62A": r"H\alpha",
        "O 3 5006.84A": "[OIII]5007",
        "N 2 6583.45A": "[NII]6583",
    }

    # if the line_id is a list (denoting a doublet or higher)
    if isinstance(line_id, list):
        line_id = ",".join(line_id)

    if line_id in special_line_labels.keys():
        line_label = special_line_labels[line_id]
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

    # if the id is a string get the lines from the LineRatios class
    if isinstance(ratio_id, str):
        ratio_line_ids = LineRatios.ratios[ratio_id]
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
    diagram_line_ids = LineRatios.diagrams[diagram_id]
    xlabel = get_ratio_label(diagram_line_ids[0])
    ylabel = get_ratio_label(diagram_line_ids[1])

    return xlabel, ylabel


def get_roman_numeral(number):
    """
    Function to convert an integer into a roman numeral str.

    Used for renaming emission lines from the cloudy defaults.

    Returns
    ---------
    str
        string reprensentation of the roman numeral
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

    Args:
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


@dataclass
class LineRatios:
    """
    A dataclass holding useful line ratios (e.g. R23) and
    diagrams (pairs of ratios), e.g. BPT.
    """

    # Shorthand for common lines
    O3b: str = "O 3 4958.91A"
    O3r: str = "O 3 5006.84A"
    O3: List[str] = field(
        default_factory=lambda: ["O 3 4958.91A", "O 3 5006.84A"]
    )
    O2b: str = "O 2 3726.03A"
    O2r: str = "O 2 3728.81A"
    O2: List[str] = field(
        default_factory=lambda: ["O 2 3726.03A", "O 2 3728.81A"]
    )
    Hb: str = "H 1 4862.69A"
    Ha: str = "H 1 6564.62A"

    # Balmer decrement, should be [2.79--2.86] (Te, ne, dependent)
    # for dust free
    ratios: Dict[str, List[List[str]]] = field(
        default_factory=lambda: {
            "BalmerDecrement": [[["H 1 6564.62A"], ["H 1 4862.69A"]]],
            "N2": [[["N 2 6583.45A"], ["H 1 6564.62A"]]],
            "S2": [[["S 2 6730.82A", "S 2 6716.44A"], ["H 1 6564.62A"]]],
            "O1": [[["O 1 6300.30A"], ["H 1 6564.62A"]]],
            "R2": [[["O 2 3726.03A"], ["H 1 4862.69A"]]],
            "R3": [[["O 3 5006.84A"], ["H 1 4862.69A"]]],
            "R23": [
                [
                    [
                        "O 3 4958.91A",
                        "O 3 5006.84A",
                        "O 2 3726.03A",
                        "O 2 3728.81A",
                    ],
                    ["H 1 4862.69A"],
                ]
            ],
            "O32": [[["O 3 5006.84A"], ["O 2 3726.03A"]]],
            "Ne3O2": [[["NE 3 3868.76A"], ["O 2 3726.03A"]]],
        }
    )

    diagrams: Dict[str, List[List[List[str]]]] = field(
        default_factory=lambda: {
            "OHNO": [
                [
                    ["O 3 5006.84A"],
                    ["NE 3 3868.76A"],
                    ["O 2 3726.03A", "O 2 3728.81A"],
                ]
            ],
            "BPT-NII": [
                [["N 2 6583.45A"], ["H 1 6564.62A"]],
                [["O 3 5006.84A"], ["H 1 4862.69A"]],
            ],
        }
    )

    def __post_init__(self):
        # Provide lists of what is included
        self.available_ratios: Tuple[str, ...] = tuple(self.ratios.keys())
        self.available_diagrams: Tuple[str, ...] = tuple(self.diagrams.keys())


class LineCollection:
    """
    A class holding a collection of emission lines

    Arguments
        lines (dictionary of Line objects)
            A dictionary of line objects.

    Methods

    """

    def __init__(self, lines):
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
        self.lineratios = LineRatios()

        # create list of available line ratios
        self.available_ratios = []
        for ratio_id, ratio in self.lineratios.ratios.items():
            # flatten line ratio list
            ratio_line_ids = [x for xs in ratio for x in xs]

            # check if line ratio is available
            if set(ratio_line_ids).issubset(self.line_ids):
                self.available_ratios.append(ratio_id)

        # create list of available line diagnostics
        self.available_diagrams = []
        for diagram_id, diagram in self.lineratios.diagrams.items():
            # flatten line ratio list
            diagram_line_ids = [x for xs in diagram[0] for x in xs] + [
                x for xs in diagram[1] for x in xs
            ]

            # check if line ratio is available
            if set(diagram_line_ids).issubset(self.line_ids):
                self.available_diagrams.append(diagram_id)

    def __getitem__(self, line_id):
        return self.lines[line_id]

    def __str__(self):
        """Function to print a basic summary of the LineCollection object.

        Returns a string containing the id, wavelength, luminosity,
        equivalent width, and flux if generated.

        Returns
        -------
        str
            Summary string containing the total mass formed and
            lists of the available SEDs, lines, and images.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 10 + "\n"
        pstr += "LINE COLLECTION\n"
        pstr += f"lines: {self.line_ids}\n"
        pstr += f"available ratios: {self.available_ratios}\n"
        pstr += f"available diagrams: {self.available_diagrams}\n"
        pstr += "-" * 10

        return pstr

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

        Arguments
        -------
        ab
            a list of lists of lines, e.g. [[l1,l2], [l3]]

        Returns
        -------
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

        Arguments
        -------
        ratio_id
            a ratio_id where the ratio lines are defined in LineRatios

        Returns
        -------
        float
            a line ratio
        """

        ab = self.lineratios.ratios[ratio_id]

        return self.get_ratio_(ab)

    def get_diagram(self, diagram_id):
        """
        Return a pair of line ratios for a given diagram_id (E.g. BPT)

        Arguments
        -------
        ratdiagram_idio_id
            a diagram_id where the pairs of ratio lines are
            defined in LineRatios

        Returns
        -------
        tuple (float)
            a pair of line ratios
        """
        ab, cd = self.lineratios.diagrams[diagram_id]

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
    ----------
    lam : wavelength of the line

    Methods
    -------

    """

    wavelength = Quantity()
    continuum = Quantity()
    luminosity = Quantity()
    flux = Quantity()
    ew = Quantity()

    def __init__(self, id_, wavelength_, luminosity_, continuum_):
        self.id_ = id_

        """ these are maintained because we may want to hold on
        to the individual lines of a doublet"""
        self.wavelength_ = wavelength_
        self.luminosity_ = luminosity_
        self.continuum_ = continuum_

        self.id = get_line_id(id_)
        self.continuum = np.mean(
            continuum_
        )  # mean continuum value in units of erg/s/Hz
        self.wavelength = np.mean(
            wavelength_
        )  # mean wavelength of the line in units of AA
        self.luminosity = np.sum(
            luminosity_
        )  # total luminosity of the line in units of erg/s/Hz
        self.flux = None  # line flux in erg/s/cm2, generated by method

        # continuum at line wavelength, erg/s/AA
        self.continuum_lam = lnu_to_llam(self.wavelength, self.continuum)
        self.equivalent_width = self.luminosity / self.continuum_lam  # AA

        # element
        self.element = self.id.split(" ")[0]

    def __str__(self):
        """Function to print a basic summary of the Line object.

        Returns a string containing the id, wavelength, luminosity,
        equivalent width, and flux if generated.

        Returns
        -------
        str
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
        -------
        obj (Line)
            New instance of Line
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
        """Calculate the line flux in units of erg/s/cm2

        Returns the line flux and (optionally) updates the line object.

        Parameters
        -------
        cosmo: obj
            Astropy cosmology object

        z: float
            Redshift

        Returns
        -------
        flux: float
            Flux of the line in units of erg/s/cm2
        """

        luminosity_distance = (
            cosmo.luminosity_distance(z).to("cm").value
        )  # the luminosity distance in cm

        self.flux = self._luminosity / (4 * np.pi * luminosity_distance**2)

        return self.flux
