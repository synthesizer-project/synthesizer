

import numpy as np
from dataclasses import dataclass
from .units import Quantity
from .utils import fnu_to_flam
from . import exceptions


interesting_lines = [
    'H 1 1215.67A', # LyA
    'H 1 6564.62A', # Ha
    'H 1 4862.69A', # Hb
    'H 1 4341.68A', # Hg
    'H 1 4102.89A', # Hd
    'H 1 3971.19A',
    'H 1 3890.15A',
    'H 1 3836.47A',
    'H 1 1.87561m', # Pa
    'H 1 1.28215m', # Pb
    'H 1 1.09410m', # Pg
    'H 1 1.00521m', # Pd
    'H 1 9548.54A',
    'H 1 9231.50A',
    'H 1 4.05224m', # Bra
    'H 1 2.62585m', # Brb
    'H 1 2.16611m', # Brg
    'H 1 1.94507m', # Brd
    'H 1 1.81790m',
    'Ar 3 7135.79A',
    'Ar 3 7751.11A',
    'C 2 1037.02A',
    'C 2 2325.40A',
    'C 2 2326.93A',
    'C 3 1908.73A',
    'C 3 1906.68A',
    'Fe 2 2382.04A',
    'Fe 2 2625.67A',
    'Fe 3 4881.12A',
    'Fe 3 4658.01A',
    'Fe 3 5270.40A',
    'Fe 3 4701.62A',
    'Ne 3 3868.76A',
    'Ne 3 3967.47A',
    'O 1 6300.30A',
    'O 2 2470.34A',
    'O 2 3728.81A',
    'O 2 3726.03A',
    'O 3 4958.91A',
    'O 3 5006.84A',
    'O 3 4363.21A',
    'O 3 1666.15A',
    'S 2 6730.82A',
    'S 2 6716.44A',
    'S 3 9068.62A',
    'S 3 9530.62A',
    'S 3 6312.06A',
    'Si 2 1265.00A',
    'Si 2 1197.39A',
    'Si 2 1194.50A',
    'Si 3 1892.03A',
    'Si 3 1882.71A',
    'Si 3 1206.50A',
    'Al 2 2660.35A',
    'Ca 2 7291.47A',
    'Cl 3 5517.71A',
    'Fe 4 3094.96A',
    'Fe 4 2835.74A',
    'Fe 4 2829.36A',
    'Fe 4 2567.61A',
    'Mg 2 2802.71A',
    'Mg 2 2795.53A',
    'N 2 6548.05A',
    'N 2 6583.45A',
    'N 3 991.511A',
    'Si 2 1179.59A',
]



def get_line_id(id):
    """
    A class representing a spectral line or set of lines (e.g. a doublet)

    Arguments
    ----------
    id : str, list, tuple
        a str, list, or tuple containing the id(s) of the lines

    Returns
    -------
    string
        string representation of the id

    """'H 1 6564.62A'

    if isinstance(id, list):
        return ','.join(id)
    else:
        return id


def get_roman_numeral(number):
    """
    Function to convert an integer into a roman numeral str.

    Used for renaming emission lines from the cloudy defaults.

    Returns
    ---------
    str
        string reprensentation of the roman numeral
    """

    num = [1, 4, 5, 9, 10, 40, 50, 90,
           100, 400, 500, 900, 1000]
    sym = ["I", "IV", "V", "IX", "X", "XL",
           "L", "XC", "C", "CD", "D", "CM", "M"]
    i = 12

    roman = ''
    while number:
        div = number // num[i]
        number %= num[i]

        while div:
            roman += sym[i]
            div -= 1
        i -= 1
    return roman


def get_fancy_line_id(id):
    """
    Function to get a nicer line id, e.g. "O 3 5008.24A" -> "OIII5008"

    Returns
    ---------
    str
        the fancy line id
    """

    element, ion, wavelength = id.split(' ')

    wavelength = float(wavelength[:-1])

    return f'{element}{get_roman_numeral(int(ion))}{wavelength: .4g}'


class LineRatios:

    """
    A dataclass holding useful line ratios (e.g. R23) and diagrams (pairs of ratios), e.g. BPT.
    """
    # short-hand

    
    def __init__(self):

        Hb = 'H 1 4862.69A'
        Ha = 'H 1 6564.62A'

        # define common line ratios
        self.ratios = {}
        self.ratios['BalmerDecrement'] = [[Ha], [Hb]] # Balmer decrement, should be ~2.86 for dust free
        self.ratios['N2'] = N2 = [['N 2 6583.45A'], [Ha]] #  add reference
        self.ratios['S2'] = [['S 2 6730.82A', 'S 2 6716.44A'], [Ha]]  #  add reference
        self.ratios['O1'] = [['O 1 6300.30A'], [Ha]]  #  add reference
        self.ratios['O2'] = O2 = ['O 2 3728.81A', 'O 2 3726.03A']
        self.ratios['O3'] = O3 = ['O 3 4958.91A', 'O 3 5006.84A']
        self.ratios['R2'] = [['O 2 3728.81A'], [Hb]]  #  add reference
        self.ratios['R3'] = R3 = [['O 3 5006.84A'], [Hb]]  #  add reference
        self.ratios['R23'] = [O3+O2, [Hb]]  #  add reference
        self.ratios['O32'] = [['O 3 5006.84A'], ['O 2 3728.81A']]  #  add reference
        self.ratios['Ne3O2'] = [['Ne 3 3967.47A'], ['O 2 3728.81A']]  #  add reference

        self.available_ratios = tuple(self.ratios.keys())

        # define common line diagnostics (i.e. pairs of line ratios)
        self.diagrams = {}
        self.diagrams['OHNO'] = [R3, [['Ne 3 3967.47A'], O2]]  #  add reference
        self.diagrams['BPT-NII'] = [N2, R3]  #  add reference
        # diagrams['VO78'] = [[], []]
        # diagrams['unVO78'] = [[], []]

        self.available_diagrams = tuple(self.diagrams.keys())

    def get_ratio_label_(self, ab, fancy=False):
        """
        Get a line ratio label

        Arguments
        -------
        ab
            a list of lists of lines, e.g. [[l1,l2], [l3]]
        fancy
            flag to return fancy label instead

        Returns
        -------
        str
            a label
        """

        a, b = ab

        if fancy:
            a = map(get_fancy_line_id, a)
            b = map(get_fancy_line_id, b)

        return f"({'+'.join(a)})/({'+'.join(b)})"

    def get_ratio_label(self, ratio_id, fancy=False):
        """
        Get a line ratio label

        Arguments
        -------
        ratio_id
            a ratio_id where the ratio lines are defined in LineRatios

        Returns
        -------
        str
            a label
        """

        ab = self.ratios[ratio_id]

        return f'{ratio_id}={self.get_ratio_label_(ab, fancy = fancy)}'


class LineCollection:

    """
    A class holding a collection of emission lines

    Attributes
    ----------
    lines : dictionary of Line objects

    Methods
    -------

    """

    def __init__(self, lines):

        self.lines = lines
        self.line_ids = list(self.lines.keys())

        # these should be filtered to only show ones that are available for the availalbe line_ids

        self.lineratios = LineRatios()

        self.available_ratios = self.lineratios.available_ratios
        self.available_diagrams = self.lineratios.available_diagrams

    def __getitem__(self, line_id):

        return self.lines[line_id]

    def __str__(self):
        """Function to print a basic summary of the LineCollection object.

        Returns a string containing the id, wavelength, luminosity, equivalent width, and flux if generated.

        Returns
        -------
        str
            Summary string containing the total mass formed and lists of the available SEDs, lines, and images.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-"*10 + "\n"
        pstr += f"LINE COLLECTION\n"
        pstr += f"lines: {self.line_ids}\n"
        pstr += f"available ratios: {self.available_ratios}\n"
        pstr += f"available diagrams: {self.available_diagrams}\n"
        pstr += "-"*10

        return pstr

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

        return np.sum([self.lines[l].luminosity for l in a]) / \
            np.sum([self.lines[l].luminosity for l in b])

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

    def get_ratio_label_(self, ab, fancy=False):
        """
        Get a line ratio label

        Arguments
        -------
        ab
            a list of lists of lines, e.g. [[l1,l2], [l3]]
        fancy
            flag to return fancy label instead

        Returns
        -------
        str
            a label
        """

        a, b = ab

        if fancy:
            a = map(get_fancy_line_id, a)
            b = map(get_fancy_line_id, b)

        return f"({','.join(a)})/({','.join(b)})"

    def get_ratio_label(self, ratio_id, fancy=False):
        """
        Get a line ratio label

        Arguments
        -------
        ratio_id
            a ratio_id where the ratio lines are defined in LineRatios

        Returns
        -------
        str
            a label
        """
        ab = self.lineratios.ratios[ratio_id]

        return f'{ratio_id}={self.get_ratio_label_(ab, fancy = fancy)}'

    def get_diagram(self, diagram_id):
        """
        Return a pair of line ratios for a given diagram_id (E.g. BPT)

        Arguments
        -------
        ratdiagram_idio_id
            a diagram_id where the pairs of ratio lines are defined in LineRatios

        Returns
        -------
        tuple (float)
            a pair of line ratios
        """
        ab, cd = self.lineratios.diagrams[diagram_id]

        return self.get_ratio_(ab), self.get_ratio_(cd)

    def get_diagram_label(self, diagram_id, fancy=False):
        """
        Get a line ratio label

        Arguments
        -------
        ab
            a list of lists of lines, e.g. [[l1,l2], [l3]]

        Returns
        -------
        str
            a label
        """
        ab, cd = self.lineratios.diagrams[diagram_id]

        return self.get_ratio_label_(ab, fancy=fancy), self.get_ratio_label_(cd, fancy=fancy)


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

        # --- these are maintained because we may want to hold on to the individual lines of a doublet
        self.wavelength_ = wavelength_
        self.luminosity_ = luminosity_
        self.continuum_ = continuum_

        self.id = get_line_id(id_)
        self.continuum = np.mean(continuum_)  #  mean continuum value in units of erg/s/Hz
        self.wavelength = np.mean(wavelength_)  # mean wavelength of the line in units of AA
        self.luminosity = np.sum(luminosity_)  # total luminosity of the line in units of erg/s/Hz
        self.flux = None  # line flux in erg/s/cm2, generated by method

        # continuum at line wavelength, erg/s/AA
        self._continuum_lam = fnu_to_flam(self._wavelength, self._continuum)
        self.ew = self._luminosity / self._continuum_lam  # AA

    def __str__(self):
        """Function to print a basic summary of the Line object.

        Returns a string containing the id, wavelength, luminosity, equivalent width, and flux if generated.

        Returns
        -------
        str
            Summary string containing the total mass formed and lists of the available SEDs, lines, and images.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-"*10 + "\n"
        pstr += f"SUMMARY OF {self.id}" + "\n"
        pstr += f"wavelength: {self.wavelength:.1f}" + "\n"
        pstr += f"log10(luminosity/{self.luminosity.units}): {np.log10(self.luminosity):.2f}" + "\n"
        pstr += f"equivalent width: {self.ew:.0f}" + "\n"
        if self._flux:
            pstr += f"log10(flux/{self.flux.units}): {np.log10(self.flux):.2f}"
        pstr += "-"*10

        return pstr

    def __add__(self, second_line):
        """
        Function allowing adding of two Line objects together. This should NOT be used to add different lines together.

        Returns
        -------
        obj (Line)
            New instance of Line
        """

        if second_line.id == self.id:

            return Line(self.id, self._wavelength, self._luminosity + second_line._luminosity, self._continuum + second_line._continuum)

        else:

            exceptions.InconsistentAddition('Wavelength grids must be identical')

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

        luminosity_distance = cosmo.luminosity_distance(
            z).to('cm').value  # the luminosity distance in cm

        self.flux = self._luminosity / (4 * np.pi * luminosity_distance**2)

        return self.flux
