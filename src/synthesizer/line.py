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
from unyt import Angstrom, Hz, angstrom, cm, erg, pc, s

from synthesizer import exceptions, line_ratios
from synthesizer.conversions import lnu_to_llam, standard_to_vacuum
from synthesizer.units import Quantity, accepts
from synthesizer.warnings import deprecation


def get_line_id(id):
    """
    A function for converting a line id possibly represented as a list to
    a single string.

    Args
        id (str, list, tuple)
            a str, list, or tuple containing the id(s) of the lines

    Returns
        id (str)
            string representation of the id
    """

    if isinstance(id, list):
        return ", ".join(id)
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
        line_id = ", ".join(line_id)

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

    Args:
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

    Args:
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

    Args:
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


class LineCollection:
    """
    A class holding a collection of emission lines.

    This enables additional functionality such as quickly calculating
    line ratios or line diagrams.

    Attributes:
        lines (dict)
            A dictionary of synthesizer.line.Line objects.
        line_ids (list)
            A list of line ids.
        wavelengths (unyt_array)
            An array of line wavelengths.
        available_ratios (list)
            A list of available line ratios.
        available_diagrams (list)
            A list of available line diagrams.
    """

    # Define quantities
    wavelengths = Quantity()

    def __init__(self, lines):
        """
        Initialise LineCollection.

        Args:
            lines (dict)
                A dictionary of synthesizer.line.Line objects.
        """
        # Dictionary of synthesizer.line.Line objects.
        self.lines = lines

        # Create an array of line_ids
        self.line_ids = np.array(list(self.lines.keys()))
        self._individual_line_ids = np.array(
            [li for lis in self.line_ids for li in lis.split(",")]
        )

        # Atrributes to enable looping
        self._current_ind = 0
        self.nlines = len(self.line_ids)

        # Create list of line wavelengths
        self.wavelengths = (
            np.array(
                [
                    line.wavelength.to("Angstrom").value
                    for line in self.lines.values()
                ]
            )
            * Angstrom
        )

        # Get the arguments that would sort wavelength
        sorted_arguments = np.argsort(self.wavelengths)

        # Sort the line_ids and wavelengths
        self.line_ids = self.line_ids[sorted_arguments]
        self.wavelengths = self.wavelengths[sorted_arguments]

        # Include line ratio and diagram definitions
        self.line_ratios = line_ratios

        # Create list of available line ratios
        self.available_ratios = []
        for ratio_id, ratio in self.line_ratios.ratios.items():
            # Create a set from the ratio line ids while also unpacking
            # any comma separated lines
            ratio_line_ids = set()
            for lis in ratio:
                ratio_line_ids.update({li.strip() for li in lis.split(",")})

            # Check if line ratio is available
            if ratio_line_ids.issubset(self._individual_line_ids):
                self.available_ratios.append(ratio_id)

        # Create list of available line diagnostics
        self.available_diagrams = []
        for diagram_id, diagram in self.line_ratios.diagrams.items():
            # Create a set from the diagram line ids while also unpacking
            # any comma separated lines
            diagram_line_ids = set()
            for ratio in diagram:
                for lis in ratio:
                    diagram_line_ids.update(
                        {li.strip() for li in lis.split(",")}
                    )

            # Check if line ratio is available
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

    def concatenate(self, other):
        """
        Concatenate two LineCollection objects together.

        Note that any duplicate lines will be taken from other (i.e. the
        LineCollection passed to concatenate).

        Args:
            other (LineCollection)
                A LineCollection object to concatenate with the current
                LineCollection object.

        Returns:
            LineCollection
                A new LineCollection object containing the lines from
                both LineCollection objects.
        """
        # Ensure other is a line collection
        if not isinstance(other, LineCollection):
            raise TypeError(
                "Can only concatenate LineCollection objects together"
            )
        # Combine the lines from each LineCollection object
        my_lines = self.lines.copy()
        my_lines.update(other.lines)

        return LineCollection(my_lines)

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

    def sum(self):
        """
        For collections containing lines from multiple particles calculate the
        integrated line properties and create a new LineCollection object.
        """

        summed_lines = {}
        for line_id, line in self.lines.items():
            summed_lines[line_id] = line.sum()

        return LineCollection(summed_lines)

    def _get_ratio(self, line1, line2):
        """
        Measure (and return) a line ratio.

        Args:
            line1 (str)
                The line or lines in the numerator.
            line2 (str)
                The line or lines in the denominator.

        Returns:
            float
                a line ratio
        """
        # If either line is a combination of lines check if we need to split
        if line1 in self.lines:
            line1 = [line1]
        else:
            line1 = [li.strip() for li in line1.split(",")]
        if line2 in self.lines:
            line2 = [line2]
        else:
            line2 = [li.strip() for li in line2.split(",")]

        return np.sum(
            [self.lines[_line].luminosity for _line in line1], axis=0
        ) / np.sum([self.lines[_line].luminosity for _line in line2], axis=0)

    def get_ratio(self, ratio_id):
        """
        Measure (and return) a line ratio.

        Args:
            ratio_id (str, list)
                Either a ratio_id where the ratio lines are defined in
                line_ratios or a list of lines.

        Returns:
            float
                a line ratio
        """
        # If ratio_id is a string interpret as a ratio_id for the ratios
        # defined in the line_ratios module...
        if isinstance(ratio_id, str):
            # Check if ratio_id exists
            if ratio_id not in self.line_ratios.available_ratios:
                raise exceptions.UnrecognisedOption(
                    f"ratio_id not recognised ({ratio_id})"
                )

            # Check if ratio_id exists
            elif ratio_id not in self.available_ratios:
                raise exceptions.UnrecognisedOption(
                    "LineCollection is missing the lines required for "
                    f"this ratio ({ratio_id})"
                )

            line1, line2 = self.line_ratios.ratios[ratio_id]

        # Otherwise interpret as a list
        elif isinstance(ratio_id, list):
            line1, line2 = ratio_id

        return self._get_ratio(line1, line2)

    def get_diagram(self, diagram_id):
        """
        Return a pair of line ratios for a given diagram_id (E.g. BPT).

        Args:
            diagram_id (str, list)
                Either a diagram_id where the pairs of ratio lines are defined
                in line_ratios or a list of lists defining the ratios.

        Returns:
            tuple (float)
                a pair of line ratios
        """
        # If ratio_id is a string interpret as a ratio_id for the ratios
        # defined in the line_ratios module...
        if isinstance(diagram_id, str):
            # check if ratio_id exists
            if diagram_id not in self.line_ratios.available_diagrams:
                raise exceptions.UnrecognisedOption(
                    f"diagram_id not recognised ({diagram_id})"
                )

            # check if ratio_id exists
            elif diagram_id not in self.available_diagrams:
                raise exceptions.UnrecognisedOption(
                    "LineCollection is missing the lines required for "
                    f"this diagram ({diagram_id})"
                )

            ab, cd = self.line_ratios.diagrams[diagram_id]

        # Otherwise interpret as a list
        elif isinstance(diagram_id, list):
            ab, cd = diagram_id

        return self._get_ratio(*ab), self._get_ratio(*cd)

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

    def get_flux0(self):
        """
        Calculate the rest frame line flux for all lines.

        Uses a standard distance of 10pc to calculate the flux.

        Returns:
            flux (unyt_quantity)
                Flux of the line in units of erg/s/cm2 by default.
        """
        for line in self.lines.values():
            line.get_flux0()

    def get_flux(self, cosmo, z, igm=None):
        """
        Calculate the line flux given a redshift and cosmology for all lines.

        This will also populate the observed_wavelength attribute with the
        wavelength of the line when observed.

        NOTE: if a redshift of 0 is passed the flux return will be calculated
        assuming a distance of 10 pc omitting IGM since at this distance
        IGM contribution makes no sense.

        Args:
            cosmo (astropy.cosmology.)
                Astropy cosmology object.
            z (float)
                The redshift.
            igm (igm)
                The IGM class. e.g. `synthesizer.igm.Inoue14`.
                Defaults to None.

        Returns:
            flux (unyt_quantity)
                Flux of the line in units of erg/s/cm2 by default.
        """
        for line in self.lines.values():
            line.get_flux(cosmo, z, igm)

    @accepts(wavelength_bins=angstrom)
    def get_blended_lines(self, wavelength_bins):
        """
        Blend lines separated by less than the provided wavelength resolution.

        We use a set of wavelength bins to enable the user to control exactly
        which lines are blended together. This also enables an array to be
        used emulating an instrument resolution.

        A simple resolution would lead to ambiguity in situations where A and
        B are blended, and B and C are blended, but A and C are not.

        Args:
            wavelength_bins (unyt_array)
                The wavelength bins into which the lines will be blended.

        Returns:
            LineCollection
                A new LineCollection object containing the blended lines.
        """
        # Sort wavelengths into the bins getting the indices in each bin
        bin_inds = np.digitize(self.wavelengths, wavelength_bins)

        # Create a dictionary to hold the blended lines
        blended_lines = np.empty(len(wavelength_bins), dtype=object)

        # Initialise the array of blended lines to None
        for i in range(blended_lines.size):
            blended_lines[i] = None

        # Loop bin indices and combine the lines into the blended_lines array
        for i, bin_ind in enumerate(bin_inds):
            # Get the line id
            line_id = self.line_ids[i]

            # Get the line itself
            line = self.lines[line_id]

            # If the bin is empty, just store the line
            if blended_lines[bin_ind] is None:
                blended_lines[bin_ind] = line

            # Otherwise, combine the line with the existing line
            else:
                blended_lines[bin_ind] = blended_lines[bin_ind] + line

        # Convert the array of lines to a dictionary ready to make a new
        # LineCollection
        new_lines = {}
        for line in blended_lines:
            if line is not None:
                new_lines[line.id] = line

        return LineCollection(new_lines)


class Line:
    """
    A class representing a spectral line or set of lines (e.g. a doublet).

    Although a Line can be instatiated directly most users should generate
    them using the various different "get_line" methods implemented across
    Synthesizer.

    A Line object can either be a single line or a combination of multiple,
    individually unresolved lines.

    A collection of Line objects are stored within a LineCollection which
    provides an interface to interact with multiple lines at once.

    Attributes:
        wavelength (Quantity)
            The standard (not vacuum) wavelength of the line.
        vacuum_wavelength (Quantity)
            The vacuum wavelength of the line.
        continuum (Quantity)
            The continuum at the line.
        luminosity (Quantity)
            The luminosity of the line.
        flux (Quantity)
            The flux of the line.
        equivalent_width (Quantity)
            The equivalent width of the line.
        individual_lines (list)
            A list of individual lines that make up this line.
        element (list)
            A list of the elements that make up this line.
    """

    # Define quantities
    wavelength = Quantity()
    vacuum_wavelength = Quantity()
    continuum = Quantity()
    luminosity = Quantity()
    flux = Quantity()
    obslam = Quantity()

    @accepts(
        wavelength=angstrom,
        luminosity=erg / s,
        continuum=erg / s / Hz,
    )
    def __init__(
        self,
        line_id=None,
        wavelength=None,
        luminosity=None,
        continuum=None,
        combine_lines=(),
    ):
        """
        Initialise the Line object.

        Args:
            line_id (str)
                The id of the line. If creating a >=doublet the line id will be
                derived while combining lines. This will not be used if lines
                are passed.
            wavelength (unyt_quantity)
                The standard (not vacuum) wavelength of the line. This
                will not be used if lines are passed.
            luminosity (unyt_quantity)
                The luminosity the line. This will not be used if
                lines are passed.
            continuum (unyt_quantity)
                The continuum at the line. This will not be used if
                lines are passed.
            combine_lines (tuple, Line)
                Any number of Line objects to combine into a single Line. If
                these are passed all other kwargs are ignored.
        """
        # Flag deprecation of list and tuple ids
        if isinstance(line_id, (list, tuple)):
            deprecation(
                "Line objects should be created with a string id, not a list"
                " or tuple. This will be removed in a future version."
            )

        # We need to check which version of the inputs we've been given, 3
        # values describing a single line or a set of lines to combine?
        if (
            len(combine_lines) == 0
            and line_id is not None
            and wavelength is not None
            and luminosity is not None
            and continuum is not None
        ):
            self._make_line_from_values(
                line_id,
                wavelength,
                luminosity,
                continuum,
            )
        elif len(combine_lines) > 0:
            self._make_line_from_lines(combine_lines)
        else:
            raise exceptions.InconsistentArguments(
                "A Line needs either its wavelength, luminosity, and continuum"
                " passed, or an arbitrary number of Lines to combine"
            )

        # Initialise an attribute to hold any individual lines used to make
        # this one.
        self.individual_lines = (
            combine_lines if len(combine_lines) > 0 else [self]
        )

        # Initialise the flux and observed wavelength (populated by
        # get_flux/get_flux0 when called)
        self.flux = None
        self.observed_wavelength = None

        # Calculate the vacuum wavelength.
        self.vacuum_wavelength = standard_to_vacuum(self.wavelength)

        # Element
        self.element = [li.strip().split(" ")[0] for li in self.id.split(",")]

    @property
    def continuum_llam(self):
        """Return the continuum in units of Llam (erg/s/AA)."""
        return lnu_to_llam(self.wavelength, self.continuum)

    @property
    def equivalent_width(self):
        """Return the equivalent width."""
        return self.luminosity / self.continuum_llam

    @accepts(
        wavelength=angstrom,
        luminosity=erg / s,
        continuum=erg / s / Hz,
    )
    def _make_line_from_values(
        self, line_id, wavelength, luminosity, continuum
    ):
        """
        Create line from explicit values.

        Args:
            line_id (str)
                The identifier for the line.
            wavelength (unyt_quantity)
                The standard (not vacuum) wavelength of the line.
            luminoisty (unyt_quantity)
                The luminoisty of the line.
            continuum (unyt_quantity)
                The continuum of the line.
        """
        # Set the line attributes
        self.wavelength = wavelength
        self.luminosity = luminosity
        self.continuum = continuum
        self.id = get_line_id(line_id)

    def _make_line_from_lines(self, lines):
        """
        Create a line by combining other lines.

        Args:
            lines (tuple, Line)
                Any number of Line objects to combine into a single line.
        """
        # Ensure we've been handed lines
        if any([not isinstance(line, Line) for line in lines]):
            raise exceptions.InconsistentArguments(
                "args passed to a Line must all be Lines. Did you mean to "
                "pass keyword arguments for wavelength, luminosity and "
                f"continuum? (Got: {[*lines]})"
            )

        # Combine the Line attributes (units are guaranteed here since the
        # quantities are coming directly from a Line)
        self.wavelength = np.mean([line.wavelength for line in lines], axis=0)
        self.luminosity = np.sum([line.luminosity for line in lines], axis=0)
        self.continuum = np.sum([line.continuum for line in lines], axis=0)

        # Derive the line id
        self.id = get_line_id([line.id for line in lines])

    def __str__(self):
        """
        Return a basic summary of the Line object.

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
        if isinstance(self.luminosity, np.ndarray):
            mean_lum = np.mean(self._luminosity)
            pstr += f"Npart: {self.luminosity.size}\n"
            pstr += (
                f"<log10(luminosity/{self.luminosity.units})>: "
                f"{np.log10(mean_lum):.2f}\n"
            )
            mean_eq = np.mean(self.equivalent_width)
            pstr += f"<equivalent width>: {mean_eq:.0f}" + "\n"
            mean_flux = np.mean(self.flux) if self.flux is not None else None
            pstr += (
                f"<log10(flux/{self.flux.units}): {np.log10(mean_flux):.2f}"
                if self.flux is not None
                else ""
            )
        else:
            pstr += (
                f"log10(luminosity/{self.luminosity.units}): "
                f"{np.log10(self.luminosity):.2f}\n"
            )
            pstr += f"equivalent width: {self.equivalent_width:.0f}" + "\n"
            pstr += (
                f"log10(flux/{self.flux.units}): {np.log10(self.flux):.2f}"
                if self.flux is not None
                else ""
            )
        pstr += "-" * 10

        return pstr

    def __add__(self, second_line):
        """
        Add another line to self.

        Overloads + operator to allow direct addition of Line objects.

        Returns
            (Line)
                New instance of Line containing both lines.
        """
        return Line(combine_lines=(self, second_line))

    def sum(self):
        """
        For objects containing lines of multiple particles sum them to produce
        the integrated quantities.
        """

        return Line(
            line_id=self.id,
            wavelength=self.wavelength,
            luminosity=np.sum(self.luminosity),
            continuum=np.sum(self.continuum),
        )

    def get_flux0(self):
        """
        Calculate the rest frame line flux.

        Uses a standard distance of 10pc to calculate the flux.

        This will also populate the observed_wavelength attribute with the
        wavelength of the line when observed (which in the rest frame is the
        same as the emitted wavelength).

        Returns:
            flux (unyt_quantity)
                Flux of the line in units of erg/s/cm2 by default.
        """
        # Compute flux
        self.flux = self.luminosity / (4 * np.pi * (10 * pc) ** 2)

        # Set the observed wavelength (in this case this is the rest frame
        # wavelength)
        self.obslam = self.wavelength

        return self.flux

    def get_flux(self, cosmo, z, igm=None):
        """
        Calculate the line flux given a redshift and cosmology.

        This will also populate the observed_wavelength attribute with the
        wavelength of the line when observed.

        NOTE: if a redshift of 0 is passed the flux return will be calculated
        assuming a distance of 10 pc omitting IGM since at this distance
        IGM contribution makes no sense.

        Args:
            cosmo (astropy.cosmology.)
                Astropy cosmology object.
            z (float)
                The redshift.
            igm (igm)
                The IGM class. e.g. `synthesizer.igm.Inoue14`.
                Defaults to None.

        Returns:
            flux (unyt_quantity)
                Flux of the line in units of erg/s/cm2 by default.
        """
        # If the redshift is 0 we can assume a distance of 10pc and ignore
        # the IGM
        if z == 0:
            return self.get_flux0()

        # Get the luminosity distance
        luminosity_distance = (
            cosmo.luminosity_distance(z).to("cm").value
        ) * cm

        # Compute flux
        self.flux = self.luminosity / (4 * np.pi * luminosity_distance**2)

        # Set the observed wavelength
        self.obslam = self.wavelength * (1 + z)

        # If we are applying an IGM model apply it
        if igm is not None:
            self.flux *= igm().get_transmission(z, self._obslam)

        return self.flux

    def combine(self, *lines):
        """
        Combine this line with an arbitrary number of other lines.

        This is important for combing >2 lines together since the simple
        line1 + line2 + line3 addition of multiple lines will not correctly
        average over all lines.

        Args:
            lines (Line)
                Any number of Line objects to combine into a single line.

        Returns:
            (Line)
                A new Line object containing the combined lines.
        """
        # Ensure we've been handed lines
        if any([not isinstance(line, Line) for line in lines]):
            raise exceptions.InconsistentArguments(
                "args passed to a Line must all be Lines. Did you mean to "
                "pass keyword arguments for wavelength, luminosity and "
                "continuum"
            )

        return Line(self, combine_lines=lines)

    def apply_attenuation(
        self,
        tau_v,
        dust_curve,
        mask=None,
    ):
        """
        Apply attenuation to this line.

        Args:
            tau_v (float/array-like, float)
                The V-band optical depth for every star particle.
            dust_curve (synthesizer.emission_models.attenuation.*)
                An instance of one of the dust attenuation models. (defined in
                synthesizer/emission_models.attenuation.py)
            mask (array-like, bool)
                A mask array with an entry for each line. Masked out
                spectra will be ignored when applying the attenuation. Only
                applicable for multidimensional lines.

        Returns:
                Line
                    A new Line object containing the attenuated line.
        """
        # Ensure the mask is compatible with the spectra
        if mask is not None:
            if self._luminosity.ndim < 1:
                raise exceptions.InconsistentArguments(
                    "Masks are only applicable for Lines containing "
                    "multiple elements"
                )
            if self._luminosity.shape[0] != mask.size:
                raise exceptions.InconsistentArguments(
                    "Mask and lines are incompatible shapes "
                    f"({mask.shape}, {self._lnu.shape})"
                )

        # If tau_v is an array it needs to match the spectra shape
        if isinstance(tau_v, np.ndarray):
            if self._luminosity.ndim < 1:
                raise exceptions.InconsistentArguments(
                    "Arrays of tau_v values are only applicable for Lines"
                    " containing multiple elements"
                )
            if self._luminosity.shape[0] != tau_v.size:
                raise exceptions.InconsistentArguments(
                    "tau_v and lines are incompatible shapes "
                    f"({tau_v.shape}, {self._lnu.shape})"
                )

        # Compute the transmission
        transmission = dust_curve.get_transmission(tau_v, self.wavelength)

        # Apply the transmision
        att_lum = self.luminosity
        att_cont = self.continuum
        if mask is None:
            att_lum *= transmission
            att_cont *= transmission
        else:
            att_lum[mask] *= transmission[mask]
            att_cont[mask] *= transmission[mask]

        return Line(
            line_id=self.id,
            wavelength=self.wavelength,
            luminosity=att_lum,
            continuum=att_cont,
        )
