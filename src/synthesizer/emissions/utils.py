"""A submodule containing utility functions for working with emission lines."""

from synthesizer import line_ratios


def get_composite_line_id_from_list(id):
    """
    Convert a list of line ids to a single string describing a composite line.

    A composite line is a line that is made up of multiple lines, e.g.
    a doublet or triplet. This function takes a list of line ids and converts
    them to a single string.

    e.g. ["H 1 4861.32A", "H 1 6562.80A"] -> "H 1 4861.32A, H 1 6562.80A"

    Args
        id (list, tuple)
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
    Get a line label for a given line_id, ratio, or diagram.

    Where the line_id is one of several predifined lines in line_labels this
    label is used, otherwise the label is constructed from the line_id.

    Argumnents
        line_id (str)
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

    if line_id in line_labels.keys():
        line_label = line_labels[line_id]
    else:
        line_id = line_id.split(",")
        _line_labels = []
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

            _line_labels.append(
                f"{element}{get_roman_numeral(int(ion))}{wavelength}"
            )

        line_label = "+".join(_line_labels)

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
    # If the id is a string get the lines from the line_ratios sub-module
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
    Get a x and y labels for a given diagram_id.

    Args:
        diagram_id (str)
            The diagram identificantion, e.g. OHNO.

    Returns:
        xlabel (str)
            A string representation of the x-label.
        ylabel (str)
            A string representation of the y-label.
    """
    # Get the list of lines for a given ratio_id
    diagram_line_ids = line_ratios.diagrams[diagram_id]
    xlabel = get_ratio_label(diagram_line_ids[0])
    ylabel = get_ratio_label(diagram_line_ids[1])

    return xlabel, ylabel


def get_roman_numeral(number):
    """
    Convert an integer into a roman numeral str.

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


# Shorthand for common lines
aliases = {
    "Hb": "H 1 4861.32A",
    "Ha": "H 1 6562.80A",
    "Hg": "H 1 4340.46A",
    "O1": "O 1 6300.30A",
    "O2b": "O 2 3726.03A",
    "O2r": "O 2 3728.81A",
    "O2": "O 2 3726.03A, O 2 3728.81A",
    "O3b": "O 3 4958.91A",
    "O3r": "O 3 5006.84A",
    "O3": "O 3 4958.91A, O 3 5006.84A",
    "Ne3": "Ne 3 3868.76A",
    "N2": "N 2 6583.45A",
    "S2": "S 2 6730.82A, S 2 6716.44A",
}


# Standard names
Ha = aliases["Ha"]
Hb = aliases["Hb"]
O1 = aliases["O1"]
O2b = aliases["O2b"]
O2r = aliases["O2r"]
O2 = aliases["O2"]
O3b = aliases["O3b"]
O3r = aliases["O3r"]
O3 = aliases["O3"]
Ne3 = aliases["Ne3"]
N2 = aliases["N2"]
S2 = aliases["S2"]


# Dictionary of common line labels to use by default
line_labels = {
    "O 2 3726.03A,O 2 3728.81A": "[OII]3726,3729",
    "H 1 4861.32A": r"H\beta",
    "O 3 4958.91A,O 3 5006.84A": "[OIII]4959,5007",
    "H 1 6562.80A": r"H\alpha",
    "O 3 5006.84A": "[OIII]5007",
    "N 2 6583.45A": "[NII]6583",
}


def alias_to_line_id(alias):
    """
    Convert a line alias to a line id.

    Args:
        alias (str)
            The line alias.

    Returns:
        line_id (str)
            The line id.
    """
    if alias in aliases:
        return aliases[alias]
    return alias
