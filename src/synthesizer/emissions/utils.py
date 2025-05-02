"""A submodule containing utility functions for working with emissions.

This module contains functions for converting between different line
identifiers, calculating attenuation and transmission, and
flattening lists of lines. It also includes functions for working with
spectral energy distributions (SEDs) and calculating attenuation at
specific wavelengths.

Example usage:

    # Example usage of the functions
    line_id = ["H 1 4861.32A", "H 1 6562.80A"]
    composite_line_id = get_composite_line_id_from_list(line_id)
    print(composite_line_id)  # Output: "H 1 4861.32A, H 1 6562.80A"

    # Example usage of the get_line_label function
    line_label = get_line_label(line_id)
    print(line_label)  # Output: "HII4861.32A+HII6562.80A"

    # Example usage of the flatten_linelist function
    list_to_flatten = [["H 1 4861.32A"], ["H 1 6562.80A"]]
    flattened_list = flatten_linelist(list_to_flatten)
    print(flattened_list)  # Output: ["H 1 4861.32A", "H 1 6562.80A"]

    # Example usage of the get_transmission function
    intrinsic_sed = Sed(...)
    attenuated_sed = Sed(...)
    transmission = get_transmission(intrinsic_sed, attenuated_sed)

    # Example usage of the get_attenuation function
    intrinsic_sed = Sed(...)
    attenuated_sed = Sed(...)
    attenuation = get_attenuation(intrinsic_sed, attenuated_sed)

    # Example usage of the get_attenuation_at_lam function
    lam = 5500.0 * angstrom
    intrinsic_sed = Sed(...)
    attenuated_sed = Sed(...)
    attenuation_at_lam = get_attenuation_at_lam(
        lam, intrinsic_sed, attenuated_sed
    )

"""

import numpy as np
from unyt import angstrom

from synthesizer import exceptions
from synthesizer.units import accepts


def get_composite_line_id_from_list(id):
    """Convert a list of line ids to a string describing a composite line.

    A composite line is a line that is made up of multiple lines, e.g.
    a doublet or triplet. This function takes a list of line ids and converts
    them to a single string.

    e.g. ["H 1 4861.32A", "H 1 6562.80A"] -> "H 1 4861.32A, H 1 6562.80A"

    Args:
        id (list, tuple):
            a str, list, or tuple containing the id(s) of the lines

    Returns:
        id (str):
            string representation of the id
    """
    if isinstance(id, list):
        return ", ".join(id)
    else:
        return id


def get_line_label(line_id):
    """Get a line label for a given line_id, ratio, or diagram.

    Where the line_id is one of several predifined lines in line_labels this
    label is used, otherwise the label is constructed from the line_id.

    Argumnents
        line_id (str):
            The line_id either as a list of individual lines or a string. If
            provided as a list this is automatically converted to a single
            string so it can be used as a key.

    Returns:
        line_label (str):
            A nicely formatted line label.
    """
    # if the line_id is a list (denoting a doublet or higher)
    if isinstance(line_id, list):
        line_id = ", ".join(line_id)

    if line_id in line_labels.keys():
        line_label = line_labels[line_id]
    else:
        line_id = [li.strip() for li in line_id.split(",")]
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
    """Flatten a mixed list of lists and strings and remove duplicates.

    Used when converting a line list which may contain single lines
    and doublets.

    Args:
        list_to_flatten (list):
            list containing lists and/or strings and integers

    Returns:
        (list):
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


def get_roman_numeral(number):
    """Convert an integer into a roman numeral str.

    Used for renaming emission lines from the cloudy defaults.

    Args:
        number (int):
            The number to convert into a roman numeral.

    Returns:
        number_representation (str):
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
    """Convert a line alias to a line id.

    Args:
        alias (str):
            The line alias.

    Returns:
        line_id (str):
            The line id.
    """
    if alias in aliases:
        return aliases[alias]
    return alias


def get_transmission(intrinsic_sed, attenuated_sed):
    """Calculate transmission as a function of wavelength.

    The transmission is defined as the ratio between an attenuated and
    an intrinsic sed.

    Args:
        intrinsic_sed (Sed):
            The intrinsic spectra object.
        attenuated_sed (Sed):
            The attenuated spectra object.

    Returns:
        np.ndarray of float:
            The transmission array.
    """
    # Ensure wavelength arrays are equal
    if not np.array_equal(attenuated_sed._lam, intrinsic_sed._lam):
        raise exceptions.InconsistentArguments(
            "Wavelength arrays of input spectra must be the same!"
        )

    return attenuated_sed.lnu / intrinsic_sed.lnu


def get_attenuation(intrinsic_sed, attenuated_sed):
    """Calculate attenuation as a function of wavelength.

    Args:
        intrinsic_sed (Sed):
            The intrinsic spectra object.
        attenuated_sed (Sed):
            The attenuated spectra object.

    Returns:
        np.ndarray of float
            The attenuation array in magnitudes.
    """
    # Calculate the transmission array
    transmission = get_transmission(intrinsic_sed, attenuated_sed)

    return -2.5 * np.log10(transmission)


@accepts(lam=angstrom)
def get_attenuation_at_lam(lam, intrinsic_sed, attenuated_sed):
    """Calculate attenuation at a given wavelength.

    Args:
        lam (float/np.ndarray of float):
            The wavelength/s at which to evaluate the attenuation in
            the same units as sed.lam (by default angstrom).
        intrinsic_sed (Sed):
            The intrinsic spectra object.
        attenuated_sed (Sed):
            The attenuated spectra object.

    Returns:
        float/np.ndarray of float
            The attenuation at the passed wavelength/s in magnitudes.
    """
    # Ensure lam is in the same units as the sed
    if lam.units != intrinsic_sed.lam.units:
        lam = lam.to(intrinsic_sed.lam.units)

    # Calcilate the transmission array
    attenuation = get_attenuation(intrinsic_sed, attenuated_sed)

    return np.interp(lam.value, intrinsic_sed._lam, attenuation)


def get_attenuation_at_5500(intrinsic_sed, attenuated_sed):
    """Calculate rest-frame FUV attenuation at 5500 angstrom.

    Args:
        intrinsic_sed (Sed):
            The intrinsic spectra object.
        attenuated_sed (Sed):
            The attenuated spectra object.

    Returns:
         float
            The attenuation at rest-frame 5500 angstrom in magnitudes.
    """
    return get_attenuation_at_lam(
        5500.0 * angstrom,
        intrinsic_sed,
        attenuated_sed,
    )


def get_attenuation_at_1500(intrinsic_sed, attenuated_sed):
    """Calculate rest-frame FUV attenuation at 1500 angstrom.

    Args:
        intrinsic_sed (Sed):
            The intrinsic spectra object.
        attenuated_sed (Sed):
            The attenuated spectra object.

    Returns:
         float
            The attenuation at rest-frame 1500 angstrom in magnitudes.
    """
    return get_attenuation_at_lam(
        1500.0 * angstrom,
        intrinsic_sed,
        attenuated_sed,
    )


def combine_list_of_seds(sed_list):
    """Convert a list of Seds into a single Sed object.

    Combines a list of `Sed` objects (length `Ngal`) into a single
    `Sed` object, with dimensions `Ngal x Nlam`. Each `Sed` object
    in the list should have an identical wavelength range.

    Args:
        sed_list (list):
            list of `Sed` objects
    """
    out_sed = sed_list[0]
    for sed in sed_list[1:]:
        out_sed = out_sed.concat(sed)

    return out_sed
