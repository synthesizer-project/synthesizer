"""A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
"""

import numpy as np
import unyt.physical_constants as const
from unyt import Hz, K, erg, pc, s, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.synth_warnings import warn
from synthesizer.units import accepts


@accepts(frequency=Hz, temperature=K)
def planck(frequency, temperature):
    """Compute the planck distribution for a given frequency and temperature.

    This function computes the spectral radiance of a black body at a given
    frequency and temperature using Planck's law. The spectral radiance is
    then converted to spectral luminosity density assuming a luminosity
    distance of 10 pc.

    Parameters:
        frequency (float or unyt_quantity): Frequency of the radiation in Hz.
        temperature (float or unyt_quantity): Temperature in Kelvin.

    Returns:
        unyt_quantity: Spectral luminosity density in erg/s/Hz.
    """
    # Planck's law: B(ν, T) = (2*h*ν^3) / (c^2 * (exp(hν / kT) - 1))
    exponent = (const.h * frequency) / (const.kb * temperature)
    spectral_radiance = (2 * const.h * frequency**3) / (
        const.c**2 * (np.exp(exponent) - 1)
    )

    # Convert from spectral radiance density to spectral luminosity density,
    # here we'll assume a luminosity distance of 10 pc
    lnu = spectral_radiance * 4 * np.pi * (10 * pc) ** 2

    # Convert the result to erg/s/Hz and return
    return lnu.to(erg / s / Hz)


def rebin_1d(arr, resample_factor, func=np.sum):
    """Rebin a 1D array.

    This function takes a 1D array and rebins it by a specified factor using
    a specified function (e.g. mean or sum).

    Args:
        arr (np.ndarray, list or unyt_array):
            The input 1D array.
        resample_factor (int):
            The integer rebinning factor, i.e. how many bins to rebin by.
        func (function):
            The function to use (e.g. mean or sum).

    Returns:
        array-like: The input array resampled by i.
    """
    # Ensure the array is 1D
    if arr.ndim != 1:
        raise exceptions.InconsistentArguments(
            f"Input array must be 1D (input was {arr.ndim}D)"
        )

    # Safely handle no integer resamples
    if not isinstance(resample_factor, int):
        warn(
            f"resample factor ({resample_factor}) is not an"
            f" integer, converting it to {int(resample_factor)}",
        )
        resample_factor = int(resample_factor)

    # How many elements in the input?
    n = len(arr)

    # If array is not the right size truncate it
    if n % resample_factor != 0:
        arr = arr[: int(resample_factor * np.floor(n / resample_factor))]

    # Set up the 2D array ready to have func applied
    rows = len(arr) // resample_factor
    brr = arr.reshape(rows, resample_factor)

    return func(brr, axis=1)


def scalar_to_array(value):
    """Convert a passed scalar to an array.

    Args:
        value (Any):
            The value to wrapped into an array. If already an array-like
            object then it is returned as is.

    Returns:
        array-like/unyt_array:
            The scalar value wrapped in an array or the array-like object
            passed in.

    Raises:
        InconsistentArguments
            If the value is not a scalar or array-like object.
    """
    # If the value is None just return it straight away
    if value is None:
        return None

    # Do we have units? If so strip them away for now for simplicity
    if isinstance(value, (unyt_quantity, unyt_array)):
        units = value.units
        value = value.ndview
    else:
        units = None

    # Just return it if we have been handed a ndim > 0 array already or None
    if not np.isscalar(value) and value.shape != ():
        arr = value

    # If we have a scalar or a 0D array then wrap it in a 1D array
    elif np.isscalar(value) or value.shape == ():
        arr = np.array(
            [
                value,
            ]
        )

    else:
        raise exceptions.InconsistentArguments(
            "Value to convert to an array wasn't a float or a unyt_quantity:"
            f"type(value) = {type(value)}"
        )

    # Reattach the units if they were stripped
    if units is not None:
        arr = unyt_array(arr, units)

    return arr


def parse_grid_id(grid_id):
    """Parse a grid name for the properties of the grid.

    This is used for parsing a grid ID to return the SPS model,
    version, and IMF

    Args:
        grid_id (str):
            string grid identifier
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
        imf = rf"$\alpha={float(imf) / 100}$"

    return {
        "sps_model": sps_model,
        "sps_model_version": sps_model_version,
        "imf": imf,
        "imf_hmc": imf_hmc,
    }


def wavelength_to_rgba(
    wavelength,
    gamma=0.8,
    fill_red=(0, 0, 0, 0.5),
    fill_blue=(0, 0, 0, 0.5),
    alpha=1.0,
):
    """Convert wavelength float to RGBA tuple.

    Taken from https://stackoverflow.com/questions/44959955/\
        matplotlib-color-under-curve-based-on-spectral-color

    Who originally took it from http://www.noah.org/wiki/\
        Wavelength_to_RGB_in_Python

    Args:
        wavelength (float):
            Wavelength in nm.
        gamma (float):
            Gamma value.
        fill_red (bool or tuple):
            The colour (RGBA) to use for wavelengths red of the visible. If
            None use final nearest visible colour.
        fill_blue (bool or tuple):
            The colour (RGBA) to use for wavelengths red of the visible. If
            None use final nearest visible colour.
        alpha (float):
            The value of the alpha channel (between 0 and 1).


    Returns:
        rgba (tuple):
            RGBA tuple.
    """
    if wavelength < 380:
        return fill_blue
    if wavelength > 750:
        return fill_red
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0

    return (R, G, B, alpha)


def wavelengths_to_rgba(wavelengths, gamma=0.8):
    """Convert wavelength array to RGBA list.

    Arguments:
        wavelengths (unyt_array):
            The wavelengths to convert to RGBA tuples.
        gamma (float):
            The gamma value to use for the conversion.

    Returns:
        list : list of RGBA tuples.
    """
    # If wavelengths provided as a unyt_array convert to nm otherwise assume
    # in Angstrom and convert.
    if isinstance(wavelengths, unyt_array):
        wavelengths_ = wavelengths.to("nm").ndview
    else:
        wavelengths_ = wavelengths / 10.0

    rgba = []
    for wavelength in wavelengths_:
        rgba.append(wavelength_to_rgba(wavelength, gamma=gamma))

    return rgba


def combine_arrays(arr1, arr2, verbose=False):
    """Combine two arrays into a single array.

    This function is a helper used to combine two arrays of the same length
    into a single array while abstracting some checks and handling improper
    combinations.

    If both arrays are None then None is returned. If one array is None and
    the other is not then None is returned along with a warning.

    Args:
        arr1 (np.ndarray):
            The first array to combine.
        arr2 (np.ndarray):
            The second array to combine.
        verbose (bool):
            If True, print warnings for None arrays.

    Returns:
        np.ndarray or None: The combined array.
    """
    # Are both arrays None?
    if arr1 is None and arr2 is None:
        return None

    # If one is None and the other is not then return None
    elif arr1 is None or arr2 is None:
        if verbose:
            warn("One of the arrays is None, one is not. Returning None.")

        return None

    # Ensure both arrays aren't 0 dimensional
    elif arr1.ndim == 0 or arr2.ndim == 0:
        return None

    # If both are not None then combine them
    else:
        return np.concatenate([arr1, arr2])


def pluralize(word: str) -> str:
    """Pluralize a singular word.

    Args:
        word (str):
            The word to pluralize.

    Returns:
        str: The pluralized word.
    """
    if (
        word.endswith("s")
        or word.endswith("x")
        or word.endswith("z")
        or word.endswith("sh")
        or word.endswith("ch")
    ):
        return word + "es"
    elif word.endswith("y") and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    elif word.endswith("f"):
        return word[:-1] + "ves"
    elif word.endswith("fe"):
        return word[:-2] + "ves"
    elif word.endswith("o") and word[-2] not in "aeiou":
        return word + "es"
    else:
        return word + "s"


def depluralize(word: str) -> str:
    """Convert a plural word to its singular form based on simple rules.

    Args:
        word (str): The word to depluralize.

    Returns:
        str: The depluralized word.
    """
    if word.endswith("ies") and len(word) > 3:  # babies -> baby
        return word[:-3] + "y"
    elif word.endswith("ves"):  # leaves -> leaf, knives -> knife
        return word[:-3] + "f"
    elif word.endswith("oes"):  # heroes -> hero, potatoes -> potato
        return word[:-2]
    elif word.endswith(
        ("ches", "shes", "xes", "sses")
    ):  # boxes -> box, churches -> church
        return word[:-2]
    elif word.endswith("s") and len(word) > 2:  # general case: cats -> cat
        return word[:-1]

    return word  # Return unchanged if no rule applies


def ensure_double_precision(value):
    """Ensure that the input value is a double precision float.

    Args:
        value (float or unyt_quantity): The value to be converted.

    Returns:
        unyt_quantity: The input value as a double precision float.
    """
    # If the value is None, return it as is
    if value is None:
        return value

    # Convert the value to double precision
    if isinstance(value, (unyt_quantity, unyt_array, np.ndarray)):
        return value.astype(np.float64)
    elif isinstance(value, (int, float)):
        return np.float64(value)
    elif np.isscalar(value):
        return np.float64(value)
    else:
        raise exceptions.InconsistentArguments(
            "Value to convert to double precision wasn't compatible:"
            f"type(value) = {type(value)}"
        )


def is_c_compatible_double(arr):
    """Check if the input array is compatible with our C extensions.

    Being "compatible" means that the numpy array is both C contiguous and
    is a double array for floating point numbers.

    If we don't do this then the C extensions will produce garbage due to the
    mismatch between the data types.

    Args:
        arr (np.ndarray): The input array to be checked.

    Returns:
        bool: True if the array is C contiguous and of double precision,
              False otherwise.
    """
    return arr.flags["C_CONTIGUOUS"] and arr.dtype == np.float64


def is_c_compatible_int(arr):
    """Check if the input array is compatible with our C extensions.

    Being "compatible" means that the numpy array is both C contiguous and
    is an int array for integer numbers.

    If we don't do this then the C extensions will produce garbage due to the
    mismatch between the data types.

    Args:
        arr (np.ndarray): The input array to be checked.

    Returns:
        bool: True if the array is C contiguous and of int type,
              False otherwise.
    """
    return arr.flags["C_CONTIGUOUS"] and arr.dtype == np.intc


def ensure_array_c_compatible_double(arr):
    """Ensure that the input array is compatible with our C extensions.

    Being "compatible" means that the numpy array is both C contiguous and
    is a double array for floating point numbers.

    If we don't do this then the C extensions will produce garbage due to the
    mismatch between the data types.

    Args:
        arr (np.ndarray): The input array to be checked.
    """
    # If the array is None, return it as is
    if arr is None:
        return arr

    # Convert a list to a numpy array before we move on
    if isinstance(arr, list):
        arr = np.array(arr)

    # If we have units we need to strip them off temporarily
    units = None
    if isinstance(arr, (unyt_array, unyt_quantity)):
        units = arr.units
        arr = arr.ndview

    # If its a scalar then just return it as a double
    if np.isscalar(arr):
        return np.float64(arr)

    # Do we need to do anything?
    need_contiguous = False
    need_double = False
    if not arr.flags["C_CONTIGUOUS"]:
        need_contiguous = True
    if arr.dtype != np.float64:
        need_double = True

    # If there's nothing to do then just return
    if not need_double and not need_contiguous:
        return arr

    # If we need both we can do it all at once
    if need_double and need_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float64)

    # If we only need to make it contiguous then do that
    elif need_contiguous:
        arr = np.ascontiguousarray(arr)

    # If we only need to make it double then do that
    elif need_double:
        arr = arr.astype(np.float64)

    # If we had units then reattach them
    if units is not None:
        arr = unyt_array(arr, units)

    return arr


def get_attr_c_compatible_double(obj, attr):
    """Ensure an attribute of an object is compatible with our C extensions.

    This function checks if the attribute of the object is a numpy array and
    ensures that it is both C contiguous and of double precision. If the
    attribute is not compatible, it modifies it in place.

    Args:
        obj (object): The object containing the attribute to be checked.
        attr (str): The name of the attribute to be checked.
    """
    # Get the attribute from the object
    arr = getattr(obj, attr)

    # Just return it if it's None
    if arr is None:
        return arr

    # Handle singular floats
    if np.isscalar(arr):
        return np.float64(arr)

    # Ensure the attribute is compatible with C extensions
    if not is_c_compatible_double(arr):
        # It's not compatible, make it compatible
        arr = ensure_array_c_compatible_double(arr)

        # Assign it inplace so we only do this conversion once (but only if we
        # can actually set it)
        if hasattr(obj, attr):
            # Set the attribute to the new array
            setattr(obj, attr, arr)

    # Also return the array
    return arr
