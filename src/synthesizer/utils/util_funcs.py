"""A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
"""

import numpy as np
import unyt.physical_constants as const
from unyt import Hz, erg, pc, s, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.units import has_units
from synthesizer.warnings import warn


def planck(frequency, temperature):
    """
    Compute the planck distribution for a given frequency and temperature.

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
    # Ensure we have unyt quantities
    if not has_units(frequency):
        raise exceptions.InconsistentArguments(
            "Frequency must have units (e.g. Hz) to calculate Planck's law."
        )
    if not has_units(temperature):
        raise exceptions.InconsistentArguments(
            "Temperature must have units (e.g. K) to calculate Planck's law."
        )

    # Ensure frequency is in Hz and temperature is in K
    frequency = frequency.to("Hz")
    temperature = temperature.to("K")

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
    """
    A simple function for rebinning a 1D array using a specificed
    function (e.g. sum or mean).

    Args:
        arr (array-like)
            The input 1D array.
        resample_factor (int)
            The integer rebinning factor, i.e. how many bins to rebin by.
        func (func)
            The function to use (e.g. mean or sum).

    Returns:
        array-like
            The input array resampled by i.
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


def value_to_array(value):
    """
    A helper functions for converting a single value to an array holding
    a single value.

    Args:
        value (float/unyt_quantity)
            The value to wrapped into an array.

    Returns:
        array-like/unyt_array
            An array containing the single value

    Raises:
        InconsistentArguments
            If the argument is not a float or unyt_quantity.
    """

    # Just return it if we have been handed an array already or None
    # NOTE: unyt_arrays and quantities are by definition arrays and thus
    # return True for the isinstance below.
    if (isinstance(value, np.ndarray) and value.size > 1) or value is None:
        return value

    if isinstance(value, float):
        arr = np.array(
            [
                value,
            ]
        )

    elif isinstance(value, (unyt_quantity, unyt_array)):
        arr = (
            np.array(
                [
                    value.value,
                ]
            )
            * value.units
        )
    else:
        raise exceptions.InconsistentArguments(
            "Value to convert to an array wasn't a float or a unyt_quantity:"
            f"type(value) = {type(value)}"
        )

    return arr


def parse_grid_id(grid_id):
    """
    Parse a grid name for the properties of the grid.

    This is used for parsing a grid ID to return the SPS model,
    version, and IMF

    Args:
        grid_id (str)
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
        imf = rf"$\alpha={float(imf)/100}$"

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
    """
    Convert wavelength float to RGBA tuple.

    Taken from https://stackoverflow.com/questions/44959955/\
        matplotlib-color-under-curve-based-on-spectral-color

    Who originally took it from http://www.noah.org/wiki/\
        Wavelength_to_RGB_in_Python

    Arguments:
        wavelength (float)
            Wavelength in nm.
        gamma (float)
            Gamma value.
        fill_red (bool or tuple)
            The colour (RGBA) to use for wavelengths red of the visible. If
            None use final nearest visible colour.
        fill_blue (bool or tuple)
            The colour (RGBA) to use for wavelengths red of the visible. If
            None use final nearest visible colour.
        alpha (float)
            The value of the alpha channel (between 0 and 1).


    Returns:
        rgba (tuple)
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
    """
    Convert wavelength array to RGBA list.

    Arguments:
        wavelength (unyt_array)
            Wavelength in nm.

    Returns:
        rgba (list)
            list of RGBA tuples.
    """

    # If wavelengths provided as a unyt_array convert to nm otherwise assume
    # in Angstrom and convert.
    if isinstance(wavelengths, unyt_array):
        wavelengths_ = wavelengths.to("nm").value
    else:
        wavelengths_ = wavelengths / 10.0

    rgba = []
    for wavelength in wavelengths_:
        rgba.append(wavelength_to_rgba(wavelength, gamma=gamma))

    return rgba
