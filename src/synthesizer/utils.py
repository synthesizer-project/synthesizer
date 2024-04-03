"""A module containing general utility functions.

Example usage:

    planck(frequency, temperature=10000 * K)
    rebin_1d(arr, 10, func=np.sum)
    value_to_array(1.0)
    has_units(1.0 * K)
    parse_grid_id("bc03_chab")
"""

from typing import Callable, Dict, List, Union

import numpy as np
from numpy.typing import NDArray
from unyt import c, h, kb, unyt_array, unyt_quantity

from synthesizer import exceptions


def planck(
    nu: unyt_array,
    temperature: Union[unyt_array, unyt_quantity],
) -> NDArray[np.float64]:
    """
    Planck's law.

    Args:
        nu: The frequencies at which to calculate the distribution.
        temperature: The dust temperature. Either a single value or the same
                     size as nu.

    Returns:
        The values of the distribution at nu.
    """
    return (2.0 * h * (nu**3) * (c**-2)) * (
        1.0 / (np.exp(h * nu / (kb * temperature)) - 1.0)
    )


def has_units(
    x: Union[NDArray, unyt_array, unyt_quantity, float, int],
) -> bool:
    """
    Check whether the passed variable has units.

    i.e., Ensure the passed variable is a unyt_quantity or unyt_array.

    Args:
        x: The variables to check.

    Returns:
        True if the variable has units, False otherwise.
    """
    return isinstance(x, (unyt_array, unyt_quantity))


def rebin_1d(
    arr: NDArray[np.float64],
    resample_factor: int,
    func: Callable = np.sum,
) -> NDArray[np.float64]:
    """
    Rebin a 1D array.

    The rebinning can be done using a specified function (e.g., sum or mean).

    Args:
        arr: The input 1D array.
        resample_factor: The integer rebinning factor, i.e., how many bins to
                         rebin by.
        func: The function to use (e.g., mean or sum).

    Returns:
        The input array resampled by i.
    """
    if arr.ndim != 1:
        raise exceptions.InconsistentArguments(
            f"Input array must be 1D (input was {arr.ndim}D)"
        )

    n: int = len(arr)
    if n % resample_factor != 0:
        arr = arr[: int(resample_factor * np.floor(n / resample_factor))]

    rows: int = len(arr) // resample_factor
    brr: NDArray[np.float64] = arr.reshape(rows, resample_factor)

    return func(brr, axis=1)


def value_to_array(
    value: Union[float, unyt_quantity, unyt_array, None],
) -> Union[NDArray, unyt_array]:
    """
    Convert a single value to an array holding a single value.

    Args:
        value: The value to be wrapped into an array.

    Returns:
        An array containing the single value

    Raises:
        InconsistentArguments
            If the argument is not a float or unyt_quantity.
    """
    if (isinstance(value, np.ndarray) and value.size > 1) or value is None:
        return value

    arr: Union[NDArray[np.float64], unyt_array]
    if isinstance(value, float):
        arr = np.array([value])
    elif isinstance(value, (unyt_quantity, unyt_array)):
        arr = np.array([value.value]) * value.units
    else:
        raise exceptions.InconsistentArguments(
            "Value to convert to an array wasn't a float or a unyt_quantity:"
            f" type(value) = {type(value)}"
        )

    return arr


def parse_grid_id(grid_id: str) -> Dict[str, str]:
    """
    Parse a grid name for the properties of the grid.

    This is used for parsing a grid ID to return the SPS model,
    version, and IMF

    Args:
        grid_id: The string grid identifier.

    Returns:
        A dictionary containing parsed grid properties.
    """
    parts: List[str] = grid_id.split("_")
    sps_model_: str
    imf_: str
    sps_model_, imf_ = parts[0], parts[1] if len(parts) > 1 else ""
    cloudy: str
    cloudy_model: str
    cloudy, cloudy_model = parts[2], parts[3] if len(parts) == 4 else ""

    sps_model_parts: List[str] = sps_model_.split("-")
    sps_model: str = sps_model_parts[0]
    sps_model_version: str = (
        "-".join(sps_model_parts[1:]) if len(sps_model_parts) > 1 else ""
    )

    imf_parts: List[str] = imf_.split("-")
    imf: str = imf_parts[0]
    imf_hmc: str = imf_parts[1] if len(imf_parts) > 1 else ""

    # Translate IMFs to readable format
    imf_translation: Dict[str, str] = {
        "chab": "Chabrier (2003)",
        "chabrier03": "Chabrier (2003)",
        "Kroupa": "Kroupa (2003)",
        "salpeter": "Salpeter (1955)",
        "135all": "Salpeter (1955)",
    }
    translated_imf: str = imf_translation.get(imf, imf)

    return {
        "sps_model": sps_model,
        "sps_model_version": sps_model_version,
        "imf": translated_imf,
        "imf_hmc": imf_hmc,
        "cloudy": cloudy,
        "cloudy_model": cloudy_model,
    }
