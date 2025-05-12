"""A submodule for instrument related utility functions.

Example usage:

    # Generate a wavelength array with a constant resolving power of 10000
    lams = generate_wavelength_array(400, 700, 10000)

    # Generate a wavelength array with a variable resolving power
    func = lambda wav: 10000 + (wav - 400) / 300 * 5000
    lams = generate_wavelength_array(400, 700, func)
"""

import inspect
import os
from typing import Callable, Union

import numpy as np
from unyt import angstrom, unyt_array, unyt_quantity

from synthesizer.units import accepts


@accepts(lam_min=angstrom, lam_max=angstrom)
def get_lams_from_resolving_power(
    lam_min: unyt_quantity,
    lam_max: unyt_quantity,
    resolving_power: Union[float, Callable[[unyt_quantity], float]],
) -> unyt_array:
    """Generate a wavelength array with variable resolving power.

    This function creates an array of wavelengths between `lam_min` and
    `lam_max`, where the spacing between wavelengths is determined by the
    resolving power. The resolving power can be specified as a constant
    or as a function that varies with wavelength.

    Args:
        lam_min (unyt_quantity): Minimum wavelength in nanometers.
        lam_max (unyt_quantity): Maximum wavelength in nanometers.
        resolving_power (float or callable): Resolving power (R = λ / Δλ).
            Can be a constant value or a function that takes a wavelength
            with units and returns a resolving power.

    Returns:
        unyt_array: Array of wavelengths in nanometers.
    """
    wavelengths = [lam_min]
    current_wav = lam_min

    while current_wav < lam_max:
        # Determine resolving power at current wavelength
        _resolving_power = (
            resolving_power(current_wav)
            if callable(resolving_power)
            else resolving_power
        )

        # Calculate wavelength step
        delta_lambda = current_wav / _resolving_power

        # Update current wavelength
        current_wav += delta_lambda

        # Include the current wavelength in the array if it's within bounds
        if current_wav <= lam_max:
            wavelengths.append(current_wav)

    return np.array(wavelengths) * angstrom


def print_premade_instruments() -> None:
    """List all available premade instruments.

    This will also count the filters (if any) available for each instrument
    and check whether the cached instrument file has been downloaded or not.
    """
    # Avoid circular import
    from synthesizer.instruments import premade as instruments

    # Find all subclasses of PremadeInstrument
    instrument_classes = []
    for name, obj in inspect.getmembers(instruments):
        if (
            inspect.isclass(obj)
            and issubclass(obj, instruments.PremadeInstrument)
            and obj is not instruments.PremadeInstrument
        ):
            instrument_classes.append(obj)

    # Prepare table rows
    rows = []
    for cls in instrument_classes:
        name = cls.__name__
        try:
            num_filters = (
                len(getattr(cls, "available_filters", None))
                if getattr(cls, "available_filters", None) is not None
                else "N/A"
            )
            cache_file = getattr(cls, "_instrument_cache_file", None)
            available = (
                "Yes" if cache_file and os.path.exists(cache_file) else "No"
            )
        except Exception:
            num_filters = "N/A"
            available = "Error"
        rows.append((name, str(num_filters), available))

    # Determine column widths
    col_widths = [
        max(len(row[0]) for row in rows + [("Instrument", "", "")]),
        max(len(row[1]) for row in rows + [("", "NFilters", "")]),
        max(len(row[2]) for row in rows + [("", "", "Cached?")]),
    ]

    # Build the table
    separator = (
        f"+{'-' * (col_widths[0] + 2)}+"
        f"{'-' * (col_widths[1] + 2)}+"
        f"{'-' * (col_widths[2] + 3)}+"
    )
    header = (
        f"| {'Instrument':<{col_widths[0]}} | "
        f"{'NFilters':<{col_widths[1]}} | "
        f" {'Cached?':<{col_widths[2]}} |"
    )
    lines = [separator, header, separator]
    for row in rows:
        lines.append(
            f"| {row[0]:<{col_widths[0]}} | "
            f"{row[1]:<{col_widths[1]}} | "
            f" {row[2]:<{col_widths[2]}} |"
        )
    lines.append(separator)

    print("\n".join(lines))
