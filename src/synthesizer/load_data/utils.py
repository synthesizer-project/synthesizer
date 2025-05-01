"""Utilities for data loading methods.

These utilities are used through the load_data module as helpers for loading
data from different simulations sources.

Examples usage:

    lengths = np.array([10, 20, 30])
    begin, end = get_begin_end_pointers(lengths)
    print(begin)  # Output: [ 0 10 30]

    table = age_lookup_table(ages, redshift=0.5, delta_a=0.1)
    print(table)  # Output: (array([0.1, 0.2, 0.3]), array([10., 20., 30.]))

    ages = lookup_age(0.2, table[0], table[1])
"""

import math

import numpy as np


def get_begin_end_pointers(length):
    """Find the beginning and ending indices from a length array.

    Args:
        length (np.ndarray of int):
            The number of particles in each galaxy.

    Returns:
        begin (np.ndarray of int): Beginning indices.
        end (np.ndarray of int): Ending indices.
    """
    begin = np.zeros(len(length), dtype=np.int64)
    end = np.zeros(len(length), dtype=np.int64)
    begin[1:] = np.cumsum(length)[:-1]
    end = np.cumsum(length)
    return begin, end


def age_lookup_table(cosmo, redshift=0.0, delta_a=1e-3, low_lim=1e-4):
    """Create a look-up table for age as a function of scale factor.

    Defaults to start at the lower resolution limit (`delta_a`),
    and proceeds in steps of `delta-a` until the scale factor given
    by the input `redshift` minus the `low_lim`.

    Args:
        cosmo (astropy.cosmology):
            Astropy cosmology object.
        redshift (float):
            Redshift of the snapshot.
        delta_a (int):
            Scale factor resolution to approximate.
        low_lim (float):
            Lower limit of scale factor.

    Returns:
        scale_factor (np.ndarray of float):
            Array of scale factors.
        age (unyt_array of float):
            Array of ages (Gyr).
    """
    # Find the scale factor for the input snapshot
    root_scale_factor = 1.0 / (1.0 + redshift)

    # Find the (integer) resolution of the grid
    resolution = (root_scale_factor - low_lim) / delta_a
    resolution = math.ceil(resolution)

    # Create the (linear) scale factor array
    scale_factor = np.linspace(
        delta_a, root_scale_factor - low_lim, resolution
    )

    # Find the ages at these scale factors
    ages = cosmo.age(1.0 / scale_factor - 1)

    return scale_factor, ages


def lookup_age(scale_factor, scale_factors, ages):
    """Look up the age given a scale factor.

    Args:
        scale_factor (np.ndarray of float):
            Scale factors to convert to ages.
        scale_factors (np.ndarray of float):
            Array of lookup scale factors.
        ages (unyt_array of float):
            Array of lookup ages.

    Returns:
        age (float): Age based on the input scale factor/s.
    """
    return np.interp(scale_factor, scale_factors, ages)
