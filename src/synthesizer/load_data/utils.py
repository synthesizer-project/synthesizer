"""
Utilities for data loading methods
"""

import numpy as np


def get_len(Length):
    """
    Find the beginning and ending indices from a length array

    Args:
        Length (array)
            array of number of particles
    Returns:
        begin (array)
            beginning indices
        end (array)
            ending indices
    """

    begin = np.zeros(len(Length), dtype=np.int64)
    end = np.zeros(len(Length), dtype=np.int64)
    begin[1:] = np.cumsum(Length)[:-1]
    end = np.cumsum(Length)
    return begin, end


def age_lookup_table(cosmo, delta_a=1e-3, low_lim=1e-4):
    """
    Create a look-up table for age as a function of scale factor

    Args:
        cosmo (astropy.cosmology)
            astropy cosmology object
        delta_a (int)
            scale factor resolution to approximate
        low_lim (float)
            lower limit of scale factor
    Returns:
        scale_factor (array)
            array of scale factors
        age (array)
            array of ages (Gyr)
    """
    resolution = (1.0 - low_lim) / delta_a
    scale_factor = np.linspace(low_lim, 1.0, resolution)
    ages = cosmo.age(1.0 / scale_factor - 1)
    return scale_factor, ages


def lookup_age(scale_factor, scale_factors, ages):
    """
    Look up the age of a galaxy given its scale factor

    Args:
        scale_factor (array.float)
            scale factors to convert to ages
        scale_factors (array)
            array of lookup scale factors
        ages (array)
            array of lookup ages

    Returns:
        age (float)
            age of galaxy
    """
    return np.interp(scale_factor, scale_factors, ages)
