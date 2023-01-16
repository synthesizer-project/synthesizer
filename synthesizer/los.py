"""
Script containing all the functions to calculate the line of sight attenuation.

...

Licence..

"""
import numpy as np
from scipy.spatial import cKDTree


def kd_los():
    """ KD-Tree flavour of LOS calculation. """


def numba_los():
    """ Original LOS calculation. Faster for smaller particle numbers. """


def calc_los(npart, kernel_func, kernel_dict):
    """ User facing wrapper for line of sight calculations.

        This will choose the best LOS implementation to apply to the
        situaition.
    """

    # If lightweight use lightweight caluclation
    if npart < 100:
        los = numba_los()
    else:
        los = kd_los()

    return los
