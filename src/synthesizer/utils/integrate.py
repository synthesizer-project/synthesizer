"""A module containing integration helper functions.

This module contains functions that help with numerical integration. These
functions wrap C extensions and abstract away boilerplate code (i.e. deciding
which integration method to use, etc.).

Example:
    integrate_last_axis(xs, ys, nthreads=1, method="trapz")
"""

import os

import numpy as np

from synthesizer import exceptions
from synthesizer.extensions.integration import (
    simps_last_axis,
    trapz_last_axis,
    weighted_simps_last_axis,
    weighted_trapz_last_axis,
)

# Import trapezoid or trapz based on numpy version
if np.__version__.startswith("1."):
    from numpy import trapz as trapezoid
else:
    from numpy import trapezoid  # noqa: F401, I001


def integrate_last_axis(xs, ys, nthreads=1, method="trapz"):
    """Integrate the last axis of an N-dimensional array.

    Args:
        xs (array-like):
            The x-values to integrate over.
        ys (array-like):
            The y-values to integrate.
        nthreads (int):
            The number of threads to use for the integration. If -1, all
            available threads will be used.
        method (str):
            The integration method to use. Options are 'trapz' or
            'simps'.

    Returns:
        array-like:
            The result of the integration.

    Raises:
        InconsistentArguments:
            If an invalid method is passed.
    """
    # Ensure we have been asked for a valid method
    if method not in ["trapz", "simps"]:
        raise exceptions.InconsistentArguments(
            f"Unrecognised integration method ({method}). "
            "Options are 'trapz' or 'simps'"
        )

    # Handle nthreads
    if nthreads == -1:
        nthreads = os.cpu_count()

    integration_function = (
        trapz_last_axis if method == "trapz" else simps_last_axis
    )

    # Ensure arrays are C-contiguous float64 for C extension safety/performance
    _xs = np.ascontiguousarray(xs, dtype=np.float64)
    _ys = np.ascontiguousarray(ys, dtype=np.float64)

    # If either input is empty or trivially zero, return zeros
    if _xs.size == 0 or _ys.size == 0:
        out_shape = ys.shape[:-1]
        return np.zeros(out_shape) if out_shape else 0.0

    if _xs.max() == 0 or _ys.max() == 0:
        out_shape = ys.shape[:-1]
        return np.zeros(out_shape) if out_shape else 0.0

    return integration_function(_xs, _ys, nthreads)


def integrate_weighted_last_axis(xs, ys, weights, nthreads=1, method="trapz"):
    """Compute a weighted average over the final axis of an ND array.

    This computes:
        integral(ys * weights, xs) / integral(weights, xs)

    in a single C-extension pass over ys.

    Args:
        xs (array-like):
            The x-values to integrate over.
        ys (array-like):
            The y-values to integrate over the final axis.
        weights (array-like):
            1D weights defined over xs.
        nthreads (int):
            Number of threads to use. If -1, all available threads are used.
        method (str):
            Integration method: 'trapz' or 'simps'.

    Returns:
        array-like:
            Weighted average over the final axis.

    Raises:
        InconsistentArguments:
            If an invalid method is passed.
    """
    if method not in ["trapz", "simps"]:
        raise exceptions.InconsistentArguments(
            f"Unrecognised integration method ({method}). "
            "Options are 'trapz' or 'simps'"
        )

    if nthreads == -1:
        nthreads = os.cpu_count()

    integration_function = (
        weighted_trapz_last_axis
        if method == "trapz"
        else weighted_simps_last_axis
    )

    _xs = np.ascontiguousarray(xs, dtype=np.float64)
    _ys = np.ascontiguousarray(ys, dtype=np.float64)
    _weights = np.ascontiguousarray(weights, dtype=np.float64)

    if _xs.size == 0 or _ys.size == 0 or _weights.size == 0:
        out_shape = ys.shape[:-1]
        return np.zeros(out_shape) if out_shape else 0.0

    if _weights.max() == 0:
        out_shape = ys.shape[:-1]
        return np.zeros(out_shape) if out_shape else 0.0

    return integration_function(_xs, _ys, _weights, nthreads)
