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
    simps_last_axis_scaled,
    simps_last_axis_weighted,
    trapz_last_axis_scaled,
    trapz_last_axis_weighted,
)
from synthesizer.utils.precision import accept_precisions

# Import trapezoid or trapz based on numpy version
if np.__version__.startswith("1."):
    from numpy import trapz as trapezoid
else:
    from numpy import trapezoid  # noqa: F401, I001


@accept_precisions()
def integrate_last_axis(xs, ys, weights=None, nthreads=1, method="trapz"):
    """Integrate the last axis of an N-dimensional array.

    This function provides a unified interface for numerical integration with
    optional weighting. When weights are provided, it computes the integral
    ∫ ys(x)*weights(x) dx in a single optimized C pass. Scaling,
    multiplication, and integration are all done inside C to avoid
    intermediate Python array allocations.

    Note that this is one of the few cases where the install precision choice
    is always ignored, i.e. SINGLE_PRECISION. This is to avoid common overflow
    issues in single precision when working in standard unit systems for
    astrophysics.

    Args:
        xs (array-like):
            The x-values to integrate over (1D array of length n).
        ys (array-like):
            The y-values to integrate (ND array where last axis has length n).
        weights (array-like, optional):
            Optional 1D weight vector (length n). When provided, the integrand
            becomes ys * weights. Default is None (no weighting).
        nthreads (int):
            The number of threads to use for the integration. If -1, all
            available threads will be used. Default is 1.
        method (str):
            The integration method to use. Options are 'trapz'
            (trapezoidal rule) or 'simps' (Simpson's rule).
            Default is 'trapz'.

    Returns:
        np.ndarray (float64):
            The result of the integration in float64 regardless of the
            input precision. Shape is ys.shape[:-1].

    Raises:
        InconsistentArguments:
            If an invalid method is passed.

    Example:
        >>> xs = np.array([0.0, 1.0, 2.0])
        >>> ys = np.array([[0.0, 1.0, 4.0], [0.0, 2.0, 8.0]])
        >>> integrate_last_axis(xs, ys, method="trapz")
        array([2.5, 5.0])
        >>> weights = np.array([1.0, 2.0, 1.0])
        >>> integrate_last_axis(xs, ys, weights=weights, method="trapz")
        array([3.0, 6.0])
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

    # Determine if we're doing weighted integration
    if weights is not None:
        # Weighted integration path
        if method == "trapz":
            integral, scale = trapz_last_axis_weighted(
                xs, ys, weights, nthreads
            )
        else:  # method == "simps"
            integral, scale = simps_last_axis_weighted(
                xs, ys, weights, nthreads
            )
    else:
        # Unweighted integration path
        if method == "trapz":
            integral, scale = trapz_last_axis_scaled(xs, ys, nthreads)
        else:  # method == "simps"
            integral, scale = simps_last_axis_scaled(xs, ys, nthreads)

    # Handle the case where the scale is zero, i.e. the result is zero.
    # Note: The result of integration is always float64, even if the input is
    # float32. This is because the C code always accumulates in float64 to
    # avoid precision issues. The scale factor is also float64.
    if scale == 0.0:
        if ys.ndim > 1:
            return np.zeros(ys.shape[:-1], dtype=np.float64)
        return np.float64(0.0)

    # Reapply the scaling and return the result.
    integral *= scale
    return integral
