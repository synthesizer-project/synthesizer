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
from synthesizer.extensions.integration import simps_last_axis, trapz_last_axis
from synthesizer.utils import precision

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

    # Get the target dtype for precision
    dtype = precision.get_numpy_dtype()

    # We need to make a copy of xs and ys and convert to the correct precision
    _xs = np.array(xs, dtype=dtype, copy=True)
    _ys = np.array(ys, dtype=dtype, copy=True)

    # Scale the integrand and xs to avoid numerical issues
    xscale = np.max(np.abs(_xs))
    yscale = np.max(np.abs(_ys))

    # If the maximum is zero, we return zero
    if xscale == 0 or yscale == 0:
        ndim = ys.ndim - 1
        return np.zeros(ndim, dtype=dtype) if ndim > 0 else 0.0

    # Scale to internal units (0..1)
    _xs /= xscale
    _ys /= yscale

    # Perform integration and rescale result
    # We cast to double precision here to ensure the scaling calculation itself
    # doesn't overflow float32 limits if the result is large.
    return (
        integration_function(_xs, _ys, nthreads).astype(float)
        * float(xscale)
        * float(yscale)
    )
