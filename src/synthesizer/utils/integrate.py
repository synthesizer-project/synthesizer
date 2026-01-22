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
    # Use float64 for scaling to avoid overflow with large values
    # Ensure we copy the array so we don't modify the original in place
    # NOTE: The C extension trapz_last_axis seems to expect float64 even if
    # the package is in single precision mode, or at least requires it for
    # stability.
    _xs = np.array(xs, dtype=np.float64, copy=True)
    _ys = np.array(ys, dtype=np.float64, copy=True)

    _xs = np.ascontiguousarray(_xs)
    _ys = np.ascontiguousarray(_ys)

    # Scale the integrand and xs to avoid numerical issues
    xscale = _xs.max()
    yscale = _ys.max()

    # Handle edge cases where scale is 0 to avoid division by zero
    if xscale == 0:
        xscale = 1.0
    if yscale == 0:
        yscale = 1.0

    _xs /= xscale
    _ys /= yscale

    # Now convert to the compiled precision for the C extension
    _xs = _xs.astype(dtype)
    _ys = _ys.astype(dtype)

    integral = integration_function(_xs, _ys, nthreads)

    # Cast integral to float64 to ensure the scaling calculation doesn't
    # overflow if integral is float32 (which it will be if compiled in single
    # precision)
    return integral.astype(np.float64) * float(xscale) * float(yscale)
