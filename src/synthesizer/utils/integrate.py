"""A module containing integration helper functions.

This module contains functions that help with numerical integration. These
functions wrap C extensions and abstract away boilerplate code (i.e. deciding
which integration method to use, etc.).

Example:
    integrate_last_axis(xs, ys, nthreads=1, method="trapz")
"""

import os

from synthesizer import exceptions
from synthesizer.extensions.integration import simps_last_axis, trapz_last_axis


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

    # We need to make a copy of xs and ys to avoid modifying in place
    _xs = xs.copy()
    _ys = ys.copy()

    # Scale the integrand and xs to avoid numerical issues
    xscale = _xs.max()
    yscale = _ys.max()
    _xs /= xscale
    _ys /= yscale

    return integration_function(_xs, _ys, nthreads) * xscale * yscale
