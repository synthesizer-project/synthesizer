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
from synthesizer.utils.precision import _NUMPY_DTYPE

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

    # Use the module-level cached dtype (avoids a C extension round-trip
    # on every call to get_numpy_dtype())
    dtype = _NUMPY_DTYPE

    # Scale the integrand and xs to avoid numerical issues.
    # We do this calculation in the input's precision to avoid overflow
    # before we have a chance to normalize.
    xscale = np.max(np.abs(xs))
    yscale = np.max(np.abs(ys))

    # If the maximum is zero, we return zero
    if xscale == 0 or yscale == 0:
        ndim = ys.ndim - 1
        return (
            np.zeros(ndim, dtype=ys.dtype) if ndim > 0 else ys.dtype.type(0.0)
        )

    # Create normalized arrays in the compiled precision for the C extension.
    # Pre-allocate output buffers so the division + dtype cast happen in a
    # single pass rather than allocating an intermediate float64 array from
    # the division and then copying into float32 via ascontiguousarray.
    if np.issubdtype(np.asarray(xs).dtype, np.floating) and (
        np.asarray(xs).dtype == dtype
    ):
        # xs is already the right dtype; if contiguous just scale in-place
        # into a fresh buffer of the same type.
        _xs_raw = np.asarray(xs)
        if _xs_raw.flags["C_CONTIGUOUS"]:
            _xs = np.empty_like(_xs_raw)
            np.divide(_xs_raw, xscale, out=_xs, casting="same_kind")
        else:
            _xs = np.ascontiguousarray(_xs_raw / xscale, dtype=dtype)
    else:
        _xs = np.array(np.asarray(xs) / xscale, dtype=dtype, order="C")

    _ys_raw = np.asarray(ys)
    if _ys_raw.dtype == dtype and _ys_raw.flags["C_CONTIGUOUS"]:
        _ys = np.empty_like(_ys_raw)
        np.divide(_ys_raw, yscale, out=_ys, casting="same_kind")
    else:
        _ys = np.array(_ys_raw / yscale, dtype=dtype, order="C")

    # Perform the integration in the compiled precision
    integral = integration_function(
        _xs,
        _ys,
        nthreads,
    )

    # Rescale the result back to physical units.
    # We cast to double precision here to ensure the scaling calculation itself
    # doesn't overflow float32 limits if the result is large.
    # The C extension returns a freshly-allocated array so we can view it as
    # float64 without a copy when already in double; otherwise we cast once.
    scale = float(xscale) * float(yscale)
    if integral.dtype == np.float64:
        integral *= scale
        return integral
    return integral.astype(np.float64) * scale
