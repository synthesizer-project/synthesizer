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
    trapz_last_axis_scaled,
    trapz_last_axis_weighted,
)
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

    # Use the module-level cached dtype (avoids a C extension round-trip
    # on every call to get_numpy_dtype())
    dtype = _NUMPY_DTYPE

    # --- Trapz path: fused scale+integrate in C ---
    # The C kernel reads Float (float32 in SINGLE_PRECISION builds) inputs,
    # accumulates in double, and returns a float64 array + scale.  This
    # avoids two full Python-side passes (max-abs + divide) and the
    # float32->float64 cast on the output.
    if method == "trapz":
        _xs = np.asarray(xs)
        _ys = np.asarray(ys)
        # Ensure inputs are in the compiled Float dtype and C-contiguous
        # before handing off to C.  On the common hot-path they already are.
        if _xs.dtype != dtype or not _xs.flags["C_CONTIGUOUS"]:
            _xs = np.ascontiguousarray(_xs, dtype=dtype)
        if _ys.dtype != dtype or not _ys.flags["C_CONTIGUOUS"]:
            _ys = np.ascontiguousarray(_ys, dtype=dtype)

        # C returns (float64 integral array, double scale).
        # scale == 0 means all inputs were zero.
        integral, scale = trapz_last_axis_scaled(_xs, _ys, nthreads)
        if scale == 0.0:
            if _ys.ndim > 1:
                return np.zeros(_ys.shape[:-1], dtype=np.float64)
            return np.float64(0.0)
        # integral is already float64; multiply scale in-place.
        integral *= scale
        return integral

    # --- Simps path: fused scale+integrate in C (same as trapz) ---
    _xs = np.asarray(xs)
    _ys = np.asarray(ys)
    if _xs.dtype != dtype or not _xs.flags["C_CONTIGUOUS"]:
        _xs = np.ascontiguousarray(_xs, dtype=dtype)
    if _ys.dtype != dtype or not _ys.flags["C_CONTIGUOUS"]:
        _ys = np.ascontiguousarray(_ys, dtype=dtype)

    integral, scale = simps_last_axis_scaled(_xs, _ys, nthreads)
    if scale == 0.0:
        if _ys.ndim > 1:
            return np.zeros(_ys.shape[:-1], dtype=np.float64)
        return np.float64(0.0)
    integral *= scale
    return integral


def integrate_weighted(xs, ys, weights, nthreads=1):
    """Integrate ys * weights over the last axis using trapezoid rule.

    Fused kernel: computes ∫ ys(x)*weights(x) dx in a single C pass.
    Scaling, multiplication, and integration are all done inside C so
    that no intermediate Python arrays are allocated.

    Args:
        xs (array-like):
            1D x-values (length n).
        ys (array-like):
            ND y-values; last axis has length n.
        weights (array-like):
            1D weight vector (length n).  The integrand is ys * weights.
        nthreads (int):
            Number of threads.  -1 means all available.

    Returns:
        np.ndarray (float64):
            Integrated values, shape = ys.shape[:-1].
    """
    if nthreads == -1:
        nthreads = os.cpu_count()

    dtype = _NUMPY_DTYPE

    _xs = np.asarray(xs)
    _ys = np.asarray(ys)
    _ws = np.asarray(weights)

    if _xs.dtype != dtype or not _xs.flags["C_CONTIGUOUS"]:
        _xs = np.ascontiguousarray(_xs, dtype=dtype)
    if _ys.dtype != dtype or not _ys.flags["C_CONTIGUOUS"]:
        _ys = np.ascontiguousarray(_ys, dtype=dtype)
    if _ws.dtype != dtype or not _ws.flags["C_CONTIGUOUS"]:
        _ws = np.ascontiguousarray(_ws, dtype=dtype)

    integral, scale = trapz_last_axis_weighted(_xs, _ys, _ws, nthreads)
    if scale == 0.0:
        if _ys.ndim > 1:
            return np.zeros(_ys.shape[:-1], dtype=np.float64)
        return np.float64(0.0)
    integral *= scale
    return integral
