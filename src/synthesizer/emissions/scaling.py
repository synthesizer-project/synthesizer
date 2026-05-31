"""Shared helpers for scaling emission arrays.

Provides normalised mask handling and a unified scaling dispatcher that
uses the specialised C++ kernels where they help, while keeping simple
no-mask broadcasts on the NumPy path. Both ``Sed.scale`` and
``LineCollection.scale`` delegate here for the single-array path.

Example usage::

    from synthesizer.emissions.scaling import (
        normalise_scale_masks,
        scale_array,
    )

    scaled = scale_array(array, scaling, mask=mask, nthreads=4)
"""

import numpy as np
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.extensions.spectra_operations import (
    scale_line_2d,
    scale_spectra_2d,
)
from synthesizer.units import get_array_quantity_view


def normalise_scale_masks(mask, lam_mask, shape):
    """Normalise row and wavelength masks for scaling helpers.

    Args:
        mask (np.ndarray or None):
            Optional mask supplied by the caller.
        lam_mask (np.ndarray or None):
            Optional explicit wavelength mask.
        shape (tuple):
            Target array shape.

    Returns:
        tuple:
            ``(mask, lam_mask)`` in a form the scaling helper understands.
    """
    if lam_mask is not None:
        lam_mask = np.asarray(lam_mask)

    # No mask at all — nothing to normalise, return as-is
    if mask is None:
        return None, lam_mask

    mask = np.asarray(mask)

    # The mask matches the full array shape so it's an element-level mask.
    # If we also have a wavelength mask we can combine them into one
    if mask.shape == shape:
        if lam_mask is not None:
            mask = np.logical_and(mask, lam_mask)
        return mask, None

    # The mask is 1D and matches the row count so it works as a row selector
    # with an optional wavelength mask alongside
    if mask.ndim == 1 and mask.shape[0] == shape[0]:
        return mask, lam_mask

    # The mask is 1D and matches the wavelength count so it works as a
    # wavelength selector. If we also have a lam_mask we combine them
    if mask.ndim == 1 and mask.shape[0] == shape[-1]:
        if lam_mask is None:
            return None, mask
        return None, np.logical_and(mask, lam_mask)

    # The mask is 1D but doesn't match either dimension — we can still
    # use it if we have an explicit lam_mask by broadcasting row-wise
    if lam_mask is not None and mask.ndim == 1:
        return np.logical_and(mask[:, None], lam_mask), None

    # None of the known shapes matched — the caller gave us something
    # incompatible
    raise exceptions.InconsistentArguments(
        f"Mask shape {mask.shape} is incompatible with target shape {shape}."
    )


def normalise_scaling_for_units(scaling, units):
    """Convert a scaling factor into raw values compatible with ``units``.

    Args:
        scaling (float, np.ndarray, or unyt quantity):
            Scaling factor supplied by the caller.
        units (unyt.Unit):
            Units expected by the target array.

    Returns:
        float or np.ndarray:
            Raw scaling values with any units stripped.
    """
    if not isinstance(scaling, (unyt_array, unyt_quantity)):
        return scaling

    if units.dimensions != scaling.units.dimensions:
        raise exceptions.InconsistentMultiplication(
            f"Incompatible units {units} and {scaling.units}"
        )

    return scaling.to(units).value


def normalise_line_scaling(scaling, get_nu, lum_units, cont_units):
    """Resolve a line scaling into luminosity and continuum factors.

    Line luminosity and continuum carry different physical units, so a single
    unit-bearing scaling may need to be converted into two different raw
    arrays. The frequency coordinate is only constructed when we actually need
    one of those unit conversions.

    Args:
        scaling (float, np.ndarray, or unyt quantity):
            Scaling factor supplied by the caller.
        get_nu (Callable[[], unyt_array]):
            Callable returning the frequency coordinate for the lines.
        lum_units (unyt.Unit):
            Units of the luminosity array.
        cont_units (unyt.Unit):
            Units of the continuum array.

    Returns:
        tuple:
            ``(scaling_lum, scaling_cont)`` as raw values.
    """
    if not isinstance(scaling, (unyt_array, unyt_quantity)):
        return scaling, scaling

    nu = get_nu()

    if cont_units.dimensions == scaling.units.dimensions:
        scaling_cont = scaling.to(cont_units).value
        scaling_lum = (scaling * nu).to(lum_units).value
        return scaling_lum, scaling_cont

    if lum_units.dimensions == scaling.units.dimensions:
        scaling_lum = scaling.to(lum_units).value
        scaling_cont = (scaling / nu).to(cont_units).value
        return scaling_lum, scaling_cont

    raise exceptions.InconsistentMultiplication(
        f"{scaling.units} is neither compatible with the "
        f"continuum ({cont_units}) nor the luminosity ({lum_units})"
    )


def scale_to_quantity(
    array,
    scaling,
    units,
    mask=None,
    lam_mask=None,
    nthreads=1,
):
    """Scale an array and wrap the result in units without copying again.

    Args:
        array (np.ndarray):
            Raw array to scale.
        scaling (float or np.ndarray):
            Raw scaling values.
        units (unyt.Unit):
            Units to attach to the scaled result.
        mask (np.ndarray or None):
            Optional row or element mask.
        lam_mask (np.ndarray or None):
            Optional wavelength mask.
        nthreads (int):
            The number of OpenMP threads available to compatible kernels.

    Returns:
        unyt_array:
            Scaled result wrapped in ``units``.
    """
    return get_array_quantity_view(
        scale_array(
            array,
            scaling,
            mask=mask,
            lam_mask=lam_mask,
            nthreads=nthreads,
        ),
        units,
    )


def scale_inplace(array, scaling, mask=None, lam_mask=None, nthreads=1):
    """Scale an array into its existing buffer.

    Args:
        array (np.ndarray):
            Raw array to mutate.
        scaling (float or np.ndarray):
            Raw scaling values.
        mask (np.ndarray or None):
            Optional row or element mask.
        lam_mask (np.ndarray or None):
            Optional wavelength mask.
        nthreads (int):
            The number of OpenMP threads available to compatible kernels.

    Returns:
        np.ndarray:
            The mutated ``array`` buffer.
    """
    return scale_array(
        array,
        scaling,
        mask=mask,
        lam_mask=lam_mask,
        nthreads=nthreads,
        out=array,
    )


def scale_line_arrays(
    luminosity,
    continuum,
    scaling_lum,
    scaling_cont,
    mask=None,
    lam_mask=None,
    nthreads=1,
    out_lum=None,
    out_cont=None,
):
    """Scale a luminosity/continuum pair, using the fused kernel when useful.

    This keeps the Python-side dispatch rules for line scaling in one place.
    The fused C++ kernel is only worthwhile for the 2D masked row-scaling
    case; otherwise we delegate to ``scale_array`` for each array.

    Args:
        luminosity (np.ndarray):
            Raw luminosity array.
        continuum (np.ndarray):
            Raw continuum array.
        scaling_lum (float or np.ndarray):
            Raw luminosity scaling values.
        scaling_cont (float or np.ndarray):
            Raw continuum scaling values.
        mask (np.ndarray or None):
            Optional row or element mask.
        lam_mask (np.ndarray or None):
            Optional wavelength mask.
        nthreads (int):
            The number of OpenMP threads available to compatible kernels.
        out_lum (np.ndarray or None):
            Optional output buffer for luminosity.
        out_cont (np.ndarray or None):
            Optional output buffer for continuum.

    Returns:
        tuple:
            Tuple of scaled ``(luminosity, continuum)`` arrays.
    """
    mask, lam_mask = normalise_scale_masks(mask, lam_mask, luminosity.shape)

    nspec = luminosity.shape[0]
    nlam = luminosity.shape[-1]
    lum_1d = isinstance(scaling_lum, np.ndarray) and scaling_lum.ndim == 1
    cont_1d = isinstance(scaling_cont, np.ndarray) and scaling_cont.ndim == 1
    use_fused = (
        luminosity.ndim == 2
        and lum_1d
        and cont_1d
        and scaling_lum.shape[0] == nspec
        and scaling_cont.shape[0] == nspec
        and (mask is not None or lam_mask is not None)
        and (mask is None or (mask.ndim == 1 and mask.shape[0] == nspec))
        and (
            lam_mask is None
            or (lam_mask.ndim == 1 and lam_mask.shape[0] == nlam)
        )
    )

    # If the fused kernel matches the shapes we use it, otherwise we fall back
    # to two independent array scales and let ``scale_array`` pick the best
    # path for each one.
    if use_fused:
        return scale_line_2d(
            luminosity,
            continuum,
            scaling_lum,
            scaling_cont,
            mask,
            lam_mask,
            nthreads,
            out_lum,
            out_cont,
        )

    return (
        scale_array(
            luminosity,
            scaling_lum,
            mask=mask,
            lam_mask=lam_mask,
            nthreads=nthreads,
            out=out_lum,
        ),
        scale_array(
            continuum,
            scaling_cont,
            mask=mask,
            lam_mask=lam_mask,
            nthreads=nthreads,
            out=out_cont,
        ),
    )


def scale_array(
    array, scaling, mask=None, lam_mask=None, nthreads=1, out=None
):
    """Scale an emission array with optional row and wavelength masks.

    Args:
        array (np.ndarray):
            The array to scale.
        scaling (float or np.ndarray):
            The scaling to apply.
        mask (np.ndarray or None):
            Optional row or element mask.
        lam_mask (np.ndarray or None):
            Optional wavelength mask.
        nthreads (int):
            The number of OpenMP threads available to compatible kernels.
        out (np.ndarray or None):
            Optional output buffer. When provided and compatible with the fast
            2D path the result is written directly into this array, avoiding
            an allocation.

    Returns:
        np.ndarray:
            The scaled array (may be ``out`` if the fast path was used).
    """
    scaling_ndim = getattr(scaling, "ndim", 0)

    # With no masks a scalar multiply is already the simple fast path.
    # Avoid building a temporary row vector and dispatching into C++.
    if (
        array.ndim == 2
        and np.isscalar(scaling)
        and mask is None
        and lam_mask is None
    ):
        if out is not None:
            np.multiply(array, scaling, out=out)
            return out
        return array * scaling

    # If the scaling is a 1D array that matches the first dimension of the
    # array we can use the dedicated 2D C++ kernel. This will handle per-
    # spectrum scaling with optional row and wavelength masks for us
    use_fast_2d_scaling = (
        isinstance(scaling, np.ndarray)
        and array.ndim == 2
        and scaling_ndim == 1
        and scaling.shape[0] == array.shape[0]
        and (
            mask is None
            or (
                getattr(mask, "ndim", 0) == 1
                and mask.shape[0] == array.shape[0]
            )
        )
        and (
            lam_mask is None
            or (
                getattr(lam_mask, "ndim", 0) == 1
                and lam_mask.shape[0] == array.shape[-1]
            )
        )
    )
    if use_fast_2d_scaling:
        # Without masks the C++ dispatch overhead outweighs its benefit.
        # NumPy broadcasting (scaling[:, np.newaxis]) is a strided view with
        # zero allocation and matches the C++ kernel's element throughput.
        if mask is None and lam_mask is None:
            scaling_2d = scaling[:, np.newaxis]
            if out is not None:
                np.multiply(array, scaling_2d, out=out)
                return out
            return array * scaling_2d
        return scale_spectra_2d(array, scaling, mask, lam_mask, nthreads, out)

    # If the scaling is a scalar and the array is 2D we can broadcast it to a
    # per-row array and use the C++ kernel for the masked cases. The masks
    # must be in the simple form the kernel accepts (1D row/lambda).
    if (
        array.ndim == 2
        and np.isscalar(scaling)
        and (
            mask is None
            or (
                isinstance(mask, np.ndarray)
                and mask.ndim == 1
                and mask.shape[0] == array.shape[0]
            )
        )
        and (
            lam_mask is None
            or (
                isinstance(lam_mask, np.ndarray)
                and lam_mask.ndim == 1
                and lam_mask.shape[0] == array.shape[-1]
            )
        )
    ):
        scaling_arr = np.empty(array.shape[0], dtype=float)
        scaling_arr.fill(scaling)
        return scale_spectra_2d(
            array, scaling_arr, mask, lam_mask, nthreads, out
        )

    # If we are scaling in place (out is the same buffer as array) we can
    # mutate the buffer directly without the copy-modify-copy-back dance at
    # the end. This is the common case for Sed.scale with inplace=True
    if out is not None and out is array and lam_mask is None:
        if np.isscalar(scaling):
            if mask is not None:
                array[mask] *= scaling
            else:
                array *= scaling
            return array
        if isinstance(scaling, np.ndarray) and scaling_ndim == 1:
            if scaling.shape[0] == array.shape[0]:
                array *= scaling[:, np.newaxis] if array.ndim == 2 else scaling
                return array
            if scaling.shape[0] == array.shape[-1]:
                array *= scaling
                return array

    # We couldn't use any of the fast paths — fall back to NumPy.
    # Start by making a copy of the array so we don't modify the original.
    # If we have a wavelength mask we only copy the relevant columns
    if lam_mask is None:
        work = np.array(array, copy=True)
    else:
        work = array[..., lam_mask]

    # Scalar scaling — multiply every element by the same number, with
    # or without a row/element mask
    if np.isscalar(scaling):
        if mask is not None:
            work[mask] *= scaling
        else:
            work *= scaling

    # The scaling array has the same shape as the (possibly masked) work
    # array so we can multiply element-by-element
    elif isinstance(scaling, np.ndarray) and scaling.shape == work.shape:
        if mask is not None:
            work[mask] *= scaling[mask]
        else:
            work *= scaling

    # The scaling is a 1D array matching the last dimension — this is the
    # per-wavelength case where each wavelength gets a different factor
    # applied to all rows
    elif (
        isinstance(scaling, np.ndarray)
        and scaling_ndim == 1
        and scaling.shape[0] == work.shape[-1]
    ):
        if mask is None:
            work *= scaling
        elif getattr(mask, "ndim", 0) == 1:
            work[mask] *= scaling
        else:
            work[mask] *= np.broadcast_to(scaling, work.shape)[mask]

    # The scaling has fewer dimensions than the work array — we expand
    # it to match by inserting dimensions at the end, then broadcast
    elif isinstance(scaling, np.ndarray) and scaling_ndim < work.ndim:
        expand_axes = tuple(range(scaling_ndim, work.ndim))
        expanded_scaling = np.expand_dims(scaling, axis=expand_axes)
        if mask is not None:
            if expanded_scaling.shape == work.shape:
                work[mask] *= expanded_scaling[mask]
            else:
                work[mask] *= expanded_scaling
        else:
            work *= expanded_scaling

    # The shapes are completely different — let NumPy figure out the
    # broadcast with an explicit trailing dimension. Masking is not
    # supported in this case since the shapes don't align
    elif isinstance(scaling, np.ndarray):
        work = scaling[..., np.newaxis] * work
        if mask is not None:
            raise exceptions.InconsistentMultiplication(
                "Masking is not supported for scaling arrays with "
                "different shapes"
            )

    # We don't know how to handle this type of scaling at all
    else:
        out_str = f"Incompatible scaling factor with type {type(scaling)} "
        if hasattr(scaling, "shape"):
            out_str += f"and shape {scaling.shape}"
        else:
            out_str += f"and value {scaling}"
        raise exceptions.InconsistentMultiplication(out_str)

    # If we didn't use a wavelength mask the work array is a direct copy
    # of the full array and we can write it back as-is (or return it)
    if lam_mask is None:
        if out is not None:
            out[...] = work
            return out
        return work

    # With a wavelength mask the work array is smaller than the original
    # (only the masked columns). We need to create a fresh copy of the
    # original and place the scaled columns back in the right positions
    out_arr = np.array(array, copy=True)
    out_arr[..., lam_mask] = work
    if out is not None:
        out[...] = out_arr
        return out
    return out_arr
