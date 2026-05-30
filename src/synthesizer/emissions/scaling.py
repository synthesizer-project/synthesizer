"""Shared helpers for scaling emission arrays.

Provides normalised mask handling and a unified scaling dispatcher that
tries the mask-specialised C++ kernels first before falling back to
NumPy broadcasting. Both ``Sed.scale`` and ``LineCollection.scale``
delegate here for the single-array path.

Example usage::

    from synthesizer.emissions.scaling import (
        normalise_scale_masks,
        scale_array,
    )

    scaled = scale_array(array, scaling, mask=mask, nthreads=4)
"""

import numpy as np

from synthesizer import exceptions
from synthesizer.extensions.spectra_operations import scale_spectra_2d


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
    if mask is None:
        return None, lam_mask

    if mask.shape == shape:
        if lam_mask is not None:
            mask = np.logical_and(mask, lam_mask)
        return mask, None

    if mask.ndim == 1 and mask.shape[0] == shape[0]:
        return mask, lam_mask

    if mask.ndim == 1 and mask.shape[0] == shape[-1]:
        if lam_mask is None:
            return None, mask
        return None, np.logical_and(mask, lam_mask)

    if lam_mask is not None and mask.ndim == 1:
        return np.logical_and(mask[:, None], lam_mask), None

    raise exceptions.InconsistentArguments(
        f"Mask shape {mask.shape} is incompatible with target shape {shape}."
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

    # Use the dedicated 2D kernel when the scaling is per-row and any masks
    # can be expressed as simple row/column selectors.
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
        return scale_spectra_2d(array, scaling, mask, lam_mask, nthreads, out)

    # Fast 2D path for scalar scaling: broadcast to per-row array so the
    # C++ kernel (with OpenMP) handles it instead of the NumPy scalar path.
    # Mask/lam_mask must be in the simple form the kernel accepts
    # (1D row/lambda).
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

    # Fast path for true in-place mutation: when ``out is array`` (same
    # buffer) we can mutate the buffer directly instead of the
    # copy-modify-copy-back dance below.  This is the common case for
    # ``Sed.scale(..., inplace=True)`` with a scalar scaling factor.
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

    if lam_mask is None:
        work = np.array(array, copy=True)
    else:
        work = np.array(array[..., lam_mask], copy=True)

    if np.isscalar(scaling):
        if mask is not None:
            work[mask] *= scaling
        else:
            work *= scaling

    elif isinstance(scaling, np.ndarray) and scaling.shape == work.shape:
        if mask is not None:
            work[mask] *= scaling[mask]
        else:
            work *= scaling

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

    elif isinstance(scaling, np.ndarray):
        work = scaling[..., np.newaxis] * work
        if mask is not None:
            raise exceptions.InconsistentMultiplication(
                "Masking is not supported for scaling arrays with "
                "different shapes"
            )

    else:
        out_str = f"Incompatible scaling factor with type {type(scaling)} "
        if hasattr(scaling, "shape"):
            out_str += f"and shape {scaling.shape}"
        else:
            out_str += f"and value {scaling}"
        raise exceptions.InconsistentMultiplication(out_str)

    if lam_mask is None:
        if out is not None:
            out[...] = work
            return out
        return work

    out_arr = np.array(array, copy=True)
    out_arr[..., lam_mask] = work
    if out is not None:
        out[...] = out_arr
        return out
    return out_arr
