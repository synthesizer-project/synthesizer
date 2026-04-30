"""Photometric noise generation for synthesizer imaging.

This module contains functions for modelling and generating noise in
photometric images, including correlated noise estimated from observed
noise templates via power-spectrum analysis.

The correlated noise pipeline has two stages that are deliberately separated
to allow caching:

1. **Estimation** (``_estimate_correlated_noise_cf``): computes the correlation
   function (CF) of a noise template via FFT.  This is the expensive step and
   only needs to run once per template.

2. **Generation** (``_generate_correlated_noise``): uses a pre-computed CF to
   produce a noise realisation for an image of the target size.  This is cheap
   and is called once per image.

These functions are intended to be used internally by
``Instrument.apply_noise`` / ``Instrument.apply_noises``.  Direct use by end
users is possible but not required.
"""

import numpy as np

from synthesizer import exceptions
from synthesizer.synth_warnings import warn


def _cf_periodicity_dilution_correction_standalone(
    cf_shape,
) -> np.ndarray:
    """Calculate the correction factor for DFT-based Correlation Functions.

    Accounts for the assumption of periodicity inherent in the DFT.
    Ported from an internal GalSim function.

    Args:
        cf_shape (tuple):
            Tuple (Ny, Nx) representing the shape of the correlation function
            array.

    Returns:
        np.ndarray:
            A 2D array with the correction factors.
    """
    ny, nx = cf_shape

    dx_coords = np.fft.fftfreq(nx) * float(nx)
    dy_coords = np.fft.fftfreq(ny) * float(ny)
    deltax, deltay = np.meshgrid(dx_coords, dy_coords)

    denominator = (nx - np.abs(deltax)) * (ny - np.abs(deltay))
    valid_denominator = np.where(denominator == 0, 1.0, denominator)
    correction = (float(nx * ny)) / valid_denominator
    if np.any(denominator == 0):
        correction[denominator == 0] = 0

    return correction


def _generate_noise_from_rootps_standalone(
    rng: np.random.Generator, shape, rootps: np.ndarray
) -> np.ndarray:
    """Generate a real-space noise field from its sqrt(PowerSpectrum).

    Ported and adapted from an internal GalSim function.

    Args:
        rng (np.random.Generator):
            NumPy random number generator.
        shape (tuple):
            Tuple (Ny, Nx) of the output real-space noise field.
        rootps (np.ndarray):
            The half-complex 2D array (Ny, Nx//2 + 1) representing the square
            root of the Power Spectrum from rfft2.

    Returns:
        np.ndarray:
            A 2D array representing the generated correlated noise field.
    """
    ny, nx = shape
    sigma_val_for_gvec_parts = np.sqrt(0.5 * ny * nx)

    gvec_real = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec_imag = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec = gvec_real + 1j * gvec_imag

    rt2 = np.sqrt(2.0)

    # DC component
    gvec[0, 0] = rt2 * gvec[0, 0].real

    # Nyquist terms (enforce Hermitian symmetry for real output)
    if ny % 2 == 0:
        gvec[ny // 2, 0] = rt2 * gvec[ny // 2, 0].real
    if nx % 2 == 0:
        gvec[0, nx // 2] = rt2 * gvec[0, nx // 2].real
    if ny % 2 == 0 and nx % 2 == 0:
        gvec[ny // 2, nx // 2] = rt2 * gvec[ny // 2, nx // 2].real

    if ny > 1:
        gvec[ny - 1 : ny // 2 : -1, 0] = np.conj(gvec[1 : (ny + 1) // 2, 0])
    if nx % 2 == 0:
        kx_nyq_idx = nx // 2
        if ny > 1:
            gvec[ny - 1 : ny // 2 : -1, kx_nyq_idx] = np.conj(
                gvec[1 : (ny + 1) // 2, kx_nyq_idx]
            )

    noise_field_k_space = gvec * rootps
    noise_real_space = np.fft.irfft2(noise_field_k_space, s=shape)
    return noise_real_space


def _estimate_correlated_noise_cf(
    source_arr: np.ndarray,
    subtract_mean: bool = False,
    correct_periodicity: bool = True,
) -> np.ndarray:
    """Estimate the correlation function (CF) of a 2D noise array.

    Computes an unrolled correlation function (DFT convention, origin at
    [0, 0]) from the power spectrum of the source array.  The result is
    intended to be cached on the ``Instrument`` and later consumed by
    ``_generate_correlated_noise`` to produce noise realisations for images
    of arbitrary shape.

    This function is deliberately separated from noise generation so that the
    expensive FFT step can be computed once (e.g. per instrument filter) and
    reused across many images.

    Args:
        source_arr (np.ndarray):
            A 2D float array of observed noise (e.g. a blank-sky cutout).
            Units must be stripped before calling.
        subtract_mean (bool):
            If True the DC component of the power spectrum is zeroed, removing
            the mean offset from the noise model.
        correct_periodicity (bool):
            If True a correction factor is applied to compensate for the
            periodicity assumption of the DFT.

    Returns:
        np.ndarray:
            A 2D array of the same shape as ``source_arr`` holding the
            unrolled CF with the origin at [0, 0].
    """
    if source_arr.ndim != 2:
        raise ValueError("source_arr must be a 2D numpy array.")

    source_shape = source_arr.shape
    ft_array = np.fft.rfft2(source_arr)
    ps_array = np.abs(ft_array) ** 2
    ps_array /= np.prod(source_shape)

    if subtract_mean:
        ps_array[0, 0] = 0.0

    cf_array = np.fft.irfft2(ps_array, s=source_shape)

    if correct_periodicity:
        correction = _cf_periodicity_dilution_correction_standalone(
            source_shape
        )
        cf_array *= correction

    return cf_array


def _generate_correlated_noise(
    cf_source: np.ndarray,
    target_shape: tuple,
    rng_seed: int = None,
) -> np.ndarray:
    """Generate a correlated noise realisation for a given target shape.

    Uses a pre-computed correlation function (CF) — typically returned by
    ``_estimate_correlated_noise_cf`` and cached on the ``Instrument`` — to
    produce a noise field whose spatial statistics match those of the original
    noise template.  The CF is resampled to the target pixel dimensions before
    noise generation.

    Args:
        cf_source (np.ndarray):
            The unrolled 2D CF (DFT convention, origin at [0, 0]) previously
            returned by ``_estimate_correlated_noise_cf``.
        target_shape (tuple of int):
            The ``(ny, nx)`` pixel dimensions of the desired noise field.
        rng_seed (int, optional):
            Seed for the random number generator.  Pass the same seed to
            reproduce an identical noise realisation.

    Returns:
        np.ndarray:
            A 2D float array of shape ``target_shape`` containing the
            generated correlated noise field.

    Raises:
        InconsistentArguments:
            If the source CF is too small to contribute any pixels to the
            target CF grid.
    """
    rng = np.random.default_rng(rng_seed)
    source_shape = cf_source.shape

    if source_shape[0] < target_shape[0] or source_shape[1] < target_shape[1]:
        warn(
            "Source noise map is smaller than the target image, which may"
            " result in truncation of the noise model."
        )

    cf_source_rolled = np.roll(
        cf_source,
        shift=(source_shape[0] // 2, source_shape[1] // 2),
        axis=(0, 1),
    )

    cf_target_rolled = np.zeros(target_shape, dtype=float)

    src_cy, src_cx = source_shape[0] // 2, source_shape[1] // 2
    trg_cy, trg_cx = target_shape[0] // 2, target_shape[1] // 2

    y_start_src = max(0, src_cy - trg_cy)
    y_end_src = min(source_shape[0], src_cy + (target_shape[0] - trg_cy))
    x_start_src = max(0, src_cx - trg_cx)
    x_end_src = min(source_shape[1], src_cx + (target_shape[1] - trg_cx))

    y_start_trg = max(0, trg_cy - src_cy)
    y_end_trg = min(target_shape[0], trg_cy + (source_shape[0] - src_cy))
    x_start_trg = max(0, trg_cx - src_cx)
    x_end_trg = min(target_shape[1], trg_cx + (source_shape[1] - src_cx))

    dy = min(y_end_src - y_start_src, y_end_trg - y_start_trg)
    dx = min(x_end_src - x_start_src, x_end_trg - x_start_trg)

    if dy <= 0 or dx <= 0:
        raise exceptions.InconsistentArguments(
            "Source CF is too small to contribute to the target CF. "
            "Ensure the source noise map is large enough to model noise "
            "for the target image."
        )

    cf_target_rolled[
        y_start_trg : y_start_trg + dy, x_start_trg : x_start_trg + dx
    ] = cf_source_rolled[
        y_start_src : y_start_src + dy, x_start_src : x_start_src + dx
    ]

    cf_on_target_grid_unrolled = np.roll(
        cf_target_rolled,
        shift=(-(target_shape[0] // 2), -(target_shape[1] // 2)),
        axis=(0, 1),
    )

    ps_target_fft = np.fft.rfft2(cf_on_target_grid_unrolled)
    rootps_target = np.sqrt(np.abs(ps_target_fft))

    return _generate_noise_from_rootps_standalone(
        rng, target_shape, rootps_target
    )


def _model_and_apply_correlated_noise(
    source_image_arr: np.ndarray,
    target_image_arr: np.ndarray,
    subtract_mean: bool = False,
    correct_periodicity: bool = True,
    rng_seed: int = None,
) -> np.ndarray:
    """Model correlated noise from a source image and apply it to a target.

    Convenience wrapper around ``_estimate_correlated_noise_cf`` and
    ``_generate_correlated_noise``.  Prefer calling those two functions
    directly when the correlation function should be cached across multiple
    calls.

    Args:
        source_image_arr (np.ndarray):
            A 2D array used to model the correlated noise characteristics.
        target_image_arr (np.ndarray):
            A 2D array to which the generated correlated noise is added.
        subtract_mean (bool):
            If True the DC component is zeroed before estimating the CF.
        correct_periodicity (bool):
            If True a periodicity-dilution correction is applied.
        rng_seed (int, optional):
            Seed for the random number generator.

    Returns:
        np.ndarray:
            ``target_image_arr`` with the synthesised correlated noise added.
    """
    if source_image_arr.ndim != 2 or target_image_arr.ndim != 2:
        raise ValueError("Input images must be 2D numpy arrays.")

    cf = _estimate_correlated_noise_cf(
        source_image_arr,
        subtract_mean=subtract_mean,
        correct_periodicity=correct_periodicity,
    )
    noise = _generate_correlated_noise(cf, target_image_arr.shape, rng_seed)
    return target_image_arr + noise
