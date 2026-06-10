"""Shared utilities for particle spatial resampling.

Provides kernel-based position sampling from the SPH kernel, array tiling,
velocity dispersion helpers, and higher-level engine functions used by
both Gas.spatially_resample and Stars.spatially_resample.

Private helpers (_tile_array, _divide_and_tile) handle the low-level numpy
mechanics. Public engine functions (resample_coordinates,
resample_velocities, resample_smoothing_lengths, resample_by_mode,
split_by_mask, validate_required_inputs) form the stable API consumed by
the class methods.
"""

import numpy as np
from unyt import Mpc, km, s, unyt_array

from synthesizer import exceptions
from synthesizer.units import accepts

#: Valid string mode names for :func:`resample_by_mode`.
RESAMPLE_MODES = frozenset(
    {"duplicated", "proportional", "normal", "lognormal", "conserved_normal"}
)


def _tile_array(arr, rep):
    """Tile *arr* along the first axis *rep* times, preserving units.

    Args:
        arr (np.ndarray or unyt_array):
            (N, ...) array to tile.
        rep (int):
            Number of repetitions.

    Returns:
        np.ndarray or unyt_array:
            (N * rep, ...) tiled array.
    """
    if hasattr(arr, "units"):
        return unyt_array(
            np.repeat(arr.ndview, rep, axis=0),
            arr.units,
            bypass_validation=True,
        )
    return np.repeat(arr, rep, axis=0)


def _divide_and_tile(arr, factor):
    """Divide *arr* by *factor* and tile along the first axis.

    The total sum of the array is conserved.

    Args:
        arr (np.ndarray or unyt_array):
            (N,) array to divide and tile.
        factor (int):
            Division factor (resample_factor).

    Returns:
        np.ndarray or unyt_array:
            (N * factor,) tiled and scaled array.
    """
    if hasattr(arr, "units"):
        return unyt_array(
            np.repeat(arr.ndview / factor, factor, axis=0),
            arr.units,
            bypass_validation=True,
        )
    return np.repeat(arr / factor, factor, axis=0)


def validate_resample_factor(resample_factor):
    """Validate that resample_factor >= 2.

    Raises:
        ValueError: If resample_factor < 2.
    """
    if resample_factor < 2:
        raise ValueError(
            f"resample_factor must be >= 2, got {resample_factor}."
        )


def validate_required_inputs(coordinates, smoothing_lengths, kernel):
    """Validate that required arrays for spatial resampling are provided.

    Args:
        coordinates (unyt_array or None): The particle coordinates array.
        smoothing_lengths (unyt_array or None): The particle smoothing lengths
            array.
        kernel (Kernel or None): The SPH kernel for position sampling.

    Raises:
        InconsistentArguments: If anything is missing.
    """
    if coordinates is None:
        raise exceptions.InconsistentArguments(
            "coordinates is required for spatial resampling."
        )
    if smoothing_lengths is None:
        raise exceptions.InconsistentArguments(
            "smoothing_lengths is required for spatial resampling."
        )
    if kernel is None:
        raise exceptions.InconsistentArguments(
            "A kernel is required for spatial resampling. "
            "Pass kernel=Kernel('cubic') or similar."
        )


def validate_mask(mask, nparticles):
    """Validate a boolean mask array and convert to bool ndarray.

    Args:
        mask (array-like):
            Boolean mask of length nparticles.
        nparticles (int):
            Expected length.

    Returns:
        np.ndarray: bool array.

    Raises:
        InconsistentArguments: If mask is the wrong length.
    """
    mask = np.asarray(mask, dtype=bool)
    if mask.size != nparticles:
        raise exceptions.InconsistentArguments(
            f"mask must have length nparticles ({nparticles}), "
            f"got {mask.size}."
        )
    return mask


def sample_kernel_positions(kernel, smoothing_lengths, n_samples, seed=None):
    """Sample 3D offsets from an SPH kernel for many particles.

    For each particle with smoothing length h, *n_samples* 3D offsets are
    sampled from the kernel W(r) treated as a PDF. The radial part is
    sampled via inverse-CDF from p(q) ∝ 4π q² f(q) where q = r/h and
    f(q) is the dimensionless kernel. Angles are sampled uniformly on the
    sphere.

    All operations are vectorised across particles × samples.

    Args:
        kernel (Kernel):
            A Kernel instance providing ``kernel.f`` for the dimensionless
            kernel function.
        smoothing_lengths (np.ndarray):
            (N,) array of raw smoothing lengths (numeric values in whatever
            length unit the caller uses — units are not attached).
        n_samples (int):
            Number of samples per particle.
        seed (int, optional):
            Random seed for reproducibility.

    Returns:
        np.ndarray:
            (N, n_samples, 3) array of raw Cartesian offset vectors in the
            same units as *smoothing_lengths*.
    """
    # Set up the random generator
    rng = np.random.default_rng(seed)

    # How many particles?
    n = smoothing_lengths.size

    # Tabulate the radial CDF from the kernel on q in [0, 1]
    n_q_bins = 10001
    q_bins = np.linspace(0.0, 1.0, n_q_bins)
    f_q = kernel.f(q_bins)
    pdf_radial = 4.0 * np.pi * q_bins**2 * f_q
    cdf = np.cumsum(pdf_radial)
    cdf /= cdf[-1]

    # Inverse-CDF: sample q for each (particle, sample) pair
    u = rng.uniform(0.0, 1.0, (n, n_samples))
    q = np.interp(u, cdf, q_bins)
    r = q * smoothing_lengths[:, np.newaxis]

    # Uniform sampling on the sphere
    cos_theta = 2.0 * rng.uniform(0.0, 1.0, (n, n_samples)) - 1.0
    theta = np.arccos(cos_theta)
    phi = rng.uniform(0.0, 2.0 * np.pi, (n, n_samples))

    # Convert to Cartesian offsets
    offsets = np.zeros((n, n_samples, 3), dtype=np.float64)
    offsets[:, :, 0] = r * np.sin(theta) * np.cos(phi)
    offsets[:, :, 1] = r * np.sin(theta) * np.sin(phi)
    offsets[:, :, 2] = r * np.cos(theta)

    return offsets


def resample_by_mode(arr, mode_spec, resample_factor, rng):
    """Resample a per-particle array according to a named mode.

    Applies one of five physically-motivated splitting strategies to *arr*:

    ``"duplicated"``
        Every child inherits the parent value unchanged.  Use for intensive
        quantities (metallicity, optical depth, flags).

    ``"proportional"``
        The parent value is divided equally: each child gets
        ``value / resample_factor``.  The total sum is exactly conserved.
        Use for extensive quantities (mass, dust mass).

    ``"normal"``
        Each child is drawn from ``Normal(value, σ)`` independently.  By
        default σ is the population standard deviation ``np.std(arr)``; pass
        ``mode_spec = ("normal", sigma)`` to supply a fixed σ instead.

    ``"lognormal"``
        Like *normal* but in log-space: each child is drawn from
        ``LogNormal(log(value), σ_log)`` so values remain positive.  By
        default σ_log = ``np.std(np.log(arr))``; pass
        ``("lognormal", sigma_log)`` to fix it.

    ``"conserved_normal"``
        Proportional split with additive Gaussian scatter, renormalised so
        children sum exactly to the parent value.  Default σ =
        ``np.std(arr) / resample_factor``; pass
        ``("conserved_normal", sigma)`` to fix it.

    Args:
        arr (np.ndarray or unyt_array):
            Per-particle array of length N.  Must not be ``None``; check
            before calling.
        mode_spec (str or tuple):
            Either a mode name string or a ``(mode_name, sigma)`` tuple.
        resample_factor (int):
            Number of new particles per original.
        rng (np.random.Generator):
            A seeded :class:`numpy.random.Generator` for reproducibility.

    Returns:
        np.ndarray or unyt_array:
            Resampled array of length ``N * resample_factor``, with units
            preserved if *arr* carries them.

    Raises:
        ValueError: If *mode_spec* is not a recognised mode name or tuple.
    """
    # Parse mode and optional sigma
    if isinstance(mode_spec, str):
        mode = mode_spec
        sigma = None
    else:
        mode, sigma = mode_spec

    # Make sure we have a valid mode
    if mode not in RESAMPLE_MODES:
        raise ValueError(
            f"Unknown resampling mode {mode!r}. "
            f"Valid modes are: {sorted(RESAMPLE_MODES)}."
        )

    # Unpack the units and get a raw numpy array to work with, note that we
    # use ndview to avoid copying
    units = getattr(arr, "units", None)
    if mode == "duplicated":
        raw = arr.ndview if units is not None else np.asarray(arr)
    else:
        raw = arr.ndview if units is not None else np.asarray(arr, dtype=float)

    # How many particles?
    n = raw.shape[0]

    # Handle the duplicated case where we just replicate the value with
    # resample_factor times for each particle
    if mode == "duplicated":
        out = _tile_array(raw, resample_factor)

    # Handle the proportional case where we divide the value equally among
    # resample_factor children, conserving the total sum
    elif mode == "proportional":
        out = _divide_and_tile(raw, resample_factor)

    # Handle the normal case where each child is drawn from Normal(value, std)
    elif mode == "normal":
        # If sigma is not provided, use the population standard deviation
        if sigma is None:
            sigma = float(np.std(raw))

        # Draw independent noise for each child and add to the parent value
        noise = rng.normal(0.0, sigma, (n, resample_factor))

        # Add the noise to the parent value for each child, then flatten to 1-D
        out = (raw[:, np.newaxis] + noise).ravel()

    # Handle the lognormal case where we draw in log-space and exponentiate
    elif mode == "lognormal":
        # Get the log of the raw values
        log_raw = np.log(raw)

        # If sigma is not provided, use the population standard deviation of
        # the log values
        if sigma is None:
            sigma = float(np.std(log_raw))

        # Draw in log-space, then exponentiate to get back to the original
        log_noise = rng.normal(0.0, sigma, (n, resample_factor))

        # Add the log noise to the log of the raw values, then exponentiate and
        # flatten to 1-D
        out = np.exp(log_raw[:, np.newaxis] + log_noise).ravel()

    # Handle the conserved_normal case: start from the proportional split, add
    # independent Gaussian scatter, then renormalise so children sum exactly
    # to the parent value
    elif mode == "conserved_normal":
        # Find the proportional split as the base for the scatter
        base = raw / resample_factor

        # If sigma is not provided, use the population standard deviation
        # scaled by the factor to keep noise amplitude proportionate
        if sigma is None:
            sigma = float(np.std(raw)) / resample_factor

        # Add independent Gaussian noise for each child
        noise = rng.normal(0.0, sigma, (n, resample_factor))
        children = base[:, np.newaxis] + noise

        # Clip negative children to zero — they'd break the renormalisation
        children = np.maximum(children, 0.0)

        # Renormalise each group so children sum exactly to the parent value
        child_sums = children.sum(axis=1, keepdims=True)
        children = np.where(
            child_sums > 0,
            children * (raw[:, np.newaxis] / child_sums),
            raw[:, np.newaxis] / resample_factor,
        )
        out = children.ravel()

    if units is not None:
        return unyt_array(out, units, bypass_validation=True)
    return out


def add_velocity_dispersion(velocities, dispersion, seed=None):
    """Add Gaussian random velocity dispersion to velocity arrays.

    Args:
        velocities (np.ndarray or unyt_array):
            (N_total, 3) array of velocities.
        dispersion (float or unyt_quantity):
            Standard deviation of the Gaussian noise.
        seed (int, optional):
            Random seed for reproducibility.

    Returns:
        np.ndarray or unyt_array:
            Velocities with added dispersion.
    """
    # Set up the random generator
    rng = np.random.default_rng(seed)

    # Get the raw numeric value of the dispersion, converting units
    # if necessary
    if hasattr(dispersion, "units"):
        disp_value = dispersion.to(velocities.units).value
    else:
        disp_value = dispersion

    # Draw Gaussian noise for each velocity component and add to the Velocities
    noise = rng.normal(0.0, disp_value, velocities.shape).astype(np.float64)

    # Reassociate units if velocities is a unyt_array
    if hasattr(velocities, "units"):
        return velocities + noise * velocities.units
    return velocities + noise


@accepts(
    coordinates=Mpc,
    smoothing_lengths=Mpc,
)
def resample_coordinates(
    coordinates,
    smoothing_lengths,
    kernel,
    resample_factor,
    seed=None,
):
    """Resample particle coordinates by adding kernel-sampled offsets.

    Each original position is replaced by *resample_factor* new positions
    offset by vectors sampled from the SPH kernel.

    Note: ``@accepts`` converts *coordinates* and *smoothing_lengths* to Mpc
    before this function is called, so ``smoothing_lengths.ndview`` below is
    always in Mpc.  The returned coordinates are therefore also in Mpc,
    matching the input unit.

    Args:
        coordinates (unyt_array, (N, 3)):
            Original particle coordinates.
        smoothing_lengths (unyt_array, (N,)):
            Per-particle smoothing lengths.
        kernel (Kernel):
            SPH kernel for position sampling.
        resample_factor (int):
            Number of new particles per original particle.
        seed (int, optional):
            Random seed.

    Returns:
        unyt_array, (N * resample_factor, 3):
            Resampled coordinates in the same units as *coordinates*.
    """
    # Strip units for the pure-numpy kernel sampler; units are re-attached
    # to the offsets before adding to the tiled coordinates.
    sml_raw = (
        smoothing_lengths.ndview
        if hasattr(smoothing_lengths, "ndview")
        else smoothing_lengths
    )

    # Sample offsets from the kernel and tile the original coordinates
    offsets = sample_kernel_positions(
        kernel, sml_raw, resample_factor, seed=seed
    )
    coords = _tile_array(coordinates, resample_factor)
    offsets_flat = offsets.reshape(-1, 3) * (
        coordinates.units if hasattr(coordinates, "units") else 1
    )
    return coords + offsets_flat


@accepts(
    velocities=km / s,
    velocity_dispersion=km / s,
)
def resample_velocities(
    velocities,
    resample_factor,
    velocity_dispersion=None,
    seed=None,
):
    """Resample velocities: tile each velocity and optionally add dispersion.

    Args:
        velocities (unyt_array, (N, 3), optional):
            Original velocities.  Returns ``None`` if ``None`` is passed.
        resample_factor (int):
            Number of new particles per original.
        velocity_dispersion (float or unyt_quantity, optional):
            Std. dev. of Gaussian velocity noise to add after tiling.
        seed (int, optional):
            Random seed for the dispersion noise.

    Returns:
        unyt_array or None:
            Resampled velocities (``None`` if *velocities* is ``None``).
    """
    if velocities is None:
        return None
    new_vels = _tile_array(velocities, resample_factor)
    if velocity_dispersion is not None:
        new_vels = add_velocity_dispersion(
            new_vels, velocity_dispersion, seed=seed
        )
    return new_vels


def resample_smoothing_lengths(smoothing_lengths, resample_factor):
    """Scale smoothing lengths for volume conservation and tile.

    Each smoothing length is scaled by ``resample_factor ** (-1/3)`` so that
    the total kernel volume is conserved across the resampled particles, then
    tiled *resample_factor* times.

    Args:
        smoothing_lengths (unyt_array, (N,)):
            Original smoothing lengths.
        resample_factor (int):
            Number of new particles per original.

    Returns:
        unyt_array, (N * resample_factor,):
            Resampled smoothing lengths.
    """
    return _tile_array(
        smoothing_lengths / (resample_factor ** (1.0 / 3.0)),
        resample_factor,
    )


def _sample_sfzh_arrays(sfzh, log10ages, log10metallicities, nstar, rng):
    """Sample ages and metallicities continuously from an SFZH array.

    Unlike :func:`sample_sfzh`, which returns values at the exact grid
    points, this function samples uniformly *within* each 2-D histogram
    cell, giving a continuous distribution of ages and metallicities.

    The SFZH is treated as a probability density; the CDF is built from
    the flattened 2-D array, a cell index is chosen by inverse-CDF
    sampling, and then the final value is linearly interpolated between
    the selected grid point and its neighbour.

    Args:
        sfzh (np.ndarray):
            2-D SFZH array (ages × metallicities).
        log10ages (np.ndarray):
            1-D array of log10(age) grid edges/centres, sampled linearly
            within cells.
        log10metallicities (np.ndarray):
            1-D array of metallicity grid edges/centres.
        nstar (int):
            Number of stellar particles to produce.
        rng (np.random.Generator):
            Seeded random generator.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            ``(ages, metallicities)`` — plain float64 arrays of length
            *nstar* with ages in yr and metallicities as dimensionless
            mass fractions.
    """
    # Build the CDF from the flattened SFZH histogram
    hist = sfzh / np.sum(sfzh)
    cdf = np.cumsum(hist.flatten())
    cdf = cdf / cdf[-1]

    # Sample cell indices by inverse-CDF sampling
    values = rng.random(nstar)
    value_bins = np.searchsorted(cdf, values)

    # Convert the flat cell indices back to 2-D age and metallicity indices
    x_idx, y_idx = np.unravel_index(
        value_bins, (len(log10ages), len(log10metallicities))
    )

    # Interpolate smoothly within each cell for a continuous distribution
    # instead of picking exact grid points
    x_frac = rng.random(nstar)
    y_frac = rng.random(nstar)

    # Interpolate log10(age) and log10(metallicity) separately, then
    # exponentiate to get back to the ages and metallicities
    ages = 10 ** (
        log10ages[x_idx]
        + x_frac
        * (
            np.asarray(log10ages)[np.clip(x_idx + 1, 0, len(log10ages) - 1)]
            - np.asarray(log10ages)[x_idx]
        )
    )
    metallicities = np.asarray(log10metallicities)[y_idx] + y_frac * (
        np.asarray(log10metallicities)[
            np.clip(y_idx + 1, 0, len(log10metallicities) - 1)
        ]
        - np.asarray(log10metallicities)[y_idx]
    )

    return ages, metallicities


def split_by_mask(mask, **arrays):
    """Split arrays by a boolean mask, preserving input order.

    Each array is either indexed by the mask (if it is a 1-D array matching
    the mask length), or left unchanged if it is a scalar, ``None``, or a
    shorter/longer array.

    Input order is preserved in the returned lists::

        m, u = split_by_mask(mask, mass=..., met=..., sfr=...)
        # m[0] is masked mass, u[0] is unmasked mass, etc.

    Args:
        mask (array-like):
            Boolean mask.
        **arrays:
            Named arrays, scalars, or ``None`` to split.

    Returns:
        tuple[list, list]:
            ``(to_resample, no_resample)`` — two lists, each of the
            same length as the number of input arrays, in input order.
    """
    mask = np.asarray(mask, dtype=bool)
    n = mask.size
    to_resample = []
    no_resample = []
    for _name, arr in arrays.items():
        if arr is None or not hasattr(arr, "shape"):
            to_resample.append(arr)
            no_resample.append(arr)
        elif arr.shape == () or arr.ndim == 0:
            to_resample.append(arr)
            no_resample.append(arr)
        elif arr.shape[0] == n:
            to_resample.append(arr[mask])
            no_resample.append(arr[~mask])
        else:
            to_resample.append(arr)
            no_resample.append(arr)
    return to_resample, no_resample
