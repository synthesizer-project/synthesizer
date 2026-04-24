"""A submodule containing generator functions for producing images.

Synthesizer supports generating both histogram and smoothed images depending on
the inputs. Particle based emitters can be histogram or smoothed, while
parametric emitters can only be smoothed. These functions abstract away the
complications of these different methods.

These can be accessed either by calling the low level Image and ImageCollection
classes directly, or by using the higher level functions on galaxies and
their components (get_images_luminoisty/get_images_flux).

The functions in this module are not intended to be called directly by the
user.
"""

from copy import deepcopy

import numpy as np
from unyt import angstrom, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.conversions import (
    angular_to_spatial_at_z,
    spatial_to_angular_at_z,
)
from synthesizer.imaging.extensions.image import make_img
from synthesizer.kernel_functions import Kernel
from synthesizer.synth_warnings import warn
from synthesizer.units import unit_is_compatible
from synthesizer.utils import (
    ensure_array_c_compatible_double,
)
from synthesizer.utils.operation_timers import timed, timer

_CENTERING_TOLERANCE = 1e-6


def _validate_centered_coordinates(cent_coords, *, warn_only=False):
    """Ensure coordinates are centred on zero along each axis.

    Args:
        cent_coords (unyt_array, float):
            The centred coordinates to validate.
        warn_only (bool):
            If True, only issue a warning if the coordinates are not centred.
            If False, raise an exception.

    Raises:
        InconsistentArguments:
            If the coordinates are not centred and warn_only is False.
    """
    # Nothing to do for empty coordinates
    if cent_coords.size == 0:
        return

    # Determine the tolerance for centering based on the span of the
    # coordinates to allow for relative precision in the centering check.
    coord_min = cent_coords.min(axis=0)
    coord_max = cent_coords.max(axis=0)
    span = np.max(np.abs(coord_max - coord_min))
    tolerance = (
        span * _CENTERING_TOLERANCE if span != 0 else _CENTERING_TOLERANCE
    )

    # Coordinates should straddle zero (or sit very near zero) in every axis
    spans_zero = np.all(
        (coord_min <= tolerance) & (coord_max >= -tolerance)
    ) or np.all(np.isclose(cent_coords, 0, atol=tolerance, rtol=0.0))
    centred = spans_zero
    if centred:
        return

    # Not centred, either warn or raise
    msg = (
        "Coordinates must be centered for imaging"
        f" (got min={coord_min} and max={coord_max})."
    )
    if warn_only:
        warn(msg)
    else:
        raise exceptions.InconsistentArguments(msg)


def _standardize_imaging_units(
    resolution,
    fov,
    emitter,
    cosmo=None,
    include_smoothing_lengths=False,
):
    """Standardize all imaging inputs to the same unit system.

    This function ensures that resolution, fov, and emitter coordinates
    are all in the same unit system (either all angular or all Cartesian).
    If they are in different systems, it converts them to a common system
    using the provided cosmology and the emitter's redshift.

    The target system is determined by the resolution units:
    - If resolution is angular, everything will be converted to angular
    - If resolution is Cartesian, everything will be converted to Cartesian

    This should be called at the start of any imaging operation to ensure
    consistent units throughout.

    IMPORTANT: This function does NOT modify the emitter's underlying data.
    It returns new coordinate arrays.

    Args:
        resolution (unyt_quantity):
            The size of a pixel. Can be in angular (e.g., arcsec) or
            Cartesian (e.g., kpc) units.
        fov (unyt_quantity/tuple of unyt_quantity):
            The width of the image. Can be in angular or Cartesian units.
            If a single value is given it will be used for both axes.
        emitter (Stars/BlackHoles/BlackHole):
            The emitter object containing the coordinates (and optionally
            smoothing lengths) to standardize.
        cosmo (astropy.cosmology.Cosmology, optional):
            The cosmology object used for unit conversions. Required if
            any inputs are in different unit systems.
        include_smoothing_lengths (bool):
            If True, also standardize and return smoothing lengths from
            the emitter. Default is False.

    Returns:
        tuple:
            - standardized_resolution (unyt_quantity)
            - standardized_fov (unyt_array)
            - standardized_coordinates (unyt_array)
            - standardized_smoothing_lengths (unyt_array or None)
              Only returned if include_smoothing_lengths=True, otherwise None

    Raises:
        InconsistentArguments:
            If inputs are in different unit systems but cosmo or
            emitter.redshift are not available.

    Examples:
        >>> # All Cartesian - no conversion needed
        >>> res, fov, coords, smls = _standardize_imaging_units(
        ...     0.1 * kpc,
        ...     10 * kpc,
        ...     emitter,
        ...     include_smoothing_lengths=True,
        ... )
        >>> # Mixed units - conversion needed
        >>> res, fov, coords, smls = _standardize_imaging_units(
        ...     0.1 * arcsecond,
        ...     10 * kpc,
        ...     emitter,
        ...     cosmo=cosmo,
        ...     include_smoothing_lengths=True,
        ... )
    """
    from unyt import arcsecond, kpc

    # Validate resolution is a unyt object
    if not isinstance(resolution, (unyt_quantity, unyt_array)):
        raise exceptions.InconsistentArguments(
            "Resolution must be a unyt_quantity or unyt_array. "
            f"Got {type(resolution).__name__}."
        )

    # Validate fov is a unyt object
    if not isinstance(fov, (unyt_quantity, unyt_array)):
        raise exceptions.InconsistentArguments(
            "Field of view (fov) must be a unyt_quantity or unyt_array. "
            f"Got {type(fov).__name__}."
        )

    # Ensure fov has an entry for each axis if it doesn't already
    if isinstance(fov, unyt_quantity):
        fov = unyt_array((fov.value, fov.value), fov.units)
    elif fov.size == 1:
        fov = unyt_array((fov.value, fov.value), fov.units)

    # Validate emitter has required attributes
    if not hasattr(emitter, "centered_coordinates"):
        raise exceptions.InconsistentArguments(
            "Emitter must have 'centered_coordinates' attribute. "
            f"Got {type(emitter).__name__} which does not have this attribute."
        )

    if include_smoothing_lengths and not hasattr(emitter, "smoothing_lengths"):
        raise exceptions.InconsistentArguments(
            "Emitter must have 'smoothing_lengths' attribute when "
            "include_smoothing_lengths=True. "
            f"Got {type(emitter).__name__} which does not have this attribute."
        )

    # Get the redshift from the emitter
    redshift = getattr(emitter, "redshift", None)

    # Determine the target unit system based on resolution
    resolution_is_angular = unit_is_compatible(resolution, arcsecond)
    resolution_is_cartesian = unit_is_compatible(resolution, kpc)

    if not resolution_is_angular and not resolution_is_cartesian:
        raise exceptions.InconsistentArguments(
            "Resolution must be in either angular (e.g., arcsec) or "
            f"Cartesian (e.g., kpc) units. Got {resolution.units}."
        )

    # Get emitter coordinates and optionally smoothing lengths
    # Use centered coordinates which should already exist on the emitter
    emitter_coords = emitter.centered_coordinates
    emitter_smls = (
        emitter.smoothing_lengths if include_smoothing_lengths else None
    )

    # Validate that coordinates are unyt arrays
    if not isinstance(emitter_coords, unyt_array):
        raise exceptions.InconsistentArguments(
            "Emitter centered_coordinates must be a unyt_array. "
            f"Got {type(emitter_coords).__name__}."
        )

    # Validate smoothing lengths if requested
    if emitter_smls is not None and not isinstance(emitter_smls, unyt_array):
        raise exceptions.InconsistentArguments(
            "Emitter smoothing_lengths must be a unyt_array. "
            f"Got {type(emitter_smls).__name__}."
        )

    # Check if conversion is needed
    coords_compatible = unit_is_compatible(resolution, emitter_coords.units)
    fov_compatible = unit_is_compatible(resolution, fov.units)
    smls_compatible = emitter_smls is None or unit_is_compatible(
        resolution, emitter_smls.units
    )

    # If everything is already compatible, return copies to avoid
    # accidentally modifying the emitter
    if coords_compatible and fov_compatible and smls_compatible:
        standardized_coords = emitter_coords.copy()
        standardized_fov = fov.copy()
        standardized_smls = (
            emitter_smls.copy() if emitter_smls is not None else None
        )
        return (
            resolution,
            standardized_fov,
            standardized_coords,
            standardized_smls,
        )

    # Need to convert - check we have the required cosmology and redshift
    if cosmo is None or redshift is None:
        raise exceptions.InconsistentArguments(
            "Cannot convert between angular and Cartesian units without "
            "both a cosmology and the emitter's redshift. "
            f"Got resolution={resolution.units}, "
            f"emitter coords={emitter_coords.units}, "
            f"fov={fov.units}, cosmo={cosmo}, redshift={redshift}."
        )

    # Convert everything to the target system (based on resolution)
    if resolution_is_angular:
        # Convert to angular units
        if not coords_compatible:
            standardized_coords = spatial_to_angular_at_z(
                emitter_coords, cosmo, redshift
            )
        else:
            standardized_coords = emitter_coords.copy()

        if not fov_compatible:
            standardized_fov = spatial_to_angular_at_z(fov, cosmo, redshift)
        else:
            standardized_fov = fov.copy()

        if emitter_smls is not None and not smls_compatible:
            standardized_smls = spatial_to_angular_at_z(
                emitter_smls, cosmo, redshift
            )
        elif emitter_smls is not None:
            standardized_smls = emitter_smls.copy()
        else:
            standardized_smls = None

    else:  # resolution_is_cartesian
        # Convert to Cartesian units
        if not coords_compatible:
            standardized_coords = angular_to_spatial_at_z(
                emitter_coords, cosmo, redshift
            )
        else:
            standardized_coords = emitter_coords.copy()

        if not fov_compatible:
            standardized_fov = angular_to_spatial_at_z(fov, cosmo, redshift)
        else:
            standardized_fov = fov.copy()

        if emitter_smls is not None and not smls_compatible:
            standardized_smls = angular_to_spatial_at_z(
                emitter_smls, cosmo, redshift
            )
        elif emitter_smls is not None:
            standardized_smls = emitter_smls.copy()
        else:
            standardized_smls = None

    return resolution, standardized_fov, standardized_coords, standardized_smls


@timed("_generate_image_particle_hist")
def _generate_image_particle_hist(
    img,
    signal,
    coordinates,
    normalisation=None,
):
    """Generate a histogram image for a particle emitter.

    Args:
        img (Image):
            The image to create.
        signal (unyt_array, float):
            The signal of each particle to be sorted into pixels.
        coordinates (unyt_array, float):
            The coordinates of the particles.
        normalisation (unyt_quantity, float):
            The normalisation to apply to the image.

    Returns:
        Image: The histogram image.
    """
    with timer("_generate_image_particle_hist.setup"):
        # Ensure the signal is a 1D array and is a compatible size with the
        # coordinates
        if signal.ndim != 1:
            raise exceptions.InconsistentArguments(
                "Signal must be a 1D array for a histogram image"
                f" (got {signal.ndim})."
            )
        if signal.size != coordinates.shape[0]:
            raise exceptions.InconsistentArguments(
                "Signal and coordinates must be the same size"
                f" for a histogram image (got {signal.size} and "
                f"{coordinates.shape[0]})."
            )

        # Ensure the coordinates are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(coordinates, img.resolution.units):
            raise exceptions.InconsistentArguments(
                "Coordinates must be compatible with the image resolution "
                f"units (got {coordinates.units} and {img.resolution.units})."
            )

        # Ensure coordinates have been centred
        _validate_centered_coordinates(coordinates, warn_only=True)

        # Strip off and store the units on the signal if they are present
        if isinstance(signal, (unyt_quantity, unyt_array)):
            img.units = signal.units
            signal = signal.value

        # Return an empty image if there are no particles
        if signal.size == 0:
            img.arr = np.zeros(img.npix)
            return img.arr * img.units if img.units is not None else img.arr

        # Unpack the image properties and ensure we agree on the units
        spatial_units = img.resolution.units
        fov = img.fov.to_value(spatial_units)

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        coordinates = coordinates.to_value(spatial_units)

    with timer("_generate_image_particle_hist.generate"):
        # Include normalisation in the original signal if we have one
        # (we'll divide by it later)
        if normalisation is not None:
            # Make sure we are working on a copy of the signal so as not to
            # modify the original
            signal = signal.copy()
            signal *= normalisation.value

        img.arr = np.histogram2d(
            coordinates[:, 0],
            coordinates[:, 1],
            bins=(
                np.linspace(-fov[0] / 2, fov[0] / 2, img.npix[0] + 1),
                np.linspace(-fov[1] / 2, fov[1] / 2, img.npix[1] + 1),
            ),
            weights=signal,
        )[0]

    # Normalise the image by the normalisation if applicable
    if normalisation is not None:
        with timer("_generate_image_particle_hist.normalise"):
            norm_img = np.histogram2d(
                coordinates[:, 0],
                coordinates[:, 1],
                bins=(
                    np.linspace(-fov[0] / 2, fov[0] / 2, img.npix[0] + 1),
                    np.linspace(-fov[1] / 2, fov[1] / 2, img.npix[1] + 1),
                ),
                weights=normalisation.value,
            )[0]

            img.arr /= norm_img

    return img


def _generate_images_particle_hist(
    imgs,
    coordinates,
    signals,
    normalisations=None,
):
    """Generate histogram images for a particle emitter.

    This is a wrapper around _generate_image_particle_hist to allow for
    multiple signals to be passed in at once to generate an ImageCollection.

    Args:
        imgs (ImageCollection):
            The image collection to create the images for.
        coordinates (unyt_array, float):
            The coordinates of the particles.
        signals (dict):
            The signals to use for the images. The resulting Images will be
            labelled with the keys of this dict.
        normalisations (dict):
            The normalisations to use for the images. The keys must match the
            keys in signals. If not provided, normalisation is set to None.

    Returns:
        ImageCollection: An image collection containing the histogram images.
    """
    # Avoid cyclic imports
    from synthesizer.imaging import Image

    # Loop over the signals and create the images
    for key, signal in signals.items():
        # Create an Image object for this filter
        img = Image(imgs.resolution, imgs.fov)

        # Get the image for this filter
        imgs[key] = _generate_image_particle_hist(
            img,
            signal,
            coordinates=coordinates,
            normalisation=normalisations[key]
            if normalisations is not None and key in normalisations
            else None,
        )

    return imgs


@timed("_generate_image_particle_smoothed")
def _generate_image_particle_smoothed(
    img,
    signal,
    cent_coords,
    smoothing_lengths,
    kernel,
    kernel_threshold,
    nthreads,
    normalisation=None,
):
    """Generate smoothed images for a particle emitter.

    Args:
        img (Image):
            The image object to populate with the image.
        signal (unyt_array of float):
            The signal of each particle to be sorted into pixels.
        cent_coords (unyt_array of float):
            The centred coordinates of the particles. These will be converted
            to the image resolution units. These will be shifted to fall in the
            range [0, FOV].
        smoothing_lengths (unyt_array of float):
            The smoothing lengths of the particles. These will be converted to
            the image resolution units.
        kernel (str):
            The array describing the kernel. This is dervied from the
            kernel_functions module.
        kernel_threshold (float):
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image.
        nthreads (int):
            The number of threads to use when smoothing the image. This
            only applies to particle imaging.
        normalisation (unyt_quantity of float):
            The optional normalisation to apply to the image.

    Returns:
        Image: The smoothed image.
    """
    with timer("_generate_image_particle_smoothed.setup"):
        # Avoid cyclic imports
        from synthesizer.imaging import Image

        # Ensure the signal is a 1D array and is a compatible size with the
        # coordinates and smoothing lengths
        if signal.ndim != 1:
            raise exceptions.InconsistentArguments(
                "Signal must be a 1D array for a smoothed image"
                f" (got {signal.ndim})."
            )
        if signal.size != cent_coords.shape[0]:
            raise exceptions.InconsistentArguments(
                "Signal and coordinates must be the same size"
                f" for a smoothed image (got {signal.size} and "
                f"{cent_coords.shape[0]})."
            )
        if signal.size != smoothing_lengths.shape[0]:
            raise exceptions.InconsistentArguments(
                "Signal and smoothing lengths must be the same size"
                f" for a smoothed image (got {signal.size} and "
                f"{smoothing_lengths.shape[0]})."
            )
        if cent_coords.shape[0] != smoothing_lengths.shape[0]:
            raise exceptions.InconsistentArguments(
                "Coordinates and smoothing lengths must be the same size"
                f" for a smoothed image (got {cent_coords.shape[0]} and "
                f"{smoothing_lengths.shape[0]})."
            )

        # Ensure the coordinates are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(cent_coords, img.resolution.units):
            raise exceptions.InconsistentArguments(
                "Coordinates must be compatible with the image resolution "
                f"units (got {cent_coords.units} and {img.resolution.units})."
            )

        # Ensure the smoothing lengths are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(smoothing_lengths, img.resolution.units):
            raise exceptions.InconsistentArguments(
                "Smoothing lengths must be compatible with the image "
                f"resolution units (got {smoothing_lengths.units} and "
                f"{img.resolution.units})."
            )

        # Ensure coordinates have been centred
        _validate_centered_coordinates(cent_coords)

        # Get the spatial units we'll work with
        spatial_units = img.resolution.units

        # Unpack the image properties we need
        fov = img.fov.to_value(spatial_units)
        res = img.resolution.to_value(spatial_units)

        # Shift the centred coordinates by half the FOV
        # (this is to ensure the image is centered on the emitter)
        _coords = cent_coords.to(spatial_units).value
        _coords[:, 0] += fov[0] / 2.0
        _coords[:, 1] += fov[1] / 2.0
        _smoothing_lengths = smoothing_lengths.to_value(spatial_units)

        # Apply normalisation to original signal if needed
        if normalisation is not None:
            # Make sure we are working on a copy of the signal so as not to
            # modify the original
            signal = signal.copy()
            signal *= normalisation.value

    # Get the (npix_x, npix_y, Nimg) array of images
    imgs_arr = make_img(
        ensure_array_c_compatible_double(signal),
        ensure_array_c_compatible_double(_smoothing_lengths),
        ensure_array_c_compatible_double(_coords),
        kernel,
        res,
        img.npix[0],
        img.npix[1],
        cent_coords.shape[0],
        kernel_threshold,
        kernel.size,
        1,
        nthreads,
    )

    # Store the image array into the image object
    img.arr = imgs_arr[:, :, 0]
    img.units = (
        signal.units
        if isinstance(signal, (unyt_quantity, unyt_array))
        else None
    )

    # Apply the normalisation if needed
    if normalisation is not None:
        with timer("_generate_image_particle_smoothed.normalise"):
            norm_img = Image(resolution=img.resolution, fov=img.fov)
            norm_img = _generate_image_particle_smoothed(
                norm_img,
                signal=normalisation,
                cent_coords=cent_coords,
                smoothing_lengths=smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

            # Normalise the image by the normalisation property
            img.arr /= norm_img.arr

    return img


@timed("_generate_images_particle_smoothed")
def _generate_images_particle_smoothed(
    imgs,
    signals,
    cent_coords,
    smoothing_lengths,
    labels,
    kernel,
    kernel_threshold,
    nthreads,
    normalisations=None,
):
    """Generate smoothed images for a particle emitter.

    Args:
        imgs (ImageCollection):
            The image collection to populate with the images.
        signals (unyt_array of float):
            The signals of each particle to be sorted into pixels. This should
            be (Nimg, Nparticles) in shape.
        cent_coords (unyt_array of float):
            The centred coordinates of the particles. These will be
            converted to the image resolution units. These will be shifted
            to fall in the range [0, FOV].
        smoothing_lengths (unyt_array of float):
            The smoothing lengths of the particles. These will be converted
            to the image resolution units.
        labels (list):
            The labels of the signals to use for the images. This must have a
            length equal to the number of images in the collection.
        kernel (str):
            The array describing the kernel. This is dervied from the
            kernel_functions module.
        kernel_threshold (float):
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image.
        nthreads (int):
            The number of threads to use when smoothing the image. This
            only applies to particle imaging.
        normalisations (dict):
            The normalisations to use for the images. The keys must match the
            labels of the signals. If not provided, normalisation is set to
            None.

    Returns:
        ImageCollection: An image collection containing the smoothed images.
    """
    with timer("_generate_images_particle_smoothed.setup"):
        # Avoid cyclic imports
        from synthesizer.imaging import Image

        # Ensure the signals are 2D arrays and are compatible sizes with the
        # coordinates and smoothing lengths and the number of images
        if signals.ndim != 2:
            raise exceptions.InconsistentArguments(
                "Signals must be a 2D array for a smoothed image"
                f" (got {signals.ndim})."
            )
        if signals.shape[1] != cent_coords.shape[0]:
            raise exceptions.InconsistentArguments(
                "Signals and coordinates must be the same size"
                f" for a smoothed image (got {signals.shape[1]} and "
                f"{cent_coords.shape[0]})."
            )
        if signals.shape[1] != smoothing_lengths.shape[0]:
            raise exceptions.InconsistentArguments(
                "Signals and smoothing lengths must be the same size"
                f" for a smoothed image (got {signals.shape[1]} and "
                f"{smoothing_lengths.shape[0]})."
            )
        if cent_coords.shape[0] != smoothing_lengths.shape[0]:
            raise exceptions.InconsistentArguments(
                "Coordinates and smoothing lengths must be the same size"
                f" for a smoothed image (got {cent_coords.shape[0]} and "
                f"{smoothing_lengths.shape[0]})."
            )
        if signals.shape[0] != len(labels):
            raise exceptions.InconsistentArguments(
                "Signals should have an entry for each signal "
                f"label (got {signals.shape[0]} and {len(labels)})."
            )

        # Ensure the coordinates are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(cent_coords, imgs.resolution.units):
            raise exceptions.InconsistentArguments(
                "Coordinates must be compatible with the image resolution "
                f"units (got {cent_coords.units} and {imgs.resolution.units})."
            )

        # Ensure the smoothing lengths are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(smoothing_lengths, imgs.resolution.units):
            raise exceptions.InconsistentArguments(
                "Smoothing lengths must be compatible with the image "
                f"resolution units (got {smoothing_lengths.units} and "
                f"{imgs.resolution.units})."
            )

        # Ensure coordinates have been centred
        _validate_centered_coordinates(cent_coords, warn_only=True)

        # Get the spatial units we'll work with
        spatial_units = imgs.resolution.units

        # Unpack the image properties we need
        fov = imgs.fov.to_value(spatial_units)
        res = imgs.resolution.to_value(spatial_units)

        # Shift the centred coordinates by half the FOV
        # (this is to ensure the image is centered on the emitter)
        _coords = cent_coords.to(spatial_units).value
        _coords[:, 0] += fov[0] / 2.0
        _coords[:, 1] += fov[1] / 2.0
        _smoothing_lengths = smoothing_lengths.to_value(spatial_units)

        # Apply normalisation to original signal if needed
        if normalisations is not None:
            # Make sure we are working on a copy of the signals so as not to
            # modify the original
            signals = signals.copy()

            # Apply the normalisation to the corresponding signal
            for ind, key in enumerate(labels):
                signals[ind, :] *= normalisations[key].value

        # In the C++ extension we want to be dealing with (Npart, Nimg)
        # signals to make the most of cache locality, so we transpose them.
        signals = signals.T

    # Get the (Nimg, npix_x, npix_y) array of images
    imgs_arr = make_img(
        ensure_array_c_compatible_double(signals),
        ensure_array_c_compatible_double(_smoothing_lengths),
        ensure_array_c_compatible_double(_coords),
        kernel,
        res,
        imgs.npix[0],
        imgs.npix[1],
        cent_coords.shape[0],
        kernel_threshold,
        kernel.size,
        signals.shape[1],
        nthreads,
    )

    # Apply units if needs be
    if isinstance(signals, (unyt_quantity, unyt_array)):
        with timer("_generate_images_particle_smoothed.apply_units"):
            imgs_arr = unyt_array(
                imgs_arr,
                units=signals.units,
            )

    # Store the image arrays on the image collection (this will
    # automatically convert them to Image objects)
    with timer("_generate_images_particle_smoothed.unpack"):
        for ind, key in enumerate(labels):
            imgs[key] = imgs_arr[:, :, ind]

    # Apply normalisation if needed
    if normalisations is not None:
        with timer("_generate_images_particle_smoothed.normalise"):
            for ind, key in enumerate(labels):
                norm_img = Image(resolution=imgs.resolution, fov=imgs.fov)
                norm_img = _generate_image_particle_smoothed(
                    norm_img,
                    signal=normalisations[key],
                    cent_coords=cent_coords,
                    smoothing_lengths=smoothing_lengths,
                    kernel=kernel,
                    kernel_threshold=kernel_threshold,
                    nthreads=nthreads,
                )

                # Normalise the image by the normalisation property
                imgs[key].arr /= norm_img.arr

    return imgs


@timed("_generate_image_parametric_smoothed")
def _generate_image_parametric_smoothed(
    img,
    density_grid,
    signal,
):
    """Generate a smoothed image for a parametric emitter.

    Args:
        img (Image):
            The image to create.
        density_grid (unyt_array of float):
            The density grid to be smoothed over.
        signal (unyt_array of float):
            The signal to be sorted into pixels.

    Returns:
        ImageCollection: An image collection containing the smoothed images.
    """
    # Multiply the density grid by the sed to get the image
    img.arr = density_grid[:, :] * signal.value
    img.units = signal.units

    return img


def _generate_images_parametric_smoothed(
    imgs,
    density_grid,
    signals,
):
    """Generate smoothed images for a parametric emitter.

    Args:
        imgs (ImageCollection):
            The image collection to populate with the images.
        density_grid (unyt_array of float):
            The density grid to be smoothed over.
        signals (dict/PhotometryCollection):
            The signals to be sorted into pixels. Each entry in the dict should
            be a single "integrated" signal to smooth over the density grid.

    Returns:
        ImageCollection: An image collection containing the smoothed images.
    """
    # Avoid cyclic imports
    from synthesizer.imaging import Image

    # Loop over the signals and create the images
    for key, signal in signals.items():
        # Create an Image object for this filter
        img = Image(imgs.resolution, imgs.fov)

        # Get the image for this filter
        imgs[key] = _generate_image_parametric_smoothed(
            img,
            density_grid=density_grid,
            signal=signal,
        )

    return imgs


@timed("_generate_image_collection_generic")
def _generate_image_collection_generic(
    instrument,
    photometry,
    fov,
    img_type,
    kernel,
    kernel_threshold,
    nthreads,
    emitter,
    cosmo,
):
    """Generate an image collection for a generic emitter.

    This function can be used to avoid repeating image generation code in
    wrappers elsewhere in the code. It'll produce an image collection based
    on the input photometry.

    Particle based imaging can either be hist or smoothed, while parametric
    imaging can only be smoothed.

    Args:
        instrument (Instrument):
            The instrument to create the images for.
        photometry (PhotometryCollection):
            The photometry to use for the images. This should be a a collection
            of 2D arrays of photometry with shape (Nfilters, Nparticles).
        fov (unyt_quantity/tuple, unyt_quantity):
            The width of the image.
        img_type (str):
            The type of image to create. Options are "hist" or "smoothed".
        kernel (str):
            The array describing the kernel. This is derived from the
            kernel_functions module. (Only applicable to particle imaging)
        kernel_threshold (float):
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image. (Only
            applicable to particle imaging)
        nthreads (int):
            The number of threads to use when smoothing the image. This
            only applies to particle imaging.
        emitter (Stars/BlackHoles/BlackHole):
            The emitter object to create the images for.
        cosmo (astropy.cosmology.Cosmology):
            A cosmology object defining the cosmology to use for the images.
            This is only relevant for angular images where a conversion to
            projected angular coordinates is needed.

    Returns:
        ImageCollection
            An image collection object containing the images.
    """
    # Avoid cyclic imports
    from synthesizer.imaging import ImageCollection
    from synthesizer.particle import Particles

    # For particle emitters, standardize units to ensure resolution, fov,
    # and emitter data are all in the same system (both angular or both
    # Cartesian). Parametric emitters handle their own geometry via
    # morphology.get_density_grid() and don't need this standardization.
    # NOTE: This does NOT modify the emitter's underlying data
    if isinstance(emitter, Particles):
        needs_smoothing_lengths = img_type == "smoothed"
        resolution, fov, coords, smls = _standardize_imaging_units(
            resolution=instrument.resolution,
            fov=fov,
            emitter=emitter,
            cosmo=cosmo,
            include_smoothing_lengths=needs_smoothing_lengths,
        )

        # Validate that smoothing lengths exist for smoothed particle imaging
        if img_type == "smoothed" and smls is None:
            raise exceptions.InconsistentArguments(
                "Smoothed particle imaging requires smoothing_lengths. "
                "The emitter must have a smoothing_lengths attribute."
            )
    else:
        # Parametric emitters: keep original resolution/fov, skip coord
        # standardization
        resolution = instrument.resolution
        fov = fov
        coords = smls = None

    # Create the image collection
    imgs = ImageCollection(
        resolution=resolution,
        fov=fov,
    )

    # Make the image handling the different types of image creation
    # NOTE: Black holes are always a histogram, safer to just hack this here
    # since a user can set the "global" method as smoothed for a galaxy
    # with both stars and black holes.
    if (img_type == "hist" and isinstance(emitter, Particles)) or (
        getattr(emitter, "name", None) == "Black Holes"
    ):
        # Use the standardized coordinates (already in correct units)
        return _generate_images_particle_hist(
            imgs,
            coordinates=coords,
            signals=photometry,
        )

    elif img_type == "hist":
        raise exceptions.InconsistentArguments(
            "Parametric images can only be made using the smoothed image type."
        )

    elif img_type == "smoothed" and isinstance(emitter, Particles):
        # Use the standardized coordinates and smoothing lengths
        # (already in correct units)
        return _generate_images_particle_smoothed(
            imgs=imgs,
            signals=photometry.photometry,
            cent_coords=coords,
            smoothing_lengths=smls,
            labels=photometry.filter_codes,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
        )

    elif img_type == "smoothed":
        # FYI, the parametric imaging handles the angular vs cartesian
        # image properties internally so we don't need to worry about it here
        return _generate_images_parametric_smoothed(
            imgs,
            density_grid=emitter.morphology.get_density_grid(
                imgs.resolution, imgs.npix
            ),
            signals=photometry,
        )

    else:
        raise exceptions.UnknownImageType(
            f"Unknown img_type {img_type} for a {type(emitter)} emitter. "
            " (Options are 'hist' (only for particle based emitters)"
            " or 'smoothed')"
        )

    return imgs


def _combine_image_collections(images, label, model_cache):
    """Combine multiple image collections into a single image collection.

    Args:
        images: Dictionary of existing images.
        label: The label of the combined image to create.
        model_cache: The model parameter cache to use for lookup.

    Returns:
        ImageCollection: The combined images.

    Raises:
        MissingModel: If label not found in model_cache.
        MissingImage: If any required component images are missing.
    """
    # Validate label exists in model_cache
    if label not in model_cache:
        raise exceptions.MissingModel(
            f"Label '{label}' not found in model cache."
        )

    # Validate that the model cache entry contains combine keys
    if "combine" not in model_cache[label]:
        raise exceptions.MissingModel(
            f"Label '{label}' in model cache missing 'combine' key."
        )

    # Find the images we need to combine from the provided model cache
    combine_keys = model_cache[label]["combine"]

    # Validate all combine_keys exist in images
    missing_keys = [key for key in combine_keys if key not in images]
    if missing_keys:
        raise exceptions.MissingImage(
            f"Cannot combine images for '{label}': missing images for "
            f"{', '.join(missing_keys)}"
        )

    # Get all the images to add
    combine_imgs = [images[key] for key in combine_keys]

    # Combine the images
    combined_img = deepcopy(combine_imgs[0])
    for img in combine_imgs[1:]:
        combined_img += img

    return combined_img


@timed("_generate_ifu_particle_hist")
def _generate_ifu_particle_hist(
    ifu,
    sed,
    quantity,
    cent_coords,
    nthreads,
):
    """Generate a histogram IFU for a particle emitter.

    Args:
        ifu (SpectralCube):
            The SpectralCube object to populate with the ifu.
        sed (Sed):
            The Sed containing the spectra to sort into the IFU. For a
            particle emitter this should be a spectrum per particle, i.e.
            a 2D array of spectra with shape (Nparticles, Nlam).
        quantity (str):
            The quantity to use for the spectra. This can be any valid
            spectra quantity on an Sed object, e.g. 'lnu', 'fnu', 'luminosity',
            'flux', etc.
        cent_coords (unyt_array of float):
            The centered coordinates of the particles.
        nthreads (int):
            The number of threads to use when smoothing the image.

    Returns:
        SpectralCube: The histogram image.
    """
    with timer("_generate_ifu_particle_hist.setup"):
        # Sample the spectra onto the wavelength grid
        sed = sed.get_resampled_sed(new_lam=ifu.lam)

        # Store the Sed and quantity
        ifu.sed = sed
        ifu.quantity = quantity

        # Get the spectra we will be sorting into the spectral cube
        spectra = getattr(sed, quantity, None)
        if spectra is None:
            raise exceptions.MissingSpectraType(
                f"Can't make an image for {quantity},"
                " it does not exist in the Sed."
            )

        # Strip off and store the units on the spectra for later
        ifu.units = spectra.units
        spectra = spectra.ndview

        # Ensure the spectra is 2D with a spectra per particle
        if spectra.ndim != 2:
            raise exceptions.InconsistentArguments(
                "Spectra must be a 2D array for an IFU image"
                f" (got {spectra.ndim})."
            )
        if spectra.shape[0] != cent_coords.shape[0]:
            raise exceptions.InconsistentArguments(
                "Spectra and coordinates must be the same size"
                f" for an IFU image (got {spectra.shape[0]} and "
                f"{cent_coords.shape[0]})."
            )

        # Ensure the coordinates are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(cent_coords, ifu.resolution.units):
            raise exceptions.InconsistentArguments(
                "Coordinates must be compatible with the IFU resolution units"
                f" (got {cent_coords.units} and {ifu.resolution.units})."
            )

        # Get the spatial units we'll work with
        spatial_units = ifu.resolution.units

        # Get some IFU properties we'll need
        fov = ifu.fov.to_value(spatial_units)
        res = ifu.resolution.to_value(spatial_units)

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        _coords = cent_coords.to(spatial_units).value

        # Ensure coordinates have been centred
        _validate_centered_coordinates(cent_coords, warn_only=True)

        # Prepare the inputs, we need to make sure we are passing C contiguous
        # arrays.
        _coords[:, 0] += fov[0] / 2
        _coords[:, 1] += fov[1] / 2
        smls = np.zeros(cent_coords.shape[0], dtype=np.float64)

        # Get the kernel
        # TODO: We should do away with this and write a histogram backend
        kernel = Kernel().get_kernel()

    ifu.arr = make_img(
        ensure_array_c_compatible_double(spectra),
        smls,
        ensure_array_c_compatible_double(_coords),
        kernel,
        res,
        ifu.npix[0],
        ifu.npix[1],
        cent_coords.shape[0],
        1,
        kernel.size,
        sed.nlam,
        nthreads,
    )

    return ifu


@timed("_generate_ifu_particle_smoothed")
def _generate_ifu_particle_smoothed(
    ifu,
    sed,
    quantity,
    cent_coords,
    smoothing_lengths,
    kernel,
    kernel_threshold,
    nthreads,
):
    """Generate a histogram IFU for a particle emitter.

    Args:
        ifu (SpectralCube):
            The SpectralCube object to populate with the ifu.
        sed (Sed):
            The Sed containing the spectra to sort into the IFU. For a
            particle emitter this should be a spectrum per particle, i.e.
            a 2D array of spectra with shape (Nparticles, Nlam).
        quantity (str):
            The quantity to use for the spectra. This can be any valid
            spectra quantity on an Sed object, e.g. 'lnu', 'fnu', 'luminosity',
            'flux', etc.
        cent_coords (unyt_array of float):
            The centered coordinates of the particles.
        smoothing_lengths (unyt_array of float):
            The smoothing lengths of the particles. These will be
            converted to the image resolution units.
        kernel (str):
            The array describing the kernel. This is derived from the
            kernel_functions module.
        kernel_threshold (float):
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image.
        nthreads (int):
            The number of threads to use when smoothing the image.

    Returns:
        SpectralCube: The histogram image.
    """
    with timer("_generate_ifu_particle_smoothed.setup"):
        # Sample the spectra onto the wavelength grid
        sed = sed.get_resampled_sed(new_lam=ifu.lam)

        # Store the Sed and quantity
        ifu.sed = sed
        ifu.quantity = quantity

        # Get the spectra we will be sorting into the spectral cube
        spectra = getattr(sed, quantity, None)
        if spectra is None:
            raise exceptions.MissingSpectraType(
                f"Can't make an image for {quantity},"
                " it does not exist in the Sed."
            )

        # Strip off and store the units on the spectra for later
        ifu.units = spectra.units
        # TODO: Rethink IFU path to avoid contiguous conversion.
        # Consider an IFU-specific backend that consumes native layout.
        spectra = ensure_array_c_compatible_double(spectra.ndview)

        # Ensure the spectra is 2D with a spectra per particle
        if spectra.ndim != 2:
            raise exceptions.InconsistentArguments(
                f"Spectra must be a 2D array for an IFU (got {spectra.ndim})."
            )
        if spectra.shape[0] != cent_coords.shape[0]:
            raise exceptions.InconsistentArguments(
                "Spectra and coordinates must be the same size"
                f" for an IFU (got {spectra.shape[0]} and "
                f"{cent_coords.shape[0]})."
            )

        # Ensure the coordinates are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(cent_coords, ifu.resolution.units):
            raise exceptions.InconsistentArguments(
                "Coordinates must be compatible with the IFU resolution units"
                f" (got {cent_coords.units} and {ifu.resolution.units})."
            )

        # Ensure the smoothing lengths are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point
        if not unit_is_compatible(smoothing_lengths, ifu.resolution.units):
            raise exceptions.InconsistentArguments(
                "Smoothing lengths must be compatible with the IFU resolution "
                f"units (got {smoothing_lengths.units} and "
                f"{ifu.resolution.units})."
            )

        # Get the spatial units we'll work with
        spatial_units = ifu.resolution.units

        # Get some IFU properties we'll need
        fov = ifu.fov.to_value(spatial_units)
        res = ifu.resolution.to_value(spatial_units)

        # Convert coordinates and smoothing lengths to the correct units and
        # strip them off
        _coords = cent_coords.to_value(spatial_units)

        # Ensure coordinates have been centred
        _validate_centered_coordinates(cent_coords)

        # Shift the centred coordinates by half the FOV to lie in the
        # range [0, FOV]
        _coords[:, 0] += fov[0] / 2
        _coords[:, 1] += fov[1] / 2

    # Generate the IFU
    ifu.arr = make_img(
        spectra,
        ensure_array_c_compatible_double(
            smoothing_lengths.to_value(spatial_units)
        ),
        ensure_array_c_compatible_double(_coords),
        kernel,
        res,
        ifu.npix[0],
        ifu.npix[1],
        cent_coords.shape[0],
        kernel_threshold,
        kernel.size,
        sed.nlam,
        nthreads,
    )

    return ifu


@timed("_generate_ifu_parametric_smoothed")
def _generate_ifu_parametric_smoothed(
    ifu,
    sed,
    quantity,
    density_grid,
):
    """Generate a smoothed IFU for a parametric emitter.

    Args:
        ifu (SpectralCube):
            The SpectralCube object to populate with the ifu.
        sed (Sed):
            The Sed containing the spectra to sort into the IFU. For a
            parametric emitter this should be a single integrated spectrum.
        quantity (str):
            The quantity to use for the spectra. This can be any valid
            spectra quantity on an Sed object, e.g. 'lnu', 'fnu', 'luminosity',
            'flux', etc.
        density_grid (unyt_array of float):
            The density grid to be smoothed over.
    """
    # Sample the spectra onto the wavelength grid if we need to
    sed = sed.get_resampled_sed(new_lam=ifu.lam)

    # Store the Sed and quantity
    ifu.sed = sed
    ifu.quantity = quantity

    # Get the spectra we will be sorting into the spectral cube
    spectra = getattr(sed, quantity, None)
    if spectra is None:
        raise exceptions.MissingSpectraType(
            f"Can't make an image for {quantity},"
            " it does not exist in the Sed."
        )

    # Strip off and store the units on the spectra for later
    ifu.units = spectra.units
    spectra = spectra.value

    # Ensure the spectra is integrated, i.e. 1D
    if spectra.ndim != 1:
        raise exceptions.InconsistentArguments(
            "Spectra must be a 1D array for a parametric IFU"
            f" (got {spectra.ndim})."
        )

    # Multiply the density grid by the sed to get the IFU
    ifu.arr = density_grid[:, :, None] * spectra

    return ifu


@timed("_generate_ifu_generic")
def _generate_ifu_generic(
    instrument,
    fov,
    lam,
    img_type,
    quantity,
    per_particle,
    kernel,
    kernel_threshold,
    nthreads,
    label,
    emitter,
    cosmo,
):
    """Generate a spectral cube.

    This function can be used to avoid repeating IFU generation code in
    wrappers elsewhere in the code. It'll produce a SpectralCube based
    on the input Sed.

    Particle based imaging can either be hist or smoothed, while parametric
    imaging can only be smoothed.

    Args:
        instrument (Instrument):
            The instrument to create the images for.
        fov (unyt_quantity/tuple, unyt_quantity):
            The width of the image.
        lam (unyt_array):
            The wavelength array of the spectra.
        img_type (str):
            The type of image to create. Options are "hist" or "smoothed".
        quantity (str):
            The quantity to use for the spectra. This can be any valid
            spectra quantity on an Sed object, e.g. 'lnu', 'fnu', 'luminosity',
            'flux', etc.
        per_particle (bool):
            Whether to create an image per particle or not.
        kernel (str):
            The array describing the kernel. This is dervied from the
            kernel_functions module. (Only applicable to particle imaging)
        kernel_threshold (float):
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image. (Only
            applicable to particle imaging)
        nthreads (int):
            The number of threads to use when smoothing the image. This
            only applies to particle imaging.
        label (str):
            The label of the photometry to use.
        emitter (Stars/BlackHoles/BlackHole):
            The emitter object to create the images for.
        cosmo (astropy.cosmology.Cosmology):
            A cosmology object defining the cosmology to use for the IFU.
            This is only relevant for angular IFUs where a conversion to
            projected angular coordinates is needed.

    Returns:
        SpectralCube: The generated spectral data cube.
    """
    # Avoid cyclic imports
    from synthesizer.imaging import SpectralCube
    from synthesizer.particle import Particles

    # Extract the Sed from the emitter
    try:
        sed = (
            emitter.particle_spectra[label]
            if per_particle
            else emitter.spectra[label]
        )
    except KeyError:
        # Ok we are missing the photometry
        raise exceptions.MissingSpectraType(
            f"Can't make a SpectralCube for {label} without an spectra. "
            "Did you not save the spectra or produce the photometry?"
        )

    # For particle emitters, standardize units to ensure resolution, fov,
    # and emitter data are all in the same system (both angular or both
    # Cartesian). Parametric emitters handle their own geometry via
    # morphology.get_density_grid() and don't need this standardization.
    # NOTE: This does NOT modify the emitter's underlying data
    if isinstance(emitter, Particles):
        needs_smoothing_lengths = img_type == "smoothed"
        resolution, fov, coords, smls = _standardize_imaging_units(
            resolution=instrument.resolution,
            fov=fov,
            emitter=emitter,
            cosmo=cosmo,
            include_smoothing_lengths=needs_smoothing_lengths,
        )

        # Validate that smoothing lengths exist for smoothed particle imaging
        if img_type == "smoothed" and smls is None:
            raise exceptions.InconsistentArguments(
                "Smoothed particle imaging requires smoothing_lengths. "
                "The emitter must have a smoothing_lengths attribute."
            )
    else:
        # Parametric emitters: keep original resolution/fov, skip coord
        # standardization
        resolution = instrument.resolution
        fov = fov
        coords = smls = None

    # Create the IFU with standardized units
    ifu = SpectralCube(
        resolution=resolution,
        lam=lam * angstrom,
        fov=fov,
    )

    # Make the IFU handling the different types of generation
    # NOTE: Black holes are always a histogram, safer to just hack this here
    # since a user can set the "global" method as smoothed for a galaxy
    # with both stars and black holes.
    if (img_type == "hist" and isinstance(emitter, Particles)) or (
        getattr(emitter, "name", None) == "Black Holes"
    ):
        # Use the standardized coordinates (already in correct units)
        return _generate_ifu_particle_hist(
            ifu,
            sed=sed,
            quantity=quantity,
            cent_coords=coords,
            nthreads=nthreads,
        )

    elif img_type == "hist":
        raise exceptions.InconsistentArguments(
            "Parametric IFU can only be made using the smoothed img type."
        )

    elif img_type == "smoothed" and isinstance(emitter, Particles):
        # Use the standardized coordinates and smoothing lengths
        # (already in correct units)
        return _generate_ifu_particle_smoothed(
            ifu,
            sed=sed,
            quantity=quantity,
            cent_coords=coords,
            smoothing_lengths=smls,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
        )

    elif img_type == "smoothed":
        # FYI, the parametric imaging handles the angular vs cartesian
        # image properties internally so we don't need to worry about it here
        return _generate_ifu_parametric_smoothed(
            ifu,
            sed=sed,
            quantity=quantity,
            density_grid=emitter.morphology.get_density_grid(
                ifu.resolution, ifu.npix
            ),
        )
    else:
        raise exceptions.UnknownImageType(
            f"Unknown img_type {img_type} for a {type(emitter)} emitter. "
            " (Options are 'hist' (only for particle based emitters)"
            " or 'smoothed')"
        )

    return ifu


def _prepare_component_image_labels(
    labels: list[str],
    model_cache: dict,
    remove_missing: bool = False,
) -> tuple[list[str], list[str]]:
    """Split component image labels into combined and generated labels.

    This function checks if any of the requested labels can be combined
    from other labels already in the request, avoiding unnecessary generation.

    Args:
        labels (list of str):
            The requested image labels (already fully expanded).
        model_cache (dict):
            The model parameter cache from the component emitter.
        remove_missing (bool):
            Whether to remove labels missing from the model cache entirely.

    Returns:
        tuple of list of str:
            - combine_labels: Labels that can be combined from others.
            - generate_labels: Labels that must be generated directly.
    """
    combine_labels_set = set()
    generate_labels = []
    labels_set = set(labels)

    # Check each label to see if it can be combined from others already in
    # the list
    for label in labels:
        # Skip if not in cache
        if label not in model_cache:
            if not remove_missing:
                generate_labels.append(label)
            continue

        # Check if this label has combine keys
        combine_keys = model_cache[label].get("combine", [])

        # If all combine keys are already in our list, we can combine instead
        # of generate
        if combine_keys and all(key in labels_set for key in combine_keys):
            combine_labels_set.add(label)
        else:
            # Either no combine keys, or not all dependencies are in the list
            generate_labels.append(label)

    # Sort combine_labels in dependency order (dependencies first)
    combine_labels = []
    added = set()

    def add_in_order(label):
        """Add label after its dependencies."""
        if label in added or label not in model_cache:
            return
        # Add dependencies first
        combine_keys = model_cache[label].get("combine", [])
        for key in combine_keys:
            if key in combine_labels_set:
                add_in_order(key)
        # Then add this label
        if label not in added:
            combine_labels.append(label)
            added.add(label)

    # Add all combine labels in order to ensure dependencies are met before
    # we try to combine them
    for label in combine_labels_set:
        add_in_order(label)

    return combine_labels, generate_labels


def _prepare_galaxy_image_labels(
    labels: list[str],
    model_cache: dict,
) -> tuple[list[str], dict[str, list[str]]]:
    """Prepare galaxy-level image generation by routing to components.

    This function takes the requested labels and recursively expands
    galaxy-level combination models, routing component-level labels to
    the appropriate emitters. Component labels are recursively expanded
    until we get to the actual labels each component needs to handle.

    Args:
        labels (list of str):
            The requested image labels.
        model_cache (dict):
            The combined model parameter cache from all components and galaxy.

    Returns:
        tuple:
            - galaxy_combine_labels (list of str): Labels for galaxy-level
              combinations.
            - component_labels_by_emitter (dict): Dict mapping emitter type
              to list of labels that emitter needs to handle.
    """
    # Track galaxy-level combinations and all labels needed
    galaxy_combine_labels = []
    all_component_labels = set()
    visited = set()

    # Recursively expand galaxy-level models
    def expand_galaxy_label(label):
        """Recursively expand a label with cycle protection."""
        # Skip if already visited (cycle protection)
        if label in visited:
            return

        # Skip if not in cache
        if label not in model_cache:
            return

        # Mark as visited to prevent cycles
        visited.add(label)

        # Get the emitter for this model
        emitter = model_cache[label].get("emitter", None)

        # If this is a galaxy-level model, expand it
        if emitter == "galaxy":
            # Check if it's a combination model
            combine_keys = model_cache[label].get("combine", [])
            if combine_keys:
                # Recursively expand the combination keys first so that
                # dependencies are combined before their parents
                for key in combine_keys:
                    expand_galaxy_label(key)
                # Then add this label if not already present
                if label not in galaxy_combine_labels:
                    galaxy_combine_labels.append(label)
        else:
            # This is a component-level label
            if label not in all_component_labels:
                all_component_labels.add(label)

    # Expand all requested labels
    for label in labels:
        expand_galaxy_label(label)

    # Now route component labels to the appropriate emitters
    component_labels_by_emitter = {}
    for label in all_component_labels:
        if label in model_cache:
            emitter = model_cache[label].get("emitter", None)
            if emitter is not None:
                if emitter not in component_labels_by_emitter:
                    component_labels_by_emitter[emitter] = []
                component_labels_by_emitter[emitter].append(label)

    return galaxy_combine_labels, component_labels_by_emitter


def _cf_periodicity_dilution_correction_standalone(
    cf_shape,
) -> np.ndarray:
    """Calculates the correction factor for DFT-based Correlation Functions.

    to account for the assumption of periodicity.
    Ported from an internal GalSim function.

    Args:
        cf_shape: Tuple (Ny, Nx) representing the shape of the
        correlation function array.

    Returns:
        A 2D NumPy array with the correction factors.
    """
    ny, nx = cf_shape

    # Create coordinate arrays for frequency domain
    # dx_coords corresponds to frequencies in the x-direction (columns)
    # dy_coords corresponds to frequencies in the y-direction (rows)
    dx_coords = np.fft.fftfreq(nx) * float(nx)
    dy_coords = np.fft.fftfreq(ny) * float(ny)

    # Create 2D grids of these coordinates
    # deltax will have shape (Ny, Nx) and vary along columns (axis 1)
    # deltay will have shape (Ny, Nx) and vary along rows (axis 0)
    deltax, deltay = np.meshgrid(dx_coords, dy_coords)

    denominator = (nx - np.abs(deltax)) * (ny - np.abs(deltay))

    # Avoid division by zero if denominator entries are zero,
    # though this is unlikely with standard fftfreq outputs for valid shapes.
    # np.finfo(float).eps could be added for numerical stability if needed.
    valid_denominator = np.where(
        denominator == 0, 1.0, denominator
    )  # Avoid true div by zero

    correction = (float(nx * ny)) / valid_denominator
    if np.any(
        denominator == 0
    ):  # Put back zeros where they were to avoid infs if original was 0
        correction[denominator == 0] = 0  # Or handle as error/warning

    return correction


def _generate_noise_from_rootps_standalone(
    rng: np.random.Generator, shape, rootps: np.ndarray
) -> np.ndarray:
    """Generates a real-space noise field from its sqrt(PowerSpectrum).

    Ported and adapted from an internal GalSim function.

    Args:
        rng: NumPy random number generator.
        shape: Tuple (Ny, Nx) of the output real-space noise field.
        rootps: The half-complex 2D array (Ny, Nx//2 + 1) representing
                the square root of the Power Spectrum from rfft2.

    Returns:
        A 2D NumPy array representing the generated correlated noise field.
    """
    ny, nx = shape

    # GalSim's GaussianDeviate implies specific scaling for random numbers.
    # E[|gvec_k|^2] = Ny * Nx for each complex component k
    # if parts are N(0, 0.5*Ny*Nx).
    sigma_val_for_gvec_parts = np.sqrt(0.5 * ny * nx)

    gvec_real = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec_imag = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec = gvec_real + 1j * gvec_imag

    # Impose Hermitian symmetry properties and scaling for DC/Nyquist terms.
    # This ensures the iFFT results in a real field
    # with correct variance distribution.
    rt2 = np.sqrt(2.0)

    # DC component (ky=0, kx=0)
    gvec[0, 0] = rt2 * gvec[0, 0].real

    # Nyquist frequency for y-axis (ky=Ny/2) at kx=0 (if Ny is even)
    if ny % 2 == 0:
        gvec[ny // 2, 0] = rt2 * gvec[ny // 2, 0].real

    # Nyquist frequency for x-axis (kx=Nx/2) at ky=0 (if Nx is even)
    if nx % 2 == 0:
        gvec[0, nx // 2] = rt2 * gvec[0, nx // 2].real

    # Corner Nyquist component (ky=Ny/2, kx=Nx/2) (if both Ny and Nx are even)
    if ny % 2 == 0 and nx % 2 == 0:
        gvec[ny // 2, nx // 2] = rt2 * gvec[ny // 2, nx // 2].real

    # Conjugate symmetry for the kx=0 column (y-axis frequencies)
    # gvec[Ny-row, 0] = conj(gvec[row, 0]) for row in 1 .. (Ny/2 - 1)
    # or ((Ny-1)/2)
    # This handles the negative y-frequencies for kx=0.
    # The slice `ny-1 : ny//2 : -1` covers rows from Ny-1 down to (Ny//2 + 1).
    # The slice `1 : (ny+1)//2` covers rows from 1 up to Ny//2.
    if ny > 1:  # This symmetry operation is relevant
        # if Ny > 1 (or Ny > 2 for some ranges)
        gvec[ny - 1 : ny // 2 : -1, 0] = np.conj(gvec[1 : (ny + 1) // 2, 0])

    # Conjugate symmetry for the kx=Nx/2 column (if Nx is even)
    # This handles negative y-frequencies for the Nyquist x-frequency.
    if nx % 2 == 0:
        kx_nyq_idx = nx // 2
        if ny > 1:
            gvec[ny - 1 : ny // 2 : -1, kx_nyq_idx] = np.conj(
                gvec[1 : (ny + 1) // 2, kx_nyq_idx]
            )

    # Element-wise multiplication in Fourier space
    noise_field_k_space = gvec * rootps

    # Inverse FFT to get the real-space noise field
    # The `s` parameter ensures the output shape matches the desired `shape`.
    noise_real_space = np.fft.irfft2(noise_field_k_space, s=shape)

    return noise_real_space


def _estimate_correlated_noise_cf(
    source_arr: np.ndarray,
    subtract_mean: bool = False,
    correct_periodicity: bool = True,
) -> np.ndarray:
    """Estimate the correlation function (CF) of a 2D noise array.

    Computes an unrolled correlation function (DFT convention, origin at
    [0, 0]) from the power spectrum of the source array. The result is
    intended to be cached and later consumed by ``_generate_correlated_noise``
    to produce noise realisations for images of arbitrary shape.

    This function is deliberately separated from noise generation so that the
    expensive FFT step can be computed once (e.g. per instrument filter) and
    reused across many images.

    Args:
        source_arr (np.ndarray):
            A 2D float array of observed noise (e.g. a blank-sky cutout).
            Must already have units stripped before calling.
        subtract_mean (bool):
            If True the DC component of the power spectrum is zeroed, which
            effectively removes the mean offset from the noise model.
        correct_periodicity (bool):
            If True a correction factor is applied to compensate for the
            assumption of periodicity in the DFT.

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
    ``_estimate_correlated_noise_cf`` and cached at the call site — to
    produce a noise field whose spatial statistics match those of the
    original noise template.  The CF is resampled to the target pixel
    dimensions before noise generation.

    Args:
        cf_source (np.ndarray):
            The unrolled 2D correlation function (DFT convention, origin at
            [0, 0]) previously returned by ``_estimate_correlated_noise_cf``.
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

    # Roll source CF so that the peak (variance, at [0,0]) moves to the centre
    cf_source_rolled = np.roll(
        cf_source,
        shift=(source_shape[0] // 2, source_shape[1] // 2),
        axis=(0, 1),
    )

    # Copy the central region of the source CF into an array of target_shape
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

    # Unroll back to DFT convention (origin at [0, 0])
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

    This is a convenience wrapper around ``_estimate_correlated_noise_cf``
    and ``_generate_correlated_noise``.  Prefer calling those two functions
    directly when the correlation function should be cached across multiple
    calls (e.g. when applying the same noise model to many images).

    Args:
        source_image_arr (np.ndarray):
            A 2D array used to model the correlated noise characteristics.
        target_image_arr (np.ndarray):
            A 2D array to which the generated correlated noise is added.
        subtract_mean (bool):
            If True the DC component of the power spectrum is zeroed before
            estimating the CF. Default is False.
        correct_periodicity (bool):
            If True a correction factor is applied to compensate for the
            assumption of periodicity in the DFT. Default is True.
        rng_seed (int, optional):
            Seed for the random number generator for reproducibility.

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
