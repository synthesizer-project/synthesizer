"""A submodule containing generator functions for producing data cubes.

Synthesizer supports generating both histogram and smoothed data cubes
depending on the inputs. Particle based emitters can be histogram or
smoothed, while parametric emitters can only be smoothed. These functions
abstract away the complications of these different methods.

These can be accessed either by calling the low level SpectralCube class
directly, or by using the higher level functions on galaxies and their
components (get_data_cube).

The functions in this module are not intended to be called directly by the
user.
"""

from copy import deepcopy

import numpy as np
from unyt import angstrom

from synthesizer import exceptions
from synthesizer.imaging.extensions.image import make_img
from synthesizer.imaging.image_generators import (
    _prepare_component_image_labels,
    _prepare_galaxy_image_labels,
    _standardize_imaging_units,
    _standardize_sph_kernel,
    _validate_centered_coordinates,
)
from synthesizer.kernel_functions import Kernel
from synthesizer.units import unit_is_compatible
from synthesizer.utils import ensure_array_c_compatible_double
from synthesizer.utils.operation_timers import timed, timer


def _combine_spectral_cubes(cubes, label, model_cache):
    """Combine multiple spectral cubes into a single spectral cube.

    Args:
        cubes: Dictionary of existing spectral cubes.
        label: The label of the combined spectral cube to create.
        model_cache: The model parameter cache to use for lookup.

    Returns:
        SpectralCube: The combined spectral cube.

    Raises:
        MissingModel: If label not found in model_cache.
        MissingIFU: If any required component cubes are missing.
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

    # Find the cubes we need to combine from the provided model cache
    combine_keys = model_cache[label]["combine"]

    # Validate all combine_keys exist in cubes
    missing_keys = [key for key in combine_keys if key not in cubes]

    # Stop early if any dependency cubes are unavailable
    if missing_keys:
        raise exceptions.MissingIFU(
            f"Cannot combine data cubes for '{label}': missing cubes for "
            f"{', '.join(missing_keys)}"
        )

    # Get all the cubes to add
    combine_cubes = [cubes[key] for key in combine_keys]

    # Combine the cubes
    combined_cube = deepcopy(combine_cubes[0])
    for cube in combine_cubes[1:]:
        combined_cube += cube

    return combined_cube


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

        # Stop early if the requested spectral quantity is unavailable
        if spectra is None:
            raise exceptions.MissingSpectraType(
                f"Can't make an image for {quantity},"
                " it does not exist in the Sed."
            )

        # Strip off and store the units on the spectra for later
        ifu.units = spectra.units
        spectra = spectra.ndview

        # Ensure the spectra is 2D with a spectra per particle

        # Reject non-particle spectral layouts before calling the backend
        if spectra.ndim != 2:
            raise exceptions.InconsistentArguments(
                "Spectra must be a 2D array for an IFU image"
                f" (got {spectra.ndim})."
            )

        # Reject coordinate arrays that do not match the spectra count
        if spectra.shape[0] != cent_coords.shape[0]:
            raise exceptions.InconsistentArguments(
                "Spectra and coordinates must be the same size"
                f" for an IFU image (got {spectra.shape[0]} and "
                f"{cent_coords.shape[0]})."
            )

        # Ensure the coordinates are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point

        # Reject coordinates in incompatible spatial units
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
        kernel = _standardize_sph_kernel(Kernel())

    # Generate the histogram IFU from the prepared particle inputs
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
    """Generate a smoothed IFU for a particle emitter.

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
        kernel (np.ndarray or Kernel):
            The kernel lookup table, or a ``Kernel`` instance to extract it
            from.
        kernel_threshold (float):
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image.
        nthreads (int):
            The number of threads to use when smoothing the image.

    Returns:
        SpectralCube: The smoothed IFU.
    """
    with timer("_generate_ifu_particle_smoothed.setup"):
        # Sample the spectra onto the wavelength grid
        sed = sed.get_resampled_sed(new_lam=ifu.lam)

        # Store the Sed and quantity
        ifu.sed = sed
        ifu.quantity = quantity

        # Get the spectra we will be sorting into the spectral cube
        spectra = getattr(sed, quantity, None)

        # Stop early if the requested spectral quantity is unavailable
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

        # Reject non-particle spectral layouts before calling the backend
        if spectra.ndim != 2:
            raise exceptions.InconsistentArguments(
                f"Spectra must be a 2D array for an IFU (got {spectra.ndim})."
            )

        # Reject coordinate arrays that do not match the spectra count
        if spectra.shape[0] != cent_coords.shape[0]:
            raise exceptions.InconsistentArguments(
                "Spectra and coordinates must be the same size"
                f" for an IFU (got {spectra.shape[0]} and "
                f"{cent_coords.shape[0]})."
            )

        # Ensure the coordinates are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point

        # Reject coordinates in incompatible spatial units
        if not unit_is_compatible(cent_coords, ifu.resolution.units):
            raise exceptions.InconsistentArguments(
                "Coordinates must be compatible with the IFU resolution units"
                f" (got {cent_coords.units} and {ifu.resolution.units})."
            )

        # Ensure the smoothing lengths are compatible with the fov/resolution
        # Note that the resolution and fov are already guaranteed to be
        # compatible with each other at this point

        # Reject smoothing lengths in incompatible spatial units
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

    # Convert the public kernel input into the lookup array used by the
    # smoothing backend.
    kernel_arr = _standardize_sph_kernel(kernel)

    # Generate the smoothed IFU from the prepared particle inputs
    ifu.arr = make_img(
        spectra,
        ensure_array_c_compatible_double(
            smoothing_lengths.to_value(spatial_units)
        ),
        ensure_array_c_compatible_double(_coords),
        kernel_arr,
        res,
        ifu.npix[0],
        ifu.npix[1],
        cent_coords.shape[0],
        kernel_threshold,
        kernel_arr.size,
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

    # Stop early if the requested spectral quantity is unavailable
    if spectra is None:
        raise exceptions.MissingSpectraType(
            f"Can't make an image for {quantity},"
            " it does not exist in the Sed."
        )

    # Strip off and store the units on the spectra for later
    ifu.units = spectra.units
    spectra = spectra.value

    # Ensure the spectra is integrated, i.e. 1D

    # Reject per-particle spectra on the parametric IFU path
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
            The saved spectrum label to use.
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

    # Resolve the saved spectrum from the emitter using the requested label
    try:
        sed = (
            emitter.particle_spectra[label]
            if per_particle
            else emitter.spectra[label]
        )
    except KeyError:
        # Fail explicitly when the required saved spectrum is unavailable
        raise exceptions.MissingSpectraType(
            f"Can't make a SpectralCube for {label} without an spectra. "
            "Did you not save the spectra or produce the photometry?"
        )

    # For particle emitters, standardize units to ensure resolution, fov,
    # and emitter data are all in the same system (both angular or both
    # Cartesian). Parametric emitters handle their own geometry via
    # morphology.get_density_grid() and don't need this standardization.
    # NOTE: This does NOT modify the emitter's underlying data

    # Standardize particle geometry inputs before selecting a cube backend
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

        # Stop early if the smoothed path lacks smoothing lengths
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

    # Create the output cube container using the standardized geometry
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
        # Route histogram cubes through the shared particle-IFU backend
        return _generate_ifu_particle_hist(
            ifu,
            sed=sed,
            quantity=quantity,
            cent_coords=coords,
            nthreads=nthreads,
        )

    elif img_type == "hist":
        # Reject histogram requests for parametric cube generation
        raise exceptions.InconsistentArguments(
            "Parametric IFU can only be made using the smoothed img type."
        )

    elif img_type == "smoothed" and isinstance(emitter, Particles):
        # Route smoothed particle cubes through the shared IFU backend
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
        # Route parametric cubes through the density-grid IFU backend
        return _generate_ifu_parametric_smoothed(
            ifu,
            sed=sed,
            quantity=quantity,
            density_grid=emitter.morphology.get_density_grid(
                ifu.resolution, ifu.npix
            ),
        )
    else:
        # Reject any unsupported cube-generation mode explicitly
        raise exceptions.UnknownImageType(
            f"Unknown img_type {img_type} for a {type(emitter)} emitter. "
            " (Options are 'hist' (only for particle based emitters)"
            " or 'smoothed')"
        )

    return ifu


def _prepare_component_data_cube_labels(
    labels: list[str],
    model_cache: dict,
    remove_missing: bool = False,
) -> tuple[list[str], list[str]]:
    """Split component data-cube labels into combined and generated labels.

    Args:
        labels (list of str):
            The requested data-cube labels.
        model_cache (dict):
            The model parameter cache from the component emitter.
        remove_missing (bool):
            Whether to remove labels missing from the model cache entirely.

    Returns:
        tuple of list of str:
            - combine_labels: Labels that can be combined from others.
            - generate_labels: Labels that must be generated directly.
    """
    # Reuse the image-label dependency logic for cube label routing
    return _prepare_component_image_labels(
        labels,
        model_cache,
        remove_missing=remove_missing,
    )


def _prepare_galaxy_data_cube_labels(
    labels: list[str],
    model_cache: dict,
) -> tuple[list[str], dict[str, list[str]]]:
    """Prepare galaxy-level data-cube generation by routing to components.

    Args:
        labels (list of str):
            The requested data-cube labels.
        model_cache (dict):
            The combined model parameter cache from all components and galaxy.

    Returns:
        tuple:
            - galaxy_combine_labels (list of str): Labels for galaxy-level
              combinations.
            - component_labels_by_emitter (dict): Dict mapping emitter type
              to list of labels that emitter needs to handle.
    """
    # Reuse the galaxy image-routing logic because cube label expansion is
    # structurally identical
    return _prepare_galaxy_image_labels(labels, model_cache)
