"""Helpers for post-processing existing observables.

This module contains shared orchestration helpers for applying instrument
effects to already-generated images and data cubes.
"""

from synthesizer import exceptions


def _get_image_postprocess_stores(owner, phot_type):
    """Return the raw, PSF, and noise stores for an image family.

    Args:
        owner:
            Object holding the image stores.
        phot_type (str):
            Either ``"lnu"`` or ``"fnu"``.

    Returns:
        tuple:
            ``(raw_store, psf_store, noise_store)`` for the requested
            photometry family.
    """
    # Select the correct raw-image cache and lazily ensure the matching
    # post-processed stores exist on the owner.
    if phot_type == "lnu":
        raw_store = owner.images_lnu
        if not hasattr(owner, "images_psf_lnu"):
            owner.images_psf_lnu = {}
        if not hasattr(owner, "images_noise_lnu"):
            owner.images_noise_lnu = {}
        return raw_store, owner.images_psf_lnu, owner.images_noise_lnu

    if phot_type == "fnu":
        raw_store = owner.images_fnu
        if not hasattr(owner, "images_psf_fnu"):
            owner.images_psf_fnu = {}
        if not hasattr(owner, "images_noise_fnu"):
            owner.images_noise_fnu = {}
        return raw_store, owner.images_psf_fnu, owner.images_noise_fnu

    raise exceptions.InconsistentArguments(
        f"Photometry type {phot_type} not recognised. Must be 'lnu' or 'fnu'."
    )


def _get_raw_images_for_postprocess(
    raw_store, instrument_label, limit_to=None
):
    """Resolve the raw images to post-process from image storage.

    Args:
        raw_store (dict):
            Instrument-keyed store of raw images.
        instrument_label (str):
            Label of the instrument whose images should be post-processed.
        limit_to (list, optional):
            Specific labels to post-process.

    Returns:
        dict:
            Raw image collections keyed by label.
    """
    # Pull the raw images for this instrument and optionally trim to a caller-
    # supplied subset of labels.
    raw_images = raw_store.get(instrument_label, {})
    labels = raw_images.keys() if limit_to is None else limit_to
    return {
        label: raw_images[label] for label in labels if label in raw_images
    }


def _apply_image_psfs(final_images, psf_store, instrument):
    """Apply the instrument PSF configuration to image collections.

    Args:
        final_images (dict):
            Current image collections keyed by label.
        psf_store (dict):
            Store for PSF-processed images.
        instrument (Instrument):
            Instrument defining the PSF application.

    Returns:
        dict:
            Updated image collections after PSF processing.
    """
    # Skip this stage entirely when the instrument has no imaging PSF model.
    if not instrument.can_do_psf_imaging:
        return final_images

    # Write the latest PSF-convolved state back to the owner cache so later
    # accesses see the same observable returned to the caller.
    psf_store.setdefault(instrument.label, {})
    for label, imgs in final_images.items():
        psf_store[instrument.label][label] = instrument.apply_psfs(imgs)

    # Return the updated view of the just-written PSF-processed images.
    return {
        label: psf_store[instrument.label][label] for label in final_images
    }


def _apply_image_noise(final_images, noise_store, instrument):
    """Apply the instrument noise configuration to image collections.

    Args:
        final_images (dict):
            Current image collections keyed by label.
        noise_store (dict):
            Store for noise-processed images.
        instrument (Instrument):
            Instrument defining the noise application.

    Returns:
        dict:
            Updated image collections after noise processing.
    """
    # Skip this stage entirely when the instrument has no configured imaging
    # noise model.
    if not instrument.can_do_noisy_imaging:
        return final_images

    # Write the latest noisy state back to the owner cache so later accesses
    # see the same observable returned to the caller.
    noise_store.setdefault(instrument.label, {})
    for label, imgs in final_images.items():
        noise_store[instrument.label][label] = instrument.apply_noises(
            imgs,
            aperture_radius=instrument.depth_app_radius,
        )

    # Return the updated view of the just-written noisy images.
    return {
        label: noise_store[instrument.label][label] for label in final_images
    }


def _postprocess_existing_images(
    owner,
    instrument,
    phot_type,
    limit_to=None,
):
    """Apply the instrument-defined imaging post-processing to images.

    Args:
        owner:
            Object holding the raw and post-processed image stores. This is
            typically a Component or BaseGalaxy instance.
        instrument (Instrument):
            Instrument defining the observation.
        phot_type (str):
            Either ``"lnu"`` or ``"fnu"``.
        limit_to (list, optional):
            Specific labels to post-process.

    Returns:
        dict:
            Final image collections keyed by label.
    """
    # Resolve the appropriate raw and post-processed stores for the requested
    # photometry family.
    raw_store, psf_store, noise_store = _get_image_postprocess_stores(
        owner, phot_type
    )

    # Resolve the raw images we are post-processing from storage populated
    # during generation.
    final_images = _get_raw_images_for_postprocess(
        raw_store,
        instrument.label,
        limit_to=limit_to,
    )

    # Apply PSFs first so any subsequent noise model acts on the observed
    # PSF-convolved image.
    final_images = _apply_image_psfs(final_images, psf_store, instrument)

    # Apply the configured instrument noise to the latest image state.
    final_images = _apply_image_noise(final_images, noise_store, instrument)

    return final_images


def _get_data_cube_postprocess_store(owner, quantity):
    """Return the raw cube store for a spectral quantity family.

    Args:
        owner:
            Object holding the data-cube stores.
        quantity (str):
            Spectral quantity family.

    Returns:
        dict:
            Instrument-keyed store of raw cubes.
    """
    # Map the requested spectral quantity family onto the corresponding raw
    # cube cache on the owner.
    if quantity in {"lnu", "llam", "luminosity"}:
        return owner.data_cubes_lnu
    if quantity in {"fnu", "flam", "flux"}:
        return owner.data_cubes_fnu

    # Unknown quantity families are treated as non-cacheable here and simply
    # yield no post-processing work.
    return {}


def _get_raw_data_cubes_for_postprocess(
    cube_store, instrument_label, limit_to=None
):
    """Resolve the raw data cubes to post-process from storage.

    Args:
        cube_store (dict):
            Instrument-keyed store of raw data cubes.
        instrument_label (str):
            Label of the instrument whose cubes should be post-processed.
        limit_to (list, optional):
            Specific labels to post-process.

    Returns:
        dict:
            Raw cubes keyed by label.
    """
    # Pull the raw cubes for this instrument and optionally trim to a caller-
    # supplied subset of labels.
    raw_cubes = cube_store.get(instrument_label, {})
    labels = raw_cubes.keys() if limit_to is None else limit_to
    return {label: raw_cubes[label] for label in labels if label in raw_cubes}


def _apply_data_cube_psf(final_cubes, cube_store, instrument):
    """Apply the instrument PSF configuration to data cubes.

    Args:
        final_cubes (dict):
            Current data cubes keyed by label.
        cube_store (dict):
            Store containing the latest cube state.
        instrument (IntegratedFieldUnit):
            Instrument defining the PSF application.

    Returns:
        dict:
            Updated cubes after PSF processing.
    """
    # Skip this stage entirely when the IFU has no resolved-spectroscopy PSF.
    if not instrument.can_do_psf_spectroscopy:
        return final_cubes

    # Write the latest PSF-convolved cube state back to the owner cache so
    # later accesses see the same observable returned to the caller.
    cube_store.setdefault(instrument.label, {})
    for label, cube in final_cubes.items():
        cube_store[instrument.label][label] = instrument.apply_psf(cube)

    # Return the updated view of the just-written PSF-processed cubes.
    return {
        label: cube_store[instrument.label][label] for label in final_cubes
    }


def _apply_data_cube_noise(final_cubes, cube_store, instrument):
    """Apply the instrument noise configuration to data cubes.

    Args:
        final_cubes (dict):
            Current data cubes keyed by label.
        cube_store (dict):
            Store containing the latest cube state.
        instrument (IntegratedFieldUnit):
            Instrument defining the noise application.

    Returns:
        dict:
            Updated cubes after noise processing.
    """
    # Skip this stage entirely when the IFU has no configured resolved-spectra
    # noise model.
    if not getattr(instrument, "can_do_noisy_resolved_spectroscopy", False):
        return final_cubes

    # Write the latest noisy cube state back to the owner cache so later
    # accesses see the same observable returned to the caller.
    cube_store.setdefault(instrument.label, {})
    for label, cube in final_cubes.items():
        cube_store[instrument.label][label] = instrument.apply_noise(cube)

    # Return the updated view of the just-written noisy cubes.
    return {
        label: cube_store[instrument.label][label] for label in final_cubes
    }


def _postprocess_existing_data_cubes(
    owner,
    instrument,
    quantity,
    limit_to=None,
):
    """Apply the instrument-defined IFU post-processing to cubes.

    Args:
        owner:
            Object holding the raw cube stores. This is typically a Component
            or BaseGalaxy instance.
        instrument (IntegratedFieldUnit):
            Instrument defining the observation.
        quantity (str):
            Spectral quantity family for selecting the store.
        limit_to (list, optional):
            Specific labels to post-process.

    Returns:
        dict:
            Final cubes keyed by label.
    """
    # Resolve the appropriate cube store for the requested quantity family.
    cube_store = _get_data_cube_postprocess_store(owner, quantity)

    # If the quantity family does not map onto a stored raw-cube cache there is
    # nothing to post-process here.
    if len(cube_store) == 0:
        return {}

    # Resolve the raw cubes we are post-processing from storage populated
    # during generation.
    final_cubes = _get_raw_data_cubes_for_postprocess(
        cube_store,
        instrument.label,
        limit_to=limit_to,
    )

    # Apply any configured IFU PSF before any configured IFU noise.
    final_cubes = _apply_data_cube_psf(final_cubes, cube_store, instrument)

    # Apply any configured IFU noise to the latest cube state.
    final_cubes = _apply_data_cube_noise(final_cubes, cube_store, instrument)

    return final_cubes
