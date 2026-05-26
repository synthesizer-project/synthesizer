"""Helpers for post-processing existing imaging products.

This module contains shared orchestration helpers for applying instrument
effects to already-generated images.
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
    if not instrument.can_do_psf_imaging:
        return final_images

    psf_store.setdefault(instrument.label, {})
    for label, imgs in final_images.items():
        psf_store[instrument.label][label] = instrument.apply_psfs(imgs)

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
    if not instrument.can_do_noisy_imaging:
        return final_images

    noise_store.setdefault(instrument.label, {})
    for label, imgs in final_images.items():
        noise_store[instrument.label][label] = instrument.apply_noises(
            imgs,
            aperture_radius=instrument.depth_app_radius,
        )

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
