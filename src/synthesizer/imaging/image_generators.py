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

import numpy as np
from unyt import angstrom, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.imaging.extensions.image import make_img
from synthesizer.kernel_functions import Kernel
from synthesizer.utils import (
    ensure_array_c_compatible_double,
)


def _generate_image_particle_hist(
    img,
    signal,
    coordinates,
    normalisation=None,
):
    """
    Generate a histogram image for a particle emitter.

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

    # Ensure coordinates have been centred
    if not (coordinates.min() < 0 and coordinates.max() > 0):
        raise exceptions.InconsistentArguments(
            "Coordinates must be centered for imaging"
            f" (got min={coordinates.min()} and max={coordinates.max()})."
        )

    # Strip off and store the units on the signal if they are present
    if isinstance(signal, (unyt_quantity, unyt_array)):
        img.units = signal.units
        signal = signal.value

    # Return an empty image if there are no particles
    if signal.size == 0:
        img.arr = np.zeros(img.npix)
        return img.arr * img.units if img.units is not None else img.arr

    # Convert coordinates and smoothing lengths to the correct units and
    # strip them off
    coordinates = coordinates.to(img.resolution.units).value

    # Include normalisation in the original signal if we have one
    # (we'll divide by it later)
    if normalisation is not None:
        signal *= normalisation.value

    img.arr = np.histogram2d(
        coordinates[:, 0],
        coordinates[:, 1],
        bins=(
            np.linspace(-img._fov[0] / 2, img._fov[0] / 2, img.npix[0] + 1),
            np.linspace(-img._fov[1] / 2, img._fov[1] / 2, img.npix[1] + 1),
        ),
        weights=signal,
    )[0]

    # Normalise the image by the normalisation if applicable
    if normalisation is not None:
        norm_img = np.histogram2d(
            coordinates[:, 0],
            coordinates[:, 1],
            bins=(
                np.linspace(
                    -img._fov[0] / 2, img._fov[0] / 2, img.npix[0] + 1
                ),
                np.linspace(img._fov[1] / 2, img._fov[1] / 2, img.npix[1] + 1),
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
    """
    Generate histogram images for a particle emitter.

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
    """
    Generate smoothed images for a particle emitter.

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

    # Ensure coordinates have been centred
    if not (cent_coords.min() < 0 and cent_coords.max() > 0):
        raise exceptions.InconsistentArguments(
            "Coordinates must be centered for imaging"
            f" (got min={cent_coords.min()} and max={cent_coords.max()})."
        )

    # Get the spatial units we'll work with
    spatial_units = img.resolution.units

    # Unpack the image properties we need
    fov = img.fov.to_value(spatial_units)
    res = img.resolution.to_value(spatial_units)

    # Shift the centred coordinates by half the FOV
    # (this is to ensure the image is centered on the emitter)
    cent_coords = cent_coords.to(spatial_units).value
    cent_coords[:, 0] += fov[0] / 2.0
    cent_coords[:, 1] += fov[1] / 2.0
    smoothing_lengths = smoothing_lengths.to_value(spatial_units)

    # Apply normalisation to original signal if needed
    if normalisation is not None:
        signal *= normalisation.value

    # Get the (npix_x, npix_y) image
    imgs_arr = make_img(
        signal,
        ensure_array_c_compatible_double(smoothing_lengths),
        ensure_array_c_compatible_double(cent_coords),
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
    img.arr = imgs_arr[0, :, :]
    img.units = (
        signal.units
        if isinstance(signal, (unyt_quantity, unyt_array))
        else None
    )

    # Apply the normalisation if needed
    if normalisation is not None:
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
    """
    Generate smoothed images for a particle emitter.

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

    # Ensure coordinates have been centred
    if not (cent_coords.min() < 0 and cent_coords.max() > 0):
        raise exceptions.InconsistentArguments(
            "Coordinates must be centered for imaging"
            f" (got min={cent_coords.min()} and max={cent_coords.max()})."
        )

    # Get the spatial units we'll work with
    spatial_units = imgs.resolution.units

    # Unpack the image properties we need
    fov = imgs.fov.to_value(spatial_units)
    res = imgs.resolution.to_value(spatial_units)

    # Shift the centred coordinates by half the FOV
    # (this is to ensure the image is centered on the emitter)
    cent_coords = cent_coords.to(spatial_units).value
    cent_coords[:, 0] += fov[0] / 2.0
    cent_coords[:, 1] += fov[1] / 2.0
    smoothing_lengths = smoothing_lengths.to_value(spatial_units)

    # Apply normalisation to original signal if needed
    if normalisations is not None:
        for ind, key in enumerate(labels):
            signals[ind, :] *= normalisations[key].value

    # Get the (Nimg, npix_x, npix_y) array of images
    imgs_arr = make_img(
        signals,
        ensure_array_c_compatible_double(smoothing_lengths),
        ensure_array_c_compatible_double(cent_coords),
        kernel,
        res,
        imgs.npix[0],
        imgs.npix[1],
        cent_coords.shape[0],
        kernel_threshold,
        kernel.size,
        signals.shape[0],
        nthreads,
    )

    # Store the image arrays on the image collection (this will
    # automatically convert them to Image objects)
    for ind, key in enumerate(labels):
        if isinstance(signals, (unyt_quantity, unyt_array)):
            imgs[key] = imgs_arr[ind, :, :] * signals.units
        else:
            imgs[key] = imgs_arr[ind, :, :]

    # Apply normalisation if needed
    if normalisations is not None:
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


def _generate_image_parametric_smoothed(
    img,
    density_grid,
    signal,
):
    """
    Generate a smoothed image for a parametric emitter.

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
    img.arr = density_grid[:, :] * signal
    img.units = signal.units

    return img


def _generate_images_parametric_smoothed(
    imgs,
    density_grid,
    signals,
):
    """
    Generate smoothed images for a parametric emitter.

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


def _generate_image_collection_generic(
    instrument,
    photometry,
    fov,
    img_type,
    kernel,
    kernel_threshold,
    nthreads,
    emitter,
):
    """
    Generate an image collection for a generic emitter.

    This function can be used to avoid repeating image generation code in
    wrappers elsewhere in the code. It'll produce an image collection based
    on the input photometry.

    Particle based imaging can either be hist or smoothed, while parametric
    imaging can only be smoothed.

    Args:
        instrument (Instrument)
            The instrument to create the images for.
        photometry (PhotometryCollection)
            The photometry to use for the images. This should be a a collection
            of 2D arrays of photometry with shape (Nfilters, Nparticles).
        fov (unyt_quantity/tuple, unyt_quantity)
            The width of the image.
        img_type (str)
            The type of image to create. Options are "hist" or "smoothed".
        kernel (str)
            The array describing the kernel. This is dervied from the
            kernel_functions module. (Only applicable to particle imaging)
        kernel_threshold (float)
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image. (Only
            applicable to particle imaging)
        nthreads (int)
            The number of threads to use when smoothing the image. This
            only applies to particle imaging.
        emitter (Stars/BlackHoles/BlackHole)
            The emitter object to create the images for.

    Returns:
        ImageCollection
            An image collection object containing the images.
    """
    # Avoid cyclic imports
    from synthesizer.imaging import ImageCollection
    from synthesizer.particle import Particles

    # Create the image collection
    imgs = ImageCollection(
        resolution=instrument.resolution,
        fov=fov,
    )

    # Make the image handling the different types of image creation
    # NOTE: Black holes are always a histogram, safer to just hack this here
    # since a user can set the "global" method as smoothed for a galaxy
    # with both stars and black holes.
    if (img_type == "hist" and isinstance(emitter, Particles)) or (
        getattr(emitter, "name", None) == "Black Holes"
    ):
        return _generate_images_particle_hist(
            imgs,
            coordinates=emitter.centered_coordinates,
            signals=photometry,
        )

    elif img_type == "hist":
        raise exceptions.InconsistentArguments(
            "Parametric images can only be made using the smoothed "
            "image type."
        )

    elif img_type == "smoothed" and isinstance(emitter, Particles):
        return _generate_images_particle_smoothed(
            imgs=imgs,
            signals=photometry.photometry,
            cent_coords=emitter.centered_coordinates,
            smoothing_lengths=emitter.smoothing_lengths,
            labels=photometry.filter_codes,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
        )

    elif img_type == "smoothed":
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


def _generate_ifu_particle_hist(
    ifu,
    sed,
    quantity,
    cent_coords,
    nthreads,
):
    """
    Generate a histogram IFU for a particle emitter.

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
    spectra = spectra.value.T

    # Ensure the spectra is 2D with a spectra per particle
    if spectra.ndim != 2:
        raise exceptions.InconsistentArguments(
            "Spectra must be a 2D array for an IFU image"
            f" (got {spectra.ndim})."
        )
    if spectra.shape[1] != cent_coords.shape[0]:
        raise exceptions.InconsistentArguments(
            "Spectra and coordinates must be the same size"
            f" for an IFU image (got {spectra.shape[0]} and "
            f"{cent_coords.shape[0]})."
        )

    # Get the spatial units we'll work with
    spatial_units = ifu.resolution.units

    # Get some IFU properties we'll need
    fov = ifu.fov.to_value(spatial_units)
    res = ifu.resolution.to_value(spatial_units)

    # Convert coordinates and smoothing lengths to the correct units and
    # strip them off
    cent_coords = cent_coords.to(spatial_units).value

    # Ensure coordinates have been centred
    if not (cent_coords.min() < 0 and cent_coords.max() > 0):
        raise exceptions.InconsistentArguments(
            "Coordinates must be centered for imaging"
            f" (got min={cent_coords.min()} and max={cent_coords.max()})."
        )

    # Prepare the inputs, we need to make sure we are passing C contiguous
    # arrays.
    spectra = np.ascontiguousarray(spectra, dtype=np.float64)
    cent_coords[:, 0] += fov[0] / 2
    cent_coords[:, 1] += fov[1] / 2
    smls = np.zeros(cent_coords.shape[0], dtype=np.float64)

    # Get the kernel
    # TODO: We should do away with this and write a histogram backend
    kernel = Kernel().get_kernel()

    ifu.arr = make_img(
        spectra,
        smls,
        ensure_array_c_compatible_double(cent_coords),
        kernel,
        res,
        ifu.npix[0],
        ifu.npix[1],
        cent_coords.shape[0],
        1,
        kernel.size,
        sed.nlam,
        nthreads,
    ).T

    return ifu


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
    """
    Generate a histogram IFU for a particle emitter.

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
            The array describing the kernel. This is dervied from the
            kernel_functions module.
        kernel_threshold (float):
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image.
        nthreads (int):
            The number of threads to use when smoothing the image.

    Returns:
        SpectralCube: The histogram image.
    """
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
    spectra = spectra.value.T

    # Ensure the spectra is 2D with a spectra per particle
    if spectra.ndim != 2:
        raise exceptions.InconsistentArguments(
            f"Spectra must be a 2D array for an IFU (got {spectra.ndim})."
        )
    if spectra.shape[1] != cent_coords.shape[0]:
        raise exceptions.InconsistentArguments(
            "Spectra and coordinates must be the same size"
            f" for an IFU (got {spectra.shape[0]} and "
            f"{cent_coords.shape[0]})."
        )

    # Get the spatial units we'll work with
    spatial_units = ifu.resolution.units

    # Get some IFU properties we'll need
    fov = ifu.fov.to_value(spatial_units)
    res = ifu.resolution.to_value(spatial_units)

    # Convert coordinates and smoothing lengths to the correct units and
    # strip them off
    cent_coords = cent_coords.to_value(spatial_units)

    # Ensure coordinates have been centred
    if not (cent_coords.min() < 0 and cent_coords.max() > 0):
        raise exceptions.InconsistentArguments(
            "Coordinates must be centered for imaging"
            f" (got min={cent_coords.min()} and max={cent_coords.max()})."
        )

    # Prepare the inputs, we need to make sure we are passing C contiguous
    # arrays.
    spectra = np.ascontiguousarray(spectra, dtype=np.float64)
    cent_coords[:, 0] += fov[0] / 2
    cent_coords[:, 1] += fov[1] / 2
    smls = ensure_array_c_compatible_double(
        smoothing_lengths.to_value(spatial_units)
    )

    # Generate the IFU
    ifu.arr = make_img(
        spectra,
        smls,
        ensure_array_c_compatible_double(cent_coords),
        kernel,
        res,
        ifu.npix[0],
        ifu.npix[1],
        cent_coords.shape[0],
        kernel_threshold,
        kernel.size,
        sed.nlam,
        nthreads,
    ).T

    return ifu


def _generate_ifu_parametric_smoothed(
    ifu,
    sed,
    quantity,
    density_grid,
):
    """
    Generate a smoothed IFU for a parametric emitter.

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
):
    """
    Generate a spectral cube.

    This function can be used to avoid repeating IFU generation code in
    wrappers elsewhere in the code. It'll produce a SpectralCube based
    on the input Sed.

    Particle based imaging can either be hist or smoothed, while parametric
    imaging can only be smoothed.

    Args:
        instrument (Instrument)
            The instrument to create the images for.
        fov (unyt_quantity/tuple, unyt_quantity)
            The width of the image.
        lam (unyt_array)
            The wavelength array of the spectra.
        img_type (str)
            The type of image to create. Options are "hist" or "smoothed".
        quantity (str)
            The quantity to use for the spectra. This can be any valid
            spectra quantity on an Sed object, e.g. 'lnu', 'fnu', 'luminosity',
            'flux', etc.
        per_particle (bool)
            Whether to create an image per particle or not.
        kernel (str)
            The array describing the kernel. This is dervied from the
            kernel_functions module. (Only applicable to particle imaging)
        kernel_threshold (float)
            The threshold for the kernel. Particles with a kernel value
            below this threshold are included in the image. (Only
            applicable to particle imaging)
        nthreads (int)
            The number of threads to use when smoothing the image. This
            only applies to particle imaging.
        label (str)
            The label of the photometry to use.
        emitter (Stars/BlackHoles/BlackHole)
            The emitter object to create the images for.

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

    # Create the IFU
    ifu = SpectralCube(
        resolution=instrument.resolution,
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
        return _generate_ifu_particle_hist(
            ifu,
            sed=sed,
            quantity=quantity,
            cent_coords=emitter.centered_coordinates,
            nthreads=nthreads,
        )

    elif img_type == "hist":
        raise exceptions.InconsistentArguments(
            "Parametric IFU can only be made using the smoothed img type."
        )

    elif img_type == "smoothed" and isinstance(emitter, Particles):
        return _generate_ifu_particle_smoothed(
            ifu,
            sed=sed,
            quantity=quantity,
            cent_coords=emitter.centered_coordinates,
            smoothing_lengths=emitter.smoothing_lengths,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
        )

    elif img_type == "smoothed":
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
