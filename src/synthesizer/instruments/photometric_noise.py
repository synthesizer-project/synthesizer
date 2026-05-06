"""Photometric noise generation for synthesizer imaging.

This module contains tools for modelling and generating photometric noise
for imaging. Correlated noise is represented by the
``CorrelatedNoiseModel`` class, which stores a source noise template and
derives reusable statistical quantities from it. These cached quantities are
then used to generate new noise realisations for images of arbitrary shape.
"""

import numpy as np
from unyt import unyt_array

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
            Tuple ``(Ny, Nx)`` representing the shape of the correlation
            function array.

    Returns:
        np.ndarray:
            A 2D array with the correction factors.
    """
    ny, nx = cf_shape

    # Work in pixel offsets measured from the CF origin. ``fftfreq`` gives the
    # wrapped DFT indexing convention, which is the same convention used by the
    # unrolled correlation functions returned by the FFT pipeline below.
    dx_coords = np.fft.fftfreq(nx) * float(nx)
    dy_coords = np.fft.fftfreq(ny) * float(ny)
    deltax, deltay = np.meshgrid(dx_coords, dy_coords)

    # At large separations fewer pixel pairs contribute to the empirical CF.
    # This factor approximately undoes that dilution introduced by the DFT's
    # implicit periodic wrapping of the source image.
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
            Tuple ``(Ny, Nx)`` of the output real-space noise field.
        rootps (np.ndarray):
            The half-complex 2D array ``(Ny, Nx//2 + 1)`` representing the
            square root of the Power Spectrum from ``rfft2``.

    Returns:
        np.ndarray:
            A 2D array representing the generated correlated noise field.
    """
    ny, nx = shape
    sigma_val_for_gvec_parts = np.sqrt(0.5 * ny * nx)

    # Draw a complex Gaussian field in Fourier space. Once it is multiplied by
    # the square root power spectrum, the inverse transform will have the
    # desired second-order statistics.
    gvec_real = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec_imag = rng.normal(scale=sigma_val_for_gvec_parts, size=rootps.shape)
    gvec = gvec_real + 1j * gvec_imag

    rt2 = np.sqrt(2.0)

    # The purely real modes of the half-complex FFT representation must remain
    # real so that the inverse transform returns a real-valued image.
    gvec[0, 0] = rt2 * gvec[0, 0].real

    # Treat the Nyquist frequencies in the same way for even-sized axes.
    if ny % 2 == 0:
        gvec[ny // 2, 0] = rt2 * gvec[ny // 2, 0].real
    if nx % 2 == 0:
        gvec[0, nx // 2] = rt2 * gvec[0, nx // 2].real
    if ny % 2 == 0 and nx % 2 == 0:
        gvec[ny // 2, nx // 2] = rt2 * gvec[ny // 2, nx // 2].real

    # Fill the modes on the kx=0 and kx=Nyquist axes so the spectrum obeys the
    # Hermitian symmetry required for a real inverse FFT.
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


class CorrelatedNoiseModel:
    """Model correlated image noise from a source template.

    This class encapsulates the correlated-noise machinery used by imaging.
    A source noise template (i.e. existing noise field) is supplied at
    construction time and is used to estimate a correlation function (CF).
    The expensive CF estimation is cached on the model so that many target
    images can draw fresh correlated noise realisations from the same
    underlying template.

    Attributes:
        source_noise_map (np.ndarray or unyt_array):
            The observed/template noise map used to characterise the noise.
        units (unyt.Unit or None):
            Units carried by the source template, if present.
        _cf_cache (dict):
            Cache of estimated correlation functions keyed by
            ``(subtract_mean, correct_periodicity)``.
    """

    def __init__(self, source_noise_map):
        """Initialise the correlated noise model.

        Args:
            source_noise_map (np.ndarray or unyt_array):
                A 2D observed noise map used to model the spatial noise
                correlations.

        Raises:
            ValueError:
                If the source noise map is not two dimensional.
        """
        # We need an image-like array...
        if source_noise_map.ndim != 2:
            raise ValueError("source_noise_map must be a 2D array.")

        # Attach the source noise map
        self.source_noise_map = source_noise_map

        # Extract and store units if we have them
        self.units = (
            source_noise_map.units
            if isinstance(source_noise_map, unyt_array)
            else None
        )
        self._cf_cache = {}

    @property
    def source_array(self) -> np.ndarray:
        """Return the source noise template without units.

        Returns:
            np.ndarray:
                The source noise template as a plain NumPy array.
        """
        if isinstance(self.source_noise_map, unyt_array):
            return self.source_noise_map.value
        return self.source_noise_map

    def estimate_correlation_function(
        self,
        subtract_mean: bool = False,
        correct_periodicity: bool = True,
    ) -> np.ndarray:
        """Estimate and cache the source correlation function.

        Computes an unrolled correlation function (DFT convention, origin at
        ``[0, 0]``) from the power spectrum of the source array. The result is
        cached for the requested modelling options so later calls can reuse it
        without recomputing the FFT.

        Args:
            subtract_mean (bool):
                If True the DC component of the power spectrum is zeroed,
                removing the mean offset from the noise model.
            correct_periodicity (bool):
                If True a correction factor is applied to compensate for the
                periodicity assumption of the DFT.

        Returns:
            np.ndarray:
                A 2D array of the same shape as the source template holding the
                unrolled CF with the origin at ``[0, 0]``.
        """
        # The modelling switches affect the derived CF, so they form the cache
        # key for reusing work across many target images.
        cache_key = (subtract_mean, correct_periodicity)
        if cache_key not in self._cf_cache:
            source_arr = self.source_array
            source_shape = source_arr.shape

            # Estimate the power spectrum of the template and normalise by the
            # number of pixels so the inverse transform returns the empirical
            # correlation function on the source grid.
            ft_array = np.fft.rfft2(source_arr)
            ps_array = np.abs(ft_array) ** 2
            ps_array /= np.prod(source_shape)

            # Zeroing the DC mode removes any constant offset from the model so
            # only spatially varying noise is propagated into later draws.
            if subtract_mean:
                ps_array[0, 0] = 0.0

            cf_array = np.fft.irfft2(ps_array, s=source_shape)

            # Apply the finite-image correction only when requested, since some
            # callers may want the raw DFT-periodic estimate.
            if correct_periodicity:
                correction = _cf_periodicity_dilution_correction_standalone(
                    source_shape
                )
                cf_array *= correction

            self._cf_cache[cache_key] = cf_array

        return self._cf_cache[cache_key]

    def generate_noise_array(
        self,
        target_shape: tuple,
        subtract_mean: bool = False,
        correct_periodicity: bool = True,
        rng_seed: int = None,
    ) -> np.ndarray:
        """Generate a correlated noise realisation for a target shape.

        A cached source CF is first fetched or computed, then resampled to the
        target dimensions before a fresh random realisation is drawn. The same
        source CF can therefore be used to generate many independent target
        noise fields.

        Args:
            target_shape (tuple of int):
                The ``(ny, nx)`` pixel dimensions of the desired noise field.
            subtract_mean (bool):
                Passed through to :meth:`estimate_correlation_function`.
            correct_periodicity (bool):
                Passed through to :meth:`estimate_correlation_function`.
            rng_seed (int, optional):
                Seed for the random number generator. Pass the same seed to
                reproduce an identical noise realisation.

        Returns:
            np.ndarray or unyt_array:
                A 2D array of shape ``target_shape`` containing the generated
                correlated noise field. Units are attached if the source
                template carried them.

        Raises:
            InconsistentArguments:
                If the source CF is too small to contribute any pixels to the
                target CF grid.
        """
        # Get the source CF
        cf_source = self.estimate_correlation_function(
            subtract_mean=subtract_mean,
            correct_periodicity=correct_periodicity,
        )

        # Seed the random number generator for reproducibility
        rng = np.random.default_rng(rng_seed)
        source_shape = cf_source.shape

        # Warn if the source CF is smaller than the target shape
        if (
            source_shape[0] < target_shape[0]
            or source_shape[1] < target_shape[1]
        ):
            warn(
                "Source noise map is smaller than the target image, which may"
                " result in truncation of the noise model."
            )

        # Roll the source CF so that zero lag sits at the array centre. This
        # makes it straightforward to copy the central overlap region onto a
        # target grid with a different shape.
        cf_source_rolled = np.roll(
            cf_source,
            shift=(source_shape[0] // 2, source_shape[1] // 2),
            axis=(0, 1),
        )

        cf_target_rolled = np.zeros(target_shape, dtype=float)

        # Compute the overlap of the centred source CF with the centred target
        # grid. This preserves the small-lag structure that dominates the noise
        # correlations while cropping or zero-padding the outer lags as needed.
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

        # Copy the overlapping central block from the source CF onto the target
        # grid, leaving any non-overlapping regions at zero correlation.
        cf_target_rolled[
            y_start_trg : y_start_trg + dy, x_start_trg : x_start_trg + dx
        ] = cf_source_rolled[
            y_start_src : y_start_src + dy, x_start_src : x_start_src + dx
        ]

        # Undo the centring roll so the target CF is back in the unrolled DFT
        # convention expected by ``rfft2``.
        cf_on_target_grid_unrolled = np.roll(
            cf_target_rolled,
            shift=(-(target_shape[0] // 2), -(target_shape[1] // 2)),
            axis=(0, 1),
        )

        # Transform back to Fourier space and draw a fresh random field with
        # this target-grid power spectrum.
        ps_target_fft = np.fft.rfft2(cf_on_target_grid_unrolled)
        rootps_target = np.sqrt(np.abs(ps_target_fft))
        noise_arr = _generate_noise_from_rootps_standalone(
            rng, target_shape, rootps_target
        )

        # Reattach template units so downstream image-noise application can use
        # the standard unit-aware code paths.
        if self.units is not None:
            return unyt_array(noise_arr, self.units)

        return noise_arr

    def apply_noise(
        self,
        image,
        correct_periodicity: bool = True,
        rng_seed: int = None,
        inplace: bool = False,
    ):
        """Apply correlated noise to an Image.

        The correlated-noise realisation is generated for the image's pixel
        shape and then added via the image's existing noise-array machinery.

        Args:
            image (Image):
                The image to which the correlated noise should be added.
            correct_periodicity (bool):
                If True the DFT periodicity correction is applied.
            rng_seed (int, optional):
                Seed for the random number generator.
            inplace (bool):
                If True, update the input image in place and return it.
                Otherwise return a new image. Default is False.

        Returns:
            Image:
                The noisy image.

        Raises:
            InconsistentArguments:
                If the image has units but the noise model is dimensionless.
        """
        # Draw one correlated-noise realisation matched to this image's pixel
        # grid from the cached source statistics.
        noise_arr = self.generate_noise_array(
            image.arr.shape,
            subtract_mean=True,
            correct_periodicity=correct_periodicity,
            rng_seed=rng_seed,
        )

        # Keep the same unit contract as ``Image.apply_noise_array``.
        if self.units is None and image.units is not None:
            raise exceptions.InconsistentArguments(
                "The image has units but the noise map on the instrument is "
                "dimensionless. Provide a noise map with units compatible "
                f"with the image units ({image.units})."
            )

        # Apply the noise to the image using the existing machinery for
        # handling explicit noise arrays.
        # TODO: This too needs to move instrument side eventually.
        noisy_image = image.apply_noise_array(noise_arr)

        # If not inplace, return the new image with the noise applied
        if not inplace:
            return noisy_image

        # Otherwise, we need to update the existing image in place
        image.arr = noisy_image.arr
        image.units = noisy_image.units
        image.noise_arr = noisy_image.noise_arr
        image.weight_map = noisy_image.weight_map

        # And delete the temporary noisy image
        del noisy_image

        return image
