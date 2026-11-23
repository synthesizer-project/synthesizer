"""Specialised spectroscopic instrument.

This instrument is designed to hold the attributes required for one-
dimensional spectroscopy. It stores a wavelength grid together with optional
depth, signal-to-noise, and noise-map definitions.

The instrument holds the noise level (depth and snrs) and shape as a function
of wavelength.
"""

from collections import OrderedDict
from numbers import Real

import h5py
import numpy as np
from scipy import sparse
from scipy.interpolate import interp1d
from unyt import angstrom, unyt_array

from synthesizer import exceptions
from synthesizer.emissions import Sed
from synthesizer.instruments.instrument_base import (
    InstrumentBase,
    _hashable_state,
)
from synthesizer.instruments.utils import (
    read_instrument_attribute,
    write_instrument_attribute,
)
from synthesizer.synth_warnings import deprecated, warn
from synthesizer.units import (
    Quantity,
    accepts,
    deepcopy_with_shared_units,
    get_quantity_unit,
)
from synthesizer.utils.operation_timers import timed


class SpectroscopicInstrument(InstrumentBase):
    """Spectroscopic instrument class.

    A class containing the attributes and methods required for integrated
    spectroscopy. It holds the wavelength array defining the spectral coverage
    of the instrument together with the optional noise information required to
    generate noisy spectra. It does not include any spatially resolved state.

    Attributes:
        lam (unyt_array): The wavelength array defining the spectral coverage
            of the instrument.
        resolving_power (float or callable or array, optional): Resolving
            power (R = lambda / delta_lambda) of the instrument. May be a
            constant or a function of wavelength or an array as a function of
            wavelength (resolving_power_lam must be supplied). Constants and
            arrays are persisted by :meth:`to_hdf5`; a callable resolving power
            cannot be serialised to HDF5 and is silently dropped on save (it is
            not restored on load either, since there is nothing to restore it
            from).
        lsfs (array, optional): Normalised line spread function kernels
            with shape (nwave, nkernel). Each row corresponds to a
            wavelength in lam, and is the profile of a narrow line
            there, centred on the middle bin. Kernels must have an odd
            number of bins and sum to unity. They are in wavelength bins, so
            the spacing of lam should change slowly across a kernel; a
            wavelength array with a gap or a jump in spacing should be split
            into separate instruments. Gaussian kernels can be made from a
            resolving power with get_lsfs_from_resolving_power. To give an
            existing instrument its LSFs, or to change them, use set_lsfs:
            the LSFs are prepared when they are set, so later changes to
            the lsfs, lam or lsf_resample_factor attributes are not picked
            up.
        resolving_power_lam (unyt_array, optional): Wavelength array
            corresponding to tabulated resolving_power values. Required
            when resolving_power is provided as an array.
        lsf_resample_factor (int, optional): The LSF supersampling factor.
            The kernels are sampled, and applied, on wavelength bins this
            many times finer than the wavelength array.
        depth (unyt_quantity, optional): The depth of the instrument, the
            spectral flux density (fnu) or luminosity density (lnu) reached
            at S/N = snrs, at depth_lam (or over depth_window). Its units
            set which of the fluxes or luminosities get noise.
        snrs (float, optional): The signal-to-noise ratio at the depth.
        depth_lam (unyt_quantity, optional): The wavelength at which the
            noise equals depth / snrs.
        depth_window (unyt_array, optional): A wavelength window (lam_min,
            lam_max) over whose median the noise equals depth / snrs. With a
            noise template, exactly one of this or depth_lam is required;
            flat noise needs neither.
        noise_shape_lam (unyt_array, optional): Wavelengths of the noise
            template noise_shape_std(noise_shape_lam), strictly increasing.
        noise_shape_std (unyt_array, optional): The noise template, the
            1-sigma noise at noise_shape_lam (NaN in detector gaps). Only
            its shape is used; depth / snrs sets the noise level.
        template_on_detector (bool): Whether noise_shape_lam are the
            detector pixel centres. If so, noise is drawn at the detector
            pixels and interpolated, giving correlated wavelength elements;
            if not, each wavelength element gets independent noise.
        noise_maps (unyt_array, optional): A fixed noise realisation added
            to spectra as it is, with one value per wavelength in lam. An
            alternative to the inline calculation; its units set which of
            the fluxes or luminosities get noise.
    """

    lam = Quantity("wavelength")

    # Share units rather than deep copying unyt's whole unit registry
    __deepcopy__ = deepcopy_with_shared_units

    @accepts(
        lam=angstrom,
        noise_shape_lam=angstrom,
        depth_lam=angstrom,
        depth_window=angstrom,
    )
    @timed("SpectroscopicInstrument.__init__")
    def __init__(
        self,
        label,
        lam,
        resolving_power=None,
        lsfs=None,
        resolving_power_lam=None,
        lsf_resample_factor=1,
        depth=None,
        snrs=None,
        depth_lam=None,
        depth_window=None,
        noise_shape_lam=None,
        noise_shape_std=None,
        template_on_detector=True,
        noise_maps=None,
    ):
        """Initialise a spectroscopic instrument.

        Args:
            label (str): A label for the instrument.
            lam (unyt_array): The wavelength array defining the spectral
                coverage of the instrument.
            resolving_power (float or callable or array, optional): Resolving
                power (R = lambda / delta_lambda) of the instrument.
                May be a constant, array, or function of wavelength. Constants
                and arrays are persisted when the instrument is saved to HDF5.
            lsfs (array, optional): Normalised line spread function
                kernels with shape (nwave, nkernel), sampled on the
                instrument wavelength grid (or on bins lsf_resample_factor
                times finer). If a 1D array is supplied, every wavelength
                is assumed to have the same kernel. Gaussian kernels can
                be made from a resolving power with
                get_lsfs_from_resolving_power.
            resolving_power_lam (unyt_array, optional): Wavelength array
                corresponding to tabulated resolving_power values. Required
                when resolving_power is provided as an array.
            lsf_resample_factor (int, optional): The LSF supersampling
                factor. The kernels are sampled, and applied, on wavelength
                bins this many times finer than the wavelength array.
            depth (unyt_quantity, optional): The depth of the instrument,
                the spectral flux density (fnu) or luminosity density (lnu)
                reached at S/N = snrs, at depth_lam (or over depth_window).
                Its units set which of the fluxes or luminosities get noise.
            snrs (float, optional): The signal-to-noise ratio at the depth.
            depth_lam (unyt_quantity, optional): The wavelength at which the
                noise equals depth / snrs.
            depth_window (unyt_array, optional): A wavelength window
                (lam_min, lam_max) over whose median the noise equals
                depth / snrs. With a noise template, exactly one of this or
                depth_lam is required; flat noise needs neither.
            noise_shape_lam (unyt_array, optional): Wavelengths of the
                noise template, strictly increasing. Requires depth and snrs.
            noise_shape_std (unyt_array, optional): The noise template,
                the 1-sigma noise at noise_shape_lam (NaN in detector
                gaps), in units convertible to the noisy spectra. Only its
                shape is used. Defaults to flat noise if not supplied.
            template_on_detector (bool): Whether noise_shape_lam are the
                detector pixel centres. If True, allows correlated
                noise to be calculated. If False each wavelength element gets
                independent noise.
            noise_maps (unyt_array, optional): A fixed noise realisation
                added to spectra as it is, with one value per wavelength in
                lam. An alternative to the inline calculation.
        """
        super().__init__(label)
        self.lam = lam
        self.resolving_power = resolving_power
        self.resolving_power_lam = resolving_power_lam
        self.lsf_resample_factor = lsf_resample_factor
        self.depth = depth
        self.snrs = snrs
        self.noise_shape_lam = noise_shape_lam
        self.noise_shape_std = noise_shape_std
        self.template_on_detector = template_on_detector
        self.depth_lam = depth_lam
        self.depth_window = depth_window
        self.noise_maps = noise_maps

        # Set the LSFs (this also validates the instrument)
        self.set_lsfs(lsfs)

    @timed("SpectroscopicInstrument.set_lsfs")
    def set_lsfs(self, lsfs, lsf_resample_factor=None):
        """Set the LSF kernels of the instrument.

        This also sets the wavelength bins the LSFs are applied on,
        so use this to give an existing instrument its LSFs or to
        change them.

        Args:
            lsfs (array): Normalised line spread function kernels with
                shape (nwave, nkernel). If a 1D array is supplied, every
                wavelength is assumed to have the same kernel. None
                removes the LSFs.
            lsf_resample_factor (int, optional): The LSF supersampling
                factor the kernels are sampled at. If not given, the
                instrument keeps the factor it has.
        """
        # Store the kernels (a single LSF kernel is used at every
        # wavelength)
        self.lsfs = None if lsfs is None else np.asarray(lsfs).copy()
        if self.lsfs is not None and self.lsfs.ndim == 1:
            self.lsfs = np.tile(self.lsfs, (np.size(self.lam), 1))
        if lsf_resample_factor is not None:
            self.lsf_resample_factor = lsf_resample_factor

        # Anything made for a previous set of LSFs is removed
        self._lsf_bins = None
        self._lsf_matrices = OrderedDict()

        SpectroscopicInstrument._validate(self)

        # Make the bins the LSFs are applied on and the matrix that collects
        # the light in them. They only depend on the instrument, so we make
        # them once here.
        if self.lsfs is not None:
            self._lsf_bins = self._get_lsf_bins()

    @timed("SpectroscopicInstrument._validate")
    def _validate(self):
        """Validate the instrument attributes.

        Raises:
            MissingArgument: If any required attributes are missing.
        """
        # Ensure we actually have a wavelength array defining the instrument
        if self.lam is None:
            raise exceptions.MissingArgument(
                "SpectroscopicInstrument requires lam."
            )

        # Depths only make sense when paired with SNR definitions
        if self.depth is not None and self.snrs is None:
            raise exceptions.MissingArgument(
                "If you set a depth you must also set the SNRs"
            )

        # SNR definitions only make sense when paired with depths
        if self.snrs is not None and self.depth is None:
            raise exceptions.MissingArgument(
                "If you set a SNR you must also set the depth"
            )

        # Noise maps are an alternative noise definition to depth+SNR pairs
        if self.snrs is not None and self.noise_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as noise maps"
            )

        # A spectrum has no labels to key depths and SNRs by (imaging keys
        # them by filter), so they must be single values
        if isinstance(self.depth, dict) or isinstance(self.snrs, dict):
            raise exceptions.InconsistentArguments(
                "The depth and SNR must be single values, not dictionaries."
            )

        # The noise is added in physical units, so the depth and noise maps
        # need units
        if self.depth is not None and not isinstance(self.depth, unyt_array):
            raise exceptions.InconsistentArguments(
                "The depth must have units."
            )
        if self.noise_maps is not None and not isinstance(
            self.noise_maps, unyt_array
        ):
            raise exceptions.InconsistentArguments(
                "noise_maps must have units."
            )

        # The depth wavelength or window says where the depth applies
        if self.depth is None and (
            self.depth_lam is not None or self.depth_window is not None
        ):
            raise exceptions.MissingArgument(
                "depth_lam and depth_window need a depth and SNR"
            )
        if self.depth_lam is not None and self.depth_window is not None:
            raise exceptions.InconsistentArguments(
                "Give at most one of depth_lam or depth_window."
            )

        # The noise template only shapes depth+SNR noise
        if self.snrs is None and self.noise_shape_std is not None:
            raise exceptions.MissingArgument(
                "A noise template needs a depth and SNR to set the noise level"
            )

        # A noise template needs both its wavelengths and values
        if (self.noise_shape_lam is None) != (self.noise_shape_std is None):
            raise exceptions.MissingArgument(
                "Give both noise_shape_lam and noise_shape_std, or "
                "neither (flat noise)"
            )
        if self.noise_shape_std is not None:
            if np.shape(self.noise_shape_lam) != np.shape(
                self.noise_shape_std
            ):
                raise exceptions.InconsistentArguments(
                    "noise_shape_lam and noise_shape_std must have the "
                    "same shape."
                )
            if np.ndim(self.noise_shape_lam) != 1 or (
                np.size(self.noise_shape_lam) < 2
            ):
                raise exceptions.InconsistentArguments(
                    "The noise template must be one-dimensional with at "
                    "least two wavelengths."
                )
            if np.any(np.diff(self.noise_shape_lam.value) <= 0):
                raise exceptions.InconsistentArguments(
                    "noise_shape_lam must be strictly increasing."
                )

            # A noise template needs a reference at which to scale it
            if self.depth_lam is None and self.depth_window is None:
                raise exceptions.MissingArgument(
                    "Give one of depth_lam or depth_window with a noise "
                    "template."
                )

        if self.resolving_power is not None and not callable(
            self.resolving_power
        ):
            if isinstance(self.resolving_power, np.ndarray):
                if self.resolving_power_lam is None:
                    raise exceptions.MissingArgument(
                        "resolving_power_lam is required when resolving_power "
                        "is an array."
                    )

                if self.resolving_power.ndim != 1:
                    raise exceptions.InconsistentArguments(
                        "resolving_power must be a one-dimensional array."
                    )

                if np.ndim(self.resolving_power_lam) != 1:
                    raise exceptions.InconsistentArguments(
                        "resolving_power_lam must be a one-dimensional "
                        "wavelength array."
                    )

                if len(self.resolving_power) != len(self.resolving_power_lam):
                    raise exceptions.InconsistentArguments(
                        "resolving_power and resolving_power_lam must have "
                        "the same length."
                    )

                if (
                    not np.issubdtype(self.resolving_power.dtype, np.number)
                    or not np.all(np.isfinite(self.resolving_power))
                    or np.any(self.resolving_power <= 0)
                ):
                    raise exceptions.InconsistentArguments(
                        "resolving_power must contain only positive, finite "
                        "values."
                    )
            elif (
                isinstance(self.resolving_power, bool)
                or not isinstance(self.resolving_power, Real)
                or not np.isfinite(self.resolving_power)
                or self.resolving_power <= 0
            ):
                raise exceptions.InconsistentArguments(
                    "resolving_power must be a positive number, array, or "
                    "callable."
                )

        # The wavelength elements can only be split into a whole number of
        # bins
        if self.lsf_resample_factor < 1 or self.lsf_resample_factor % 1 != 0:
            raise exceptions.InconsistentArguments(
                "lsf_resample_factor must be an integer greater than or "
                "equal to 1."
            )

        # Validate the LSF kernels
        if self.lsfs is not None:
            if self.lsfs.ndim != 2:
                raise exceptions.InconsistentArguments(
                    "LSFs must be a 2D array with shape (nwave, nkernel)."
                )

            if self.lsfs.shape[0] != len(self.lam):
                raise exceptions.InconsistentArguments(
                    "The number of LSFs must match the instrument "
                    "wavelength grid."
                )

            if self.lsfs.shape[1] < 1 or self.lsfs.shape[1] % 2 == 0:
                raise exceptions.InconsistentArguments(
                    "LSF kernels must have an odd, non-zero length."
                )

            if not np.issubdtype(self.lsfs.dtype, np.floating):
                raise exceptions.InconsistentArguments(
                    "LSFs must have a floating-point dtype."
                )

            if not np.all(np.isfinite(self.lsfs)):
                raise exceptions.InconsistentArguments(
                    "LSFs must contain only finite values."
                )

            if np.any(self.lsfs < 0):
                raise exceptions.InconsistentArguments(
                    "LSF kernel weights must be non-negative."
                )

            if not np.allclose(self.lsfs.sum(axis=1), 1.0, rtol=0, atol=1e-6):
                raise exceptions.InconsistentArguments(
                    "Each LSF kernel must be normalised to unity."
                )

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "spectroscopic"

    @property
    def can_do_spectroscopy(self):
        """Return whether this instrument supports spectroscopy."""
        return True

    @property
    def can_do_noisy_spectroscopy(self):
        """Return whether this instrument supports noisy spectroscopy."""
        return (
            self.depth is not None and self.snrs is not None
        ) or self.noise_maps is not None

    @property
    def can_do_lsf_spectroscopy(self):
        """Return whether this instrument supports LSF spectroscopy."""
        return self.lsfs is not None

    @timed("SpectroscopicInstrument._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the spectroscopic comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        # Keep every resolving-power grid coordinate. The generic numeric
        # state helper intentionally uses a compact fingerprint, which can
        # make grids with different interior wavelengths compare equal.
        resolving_power_lam_state = (
            None
            if self.resolving_power_lam is None
            else tuple(self.resolving_power_lam.to_value("angstrom"))
        )

        return (
            _hashable_state(self._lam),
            _hashable_state(self.resolving_power),
            resolving_power_lam_state,
            _hashable_state(self.lsfs),
            _hashable_state(self.lsf_resample_factor),
            _hashable_state(self.depth),
            _hashable_state(self.snrs),
            _hashable_state(self.noise_shape_lam),
            _hashable_state(self.noise_shape_std),
            _hashable_state(self.template_on_detector),
            _hashable_state(self.depth_lam),
            _hashable_state(self.depth_window),
            _hashable_state(self.noise_maps),
        )

    @timed("SpectroscopicInstrument.apply_lam_array_lnu")
    def apply_lam_array_lnu(self, sed, nthreads=1):
        """Apply the instrument wavelength array in the rest frame.

        The instrument wavelengths are interpreted as rest frame wavelengths,
        so the luminosities are resampled directly onto them. Any fluxes are
        resampled onto the matching observer frame wavelengths,
        instrument.lam * (1 + redshift).

        Args:
            sed (Sed): Spectral energy distribution to resample.
            nthreads (int): Number of threads to use in the low-level
                resampling call. Currently unused.

        Returns:
            Sed: New SED resampled onto the instrument wavelength grid in the
                rest frame.
        """
        # Delegate the low-level rest frame resampling to the SED
        return sed.get_resampled_sed(new_lam=self.lam)

    @timed("SpectroscopicInstrument.apply_lam_array_fnu")
    def apply_lam_array_fnu(self, sed, nthreads=1):
        """Apply the instrument wavelength array in the observer frame.

        The instrument wavelengths are interpreted as observer frame
        wavelengths, so the fluxes are resampled directly onto them and the
        observer frame wavelengths of the result are exactly the instrument's.
        The luminosities are resampled onto the matching rest frame
        wavelengths, instrument.lam / (1 + redshift).

        Args:
            sed (Sed): Spectral energy distribution to observe. Its fluxes
                must already have been computed (e.g. with Sed.get_fnu).
            nthreads (int): Number of threads to use in the low-level
                resampling call. Currently unused.

        Returns:
            Sed: New SED resampled onto the instrument wavelength grid in the
                observer frame.
        """
        # Delegate the low-level observer frame resampling to the SED
        return sed.get_resampled_observed_sed(self.lam)

    @deprecated(
        "is deprecated and will be removed in version 1.4.0. "
        "It is an alias for instrument.apply_lam_array_fnu(...) (observer "
        "frame); use that, or instrument.apply_lam_array_lnu(...) for the "
        "rest frame, instead."
    )
    @timed("SpectroscopicInstrument.apply_lam_array")
    def apply_lam_array(self, sed, nthreads=1):
        """Apply the instrument wavelength array to an SED.

        Deprecated alias for apply_lam_array_fnu, which treats the instrument
        wavelengths as observer frame wavelengths.

        Args:
            sed (Sed): Spectral energy distribution to observe.
            nthreads (int): Number of threads to use in the low-level
                resampling call.

        Returns:
            Sed: New SED resampled onto the instrument wavelength grid.
        """
        # Defer to the observer frame version
        return self.apply_lam_array_fnu(sed, nthreads=nthreads)

    @timed("SpectroscopicInstrument.apply_lsf")
    def apply_lsf(self, sed, frame):
        """Apply the instrument's line spread function (LSF) to an SED.

        Each element of the instrument wavelength array collects the light
        around it with its own kernel, so the new SED is on the instrument
        wavelength grid (as from apply_lam_array_lnu or apply_lam_array_fnu).

        The SED can be on any wavelengths, including the instrument's own
        (e.g. spectroscopy made without the LSF). Where its wavelengths are
        finer than the instrument's, the LSF acts on the light before it is
        resampled, which places narrow lines more accurately. If it extends
        beyond the instrument wavelengths, the ends also collect the light
        from just outside.

        Args:
            sed (Sed): Spectral energy distribution to apply the LSF to.
            frame (str): The frame the instrument wavelengths are in,
                either "rest" or "observer" ("observed" is also accepted).
                For the observer frame the fluxes must already have been
                computed (e.g. with Sed.get_fnu).

        Returns:
            Sed: New SED on the instrument wavelength grid with the LSF
                applied.

        Raises:
            MissingArgument: If the instrument has no LSFs configured.
            InconsistentArguments: If frame is not recognised.
            MissingSpectraType: If frame is "observer" and the fluxes have
                not been computed.
        """
        # Ensure the instrument actually has LSFs to apply
        if self.lsfs is None or self._lsf_bins is None:
            raise exceptions.MissingArgument(
                "No LSFs are set on this Instrument. Provide lsfs when "
                "constructing the Instrument, or set them with set_lsfs."
            )

        # Ensure we know which frame the instrument wavelengths are in
        # ("observed" is accepted for the observer frame)
        if frame == "observed":
            frame = "observer"
        if frame not in ("rest", "observer"):
            raise exceptions.InconsistentArguments(
                f"Frame {frame} not recognised. Must be 'rest' or 'observer'."
            )

        # We can only work in the observer frame if we have fluxes, as the
        # SED has no observed wavelengths until they are computed
        if frame == "observer" and sed.fnu is None:
            raise exceptions.MissingSpectraType(
                "Observer frame spectroscopy needs fluxes, call get_fnu first."
            )

        # Get the matrix that applies the LSF to spectra on the SED's
        # wavelengths, in the frame the instrument wavelengths are in
        lsf_matrix = self._get_lsf_matrix(
            sed._obslam if frame == "observer" else sed._lam
        )

        # Apply it to the luminosities and any fluxes. The new SED holds
        # both whichever frame the instrument wavelengths are in (as from
        # apply_lam_array_lnu and apply_lam_array_fnu), and one matrix
        # serves both, as the rest and observer frame wavelengths differ by
        # a constant factor. NaNs are cleaned up where those methods clean
        # them up (we shouldn't get them, but a spectrum can hold them).
        lnu = unyt_array(
            _apply_lsf_matrix(lsf_matrix, sed._lnu),
            get_quantity_unit(sed, "lnu"),
            bypass_validation=True,
        )
        fnu = None
        if sed.fnu is not None:
            fnu = unyt_array(
                np.nan_to_num(
                    _apply_lsf_matrix(lsf_matrix, sed._fnu), copy=False
                ),
                get_quantity_unit(sed, "fnu"),
                bypass_validation=True,
            )

        # Build the new SED as apply_lam_array_lnu and apply_lam_array_fnu do
        if frame == "observer":
            new_sed = Sed.from_observed(
                self.lam, fnu, sed.redshift, rest_frame_lnu=lnu
            )
        else:
            new_sed = Sed(self.lam, np.nan_to_num(lnu, copy=False))
            if fnu is not None:
                new_sed.obslam = new_sed.lam * (1.0 + sed.redshift)
                new_sed.obsnu = new_sed.nu / (1.0 + sed.redshift)
                new_sed.fnu = fnu
                new_sed.redshift = sed.redshift
        new_sed.vel_shifted = sed.vel_shifted

        return new_sed

    @timed("SpectroscopicInstrument._get_lsf_matrix")
    def _get_lsf_matrix(self, lam):
        """Return the matrix that applies the LSF to spectra on lam.

        The spectra are first resampled onto the wavelength bins the LSF is
        applied on (see _get_lsf_bins), assuming the light at each of the
        spectra's own wavelengths to be spread evenly across its wavelength
        bin. The kernels are then applied to those bins.

        Args:
            lam (np.ndarray): The wavelengths of the spectra, in the same
                frame and units as the instrument wavelengths.

        Returns:
            scipy.sparse.csr_matrix: Matrix with shape (nwave, len(lam))
                taking spectra per unit frequency on lam to spectra per unit
                frequency on the instrument wavelengths.
        """
        # If we have already made the matrix for these wavelengths, reuse it
        key = lam.tobytes()
        lsf_matrix = self._lsf_matrices.get(key)
        if lsf_matrix is not None:
            return lsf_matrix

        # Get the bins the LSFs are applied on and the matrix that collects
        # the light in them
        bin_edges, bin_matrix = self._lsf_bins

        # Get the edges of the spectra's own wavelength bins, which lie
        # midway between their wavelengths
        edges = interp1d(np.arange(lam.size), lam, fill_value="extrapolate")(
            np.arange(lam.size + 1) - 0.5
        )

        # Cut the wavelength range covered by both sets of bins at every bin
        # edge of either set, so that each piece lies in one bin of each set
        cuts = np.union1d(edges, bin_edges)
        cuts = cuts[
            (cuts >= max(edges[0], bin_edges[0]))
            & (cuts <= min(edges[-1], bin_edges[-1]))
        ]

        # The spectra are per unit frequency, so the light a spectrum has in
        # each piece is its value there times the frequency width of the
        # piece.
        light_matrix = sparse.csr_matrix(
            (
                1 / cuts[:-1] - 1 / cuts[1:],
                (
                    np.searchsorted(bin_edges, cuts[:-1], side="right") - 1,
                    np.searchsorted(edges, cuts[:-1], side="right") - 1,
                ),
            ),
            shape=(bin_edges.size - 1, lam.size),
        )

        # Keep the matrix for later spectra on the same wavelengths, holding
        # on to no more than the eight made most recently
        lsf_matrix = bin_matrix @ light_matrix
        self._lsf_matrices[key] = lsf_matrix
        while len(self._lsf_matrices) > 8:
            self._lsf_matrices.popitem(last=False)

        return lsf_matrix

    @timed("SpectroscopicInstrument._get_lsf_bins")
    def _get_lsf_bins(self):
        """Return the wavelength bins the LSF is applied on and their matrix.

        Each element of the instrument wavelength array is split into
        lsf_resample_factor bins, and the bins continue past both ends
        of the array as far as the kernels reach. Each element collects the
        light in the bins around it with its own kernel, so the light from
        just outside the array is collected by the elements at its ends.

        The bins only depend on the instrument, so this is called once,
        when the LSFs are set.

        Returns:
            tuple: The edges of the bins (np.ndarray, in the units of the
                instrument wavelengths) and the matrix
                (scipy.sparse.csr_matrix) with shape (nwave, nbins) taking
                the light in each bin to a spectrum per unit frequency on
                the instrument wavelengths.
        """
        nwave, nkernel = self.lsfs.shape
        factor = int(self.lsf_resample_factor)

        # Get the number of wavelength elements the kernels reach past each
        # end of the wavelength array
        pad = int(np.ceil(nkernel // 2 / factor))

        # Get the bin edges in units of wavelength elements (element i spans
        # i - 0.5 to i + 0.5)
        edges = np.arange(-pad * factor, (nwave + pad) * factor + 1)
        edges = edges / factor - 0.5

        # Convert the edges to wavelengths (past each end of the wavelength
        # array, the spacing found there carries on)
        edges = interp1d(
            np.arange(nwave), self._lam, fill_value="extrapolate"
        )(edges)

        # Get the frequency width of each wavelength element, to convert the
        # light it collects to a spectrum per unit frequency (any constant
        # factors cancel)
        element_edges = edges[
            pad * factor : (pad + nwave) * factor + 1 : factor
        ]
        widths = 1 / element_edges[:-1] - 1 / element_edges[1:]

        # Each bin of a wavelength element collects the light in the bins
        # around it, weighted by the element's kernel.
        rows = np.repeat(np.arange(nwave), factor * nkernel)
        columns = pad * factor + np.arange(nwave * factor)[:, None]
        columns = columns - (np.arange(nkernel) - nkernel // 2)
        kernels = np.repeat(self.lsfs / widths[:, None], factor, axis=0)

        # Build the matrix (values with the same wavelength element and bin
        # are summed)
        bin_matrix = sparse.csr_matrix(
            (kernels.ravel(), (rows, columns.ravel())),
            shape=(nwave, edges.size - 1),
        )

        return edges, bin_matrix

    @timed("SpectroscopicInstrument._get_noise_std")
    def _get_noise_std(self, lam, units, noise_level):
        """Return the 1-sigma depth+SNR noise of each wavelength element.

        The noise is alpha * shape(lambda). Without a noise template the
        shape is flat (ones). With one, the template is interpolated onto
        lam: for "correlated" noise (template_on_detector) the shape is the
        noise the interpolated detector noise has,
        sigma_j^2 = [(1 - w)^2 sigma_lo^2 + w^2 sigma_hi^2]
        / [(1 - w)^2 + w^2], and for "independent" noise it is the template
        linearly interpolated onto lam. Wavelength elements next to template
        NaNs (detector gaps) are NaN. alpha scales the shape so the noise
        equals noise_level at depth_lam, or its median over depth_window
        does.

        Args:
            lam (unyt_array): The wavelengths of the spectra.
            units (unyt.Unit): The units of the spectra.
            noise_level (unyt_quantity): The target 1-sigma noise.

        Returns:
            tuple: The 1-sigma noise of each element (np.ndarray, in units),
                alpha (float), and the template indices and weights
                (i_lo, w) interpolating it onto lam (None without a
                template).

        Raises:
            InconsistentArguments: If the reference is outside the spectra,
                or alpha is not positive and finite.
        """
        spec_lam = lam.value

        # Without a template the noise is flat
        if self.noise_shape_std is None:
            shape = np.ones(lam.size)
            weights = None
            shape_ref = 1.0

        else:
            template_lam = self.noise_shape_lam.to(lam.units).value
            template_std = self.noise_shape_std.to(units).value

            # Warn if the spectra extend beyond the template
            outside = (spec_lam < template_lam[0]) | (
                spec_lam > template_lam[-1]
            )
            if np.any(outside):
                warn(
                    f"{self.label}: {outside.sum()} wavelength(s) fall "
                    "outside the noise template; using the nearest end of "
                    "the template there."
                )

            # Get the template pixels either side of each wavelength element
            # and the interpolation weights: element j is
            # (1 - w[j]) * template[i_lo[j]] + w[j] * template[i_lo[j] + 1],
            # and elements outside the template use its nearest end
            i_lo = np.clip(
                np.searchsorted(template_lam, spec_lam, side="right") - 1,
                0,
                template_lam.size - 2,
            )
            w = np.clip(
                (spec_lam - template_lam[i_lo])
                / (template_lam[i_lo + 1] - template_lam[i_lo]),
                0.0,
                1.0,
            )
            weights = (i_lo, w)

            # Interpolate the template, only including template values with
            # non-zero weight, so a NaN next to a gap doesn't spread to
            # elements that don't use it
            if self.template_on_detector:
                a, b = (1 - w) ** 2, w**2
                lo = np.where(a > 0, a * template_std[i_lo] ** 2, 0.0)
                hi = np.where(b > 0, b * template_std[i_lo + 1] ** 2, 0.0)
                shape = np.sqrt((lo + hi) / (a + b))
            else:
                lo = np.where(w < 1, (1 - w) * template_std[i_lo], 0.0)
                hi = np.where(w > 0, w * template_std[i_lo + 1], 0.0)
                shape = lo + hi

            # Get the shape at the reference wavelength
            if self.depth_window is not None:
                lam_min, lam_max = self.depth_window.to(lam.units).value
                in_window = (spec_lam >= lam_min) & (spec_lam <= lam_max)
                if not np.any(in_window):
                    raise exceptions.InconsistentArguments(
                        f"{self.label}: no wavelengths inside depth_window "
                        f"{self.depth_window}."
                    )
                shape_ref = np.nanmedian(shape[in_window])
            elif self.depth_lam is not None:
                lam_ref = self.depth_lam.to(lam.units).value
                if not spec_lam[0] <= lam_ref <= spec_lam[-1]:
                    raise exceptions.InconsistentArguments(
                        f"{self.label}: depth_lam {self.depth_lam} is "
                        "outside the spectra."
                    )
                shape_ref = np.interp(lam_ref, spec_lam, shape)

        # Scale the shape to the noise level
        alpha = float(noise_level.to(units).value / shape_ref)
        if not (np.isfinite(alpha) and alpha > 0):
            raise exceptions.InconsistentArguments(
                f"{self.label}: noise scale factor is {alpha}; check the "
                "depth, SNR, noise template and reference."
            )
        return alpha * shape, alpha, weights

    @timed("SpectroscopicInstrument._generate_noise")
    def _generate_noise(
        self,
        arr,
        lam,
        units,
        rng,
        noise_level=None,
        simspin_snr=None,
    ):
        """Generate noise for some spectra, and its weight map.

        The noise machinery here works on arrays whose last axis is wavelength,
        so it serves single spectra and spectral cubes. Two noise models are
        available:

        - Source independent: sigma(lambda) = alpha * shape(lambda). The shape
        is an optional noise template (defaults to flat if not supplied), and
        alpha scales it so sigma = depth / snr at a reference wavelength (or
        the median over a reference window). If the template is sampled on the
        detector pixels, noise is drawn there and linearly interpolated onto
        the spectrum's wavelengths ("correlated"), so wavelength elements
        sharing detector pixels are correlated. Otherwise each wavelength
        element gets independent noise ("independent").
        - SimSpin: sigma(lambda) = |F(lambda)| sqrt(F_med) / (snr sqrt(F)),
        where F is a spectrum summed over wavelength and F_med the median of F
        over spectra with F > 0. The median spectrum then has S/N = snr at
        every wavelength.

        As for imaging, the noise comes from SimSpin if simspin_snr is given,
        otherwise from the instrument's noise maps (a fixed realisation added
        as it is), or its depth and SNR. Any leading axes of arr hold
        independent spectra (e.g. the pixels of a spectral cube), each
        getting its own realisation.

        Args:
            arr (np.ndarray): The spectra, with shape (..., nlam), in units.
            lam (unyt_array): The wavelengths of the spectra.
            units (unyt.Unit): The units of the spectra.
            rng (np.random.Generator): The random number generator.
            noise_level (unyt_quantity, optional): The target 1-sigma
                depth+SNR noise, depth / snrs (per pixel for spectral
                cubes).
            simspin_snr (float, optional): The S/N of the median spectrum
                for SimSpin noise.

        Returns:
            tuple: The noise (np.ndarray, in units, shaped like arr, or like
                the noise maps if they are used) and the weight map
                (np.ndarray shaped like arr).

        Raises:
            MissingArgument: If there is no noise model to apply.
            InconsistentArguments: If simspin_snr is not positive, or the
                noise maps don't match the spectra.
        """
        # SimSpin noise needs only its S/N
        if simspin_snr is not None:
            if not simspin_snr > 0:
                raise exceptions.InconsistentArguments(
                    "simspin_snr must be positive."
                )
        # Otherwise the instrument must define noise maps or depth+SNR noise
        elif self.noise_maps is None and (
            self.depth is None or self.snrs is None
        ):
            raise exceptions.MissingArgument(
                "The instrument has no spectroscopic noise configuration. "
                "Provide depth and snrs or noise_maps, or pass simspin_snr."
            )

        if simspin_snr is not None:
            # Sum each spectrum over wavelength and take the median of the
            # spectra with flux
            flux = np.nansum(arr, axis=-1)
            has_flux = flux > 0
            if not np.any(has_flux):
                raise exceptions.InconsistentArguments(
                    f"{self.label}: SimSpin noise needs spectra with positive "
                    "flux."
                )
            median_flux = np.median(flux[has_flux])

            # Scale so the median spectrum has S/N = simspin_snr at every
            # wavelength (spectra without flux get no noise)
            scale = np.zeros(flux.shape)
            scale[has_flux] = np.sqrt(median_flux) / (
                simspin_snr * np.sqrt(flux[has_flux])
            )
            err = np.abs(arr) * scale[..., None]
            noise = rng.standard_normal(arr.shape) * err

        elif self.noise_maps is not None:
            # The fixed noise must match the trailing axes of the spectra:
            # (nlam,) for spectra, or the whole cube for a spectral cube
            ndim = self.noise_maps.ndim
            if ndim > arr.ndim or self.noise_maps.shape != arr.shape[-ndim:]:
                raise exceptions.InconsistentArguments(
                    f"{self.label}: noise_maps has shape "
                    f"{self.noise_maps.shape}, which doesn't match the "
                    f"spectra's shape {arr.shape}."
                )
            noise = self.noise_maps.to(units).value

            # As for imaging, the weight map comes from the noise's std
            err = np.full(arr.shape, np.std(noise))

        else:
            # Get the 1-sigma noise of each wavelength element
            err, alpha, weights = self._get_noise_std(lam, units, noise_level)
            err = np.broadcast_to(err, arr.shape).copy()

            if weights is None or not self.template_on_detector:
                noise = rng.standard_normal(arr.shape) * err

            else:
                # Interpolation doesn't average detector pixels down, so
                # wavelength elements coarser than the detector pixels would
                # get too much noise
                template_lam = self.noise_shape_lam.to(lam.units).value
                ratio = np.interp(
                    lam.value, template_lam, np.gradient(template_lam)
                ) / np.gradient(lam.value)
                coarse = ratio < 0.9
                if np.any(coarse):
                    warn(
                        f"{self.label}: {coarse.sum()} of {lam.size} "
                        "wavelength elements are coarser than the detector "
                        f"pixels (ratio down to {ratio.min():.2f}, between "
                        f"{lam[coarse].min()} and {lam[coarse].max()}); "
                        "correlated noise does not average down there."
                    )

                # Draw independent noise at the detector pixels for each
                # spectrum, in (detector pixel, spectrum) order
                template_std = self.noise_shape_std.to(units).value
                nspec = int(np.prod(arr.shape[:-1]))
                detector_noise = (
                    rng.standard_normal((template_std.size, nspec)).T
                    * template_std
                )

                # Interpolate it onto the wavelength elements, only including
                # detector pixels with non-zero weight (to avoid 0 * NaN next
                # to gaps), and renormalise so each element's sigma is the
                # detector value. Elements sharing detector pixels are then
                # correlated.
                i_lo, w = weights
                lo = np.where(w < 1, (1 - w) * detector_noise[:, i_lo], 0.0)
                hi = np.where(w > 0, w * detector_noise[:, i_lo + 1], 0.0)
                noise = alpha * (lo + hi) / np.sqrt((1 - w) ** 2 + w**2)
                noise = noise.reshape(arr.shape)

        err[np.isnan(arr)] = np.nan

        # Weight map is 1 / err**2, with 0 where err is NaN and inf where
        # it is 0 (e.g. spectra without flux under SimSpin noise).
        with np.errstate(divide="ignore", invalid="ignore"):
            weight_map = np.where(np.isnan(err), 0.0, 1.0 / err**2)

        return noise, weight_map

    @timed("SpectroscopicInstrument.apply_noise")
    def apply_noise(self, sed, rng_seed=None, simspin_snr=None):
        """Apply spectroscopic noise to an SED.

        As for imaging, the noise comes from SimSpin if simspin_snr is given,
        otherwise from the instrument's noise maps (added as they are), or
        its depth and SNR. The noise is added to the observer frame fluxes
        if the depth (or noise maps) is a flux density, or to the rest frame
        luminosities if it is a luminosity density; SimSpin noise is added
        to the fluxes if the SED has them. The other spectrum is carried
        over without noise.

        Args:
            sed (Sed): The SED to add noise to, e.g. from apply_lam_array_fnu
                or apply_lam_array_lnu. For noise on the fluxes they must
                already have been computed (e.g. with Sed.get_fnu).
            rng_seed (int, optional): Seed for the noise realisation. None
                gives a fresh realisation.
            simspin_snr (float, optional): The S/N of the median spectrum
                for SimSpin noise.

        Returns:
            Sed: New SED with noise added, with noise_arr (the noise added,
                unyt_array) and weight_map (the inverse variance of each
                element; 0 where it is undefined) attached.

        Raises:
            MissingSpectraType: If the noise is for the fluxes and they have
                not been computed.
            MissingArgument: If there is no noise model to apply.
            InconsistentArguments: If the noise units are neither fnu nor
                lnu, or the noise maps don't match the wavelengths.
        """
        # Choose the spectrum to add noise to: the one sharing the units of
        # the noise maps or depth, or for SimSpin noise the fluxes if there
        # are any (without any noise model, _generate_noise raises below)
        noise_def = (
            self.noise_maps if self.noise_maps is not None else self.depth
        )
        if simspin_snr is not None or noise_def is None:
            quantity = "lnu" if sed.fnu is None else "fnu"
        else:
            for quantity in ("fnu", "lnu"):
                quantity_units = get_quantity_unit(sed, quantity)
                if noise_def.units.dimensions == quantity_units.dimensions:
                    break
            else:
                raise exceptions.InconsistentArguments(
                    f"{self.label}: the noise units {noise_def.units} are "
                    "neither a flux density (fnu) nor a luminosity density "
                    "(lnu) per unit frequency."
                )
        if quantity == "fnu" and sed.fnu is None:
            raise exceptions.MissingSpectraType(
                "The noise is a flux density, so noisy spectroscopy needs "
                "fluxes, call get_fnu first."
            )
        units = get_quantity_unit(sed, quantity)
        spectra = getattr(sed, f"_{quantity}")
        lam = sed.obslam if quantity == "fnu" else sed.lam

        # Generate the noise, with the noise level the depth defines for a
        # point source
        noise_level = None if self.depth is None else self.depth / self.snrs
        rng = np.random.default_rng(rng_seed)
        noise, weight_map = self._generate_noise(
            spectra,
            lam,
            units,
            rng,
            noise_level=noise_level,
            simspin_snr=simspin_snr,
        )

        # Build the new SED as apply_lsf does in each frame
        if quantity == "fnu":
            new_sed = Sed.from_observed(
                sed.obslam,
                unyt_array(sed._fnu + noise, units),
                sed.redshift,
                rest_frame_lnu=sed.lnu,
            )
        else:
            new_sed = Sed(sed.lam, unyt_array(sed._lnu + noise, units))
            if sed.fnu is not None:
                new_sed.obslam = new_sed.lam * (1.0 + sed.redshift)
                new_sed.obsnu = new_sed.nu / (1.0 + sed.redshift)
                new_sed.fnu = sed.fnu
                new_sed.redshift = sed.redshift
        new_sed.vel_shifted = sed.vel_shifted

        # Store the noise and the weight map on the new SED
        new_sed.noise_arr = unyt_array(noise, units)
        new_sed.weight_map = weight_map

        return new_sed

    @timed("SpectroscopicInstrument.to_hdf5")
    def to_hdf5(self, group):
        """Write the spectroscopic instrument to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised.
        """
        group.attrs["label"] = self.label
        group.attrs["instrument_type"] = self.instrument_type

        # A callable resolving_power(wavelength) is not serialisable and is
        # intentionally dropped here. NumPy scalars count as constants:
        # np.float32 and np.int64 are not instances of float or int, so
        # testing those alone would silently discard a resolving power that
        # came out of an array.
        if isinstance(self.resolving_power, np.ndarray):
            ds = group.create_dataset(
                "ResolvingPower", data=self.resolving_power, dtype=float
            )
            ds.attrs["units"] = "dimensionless"

            write_instrument_attribute(
                group, "ResolvingPowerWavelength", self.resolving_power_lam
            )
        elif isinstance(self.resolving_power, Real) and not isinstance(
            self.resolving_power, bool
        ):
            group.attrs["resolving_power"] = float(self.resolving_power)

        ds = group.create_dataset(
            "Wavelength", data=self.lam.ndview, dtype=float
        )
        ds.attrs["units"] = str(self.lam.units)

        if self.lsfs is not None:
            ds = group.create_dataset("LSFs", data=self.lsfs)
            ds.attrs["units"] = "dimensionless"

        ds = group.create_dataset(
            "LSFResampleFactor", data=self.lsf_resample_factor, dtype=int
        )
        ds.attrs["units"] = "dimensionless"

        write_instrument_attribute(group, "Depth", self.depth)
        write_instrument_attribute(group, "DepthWavelength", self.depth_lam)
        write_instrument_attribute(group, "DepthWindow", self.depth_window)
        write_instrument_attribute(group, "SNRs", self.snrs)
        write_instrument_attribute(group, "NoiseMaps", self.noise_maps)

        # Noise template
        group.attrs["template_on_detector"] = bool(self.template_on_detector)
        write_instrument_attribute(
            group, "NoiseShapeWavelength", self.noise_shape_lam
        )
        write_instrument_attribute(
            group, "NoiseShapeStd", self.noise_shape_std
        )

    @classmethod
    @timed("SpectroscopicInstrument.load")
    def load(cls, filepath, **kwargs):
        """Load a spectroscopic instrument from an HDF5 file.

        Args:
            filepath (str or PathLike): Path to the HDF5 file.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            SpectroscopicInstrument: The loaded instrument.
        """
        with h5py.File(filepath, "r") as hdf:
            return cls._from_hdf5(hdf, **kwargs)

    @classmethod
    @timed("SpectroscopicInstrument._from_hdf5")
    def _from_hdf5(cls, group, **kwargs):
        """Load a spectroscopic instrument from an HDF5 group.

        Args:
            group (h5py.Group): Group containing the serialised instrument.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            SpectroscopicInstrument: The loaded instrument.
        """
        lam = unyt_array(
            group["Wavelength"][...], group["Wavelength"].attrs["units"]
        )

        depth = read_instrument_attribute(group, "Depth")
        snrs = read_instrument_attribute(group, "SNRs")

        noise_maps = read_instrument_attribute(group, "NoiseMaps")

        if "LSFs" in group:
            lsfs = group["LSFs"][...]
        else:
            lsfs = None

        if "LSFResampleFactor" in group:
            lsf_resample_factor = int(group["LSFResampleFactor"][...])
        else:
            lsf_resample_factor = 1

        if "ResolvingPower" in group:
            resolving_power = group["ResolvingPower"][...]
            resolving_power_lam = read_instrument_attribute(
                group, "ResolvingPowerWavelength"
            )
        else:
            resolving_power = group.attrs.get("resolving_power")
            resolving_power_lam = None

        payload = {
            "label": group.attrs["label"],
            "lam": lam,
            "depth": depth,
            "snrs": snrs,
            "noise_maps": noise_maps,
            "resolving_power": resolving_power,
            "resolving_power_lam": resolving_power_lam,
            "lsfs": lsfs,
            "lsf_resample_factor": lsf_resample_factor,
            "noise_shape_lam": read_instrument_attribute(
                group, "NoiseShapeWavelength"
            ),
            "noise_shape_std": read_instrument_attribute(
                group, "NoiseShapeStd"
            ),
            "template_on_detector": bool(
                group.attrs.get("template_on_detector", True)
            ),
            "depth_lam": read_instrument_attribute(group, "DepthWavelength"),
            "depth_window": read_instrument_attribute(group, "DepthWindow"),
        }
        payload.update(kwargs)

        return cls(
            label=payload["label"],
            lam=payload["lam"],
            depth=payload["depth"],
            snrs=payload["snrs"],
            noise_maps=payload["noise_maps"],
            resolving_power=payload["resolving_power"],
            resolving_power_lam=payload["resolving_power_lam"],
            lsfs=payload["lsfs"],
            lsf_resample_factor=payload["lsf_resample_factor"],
            noise_shape_lam=payload["noise_shape_lam"],
            noise_shape_std=payload["noise_shape_std"],
            template_on_detector=payload["template_on_detector"],
            depth_lam=payload["depth_lam"],
            depth_window=payload["depth_window"],
        )


def _apply_lsf_matrix(lsf_matrix, spectra):
    """Apply an LSF matrix along the wavelength axis of some spectra.

    Args:
        lsf_matrix (scipy.sparse.csr_matrix): The matrix, with shape
            (nwave, nlam).
        spectra (np.ndarray): The spectra, with shape (..., nlam).

    Returns:
        np.ndarray: The spectra on the instrument wavelengths, with shape
            (..., nwave) and the precision of the input.
    """
    # Any leading axes hold independent spectra
    flat_spectra = spectra.reshape(-1, spectra.shape[-1])
    new_spectra = np.empty(
        (flat_spectra.shape[0], lsf_matrix.shape[0]), dtype=spectra.dtype
    )

    # Apply the matrix to each spectrum
    for ind, spectrum in enumerate(flat_spectra):
        new_spectra[ind] = lsf_matrix @ spectrum

    return new_spectra.reshape(spectra.shape[:-1] + (lsf_matrix.shape[0],))
