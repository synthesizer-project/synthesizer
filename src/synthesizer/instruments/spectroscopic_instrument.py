"""Specialised spectroscopic instrument.

This instrument is designed to hold the attributes required for one-
dimensional spectroscopy. It stores a wavelength grid together with optional
depth, signal-to-noise, and noise-map definitions.
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
from synthesizer.synth_warnings import deprecated
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
        depth (dict or unyt_quantity, optional): The depth of the instrument.
            If depths are provided in multiple bins or regions, this should be
            a dictionary keyed by the relevant labels.
        depth_app_radius (unyt_quantity, optional): The aperture radius for
            the depth measurement. If this is omitted but SNRs and depths are
            provided, the depth is assumed to be a point-source depth.
        snrs (dict or unyt_quantity, optional): The signal-to-noise ratios of
            the instrument. If values are provided in multiple bins or regions,
            this should be a dictionary keyed by the relevant labels.
        noise_maps (unyt_array, optional): An optional array with noise as a
            function of wavelength, in the same units as the spectral noise.
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
    """

    lam = Quantity("wavelength")

    # Share units rather than deep copying unyt's whole unit registry
    __deepcopy__ = deepcopy_with_shared_units

    @accepts(lam=angstrom)
    @timed("SpectroscopicInstrument.__init__")
    def __init__(
        self,
        label,
        lam,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        noise_maps=None,
        resolving_power=None,
        lsfs=None,
        resolving_power_lam=None,
        lsf_resample_factor=1,
    ):
        """Initialise a spectroscopic instrument.

        Args:
            label (str): A label for the instrument.
            lam (unyt_array): The wavelength array defining the spectral
                coverage of the instrument.
            depth (dict or unyt_quantity, optional): The depth of the
                instrument. If depths are provided in multiple bins or regions,
                this should be a dictionary keyed by the relevant labels.
            depth_app_radius (unyt_quantity, optional): The aperture radius for
                the depth measurement. If this is omitted but SNRs and depths
                are provided, the depth is assumed to be a point-source depth.
            snrs (dict or unyt_quantity, optional): The signal-to-noise ratios
                of the instrument. If values are provided in multiple bins or
                regions, this should be a dictionary keyed by the relevant
                labels.
            noise_maps (unyt_array, optional): An optional array with noise as
                a function of wavelength, in the same units as the spectral
                noise.
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
        """
        super().__init__(label)
        self.lam = lam
        self.depth = depth
        self.depth_app_radius = depth_app_radius
        self.snrs = snrs
        self.noise_maps = noise_maps
        self.resolving_power = resolving_power
        self.resolving_power_lam = resolving_power_lam
        self.lsf_resample_factor = lsf_resample_factor

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
        return False

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
            _hashable_state(self.depth),
            _hashable_state(self.depth_app_radius),
            _hashable_state(self.snrs),
            _hashable_state(self.noise_maps),
            _hashable_state(self.resolving_power),
            resolving_power_lam_state,
            _hashable_state(self.lsfs),
            _hashable_state(self.lsf_resample_factor),
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

    @timed("SpectroscopicInstrument.apply_noise")
    def apply_noise(self, spectrum, **kwargs):
        """Apply spectroscopic noise to an observed spectrum.

        This method is the placeholder instrument-owned entry point for noisy
        one-dimensional spectroscopy. The public behaviour surface should exist
        now even though the underlying machinery has not yet been implemented.

        Args:
            spectrum: Observed spectrum-like object to which noise should be
                applied.
            **kwargs: Future keyword arguments for controlling the noise model
                and its application.

        Raises:
            UnimplementedFunctionality: Raised because noisy one-dimensional
                spectroscopy has not yet been implemented on the instrument
                side.
        """
        # The instrument should eventually own spectroscopy-noise application,
        # but the machinery for that does not exist yet.
        raise exceptions.UnimplementedFunctionality(
            "SpectroscopicInstrument.apply_noise is not implemented yet."
        )

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
        write_instrument_attribute(
            group, "DepthApertureRadius", self.depth_app_radius
        )
        write_instrument_attribute(group, "SNRs", self.snrs)
        write_instrument_attribute(group, "NoiseMaps", self.noise_maps)

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
        depth_app_radius = read_instrument_attribute(
            group, "DepthApertureRadius"
        )
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
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "noise_maps": noise_maps,
            "resolving_power": resolving_power,
            "resolving_power_lam": resolving_power_lam,
            "lsfs": lsfs,
            "lsf_resample_factor": lsf_resample_factor,
        }
        payload.update(kwargs)

        return cls(
            label=payload["label"],
            lam=payload["lam"],
            depth=payload["depth"],
            depth_app_radius=payload["depth_app_radius"],
            snrs=payload["snrs"],
            noise_maps=payload["noise_maps"],
            resolving_power=payload["resolving_power"],
            resolving_power_lam=payload["resolving_power_lam"],
            lsfs=payload["lsfs"],
            lsf_resample_factor=payload["lsf_resample_factor"],
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
