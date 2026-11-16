"""Specialised spectroscopic instrument.

This instrument is designed to hold the attributes required for one-
dimensional spectroscopy. It stores a wavelength grid together with optional
depth, signal-to-noise, and noise-map definitions.
"""

from numbers import Real

import h5py
import numpy as np
from unyt import angstrom, unyt_array

from synthesizer import exceptions
from synthesizer.instruments.instrument_base import (
    InstrumentBase,
    _hashable_state,
)
from synthesizer.instruments.utils import (
    read_instrument_attribute,
    write_instrument_attribute,
)
from synthesizer.synth_warnings import deprecated
from synthesizer.units import Quantity, accepts, deepcopy_with_shared_units
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
        resolving_power (float or callable, optional): Resolving power
            (R = lambda / delta_lambda) of the instrument. May be a constant
            or a function of wavelength. Only a constant (float/int) value is
            persisted by :meth:`to_hdf5`; a callable resolving power cannot be
            serialised to HDF5 and is silently dropped on save (it is not
            restored on load either, since there is nothing to restore it
            from).
        lsfs (array, optional): Normalised line spread function kernels
            with shape (nwave, nkernel). Each row corresponds to a
            wavelength in lam. Kernels must have an odd number of bins
            and sum to unity.
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
            resolving_power (float or callable, optional): Resolving power
                (R = lambda / delta_lambda) of the instrument. May be a
                constant or a function of wavelength. Only a constant value is
                persisted when the instrument is saved to HDF5.
            lsfs (array, optional): Normalised line spread function
                kernels with shape (nwave, nkernel), sampled on the
                instrument wavelength grid.
        """
        super().__init__(label)
        self.lam = lam
        self.depth = depth
        self.depth_app_radius = depth_app_radius
        self.snrs = snrs
        self.noise_maps = noise_maps
        self.resolving_power = resolving_power
        self.lsfs = None if lsfs is None else np.asarray(lsfs).copy()
        SpectroscopicInstrument._validate(self)

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
            if (
                isinstance(self.resolving_power, bool)
                or not isinstance(self.resolving_power, Real)
                or not np.isfinite(self.resolving_power)
                or self.resolving_power <= 0
            ):
                raise exceptions.InconsistentArguments(
                    "resolving_power must be a positive number or callable."
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

    @timed("SpectroscopicInstrument._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the spectroscopic comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        return (
            _hashable_state(self._lam),
            _hashable_state(self.depth),
            _hashable_state(self.depth_app_radius),
            _hashable_state(self.snrs),
            _hashable_state(self.noise_maps),
            _hashable_state(self.resolving_power),
            _hashable_state(self.lsfs),
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

        # Only a constant resolving power can be represented as an HDF5
        # attribute; a callable resolving_power(wavelength) is not
        # serialisable and is intentionally dropped here. NumPy scalars count
        # as constants: np.float32 and np.int64 are not instances of float or
        # int, so testing those alone would silently discard a resolving
        # power that came out of an array.
        if isinstance(self.resolving_power, Real) and not isinstance(
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

        payload = {
            "label": group.attrs["label"],
            "lam": lam,
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "noise_maps": noise_maps,
            "resolving_power": group.attrs.get("resolving_power"),
            "lsfs": lsfs,
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
            lsfs=payload["lsfs"],
        )
