"""Specialised spectroscopic instrument.

This instrument is designed to hold the attributes required for one-
dimensional spectroscopy. It stores a wavelength grid together with optional
depth, signal-to-noise, and noise-map definitions.
"""

import h5py
from unyt import angstrom, unyt_array

from synthesizer import exceptions
from synthesizer.instruments.instrument_base import (
    InstrumentBase,
    _hashable_state,
)
from synthesizer.units import Quantity, accepts


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
    """

    lam = Quantity("wavelength")

    @accepts(lam=angstrom)
    def __init__(
        self,
        label,
        lam,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        noise_maps=None,
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
        """
        super().__init__(label)
        self.lam = lam
        self.depth = depth
        self.depth_app_radius = depth_app_radius
        self.snrs = snrs
        self.noise_maps = noise_maps
        SpectroscopicInstrument._validate(self)

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
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None
        return have_noise

    def _validate(self):
        """Validate the spectroscopic attributes.

        Raises:
            MissingArgument: If required spectroscopic attributes are missing
                or inconsistent.
        """
        if self.lam is None:
            raise exceptions.MissingArgument(
                "SpectroscopicInstrument requires lam."
            )
        if self.depth is not None and self.snrs is None:
            raise exceptions.MissingArgument(
                "If you set a depth you must also set the SNRs"
            )
        if self.snrs is not None and self.depth is None:
            raise exceptions.MissingArgument(
                "If you set a SNR you must also set the depth"
            )
        if self.snrs is not None and self.noise_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as noise maps"
            )

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
        )

    def to_hdf5(self, group):
        """Write the spectroscopic instrument to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised.
        """
        group.attrs["label"] = self.label
        group.attrs["instrument_type"] = self.instrument_type

        ds = group.create_dataset(
            "Wavelength", data=self.lam.value, dtype=float
        )
        ds.attrs["units"] = str(self.lam.units)

        if self.depth is not None:
            if isinstance(self.depth, dict):
                depth_group = group.create_group("Depth")
                for key, value in self.depth.items():
                    ds = depth_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = "dimensionless"
            else:
                ds = group.create_dataset(
                    "Depth", data=self.depth.value, dtype=float
                )
                ds.attrs["units"] = "dimensionless"

        if self.depth_app_radius is not None:
            ds = group.create_dataset(
                "DepthApertureRadius",
                data=self.depth_app_radius.value,
                dtype=float,
            )
            ds.attrs["units"] = str(self.depth_app_radius.units)

        if self.snrs is not None:
            if isinstance(self.snrs, dict):
                snrs_group = group.create_group("SNRs")
                for key, value in self.snrs.items():
                    ds = snrs_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = "dimensionless"
            else:
                ds = group.create_dataset(
                    "SNRs", data=self.snrs.value, dtype=float
                )
                ds.attrs["units"] = "dimensionless"

        if self.noise_maps is not None:
            ds = group.create_dataset(
                "NoiseMaps", data=self.noise_maps.value, dtype=float
            )
            ds.attrs["units"] = str(self.noise_maps.units)

    @classmethod
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

        if "Depth" in group and isinstance(group["Depth"], h5py.Group):
            depth = {
                key: unyt_array(value[...], value.attrs["units"])
                for key, value in group["Depth"].items()
            }
        elif "Depth" in group:
            depth = unyt_array(
                group["Depth"][...], group["Depth"].attrs["units"]
            )
        else:
            depth = None

        if "DepthApertureRadius" in group:
            depth_app_radius = unyt_array(
                group["DepthApertureRadius"][...],
                group["DepthApertureRadius"].attrs["units"],
            )
        else:
            depth_app_radius = None

        if "SNRs" in group and isinstance(group["SNRs"], h5py.Group):
            snrs = {
                key: unyt_array(value[...], value.attrs["units"])
                for key, value in group["SNRs"].items()
            }
        elif "SNRs" in group:
            snrs = unyt_array(group["SNRs"][...], group["SNRs"].attrs["units"])
        else:
            snrs = None

        if "NoiseMaps" in group:
            noise_maps = unyt_array(
                group["NoiseMaps"][...], group["NoiseMaps"].attrs["units"]
            )
        else:
            noise_maps = None

        payload = {
            "label": group.attrs["label"],
            "lam": lam,
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "noise_maps": noise_maps,
        }
        payload.update(kwargs)

        return cls(
            label=payload["label"],
            lam=payload["lam"],
            depth=payload["depth"],
            depth_app_radius=payload["depth_app_radius"],
            snrs=payload["snrs"],
            noise_maps=payload["noise_maps"],
        )
