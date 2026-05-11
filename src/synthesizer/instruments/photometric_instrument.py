"""Specialised photometric instrument."""

import h5py

from synthesizer import exceptions
from synthesizer.instruments.instrument import unpack_instrument_payload
from synthesizer.instruments.instrument_base import (
    InstrumentBase,
    _hashable_state,
)


class PhotometricInstrument(InstrumentBase):
    """Concrete instrument for photometric-only configurations.

    A `PhotometricInstrument` owns a `FilterCollection` and the optional depth
    / signal-to-noise configuration needed for noisy photometry. It does not
    assume any spatial resolution or imaging-specific state such as PSFs or
    pixel-space noise maps.
    """

    def __init__(
        self,
        label,
        filters,
        depth=None,
        depth_app_radius=None,
        snrs=None,
    ):
        """Initialise a photometric instrument."""
        super().__init__(label)
        self.filters = filters
        self.depth = depth
        self.depth_app_radius = depth_app_radius
        self.snrs = snrs
        PhotometricInstrument._validate(self)

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "photometric"

    @property
    def can_do_photometry(self):
        """Return whether this instrument supports photometry."""
        return True

    def _validate(self):
        if self.filters is None:
            raise exceptions.MissingArgument(
                "PhotometricInstrument requires filters."
            )
        if self.depth is not None and self.snrs is None:
            raise exceptions.MissingArgument(
                "If you set a depth you must also set the SNRs"
            )
        if self.snrs is not None and self.depth is None:
            raise exceptions.MissingArgument(
                "If you set a SNR you must also set the depth"
            )

    def _comparison_state(self):
        return (
            _hashable_state(self.filters.filter_codes),
            _hashable_state(self.depth),
            _hashable_state(self.depth_app_radius),
            _hashable_state(self.snrs),
        )

    def add_filters(
        self,
        filters,
        psfs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        """Add filters and optional imaging payloads to the instrument."""
        if psfs is not None and set(psfs.keys()) != set(filters.filter_codes):
            raise exceptions.InconsistentAddition(
                "PSFs missing for filters: "
                f"{set(filters.filter_codes) - set(psfs.keys())}"
            )
        if noise_maps is not None and set(noise_maps.keys()) != set(
            filters.filter_codes
        ):
            raise exceptions.InconsistentAddition(
                "Noise maps missing for filters: "
                f"{set(filters.filter_codes) - set(noise_maps.keys())}"
            )
        if noise_source_maps is not None and set(
            noise_source_maps.keys()
        ) != set(filters.filter_codes):
            raise exceptions.InconsistentAddition(
                "Noise source maps missing for filters: "
                f"{set(filters.filter_codes) - set(noise_source_maps.keys())}"
            )

        if self.snrs is not None and noise_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as noise maps"
            )
        if self.snrs is not None and noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as "
                "noise source maps"
            )
        if (
            noise_maps is not None
            and getattr(self, "noise_source_maps", None) is not None
        ):
            raise exceptions.MissingArgument(
                "You cannot set fixed noise maps and correlated noise source "
                "maps at the same time"
            )
        if (
            noise_source_maps is not None
            and getattr(self, "noise_maps", None) is not None
        ):
            raise exceptions.MissingArgument(
                "You cannot set fixed noise maps and correlated noise source "
                "maps at the same time"
            )

        self.filters += filters

        if psfs is not None:
            if getattr(self, "psfs", None) is None:
                self.psfs = {}
            self.psfs.update(psfs)

        if noise_maps is not None:
            if getattr(self, "noise_maps", None) is None:
                self.noise_maps = {}
            self.noise_maps.update(noise_maps)

        if noise_source_maps is not None:
            if getattr(self, "noise_source_maps", None) is None:
                self.noise_source_maps = {}
            self.noise_source_maps.update(noise_source_maps)
            if hasattr(self, "_build_correlated_noise_models"):
                self.correlated_noise_models = (
                    self._build_correlated_noise_models()
                )

        self._validate()

    def to_hdf5(self, group):
        """Write the photometric instrument to an HDF5 group."""
        group.attrs["label"] = self.label
        group.attrs["instrument_type"] = self.instrument_type

        filters_group = group.create_group("Filters")
        self.filters._write_filters_to_group(filters_group)

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

    @classmethod
    def load(cls, filepath, **kwargs):
        """Load a photometric instrument from an HDF5 file."""
        with h5py.File(filepath, "r") as hdf:
            return cls._from_hdf5(hdf, **kwargs)

    @classmethod
    def _from_hdf5(cls, group, **kwargs):
        payload = unpack_instrument_payload(group, **kwargs)
        return cls(
            label=payload["label"],
            filters=payload["filters"],
            depth=payload["depth"],
            depth_app_radius=payload["depth_app_radius"],
            snrs=payload["snrs"],
        )
