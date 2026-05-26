"""Specialised photometric instrument.

This instrument is designed to hold the attributes required for integrated
photometry. It stores a :class:`FilterCollection` together with optional depth
and signal-to-noise definitions for noisy photometric measurements.
"""

import h5py
from unyt import unyt_array

from synthesizer import exceptions
from synthesizer.instruments.filters import FilterCollection
from synthesizer.instruments.instrument_base import (
    InstrumentBase,
    _hashable_state,
)
from synthesizer.utils.operation_timers import timed


class PhotometricInstrument(InstrumentBase):
    """Photometric instrument class.

    A class containing the attributes and methods required for integrated
    photometry. It holds the filters defining the instrument transmission
    curves together with the optional noise information required to generate
    noisy photometric measurements. It does not include any spatially resolved
    information such as an image resolution, PSFs, or image-plane noise maps.

    Attributes:
        filters (FilterCollection): The filters defining the photometric
            response of the instrument.
        depth (dict or unyt_quantity, optional): The depth of the instrument,
            typically in apparent magnitudes. If depths are provided per
            filter, this should be a dictionary keyed by filter code.
        depth_app_radius (unyt_quantity, optional): The aperture radius for
            the depth measurement. If omitted but depths and SNRs are provided,
            the depth is assumed to be a point-source depth.
        snrs (dict or unyt_quantity, optional): The signal-to-noise ratios of
            the instrument. If values are provided per filter, this should be a
            dictionary keyed by filter code.
    """

    @timed("PhotometricInstrument.__init__")
    def __init__(
        self,
        label,
        filters,
        depth=None,
        depth_app_radius=None,
        snrs=None,
    ):
        """Initialise a photometric instrument.

        Args:
            label (str): A label for the instrument.
            filters (FilterCollection): The filters defining the photometric
                response of the instrument.
            depth (dict or unyt_quantity, optional): The depth of the
                instrument, typically in apparent magnitudes. If depths are
                provided per filter, this should be a dictionary keyed by
                filter code.
            depth_app_radius (unyt_quantity, optional): The aperture radius for
                the depth measurement. If this is omitted but SNRs and depths
                are provided, the depth is assumed to be a point-source depth.
            snrs (dict or unyt_quantity, optional): The signal-to-noise ratios
                of the instrument. If values are provided per filter, this
                should be a dictionary keyed by filter code.
        """
        super().__init__(label)
        self.filters = filters
        self.depth = depth
        self.depth_app_radius = depth_app_radius
        self.snrs = snrs
        PhotometricInstrument._validate(self)

    @timed("PhotometricInstrument._validate")
    def _validate(self):
        """Validate the instrument attributes.

        Raises:
            MissingArgument: If any required attributes are missing.
        """
        # Ensure we actually have filters defining the instrument
        if self.filters is None:
            raise exceptions.MissingArgument(
                "PhotometricInstrument requires filters."
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

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "photometric"

    @property
    def can_do_photometry(self):
        """Return whether this instrument supports photometry."""
        return True

    @timed("PhotometricInstrument._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the photometric comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        return (
            _hashable_state(self.filters.filter_codes),
            _hashable_state(self.depth),
            _hashable_state(self.depth_app_radius),
            _hashable_state(self.snrs),
        )

    @timed("PhotometricInstrument.add_filters")
    def add_filters(
        self,
        filters,
        psfs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        """Add filters and optional imaging payloads to the instrument.

        Args:
            filters (FilterCollection): The filters to add to the instrument.
            psfs (dict, optional): Optional PSFs keyed by filter code. These
                are only meaningful for imaging-capable subclasses.
            noise_maps (dict, optional): Optional fixed noise maps keyed by
                filter code. These are only meaningful for imaging-capable
                subclasses.
            noise_source_maps (dict, optional): Optional correlated-noise
                source maps keyed by filter code. These are only meaningful for
                imaging-capable subclasses.
        """
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

        new_filter_codes = set(filters.filter_codes)
        if isinstance(getattr(self, "depth", None), dict):
            missing_depths = new_filter_codes - set(self.depth.keys())
            if len(missing_depths) > 0:
                raise exceptions.InconsistentAddition(
                    "Cannot add filters without matching depth entries for: "
                    f"{missing_depths}"
                )
        if isinstance(getattr(self, "snrs", None), dict):
            missing_snrs = new_filter_codes - set(self.snrs.keys())
            if len(missing_snrs) > 0:
                raise exceptions.InconsistentAddition(
                    "Cannot add filters without matching SNR entries for: "
                    f"{missing_snrs}"
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

    @timed("PhotometricInstrument.to_hdf5")
    def to_hdf5(self, group):
        """Write the photometric instrument to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised.
        """
        group.attrs["label"] = self.label
        group.attrs["instrument_type"] = self.instrument_type

        filters_group = group.create_group("Filters")
        self.filters._write_filters_to_group(filters_group)

        if self.depth is not None:
            if isinstance(self.depth, dict):
                depth_group = group.create_group("Depth")
                for key, value in self.depth.items():
                    raw = value.value if hasattr(value, "value") else value
                    units = (
                        str(value.units)
                        if hasattr(value, "units")
                        else "dimensionless"
                    )
                    ds = depth_group.create_dataset(key, data=raw, dtype=float)
                    ds.attrs["units"] = units
            else:
                raw = (
                    self.depth.value
                    if hasattr(self.depth, "value")
                    else self.depth
                )
                units = (
                    str(self.depth.units)
                    if hasattr(self.depth, "units")
                    else "dimensionless"
                )
                ds = group.create_dataset("Depth", data=raw, dtype=float)
                ds.attrs["units"] = units

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
                    raw = value.value if hasattr(value, "value") else value
                    units = (
                        str(value.units)
                        if hasattr(value, "units")
                        else "dimensionless"
                    )
                    ds = snrs_group.create_dataset(key, data=raw, dtype=float)
                    ds.attrs["units"] = units
            else:
                raw = (
                    self.snrs.value
                    if hasattr(self.snrs, "value")
                    else self.snrs
                )
                units = (
                    str(self.snrs.units)
                    if hasattr(self.snrs, "units")
                    else "dimensionless"
                )
                ds = group.create_dataset("SNRs", data=raw, dtype=float)
                ds.attrs["units"] = units

    @classmethod
    @timed("PhotometricInstrument.load")
    def load(cls, filepath, **kwargs):
        """Load a photometric instrument from an HDF5 file.

        Args:
            filepath (str or PathLike): Path to the HDF5 file.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            PhotometricInstrument: The loaded instrument.
        """
        with h5py.File(filepath, "r") as hdf:
            return cls._from_hdf5(hdf, **kwargs)

    @classmethod
    @timed("PhotometricInstrument._from_hdf5")
    def _from_hdf5(cls, group, **kwargs):
        """Load a photometric instrument from an HDF5 group.

        Args:
            group (h5py.Group): Group containing the serialised instrument.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            PhotometricInstrument: The loaded instrument.
        """
        filters = FilterCollection._from_hdf5(group["Filters"])

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

        payload = {
            "label": group.attrs["label"],
            "filters": filters,
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
        }
        payload.update(kwargs)

        return cls(
            label=payload["label"],
            filters=payload["filters"],
            depth=payload["depth"],
            depth_app_radius=payload["depth_app_radius"],
            snrs=payload["snrs"],
        )
