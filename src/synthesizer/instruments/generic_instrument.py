"""Generic compatibility instrument implementation."""

import h5py
import numpy as np
from unyt import angstrom, arcsecond, kpc, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.instruments.filters import FilterCollection
from synthesizer.instruments.instrument_base import (
    InstrumentBase,
    _hashable_state,
)
from synthesizer.instruments.photometric_noise import CorrelatedNoiseModel
from synthesizer.units import Quantity, accepts


class GenericInstrument(InstrumentBase):
    """Compatibility instrument supporting mixed-mode configurations."""

    lam = Quantity("wavelength")

    @accepts(resolution=(kpc, arcsecond), lam=angstrom)
    def __init__(
        self,
        label,
        filters=None,
        resolution=None,
        lam=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        """Initialise a generic mixed-capability instrument."""
        super().__init__(label)
        self.filters = filters

        if (
            isinstance(resolution, (unyt_array, unyt_quantity))
            or resolution is None
        ):
            self.resolution = resolution
        else:
            raise exceptions.InconsistentArguments(
                "Resolution must have units."
            )

        self.lam = lam
        self.depth = depth
        self.depth_app_radius = depth_app_radius
        self.snrs = snrs
        self.psfs = psfs
        self.noise_maps = noise_maps
        self.noise_source_maps = noise_source_maps
        self.correlated_noise_models = self._build_correlated_noise_models()

        self._validate()

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "generic"

    @property
    def can_do_photometry(self):
        """Return whether this instrument supports photometry."""
        return self.filters is not None

    @property
    def can_do_imaging(self):
        """Return whether this instrument supports imaging."""
        return self.can_do_photometry and self.resolution is not None

    @property
    def can_do_psf_imaging(self):
        """Return whether this instrument supports PSF imaging."""
        return self.can_do_imaging and self.psfs is not None

    @property
    def can_do_noisy_imaging(self):
        """Return whether this instrument supports noisy imaging."""
        have_noise = self.noise_maps is not None
        have_noise |= self.noise_source_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None
        return self.can_do_imaging and have_noise

    @property
    def can_do_spectroscopy(self):
        """Return whether this instrument supports spectroscopy."""
        return self.lam is not None

    @property
    def can_do_noisy_spectroscopy(self):
        """Return whether this instrument supports noisy spectroscopy."""
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None
        return self.can_do_spectroscopy and have_noise

    @property
    def can_do_resolved_spectroscopy(self):
        """Return whether this instrument supports resolved spectroscopy."""
        return self.can_do_spectroscopy and self.resolution is not None

    @property
    def can_do_psf_spectroscopy(self):
        """Return whether this instrument supports PSF spectroscopy."""
        return self.can_do_resolved_spectroscopy and self.psfs is not None

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """Return whether this instrument supports noisy IFU work."""
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None
        return self.can_do_resolved_spectroscopy and have_noise

    def _validate(self):
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
                "You cannot set depths and SNRs at the same time as "
                " noise maps"
            )

        if self.snrs is not None and self.noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as "
                "noise source maps"
            )

        if self.noise_maps is not None and self.noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set fixed noise maps and correlated noise source "
                "maps at the same time"
            )

    def _build_correlated_noise_models(self):
        if self.noise_source_maps is None:
            return None

        if not isinstance(self.noise_source_maps, dict):
            raise exceptions.InconsistentArguments(
                "noise_source_maps must be a dict keyed by filter code for "
                "correlated noise generation."
            )

        return {
            filter_code: CorrelatedNoiseModel(noise_map)
            for filter_code, noise_map in self.noise_source_maps.items()
        }

    def _comparison_state(self):
        return (
            _hashable_state(
                self.filters.filter_codes if self.filters is not None else None
            ),
            _hashable_state(self.resolution),
            _hashable_state(self._lam),
            _hashable_state(self.depth),
            _hashable_state(self.depth_app_radius),
            _hashable_state(self.snrs),
            _hashable_state(self.psfs),
            _hashable_state(self.noise_maps),
            _hashable_state(self.noise_source_maps),
        )

    def get_correlated_noise_model(self, filter_code):
        """Return the correlated-noise model for a filter."""
        if self.correlated_noise_models is None:
            raise exceptions.MissingArgument(
                "No correlated noise models are set on this Instrument. "
                "Provide noise_source_maps when constructing the Instrument."
            )

        if filter_code not in self.correlated_noise_models:
            raise exceptions.InconsistentArguments(
                "No correlated noise model found for filter "
                f"'{filter_code}'. Available filters: "
                f"{list(self.correlated_noise_models.keys())}"
            )

        return self.correlated_noise_models[filter_code]

    def apply_noise(
        self,
        image,
        filter_code,
        correct_periodicity=True,
        rng_seed=None,
        aperture_radius=None,
    ):
        """Apply the configured noise model to one image."""
        if self.noise_maps is not None:
            noise_arr = (
                self.noise_maps[filter_code]
                if isinstance(self.noise_maps, dict)
                else self.noise_maps
            )
            return image.apply_noise_array(noise_arr)
        if self.correlated_noise_models is not None:
            return image.apply_correlated_noise(
                self,
                filter_code,
                correct_periodicity=correct_periodicity,
                rng_seed=rng_seed,
            )
        if self.snrs is not None and self.depth is not None:
            snr = (
                self.snrs[filter_code]
                if isinstance(self.snrs, dict)
                else self.snrs
            )
            depth = (
                self.depth[filter_code]
                if isinstance(self.depth, dict)
                else self.depth
            )
            return image.apply_noise_from_snr(
                snr=snr, depth=depth, aperture_radius=aperture_radius
            )

        raise exceptions.MissingArgument(
            "The instrument has no noise configuration. Set either noise_maps,"
            "noise_source_maps, or snrs and depth before calling apply_noise."
        )

    def apply_noises(
        self,
        image_collection,
        correct_periodicity=True,
        rng_seed=None,
        aperture_radius=None,
    ):
        """Apply the configured noise model to an image collection."""
        from synthesizer.imaging.image_collection import ImageCollection

        rng = np.random.default_rng(rng_seed)
        noisy_imgs = {}
        for f in image_collection.filter_codes:
            filter_rng_seed = (
                None
                if rng_seed is None
                else int(rng.integers(0, np.iinfo(np.uint32).max))
            )
            noisy_imgs[f] = self.apply_noise(
                image_collection.imgs[f],
                f,
                correct_periodicity=correct_periodicity,
                rng_seed=filter_rng_seed,
                aperture_radius=aperture_radius,
            )

        return ImageCollection(
            resolution=image_collection.resolution,
            fov=image_collection.fov,
            imgs=noisy_imgs,
        )

    def add_filters(
        self,
        filters,
        psfs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        """Add filters and optional imaging payloads to the instrument."""
        if not self.can_do_photometry:
            raise exceptions.InconsistentAddition(
                "Cannot add filters to an instrument without photometric "
                "capabilities."
            )

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
        if noise_maps is not None and self.noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set fixed noise maps and correlated noise source "
                "maps at the same time"
            )
        if noise_source_maps is not None and self.noise_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set fixed noise maps and correlated noise source "
                "maps at the same time"
            )

        self.filters += filters

        if psfs is not None:
            self.psfs.update(psfs)

        if noise_maps is not None:
            if self.noise_maps is None:
                self.noise_maps = {}
            self.noise_maps.update(noise_maps)

        if noise_source_maps is not None:
            if self.noise_source_maps is None:
                self.noise_source_maps = {}
            self.noise_source_maps.update(noise_source_maps)
            self.correlated_noise_models = (
                self._build_correlated_noise_models()
            )

        self._validate()

    def to_hdf5(self, group):
        """Write the generic instrument to an HDF5 group."""
        group.attrs["label"] = self.label
        group.attrs["instrument_type"] = self.instrument_type

        if self.filters is not None:
            filters_group = group.create_group("Filters")
            self.filters._write_filters_to_group(filters_group)

        if self.resolution is not None:
            ds = group.create_dataset(
                "Resolution", data=self.resolution.value, dtype=float
            )
            ds.attrs["units"] = str(self.resolution.units)
        if self.lam is not None:
            ds = group.create_dataset(
                "Wavelength", data=self.lam.value, dtype=float
            )
            ds.attrs["units"] = str(self.lam.units)
        if self.depth_app_radius is not None:
            ds = group.create_dataset(
                "DepthApertureRadius",
                data=self.depth_app_radius.value,
                dtype=float,
            )
            ds.attrs["units"] = str(self.depth_app_radius.units)

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

        if self.psfs is not None:
            if isinstance(self.psfs, dict):
                psfs_group = group.create_group("PSFs")
                for key, value in self.psfs.items():
                    ds = psfs_group.create_dataset(
                        key, data=value, dtype=float
                    )
                    ds.attrs["units"] = "dimensionless"
            else:
                ds = group.create_dataset("PSFs", data=self.psfs, dtype=float)
                ds.attrs["units"] = "dimensionless"

        if self.noise_maps is not None:
            if isinstance(self.noise_maps, dict):
                noise_group = group.create_group("NoiseMaps")
                for key, value in self.noise_maps.items():
                    ds = noise_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = str(value.units)
            else:
                ds = group.create_dataset(
                    "NoiseMaps", data=self.noise_maps.value, dtype=float
                )
                ds.attrs["units"] = str(self.noise_maps.units)

        if self.noise_source_maps is not None:
            noise_source_group = group.create_group("NoiseSourceMaps")
            for key, value in self.noise_source_maps.items():
                ds = noise_source_group.create_dataset(
                    key, data=value.value, dtype=float
                )
                ds.attrs["units"] = str(value.units)


def unpack_instrument_payload(group, **kwargs):
    """Read common instrument fields from an HDF5 group."""
    if "Filters" in group:
        filters = FilterCollection._from_hdf5(group["Filters"])
    else:
        filters = None

    if "Resolution" in group:
        resolution = unyt_array(
            group["Resolution"][...], group["Resolution"].attrs["units"]
        )
    else:
        resolution = None

    if "Wavelength" in group:
        lam = unyt_array(
            group["Wavelength"][...], group["Wavelength"].attrs["units"]
        )
    else:
        lam = None

    if "Depth" in group and isinstance(group["Depth"], h5py.Group):
        depth = {
            key: unyt_array(value[...], value.attrs["units"])
            for key, value in group["Depth"].items()
        }
    elif "Depth" in group:
        depth = unyt_array(group["Depth"][...], group["Depth"].attrs["units"])
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

    if "PSFs" in group and isinstance(group["PSFs"], h5py.Group):
        psfs = {}
        for key in group["PSFs"]:
            if isinstance(group["PSFs"][key], h5py.Group):
                for subkey in group["PSFs"][key]:
                    psfs[f"{key}/{subkey}"] = unyt_array(
                        group["PSFs"][key][subkey][...],
                        group["PSFs"][key][subkey].attrs["units"],
                    )
            else:
                psfs[key] = unyt_array(
                    group["PSFs"][key][...],
                    group["PSFs"][key].attrs["units"],
                )
    elif "PSFs" in group:
        psfs = unyt_array(group["PSFs"][...], group["PSFs"].attrs["units"])
    else:
        psfs = None

    if "NoiseMaps" in group and isinstance(group["NoiseMaps"], h5py.Group):
        noise_maps = {
            key: unyt_array(value[...], value.attrs["units"])
            for key, value in group["NoiseMaps"].items()
        }
    elif "NoiseMaps" in group:
        noise_maps = unyt_array(
            group["NoiseMaps"][...], group["NoiseMaps"].attrs["units"]
        )
    else:
        noise_maps = None

    if "NoiseSourceMaps" in group and isinstance(
        group["NoiseSourceMaps"], h5py.Group
    ):
        noise_source_maps = {
            key: unyt_array(value[...], value.attrs["units"])
            for key, value in group["NoiseSourceMaps"].items()
        }
    else:
        noise_source_maps = None

    payload = {
        "label": group.attrs["label"],
        "filters": filters,
        "resolution": resolution,
        "lam": lam,
        "depth": depth,
        "depth_app_radius": depth_app_radius,
        "snrs": snrs,
        "psfs": psfs,
        "noise_maps": noise_maps,
        "noise_source_maps": noise_source_maps,
    }
    payload.update(kwargs)
    return payload
