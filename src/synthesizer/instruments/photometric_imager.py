"""Specialised photometric imaging instrument."""

import numpy as np
from unyt import arcsecond, kpc

from synthesizer import exceptions
from synthesizer.instruments.generic_instrument import (
    unpack_instrument_payload,
)
from synthesizer.instruments.instrument_base import _hashable_state
from synthesizer.instruments.photometric_instrument import (
    PhotometricInstrument,
)
from synthesizer.instruments.photometric_noise import CorrelatedNoiseModel
from synthesizer.units import accepts


class PhotometricImager(PhotometricInstrument):
    """Concrete instrument for imaging-capable photometric setups."""

    @accepts(resolution=(kpc, arcsecond))
    def __init__(
        self,
        label,
        filters,
        resolution,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        """Initialise a photometric imager."""
        super().__init__(
            label=label,
            filters=filters,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
        )
        self.resolution = resolution
        self.psfs = psfs
        self.noise_maps = noise_maps
        self.noise_source_maps = noise_source_maps
        self.correlated_noise_models = self._build_correlated_noise_models()
        self._validate()

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "photometric_imager"

    @property
    def can_do_imaging(self):
        """Return whether this instrument supports imaging."""
        return True

    @property
    def can_do_psf_imaging(self):
        """Return whether this instrument supports PSF imaging."""
        return self.psfs is not None

    @property
    def can_do_noisy_imaging(self):
        """Return whether this instrument supports noisy imaging."""
        have_noise = self.noise_maps is not None
        have_noise |= self.noise_source_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None
        return have_noise

    def _validate(self):
        super()._validate()

        if self.resolution is None:
            raise exceptions.MissingArgument(
                "PhotometricImager requires a resolution."
            )
        if self.snrs is not None and self.noise_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as noise maps"
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
        return super()._comparison_state() + (
            _hashable_state(self.resolution),
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
        """Apply the configured imaging noise to one image."""
        if self.noise_maps is not None:
            noise_arr = self.noise_maps[filter_code]
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
            "The instrument has no imaging noise configuration."
        )

    def apply_noises(
        self,
        image_collection,
        correct_periodicity=True,
        rng_seed=None,
        aperture_radius=None,
    ):
        """Apply the configured imaging noise to an image collection."""
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

    def to_hdf5(self, group):
        """Write the photometric imager to an HDF5 group."""
        super().to_hdf5(group)

        ds = group.create_dataset(
            "Resolution", data=self.resolution.value, dtype=float
        )
        ds.attrs["units"] = str(self.resolution.units)

        if self.psfs is not None:
            psfs_group = group.create_group("PSFs")
            for key, value in self.psfs.items():
                ds = psfs_group.create_dataset(key, data=value, dtype=float)
                ds.attrs["units"] = "dimensionless"

        if self.noise_maps is not None:
            noise_group = group.create_group("NoiseMaps")
            for key, value in self.noise_maps.items():
                ds = noise_group.create_dataset(
                    key, data=value.value, dtype=float
                )
                ds.attrs["units"] = str(value.units)

        if self.noise_source_maps is not None:
            noise_source_group = group.create_group("NoiseSourceMaps")
            for key, value in self.noise_source_maps.items():
                ds = noise_source_group.create_dataset(
                    key, data=value.value, dtype=float
                )
                ds.attrs["units"] = str(value.units)

    @classmethod
    def _from_hdf5(cls, group, **kwargs):
        payload = unpack_instrument_payload(group, **kwargs)
        return cls(
            label=payload["label"],
            filters=payload["filters"],
            resolution=payload["resolution"],
            depth=payload["depth"],
            depth_app_radius=payload["depth_app_radius"],
            snrs=payload["snrs"],
            psfs=payload["psfs"],
            noise_maps=payload["noise_maps"],
            noise_source_maps=payload["noise_source_maps"],
        )
