"""Specialised photometric imaging instrument.

This instrument is designed to hold the attributes required by photometric
imaging. It extends :class:`PhotometricInstrument` with spatial resolution,
optional PSFs, fixed noise maps, and correlated-noise source maps.
"""

import hashlib
import inspect

import h5py
from scipy import signal
from unyt import arcsecond, kpc, unyt_array

from synthesizer import exceptions
from synthesizer.imaging.image import Image
from synthesizer.imaging.image_collection import ImageCollection
from synthesizer.imaging.image_generators import (
    _generate_image_collection_generic,
)
from synthesizer.instruments.filters import FilterCollection
from synthesizer.instruments.instrument_base import _hashable_state
from synthesizer.instruments.photometric_instrument import (
    PhotometricInstrument,
)
from synthesizer.instruments.photometric_noise import CorrelatedNoiseModel
from synthesizer.units import accepts
from synthesizer.utils.operation_timers import timed


class PhotometricImager(PhotometricInstrument):
    """Photometric imager instrument class.

    A class containing the attributes and methods required to produce
    photometric images. It extends :class:`PhotometricInstrument` with spatial
    resolution, optional PSFs, fixed noise maps, and correlated-noise source
    maps. This is the instrument class to use when a photometric setup must
    produce resolved images rather than integrated photometry alone.

    Attributes:
        resolution (unyt_array): The spatial resolution of the instrument, in
            kpc or arcseconds.
        psfs (dict, optional): An optional dictionary of point spread
            functions, with one entry per filter.
        noise_maps (dict, optional): An optional dictionary of fixed noise maps
            to apply directly to images, with one entry per filter.
        noise_source_maps (dict, optional): An optional dictionary of source
            maps used to generate correlated-noise models, with one entry per
            filter.
    """

    @accepts(resolution=(kpc, arcsecond))
    @timed("PhotometricImager.__init__")
    def __init__(
        self,
        label,
        filters,
        resolution,
        psfs=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        """Initialise a photometric imager.

        Args:
            label (str): A label for the instrument.
            filters (FilterCollection): The filters defining the photometric
                response of the instrument.
            resolution (unyt_array): The spatial resolution of the instrument,
                in kpc or arcseconds.
            psfs (dict, optional): An optional dictionary of point spread
                functions, with one entry per filter.
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
            noise_maps (dict, optional): An optional dictionary of fixed noise
                maps to apply directly to images, with one entry per filter.
            noise_source_maps (dict, optional): An optional dictionary of
                source maps used to generate correlated-noise models, with one
                entry per filter.
        """
        # Initialise the shared photometric instrument first
        super().__init__(
            label=label,
            filters=filters,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
        )

        # Set the photometric imager-specific attributes
        self.resolution = resolution
        self.psfs = psfs
        self.noise_maps = noise_maps
        self.noise_source_maps = noise_source_maps
        self.correlated_noise_models = self._build_correlated_noise_models()

        # Validate the instrument configuration
        self._validate()

    @timed("PhotometricImager._validate")
    def _validate(self):
        """Validate the instrument attributes.

        Raises:
            MissingArgument: If any required attributes are missing.
        """
        # Perform the shared validation first
        super()._validate()

        # Ensure we actually have the image resolution
        if self.resolution is None:
            raise exceptions.MissingArgument(
                "PhotometricImager requires a resolution."
            )

        # Noise maps are an alternative noise definition to depth+SNR pairs
        if self.snrs is not None and self.noise_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as noise maps"
            )

        # Correlated-noise source maps are also an alternative noise definition
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

    @timed("PhotometricImager._build_correlated_noise_models")
    def _build_correlated_noise_models(self):
        """Build per-filter correlated-noise models from source maps.

        Returns:
            dict or None: Mapping of filter codes to correlated-noise models,
                or None if no source maps are configured.

        Raises:
            InconsistentArguments: If ``noise_source_maps`` is not a dict.
        """
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

    @timed("PhotometricImager._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the imaging comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        return super()._comparison_state() + (
            _hashable_state(self.resolution),
            _hashable_state(self.psfs),
            _hashable_state(self.noise_maps),
            _hashable_state(self.noise_source_maps),
        )

    @timed("PhotometricImager.get_correlated_noise_model")
    def get_correlated_noise_model(self, filter_code):
        """Return the correlated-noise model for a filter.

        Args:
            filter_code (str): Filter code identifying the required model.

        Returns:
            CorrelatedNoiseModel: The correlated-noise model for the filter.
        """
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

    @timed("PhotometricImager.generate_images")
    def generate_images(
        self,
        photometry,
        fov,
        img_type,
        kernel,
        kernel_threshold,
        nthreads,
        emitter,
        cosmo,
    ):
        """Generate an image collection for one emitter.

        Args:
            photometry (PhotometryCollection): Photometry to project into the
                output images.
            fov (unyt_quantity/tuple, unyt_quantity): Width of the image.
            img_type (str): The type of image to create.
            kernel (np.ndarray, optional): Kernel used for smoothed particle
                imaging.
            kernel_threshold (float): Kernel impact-parameter threshold.
            nthreads (int): Number of threads to use for particle smoothing.
            emitter (Component): Emitter supplying geometry and source data.
            cosmo (astropy.cosmology.Cosmology, optional): Cosmology used for
                angular-image coordinate conversions.

        Returns:
            ImageCollection: The generated image collection.
        """
        # Delegate the low-level image construction to the shared generator so
        # the instrument owns the public imaging entry point
        return _generate_image_collection_generic(
            instrument=self,
            photometry=photometry,
            fov=fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
            emitter=emitter,
            cosmo=cosmo,
        )

    @timed("PhotometricImager.apply_psf")
    def apply_psf(
        self, image, filter_code, psf_resample_factor=1, inplace=False
    ):
        """Apply the configured PSF to one image.

        Args:
            image (Image): Image to which the PSF should be applied.
            filter_code (str): Filter code identifying which PSF to use.
            psf_resample_factor (int): Factor by which to supersample the image
                before convolution and downsample it afterwards.
            inplace (bool): If ``True`` update ``image`` directly and return
                it. Otherwise return a new image.

        Returns:
            Image: New image with the PSF applied.

        Raises:
            MissingArgument: If the instrument has no PSFs configured.
            InconsistentArguments: If no PSF is defined for ``filter_code``.
                Also raised if ``psf_resample_factor`` is smaller than 1.
        """
        # Ensure the instrument actually has PSFs to apply
        if self.psfs is None:
            raise exceptions.MissingArgument(
                "No PSFs are set on this Instrument. Provide psfs when "
                "constructing the Instrument."
            )

        # Ensure the requested filter has a PSF definition
        if filter_code not in self.psfs:
            raise exceptions.InconsistentArguments(
                f"No PSF found for filter '{filter_code}'. Available filters: "
                f"{list(self.psfs.keys())}"
            )

        # Resampling factors smaller than one do not make sense for this PSF
        # application path because the image is optionally supersampled first
        if psf_resample_factor < 1:
            raise exceptions.InconsistentArguments(
                "psf_resample_factor must be greater than or equal to 1."
            )

        # Work on the original image when requested, otherwise create a fresh
        # container so the instrument owns the post-processing semantics
        if inplace:
            working_image = image
        else:
            working_image = Image(
                resolution=image.resolution,
                fov=image.fov,
                img=image.img,
            )

        # If requested, temporarily supersample before the PSF convolution
        if psf_resample_factor > 1:
            working_image.resample(psf_resample_factor)

        # Perform the PSF convolution on the instrument side rather than on the
        # image container so the observation machinery stays instrument-owned
        convolved_img = signal.fftconvolve(
            working_image.arr,
            self.psfs[filter_code],
            mode="same",
        )

        # Reapply units if the image carries them
        if working_image.units is not None:
            convolved_img *= working_image.units

        # Update the chosen image container with the convolved pixels
        working_image.img = convolved_img

        # Return to the original resolution if we temporarily supersampled
        if psf_resample_factor > 1:
            working_image.downsample(1 / psf_resample_factor)

        return working_image

    @timed("PhotometricImager.apply_psfs")
    def apply_psfs(
        self, image_collection, psf_resample_factor=1, inplace=False
    ):
        """Apply the configured PSFs to an image collection.

        Args:
            image_collection (ImageCollection): Collection to which PSFs should
                be applied.
            psf_resample_factor (int): Factor by which to supersample the
                images before convolution and downsample them afterwards.
            inplace (bool): If ``True`` update ``image_collection`` directly
                and return it. Otherwise return a new image collection.

        Returns:
            ImageCollection: New image collection with PSFs applied.

        Raises:
            InconsistentArguments: If ``psf_resample_factor`` is smaller than
                1.
        """
        # Resampling factors smaller than one do not make sense for this PSF
        # application path because the images are optionally supersampled first
        if psf_resample_factor < 1:
            raise exceptions.InconsistentArguments(
                "psf_resample_factor must be greater than or equal to 1."
            )

        # Work on the original collection when requested, otherwise construct a
        # fresh collection to hold the convolved images.
        if inplace:
            working_collection = image_collection
            target_imgs = working_collection.imgs
        else:
            target_imgs = {}
            for filter_code in image_collection.filter_codes:
                image = image_collection.imgs[filter_code]
                target_imgs[filter_code] = Image(
                    resolution=image.resolution,
                    fov=image.fov,
                    img=image.img,
                )
            working_collection = ImageCollection(
                resolution=image_collection.resolution,
                fov=image_collection.fov,
                imgs=target_imgs,
            )

        # Apply the PSF to each image through the single-image helper so the
        # instrument owns one canonical PSF-convolution path.
        for filter_code in image_collection.filter_codes:
            target_imgs[filter_code] = self.apply_psf(
                target_imgs[filter_code],
                filter_code,
                psf_resample_factor=psf_resample_factor,
                inplace=True,
            )

        return working_collection

    @timed("PhotometricImager.apply_noise")
    def apply_noise(
        self,
        image,
        filter_code,
        correct_periodicity=True,
        rng_seed=None,
        aperture_radius=None,
    ):
        """Apply the configured imaging noise to one image.

        Args:
            image (Image): Image to which noise should be applied.
            filter_code (str): Filter code identifying which noise definition
                to use.
            correct_periodicity (bool): Whether to apply periodicity
                correction when generating correlated noise.
            rng_seed (int, optional): Seed used for stochastic noise
                generation.
            aperture_radius (unyt_quantity, optional): Aperture radius used by
                SNR/depth-based noise generation.

        Returns:
            Image: New image with noise applied.
        """
        # Apply a fixed noise array directly if one has been configured
        if self.noise_maps is not None:
            noise_arr = self.noise_maps[filter_code]
            return image.apply_noise_array(noise_arr)

        # Delegate correlated-noise generation to the image primitive while the
        # instrument remains the owner of the noise-model configuration
        if self.correlated_noise_models is not None:
            return image.apply_correlated_noise(
                self,
                filter_code,
                correct_periodicity=correct_periodicity,
                rng_seed=rng_seed,
            )

        # Derive image noise from depth and SNR definitions when that is the
        # configured photometric noise model
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

    @timed("PhotometricImager.apply_noises")
    def apply_noises(
        self,
        image_collection,
        correct_periodicity=True,
        rng_seed=None,
        aperture_radius=None,
    ):
        """Apply the configured imaging noise to an image collection.

        Args:
            image_collection (ImageCollection): Collection to which noise
                should be applied.
            correct_periodicity (bool): Whether to apply periodicity
                correction when generating correlated noise.
            rng_seed (int, optional): Seed used for stochastic noise
                generation.
            aperture_radius (unyt_quantity, optional): Aperture radius used by
                SNR/depth-based noise generation.

        Returns:
            ImageCollection: New image collection with noise applied.
        """
        noisy_imgs = {}
        for f in image_collection.filter_codes:
            if rng_seed is None:
                filter_rng_seed = None
            else:
                seed_material = f"{rng_seed}:{f}".encode("utf-8")
                filter_rng_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:4], "big"
                )

            # Delegate the per-image noise policy to the single-image helper so
            # there is one canonical decision path for imaging noise
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

    @timed("PhotometricImager.to_hdf5")
    def to_hdf5(self, group):
        """Write the photometric imager to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised.
        """
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
    @timed("PhotometricImager.load")
    def load(cls, filepath=None, **kwargs):
        """Load a photometric imager from an HDF5 file.

        Args:
            filepath (str or PathLike, optional): Path to the HDF5 file. If
                omitted, subclasses may provide a cached default path.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            PhotometricImager: The loaded instrument.
        """
        if filepath is None:
            filepath = getattr(cls, "_instrument_cache_file", None)
        if filepath is None:
            raise exceptions.MissingArgument(
                f"{cls.__name__}.load requires a filepath."
            )
        with h5py.File(filepath, "r") as hdf:
            return cls._from_hdf5(hdf, **kwargs)

    @classmethod
    @timed("PhotometricImager._from_hdf5")
    def _from_hdf5(cls, group, **kwargs):
        """Load a photometric imager from an HDF5 group.

        Args:
            group (h5py.Group): Group containing the serialised instrument.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            PhotometricImager: The loaded instrument.
        """
        filters = FilterCollection._from_hdf5(group["Filters"])
        resolution = unyt_array(
            group["Resolution"][...], group["Resolution"].attrs["units"]
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
        else:
            psfs = None

        if "NoiseMaps" in group and isinstance(group["NoiseMaps"], h5py.Group):
            noise_maps = {
                key: unyt_array(value[...], value.attrs["units"])
                for key, value in group["NoiseMaps"].items()
            }
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
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "psfs": psfs,
            "noise_maps": noise_maps,
            "noise_source_maps": noise_source_maps,
        }
        payload.update(kwargs)

        init_params = inspect.signature(cls.__init__).parameters

        if "filters" not in init_params or "resolution" not in init_params:
            return cls(
                label=payload["label"],
                filter_lams=payload["filters"].lam,
                depth=payload["depth"],
                depth_app_radius=payload["depth_app_radius"],
                snrs=payload["snrs"],
                psfs=payload["psfs"],
                noise_maps=payload["noise_maps"],
                noise_source_maps=payload["noise_source_maps"],
                filter_subset=tuple(payload["filters"].filter_codes),
            )

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
