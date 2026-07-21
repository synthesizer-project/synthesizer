"""Specialised emission line imaging instrument.

This instrument is designed to hold the attributes required to turn
per-line luminosities/fluxes into resolved images. Unlike photometric
imaging, there is no filter transmission curve involved: each emission
line is projected directly into its own image, keyed by line id.
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
    _generate_line_image_collection_generic,
)
from synthesizer.instruments.instrument_base import (
    InstrumentBase,
    _hashable_state,
)
from synthesizer.instruments.photometric_noise import CorrelatedNoiseModel
from synthesizer.units import accepts
from synthesizer.utils.operation_timers import timed


class LineImager(InstrumentBase):
    """Emission line imaging instrument class.

    A class containing the attributes and methods required to produce
    resolved emission line images. It holds the set of line ids the
    instrument images together with the spatial resolution, optional PSFs,
    and optional noise definitions, each keyed by line id rather than by
    filter code.

    Attributes:
        line_ids (list): The ids of the emission lines this instrument
            images.
        resolution (unyt_array): The spatial resolution of the instrument, in
            kpc or arcseconds.
        psfs (dict, optional): An optional dictionary of point spread
            functions, with one entry per line id.
        noise_maps (dict, optional): An optional dictionary of fixed noise
            maps to apply directly to images, with one entry per line id.
        noise_source_maps (dict, optional): An optional dictionary of source
            maps used to generate correlated-noise models, with one entry per
            line id.
        depth (dict or unyt_quantity, optional): The depth of the instrument.
            If depths are provided per line, this should be a dictionary
            keyed by line id.
        depth_app_radius (unyt_quantity, optional): The aperture radius for
            the depth measurement. If this is omitted but SNRs and depths are
            provided, the depth is assumed to be a point-source depth.
        snrs (dict or unyt_quantity, optional): The signal-to-noise ratios of
            the instrument. If values are provided per line, this should be a
            dictionary keyed by line id.
    """

    @accepts(resolution=(kpc, arcsecond))
    @timed("LineImager.__init__")
    def __init__(
        self,
        label,
        line_ids,
        resolution,
        psfs=None,
        psf_resample_factor=1,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        """Initialise a line imager.

        Args:
            label (str): A label for the instrument.
            line_ids (list): The ids of the emission lines this instrument
                images.
            resolution (unyt_array): The spatial resolution of the
                instrument, in kpc or arcseconds.
            psfs (dict, optional): An optional dictionary of point spread
                functions, with one entry per line id.
            psf_resample_factor (int, optional): Instrument-owned PSF
                supersampling factor. When greater than 1, images are
                temporarily resampled by this factor before PSF convolution
                and then downsampled back to their original resolution.
            depth (dict or unyt_quantity, optional): The depth of the
                instrument. If depths are provided per line, this should be a
                dictionary keyed by line id.
            depth_app_radius (unyt_quantity, optional): The aperture radius
                for the depth measurement. If this is omitted but SNRs and
                depths are provided, the depth is assumed to be a
                point-source depth.
            snrs (dict or unyt_quantity, optional): The signal-to-noise
                ratios of the instrument. If values are provided per line,
                this should be a dictionary keyed by line id.
            noise_maps (dict, optional): An optional dictionary of fixed
                noise maps to apply directly to images, with one entry per
                line id.
            noise_source_maps (dict, optional): An optional dictionary of
                source maps used to generate correlated-noise models, with
                one entry per line id.
        """
        super().__init__(label)

        # Set the line imager specific attributes
        self.line_ids = [str(line_id) for line_id in line_ids]
        self.resolution = resolution
        self.psfs = psfs
        self.psf_resample_factor = psf_resample_factor
        self.depth = depth
        self.depth_app_radius = depth_app_radius
        self.snrs = snrs
        self.noise_maps = noise_maps
        self.noise_source_maps = noise_source_maps
        self.correlated_noise_models = self._build_correlated_noise_models()

        # Validate the instrument configuration
        self._validate()

    @timed("LineImager._validate")
    def _validate(self):
        """Validate the instrument attributes.

        Raises:
            MissingArgument: If any required attributes are missing.
        """
        if len(self.line_ids) == 0:
            raise exceptions.MissingArgument(
                "LineImager requires at least one line id."
            )

        if self.resolution is None:
            raise exceptions.MissingArgument(
                "LineImager requires a resolution."
            )

        if self.psf_resample_factor < 1:
            raise exceptions.InconsistentArguments(
                "psf_resample_factor must be greater than or equal to 1."
            )

        # Depths only make sense when paired with SNR definitions
        if self.depth is not None and self.snrs is None:
            raise exceptions.MissingArgument(
                "If you set a depth you must also set the SNRs"
            )
        if self.snrs is not None and self.depth is None:
            raise exceptions.MissingArgument(
                "If you set a SNR you must also set the depth"
            )

        # Noise maps are an alternative noise definition to depth+SNR pairs
        if self.snrs is not None and self.noise_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as noise maps"
            )

        # Correlated-noise source maps are also an alternative noise
        # definition
        if self.snrs is not None and self.noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as "
                "noise source maps"
            )
        if self.noise_maps is not None and self.noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set fixed noise maps and correlated noise "
                "source maps at the same time"
            )

        # PSFs and noise definitions are always looked up by line id, so keep
        # that configuration strict here and reject partial dictionaries at
        # construction time.
        for attr_name in (
            "psfs",
            "noise_maps",
            "noise_source_maps",
        ):
            payload = getattr(self, attr_name)
            if payload is None:
                continue
            missing_lines = set(self.line_ids) - set(payload)
            if len(missing_lines) > 0:
                raise exceptions.MissingArgument(
                    f"{attr_name} is missing entries for lines: "
                    f"{sorted(missing_lines)}"
                )

        for attr_name in ("depth", "snrs"):
            payload = getattr(self, attr_name)
            if not isinstance(payload, dict):
                continue
            missing_lines = set(self.line_ids) - set(payload)
            if len(missing_lines) > 0:
                raise exceptions.MissingArgument(
                    f"{attr_name} is missing entries for lines: "
                    f"{sorted(missing_lines)}"
                )

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "line_imager"

    @property
    def can_do_line_imaging(self):
        """Return whether this instrument supports line imaging."""
        return True

    @property
    def can_do_psf_line_imaging(self):
        """Return whether this instrument supports PSF line imaging."""
        return self.psfs is not None

    @property
    def can_do_noisy_line_imaging(self):
        """Return whether this instrument supports noisy line imaging."""
        have_noise = self.noise_maps is not None
        have_noise |= self.noise_source_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None
        return have_noise

    @timed("LineImager._build_correlated_noise_models")
    def _build_correlated_noise_models(self):
        """Build per-line correlated-noise models from source maps.

        Returns:
            dict or None: Mapping of line ids to correlated-noise models, or
                None if no source maps are configured.

        Raises:
            InconsistentArguments: If ``noise_source_maps`` is not a dict.
        """
        if self.noise_source_maps is None:
            return None

        if not isinstance(self.noise_source_maps, dict):
            raise exceptions.InconsistentArguments(
                "noise_source_maps must be a dict keyed by line id for "
                "correlated noise generation."
            )

        return {
            line_id: CorrelatedNoiseModel(noise_map)
            for line_id, noise_map in self.noise_source_maps.items()
        }

    @timed("LineImager._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the line imaging comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        return (
            _hashable_state(tuple(self.line_ids)),
            _hashable_state(self.resolution),
            _hashable_state(self.psfs),
            _hashable_state(self.psf_resample_factor),
            _hashable_state(self.depth),
            _hashable_state(self.depth_app_radius),
            _hashable_state(self.snrs),
            _hashable_state(self.noise_maps),
            _hashable_state(self.noise_source_maps),
        )

    @timed("LineImager.get_correlated_noise_model")
    def get_correlated_noise_model(self, line_id):
        """Return the correlated-noise model for a line.

        Args:
            line_id (str): Line id identifying the required model.

        Returns:
            CorrelatedNoiseModel: The correlated-noise model for the line.
        """
        if self.correlated_noise_models is None:
            raise exceptions.MissingArgument(
                "No correlated noise models are set on this Instrument. "
                "Provide noise_source_maps when constructing the Instrument."
            )

        if line_id not in self.correlated_noise_models:
            raise exceptions.InconsistentArguments(
                f"No correlated noise model found for line '{line_id}'. "
                f"Available lines: {list(self.correlated_noise_models)}"
            )

        return self.correlated_noise_models[line_id]

    @timed("LineImager.generate_images")
    def generate_images(
        self,
        lines,
        fov,
        img_type,
        kernel,
        kernel_threshold,
        nthreads,
        emitter,
        cosmo,
        quantity="luminosity",
    ):
        """Generate a line image collection for one emitter.

        Args:
            lines (LineCollection): Lines to project into the output images.
            fov (unyt_quantity/tuple, unyt_quantity): Width of the image.
            img_type (str): The type of image to create.
            kernel (np.ndarray, optional): Kernel used for smoothed particle
                imaging.
            kernel_threshold (float): Kernel impact-parameter threshold.
            nthreads (int): Number of threads to use for particle smoothing.
            emitter (Component): Emitter supplying geometry and source data.
            cosmo (astropy.cosmology.Cosmology, optional): Cosmology used for
                angular-image coordinate conversions.
            quantity (str): Either "luminosity" or "flux", selecting which
                LineCollection attribute is imaged.

        Returns:
            ImageCollection: The generated image collection, one Image per
                requested line id.
        """
        return _generate_line_image_collection_generic(
            instrument=self,
            lines=lines,
            fov=fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
            emitter=emitter,
            cosmo=cosmo,
            quantity=quantity,
        )

    @timed("LineImager.apply_psf")
    def apply_psf(self, image, line_id, inplace=False):
        """Apply the configured PSF to one image.

        Args:
            image (Image): Image to which the PSF should be applied.
            line_id (str): Line id identifying which PSF to use.
            inplace (bool): If ``True`` update ``image`` directly and return
                it. Otherwise return a new image.

        Returns:
            Image: New image with the PSF applied.

        Raises:
            MissingArgument: If the instrument has no PSFs configured.
            InconsistentArguments: If no PSF is defined for ``line_id``.
        """
        if self.psfs is None:
            raise exceptions.MissingArgument(
                "No PSFs are set on this Instrument. Provide psfs when "
                "constructing the Instrument."
            )

        if line_id not in self.psfs:
            raise exceptions.InconsistentArguments(
                f"No PSF found for line '{line_id}'. Available lines: "
                f"{list(self.psfs.keys())}"
            )

        psf_resample_factor = self.psf_resample_factor
        if psf_resample_factor < 1:
            raise exceptions.InconsistentArguments(
                "psf_resample_factor must be greater than or equal to 1."
            )

        if inplace:
            working_image = image
        else:
            working_image = Image(
                resolution=image.resolution,
                fov=image.fov,
                img=image.img,
            )

        if psf_resample_factor > 1:
            working_image.resample(psf_resample_factor)

        convolved_img = signal.fftconvolve(
            working_image.arr,
            self.psfs[line_id],
            mode="same",
        )

        if working_image.units is not None:
            convolved_img *= working_image.units

        working_image.img = convolved_img

        if psf_resample_factor > 1:
            working_image.downsample(1 / psf_resample_factor)

        return working_image

    @timed("LineImager.apply_psfs")
    def apply_psfs(self, image_collection, inplace=False):
        """Apply the configured PSFs to an image collection.

        Args:
            image_collection (ImageCollection): Collection to which PSFs
                should be applied.
            inplace (bool): If ``True`` update ``image_collection`` directly
                and return it. Otherwise return a new image collection.

        Returns:
            ImageCollection: New image collection with PSFs applied.

        Raises:
            InconsistentArguments: If ``psf_resample_factor`` is smaller than
                1.
        """
        psf_resample_factor = self.psf_resample_factor
        if psf_resample_factor < 1:
            raise exceptions.InconsistentArguments(
                "psf_resample_factor must be greater than or equal to 1."
            )

        if inplace:
            working_collection = image_collection
            target_imgs = working_collection.imgs
        else:
            target_imgs = {}
            for line_id in image_collection.keys():
                image = image_collection.imgs[line_id]
                target_imgs[line_id] = Image(
                    resolution=image.resolution,
                    fov=image.fov,
                    img=image.img,
                )
            working_collection = ImageCollection(
                resolution=image_collection.resolution,
                fov=image_collection.fov,
                imgs=target_imgs,
            )

        for line_id in list(image_collection.keys()):
            target_imgs[line_id] = self.apply_psf(
                target_imgs[line_id],
                line_id,
                inplace=True,
            )

        return working_collection

    @timed("LineImager.apply_noise")
    def apply_noise(
        self,
        image,
        line_id,
        correct_periodicity=True,
        rng_seed=None,
        aperture_radius=None,
    ):
        """Apply the configured imaging noise to one image.

        Args:
            image (Image): Image to which noise should be applied.
            line_id (str): Line id identifying which noise definition to use.
            correct_periodicity (bool): Whether to apply periodicity
                correction when generating correlated noise.
            rng_seed (int, optional): Seed used for stochastic noise
                generation.
            aperture_radius (unyt_quantity, optional): Aperture radius used by
                SNR/depth-based noise generation.

        Returns:
            Image: New image with noise applied.
        """
        if self.noise_maps is not None:
            if line_id not in self.noise_maps:
                raise exceptions.MissingArgument(
                    f"noise_maps is missing an entry for line '{line_id}'."
                )
            noise_arr = self.noise_maps[line_id]
            return image.apply_noise_array(noise_arr)

        if self.correlated_noise_models is not None:
            return image.apply_correlated_noise(
                self,
                line_id,
                correct_periodicity=correct_periodicity,
                rng_seed=rng_seed,
            )

        if self.snrs is not None and self.depth is not None:
            snr = (
                self.snrs[line_id]
                if isinstance(self.snrs, dict)
                else self.snrs
            )
            depth = (
                self.depth[line_id]
                if isinstance(self.depth, dict)
                else self.depth
            )
            return image.apply_noise_from_snr(
                snr=snr, depth=depth, aperture_radius=aperture_radius
            )

        raise exceptions.MissingArgument(
            "The instrument has no imaging noise configuration."
        )

    @timed("LineImager.apply_noises")
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
        for line_id in image_collection.keys():
            if rng_seed is None:
                line_rng_seed = None
            else:
                seed_material = f"{rng_seed}:{line_id}".encode("utf-8")
                line_rng_seed = int.from_bytes(
                    hashlib.sha256(seed_material).digest()[:4], "big"
                )

            noisy_imgs[line_id] = self.apply_noise(
                image_collection.imgs[line_id],
                line_id,
                correct_periodicity=correct_periodicity,
                rng_seed=line_rng_seed,
                aperture_radius=aperture_radius,
            )

        return ImageCollection(
            resolution=image_collection.resolution,
            fov=image_collection.fov,
            imgs=noisy_imgs,
        )

    @timed("LineImager.to_hdf5")
    def to_hdf5(self, group):
        """Write the line imager to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised.
        """
        group.attrs["label"] = self.label
        group.attrs["instrument_type"] = self.instrument_type
        group.attrs["line_ids"] = self.line_ids

        ds = group.create_dataset(
            "Resolution", data=self.resolution.value, dtype=float
        )
        ds.attrs["units"] = str(self.resolution.units)

        if self.psfs is not None:
            psfs_group = group.create_group("PSFs")
            for key, value in self.psfs.items():
                ds = psfs_group.create_dataset(key, data=value, dtype=float)
                ds.attrs["units"] = "dimensionless"

        ds = group.create_dataset(
            "PSFResampleFactor",
            data=self.psf_resample_factor,
            dtype=int,
        )
        ds.attrs["units"] = "dimensionless"

        if self.depth is not None:
            if isinstance(self.depth, dict):
                depth_group = group.create_group("Depth")
                for key, value in self.depth.items():
                    ds = depth_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = str(value.units)
            else:
                ds = group.create_dataset(
                    "Depth", data=self.depth.value, dtype=float
                )
                ds.attrs["units"] = str(self.depth.units)

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
                    ds.attrs["units"] = str(value.units)
            else:
                ds = group.create_dataset(
                    "SNRs", data=self.snrs.value, dtype=float
                )
                ds.attrs["units"] = str(self.snrs.units)

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
    @timed("LineImager.load")
    def load(cls, filepath=None, **kwargs):
        """Load a line imager from an HDF5 file.

        Args:
            filepath (str or PathLike, optional): Path to the HDF5 file.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            LineImager: The loaded instrument.
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
    @timed("LineImager._from_hdf5")
    def _from_hdf5(cls, group, **kwargs):
        """Load a line imager from an HDF5 group.

        Args:
            group (h5py.Group): Group containing the serialised instrument.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            LineImager: The loaded instrument.
        """
        line_ids = [str(lid) for lid in group.attrs["line_ids"]]
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
            psfs = {key: group["PSFs"][key][...] for key in group["PSFs"]}
        else:
            psfs = None

        if "PSFResampleFactor" in group:
            psf_resample_factor = int(group["PSFResampleFactor"][...])
        else:
            psf_resample_factor = 1

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
            "line_ids": line_ids,
            "resolution": resolution,
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "psfs": psfs,
            "psf_resample_factor": psf_resample_factor,
            "noise_maps": noise_maps,
            "noise_source_maps": noise_source_maps,
        }
        payload.update(kwargs)

        init_params = inspect.signature(cls.__init__).parameters
        return cls(**{k: v for k, v in payload.items() if k in init_params})
