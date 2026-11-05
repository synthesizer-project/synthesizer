"""Specialised Integrated Field Unit instrument.

This instrument is designed to hold the attributes required by resolved
spectroscopy. It extends :class:`SpectroscopicInstrument` with spatial
resolution, optional PSFs, and noise definitions.
"""

import h5py
import numpy as np
from scipy import signal
from unyt import angstrom, arcsecond, kpc, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.imaging import SpectralCube
from synthesizer.imaging.image_generators import _standardize_imaging_units
from synthesizer.instruments.instrument_base import _hashable_state
from synthesizer.instruments.spectroscopic_instrument import (
    SpectroscopicInstrument,
)
from synthesizer.instruments.utils import (
    read_instrument_attribute,
    write_instrument_attribute,
)
from synthesizer.units import accepts, unyt_to_ndview
from synthesizer.utils.operation_timers import timed


class IntegratedFieldUnit(SpectroscopicInstrument):
    """Integrated Field Unit instrument class.

    A class containing the attributes and methods required to produce resolved
    spectroscopy. It extends :class:`SpectroscopicInstrument` with spatial
    resolution, optional PSFs, and noise definitions.

    Attributes:
        resolution (unyt_array): The spatial resolution of the instrument, in
            kpc or arcseconds.
        psfs (array): An optional array with spatial point spread functions as
            a function of wavelength, in dimensionless units. If a 2D array is
            supplied, every wavelength is assumed to have the same PSF. If
            a 3D array is supplied, the last axis must be the wavelength axis.
    """

    @accepts(
        resolution=(kpc, arcsecond),
        psf_pixel_scale=(kpc, arcsecond),
        lam=angstrom,
        depth_app_radius=(kpc, arcsecond),
    )
    @timed("IntegratedFieldUnit.__init__")
    def __init__(
        self,
        label,
        lam,
        resolution,
        psfs=None,
        psf_pixel_scale=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        noise_maps=None,
        noise_source_maps=None,
        resolving_power=None,
    ):
        """Initialise an integrated field unit instrument.

        Args:
            label (str): A label for the instrument.
            lam (unyt_array): The wavelength array defining the spectral
                coverage of the instrument.
            resolution (unyt_array): The spatial resolution of the instrument,
                in kpc or arcseconds.
            psfs (array, optional): An optional array with spatial point spread
                functions as a function of wavelength, in dimensionless units.
                If a 2D array is supplied, every wavelength is assumed to have
                the same PSF. If a 3D array is supplied, the last axis must be
                the wavelength axis.
            psf_pixel_scale (unyt_quantity, optional): Spatial pixel scale of
                the supplied PSF kernels. Must have units compatible with the
                spectral cube resolution.
            depth (unyt_quantity, optional): The depth of the instrument, in
                the same units as the image surface brightness.
            depth_app_radius (unyt_quantity, optional): The aperture radius for
                the depth measurement, in resolution units.
            snrs (unyt_quantity, optional): The signal-to-noise ratio of the
                instrument, in dimensionless units.
            noise_maps (array, optional): An optional array with noise map as
                a function of wavelength, in the same units as the image noise.
                If a 2D array is supplied, every wavelength is assumed to have
                the same noise map. If a 3D array is supplied, the last axis
                must be the wavelength axis.
            noise_source_maps (array, optional): An optional array or mapping
                carrying source noise templates for future correlated-noise IFU
                machinery.
            resolving_power (float or callable, optional): Spectral resolving
                power (R = lambda / delta_lambda) of the instrument. Only a
                constant value is persisted when the instrument is saved to
                HDF5.
        """
        # Initialise the shared spectroscopic instrument first
        super().__init__(
            label=label,
            lam=lam,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            noise_maps=noise_maps,
            resolving_power=resolving_power,
        )

        # Attach the IFU specific attributes
        self.resolution = resolution
        self.psfs = psfs
        self.psf_pixel_scale = psf_pixel_scale
        self.noise_source_maps = noise_source_maps

        # Ensure we have been handed the correct information
        self._validate()

    @timed("IntegratedFieldUnit._validate")
    def _validate(self):
        """Validate the instrument attributes.

        Raises:
            MissingArgument: If any required attributes are missing.
        """
        # Perform the shared validation first
        super()._validate()

        # Ensure we actually have the resolution... otherwise we are not
        # really an IFU!
        if self.resolution is None:
            raise exceptions.MissingArgument(
                "IntegratedFieldUnit requires a resolution."
            )

        if self.psfs is not None and self.psf_pixel_scale is None:
            raise exceptions.MissingArgument(
                "A PSF pixel scale must be provided when PSFs are set."
            )

        if self.psf_pixel_scale is not None:
            if not np.all(np.isfinite(self.psf_pixel_scale.value)):
                raise exceptions.InconsistentArguments(
                    "psf_pixel_scale must be finite."
                )

            if np.any(self.psf_pixel_scale <= 0 * self.psf_pixel_scale.units):
                raise exceptions.InconsistentArguments(
                    "psf_pixel_scale must be greater than zero."
                )

        # Correlated-noise source maps are an alternative future noise
        # definition to depth+SNR pairs.
        if (
            self.depth is not None or self.snrs is not None
        ) and self.noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set depths and SNRs at the same time as "
                "noise source maps"
            )

        # Fixed noise maps and source-noise templates are mutually exclusive.
        if self.noise_maps is not None and self.noise_source_maps is not None:
            raise exceptions.MissingArgument(
                "You cannot set fixed noise maps and correlated noise source "
                "maps at the same time"
            )

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "ifu"

    @property
    def can_do_resolved_spectroscopy(self):
        """Return whether this instrument supports resolved spectroscopy."""
        return True

    @property
    def can_do_psf_spectroscopy(self):
        """Return whether this instrument supports PSF spectroscopy."""
        return self.psfs is not None

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """Return whether this instrument supports noisy IFU work."""
        return False

    @timed("IntegratedFieldUnit._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the IFU comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        return super()._comparison_state() + (
            _hashable_state(self.resolution),
            _hashable_state(self.psfs),
            _hashable_state(self.psf_pixel_scale),
            _hashable_state(self.noise_source_maps),
        )

    @timed("IntegratedFieldUnit.generate_data_cube")
    def generate_data_cube(
        self,
        component,
        fov,
        sed,
        cube_type="smoothed",
        kernel=None,
        kernel_threshold=1,
        quantity="lnu",
        nthreads=1,
        cosmo=None,
    ):
        """Generate a resolved-spectroscopy data cube for one saved spectrum.

        This method is the instrument-owned entry point for IFU cube
        generation. It determines which component owns the requested saved
        spectrum and then constructs the data cube directly using the relevant
        low-level cube-generation path.

        Args:
            component (Component): Component providing the geometry used to
                construct the data cube.
            fov (unyt_quantity): Width of the requested data cube.
            sed (Sed): Saved spectra to turn into a data cube.
            cube_type (str): Either ``"smoothed"`` or ``"hist"``.
            kernel (array-like, optional): Kernel used for smoothed particle
                cubes.
            kernel_threshold (float): Kernel impact-parameter threshold.
            quantity (str): Spectral quantity to store in the cube.
            nthreads (int): Number of threads to use for particle cube
                generation.
            cosmo (astropy.cosmology, optional): Cosmology used for mixed-unit
                conversions.

        Returns:
            SpectralCube: Generated spectral data cube.
        """
        if isinstance(sed, str):
            particle_spectra = getattr(component, "particle_spectra", {})
            spectra = getattr(component, "spectra", {})

            if sed in particle_spectra:
                sed = particle_spectra[sed]
            elif sed in spectra:
                sed = spectra[sed]
            else:
                component_name = getattr(
                    component,
                    "component_type",
                    type(component).__name__,
                )
                raise ValueError(
                    f"Component '{component_name}' has no saved spectrum "
                    f"label '{sed}'."
                )

        if hasattr(component, "morphology"):
            if cube_type != "smoothed":
                raise exceptions.InconsistentArguments(
                    f"Parametric {component.component_type} can only produce "
                    "smoothed data cubes."
                )
            return self._generate_parametric_component_cube(
                component=component,
                sed=sed,
                fov=fov,
                lam=self.lam,
                quantity=quantity,
            )

        return self._generate_particle_component_cube(
            component=component,
            sed=sed,
            fov=fov,
            lam=self.lam,
            cube_type=cube_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            quantity=quantity,
            nthreads=nthreads,
            cosmo=cosmo,
        )

    @timed("IntegratedFieldUnit._generate_particle_component_cube")
    def _generate_particle_component_cube(
        self,
        component,
        sed,
        fov,
        lam,
        cube_type="hist",
        kernel=None,
        kernel_threshold=1,
        quantity="lnu",
        nthreads=1,
        cosmo=None,
    ):
        """Generate a data cube for one particle component.

        Args:
            component (Particles): The particle component providing positions
                and, for smoothed cubes, smoothing lengths.
            sed (Sed): The saved particle spectra to turn into a data cube.
            fov (unyt_quantity): Width of the requested data cube.
            lam (unyt_array): Wavelength array defining the cube sampling.
            cube_type (str): Either ``"smoothed"`` or ``"hist"``.
            kernel (array-like, optional): Kernel used for smoothed particle
                cubes.
            kernel_threshold (float): Kernel impact-parameter threshold.
            quantity (str): Spectral quantity to store in the cube.
            nthreads (int): Number of threads to use for particle cube
                generation.
            cosmo (astropy.cosmology, optional): Cosmology used for mixed-unit
                conversions.

        Returns:
            SpectralCube: Generated spectral data cube for the component.
        """
        needs_smoothing = cube_type == "smoothed"
        resolution, fov, coords, smls = _standardize_imaging_units(
            resolution=self.resolution,
            fov=fov,
            emitter=component,
            cosmo=cosmo,
            include_smoothing_lengths=needs_smoothing,
        )

        cube = SpectralCube(resolution=resolution, fov=fov, lam=lam)

        if cube_type == "hist":
            cube.generate_data_cube_hist(
                sed=sed,
                coordinates=coords,
                quantity=quantity,
                nthreads=nthreads,
            )
            return cube

        if cube_type == "smoothed":
            cube.generate_data_cube_smoothed(
                sed=sed,
                coordinates=coords,
                smoothing_lengths=smls,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                quantity=quantity,
                nthreads=nthreads,
            )
            return cube

        raise exceptions.UnknownImageType(
            "Unknown cube_type %s. (Options are 'hist' or 'smoothed')"
            % cube_type
        )

    @timed("IntegratedFieldUnit._generate_parametric_component_cube")
    def _generate_parametric_component_cube(
        self,
        component,
        sed,
        fov,
        lam,
        quantity="lnu",
    ):
        """Generate a data cube for one parametric component.

        Args:
            component (Component): The parametric component providing the
                morphology density grid.
            sed (Sed): The saved spectra to turn into a data cube.
            fov (unyt_quantity): Width of the requested data cube.
            lam (unyt_array): Wavelength array defining the cube sampling.
            quantity (str): Spectral quantity to store in the cube.

        Returns:
            SpectralCube: Generated spectral data cube for the component.
        """
        cube = SpectralCube(resolution=self.resolution, fov=fov, lam=lam)
        density_grid = component.morphology.get_density_grid(
            self.resolution, cube.npix
        )
        cube.generate_data_cube_smoothed(
            sed=sed,
            density_grid=density_grid,
            quantity=quantity,
        )
        return cube

    @timed("IntegratedFieldUnit.apply_psf_to_cube")
    def apply_psf_to_cube(self, cube, inplace=False):
        """Apply the IFU PSF to a spectral cube.

        Args:
            cube (SpectralCube): Spectral cube to which the PSF should be
                applied.
            inplace (bool): If ``True`` update ``cube`` directly and return
                it. Otherwise return a new cube.

        Returns:
            SpectralCube: New cube with the PSF applied.

        Raises:
            MissingArgument: If the instrument has no PSFs configured.
            InconsistentArguments: If the PSF dimensions or wavelength
                grid are incompatible with the cube.
        """
        # Ensure the spectral cube has been generated.
        if cube.arr is None:
            raise exceptions.MissingIFU(
                "The spectral cube array hasn't been generated yet. Run "
                "generate_data_cube_hist() or generate_data_cube_smoothed() "
                "before applying the PSF."
            )

        # Ensure the instrument actually has PSFs to apply
        if self.psfs is None:
            raise exceptions.MissingArgument(
                "No PSFs are set on this Instrument. Provide psfs when "
                "constructing the Instrument."
            )

        if np.ndim(self.psfs) == 2:
            psf_type = "2D"
        elif np.ndim(self.psfs) == 3:
            psf_type = "3D"
        else:
            raise exceptions.InconsistentArguments(
                "The Instrument PSF must be 2 or 3 dimensional."
            )

        if np.shape(self.psfs)[0] == 0 or np.shape(self.psfs)[1] == 0:
            raise exceptions.InconsistentArguments(
                "The Instrument PSF must have non-empty spatial dimensions."
            )

        if not np.all(np.isfinite(self.psfs)):
            raise exceptions.InconsistentArguments(
                "The Instrument PSF must contain only finite values."
            )

        if psf_type == "3D":
            cube_lam = cube.lam.ndview
            min_channel_spacing = (
                np.min(np.abs(np.diff(cube_lam))) if len(cube_lam) > 1 else 0.0
            )
            if len(self.lam) != len(cube.lam) or not np.allclose(
                unyt_to_ndview(self.lam, cube.lam.units),
                cube_lam,
                rtol=0.0,
                atol=min_channel_spacing * 1e-6,
            ):
                raise exceptions.InconsistentArguments(
                    "The Instrument and cube wavelength grids must match."
                )
            if np.shape(self.psfs)[2] != np.shape(cube.arr)[2]:
                raise exceptions.InconsistentArguments(
                    "The Instrument PSF is specified on a different wavelength"
                    "grid to the data cube. The PSF must be specified at each"
                    "wavelength."
                )

        psf_resample_factor = unyt_to_ndview(
            cube.resolution / self.psf_pixel_scale,
            "dimensionless",
        ).item()

        # Round to an integer if within 0.1%.
        nearest_int = round(psf_resample_factor)

        if np.isclose(
            psf_resample_factor,
            nearest_int,
            rtol=1e-3,
            atol=0,
        ):
            psf_resample_factor = int(nearest_int)

        # Resampling factors smaller than one do not make sense for this PSF
        # application path because the image is optionally supersampled first
        if psf_resample_factor < 1:
            raise exceptions.InconsistentArguments(
                "PSF pixel scale must be no larger than the cube pixel scale."
            )

        # Work on the original image when requested, otherwise create a fresh
        # container so the instrument owns the post-processing semantics
        if inplace:
            working_cube = cube
        else:
            working_cube = SpectralCube(
                resolution=cube.resolution,
                fov=cube.fov,
                lam=cube.lam,
            )
            working_cube.arr = cube.arr.copy()
            working_cube.units = cube.units
            working_cube.sed = cube.sed
            working_cube.quantity = cube.quantity

        # If requested, temporarily supersample before the PSF convolution
        if psf_resample_factor > 1:
            working_cube.resample(psf_resample_factor)

        # Perform the PSF convolution on the instrument side rather than on the
        # image container so the observation machinery stays instrument-owned
        convolved_cube = np.empty_like(working_cube.arr)

        if psf_type == "3D":
            for ii in np.arange(0, np.shape(working_cube.arr)[2]):
                convolved_cube[:, :, ii] = signal.fftconvolve(
                    working_cube.arr[:, :, ii],
                    self.psfs[:, :, ii],
                    mode="same",
                )
        elif psf_type == "2D":
            for ii in np.arange(0, np.shape(working_cube.arr)[2]):
                convolved_cube[:, :, ii] = signal.fftconvolve(
                    working_cube.arr[:, :, ii],
                    self.psfs,
                    mode="same",
                )

        # Update the chosen cube container with the convolved pixels
        working_cube.arr = convolved_cube

        # Return to the original resolution if we temporarily supersampled
        if psf_resample_factor > 1:
            working_cube.downsample(1 / psf_resample_factor)

        return working_cube

    @timed("IntegratedFieldUnit.apply_psf")
    def apply_psf(self, observable, **kwargs):
        """Apply the IFU PSF to an observable.

        Args:
            observable: Observable to which the IFU PSF should be applied.
            **kwargs: Additional arguments passed to the PSF method.

        Returns:
            SpectralCube: Cube with the PSF applied.

        Raises:
            InconsistentArguments: If the observable is not a SpectralCube.
        """
        if isinstance(observable, SpectralCube):
            return self.apply_psf_to_cube(observable, **kwargs)

        raise exceptions.InconsistentArguments(
            "IntegratedFieldUnit.apply_psf only supports SpectralCube objects."
        )

    @timed("IntegratedFieldUnit.apply_noise_to_cube")
    def apply_noise_to_cube(self, cube, **kwargs):
        """Apply IFU noise to a spectral cube.

        This placeholder makes the intended IFU-owned noise behaviour explicit
        even though the underlying cube-noise machinery has not yet been
        implemented.

        Args:
            cube (SpectralCube): Spectral cube to which noise should be
                applied.
            **kwargs: Future keyword arguments for IFU noise application.

        Raises:
            UnimplementedFunctionality: Raised because IFU cube noise
                application is not yet implemented.
        """
        # The IFU should eventually own cube-side noise application, but the
        # required machinery has not yet been implemented.
        raise exceptions.UnimplementedFunctionality(
            "IntegratedFieldUnit.apply_noise_to_cube is not implemented yet."
        )

    @timed("IntegratedFieldUnit.apply_noise")
    def apply_noise(self, observable, **kwargs):
        """Apply IFU noise to an observable.

        This placeholder makes the intended IFU-owned noise behaviour explicit
        even though the underlying resolved-spectroscopy noise machinery has
        not yet been implemented.

        Args:
            observable: Observable to which IFU noise should be applied.
            **kwargs: Future keyword arguments for IFU noise application.

        Raises:
            UnimplementedFunctionality: Raised because IFU noise application is
                not yet implemented.
        """
        # The IFU should eventually own resolved-spectroscopy noise
        # application, but the required machinery has not yet been
        # implemented.
        raise exceptions.UnimplementedFunctionality(
            "IntegratedFieldUnit.apply_noise is not implemented yet."
        )

    @timed("IntegratedFieldUnit.to_hdf5")
    def to_hdf5(self, group):
        """Write the integrated field unit to an HDF5 group.

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
            ds = group.create_dataset("PSFs", data=self.psfs, dtype=float)
            ds.attrs["units"] = "dimensionless"

        if self.psf_pixel_scale is not None:
            ds = group.create_dataset(
                "PSFPixelScale",
                data=self.psf_pixel_scale.value,
                dtype=float,
            )
            ds.attrs["units"] = str(self.psf_pixel_scale.units)

        write_instrument_attribute(
            group, "NoiseSourceMaps", self.noise_source_maps
        )

    @classmethod
    @timed("IntegratedFieldUnit.load")
    def load(cls, filepath, **kwargs):
        """Load an integrated field unit from an HDF5 file.

        Args:
            filepath (str or PathLike): Path to the HDF5 file.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            IntegratedFieldUnit: The loaded instrument.
        """
        with h5py.File(filepath, "r") as hdf:
            return cls._from_hdf5(hdf, **kwargs)

    @classmethod
    @timed("IntegratedFieldUnit._from_hdf5")
    def _from_hdf5(cls, group, **kwargs):
        """Load an integrated field unit from an HDF5 group.

        Args:
            group (h5py.Group): Group containing the serialised instrument.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            IntegratedFieldUnit: The loaded instrument.
        """
        lam = unyt_array(
            group["Wavelength"][...], group["Wavelength"].attrs["units"]
        )
        resolution = unyt_array(
            group["Resolution"][...], group["Resolution"].attrs["units"]
        )

        depth = read_instrument_attribute(group, "Depth")
        depth_app_radius = read_instrument_attribute(
            group, "DepthApertureRadius"
        )
        snrs = read_instrument_attribute(group, "SNRs")

        if "PSFs" in group:
            psfs = unyt_array(group["PSFs"][...], group["PSFs"].attrs["units"])
        else:
            psfs = None

        if "PSFPixelScale" in group:
            psf_pixel_scale = unyt_quantity(
                group["PSFPixelScale"][()],
                group["PSFPixelScale"].attrs["units"],
            )
        else:
            psf_pixel_scale = None

        noise_maps = read_instrument_attribute(group, "NoiseMaps")
        noise_source_maps = read_instrument_attribute(group, "NoiseSourceMaps")

        payload = {
            "label": group.attrs["label"],
            "lam": lam,
            "resolution": resolution,
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "psfs": psfs,
            "psf_pixel_scale": psf_pixel_scale,
            "noise_maps": noise_maps,
            "noise_source_maps": noise_source_maps,
            "resolving_power": group.attrs.get("resolving_power"),
        }
        payload.update(kwargs)

        return cls(
            label=payload["label"],
            lam=payload["lam"],
            resolution=payload["resolution"],
            depth=payload["depth"],
            depth_app_radius=payload["depth_app_radius"],
            snrs=payload["snrs"],
            psfs=payload["psfs"],
            psf_pixel_scale=payload["psf_pixel_scale"],
            noise_maps=payload["noise_maps"],
            noise_source_maps=payload["noise_source_maps"],
            resolving_power=payload["resolving_power"],
        )
