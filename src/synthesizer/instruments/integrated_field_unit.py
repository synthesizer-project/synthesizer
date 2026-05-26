"""Specialised Integrated Field Unit instrument.

This instrument is designed to hold the attributes required by resolved
spectroscopy. It extends :class:`SpectroscopicInstrument` with spatial
resolution, optional PSFs, and noise definitions.
"""

import h5py
from unyt import angstrom, arcsecond, kpc, unyt_array

from synthesizer import exceptions
from synthesizer.imaging import SpectralCube
from synthesizer.imaging.image_generators import _standardize_imaging_units
from synthesizer.instruments.instrument_base import _hashable_state
from synthesizer.instruments.spectroscopic_instrument import (
    SpectroscopicInstrument,
)
from synthesizer.units import accepts
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
        depth=None,
        depth_app_radius=None,
        snrs=None,
        noise_maps=None,
        noise_source_maps=None,
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
        """
        # Initialise the shared spectroscopic instrument first
        super().__init__(
            label=label,
            lam=lam,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            noise_maps=noise_maps,
        )

        # Attach the IFU specific attributes
        self.resolution = resolution
        self.psfs = psfs
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
        return False

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
        if hasattr(component, "particle_spectra") and sed in getattr(
            component, "particle_spectra", {}
        ):
            sed = component.particle_spectra[sed]

        elif sed in getattr(component, "spectra", {}):
            sed = component.spectra[sed]

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
    def apply_psf_to_cube(self, cube, **kwargs):
        """Apply the IFU PSF to a spectral cube.

        This placeholder makes the intended IFU-owned PSF behaviour explicit
        even though the underlying cube PSF machinery has not yet been
        implemented.

        Args:
            cube (SpectralCube): Spectral cube to which the PSF should be
                applied.
            **kwargs: Future keyword arguments for IFU PSF application.

        Raises:
            UnimplementedFunctionality: Raised because IFU PSF application is
                not yet implemented.
        """
        # The IFU should eventually own cube-side PSF application, but the
        # required machinery has not yet been implemented.
        raise exceptions.UnimplementedFunctionality(
            "IntegratedFieldUnit.apply_psf_to_cube is not implemented yet."
        )

    @timed("IntegratedFieldUnit.apply_psf")
    def apply_psf(self, observable, **kwargs):
        """Apply the IFU PSF to an observable.

        This placeholder makes the intended IFU-owned PSF behaviour explicit
        even though the underlying resolved-spectroscopy PSF machinery has not
        yet been implemented.

        Args:
            observable: Observable to which the IFU PSF should be applied.
            **kwargs: Future keyword arguments for IFU PSF application.

        Raises:
            UnimplementedFunctionality: Raised because IFU PSF application is
                not yet implemented.
        """
        # The IFU should eventually own resolved-spectroscopy PSF
        # application, but the required machinery has not yet been
        # implemented.
        raise exceptions.UnimplementedFunctionality(
            "IntegratedFieldUnit.apply_psf is not implemented yet."
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

        if self.noise_source_maps is not None:
            if isinstance(self.noise_source_maps, dict):
                noise_source_group = group.create_group("NoiseSourceMaps")
                for key, value in self.noise_source_maps.items():
                    ds = noise_source_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = str(value.units)
            else:
                ds = group.create_dataset(
                    "NoiseSourceMaps",
                    data=self.noise_source_maps.value,
                    dtype=float,
                )
                ds.attrs["units"] = str(self.noise_source_maps.units)

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

        if "PSFs" in group:
            psfs = unyt_array(group["PSFs"][...], group["PSFs"].attrs["units"])
        else:
            psfs = None

        if "NoiseMaps" in group:
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
        elif "NoiseSourceMaps" in group:
            noise_source_maps = unyt_array(
                group["NoiseSourceMaps"][...],
                group["NoiseSourceMaps"].attrs["units"],
            )
        else:
            noise_source_maps = None

        payload = {
            "label": group.attrs["label"],
            "lam": lam,
            "resolution": resolution,
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "psfs": psfs,
            "noise_maps": noise_maps,
            "noise_source_maps": noise_source_maps,
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
            noise_maps=payload["noise_maps"],
            noise_source_maps=payload["noise_source_maps"],
        )
