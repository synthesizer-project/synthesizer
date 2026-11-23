"""Specialised Integrated Field Unit instrument.

This instrument is designed to hold the attributes required by resolved
spectroscopy. It extends :class:`SpectroscopicInstrument` with spatial
resolution, optional PSFs, and noise definitions.

Cube noise uses the noise models of :class:`SpectroscopicInstrument`, with
each spatial pixel of the cube an independent spectrum.
"""

import h5py
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
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
from synthesizer.synth_warnings import warn
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
        psf_lam (unyt_array): Input wavelength grid of the PSF slices.
            Wavelength-dependent PSFs are interpolated onto ``lam`` during
            initialisation. Defaults to ``lam`` when not explicitly supplied.
        lsfs (array, optional): Normalised line spread function kernels
            with shape (nwave, nkernel), inherited from
            SpectroscopicInstrument. Each row corresponds to a
            wavelength in lam.
        noise_shape_lam, noise_shape_std, template_on_detector: The noise
            template settings, inherited from SpectroscopicInstrument.
        depth_lam, depth_window: Where the depth applies, inherited from
            SpectroscopicInstrument.
        depth_app_radius (unyt_quantity, optional): The aperture radius for
            the depth. If set, depth / snrs is the noise in the aperture and
            is converted to per pixel; otherwise it is the noise per pixel.
    """

    @accepts(
        resolution=(kpc, arcsecond),
        psf_pixel_scale=(kpc, arcsecond),
        psf_lam=angstrom,
        lam=angstrom,
        depth_app_radius=(kpc, arcsecond),
        noise_shape_lam=angstrom,
        depth_lam=angstrom,
        depth_window=angstrom,
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
        lsfs=None,
        psf_lam=None,
        noise_shape_lam=None,
        noise_shape_std=None,
        template_on_detector=True,
        depth_lam=None,
        depth_window=None,
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
            psf_lam (unyt_array, optional): Input wavelength grid of the
                supplied PSF slices. Wavelength-dependent PSFs are
                interpolated onto ``lam`` during initialisation. Defaults to
                ``lam``.
            depth (unyt_quantity, optional): The depth of the instrument,
                the spectral flux or luminosity density reached at
                S/N = snrs, at depth_lam (or over depth_window), in units
                convertible to the cube units.
            depth_app_radius (unyt_quantity, optional): The aperture radius
                for the depth, in resolution units. If set, depth / snrs is
                the noise in the aperture and is converted to per pixel, as
                for imaging; otherwise it is the noise per pixel.
            snrs (float, optional): The signal-to-noise ratio at the depth.
            noise_maps (unyt_array, optional): A fixed noise cube added to
                cubes as it is, with the same shape as the cube (the last
                axis is wavelength) and units convertible to the cube units.
                An alternative to depth and snrs.
            noise_source_maps (array, optional): An optional array or mapping
                carrying source noise templates for future correlated-noise IFU
                machinery.
            resolving_power (float or callable, optional): Spectral resolving
                power (R = lambda / delta_lambda) of the instrument. Only a
                constant value is persisted when the instrument is saved to
                HDF5.
            lsfs (array, optional): Normalised line spread function
                kernels with shape (nwave, nkernel), sampled on the
                instrument wavelength grid.
            noise_shape_lam (unyt_array, optional): Wavelengths of the
                noise template, strictly increasing. Requires depth and snrs.
            noise_shape_std (unyt_array, optional): The noise template,
                the 1-sigma noise per pixel at noise_shape_lam (NaN in
                detector gaps), in units convertible to the cube units. Only
                its shape is used. Defaults to flat noise if not supplied.
            template_on_detector (bool): Whether noise_shape_lam are the
                detector pixel centres, giving correlated noise. If False
                each wavelength element gets independent noise.
            depth_lam (unyt_quantity, optional): The wavelength at which the
                noise equals depth / snrs.
            depth_window (unyt_array, optional): A wavelength window
                (lam_min, lam_max) over whose median the noise equals
                depth / snrs. With a noise template, exactly one of this or
                depth_lam is required; flat noise needs neither.
        """
        # Initialise the shared spectroscopic instrument first
        super().__init__(
            label=label,
            lam=lam,
            depth=depth,
            snrs=snrs,
            noise_maps=noise_maps,
            resolving_power=resolving_power,
            lsfs=lsfs,
            noise_shape_lam=noise_shape_lam,
            noise_shape_std=noise_shape_std,
            template_on_detector=template_on_detector,
            depth_lam=depth_lam,
            depth_window=depth_window,
        )

        # Attach the IFU specific attributes
        self.depth_app_radius = depth_app_radius
        self.resolution = resolution
        self.psfs = psfs
        self.psf_pixel_scale = psf_pixel_scale
        self.psf_lam = self.lam if psf_lam is None else psf_lam
        self.noise_source_maps = noise_source_maps

        # Ensure we have been handed the correct information
        self._validate()
        self._interpolate_psfs_to_lam()

    @timed("IntegratedFieldUnit._interpolate_psfs_to_lam")
    def _interpolate_psfs_to_lam(self):
        """Interpolate wavelength-dependent PSFs onto ``self.lam`` once."""
        if self.psfs is None or np.ndim(self.psfs) != 3:
            return

        if self.psf_lam is None:
            if self.psfs.shape[2] == len(self.lam):
                warn(
                    "PSF wavelengths not provided. Assuming PSF wavelengths "
                    "match the instrument wavelength grid."
                )
                self.psf_lam = self.lam.copy()
                return
            else:
                raise ValueError(
                    "PSF wavelengths are required when the number of PSF "
                    "slices differs from the instrument wavelength grid."
                )

        source_lam = unyt_to_ndview(self.psf_lam, self.lam.units)
        target_lam = self.lam.ndview

        if len(source_lam) != self.psfs.shape[2]:
            raise ValueError(
                "PSF wavelength array must match the PSF cube wavelength axis."
            )

        if len(source_lam) == 0 or np.any(np.diff(source_lam) <= 0):
            raise ValueError("PSF wavelengths must be strictly increasing.")

        # No interpolation if already on the correct wavelength grid
        if np.array_equal(source_lam, target_lam):
            return

        if len(source_lam) == 1:
            warn(
                "Only one PSF wavelength provided. Repeating the PSF "
                "across the instrument wavelength grid."
            )
            interpolated_psfs = np.repeat(self.psfs, len(target_lam), axis=2)
        else:
            # Warn if target wavelengths extend beyond the PSF coverage
            if np.any(
                (target_lam < source_lam[0]) | (target_lam > source_lam[-1])
            ):
                warn(
                    "Instrument wavelengths extend beyond the PSF wavelength "
                    "range. Endpoint PSFs will be used outside this range."
                )

            interpolated_psfs = interp1d(
                source_lam,
                self.psfs,
                axis=2,
                kind="linear",
                bounds_error=False,
                fill_value=(self.psfs[:, :, 0], self.psfs[:, :, -1]),
            )(target_lam)

        self.psfs = interpolated_psfs
        self.psf_lam = self.lam.copy()

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

        if not isinstance(self.psf_lam, unyt_array) or self.psf_lam.ndim != 1:
            raise exceptions.InconsistentArguments(
                "psf_lam must be a one-dimensional wavelength array."
            )

        if len(self.psf_lam) == 0:
            raise exceptions.InconsistentArguments(
                "psf_lam must contain at least one wavelength."
            )

        psf_lam = unyt_to_ndview(self.psf_lam, angstrom)
        if not np.all(np.isfinite(psf_lam)) or np.any(psf_lam <= 0):
            raise exceptions.InconsistentArguments(
                "psf_lam must contain only positive, finite wavelengths."
            )

        if len(psf_lam) > 1 and not np.all(np.diff(psf_lam) > 0):
            raise exceptions.InconsistentArguments(
                "psf_lam must be strictly increasing."
            )

        if self.psfs is not None and np.ndim(self.psfs) == 3:
            if self.psfs.shape[2] != len(self.psf_lam):
                raise exceptions.InconsistentArguments(
                    "The number of psf_lam wavelengths must match the "
                    "number of PSF slices."
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
        return (
            self.depth is not None and self.snrs is not None
        ) or self.noise_maps is not None

    @timed("IntegratedFieldUnit._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the IFU comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        return super()._comparison_state() + (
            _hashable_state(self.depth_app_radius),
            _hashable_state(self.resolution),
            _hashable_state(self.psfs),
            _hashable_state(self.psf_pixel_scale),
            _hashable_state(self.psf_lam),
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

        Wavelength-dependent PSFs are interpolated onto the instrument
        wavelength grid during initialisation.

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
                grid are incompatible with the instrument.
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
            if self.psfs.shape[2] != len(self.psf_lam):
                raise exceptions.InconsistentArguments(
                    "The number of PSF wavelength planes must match `psf_lam`."
                )

            # Wavelength-dependent PSFs are interpolated onto the instrument
            # grid during initialisation, so indexing them by cube channel is
            # only valid when the cube uses that same wavelength grid.
            cube_lam = unyt_to_ndview(cube.lam, self.psf_lam.units)
            if not np.array_equal(cube_lam, self.psf_lam.ndview):
                raise exceptions.InconsistentArguments(
                    "The cube wavelength grid must match the PSF wavelength "
                    "grid for a wavelength-dependent PSF."
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
    def apply_noise_to_cube(
        self,
        cube,
        rng_seed=None,
        inplace=False,
        simspin_snr=None,
    ):
        """Apply IFU noise to a spectral cube.

        The noise follows the SimSpin calculation if simspin_snr is given,
        otherwise from the provided noise maps (a fixed realisation added
        as it is), or calculated from the depth and SNR. For depth+SNR and
        SimSpin noise each spatial pixel is an independent spectrum, with
        noise from the noise models of SpectroscopicInstrument. For
        depth+SNR noise with depth_app_radius set, the depth is an aperture
        depth and the noise is converted to per pixel.

        Args:
            cube (SpectralCube): Spectral cube to which noise should be
                applied.
            rng_seed (int, optional): Seed for the noise realisation. None
                gives a fresh realisation.
            inplace (bool): If ``True`` update ``cube`` directly and return
                it. Otherwise return a new cube.
            simspin_snr (float, optional): The S/N of the median-flux pixel
                for SimSpin noise.

        Returns:
            SpectralCube: Cube with noise applied, with noise_arr (the noise
                added, unyt_array) and weight_map (the inverse variance of
                each voxel; 0 where it is undefined) attached.

        Raises:
            MissingIFU: If the cube has not been generated.
            MissingArgument: If there is no noise model to apply.
            InconsistentArguments: If the cube has no units, or the noise
                maps don't match the cube.
            UnimplementedFunctionality: If noise source maps are set.
        """
        # Ensure the spectral cube has been generated
        if cube.arr is None:
            raise exceptions.MissingIFU(
                "The spectral cube array hasn't been generated yet. Run "
                "generate_data_cube_hist() or generate_data_cube_smoothed() "
                "before applying noise."
            )
        # Correlated noise from noise source maps has no machinery yet
        if self.noise_source_maps is not None:
            raise exceptions.UnimplementedFunctionality(
                "Correlated noise from noise source maps is not implemented "
                "yet for spectral cubes."
            )

        # The noise is defined in physical units, so the cube needs them
        if cube.units is None:
            raise exceptions.InconsistentArguments(
                "The spectral cube must have units to apply noise."
            )
        units = getattr(cube.units, "units", cube.units)

        # Get the noise level per pixel from the depth. An aperture depth is
        # the noise in the aperture, so it is converted to per pixel,
        # sigma_pix = sigma_ap / sqrt(N_pix) with
        # N_pix = pi r^2 / resolution^2 (assuming independent pixels)
        noise_level = None
        if self.depth is not None:
            noise_level = self.depth / self.snrs
            if self.depth_app_radius is not None:
                app_radius_pix = float(
                    (self.depth_app_radius / cube.resolution).to(
                        "dimensionless"
                    )
                )
                noise_level = noise_level / np.sqrt(np.pi * app_radius_pix**2)

        # Generate the noise, treating each pixel as a spectrum
        rng = np.random.default_rng(rng_seed)
        noise, weight_map = self._generate_noise(
            cube.arr,
            cube.lam,
            units,
            rng,
            noise_level=noise_level,
            simspin_snr=simspin_snr,
        )

        # Work on the original cube when requested, otherwise create a fresh
        # container
        if inplace:
            working_cube = cube
        else:
            working_cube = SpectralCube(
                resolution=cube.resolution,
                fov=cube.fov,
                lam=cube.lam,
            )
            working_cube.units = cube.units
            working_cube.sed = cube.sed
            working_cube.quantity = cube.quantity

        # Add the noise and store it with the weight map
        working_cube.arr = cube.arr + noise
        working_cube.noise_arr = unyt_array(noise, units)
        working_cube.weight_map = weight_map

        return working_cube

    @timed("IntegratedFieldUnit.apply_noise")
    def apply_noise(self, observable, **kwargs):
        """Apply IFU noise to an observable.

        Args:
            observable: Observable to which IFU noise should be applied.
            **kwargs: Additional arguments passed to the noise method.

        Returns:
            SpectralCube: Cube with noise applied.

        Raises:
            InconsistentArguments: If the observable is not a SpectralCube.
        """
        if isinstance(observable, SpectralCube):
            return self.apply_noise_to_cube(observable, **kwargs)

        raise exceptions.InconsistentArguments(
            "IntegratedFieldUnit.apply_noise only supports SpectralCube "
            "objects."
        )

    @timed("IntegratedFieldUnit.to_hdf5")
    def to_hdf5(self, group):
        """Write the integrated field unit to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised.
        """
        super().to_hdf5(group)

        write_instrument_attribute(
            group, "DepthApertureRadius", self.depth_app_radius
        )

        ds = group.create_dataset(
            "Resolution", data=self.resolution.value, dtype=float
        )
        ds.attrs["units"] = str(self.resolution.units)

        if self.psfs is not None:
            ds = group.create_dataset("PSFs", data=self.psfs, dtype=float)
            ds.attrs["units"] = "dimensionless"

        ds = group.create_dataset(
            "PSFWavelength", data=self.psf_lam.value, dtype=float
        )
        ds.attrs["units"] = str(self.psf_lam.units)

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

        if "PSFWavelength" in group:
            psf_lam = unyt_array(
                group["PSFWavelength"][...],
                group["PSFWavelength"].attrs["units"],
            )
        else:
            psf_lam = lam

        if "PSFPixelScale" in group:
            psf_pixel_scale = unyt_quantity(
                group["PSFPixelScale"][()],
                group["PSFPixelScale"].attrs["units"],
            )
        else:
            psf_pixel_scale = None

        if "LSFs" in group:
            lsfs = group["LSFs"][...]
        else:
            lsfs = None

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
            "psf_lam": psf_lam,
            "lsfs": lsfs,
            "noise_maps": noise_maps,
            "noise_source_maps": noise_source_maps,
            "resolving_power": group.attrs.get("resolving_power"),
            "noise_shape_lam": read_instrument_attribute(
                group, "NoiseShapeWavelength"
            ),
            "noise_shape_std": read_instrument_attribute(
                group, "NoiseShapeStd"
            ),
            "template_on_detector": bool(
                group.attrs.get("template_on_detector", True)
            ),
            "depth_lam": read_instrument_attribute(group, "DepthWavelength"),
            "depth_window": read_instrument_attribute(group, "DepthWindow"),
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
            psf_lam=payload["psf_lam"],
            lsfs=payload["lsfs"],
            noise_maps=payload["noise_maps"],
            noise_source_maps=payload["noise_source_maps"],
            resolving_power=payload["resolving_power"],
            noise_shape_lam=payload["noise_shape_lam"],
            noise_shape_std=payload["noise_shape_std"],
            template_on_detector=payload["template_on_detector"],
            depth_lam=payload["depth_lam"],
            depth_window=payload["depth_window"],
        )
