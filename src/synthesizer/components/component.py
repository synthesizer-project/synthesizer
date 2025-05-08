"""A module containing generic component functionality.

This module contains the abstract base class for all components in the
synthesizer. It defines the basic structure of a component and the methods
that all components should have.

StellarComponents and BlackHoleComponents are children of this class and
contain the specific functionality for stellar and black hole components
respectively.
"""

from abc import ABC, abstractmethod

from unyt import arcsecond, kpc, pc

from synthesizer import exceptions
from synthesizer.emissions import plot_spectra
from synthesizer.instruments import Instrument
from synthesizer.synth_warnings import deprecated, deprecation
from synthesizer.units import unit_is_compatible


class Component(ABC):
    """The parent class for all components in the synthesizer.

    This class contains the basic structure of a component and the methods
    that all components should have.

    Attributes:
        component_type (str):
            The type of component, either "Stars" or "BlackHole".
        spectra (dict):
            A dictionary to hold the stellar spectra.
        lines (dict):
            A dictionary to hold the stellar emission lines.
        photo_lnu (dict):
            A dictionary to hold the stellar photometry in luminosity units.
        photo_fnu (dict):
            A dictionary to hold the stellar photometry in flux units.
        images_lnu (dict):
            A dictionary to hold the images in luminosity units.
        images_fnu (dict):
            A dictionary to hold the images in flux units
        fesc (float):
            The escape fraction of the component.
    """

    def __init__(
        self,
        component_type,
        fesc,
        **kwargs,
    ):
        """Initialise the Component.

        Args:
            component_type (str):
                The type of component, either "Stars" or "BlackHole".
            fesc (float):
                The escape fraction of the component.
            **kwargs (dict):
                Any additional keyword arguments to attach to the Component.
        """
        # Attach the component type and name to the object
        self.component_type = component_type

        # Define the spectra dictionary to hold the stellar spectra
        self.spectra = {}

        # Define the line dictionary to hold the stellar emission lines
        self.lines = {}

        # Define the photometry dictionaries to hold the stellar photometry
        self.photo_lnu = {}
        self.photo_fnu = {}

        # Define the dictionaries to hold the images (we carry 3 different
        # distionaries for both lnu and fnu images to draw a distinction
        # between images with and without a PSF and/or noise)
        self.images_lnu = {}
        self.images_fnu = {}
        self.images_psf_lnu = {}
        self.images_psf_fnu = {}
        self.images_noise_lnu = {}
        self.images_noise_fnu = {}

        # Define the dictionaries to hold instrument specific spectroscopy
        self.spectroscopy = {}
        self.particle_spectroscopy = {}

        # Attach a default escape fraction
        self.fesc = fesc if fesc is not None else 0.0

        # Set any of the extra attribute provided as kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        # A container for any grid weights we already computed
        self._grid_weights = {"cic": {}, "ngp": {}}

    @property
    def photo_fluxes(self):
        """Get the photometric fluxes.

        Returns:
            dict
                The photometry fluxes.
        """
        deprecation(
            "The `photo_fluxes` attribute is deprecated. Use "
            "`photo_fnu` instead. Will be removed in v1.0.0"
        )
        return self.photo_fnu

    @property
    def photo_luminosities(self):
        """Get the photometric luminosities.

        Returns:
            dict
                The photometry luminosities.
        """
        deprecation(
            "The `photo_luminosities` attribute is deprecated. Use "
            "`photo_lnu` instead. Will be removed in v1.0.0"
        )
        return self.photo_lnu

    @abstractmethod
    def get_mask(self, attr, thresh, op, mask=None):
        """Return a mask based on the attribute and threshold."""
        pass

    @abstractmethod
    def get_weighted_attr(self, attr, weights, **kwargs):
        """Return the weighted attribute."""
        pass

    @property
    def is_parametric(self):
        """Return whether the component is parametric.

        Returns:
            bool
                Whether the component is parametric.
        """
        # Import here to avoid circular imports
        from synthesizer.parametric import BlackHole as ParametricBlackHole
        from synthesizer.parametric import Stars as ParametricStars

        return isinstance(self, (ParametricStars, ParametricBlackHole))

    @property
    def is_particle(self):
        """Return whether the component is particle based.

        Returns:
            bool
                Whether the component is particle based.
        """
        return not self.is_parametric

    def get_luminosity_distance(self, cosmo):
        """Get the luminosity distance of the component.

        This requires the redshift to be set on the component.

        This will use the astropy cosmology module to calculate the
        luminosity distance. If the redshift is 0, the distance will be set to
        10 pc to avoid any issues with 0s.

        Args:
            cosmo (astropy.cosmology):
                The cosmology to use for the calculation.

        Returns:
            unyt_quantity:
                The luminosity distance of the component in kpc.
        """
        # If we don't have a redshift then we can't calculate the
        # luminosity distance
        if not hasattr(self, "redshift"):
            raise exceptions.InconsistentArguments(
                "The component does not have a redshift set."
            )

        # Check redshift is set
        if self.redshift is None:
            raise exceptions.InconsistentArguments(
                "The component must have a redshift set to calculate the "
                "luminosity distance."
            )

        # At redshift > 0 we can calculate the luminosity distance explicitly
        if self.redshift > 0:
            return (
                cosmo.luminosity_distance(self.redshift).to("kpc").value * kpc
            )

        # At redshift 0 just place the component at 10 pc to
        # avoid any issues with 0s
        return (10 * pc).to(kpc)

    def get_photo_lnu(self, filters, verbose=True, nthreads=1):
        """Calculate luminosity photometry using a FilterCollection object.

        Args:
            filters (filters.FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            photo_lnu (dict):
                A dictionary of rest frame broadband luminosities.
        """
        # Loop over spectra in the component
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_lnu[spectra] = self.spectra[spectra].get_photo_lnu(
                filters,
                verbose,
                nthreads=nthreads,
            )

        return self.photo_lnu

    @deprecated(
        "The `get_photo_luminosities` method is deprecated. Use "
        "`get_photo_lnu` instead. Will be removed in v1.0.0"
    )
    def get_photo_luminosities(self, filters, verbose=True):
        """Calculate luminosity photometry using a FilterCollection object.

        Alias to get_photo_lnu.

        Photometry is calculated in spectral luminosity density units.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?

        Returns:
            PhotometryCollection
                A PhotometryCollection object containing the luminosity
                photometry in each filter in filters.
        """
        return self.get_photo_lnu(filters, verbose)

    def get_photo_fnu(self, filters, verbose=True, nthreads=1):
        """Calculate flux photometry using a FilterCollection object.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            dict:
                A dictionary of fluxes in each filter in filters.
        """
        # Loop over spectra in the component
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_fnu[spectra] = self.spectra[spectra].get_photo_fnu(
                filters,
                verbose,
                nthreads=nthreads,
            )

        return self.photo_fnu

    @deprecated(
        "The `get_photo_fluxes` method is deprecated. Use "
        "`get_photo_fnu` instead. Will be removed in v1.0.0"
    )
    def get_photo_fluxes(self, filters, verbose=True):
        """Calculate flux photometry using a FilterCollection object.

        Alias to get_photo_fnu.

        Photometry is calculated in spectral flux density units.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?

        Returns:
            PhotometryCollection:
                A PhotometryCollection object containing the flux photometry
                in each filter in filters.
        """
        return self.get_photo_fnu(filters, verbose)

    def get_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        vel_shift=None,
        verbose=True,
        nthreads=1,
        grid_assignment_method="cic",
        **kwargs,
    ):
        """Generate stellar spectra as described by the emission model.

        Args:
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            vel_shift (bool):
                Flags whether to apply doppler shift to the spectra.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the tree search. If -1, all
                available threads will be used.
            grid_assignment_method (str):
                The method to use for assigning particles to the grid. Options
                are "cic" (cloud-in-cell) or "ngp" (nearest grid point)."
            **kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of spectra which can be attached to the
                appropriate spectra attribute of the component
                (spectra/particle_spectra)
        """
        # Get the spectra
        spectra, particle_spectra = emission_model._get_spectra(
            emitters={"stellar": self}
            if self.component_type == "Stars"
            else {"blackhole": self},
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            vel_shift=vel_shift,
            verbose=verbose,
            nthreads=nthreads,
            grid_assignment_method=grid_assignment_method,
            **kwargs,
        )

        # Update the spectra dictionary
        self.spectra.update(spectra)

        # Update the particle_spectra dictionary if it exists
        if hasattr(self, "particle_spectra"):
            self.particle_spectra.update(particle_spectra)

        # Return the spectra the user wants
        if emission_model.per_particle:
            return self.particle_spectra[emission_model.label]
        return self.spectra[emission_model.label]

    def get_lines(
        self,
        line_ids,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """Generate stellar lines as described by the emission model.

        Args:
            line_ids (list):
                A list of line_ids. Doublets can be specified as a nested list
                or using a comma (e.g. 'OIII4363,OIII4959').
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool):
                Are we talking?
            kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            LineCollection
                A LineCollection object containing the lines defined by the
                root model.
        """
        # Get the lines
        lines, particle_lines = emission_model._get_lines(
            line_ids=line_ids,
            emitters={"stellar": self}
            if self.component_type == "Stars"
            else {"blackhole": self},
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the lines dictionary
        self.lines.update(lines)

        # Update the particle_lines dictionary if it exists
        if hasattr(self, "particle_lines"):
            self.particle_lines.update(particle_lines)

        # Return the lines the user wants
        if emission_model.per_particle:
            return self.particle_lines[emission_model.label]
        return self.lines[emission_model.label]

    def get_images_luminosity(
        self,
        resolution,
        fov,
        emission_model,
        img_type="smoothed",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
        limit_to=None,
        instrument=None,
        cosmo=None,
    ):
        """Make an ImageCollection from component luminosities.

        For Parametric components, images can only be smoothed. An
        exception will be raised if a histogram is requested.

        For Particle components, images can either be a simple
        histogram ("hist") or an image with particles smoothed over
        their SPH kernel.

        Which images are produced is defined by the emission model. If any
        of the necessary photometry is missing for generating a particular
        image, an exception will be raised.

        The limit_to argument can be used if only a specific image is desired.

        Note that black holes will never be smoothed and only produce a
        histogram due to the point source nature of black holes.

        All images that are created will be stored on the emitter (Stars or
        BlackHole/s) under the images_lnu attribute. The image
        collection at the root of the emission model will also be returned.

        Args:
            resolution (unyt_quantity of float):
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov (float):
                The width of the image in image coordinates.
            emission_model (EmissionModel):
                The emission model to use to generate the images.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel for a particle
                galaxy. Otherwise, only smoothed is applicable.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.
            limit_to (str):
                The label of the image to limit to. If None, all images are
                returned.
            instrument (Instrument):
                The instrument to use to generate the images.
            cosmo (astropy.cosmology):
                The cosmology to use for the calculation of the luminosity
                distance. Only needed for internal conversions from cartesian
                to angular coordinates when an angular resolution is used.

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Ensure we aren't trying to make a histogram for a parametric
        # component
        if hasattr(self, "morphology") and img_type == "hist":
            raise exceptions.InconsistentArguments(
                f"Parametric {self.component_type} can only produce "
                "smoothed images."
            )

        # If we haven't got an instrument create one
        # TODO: we need to eventually fully pivot to taking only an instrument
        # this will be done when we introduced some premade instruments
        if instrument is None:
            # Get the filters from the emitters
            filters = self.particle_photo_lnu[emission_model.label].filters

            # Make the place holder instrument
            instrument = Instrument(
                "place-holder",
                resolution=resolution,
                filters=filters,
            )

        # Ensure we have a cosmology if we need it
        if unit_is_compatible(instrument.resolution, arcsecond):
            if cosmo is None:
                raise exceptions.InconsistentArguments(
                    "Cosmology must be provided when using an angular "
                    "resolution and FOV."
                )

            # Also ensure we have a redshift
            if self.redshift is None:
                raise exceptions.MissingAttribute(
                    "Redshift must be set when using an angular "
                    "resolution and FOV."
                )

        # Get the images
        images = emission_model._get_images(
            instrument=instrument,
            fov=fov,
            emitters={"stellar": self}
            if self.component_type == "Stars"
            else {"blackhole": self},
            img_type=img_type,
            mask=None,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
            limit_to=limit_to,
            do_flux=False,
            cosmo=cosmo,
        )

        # Store the images
        self.images_lnu.update(images)

        # If we are limiting to a specific image then return that
        if limit_to is not None:
            return images[limit_to]

        # Return the image at the root of the emission model
        return images[emission_model.label]

    def get_images_flux(
        self,
        resolution,
        fov,
        emission_model,
        img_type="smoothed",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
        limit_to=None,
        instrument=None,
        cosmo=None,
    ):
        """Make an ImageCollection from fluxes.

        For Parametric components, images can only be smoothed. An
        exception will be raised if a histogram is requested.

        For Particle components, images can either be a simple
        histogram ("hist") or an image with particles smoothed over
        their SPH kernel.

        Which images are produced is defined by the emission model. If any
        of the necessary photometry is missing for generating a particular
        image, an exception will be raised.

        The limit_to argument can be used if only a specific image is desired.

        Note that black holes will never be smoothed and only produce a
        histogram due to the point source nature of black holes.

        All images that are created will be stored on the emitter (Stars or
        BlackHole/s) under the images_fnu attribute. The image
        collection at the root of the emission model will also be returned.

        Args:
            resolution (unyt_quantity of float):
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov (float):
                The width of the image in image coordinates.
            emission_model (EmissionModel):
                The emission model to use to generate the images.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel for a particle
                galaxy. Otherwise, only smoothed is applicable.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.
            limit_to (str):
                The label of the image to limit to. If None, all images are
                returned.
            instrument (Instrument):
                The instrument to use to generate the images.
            cosmo (astropy.cosmology):
                The cosmology to use for the calculation of the luminosity
                distance. Only needed for internal conversions from cartesian
                to angular coordinates when an angular resolution is used.

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Ensure we aren't trying to make a histogram for a parametric
        # component
        if hasattr(self, "morphology") and img_type == "hist":
            raise exceptions.InconsistentArguments(
                f"Parametric {self.component_type} can only produce "
                "smoothed images."
            )

        # If we haven't got an instrument create one
        # TODO: we need to eventually fully pivot to taking only an instrument
        # this will be done when we introduced some premade instruments
        if instrument is None:
            # Get the filters from the emitters
            filters = self.particle_photo_lnu[emission_model.label].filters

            # Make the place holder instrument
            instrument = Instrument(
                "place-holder",
                resolution=resolution,
                filters=filters,
            )

        # Ensure we have a cosmology if we need it
        if unit_is_compatible(instrument.resolution, arcsecond):
            if cosmo is None:
                raise exceptions.InconsistentArguments(
                    "Cosmology must be provided when using an angular "
                    "resolution and FOV."
                )

            # Also ensure we have a redshift
            if self.redshift is None:
                raise exceptions.MissingAttribute(
                    "Redshift must be set when using an angular "
                    "resolution and FOV."
                )

        # Get the images
        images = emission_model._get_images(
            instrument=instrument,
            fov=fov,
            emitters={"stellar": self}
            if self.component_type == "Stars"
            else {"blackhole": self},
            img_type=img_type,
            mask=None,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
            limit_to=limit_to,
            do_flux=True,
            cosmo=cosmo,
        )

        # Store the images
        self.images_fnu.update(images)

        # If we are limiting to a specific image then return that
        if limit_to is not None:
            return images[limit_to]

        # Return the image at the root of the emission model
        return images[emission_model.label]

    def apply_psf_to_images_lnu(
        self,
        instrument,
        psf_resample_factor=1,
        limit_to=None,
    ):
        """Apply instrument PSFs to this component's luminosity images.

        Args:
            instrument (Instrument):
                The instrument with the PSF to apply.
            psf_resample_factor (int):
                The resample factor for the PSF. This should be a value greater
                than 1. The image will be resampled by this factor before the
                PSF is applied and then downsampled back to the original
                after convolution. This can help minimize the effects of
                using a generic PSF centred on the galaxy centre, a
                simplification we make for performance reasons (the
                effects are sufficiently small that this simplifications is
                justified).
            limit_to (str/list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.

        Returns:
            dict The images with the PSF applied.
        """
        # Ensure limit_to is a list
        limit_to = [limit_to] if isinstance(limit_to, str) else limit_to

        # Sanity check that we have a PSF
        if instrument.psfs is None:
            raise exceptions.InconsistentArguments(
                f"Instrument ({instrument.label}) does not have PSFs."
            )

        # Ensure we have images for this instrument
        if instrument.label not in self.images_lnu:
            raise exceptions.InconsistentArguments(
                "No images found in images_lnu for instrument"
                f" {instrument.label}."
            )

        # Create an entry for the instrument in the PSF images
        # dictionary if it doesn't exist
        if instrument.label not in self.images_psf_lnu:
            self.images_psf_lnu[instrument.label] = {}

        # Loop over the images in the component
        for key in self.images_lnu[instrument.label]:
            # Are we limiting to a specific model?
            if limit_to is not None and key not in limit_to:
                continue

            # Unpack the image
            imgs = self.images_lnu[instrument.label][key]

            # If requested, do the resampling
            if psf_resample_factor > 1:
                imgs.supersample(psf_resample_factor)

            # Apply the PSF
            self.images_psf_lnu[instrument.label][key] = imgs.apply_psfs(
                instrument.psfs
            )

            # Undo the resampling (if needed)
            if psf_resample_factor > 1:
                self.images_psf_lnu[instrument.label][key].downsample(
                    1 / psf_resample_factor
                )

        return self.images_psf_lnu[instrument.label]

    def apply_psf_to_images_fnu(
        self,
        instrument,
        psf_resample_factor=1,
        limit_to=None,
    ):
        """Apply instrument PSFs to this component's flux images.

        Args:
            instrument (Instrument):
                The instrument with the PSF to apply.
            psf_resample_factor (int):
                The resample factor for the PSF. This should be a value greater
                than 1. The image will be resampled by this factor before the
                PSF is applied and then downsampled back to the original
                after convolution. This can help minimize the effects of
                using a generic PSF centred on the galaxy centre, a
                simplification we make for performance reasons (the
                effects are sufficiently small that this simplifications is
                justified).
            limit_to (str/list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.

        Returns:
            dict The images with the PSF applied.
        """
        # Ensure limit_to is a list
        limit_to = [limit_to] if isinstance(limit_to, str) else limit_to

        # Sanity check that we have a PSF
        if instrument.psfs is None:
            raise exceptions.InconsistentArguments(
                f"Instrument ({instrument.label}) does not have PSFs."
            )

        # Ensure we have images for this instrument
        if instrument.label not in self.images_fnu:
            raise exceptions.InconsistentArguments(
                "No images found in images_fnu for instrument"
                f" {instrument.label}."
            )

        # Create an entry for the instrument in the PSF images
        # dictionary if it doesn't exist
        if instrument.label not in self.images_psf_fnu:
            self.images_psf_fnu[instrument.label] = {}

        # Loop over the images in the component
        for key in self.images_fnu[instrument.label]:
            # Are we limiting to a specific model?
            if limit_to is not None and key not in limit_to:
                continue

            # Unpack the image
            imgs = self.images_fnu[instrument.label][key]

            # If requested, do the resampling
            if psf_resample_factor > 1:
                imgs.supersample(psf_resample_factor)

            # Apply the PSF
            self.images_psf_fnu[instrument.label][key] = imgs.apply_psfs(
                instrument.psfs
            )

            # Undo the resampling (if needed)
            if psf_resample_factor > 1:
                self.images_psf_fnu[instrument.label][key].downsample(
                    1 / psf_resample_factor
                )

        return self.images_psf_fnu[instrument.label]

    def apply_noise_to_images_lnu(
        self,
        instrument,
        limit_to=None,
        apply_to_psf=True,
    ):
        """Apply instrument noise to this component's images.

        Args:
            instrument (Instrument):
                The instrument with the noise to apply.
            limit_to (str/list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.
            apply_to_psf (bool):
                If True, apply the noise to the PSF images.
                Otherwise, apply to the non-PSF images.

        Returns:
            dict The images with the noise applied.
        """
        # Ensure limit_to is a list
        limit_to = [limit_to] if isinstance(limit_to, str) else limit_to

        # Get the images we are applying the noise to
        if apply_to_psf and instrument.label in self.images_psf_lnu:
            images = self.images_psf_lnu[instrument.label]
        elif instrument.label in self.images_lnu:
            images = self.images_lnu[instrument.label]
        else:
            if apply_to_psf:
                raise exceptions.InconsistentArguments(
                    "No images found in images_psf_lnu for instrument"
                    f" {instrument.label}."
                )
            raise exceptions.InconsistentArguments(
                "No images found in images_lnu  for instrument"
                f" {instrument.label}."
            )

        # Create an entry for the instrument in the noise images
        # dictionary if it doesn't exist
        if instrument.label not in self.images_noise_lnu:
            self.images_noise_lnu[instrument.label] = {}

        # Loop over the images in the component
        for key in images:
            # Are we limiting to a specific model?
            if limit_to is not None and key not in limit_to:
                continue

            # Unpack the image
            imgs = images[key]

            # Apply the noise using the correct method
            if instrument.noise_maps is not None:
                self.images_noise_lnu[instrument.label][key] = (
                    imgs.apply_noise_arrays(
                        instrument.noise_maps,
                    )
                )
            elif instrument.snrs is not None:
                self.images_noise_lnu[instrument.label][key] = (
                    imgs.apply_noise_from_snrs(
                        snrs=instrument.snrs,
                        depths=instrument.depth,
                        aperture_radius=instrument.depth_aperture_radius,
                    )
                )
            else:
                raise exceptions.InconsistentArguments(
                    f"Instrument ({instrument.label}) cannot be used "
                    "for applying noise."
                )

        return self.images_noise_lnu[instrument.label]

    def apply_noise_to_images_fnu(
        self,
        instrument,
        limit_to=None,
        apply_to_psf=True,
    ):
        """Apply instrument noise to this component's images.

        Args:
            instrument (Instrument):
                The instrument with the noise to apply.
            limit_to (str/list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.
            apply_to_psf (bool):
                If True, apply the noise to the PSF images.
                Otherwise, apply to the non-PSF images.

        Returns:
            dict The images with the noise applied.
        """
        # Ensure limit_to is a list
        limit_to = [limit_to] if isinstance(limit_to, str) else limit_to

        # Get the images we are applying the noise to
        if apply_to_psf and instrument.label in self.images_psf_fnu:
            images = self.images_psf_fnu[instrument.label]
        elif instrument.label in self.images_fnu:
            images = self.images_fnu[instrument.label]
        else:
            if apply_to_psf:
                raise exceptions.InconsistentArguments(
                    "No images found in images_psf_fnu for instrument"
                    f" {instrument.label}."
                )
            raise exceptions.InconsistentArguments(
                "No images found in images_fnu for instrument"
                f" {instrument.label}."
            )

        # Create an entry for the instrument in the noise images
        # dictionary if it doesn't exist
        if instrument.label not in self.images_noise_fnu:
            self.images_noise_fnu[instrument.label] = {}

        # Loop over the images in the component
        for key in images:
            # Are we limiting to a specific model?
            if limit_to is not None and key not in limit_to:
                continue

            # Unpack the image
            imgs = images[key]

            # Apply the noise using the correct method
            if instrument.noise_maps is not None:
                self.images_noise_fnu[instrument.label][key] = (
                    imgs.apply_noise_arrays(
                        instrument.noise_maps,
                    )
                )
            elif instrument.snrs is not None:
                self.images_noise_fnu[instrument.label][key] = (
                    imgs.apply_noise_from_snrs(
                        snrs=instrument.snrs,
                        depths=instrument.depth,
                        aperture_radius=instrument.depth_aperture_radius,
                    )
                )
            else:
                raise exceptions.InconsistentArguments(
                    f"Instrument ({instrument.label}) cannot be used "
                    "for applying noise."
                )

        return self.images_noise_fnu[instrument.label]

    def get_spectroscopy(
        self,
        instrument,
    ):
        """Get spectroscopy for the component based on a specific instrument.

        This will apply the instrument's wavelength array to each
        spectra stored on the component.

        Args:
            instrument (Instrument):
                The instrument to use for the spectroscopy.

        Returns:
            dict
                The spectroscopy for the galaxy.
        """
        # Create an entry for the instrument in the spectroscopy
        # dictionary if it doesn't exist
        if instrument.label not in self.spectroscopy:
            self.spectroscopy[instrument.label] = {}

        # Loop over the spectra in the component and apply the instrument
        for key, sed in self.spectra.items():
            self.spectroscopy[instrument.label][key] = (
                sed.apply_instrument_lams(instrument)
            )

        # If we have particle spectra then do the same for them
        if (
            hasattr(self, "particle_spectra")
            and len(self.particle_spectra) > 0
        ):
            if instrument.label not in self.particle_spectroscopy:
                self.particle_spectroscopy[instrument.label] = {}

            # Loop over the spectra in the component and apply the instrument
            for key, sed in self.particle_spectra.items():
                self.particle_spectroscopy[instrument.label][key] = (
                    sed.apply_instrument_lams(instrument)
                )

        # Return the spectroscopy for the component
        return self.spectroscopy[instrument.label]

    def plot_spectra(
        self,
        spectra_to_plot=None,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        **kwargs,
    ):
        """Plot the spectra of the component.

        Can either plot specific spectra (specified via spectra_to_plot) or
        all spectra on the child object.

        Args:
            spectra_to_plot (string/list, string):
                The specific spectra to plot.
                    - If None all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool):
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple):
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100) times
                less than the peak of the spectrum for rest_frame (observed)
                spectra.
            xlimits (tuple):
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple):
                Tuple with size 2 defining the figure size.
            kwargs (dict):
                Arguments to the `sed.plot_spectra` method called from this
                wrapper.

        Returns:
            fig (matplotlib.pyplot.figure)
                The matplotlib figure object for the plot.
            ax (matplotlib.axes)
                The matplotlib axes object containing the plotted data.
        """
        # Handling whether we are plotting all spectra, specific spectra, or
        # a single spectra
        if spectra_to_plot is None:
            spectra = self.spectra
        elif isinstance(spectra_to_plot, (list, tuple)):
            spectra = {key: self.spectra[key] for key in spectra_to_plot}
        else:
            spectra = self.spectra[spectra_to_plot]

        return plot_spectra(
            spectra,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            **kwargs,
        )

    def plot_spectroscopy(
        self,
        instrument_label,
        spectra_to_plot=None,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        fig=None,
        ax=None,
        **kwargs,
    ):
        """Plot the instrument's spectroscopy of the component.

        This will plot the spectroscopy for the component using the
        instrument's wavelength array. The spectra are plotted
        in the order they are stored in the spectroscopy dictionary.

        Can either plot specific spectroscopy (specified via spectra_to_plot)
        or all spectroscopy on the component.

        Args:
            instrument_label (str):
                The label of the instrument to use for the spectroscopy.
            spectra_to_plot (string/list, string):
                The specific spectroscopy to plot.
                    - If None all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool):
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple):
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100) times
                less than the peak of the spectrum for rest_frame (observed)
                spectra.
            xlimits (tuple):
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple):
                Tuple with size 2 defining the figure size.
            fig (matplotlib.pyplot.figure):
                The matplotlib figure object for the plot.
            ax (matplotlib.axes):
                The matplotlib axes object containing the plotted data.
            **kwargs (dict):
                Arguments to the `sed.plot_spectra` method called from this
                wrapper.

        Returns:
            fig (matplotlib.pyplot.figure)
                The matplotlib figure object for the plot.
            ax (matplotlib.axes)
                The matplotlib axes object containing the plotted data.
        """
        # Handling whether we are plotting all spectra, specific spectra, or
        # a single spectra
        if spectra_to_plot is None:
            spectra = self.spectroscopy[instrument_label]
        elif isinstance(spectra_to_plot, (list, tuple)):
            spectra = {
                key: self.spectroscopy[instrument_label][key]
                for key in spectra_to_plot
            }
        else:
            spectra = self.spectroscopy[instrument_label][spectra_to_plot]

        # Include the instrument label in the spectra key (i.e. plot lables)
        if isinstance(spectra, dict):
            spectra = {
                f"{instrument_label}: {key}": self.spectroscopy[
                    instrument_label
                ][key]
                for key in spectra
            }

        return plot_spectra(
            spectra,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            fig=fig,
            ax=ax,
            **kwargs,
        )

    def clear_all_spectra(self):
        """Clear all spectra from the component."""
        self.spectra = {}
        if hasattr(self, "particle_spectra"):
            self.particle_spectra = {}

    def clear_all_spectroscopy(self):
        """Clear all spectroscopy from the component."""
        self.spectroscopy = {}
        if hasattr(self, "particle_spectroscopy"):
            self.particle_spectroscopy = {}

    def clear_all_lines(self):
        """Clear all lines from the component."""
        self.lines = {}
        if hasattr(self, "particle_lines"):
            self.particle_lines = {}

    def clear_all_photometry(self):
        """Clear all photometry from the component."""
        self.photo_lnu = {}
        self.photo_fnu = {}
        if hasattr(self, "particle_photo_lnu"):
            self.particle_photo_lnu = {}
        if hasattr(self, "particle_photo_fnu"):
            self.particle_photo_fnu = {}

    def clear_all_emissions(self):
        """Clear all emissions from the component.

        This clears all spectra, lines, and photometry.
        """
        self.clear_all_spectra()
        self.clear_all_lines()
        self.clear_all_photometry()
        self.clear_all_spectroscopy()

    def clear_weights(self):
        """Clear all cached grid weights from the component.

        This clears all grid weights calculated using different
        methods from this component, and resets the `_grid_weights`
        dictionary.
        """
        if hasattr(self, "_grid_weights"):
            self._grid_weights = {"cic": {}, "ngp": {}}
