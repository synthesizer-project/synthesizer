"""A module containing generic component functionality.

This module contains the abstract base class for all components in the
synthesizer. It defines the basic structure of a component and the methods
that all components should have.

StellarComponents and BlackHoleComponents are children of this class and
contain the specific functionality for stellar and black hole components
respectively.
"""

from abc import ABC, abstractmethod

from synthesizer import exceptions
from synthesizer.instruments import Instrument
from synthesizer.sed import plot_spectra
from synthesizer.warnings import deprecated, deprecation


class Component(ABC):
    """
    The parent class for all components in the synthesizer.

    This class contains the basic structure of a component and the methods
    that all components should have.

    Attributes:
        component_type (str)
            The type of component, either "Stars" or "BlackHole".
        spectra (dict)
            A dictionary to hold the stellar spectra.
        lines (dict)
            A dictionary to hold the stellar emission lines.
        photo_lnu (dict)
            A dictionary to hold the stellar photometry in luminosity units.
        photo_fnu (dict)
            A dictionary to hold the stellar photometry in flux units.
        images_lnu (dict)
            A dictionary to hold the images in luminosity units.
        images_fnu (dict)
            A dictionary to hold the images in flux units
    """

    def __init__(
        self,
        component_type,
        fesc=0.0,
        **kwargs,
    ):
        """
        Initialise the Component.

        Args:
            component_type (str)
                The type of component, either "Stars" or "BlackHole".
            **kwargs
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

        # Define the dictionaries to hold the images
        self.images_lnu = {}
        self.images_fnu = {}

        # Attach a default escape fraction
        self.fesc = fesc

        # Set any of the extra attribute provided as kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

    @property
    def photo_fluxes(self):
        """
        Get the photometric fluxes.

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
        """
        Get the photometric luminosities.

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
    def generate_lnu(self, *args, **kwargs):
        """Generate the rest frame spectra for the component."""
        pass

    @abstractmethod
    def generate_line(self, *args, **kwargs):
        """Generate the rest frame line emission for the component."""
        pass

    @abstractmethod
    def get_mask(self, attr, thresh, op, mask=None):
        """Return a mask based on the attribute and threshold."""
        pass

    @abstractmethod
    def _prepare_sed_args(self, *args, **kwargs):
        """Prepare arguments for the SED generation."""
        pass

    @abstractmethod
    def _prepare_line_args(self, *args, **kwargs):
        """Prepare arguments for the line generation."""
        pass

    def get_photo_lnu(self, filters, verbose=True, nthreads=1):
        """
        Calculate luminosity photometry using a FilterCollection object.

        Args:
            filters (filters.FilterCollection)
                A FilterCollection object.
            verbose (bool)
                Are we talking?
            nthreads (int)
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            photo_lnu (dict)
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
        """
        Calculate luminosity photometry using a FilterCollection object.

        Alias to get_photo_lnu.

        Photometry is calculated in spectral luminosity density units.

        Args:
            filters (filters.FilterCollection)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            PhotometryCollection
                A PhotometryCollection object containing the luminosity
                photometry in each filter in filters.
        """
        return self.get_photo_lnu(filters, verbose)

    def get_photo_fnu(self, filters, verbose=True, nthreads=1):
        """
        Calculate flux photometry using a FilterCollection object.

        Args:
            filters (object)
                A FilterCollection object.
            verbose (bool)
                Are we talking?
            nthreads (int)
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            (dict)
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
        """
        Calculate flux photometry using a FilterCollection object.

        Alias to get_photo_fnu.

        Photometry is calculated in spectral flux density units.

        Args:
            filters (object)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            PhotometryCollection
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
        **kwargs,
    ):
        """
        Generate stellar spectra as described by the emission model.

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
            shift (bool):
                Flags whether to apply doppler shift to the spectra.
            verbose (bool)
                Are we talking?
            kwargs (dict)
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
        """
        Generate stellar lines as described by the emission model.

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
            verbose (bool)
                Are we talking?
            kwargs (dict)
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
    ):
        """
        Make an ImageCollection from component luminosities.

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
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            emission_model (EmissionModel)
                The emission model to use to generate the images.
            img_type : str
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel for a particle
                galaxy. Otherwise, only smoothed is applicable.
            stellar_photometry (string)
                The stellar spectra key from which to extract photometry
                to use for the image.
            blackhole_photometry (string)
                The black hole spectra key from which to extract photometry
                to use for the image.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).
            nthreads (int)
                The number of threads to use in the tree search. Default is 1.
            instrument (Instrument)
                The instrument to use to generate the images.

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
        if instrument is None:
            instrument = Instrument("place-holder", resolution=resolution)

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
    ):
        """
        Make an ImageCollection from fluxes.

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
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            emission_model (EmissionModel)
                The emission model to use to generate the images.
            img_type : str
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel for a particle
                galaxy. Otherwise, only smoothed is applicable.
            stellar_photometry (string)
                The stellar spectra key from which to extract photometry
                to use for the image.
            blackhole_photometry (string)
                The black hole spectra key from which to extract photometry
                to use for the image.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).
            nthreads (int)
                The number of threads to use in the tree search. Default is 1.
            instrument (Instrument)
                The instrument to use to generate the images.

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
        if instrument is None:
            instrument = Instrument("place-holder", resolution=resolution)

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
        )

        # Store the images
        self.images_fnu.update(images)

        # If we are limiting to a specific image then return that
        if limit_to is not None:
            return images[limit_to]

        # Return the image at the root of the emission model
        return images[emission_model.label]

    def plot_spectra(
        self,
        spectra_to_plot=None,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        **kwargs,
    ):
        """
        Plot the spectra of the component.

        Can either plot specific spectra (specified via spectra_to_plot) or
        all spectra on the child object.

        Args:
            spectra_to_plot (string/list, string)
                The specific spectra to plot.
                    - If None all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool)
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple)
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100) times
                less than the peak of the spectrum for rest_frame (observed)
                spectra.
            xlimits (tuple)
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple)
                Tuple with size 2 defining the figure size.
            kwargs (dict)
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

    def clear_all_spectra(self):
        """Clear all spectra from the component."""
        self.spectra = {}
        if hasattr(self, "particle_spectra"):
            self.particle_spectra = {}

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
        """
        Clear all emissions from the component.

        This clears all spectra, lines, and photometry.
        """
        self.clear_all_spectra()
        self.clear_all_lines()
        self.clear_all_photometry()
