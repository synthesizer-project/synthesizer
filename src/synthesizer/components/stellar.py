"""A module for particle and parametric stars.

This should never be directly instantiated. Instead it is the parent class for
particle.Stars and parametric.Stars and contains attributes
and methods common between them.
"""

from unyt import unyt_quantity

from synthesizer import exceptions
from synthesizer.sed import plot_spectra
from synthesizer.units import Quantity


class StarsComponent:
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly.

    Attributes:

    """

    # Define quantities
    ages = Quantity()

    def __init__(
        self,
        ages,
        metallicities,
    ):
        """
        Initialise the StellarComponent.

        Args:
            ages (array-like, float)
                The age of each stellar particle/bin (particle/parametric).
            metallicities (array-like, float)
                The metallicity of each stellar particle/bin
                (particle/parametric).
        """

        # Define the spectra dictionary to hold the stellar spectra
        self.spectra = {}

        # Define the line dictionary to hold the stellar emission lines
        self.lines = {}

        # Define the photometry dictionaries to hold the stellar photometry
        self.photo_luminosities = {}
        self.photo_fluxes = {}

        # The common stellar attributes between particle and parametric stars
        self.ages = ages
        self.metallicities = metallicities

    def generate_lnu(self, *args, **kwargs):
        """
        This method is a prototype for generating spectra from SPS grids. It is
        redefined on the child classes.
        """
        raise Warning(
            (
                "generate_lnu should be overloaded by child classes:\n"
                "`particle.Stars`\n"
                "`parametric.Stars`\n"
                "You should not be seeing this!!!"
            )
        )

    def generate_line(self, *args, **kwargs):
        """
        This method is a prototype for generating lines from SPS grids. It is
        redefined on the child classes.
        """
        raise Warning(
            (
                "generate_line should be overloaded by child classes:\n"
                "`particle.Stars`\n"
                "`parametric.Stars`\n"
                "You should not be seeing this!!!"
            )
        )

    def _check_young_old_units(self, young, old):
        """
        Checks whether the `young` and `old` arguments to many
        spectra generation methods are in the right units (Myr)

        Args:
            young (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for young star particles.
            old (unyt_quantity):
                If not None, specifies age in Myr at which to filter
                for old star particles.
        """

        if young is not None:
            if isinstance(young, (unyt_quantity)):
                young = young.to("Myr")
            else:
                raise exceptions.InconsistentArguments(
                    "young must be a unyt_quantity (i.e. a value with units)"
                )
        if old is not None:
            if isinstance(old, (unyt_quantity)):
                old = old.to("Myr")
            else:
                raise exceptions.InconsistentArguments(
                    "young must be a unyt_quantity (i.e. a value with units)"
                )

        return young, old

    def get_photo_luminosities(self, filters, verbose=True):
        """
        Calculate luminosity photometry using a FilterCollection object.

        Args:
            filters (filters.FilterCollection)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            photo_luminosities (dict)
                A dictionary of rest frame broadband luminosities.
        """
        # Loop over spectra in the component
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_luminosities[spectra] = self.spectra[
                spectra
            ].get_photo_luminosities(filters, verbose)

        return self.photo_luminosities

    def get_photo_fluxes(self, filters, verbose=True):
        """
        Calculate flux photometry using a FilterCollection object.

        Args:
            filters (object)
                A FilterCollection object.
            verbose (bool)
                Are we talking?

        Returns:
            (dict)
                A dictionary of fluxes in each filter in filters.
        """
        # Loop over spectra in the component
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_fluxes[spectra] = self.spectra[
                spectra
            ].get_photo_fluxes(filters, verbose)

        return self.photo_fluxes

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
        Plots either specific spectra (specified via spectra_to_plot) or all
        spectra on the child Stars object.

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
        elif isinstance(spectra_to_plot, list):
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

    def get_mask(self, attr, thresh, op, mask=None):
        """
        This method is a prototype for generating masks for a stellar
        component. It is redefined on the child classes.
        """
        raise Warning(
            (
                "get_masks should be overloaded by child classes:\n"
                "`particle.Particles`\n"
                "`parametric.Stars`\n"
                "You should not be seeing this!!!"
            )
        )

    def get_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate stellar spectra as described by the emission model.

        Args:
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An overide to the emisison model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An overide to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                        should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form:
                            {<label>: float(<tau_v>)}
                        to use a specific optical depth with a particular
                        model or
                            {<label>: str(<attribute>)}
                        to use an attribute of the component as the optical
                        depth.
            fesc (dict):
                An overide to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form:
                            {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or
                            {<label>: str(<attribute>)}
                      to use an attribute of the component as the escape
                      fraction.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
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
        spectra = emission_model._get_spectra(
            components={"stellar": self},
            per_particle=False,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the spectra dictionary
        self.spectra.update(spectra)

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
                An overide to the emisison model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An overide to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                        should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form:
                            {<label>: float(<tau_v>)}
                        to use a specific optical depth with a particular
                        model or
                            {<label>: str(<attribute>)}
                        to use an attribute of the component as the optical
                        depth.
            fesc (dict):
                An overide to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form:
                            {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or
                            {<label>: str(<attribute>)}
                      to use an attribute of the component as the escape
                      fraction.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
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
        lines = emission_model._get_lines(
            line_ids=line_ids,
            components={"stellar": self},
            per_particle=False,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the lines dictionary
        self.lines.update(lines)

        return self.lines[emission_model.label]
