"""A module for common functionality in Parametric and Particle Galaxies

The class described in this module should never be directly instatiated. It
only contains common attributes and methods to reduce boilerplate.
"""

from unyt import Mpc

from synthesizer import exceptions
from synthesizer.emission_models.attenuation import Inoue14
from synthesizer.emissions import Sed, plot_observed_spectra, plot_spectra
from synthesizer.instruments import Instrument
from synthesizer.synth_warnings import deprecated, deprecation
from synthesizer.units import accepts
from synthesizer.utils import TableFormatter


class BaseGalaxy:
    """
    The base galaxy class.

    This should never be directly instantiated. It instead contains the common
    functionality and attributes needed for parametric and particle galaxies.

    Attributes:
        spectra (dict, Sed)
            The dictionary containing a Galaxy's spectra. Each entry is an
            Sed object. This dictionary only contains combined spectra from
            All components that make up the Galaxy (Stars, Gas, BlackHoles).
        stars (particle.Stars/parametric.Stars)
            The Stars object holding information about the stellar population.
        gas (particle.Gas/parametric.Gas)
            The Gas object holding information about the gas distribution.
        black_holes (particle.BlackHoles/parametric.BlackHole)
            The BlackHole/s object holding information about the black hole/s.
    """

    @accepts(centre=Mpc)
    def __init__(self, stars, gas, black_holes, redshift, centre, **kwargs):
        """
        Instantiate the base Galaxy class.

        This is the parent class of both parametric.Galaxy and particle.Galaxy.

        Note: The stars, gas, and black_holes component objects differ for
        parametric and particle galaxies but are attached at this parent level
        regardless to unify the Galaxy syntax for both cases.

        Args:
            stars (particle.Stars/parametric.Stars)
                The Stars object holding information about the stellar
                population.
            gas (particle.Gas/parametric.Gas)
                The Gas object holding information about the gas distribution.
            black_holes (particle.BlackHoles/parametric.BlackHole)
                The BlackHole/s object holding information about the
                black hole/s.
            redshift (float)
                The redshift of the galaxy.
            centre (array)
                The centre of the galaxy.
            **kwargs
                Any additional attributes to attach to the galaxy object.
        """
        # Container for the spectra and lines
        self.spectra = {}
        self.lines = {}

        # Initialise the photometry dictionaries
        self.photo_lnu = {}
        self.photo_fnu = {}

        # Intialise the image dictionaries
        self.images_lnu = {}
        self.images_fnu = {}

        # Attach the components
        self.stars = stars
        self.gas = gas
        self.black_holes = black_holes

        # The redshift of the galaxy
        self.redshift = redshift
        self.centre = centre

        if getattr(self, "galaxy_type") is None:
            raise Warning(
                "Instantiating a BaseGalaxy object is not "
                "supported behaviour. Instead, you should "
                "use one of the derived Galaxy classes:\n"
                "`particle.galaxy.Galaxy`\n"
                "`parametric.galaxy.Galaxy`"
            )

        # Attach any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

        # If the centre has been provided then we need to make sure all our
        # components agree
        if centre is not None:
            if self.stars is not None:
                self.stars.centre = centre
            if self.gas is not None:
                self.gas.centre = centre
            if self.black_holes is not None:
                self.black_holes.centre = centre

    @property
    def photo_fluxes(self):
        """
        Get the photometry fluxes.

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
        Get the photometry luminosities.

        Returns:
            dict
                The photometry luminosities.
        """
        deprecation(
            "The `photo_luminosities` attribute is deprecated. Use "
            "`photo_lnu` instead. Will be removed in v1.0.0"
        )
        return self.photo_lnu

    def __str__(self):
        """
        Return a string representation of the galaxy object.

        Returns:
            table (str)
                A string representation of the galaxy object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Galaxy")

    def get_equivalent_width(self, feature, blue, red, spectra_to_plot=None):
        """
        Get all equivalent widths associated with a sed object

        Parameters
        ----------
        index: float
            the index to be used in the computation of equivalent width.
        spectra_to_plot: float array
            An empty list of spectra to be populated.

        Returns
        -------
        equivalent_width : float
            The calculated equivalent width at the current index.
        """
        equivalent_width = None

        if not isinstance(spectra_to_plot, list):
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]

            # Compute equivalent width
            equivalent_width = sed.measure_index(feature, blue, red)

        return equivalent_width

    def get_observed_spectra(self, cosmo, igm=Inoue14):
        """
        Calculate the observed spectra for all Sed objects within this galaxy.

        This will run Sed.get_fnu(...) and populate Sed.fnu (and sed.obslam
        and sed.obsnu) for all spectra in:
        - Galaxy.spectra
        - Galaxy.stars.spectra
        - Galaxy.gas.spectra (WIP)
        - Galaxy.black_holes.spectra

        And in the case of particle galaxies
        - Galaxy.stars.particle_spectra
        - Galaxy.gas.particle_spectra (WIP)
        - Galaxy.black_holes.particle_spectra

        Args:
            cosmo (astropy.cosmology.Cosmology)
                The cosmology object containing the cosmological model used
                to calculate the luminosity distance.
            igm (igm)
                The object describing the intergalactic medium (defaults to
                Inoue14).

        Raises:
            MissingAttribute
                If a galaxy has no redshift we can't get the observed spectra.

        """
        # Ensure we have a redshift
        if self.redshift is None:
            raise exceptions.MissingAttribute(
                "This Galaxy has no redshift! Fluxes can't be"
                " calculated without one."
            )

        # Loop over all combined spectra
        for sed in self.spectra.values():
            # Calculate the observed spectra
            sed.get_fnu(
                cosmo=cosmo,
                z=self.redshift,
                igm=igm,
            )

        # Do we have stars?
        if self.stars is not None:
            # Loop over all stellar spectra
            for sed in self.stars.spectra.values():
                # Calculate the observed spectra
                sed.get_fnu(
                    cosmo=cosmo,
                    z=self.redshift,
                    igm=igm,
                )

            # Loop over all stellar particle spectra
            if getattr(self.stars, "particle_spectra", None) is not None:
                for sed in self.stars.particle_spectra.values():
                    # Calculate the observed spectra
                    sed.get_fnu(
                        cosmo=cosmo,
                        z=self.redshift,
                        igm=igm,
                    )

        # Do we have black holes?
        if self.black_holes is not None:
            # Loop over all black hole spectra
            for sed in self.black_holes.spectra.values():
                # Calculate the observed spectra
                sed.get_fnu(
                    cosmo=cosmo,
                    z=self.redshift,
                    igm=igm,
                )

            # Loop over all black hole particle spectra
            if getattr(self.black_holes, "particle_spectra", None) is not None:
                for sed in self.black_holes.particle_spectra.values():
                    # Calculate the observed spectra
                    sed.get_fnu(
                        cosmo=cosmo,
                        z=self.redshift,
                        igm=igm,
                    )

    def get_observed_lines(self, cosmo, igm=Inoue14):
        """
        Calculate the observed lines for all Line objects within this galaxy.

        This will run Line.get_fnu(...) and populate Line.fnu (and Line.obslam
        and Line.obsnu) for all lines in:
        - Galaxy.lines
        - Galaxy.stars.lines
        - Galaxy.gas.lines (WIP)
        - Galaxy.black_holes.lines

        And in the case of particle galaxies
        - Galaxy.stars.particle_lines
        - Galaxy.gas.particle_lines (WIP)
        - Galaxy.black_holes.particle_lines

        Args:
            cosmo (astropy.cosmology.Cosmology)
                The cosmology object containing the cosmological model used
                to calculate the luminosity distance.
            igm (igm)
                The object describing the intergalactic medium (defaults to
                Inoue14).

        Raises:
            MissingAttribute
                If a galaxy has no redshift we can't get the observed lines.
        """
        # Ensure we have a redshift
        if self.redshift is None:
            raise exceptions.MissingAttribute(
                "This Galaxy has no redshift! Fluxes can't be"
                " calculated without one."
            )

        # Loop over all combined lines
        for line in self.lines.values():
            # Calculate the observed lines
            line.get_flux(
                cosmo=cosmo,
                z=self.redshift,
                igm=igm,
            )

        # Do we have stars?
        if self.stars is not None:
            # Loop over all stellar lines
            for line in self.stars.lines.values():
                # Calculate the observed lines
                line.get_flux(
                    cosmo=cosmo,
                    z=self.redshift,
                    igm=igm,
                )

            # Loop over all stellar particle lines
            if getattr(self.stars, "particle_lines", None) is not None:
                # Loop over all stellar particle lines
                for line in self.stars.particle_lines.values():
                    # Calculate the observed lines
                    line.get_flux(
                        cosmo=cosmo,
                        z=self.redshift,
                        igm=igm,
                    )

        # Do we have black holes?
        if self.black_holes is not None:
            # Loop over all black hole lines
            for line in self.black_holes.lines.values():
                # Calculate the observed lines
                line.get_flux(
                    cosmo=cosmo,
                    z=self.redshift,
                    igm=igm,
                )

            # Loop over all black hole particle lines
            if getattr(self.black_holes, "particle_lines", None) is not None:
                for line in self.black_holes.particle_lines.values():
                    # Calculate the observed lines
                    line.get_flux(
                        cosmo=cosmo,
                        z=self.redshift,
                        igm=igm,
                    )

    def get_spectra_combined(self):
        """
        Combine all common component spectra from components onto the galaxy.

        e.g.:
            intrinsc = stellar_intrinsic + black_hole_intrinsic.

        For any combined spectra all components with a valid spectra will be
        combined and stored in Galaxy.spectra under the same key, but only if
        there are instances of that spectra key on more than 1 component.

        Possible combined spectra are:
            - "total"
            - "intrinsic"
            - "emergent"

        Note that this process is only applicable to integrated spectra.
        """
        # Get the spectra we have on the components to combine
        spectra = {"total": [], "intrinsic": [], "emergent": []}
        for key in spectra:
            if self.stars is not None and key in self.stars.spectra:
                spectra[key].append(self.stars.spectra[key])
            if (
                self.black_holes is not None
                and key in self.black_holes.spectra
            ):
                spectra[key].append(self.black_holes.spectra[key])
            if self.gas is not None and key in self.gas.spectra:
                spectra[key].append(self.gas.spectra[key])

        # Now combine all spectra that have more than one contributing
        # component.
        # Note that sum when applied to a list of spectra
        # with overloaded __add__ methods will produce an Sed object
        # containing the combined spectra.
        for key, lst in spectra.items():
            if len(lst) > 1:
                self.spectra[key] = sum(lst)

    def get_photo_lnu(self, filters, verbose=True, nthreads=1):
        """
        Calculate luminosity photometry using a FilterCollection object.

        Photometry is calculated in spectral luminosity density units.

        Args:
            filters (filters.FilterCollection)
                A FilterCollection object.
            verbose (bool)
                Are we talking?
            nthreads (int)
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            PhotometryCollection
                A PhotometryCollection object containing the luminosity
                photometry in each filter in filters.
        """
        # Get stellar photometry
        if self.stars is not None:
            self.stars.get_photo_lnu(filters, verbose, nthreads=nthreads)

            # If we have particle spectra do that too (not applicable to
            # parametric Galaxy)
            if getattr(self.stars, "particle_spectra", None) is not None:
                self.stars.get_particle_photo_lnu(
                    filters,
                    verbose,
                    nthreads=nthreads,
                )

        # Get black hole photometry
        if self.black_holes is not None:
            self.black_holes.get_photo_lnu(filters, verbose, nthreads=nthreads)

            # If we have particle spectra do that too (not applicable to
            # parametric Galaxy)
            if getattr(self.black_holes, "particle_spectra", None) is not None:
                self.black_holes.get_particle_photo_lnu(
                    filters,
                    verbose,
                    nthreads=nthreads,
                )

        # Get the combined photometry
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_lnu[spectra] = self.spectra[spectra].get_photo_lnu(
                filters,
                verbose,
                nthreads=nthreads,
            )

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

        Photometry is calculated in spectral flux density units.

        Args:
            filters (object)
                A FilterCollection object.
            verbose (bool)
                Are we talking?
            nthreads (int)
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            PhotometryCollection
                A PhotometryCollection object containing the flux photometry
                in each filter in filters.
        """
        # Get stellar photometry
        if self.stars is not None:
            self.stars.get_photo_fnu(filters, verbose, nthreads=nthreads)

            # If we have particle spectra do that too (not applicable to
            # parametric Galaxy)
            if getattr(self.stars, "particle_spectra", None) is not None:
                self.stars.get_particle_photo_fnu(
                    filters,
                    verbose,
                    nthreads=nthreads,
                )

        # Get black hole photometry
        if self.black_holes is not None:
            self.black_holes.get_photo_fnu(filters, verbose, nthreads=nthreads)

            # If we have particle spectra do that too (not applicable to
            # parametric Galaxy)
            if getattr(self.black_holes, "particle_spectra", None) is not None:
                self.black_holes.get_particle_photo_fnu(
                    filters,
                    verbose,
                    nthreads=nthreads,
                )

        # Get the combined photometry
        for spectra in self.spectra:
            # Create the photometry collection and store it in the object
            self.photo_fnu[spectra] = self.spectra[spectra].get_photo_fnu(
                filters,
                verbose,
                nthreads=nthreads,
            )

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

    def plot_spectra(
        self,
        combined_spectra=True,
        stellar_spectra=False,
        gas_spectra=False,
        black_hole_spectra=False,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        quantity_to_plot="lnu",
    ):
        """
        Plots either specific observed spectra (specified via combined_spectra,
        stellar_spectra, gas_spectra, and/or black_hole_spectra) or all spectra
        for any of the spectra arguments that are True. If any are false that
        component is ignored.

        Args:
            combined_spectra (bool/list, string/string)
                The specific combined galaxy spectra to plot. (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra (bool/list, string/string)
                The specific stellar spectra to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra (bool/list, string/string)
                The specific gas spectra to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra (bool/list, string/string)
                The specific black hole spectra to plot. (e.g "blr")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool)
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple)
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100)
                times less than the peak of the spectrum for rest_frame
                (observed) spectra.
            xlimits (tuple)
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple)
                Tuple with size 2 defining the figure size.
            quantity_to_plot (string)
                The sed property to plot. Can be "lnu", "luminosity" or "llam"
                for rest frame spectra or "fnu", "flam" or "flux" for observed
                spectra. Defaults to "lnu".

        Returns:
            fig (matplotlib.pyplot.figure)
                The matplotlib figure object for the plot.
            ax (matplotlib.axes)
                The matplotlib axes object containing the plotted data.
        """
        # We need to construct the dictionary of all spectra to plot for each
        # component based on what we've been passed
        spectra = {}

        # Get the combined spectra
        if combined_spectra:
            if isinstance(combined_spectra, list):
                spectra.update(
                    {key: self.spectra[key] for key in combined_spectra}
                )
            elif isinstance(combined_spectra, Sed):
                spectra.update(
                    {
                        "combined_spectra": combined_spectra,
                    }
                )
            else:
                spectra.update(self.spectra)

        # Get the stellar spectra
        if stellar_spectra:
            if isinstance(stellar_spectra, list):
                spectra.update(
                    {
                        "Stellar " + key: self.stars.spectra[key]
                        for key in stellar_spectra
                    }
                )
            elif isinstance(stellar_spectra, Sed):
                spectra.update(
                    {
                        "stellar_spectra": stellar_spectra,
                    }
                )
            else:
                spectra.update(
                    {
                        "Stellar " + key: self.stars.spectra[key]
                        for key in self.stars.spectra
                    }
                )

        # Get the gas spectra
        if gas_spectra:
            if isinstance(gas_spectra, list):
                spectra.update(
                    {
                        "Gas " + key: self.gas.spectra[key]
                        for key in gas_spectra
                    }
                )
            elif isinstance(gas_spectra, Sed):
                spectra.update(
                    {
                        "gas_spectra": gas_spectra,
                    }
                )
            else:
                spectra.update(
                    {
                        "Gas " + key: self.gas.spectra[key]
                        for key in self.gas.spectra
                    }
                )

        # Get the black hole spectra
        if black_hole_spectra:
            if isinstance(black_hole_spectra, list):
                spectra.update(
                    {
                        "Black Hole " + key: self.black_holes.spectra[key]
                        for key in black_hole_spectra
                    }
                )
            elif isinstance(black_hole_spectra, Sed):
                spectra.update(
                    {
                        "black_hole_spectra": black_hole_spectra,
                    }
                )
            else:
                spectra.update(
                    {
                        "Black Hole " + key: self.black_holes.spectra[key]
                        for key in self.black_holes.spectra
                    }
                )

        return plot_spectra(
            spectra,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            quantity_to_plot=quantity_to_plot,
        )

    def plot_observed_spectra(
        self,
        combined_spectra=True,
        stellar_spectra=False,
        gas_spectra=False,
        black_hole_spectra=False,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        filters=None,
        quantity_to_plot="fnu",
    ):
        """
        Plots either specific observed spectra (specified via combined_spectra,
        stellar_spectra, gas_spectra, and/or black_hole_spectra) or all spectra
        for any of the spectra arguments that are True. If any are false that
        component is ignored.

        Args:
            combined_spectra (bool/list, string/string)
                The specific combined galaxy spectra to plot. (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra (bool/list, string/string)
                The specific stellar spectra to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra (bool/list, string/string)
                The specific gas spectra to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra (bool/list, string/string)
                The specific black hole spectra to plot. (e.g "blr")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool)
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple)
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100)
                times less than the peak of the spectrum for rest_frame
                (observed) spectra.
            xlimits (tuple)
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple)
                Tuple with size 2 defining the figure size.
            filters (FilterCollection)
                If given then the photometry is computed and both the
                photometry and filter curves are plotted
            quantity_to_plot (string)
                The sed property to plot. Can be "lnu", "luminosity" or "llam"
                for rest frame spectra or "fnu", "flam" or "flux" for observed
                spectra. Defaults to "lnu".

        Returns:
            fig (matplotlib.pyplot.figure)
                The matplotlib figure object for the plot.
            ax (matplotlib.axes)
                The matplotlib axes object containing the plotted data.
        """
        # We need to construct the dictionary of all spectra to plot for each
        # component based on what we've been passed
        spectra = {}

        # Get the combined spectra
        if combined_spectra:
            if isinstance(combined_spectra, list):
                spectra.update(
                    {key: self.spectra[key] for key in combined_spectra}
                )
            elif isinstance(combined_spectra, Sed):
                spectra.update(
                    {
                        "combined_spectra": combined_spectra,
                    }
                )
            else:
                spectra.update(self.spectra)

        # Get the stellar spectra
        if stellar_spectra:
            if isinstance(stellar_spectra, list):
                spectra.update(
                    {
                        "Stellar " + key: self.stars.spectra[key]
                        for key in stellar_spectra
                    }
                )
            elif isinstance(stellar_spectra, Sed):
                spectra.update(
                    {
                        "stellar_spectra": stellar_spectra,
                    }
                )
            else:
                spectra.update(
                    {
                        "Stellar " + key: self.stars.spectra[key]
                        for key in self.stars.spectra
                    }
                )

        # Get the gas spectra
        if gas_spectra:
            if isinstance(gas_spectra, list):
                spectra.update(
                    {
                        "Gas " + key: self.gas.spectra[key]
                        for key in gas_spectra
                    }
                )
            elif isinstance(gas_spectra, Sed):
                spectra.update(
                    {
                        "gas_spectra": gas_spectra,
                    }
                )
            else:
                spectra.update(
                    {
                        "Gas " + key: self.gas.spectra[key]
                        for key in self.gas.spectra
                    }
                )

        # Get the black hole spectra
        if black_hole_spectra:
            if isinstance(black_hole_spectra, list):
                spectra.update(
                    {
                        "Black Hole " + key: self.black_holes.spectra[key]
                        for key in black_hole_spectra
                    }
                )
            elif isinstance(black_hole_spectra, Sed):
                spectra.update(
                    {
                        "black_hole_spectra": black_hole_spectra,
                    }
                )
            else:
                spectra.update(
                    {
                        "Black Hole " + key: self.black_holes.spectra[key]
                        for key in self.black_holes.spectra
                    }
                )

        return plot_observed_spectra(
            spectra,
            self.redshift,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            filters=filters,
            quantity_to_plot=quantity_to_plot,
        )

    def get_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        covering_fraction=None,
        mask=None,
        vel_shift=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate spectra as described by the emission model.

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
            verbose (bool)
                Are we talking?
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                The combined spectra for the galaxy.
        """
        # Get the spectra
        spectra, particle_spectra = emission_model._get_spectra(
            emitters={"stellar": self.stars, "blackhole": self.black_holes},
            dust_curves=dust_curves,
            tau_v=tau_v,
            covering_fraction=covering_fraction,
            mask=mask,
            vel_shift=vel_shift,
            verbose=verbose,
            **kwargs,
        )

        # Unpack the spectra to the right component
        for model in emission_model._models.values():
            # Skip models we aren't saving
            if not model.save:
                continue
            if model.emitter == "galaxy":
                self.spectra[model.label] = spectra[model.label]
            elif model.emitter == "stellar":
                self.stars.spectra[model.label] = spectra[model.label]
            elif model.emitter == "blackhole":
                self.black_holes.spectra[model.label] = spectra[model.label]
            else:
                raise KeyError(
                    f"Unknown emitter in emission model. ({model.emitter})"
                )

            # If the model is particle based then we need to save the particle
            # spectra
            if model.per_particle:
                if model.emitter == "stellar":
                    self.stars.particle_spectra[model.label] = (
                        particle_spectra[model.label]
                    )
                elif model.emitter == "blackhole":
                    self.black_holes.particle_spectra[model.label] = (
                        particle_spectra[model.label]
                    )
                else:
                    raise KeyError(
                        "Unknown emitter in per particle "
                        f"emission model. ({model.emitter})"
                    )

        # Return the spectra at the root from the right place
        if emission_model.emitter == "galaxy":
            return self.spectra[emission_model.label]
        elif emission_model.emitter == "stellar":
            return self.stars.spectra[emission_model.label]
        elif emission_model.emitter == "blackhole":
            return self.black_holes.spectra[emission_model.label]
        else:
            raise KeyError(
                "Unknown emitter in emission model. "
                f"({emission_model.emitter})"
            )

    def get_lines(
        self,
        line_ids,
        emission_model,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate lines as described by the emission model.

        Args:
            line_ids (list):
                A list of line ids to include in the spectra.
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
            dict
                The combined lines for the galaxy.
        """
        # Get the lines
        lines, particle_lines = emission_model._get_lines(
            line_ids=line_ids,
            emitters={"stellar": self.stars, "blackhole": self.black_holes},
            dust_curves=dust_curves,
            tau_v=tau_v,
            covering_fraction=covering_fraction,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Unpack the lines to the right component
        for model in emission_model._models.values():
            # Skip models we aren't saving
            if not model.save:
                continue
            if model.emitter == "galaxy":
                self.lines[model.label] = lines[model.label]
            elif model.emitter == "stellar":
                self.stars.lines[model.label] = lines[model.label]
            elif model.emitter == "blackhole":
                self.black_holes.lines[model.label] = lines[model.label]
            else:
                raise KeyError(
                    f"Unknown emitter in emission model. ({model.emitter})"
                )

            # If the model is particle based then we need to save the particle
            # lines
            if model.per_particle:
                if model.emitter == "stellar":
                    self.stars.particle_lines[model.label] = particle_lines[
                        model.label
                    ]
                elif model.emitter == "blackhole":
                    self.black_holes.particle_lines[model.label] = (
                        particle_lines[model.label]
                    )
                else:
                    raise KeyError(
                        "Unknown emitter in per particle "
                        f"emission model. ({model.emitter})"
                    )

        # Return the lines at the root from the right place
        if emission_model.emitter == "galaxy":
            return self.lines[emission_model.label]
        elif emission_model.emitter == "stellar":
            return self.stars.lines[emission_model.label]
        elif emission_model.emitter == "blackhole":
            return self.black_holes.lines[emission_model.label]
        else:
            raise KeyError(
                "Unknown emitter in emission model. "
                f"({emission_model.emitter})"
            )

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
        Make an ImageCollection from luminosities.

        For Parametric Galaxy objects, images can only be smoothed. An
        exception will be raised if a histogram is requested.

        For Particle Galaxy objects, images can either be a simple
        histogram ("hist") or an image with particles smoothed over
        their SPH kernel.

        Which images are produced is defined by the emission model. If any
        of the necessary photometry is missing for generating a particular
        image, an exception will be raised.

        The limit_to argument can be used if only a specific image is desired.

        Note that black holes will never be smoothed and only produce a
        histogram due to the point source nature of black holes.

        All images that are created will be stored on the emitter (Stars,
        BlackHole/s, or galaxy) under the images_lnu attribute. The image
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
            limit_to (str)
                Optionally pass a single model label to limit image generation
                to only that model.
            instrument (Instrument)
                The instrument to use for the image. This can be None but if
                not it will be used to limit the included filters and label
                the images by instrument.

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Ensure we aren't trying to make a histogram for a parametric galaxy
        if self.galaxy_type == "Parametric" and img_type == "hist":
            raise exceptions.InconsistentArguments(
                "Parametric Galaxies can only produce smoothed images."
            )

        # If we haven't got an instrument create one
        if instrument is None:
            instrument = Instrument("place-holder", resolution=resolution)

        # Get the images
        images = emission_model._get_images(
            instrument=instrument,
            fov=fov,
            emitters={
                "stellar": self.stars,
                "blackhole": self.black_holes,
                "galaxy": self,
            },
            img_type=img_type,
            mask=None,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
            limit_to=limit_to,
            do_flux=False,
        )

        # Get the instrument name if we have one
        if instrument is not None:
            instrument_name = instrument.label
        else:
            instrument_name = None

        # Unpack the images to the right component
        for model in emission_model._models.values():
            # Are we limiting to a specific model?
            if limit_to is not None and model.label != limit_to:
                continue

            # Skip models we aren't saving
            if not model.save:
                continue

            # Attach the image to the right component
            if model.emitter == "galaxy":
                if instrument_name is not None:
                    self.images_lnu.setdefault(instrument_name, {})
                    self.images_lnu[instrument_name][model.label] = images[
                        model.label
                    ]
                else:
                    self.images_lnu[model.label] = images[model.label]
            elif model.emitter == "stellar":
                if instrument_name is not None:
                    self.stars.images_lnu.setdefault(instrument_name, {})
                    self.stars.images_lnu[instrument_name][model.label] = (
                        images[model.label]
                    )
                else:
                    self.stars.images_lnu[model.label] = images[model.label]
            elif model.emitter == "blackhole":
                if instrument_name is not None:
                    self.black_holes.images_lnu.setdefault(instrument_name, {})
                    self.black_holes.images_lnu[instrument_name][
                        model.label
                    ] = images[model.label]
                else:
                    self.black_holes.images_lnu[model.label] = images[
                        model.label
                    ]
            else:
                raise KeyError(
                    f"Unknown emitter in emission model. ({model.emitter})"
                )

        # If we are limiting to a specific image then return that
        if limit_to is not None and instrument_name is not None:
            return images[instrument_name][limit_to]
        elif limit_to is not None:
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

        For Parametric Galaxy objects, images can only be smoothed. An
        exception will be raised if a histogram is requested.

        For Particle Galaxy objects, images can either be a simple
        histogram ("hist") or an image with particles smoothed over
        their SPH kernel.

        Which images are produced is defined by the emission model. If any
        of the necessary photometry is missing for generating a particular
        image, an exception will be raised.

        The limit_to argument can be used if only a specific image is desired.

        Note that black holes will never be smoothed and only produce a
        histogram due to the point source nature of black holes.

        All images that are created will be stored on the emitter (Stars,
        BlackHole/s, or galaxy) under the images_fnu attribute. The image
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
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).
            nthreads (int)
                The number of threads to use in the tree search. Default is 1.
            limit_to (str)
                Optionally pass a single model label to limit image generation
                to only that model.
            instrument (Instrument)
                The instrument to use for the image. This can be None but if
                not it will be used to limit the included filters and label
                the images by instrument.

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Ensure we aren't trying to make a histogram for a parametric galaxy
        if self.galaxy_type == "Parametric" and img_type == "hist":
            raise exceptions.InconsistentArguments(
                "Parametric Galaxies can only produce smoothed images."
            )

        # If we haven't got an instrument create one
        if instrument is None:
            instrument = Instrument("place-holder", resolution=resolution)

        # Get the images
        images = emission_model._get_images(
            instrument=instrument,
            fov=fov,
            emitters={
                "stellar": self.stars,
                "blackhole": self.black_holes,
                "galaxy": self,
            },
            img_type=img_type,
            mask=None,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
            limit_to=limit_to,
            do_flux=True,
        )

        # Get the instrument name if we have one
        if instrument is not None:
            instrument_name = instrument.label
        else:
            instrument_name = None

        # Unpack the images to the right component
        for model in emission_model._models.values():
            # Are we limiting to a specific model?
            if limit_to is not None and model.label != limit_to:
                continue

            # Skip models we aren't saving
            if not model.save:
                continue

            # Attach the image to the right component
            if model.emitter == "galaxy":
                if instrument_name is not None:
                    self.images_fnu.setdefault(instrument_name, {})
                    self.images_fnu[instrument_name][model.label] = images[
                        model.label
                    ]
                else:
                    self.images_fnu[model.label] = images[model.label]
            elif model.emitter == "stellar":
                if instrument_name is not None:
                    self.stars.images_fnu.setdefault(instrument_name, {})
                    self.stars.images_fnu[instrument_name][model.label] = (
                        images[model.label]
                    )
                else:
                    self.stars.images_fnu[model.label] = images[model.label]
            elif model.emitter == "blackhole":
                if instrument_name is not None:
                    self.black_holes.images_fnu.setdefault(instrument_name, {})
                    self.black_holes.images_fnu[instrument_name][
                        model.label
                    ] = images[model.label]
                else:
                    self.black_holes.images_fnu[model.label] = images[
                        model.label
                    ]
            else:
                raise KeyError(
                    f"Unknown emitter in emission model. ({model.emitter})"
                )

        # If we are limiting to a specific image then return that
        if limit_to is not None and instrument_name is not None:
            return images[instrument_name][limit_to]
        elif limit_to is not None:
            return images[limit_to]

        # Return the image at the root of the emission model
        return images[emission_model.label]

    def clear_all_spectra(self):
        """
        Clear all spectra.

        This method is a quick helper to clear all spectra from the
        galaxy object and its components. This will cover both integrated and
        per particle spectra if present.
        """
        # Clear spectra
        self.spectra = {}
        if self.stars is not None:
            self.stars.clear_all_spectra()
        if self.black_holes is not None:
            self.black_holes.clear_all_spectra()

    def clear_all_lines(self):
        """
        Clear all lines.

        This method is a quick helper to clear all lines from the galaxy object
        and its components. This will cover both integrated and per particle
        lines if present.
        """
        # Clear lines
        self.lines = {}
        if self.stars is not None:
            self.stars.clear_all_lines()
        if self.black_holes is not None:
            self.black_holes.clear_all_lines()

    def clear_all_photometry(self):
        """
        Clear all photometry.

        This method is a quick helper to clear all photometry from the galaxy
        object and its components. This will cover both integrated and per
        particle photometry if present.
        """
        # Clear photometry
        self.photo_lnu = {}
        self.photo_fnu = {}
        if self.stars is not None:
            self.stars.clear_all_photometry()
        if self.black_holes is not None:
            self.black_holes.clear_all_photometry()

    def clear_all_emissions(self):
        """
        Clear all spectra, lines and photometry.

        This method is a quick helper to clear all spectra, lines, and
        photometry from the galaxy object and its components. This will cover
        both integrated and per particle emission.
        """
        # Clear spectra
        self.clear_all_spectra()

        # Clear lines
        self.clear_all_lines()

        # Clear photometry
        self.clear_all_photometry()

    def clear_weights(self):
        """
        Clear all cached grid weights.

        This clears all grid weights calculated using different
        methods from this base galaxy, and resets the
        `_grid_weights` dictionary.
        """
        if self.stars is not None:
            if hasattr(self.stars, "_grid_weights"):
                self._grid_weights = {"cic": {}, "ngp": {}}
        if self.black_holes is not None:
            if hasattr(self.black_holes, "_grid_weights"):
                self._grid_weights = {"cic": {}, "ngp": {}}
