"""A module for common functionality in Parametric and Particle Galaxies.

The class described in this module should never be directly instatiated. It
only contains common attributes and methods to reduce boilerplate.
"""

from unyt import Mpc, arcsecond

from synthesizer import exceptions
from synthesizer.emission_models.attenuation import Inoue14
from synthesizer.emissions import Sed, plot_observed_spectra, plot_spectra
from synthesizer.instruments import Instrument
from synthesizer.synth_warnings import deprecated, deprecation
from synthesizer.units import accepts, unit_is_compatible
from synthesizer.utils import TableFormatter


class BaseGalaxy:
    """The base galaxy class.

    This should never be directly instantiated. It instead contains the common
    functionality and attributes needed for parametric and particle galaxies.

    Attributes:
        spectra (dict, Sed):
            The dictionary containing a Galaxy's spectra. Each entry is an
            Sed object. This dictionary only contains combined spectra from
            All components that make up the Galaxy (Stars, Gas, BlackHoles).
        stars (particle.Stars/parametric.Stars):
            The Stars object holding information about the stellar population.
        gas (particle.Gas/parametric.Gas):
            The Gas object holding information about the gas distribution.
        black_holes (particle.BlackHoles/parametric.BlackHole):
            The BlackHole/s object holding information about the black hole/s.
    """

    @accepts(centre=Mpc)
    def __init__(self, stars, gas, black_holes, redshift, centre, **kwargs):
        """Instantiate the base Galaxy class.

        This is the parent class of both parametric.Galaxy and particle.Galaxy.

        Note: The stars, gas, and black_holes component objects differ for
        parametric and particle galaxies but are attached at this parent level
        regardless to unify the Galaxy syntax for both cases.

        Args:
            stars (particle.Stars/parametric.Stars):
                The Stars object holding information about the stellar
                population.
            gas (particle.Gas/parametric.Gas):
                The Gas object holding information about the gas distribution.
            black_holes (particle.BlackHoles/parametric.BlackHole):
                The BlackHole/s object holding information about the
                black hole/s.
            redshift (float):
                The redshift of the galaxy.
            centre (unyt_array of float):
                The centre of the galaxy.
            **kwargs (dict):
                Any additional attributes to attach to the galaxy object.
        """
        # Container for the spectra and lines
        self.spectra = {}
        self.lines = {}

        # Initialise the photometry dictionaries
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

        # Initialise the dictionary to hold instrument specific spectroscopy
        self.spectroscopy = {}

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
        if self.centre is not None:
            if self.stars is not None:
                self.stars.centre = self.centre
            if self.gas is not None:
                self.gas.centre = self.centre
            if self.black_holes is not None:
                self.black_holes.centre = self.centre

    @property
    def photo_fluxes(self):
        """Get the photometry fluxes.

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
        """Get the photometry luminosities.

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
        """Return a string representation of the galaxy object.

        Returns:
            table (str):
                A string representation of the galaxy object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Galaxy")

    def get_equivalent_width(self, feature, blue, red, spectra_type):
        """Get all equivalent widths associated with a sed object.

        Args:
            feature (str):
                The feature to measure the equivalent width of.
                e.g. "Halpha", "Hbeta", "MgII", etc.
            blue (float):
                The blue side of the feature to measure.
            red (float):
                The red side of the feature to measure.
            spectra_type (str/list):
                The spectra type to measure the equivalent width of. Either
                a single type (str): or a list of types.

        Returns:
            equivalent_width (float/dict of float):
                The equivalent width of the feature in the spectra.
        """
        # If we only have one spectra type then just return the result
        if isinstance(spectra_type, str):
            return self.spectra[spectra_type].measure_index(feature, blue, red)

        # If we have a list of spectra types then loop over them and store
        # the results in a dictionary
        equivalent_widths = {}
        for sed_name in spectra_type:
            sed = self.spectra[sed_name]

            # Compute equivalent width
            equivalent_widths[sed_name] = sed.measure_index(feature, blue, red)

        return equivalent_widths

    def get_observed_spectra(self, cosmo, igm=Inoue14):
        """Calculate the observed spectra for all Seds within this galaxy.

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
            cosmo (astropy.cosmology.Cosmology):
                The cosmology object containing the cosmological model used
                to calculate the luminosity distance.
            igm (igm):
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
        """Calculate the observed lines for all Line objects.

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
            cosmo (astropy.cosmology.Cosmology):
                The cosmology object containing the cosmological model used
                to calculate the luminosity distance.
            igm (igm):
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
        """Combine all common spectra from components onto the galaxy.

        e.g.:
            intrinsc = stellar_intrinsic + black_hole_intrinsic.

        For any combined spectra all components with a valid spectra will be
        combined and stored in Galaxy.spectra under the same key, but only if
        there are instances of a spectra containing that name to combine.

        Possible combined spectra are:
            - "total"
            - "intrinsic"
            - "emergent"

        Note that this process is only applicable to integrated spectra.
        """
        # Get the spectra we have on the components to combine
        spectra = {"total": [], "intrinsic": [], "emergent": []}
        for key in spectra:
            if self.stars is not None:
                for component_key in self.stars.spectra:
                    if key in component_key:
                        spectra[key].append(self.stars.spectra[component_key])
            if self.black_holes is not None:
                for component_key in self.black_holes.spectra:
                    if key in component_key:
                        spectra[key].append(
                            self.black_holes.spectra[component_key]
                        )
            if self.gas is not None:
                for component_key in self.gas.spectra:
                    if key in component_key:
                        spectra[key].append(self.gas.spectra[component_key])

        # Now combine all spectra that have more than one contributing
        # component.
        # Note that sum when applied to a list of spectra
        # with overloaded __add__ methods will produce an Sed object
        # containing the combined spectra.
        for key, lst in spectra.items():
            if len(lst) > 1:
                self.spectra[key] = sum(lst)

    def get_photo_lnu(self, filters, verbose=True, nthreads=1):
        """Calculate luminosity photometry using a FilterCollection object.

        Photometry is calculated in spectral luminosity density units.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            PhotometryCollection:
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
        """Calculate luminosity photometry using a FilterCollection object.

        Alias to get_photo_lnu.

        Photometry is calculated in spectral luminosity density units.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?

        Returns:
            PhotometryCollection:
                A PhotometryCollection object containing the luminosity
                photometry in each filter in filters.
        """
        return self.get_photo_lnu(filters, verbose)

    def get_photo_fnu(self, filters, verbose=True, nthreads=1):
        """Calculate flux photometry using a FilterCollection object.

        Photometry is calculated in spectral flux density units.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1, all
                threads will be used.

        Returns:
            PhotometryCollection:
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
        """Plot spectra on this galaxy.

        Plots either specific observed spectra (specified via combined_spectra,
        stellar_spectra, gas_spectra, and/or black_hole_spectra) or all spectra
        for any of the spectra arguments that are True. If any are false that
        component is ignored.

        Args:
            combined_spectra (bool/list, string/string):
                The specific combined galaxy spectra to plot. (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra (bool/list, string/string):
                The specific stellar spectra to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra (bool/list, string/string):
                The specific gas spectra to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra (bool/list, string/string):
                The specific black hole spectra to plot. (e.g "blr")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool):
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple):
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100)
                times less than the peak of the spectrum for rest_frame
                (observed) spectra.
            xlimits (tuple):
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple):
                Tuple with size 2 defining the figure size.
            quantity_to_plot (string):
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
        """Plot observed spectra on this galaxy.

        Plots either specific observed spectra (specified via combined_spectra,
        stellar_spectra, gas_spectra, and/or black_hole_spectra) or all spectra
        for any of the spectra arguments that are True. If any are false that
        component is ignored.

        Args:
            combined_spectra (bool/list, string/string):
                The specific combined galaxy spectra to plot. (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra (bool/list, string/string):
                The specific stellar spectra to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra (bool/list, string/string):
                The specific gas spectra to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra (bool/list, string/string):
                The specific black hole spectra to plot. (e.g "blr")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool):
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple):
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100)
                times less than the peak of the spectrum for rest_frame
                (observed) spectra.
            xlimits (tuple):
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple):
                Tuple with size 2 defining the figure size.
            filters (FilterCollection):
                If given then the photometry is computed and both the
                photometry and filter curves are plotted
            quantity_to_plot (string):
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
        """Generate spectra as described by the emission model.

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
            covering_fraction (dict):
                An override to the emission model covering fraction. Either:
                    - None, indicating the covering fraction defined on the
                        emission model should be used.
                    - A float to use as the covering fraction for all models.
                    - A dictionary of the form
                        {<label>: float(<covering_fraction>)}
                        to use a specific covering fraction with a particular
                        model or {<label>: str(<attribute>)} to use an
                        attribute of the component as the covering fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            vel_shift (bool):
                An overide to the model level velocity shift flag. If True
                then the velocity shift is applied when generating all spectra.
            verbose (bool):
                Are we talking?
            kwargs (dict):
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
        """Generate lines as described by the emission model.

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
            covering_fraction (dict):
                An override to the emission model covering fraction. Either:
                    - None, indicating the covering fraction defined on the
                        emission model should be used.
                    - A float to use as the covering fraction for all models.
                    - A dictionary of the form
                        {<label>: float(<covering_fraction>)}
                        to use a specific covering fraction with a particular
                        model or {<label>: str(<attribute>)} to use an
                        attribute of the component as the covering fraction.
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
        cosmo=None,
    ):
        """Make an ImageCollection from luminosities.

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
            resolution (unyt_quantity of float):
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov (unyt_quantity of float):
                The width of the image in image coordinates.
            emission_model (EmissionModel):
                The emission model to use to generate the images.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel for a particle
                galaxy. Otherwise, only smoothed is applicable.
            stellar_photometry (str):
                The stellar spectra key from which to extract photometry
                to use for the image.
            blackhole_photometry (str):
                The black hole spectra key from which to extract photometry
                to use for the image.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.
            limit_to (str/list):
                Optionally pass a single model label to limit image generation
                to only that model.
            instrument (Instrument):
                The instrument to use for the image. This can be None but if
                not it will be used to limit the included filters and label
                the images by instrument.
            cosmo (astropy.cosmology):
                The cosmology to use for the calculation of the luminosity
                distance. Only needed for internal conversions from cartesian
                to angular coordinates when an angular resolution is used.

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
        # TODO: we need to eventually fully pivot to taking only an instrument
        # this will be done when we introduced some premade instruments
        if instrument is None:
            # Get the filters from the emitters
            if len(self.photo_lnu) > 0:
                filters = self.photo_lnu[emission_model.label].filters
            elif self.stars is not None and len(self.stars.photo_lnu) > 0:
                filters = self.stars.photo_lnu[emission_model.label].filters
            elif (
                self.black_holes is not None
                and len(self.black_holes.photo_lnu) > 0
            ):
                filters = self.black_holes.photo_lnu[
                    emission_model.label
                ].filters

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
                    "Redshift must be set on a Galaxy when using an angular "
                    "resolution and FOV."
                )

        # Convert `limit_to` to a list if it is a string
        limit_to = [limit_to] if isinstance(limit_to, str) else limit_to

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
            cosmo=cosmo,
        )

        # Get the instrument name if we have one
        if instrument is not None:
            instrument_name = instrument.label
        else:
            instrument_name = None

        # Unpack the images to the right component
        for model in emission_model._models.values():
            # Are we limiting to a specific model?
            if limit_to is not None and model.label not in limit_to:
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
        if limit_to is not None:
            return images[limit_to[0]]  # return the first image in list

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
            resolution (unyt_quantity of float):
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov (unyt_quantity of float):
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
            limit_to (str, list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.
            instrument (Instrument):
                The instrument to use for the image. This can be None but if
                not it will be used to limit the included filters and label
                the images by instrument.
            cosmo (astropy.cosmology):
                The cosmology to use for the calculation of the luminosity
                distance. Only needed for internal conversions from cartesian
                to angular coordinates when an angular resolution is used.

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
        # TODO: we need to eventually fully pivot to taking only an instrument
        # this will be done when we introduced some premade instruments
        if instrument is None:
            # Get the filters from the emitters
            if len(self.photo_fnu) > 0:
                filters = self.photo_fnu[emission_model.label].filters
            elif len(self.stars.photo_fnu) > 0:
                filters = self.stars.photo_fnu[emission_model.label].filters
            elif len(self.black_holes.photo_fnu) > 0:
                filters = self.black_holes.photo_fnu[
                    emission_model.label
                ].filters
            instrument = Instrument(
                "place-holder", resolution=resolution, filters=filters
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
                    "Redshift must be set on a Galaxy when using an angular "
                    "resolution and FOV."
                )

        # Convert `limit_to` to a list if it is a string
        limit_to = [limit_to] if isinstance(limit_to, str) else limit_to

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
            cosmo=cosmo,
        )

        # Get the instrument name if we have one
        if instrument is not None:
            instrument_name = instrument.label
        else:
            instrument_name = None

        # Unpack the images to the right component
        for model in emission_model._models.values():
            # Are we limiting to a specific model?
            if limit_to is not None and model.label not in limit_to:
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
        if limit_to is not None:
            return images[limit_to[0]]  # return the first image in list

        # Return the image at the root of the emission model
        return images[emission_model.label]

    def apply_psf_to_images_lnu(
        self,
        instrument,
        psf_resample_factor=1,
        limit_to=None,
    ):
        """Apply instrument PSFs to this galaxy's luminosity images.

        This will also apply the PSF to any images attached to the galaxies
        components, as well as those on the top level galaxy object.

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

        # Get the images we are applying the PSF to
        if instrument.label in self.images_lnu:
            images = self.images_lnu[instrument.label]
        else:
            # If it's not either dict we just don't have any images on the
            # galaxy but we might on the component, so no error!
            images = {}

        # Make an entry for this instrument in the images_psf_lnu dict
        # if it doesn't exist
        if instrument.label not in self.images_psf_lnu:
            self.images_psf_lnu[instrument.label] = {}

        # Do the galaxy level images
        for key in images:
            # Are we limiting to a specific model?
            if limit_to is not None and key not in limit_to:
                continue

            # Unpack the image
            imgs = images[key]

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

        # If we have stars, do those
        if (
            self.stars is not None
            and instrument.label in self.stars.images_lnu
        ):
            self.stars.apply_psf_to_images_lnu(
                instrument,
                psf_resample_factor=psf_resample_factor,
                limit_to=limit_to,
            )

        # If we have black holes, do those
        if (
            self.black_holes is not None
            and instrument.label in self.black_holes.images_lnu
        ):
            self.black_holes.apply_psf_to_images_lnu(
                instrument,
                psf_resample_factor=psf_resample_factor,
                limit_to=limit_to,
            )

        return self.images_psf_lnu[instrument.label]

    def apply_psf_to_images_fnu(
        self,
        instrument,
        psf_resample_factor=1,
        limit_to=None,
    ):
        """Apply instrument PSFs to this galaxy's flux images.

        This will also apply the PSF to any images attached to the galaxies
        components, as well as those on the top level galaxy object.

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

        # Get the images we are applying the PSF to
        if instrument.label in self.images_fnu:
            images = self.images_fnu[instrument.label]
        else:
            # If it's not either dict we just don't have any images on the
            # galaxy but we might on the component, so no error!
            images = {}

        # Make an entry for this instrument in the images_psf_fnu dict
        # if it doesn't exist
        if instrument.label not in self.images_psf_fnu:
            self.images_psf_fnu[instrument.label] = {}

        # Do the galaxy level images
        for key in images:
            # Are we limiting to a specific model?
            if limit_to is not None and key not in limit_to:
                continue

            # Unpack the image
            imgs = images[key]

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

        # If we have stars, do those
        if (
            self.stars is not None
            and instrument.label in self.stars.images_fnu
        ):
            self.stars.apply_psf_to_images_fnu(
                instrument,
                psf_resample_factor=psf_resample_factor,
                limit_to=limit_to,
            )

        # If we have black holes, do those
        if (
            self.black_holes is not None
            and instrument.label in self.black_holes.images_fnu
        ):
            self.black_holes.apply_psf_to_images_fnu(
                instrument,
                psf_resample_factor=psf_resample_factor,
                limit_to=limit_to,
            )

        return self.images_psf_fnu[instrument.label]

    def apply_noise_to_images_lnu(
        self,
        instrument,
        limit_to=None,
        apply_to_psf=True,
    ):
        """Apply instrument noise to this galaxy's and its component's images.

        Args:
            instrument (Instrument):
                The instrument with the noise to apply.
            limit_to (str/list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.
            apply_to_psf (bool):
                If True, apply the noise to the PSF images. Otherwise, apply
                it to the normal images.

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
            # If it's not either dict we just don't have any images on the
            # galaxy but we might on the component, so no error!
            images = {}

        # Make an entry for this instrument in the images_noise_lnu dict
        # if it doesn't exist
        if instrument.label not in self.images_noise_lnu:
            self.images_noise_lnu[instrument.label] = {}

        # Do the galaxy level images
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
                    "for applying noise because no noise attributes are set."
                )

        # If we have stars, do those
        if self.stars is not None and (
            instrument.label in self.stars.images_lnu
            or (apply_to_psf and instrument.label in self.stars.images_psf_lnu)
        ):
            self.stars.apply_noise_to_images_lnu(
                instrument,
                limit_to=limit_to,
                apply_to_psf=apply_to_psf,
            )

        # If we have black holes, do those
        if self.black_holes is not None and (
            instrument.label in self.black_holes.images_lnu
            or (
                apply_to_psf
                and instrument.label in self.black_holes.images_psf_lnu
            )
        ):
            self.black_holes.apply_noise_to_images_lnu(
                instrument,
                limit_to=limit_to,
                apply_to_psf=apply_to_psf,
            )

        return self.images_noise_lnu[instrument.label]

    def apply_noise_to_images_fnu(
        self,
        instrument,
        limit_to=None,
        apply_to_psf=True,
    ):
        """Apply instrument noise to this galaxy's and its component's images.

        Args:
            instrument (Instrument):
                The instrument with the noise to apply.
            limit_to (str/list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.
            apply_to_psf (bool):
                If True, apply the noise to the PSF images. Otherwise, apply
                it to the normal images.

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
            # If it's not either dict we just don't have any images on the
            # galaxy but we might on the component, so no error!
            images = {}

        # Make an entry for this instrument in the images_noise_fnu dict
        # if it doesn't exist
        if instrument.label not in self.images_noise_fnu:
            self.images_noise_fnu[instrument.label] = {}

        # Do the galaxy level images
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
                    "for applying noise because no noise attributes are set."
                )

        # If we have stars, do those
        if self.stars is not None and (
            instrument.label in self.stars.images_fnu
            or (apply_to_psf and instrument.label in self.stars.images_psf_fnu)
        ):
            self.stars.apply_noise_to_images_fnu(
                instrument,
                limit_to=limit_to,
                apply_to_psf=apply_to_psf,
            )

        # If we have black holes, do those
        if self.black_holes is not None and (
            instrument.label in self.black_holes.images_fnu
            or (
                apply_to_psf
                and instrument.label in self.black_holes.images_psf_fnu
            )
        ):
            self.black_holes.apply_noise_to_images_fnu(
                instrument,
                limit_to=limit_to,
                apply_to_psf=apply_to_psf,
            )

        return self.images_noise_fnu[instrument.label]

    def get_spectroscopy(
        self,
        instrument,
    ):
        """Get spectroscopy for the galaxy based on a specific instrument.

        This will apply the instrument's wavelength array to each
        spectra stored on the galaxy and its components.

        Args:
            instrument (Instrument):
                The instrument to use for the spectroscopy.

        Returns:
            dict
                The spectroscopy for the galaxy.
        """
        # Check we have some spectra
        nspec = len(self.spectra)
        if self.stars is not None:
            nspec += len(self.stars.spectra)
            if hasattr(self.stars, "particle_spectra"):
                nspec += len(self.stars.particle_spectra)
        if self.black_holes is not None:
            nspec += len(self.black_holes.spectra)
            if hasattr(self.black_holes, "particle_spectra"):
                nspec += len(self.black_holes.particle_spectra)
        if nspec == 0:
            raise exceptions.InconsistentArguments(
                "No spectra found in galaxy or components."
            )

        # Create an entry for this instrument in the spectroscopy dictionary
        # if it doesn't exist
        if instrument.label not in self.spectroscopy:
            self.spectroscopy[instrument.label] = {}

        # Do the galaxy level spectra
        for key, sed in self.spectra.items():
            # Get the spectroscopy
            self.spectroscopy[instrument.label][key] = (
                sed.apply_instrument_lams(instrument)
            )

        # Do the stars level spectra
        if self.stars is not None:
            self.stars.get_spectroscopy(instrument)

        # Do the black holes level spectra
        if self.black_holes is not None:
            self.black_holes.get_spectroscopy(instrument)

        return self.spectroscopy[instrument.label]

    def clear_all_spectra(self):
        """Clear all spectra.

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

    def clear_all_spectroscopy(self):
        """Clear all spectroscopy.

        This method is a quick helper to clear all spectroscopy from the
        galaxy object and its components. This will cover both integrated and
        per particle spectroscopy if present.
        """
        # Clear spectroscopy
        self.spectroscopy = {}
        if self.stars is not None:
            self.stars.clear_all_spectroscopy()
        if self.black_holes is not None:
            self.black_holes.clear_all_spectroscopy()

    def clear_all_lines(self):
        """Clear all lines.

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
        """Clear all photometry.

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
        """Clear all spectra, lines and photometry.

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

        # Clear spectroscopy
        self.clear_all_spectroscopy()

    def clear_weights(self):
        """Clear all cached grid weights.

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

    def plot_spectroscopy(
        self,
        instrument_label,
        combined_spectra=True,
        stellar_spectra=False,
        gas_spectra=False,
        black_hole_spectra=False,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        quantity_to_plot="lnu",
        fig=None,
        ax=None,
    ):
        """Plot an instrument's spectroscopy.

        This will plot the spectroscopy for the galaxy and its components
        using the instrument's wavelength array. The spectra are plotted
        in the order they are stored in the spectroscopy dictionary.

        Args:
            instrument_label (str):
                The label of the instrument whose spectroscopy to plot.
            combined_spectra (bool/list, string/string):
                The specific combined galaxy spectra to plot. (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra (bool/list, string/string):
                The specific stellar spectra to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra (bool/list, string/string):
                The specific gas spectra to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra (bool/list, string/string):
                The specific black hole spectra to plot. (e.g "blr")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool):
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple):
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100)
                times less than the peak of the spectrum for rest_frame
                (observed) spectra.
            xlimits (tuple):
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple):
                Tuple with size 2 defining the figure size.
            quantity_to_plot (string):
                The sed property to plot. Can be "lnu", "luminosity" or "llam"
                for rest frame spectra or "fnu", "flam" or "flux" for observed
                spectra. Defaults to "lnu".
            fig (matplotlib.pyplot.figure):
                The matplotlib figure object to plot on. If None a new
                figure is created.
            ax (matplotlib.axes):
                The matplotlib axes object to plot on. If None a new
                axes is created.

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
                    {
                        key: self.spectroscopy[instrument_label][key]
                        for key in combined_spectra
                    }
                )
            elif isinstance(combined_spectra, Sed):
                spectra.update(
                    {
                        "combined_spectra": combined_spectra,
                    }
                )
            else:
                spectra.update(self.spectroscopy[instrument_label])

        # Get the stellar spectra
        if stellar_spectra:
            if isinstance(stellar_spectra, list):
                spectra.update(
                    {
                        "Stellar " + key: self.stars.spectroscopy[
                            instrument_label
                        ][key]
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
                        "Stellar " + key: self.stars.spectroscopy[
                            instrument_label
                        ][key]
                        for key in self.stars.spectroscopy[instrument_label]
                    }
                )

        # Get the gas spectra
        if gas_spectra:
            if isinstance(gas_spectra, list):
                spectra.update(
                    {
                        "Gas " + key: self.gas.spectroscopy[instrument_label][
                            key
                        ]
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
                        "Gas " + key: self.gas.spectroscopy[instrument_label][
                            key
                        ]
                        for key in self.gas.spectroscopy[instrument_label]
                    }
                )

        # Get the black hole spectra
        if black_hole_spectra:
            if isinstance(black_hole_spectra, list):
                spectra.update(
                    {
                        "Black Hole " + key: self.black_holes.spectroscopy[
                            instrument_label
                        ][key]
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
                        "Black Hole " + key: self.black_holes.spectroscopy[
                            instrument_label
                        ][key]
                        for key in self.black_holes.spectroscopy[
                            instrument_label
                        ]
                    }
                )

        # Add the instrument to the key (label) of each spectra
        old_keys = list(spectra.keys())
        for key in old_keys:
            spectra[f"{instrument_label}: {key}"] = spectra[key]
            del spectra[key]

        return plot_spectra(
            spectra,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            quantity_to_plot=quantity_to_plot,
            fig=fig,
            ax=ax,
        )

    def plot_observed_spectroscopy(
        self,
        instrument_label,
        combined_spectra=True,
        stellar_spectra=False,
        gas_spectra=False,
        black_hole_spectra=False,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        quantity_to_plot="fnu",
        fig=None,
        ax=None,
    ):
        """Plot an instrument's spectroscopy.

        This will plot the spectroscopy for the galaxy and its components
        using the instrument's wavelength array. The spectra are plotted
        in the order they are stored in the spectroscopy dictionary.

        Args:
            instrument_label (str):
                The label of the instrument whose spectroscopy to plot.
            combined_spectra (bool/list, string/string):
                The specific combined galaxy spectroscopy to plot.
                (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra (bool/list, string/string):
                The specific stellar spectroscopy to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra (bool/list, string/string):
                The specific gas spectroscopy to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra (bool/list, string/string):
                The specific black hole spectroscopy to plot. (e.g "blr")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            show (bool):
                Flag for whether to show the plot or just return the
                figure and axes.
            ylimits (tuple):
                The limits to apply to the y axis. If not provided the limits
                will be calculated with the lower limit set to 1000 (100)
                times less than the peak of the spectrum for rest_frame
                (observed) spectra.
            xlimits (tuple):
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple):
                Tuple with size 2 defining the figure size.
            quantity_to_plot (string):
                The sed property to plot. Can be "lnu", "luminosity" or "llam"
                for rest frame spectra or "fnu", "flam" or "flux" for observed
                spectra. Defaults to "lnu".
            fig (matplotlib.pyplot.figure):
                The matplotlib figure object to plot on. If None a new
                figure is created.
            ax (matplotlib.axes):
                The matplotlib axes object to plot on. If None a new
                axes is created.

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
                    {
                        key: self.spectroscopy[instrument_label][key]
                        for key in combined_spectra
                    }
                )
            elif isinstance(combined_spectra, Sed):
                spectra.update(
                    {
                        "combined_spectra": combined_spectra,
                    }
                )
            else:
                spectra.update(self.spectroscopy[instrument_label])

        # Get the stellar spectra
        if stellar_spectra:
            if isinstance(stellar_spectra, list):
                spectra.update(
                    {
                        "Stellar " + key: self.stars.spectroscopy[
                            instrument_label
                        ][key]
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
                        "Stellar " + key: self.stars.spectroscopy[
                            instrument_label
                        ][key]
                        for key in self.stars.spectroscopy[instrument_label]
                    }
                )

        # Get the gas spectra
        if gas_spectra:
            if isinstance(gas_spectra, list):
                spectra.update(
                    {
                        "Gas " + key: self.gas.spectroscopy[instrument_label][
                            key
                        ]
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
                        "Gas " + key: self.gas.spectroscopy[instrument_label][
                            key
                        ]
                        for key in self.gas.spectroscopy[instrument_label]
                    }
                )

        # Get the black hole spectra
        if black_hole_spectra:
            if isinstance(black_hole_spectra, list):
                spectra.update(
                    {
                        "Black Hole " + key: self.black_holes.spectroscopy[
                            instrument_label
                        ][key]
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
                        "Black Hole " + key: self.black_holes.spectroscopy[
                            instrument_label
                        ][key]
                        for key in self.black_holes.spectroscopy[
                            instrument_label
                        ]
                    }
                )

        # Add the instrument to the key (label) of each spectra
        old_keys = list(spectra.keys())
        for key in old_keys:
            spectra[f"{instrument_label}: {key}"] = spectra[key]
            del spectra[key]

        return plot_observed_spectra(
            spectra,
            self.redshift,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            quantity_to_plot=quantity_to_plot,
            fig=fig,
            ax=ax,
        )
