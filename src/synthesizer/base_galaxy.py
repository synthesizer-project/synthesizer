""" A module for common functionality in Parametric and Particle Galaxies

The class described in this module should never be directly instatiated. It
only contains common attributes and methods to reduce boilerplate.
"""
import numpy as np
import matplotlib.pyplot as plt

from synthesizer.sed import Sed, plot_spectra, plot_observed_spectra


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

    def __init__(self, stars, gas, black_holes, **kwargs):
        """
        Instantiate the base Galaxy class.

        This is the parent class of both parametric.Galaxy and particle.Galaxy.

        Note: The stars, gas, and black_holes component objects differ for
        parametric and particle galaxies but are attached at this parent level
        regardless to unify the Galaxy syntax for both cases.

        Args:

        """
        # Add some place holder attributes which are overloaded on the children
        self.spectra = {}

        # Attach the components
        self.stars = stars
        self.gas = gas
        self.black_holes = black_holes

        if isinstance(self, BaseGalaxy):
            raise Warning(
                "Instantiating a BaseGalaxy object is not "
                "supported behaviour. Instead, you should "
                "use one of the derived Galaxy classes:\n"
                "`particle.galaxy.Galaxy`\n"
                "`parametric.galaxy.Galaxy`"
            )

    def get_spectra_dust(self, emissionmodel):
        """
        Calculates dust emission spectra using the attenuated and intrinsic
        spectra that have already been generated and an emission model.

        Parameters
        ----------
        emissionmodel : obj
            The spectral frid

        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        # use wavelength grid from attenuated spectra
        # NOTE: in future it might be good to allow a custom wavelength grid

        lam = self.spectra["emergent"].lam

        # calculate the bolometric dust lunminosity as the difference between
        # the intrinsic and attenuated

        dust_bolometric_luminosity = (
            self.spectra["intrinsic"].measure_bolometric_luminosity()
            - self.spectra["emergent"].measure_bolometric_luminosity()
        )

        # get the spectrum and normalise it properly
        lnu = dust_bolometric_luminosity.to("erg/s").value * emissionmodel.lnu(lam)

        # create new Sed object containing dust spectra
        sed = Sed(lam, lnu=lnu)

        # associate that with the component's spectra dictionarity
        self.spectra["dust"] = sed
        self.spectra["total"] = self.spectra["dust"] + self.spectra["emergent"]

        return sed

    def get_equivalent_width(self, feature, blue, red, spectra_to_plot=None):
        """
        Gets all equivalent widths associated with a sed object

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

    def transmission(self):
        """
        Calculate transmission as a function of wavelength

        Returns:
            transmission (array)
        """

        return (
            self.spectra["attenuated"].lam,
            self.spectra["attenuated"].lnu / self.spectra["intrinsic"].lnu,
        )

    def Al(self):
        """
        Calculate attenuation as a function of wavelength

        Returns:
            attenuation (array)
        """

        lam, transmission = self.transmission()

        return lam, -2.5 * np.log10(transmission)

    def A(self, l):
        """
        Calculate attenuation at a given wavelength

        Returns:
            attenuation (float)
        """

        lam, Al = self.Al()

        return np.interp(l, lam, Al)

    def AV(self):
        """
        Calculate rest-frame FUV attenuation

        Returns:
            attenuation at rest-frame 1500 angstrom (float)
        """

        return self.A(5500.0)

    def A1500(self):
        """
        Calculate rest-frame FUV attenuation

        Returns:
            attenuation at rest-frame 1500 angstrom (float)
        """

        return self.A(1500.0)

    def plot_spectra(
        self,
        spectra_to_plot=True,
        stellar_spectra_to_plot=False,
        gas_spectra_to_plot=False,
        black_hole_spectra_to_plot=False,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
    ):
        """
        Plots either a specific spectra or all spectra provided in a dictionary.

        This is a generic plotting function to be used either directly or to be
        wrapped by helper methods through Synthesizer.

        Args:
            spectra_to_plot (bool/list, string/string)
                The specific combined galaxy spectra to plot. (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra_to_plot (bool/list, string/string)
                The specific stellar spectra to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra_to_plot (bool/list, string/string)
                The specific gas spectra to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra_to_plot (bool/list, string/string)
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
        if spectra_to_plot:
            if isinstance(spectra_to_plot, list):
                spectra.update({self.spectra[key] for key in spectra_to_plot})
            else:
                spectra.update(self.spectra)

        # Get the stellar spectra
        if stellar_spectra_to_plot:
            if isinstance(stellar_spectra_to_plot, list):
                spectra.update(
                    {self.stars.spectra[key] for key in stellar_spectra_to_plot}
                )
            else:
                spectra.update(self.stars.spectra)

        # Get the gas spectra
        if gas_spectra_to_plot:
            if isinstance(gas_spectra_to_plot, list):
                spectra.update({self.gas.spectra[key] for key in gas_spectra_to_plot})
            else:
                spectra.update(self.gas.spectra)

        # Get the black hole spectra
        if black_hole_spectra_to_plot:
            if isinstance(black_hole_spectra_to_plot, list):
                spectra.update(
                    {
                        self.black_holes.spectra[key]
                        for key in black_hole_spectra_to_plot
                    }
                )
            else:
                spectra.update(self.black_holes.spectra)

        return plot_spectra(
            spectra,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
        )

    def plot_observed_spectra(
        self,
        spectra_to_plot=True,
        stellar_spectra_to_plot=False,
        gas_spectra_to_plot=False,
        black_hole_spectra_to_plot=False,
        show=False,
        ylimits=(),
        xlimits=(),
        figsize=(3.5, 5),
        filters=None,
    ):
        """
        Plots either a specific spectra or all spectra provided in a dictionary.

        This is a generic plotting function to be used either directly or to be
        wrapped by helper methods through Synthesizer.

        Args:
            spectra_to_plot (bool/list, string/string)
                The specific combined galaxy spectra to plot. (e.g "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            stellar_spectra_to_plot (bool/list, string/string)
                The specific stellar spectra to plot. (e.g. "incident")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            gas_spectra_to_plot (bool/list, string/string)
                The specific gas spectra to plot. (e.g. "total")
                    - If True all spectra are plotted.
                    - If a list of strings each specifc spectra is plotted.
                    - If a single string then only that spectra is plotted.
            black_hole_spectra_to_plot (bool/list, string/string)
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
                If given then the photometry is computed and both the photometry
                and filter curves are plotted

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
        if spectra_to_plot:
            if isinstance(spectra_to_plot, list):
                spectra.update({key: self.spectra[key] for key in spectra_to_plot})
            else:
                spectra.update(self.spectra)

        # Get the stellar spectra
        if stellar_spectra_to_plot:
            if isinstance(stellar_spectra_to_plot, list):
                spectra.update(
                    {
                        "Stellar " + key: self.stars.spectra[key]
                        for key in stellar_spectra_to_plot
                    }
                )
            else:
                spectra.update(self.stars.spectra)

        # Get the gas spectra
        if gas_spectra_to_plot:
            if isinstance(gas_spectra_to_plot, list):
                spectra.update(
                    {"Gas " + key: self.gas.spectra[key] for key in gas_spectra_to_plot}
                )
            else:
                spectra.update(self.gas.spectra)

        # Get the black hole spectra
        if black_hole_spectra_to_plot:
            if isinstance(black_hole_spectra_to_plot, list):
                spectra.update(
                    {
                        "Black Hole" + key: self.black_holes.spectra[key]
                        for key in black_hole_spectra_to_plot
                    }
                )
            else:
                spectra.update(self.black_holes.spectra)

        return plot_observed_spectra(
            spectra,
            show=show,
            ylimits=ylimits,
            xlimits=xlimits,
            figsize=figsize,
            draw_legend=isinstance(spectra, dict),
            filters=filters,
        )
