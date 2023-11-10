""" A module for creating and manipulating parametric stellar populations.

This is the parametric analog of particle.Stars. It not only computes and holds
the SFZH grid but everything describing a parametric Galaxy's stellar
component.

Example usage:

    stars = Stars(log10ages, metallicities,
                  sfzh=sfzh)
    stars.get_spectra_incident(grid)
    stars.plot_spectra()
"""
import numpy as np
from scipy import integrate
from unyt import yr, unyt_quantity


import matplotlib.pyplot as plt
import cmasher as cmr

from synthesizer import exceptions
from synthesizer.components import StarsComponent
from synthesizer.line import Line
from synthesizer.stats import weighted_median, weighted_mean
from synthesizer.plt import single_histxy, mlabel
from synthesizer.units import Quantity


class Stars(StarsComponent):
    """
    The parametric stellar population object.

    This class holds a binned star formation and metal enrichment history
    describing the age and metallicity of the stellar population, an
    optional morphology model describing the distribution of those stars,
    and various other important attributes for defining a parametric
    stellar population.

    Attributes:
        log10ages (array-like, float)
            The array of log10(ages) defining the age axis of the SFZH.
        ages (array-like, float)
            The array of ages defining the age axis of the SFZH.
        metallicities (array-like, float)
            The array of metallicitities defining the metallicity axies of
            the SFZH.
        log10metallicities (array-like, float)
            The array of log10(metallicitities) defining the metallicity axes
            of the SFZH.
        initial_mass (unyt_quantity/float)
            The total initial stellar mass.
        morphology (morphology.* e.g. Sersic2D)
            An instance of one of the morphology classes describing the
            stellar population's morphology. This can be any of the family
            of morphology classes from synthesizer.morphology.
        sfzh (array-like, float)
            An array describing the binned SFZH. If provided all following
            arguments are ignored.
        sf_hist (array-like, float)
            An array describing the star formation history.
        metal_hist (array-like, float)
            An array describing the metallity distribution.
        sf_hist_func (SFH.*)
            An instance of one of the child classes of SFH. This will be
            used to calculate sf_hist and takes precendence over a passed
            sf_hist if both are present.
        metal_hist_func (ZH.*)
            An instance of one of the child classes of ZH. This will be
            used to calculate metal_hist and takes precendence over a
            passed metal_hist if both are present.
        instant_sf (float)
            An age at which to compute an instantaneous SFH, i.e. all
            stellar mass populating a single SFH bin.
        instant_metallicity (float)
            A metallicity at which to compute an instantaneous ZH, i.e. all
            stellar populating a single ZH bin.
        log10ages_lims (array_like_float)
            The log10(age) limits of the SFZH grid.
        metallicities_lims (array-like, float)
            The metallicity limits of the SFZH grid.
        log10metallicities_lims (array-like, float)
            The log10(metallicity) limits of the SFZH grid.
        metallicity_grid_type (string)
            The type of gridding for the metallicity axis. Either:
                - Regular linear ("Z")
                - Regular logspace ("log10Z")
                - Irregular (None)
    """

    # Define quantities
    initial_mass = Quantity()

    def __init__(
            self,
            log10ages,
            metallicities,
            initial_mass=1.0,
            morphology=None,
            sfzh=None,
            sf_hist=None,
            metal_hist=None,
            sf_hist_func=None,
            metal_hist_func=None,
            instant_sf=None,
            instant_metallicity=None,
    ):
        """
        Initialise the parametric stellar population.

        Can either be instantiated by:
        - Passing a SFZH grid explictly.
        - Passing instant_sf and instant_metallicity to get an instantaneous
          SFZH.
        - Passing functions that describe the SFH and ZH.
        - Passing arrays that describe the SFH and ZH.
        - Passing any combination of SFH and ZH instant values, arrays
          or functions.

        Args:
            log10ages (array-like, float)
                The array of ages defining the log10(age) axis of the SFZH.
            metallicities (array-like, float)
                The array of metallicitities defining the metallicity axies of
                the SFZH.
            initial_mass (unyt_quantity/float)
                The total initial stellar mass.
            morphology (morphology.* e.g. Sersic2D)
                An instance of one of the morphology classes describing the
                stellar population's morphology. This can be any of the family
                of morphology classes from synthesizer.morphology.
            sfzh (array-like, float)
                An array describing the binned SFZH. If provided all following
                arguments are ignored.
            sf_hist (array-like, float)
                An array describing the star formation history.
            metal_hist (array-like, float)
                An array describing the metallity distribution.
            sf_hist_func (SFH.*)
                An instance of one of the child classes of SFH. This will be
                used to calculate sf_hist and takes precendence over a passed
                sf_hist if both are present.
            metal_hist_func (ZH.*)
                An instance of one of the child classes of ZH. This will be
                used to calculate metal_hist and takes precendence over a
                passed metal_hist if both are present.
            instant_sf (float)
                An age at which to compute an instantaneous SFH, i.e. all
                stellar mass populating a single SFH bin.
            instant_metallicity (float)
                A metallicity at which to compute an instantaneous ZH, i.e. all
                stellar populating a single ZH bin.
        """

        # Instantiate the parent
        StarsComponent.__init__(self, 10 ** log10ages, metallicities)

        # Set the age grid properties
        self.log10ages = log10ages
        self.log10ages_lims = [self.log10ages[0], self.log10ages[-1]]

        # Set the metallicity grid properties
        self.metallicities_lims = [self.metallicities[0], self.metallicities[-1]]
        self.log10metallicities = np.log10(metallicities)
        self.log10metallicities_lims = [
            self.log10metallicities[0],
            self.log10metallicities[-1],
        ]

        # Store the function used to make the star formation history if given
        self.sf_hist_func = sf_hist_func

        # Store the function used to make the metallicity history if given
        self.metal_hist_func = metal_hist_func

        # Store the SFH array (if None recalculated below)
        self.sf_hist = sf_hist

        # Store the ZH array (if None recalculated below)
        self.metal_hist = metal_hist

        # Store the total initial stellar mass
        self.initial_mass = initial_mass

        # If we have been handed an explict SFZH grid we can ignore all the
        # calculation methods
        if sfzh is not None:

            # Store the SFZH grid
            self.sfzh = sfzh

            # Project the SFZH to get the 1D SFH
            self.sf_hist = np.sum(self.sfzh, axis=1)

            # Project the SFZH to get the 1D ZH
            self.metal_hist = np.sum(self.sfzh, axis=0)

        else:

            # Set up the array ready for the calculation
            self.sfzh = np.zeros((len(log10ages), len(metallicities)))

            # Compute the SFZH grid
            self._get_sfzh(instant_sf, instant_metallicity)

        # Attach the morphology model
        self.morphology = morphology

        # Check if metallicities are uniformly binned in log10metallicity or
        # linear metallicity or not at all (e.g. BPASS)
        if len(set(self.metallicities[:-1] - self.metallicities[1:])) == 1:
            # Regular linearly
            self.metallicity_grid_type = "Z"

        elif len(set(
                self.log10metallicities[:-1] - self.log10metallicities[1:]
        )) == 1:
            # Regular in logspace
            self.metallicity_grid_type = "log10Z"

        else:
            # Irregular
            self.metallicity_grid_type = None

    def _get_sfzh(self, instant_sf, instant_metallicity):
        """
        Computes the SFZH for all possible combinations of input.

        If functions are passed for sf_hist_func and metal_hist_func then
        the SFH and ZH arrays are computed first.

        Args:
            instant_sf (unyt_quantity/float)
                An age at which to compute an instantaneous SFH, i.e. all
                stellar mass populating a single SFH bin.
            instant_metallicity (float)
                A metallicity at which to compute an instantaneous ZH, i.e. all
                stellar populating a single ZH bin.
        """

        # If no units assume unit system
        if instant_sf is not None and not isinstance(instant_sf, unyt_quantity):
            instant_sf *= self.ages.units

        # Handle the instantaneous SFH case
        if instant_sf is not None:

            # Create SFH array
            self.sf_hist = np.zeros(self.ages.size - 1)

            # Get the bin
            ia = (np.abs(self.ages - instant_sf)).argmin()
            self.sf_hist[ia] = self.initial_mass

        # A delta function for metallicity is a special case
        # equivalent to instant_metallicity = metal_hist_func.metallicity
        if self.metal_hist_func is not None:
            if self.metal_hist_func.dist == "delta":
                instant_metallicity = self.metal_hist_func.metallicity()

        # Handle the instantaneous ZH case
        if instant_metallicity is not None:

            # Create SFH array
            self.metal_hist = np.zeros(self.metallicities.size - 1)

            # Get the bin
            imetal = (np.abs(self.metallicities - instant_metallicity)).argmin()
            self.metal_hist[imetal] = self.initial_mass

        # Calculate SFH from function if necessary
        if self.sf_hist_func is not None and self.sf_hist is None:

            # Set up SFH array
            self.sf_hist = np.zeros(self.ages.size - 1)

            # Loop over age bins calculating the amount of mass in each bin
            min_age = 0
            for ia, age in enumerate(self.ages[:-1]):
                max_age = np.mean([self.ages[ia + 1], self.ages[ia]])
                sf = integrate.quad(self.sf_hist_func.sfr, min_age, max_age)[0]
                self.sf_hist[ia] = sf
                min_age = max_age

        # Calculate SFH from function if necessary
        if self.metal_hist_func is not None and self.metal_hist is None:

            # Set up SFH array
            self.metal_hist = np.zeros(self.metallicities.size - 1)

            # Loop over age bins calculating the amount of mass in each bin
            min_metal = 0
            for imetal, metal in enumerate(self.metallcities[:-1]):
                max_metal = np.mean(
                    [self.metallicities[ia + 1], self.metallicities[ia]]
                )
                sf = integrate.quad(
                    self.metal_hist_func.metallicity, min_metal, max_metal
                )[0]
                self.metal_hist[imetal] = sf
                min_metal = max_metal

        # Ensure that by this point we have an array for SFH and ZH
        if self.sf_hist is None or self.metal_hist is None:
            raise exceptions.InconsistentArguments(
                "A method for defining both the SFH and ZH must be provided!\n"
                "For each either an instantaneous"
                " value, a SFH/ZH object, or an array must be passed"
            )

        # Finally, calculate the SFZH grid based on the above calculations
        self.sfzh = self.sf_hist[:, np.newaxis] * self.metal_hist

        # Normalise the SFZH grid
        self.sfzh /= np.sum(self.sfzh)

        # ... and multiply it by the initial mass of stars
        self.sfzh *= self.initial_mass

    def generate_lnu(self, grid, spectra_name, old=False, young=False):
        """
        Calculate rest frame spectra from an SPS Grid.

        This is a flexible base method which extracts the rest frame spectra of
        this stellar popualtion from the SPS grid based on the passed
        arguments. More sophisticated types of spectra are produced by the
        get_spectra_* methods on StarsComponent, which call this method.

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra.
            spectra_name (str):
                A string denoting the desired type of spectra. Must match a
                key on the Grid.
            old (bool/float):
                Are we extracting only old stars? If so only SFZH bins with
                log10(Ages) > old will be included in the spectra. Defaults to
                False.
            young (bool/float):
                Are we extracting only young stars? If so only SFZH bins with
                log10(Ages) <= young will be included in the spectra. Defaults
                to False.

        Returns:
            The Stars's integrated rest frame spectra in erg / s / Hz.
        """

        # Ensure arguments make sense
        if old * young:
            raise ValueError("Cannot provide old and young stars together")

        # Get the indices of non-zero entries in the SFZH
        non_zero_inds = np.where(self.sfzh > 0)

        # Make the mask for relevent SFZH bins
        if old:
            sfzh_mask = self.log10ages[non_zero_inds[0]] > old
        elif young:
            sfzh_mask = self.log10ages[non_zero_inds[0]] <= young
        else:
            sfzh_mask = np.ones(
                len(self.log10ages[non_zero_inds[0]]),
                dtype=bool,
            )

        # Account for the SFZH mask in the non-zero indices
        non_zero_inds = (
            non_zero_inds[0][sfzh_mask], non_zero_inds[1][sfzh_mask]
        )

        # Compute the spectra
        spectra = np.sum(
            grid.spectra[spectra_name][non_zero_inds[0], non_zero_inds[1], :]
            * self.sfzh[non_zero_inds[0], non_zero_inds[1], :],
            axis=0,
        )

        return spectra

    def generate_line(self, grid, line_id, fesc):
        """
        Calculate rest frame line luminosity and continuum from an SPS Grid.

        This is a flexible base method which extracts the rest frame line
        luminosity of this stellar population from the SPS grid based on the
        passed arguments.

        Args:
            grid (Grid):
                A Grid object.
            line_id (list/str):
                A list of line_ids or a str denoting a single line.
                Doublets can be specified as a nested list or using a
                comma (e.g. 'OIII4363,OIII4959').
            fesc (float):
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped.

        Returns:
            Line
                An instance of Line contain this lines wavelenth, luminosity,
                and continuum.
        """

        # If the line_id is a str denoting a single line
        if isinstance(line_id, str):

            # Get the grid information we need
            grid_line = grid.lines[line_id]
            wavelength = grid_line["wavelength"]

            # Line luminosity erg/s
            luminosity = (1 - fesc) * np.sum(
                grid_line["luminosity"] * self.sfzh, axis=(0, 1)
            )

            # Continuum at line wavelength, erg/s/Hz
            continuum = np.sum(grid_line["continuum"] * self.sfzh, axis=(0, 1))

            # NOTE: this is currently incorrect and should be made of the
            # separated nebular and stellar continuum emission
            #
            # proposed alternative
            # stellar_continuum = np.sum(
            #     grid_line['stellar_continuum'] * self.sfzh.sfzh,
            #               axis=(0, 1))  # not affected by fesc
            # nebular_continuum = np.sum(
            #     (1-fesc)*grid_line['nebular_continuum'] * self.sfzh.sfzh,
            #               axis=(0, 1))  # affected by fesc

        # Else if the line is list or tuple denoting a doublet (or higher)
        elif isinstance(line_id, (list, tuple)):

            # Set up containers for the line information
            luminosity = []
            continuum = []
            wavelength = []

            # Loop over the ids in this container
            for line_id_ in line_id:
                grid_line = grid.lines[line_id_]

                # Wavelength [\AA]
                wavelength.append(grid_line["wavelength"])

                # Line luminosity erg/s
                luminosity.append(
                    (1 - fesc)
                    * np.sum(grid_line["luminosity"] * self.sfzh, axis=(0, 1))
                )

                # Continuum at line wavelength, erg/s/Hz
                continuum.append(
                    np.sum(grid_line["continuum"] * self.sfzh, axis=(0, 1))
                )

        else:
            raise exceptions.InconsistentArguments(
                "Unrecognised line_id! line_ids should contain strings"
                " or lists/tuples for doublets"
            )

        return Line(line_id, wavelength, luminosity, continuum)

    def calculate_median_age(self):
        """
        Calculate the median age of the stellar population.
        """
        return weighted_median(self.ages, self.sf_hist) * yr

    def calculate_mean_age(self):
        """
        Calculate the mean age of the stellar population.
        """
        return weighted_mean(self.ages, self.sf_hist) * yr

    def calculate_mean_metallicity(self):
        """
        Calculate the mean metallicity of the stellar population.
        """
        return weighted_mean(self.metallicities, self.metal_hist)

    def __str__(self):
        """
        Overload the print function to give a basic summary of the
        stellar population
        """

        # Define the output string
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF BINNED SFZH" + "\n"
        pstr += f'median age: {self.calculate_median_age().to("Myr"):.2f}' + "\n"
        pstr += f'mean age: {self.calculate_mean_age().to("Myr"):.2f}' + "\n"
        pstr += f"mean metallicity: {self.calculate_mean_metallicity():.4f}" + "\n"
        pstr += "-" * 10 + "\n"

        return pstr

    def __add__(self, other_stars):
        """
        Add two Stars instances together.

        In simple terms this sums the SFZH grids of both Stars instances.

        This will only work for Stars objects with the same SFZH grid axes.

        Args:
            other_stars (parametric.Stars)
                The other instance of Stars to add to this one.
        """

        if (np.all(self.log10ages == other_stars.log10ages) and
            np.all(self.metallicities == other_stars.metallicities)):
            new_sfzh = self.sfzh + other_stars.sfzh

        else:
            raise exceptions.InconsistentAddition("SFZH must be the same shape")

        return Stars(self.log10ages, self.metallicities, new_sfzh)

    def plot_sfzh(self, show=True):
        """
        Plot the binned SZFH.

        Args:
            show (bool)
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """

        # Create the figure and extra axes for histograms
        fig, ax, haxx, haxy = single_histxy()

        # Visulise the SFZH grid
        ax.pcolormesh(
            self.log10ages,
            self.log10metallicities,
            self.sfzh.T,
            cmap=cmr.sunburst,
        )

        # Add binned Z to right of the plot
        haxy.fill_betweenx(
            self.log10metallicities,
            self.metal_hist / np.max(self.metal_hist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Add binned SF_HIST to top of the plot
        haxx.fill_between(
            self.log10ages,
            self.sf_hist / np.max(self.sf_hist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Add SFR to top of the plot
        if self.sf_hist_func:
            x = np.linspace(*self.log10ages_lims, 1000)
            y = self.sf_hist_func.sfr(10**x)
            haxx.plot(x, y / np.max(y))

        # Set plot limits
        haxy.set_xlim([0.0, 1.2])
        haxy.set_ylim(*self.log10metallicities_lims)
        haxx.set_ylim([0.0, 1.2])
        haxx.set_xlim(self.log10ages_lims)

        # Set labels
        ax.set_xlabel(mlabel("log_{10}(age/yr)"))
        ax.set_ylabel(mlabel("log_{10}Z"))

        # Set the limits so all axes line up
        ax.set_ylim(*self.log10metallicities_lims)
        ax.set_xlim(*self.log10ages_lims)

        # Shall we show it?
        if show:
            plt.show()

        return fig, ax
