""" A module for creating and manipulating parametric stellar populations.


"""
import h5py
import numpy as np
from scipy import integrate
from unyt import yr


import matplotlib.pyplot as plt
import cmasher as cmr

from synthesizer import exceptions
from synthesizer.components import StarsComponent
from synthesizer.stats import weighted_median, weighted_mean
from synthesizer.plt import single_histxy, mlabel


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
        ages (Quantity, array-like, float)
        log10ages_lims (array_like_float)
        metallicities (array-like, float)
        metallicities_lims (array-like, float)
        log10metallicities (array-like, float)
        log10metallicities_lims (array-like, float)
        sfzh (array-like, float)
        sf_hist (array-like, float)
        metal_hist (array-like, float)
        sf_hist_func (function)
        metal_hist_func (function)
        morphology (synthesizer.morphology.*)
        metallicity_grid_type (string)
    """

    def __init__(
            self,
            log10ages,
            metallicities,
            sfzh,
            morphology=None,
            sf_hist_func=None,
            metal_hist_func=None,
    ):
        """
        Initialise the parametric stellar population.

        Args:
            log10ages (array-like, float)
            metallicities (array-like, float)
            sfzh (array-like, float)
            morphology (synthesizer.morphology.*)
            sf_hist_func (function)
            metal_hist_func (function)
        """

        # Instantiate the parent
        StarsComponent(10 ** log10ages, metallicities)

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

        # Set the 2D star formation metallicity history grid
        self.sfzh = sfzh

        # Project the SFZH to get the 1D SFH
        self.sf_hist = np.sum(self.sfzh, axis=1)

        # Project the SFZH to get the 1D ZH
        self.metal_hist = np.sum(self.sfzh, axis=0)

        # Store the function used to make the star formation history if given
        self.sf_hist_func = sf_hist_func

        # Store the function used to make the metallicity history if given
        self.metal_hist_func = metal_hist_func

        # Attach the morphology model
        self.morphology = morphology

        # Check if metallicities are uniformly binned in log10metallicity or
        # linear metallicity or not at all (e.g. BPASS)
        if len(set(self.metallicities[:-1] - self.metallicities[1:])) == 1:
            # Regular linearly
            self.metallicity_grid_type = "Z"

        elif len(set(self.log10metallicities[:-1] - self.log10metallicities[1:])) == 1:
            # Regular in logspace
            self.metallicity_grid_type = "log10Z"

        else:
            # Irregular
            self.metallicity_grid_type = None

    def generate_lnu(self, grid, spectra_name, old=False, young=False):
        """
        Calculate rest frame spectra from an SPS Grid.

        This is a flexible base method which extracts the rest frame spectra of
        this galaxy from the SPS grid based on the passed arguments. More
        sophisticated types of spectra are produced by the get_spectra_*
        methods on BaseGalaxy, which call this method.

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
            The Galaxy's rest frame spectra in erg / s / Hz.
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

    def calculate_median_age(self):
        """calculate the median age"""

        return weighted_median(self.ages, self.sf_hist) * yr

    def calculate_mean_age(self):
        """calculate the mean age"""

        return weighted_mean(self.ages, self.sf_hist) * yr

    def calculate_mean_metallicity(self):
        """calculate the mean metallicity"""

        return weighted_mean(self.metallicities, self.metal_hist)

    def __str__(self):
        """print basic summary of the binned star formation and metal enrichment history"""

        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF BINNED SFZH" + "\n"
        pstr += f'median age: {self.calculate_median_age().to("Myr"):.2f}' + "\n"
        pstr += f'mean age: {self.calculate_mean_age().to("Myr"):.2f}' + "\n"
        pstr += f"mean metallicity: {self.calculate_mean_metallicity():.4f}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr

    def __add__(self, second_sfzh):
        """Add two SFZH histories together"""

        if second_sfzh.sfzh.shape == self.sfzh.shape:
            new_sfzh = self.sfzh + second_sfzh.sfzh

            return Stars(self.log10ages, self.metallicities, new_sfzh)

        else:
            raise exceptions.InconsistentAddition("SFZH must be the same shape")

    def plot(self, show=True):
        """Make a nice plots of the binned SZFH"""

        fig, ax, haxx, haxy = single_histxy()

        # this is technically incorrect because metallicity is not on a an actual grid.
        ax.pcolormesh(
            self.log10ages, self.log10metallicities, self.sfzh.T, cmap=cmr.sunburst
        )

        # --- add binned Z to right of the plot
        haxy.fill_betweenx(
            self.log10metallicities,
            self.metal_hist / np.max(self.metal_hist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # --- add binned SF_HIST to top of the plot
        haxx.fill_between(
            self.log10ages,
            self.sf_hist / np.max(self.sf_hist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # --- add SFR to top of the plot
        if self.sf_hist_func:
            x = np.linspace(*self.log10ages_lims, 1000)
            y = self.sf_hist_func.sfr(10**x)
            haxx.plot(x, y / np.max(y))

        haxy.set_xlim([0.0, 1.2])
        haxy.set_ylim(*self.log10metallicities_lims)
        haxx.set_ylim([0.0, 1.2])
        haxx.set_xlim(self.log10ages_lims)

        ax.set_xlabel(mlabel("log_{10}(age/yr)"))
        ax.set_ylabel(mlabel("log_{10}Z"))

        # Set the limits so all axes line up
        ax.set_ylim(*self.log10metallicities_lims)
        ax.set_xlim(*self.log10ages_lims)

        if show:
            plt.show()

        return fig, ax


def generate_sf_hist(ages, sf_hist_, log10=False):
    if log10:
        ages = 10**ages

    SF_HIST = np.zeros(len(ages))

    min_age = 0
    for ia, age in enumerate(ages[:-1]):
        max_age = int(np.mean([ages[ia + 1], ages[ia]]))  #  years
        sf = integrate.quad(sf_hist_.sfr, min_age, max_age)[0]
        SF_HIST[ia] = sf
        min_age = max_age

    # --- normalise
    SF_HIST /= np.sum(SF_HIST)

    return SF_HIST


def generate_instant_sfzh(
    log10ages, metallicities, log10age, metallicity, stellar_mass=1
):
    """simply returns the SFZH where only bin is populated corresponding to the age and metallicity"""

    sfzh = np.zeros((len(log10ages), len(metallicities)))
    ia = (np.abs(log10ages - log10age)).argmin()
    iZ = (np.abs(metallicities - metallicity)).argmin()
    sfzh[ia, iZ] = stellar_mass

    return Stars(log10ages, metallicities, sfzh)


def generate_sfzh(log10ages, metallicities, sf_hist, metal_hist, stellar_mass=1.0):
    """return an instance of the Stars class"""

    ages = 10**log10ages

    sfzh = np.zeros((len(log10ages), len(metallicities)))

    if metal_hist.dist == "delta":
        min_age = 0
        for ia, age in enumerate(ages[:-1]):
            max_age = int(np.mean([ages[ia + 1], ages[ia]]))  #  years
            sf = integrate.quad(sf_hist.sfr, min_age, max_age)[0]
            iZ = (np.abs(metallicities - metal_hist.Z(age))).argmin()
            sfzh[ia, iZ] = sf
            min_age = max_age

    if metal_hist.dist == "dist":
        print("WARNING: NOT YET IMPLEMENTED")

    # --- normalise
    sfzh /= np.sum(sfzh)
    sfzh *= stellar_mass

    return Stars(log10ages, metallicities, sfzh, sf_hist_func=sf_hist, metal_hist_func=metal_hist)


def generate_sfzh_from_array(log10ages, metallicities, sf_hist, metal_hist, stellar_mass=1.0):
    """
    Generated a Stars from an array instead of function
    """

    if not isinstance(metal_hist, np.ndarray):
        iZ = np.abs(metallicities - metal_hist).argmin()
        metal_hist = np.zeros(len(metallicities))
        metal_hist[iZ] = 1.0

    sfzh = sf_hist[:, np.newaxis] * metal_hist

    # --- normalise
    sfzh /= np.sum(sfzh)
    sfzh *= stellar_mass

    return Stars(log10ages, metallicities, sfzh)
