"""A module for creating and manipulating parametric stellar populations.

This is the parametric analog of particle.Stars. It not only computes and holds
the SFZH grid but everything describing a parametric Galaxy's stellar
component.

Example usage::

    stars = Stars(log10ages, metallicities, sfzh=sfzh)
    stars.get_spectra(emission_model)
    stars.plot_spectra()
"""

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from unyt import Hz, Msun, angstrom, erg, nJy, s, unyt_array, unyt_quantity, yr

from synthesizer import exceptions
from synthesizer.components.stellar import StarsComponent
from synthesizer.line import Line
from synthesizer.parametric.metal_dist import Common as ZDistCommon
from synthesizer.parametric.sf_hist import Common as SFHCommon
from synthesizer.units import Quantity, accepts
from synthesizer.utils.plt import single_histxy
from synthesizer.utils.stats import weighted_mean, weighted_median


class Stars(StarsComponent):
    """
    The parametric stellar population object.

    This class holds a binned star formation and metal enrichment history
    describing the age and metallicity of the stellar population, an
    optional morphology model describing the distribution of those stars,
    and various other important attributes for defining a parametric
    stellar population.

    Attributes:
        ages (array-like, float)
            The array of ages defining the age axis of the SFZH.
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
        metal_dist (array-like, float)
            An array describing the metallity distribution.
        sf_hist_func (SFH.*)
            An instance of one of the child classes of SFH. This will be
            used to calculate sf_hist and takes precendence over a passed
            sf_hist if both are present.
        metal_dist_func (ZH.*)
            An instance of one of the child classes of ZH. This will be
            used to calculate metal_dist and takes precendence over a
            passed metal_dist if both are present.
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

    @accepts(initial_mass=Msun.in_base("galactic"))
    def __init__(
        self,
        log10ages,
        metallicities,
        initial_mass=1.0,
        morphology=None,
        sfzh=None,
        sf_hist=None,
        metal_dist=None,
        **kwargs,
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
            sf_hist (float/unyt_quantity/array-like, float/SFH.*)
                Either:
                    - An age at which to compute an instantaneous SFH, i.e. all
                      stellar mass populating a single SFH bin.
                    - An array describing the star formation history.
                    - An instance of one of the child classes of SFH. This
                      will be used to calculate an array describing the SFH.
            metal_dist (float/unyt_quantity/array-like, float/ZDist.*)
                Either:
                    - A metallicity at which to compute an instantaneous
                      ZH, i.e. all stellar mass populating a single Z bin.
                    - An array describing the metallity distribution.
                    - An instance of one of the child classes of ZH. This
                      will be used to calculate an array describing the
                      metallicity distribution.
        """

        # Instantiate the parent
        StarsComponent.__init__(
            self,
            10**log10ages * yr,
            metallicities,
            **kwargs,
        )

        # Set the age grid lims
        self.log10ages_lims = [self.log10ages[0], self.log10ages[-1]]

        # Set the metallicity grid lims
        self.metallicities_lims = [
            self.metallicities[0],
            self.metallicities[-1],
        ]
        self.log10metallicities_lims = [
            self.log10metallicities[0],
            self.log10metallicities[-1],
        ]

        # Store the SFH we've been given, this is either...
        if issubclass(type(sf_hist), SFHCommon):
            self.sf_hist_func = sf_hist  # a SFH function
            self.sf_hist = None
            instant_sf = None
        elif isinstance(sf_hist, (unyt_quantity, float)):
            instant_sf = sf_hist  # an instantaneous SFH
            self.sf_hist_func = None
            self.sf_hist = None
        elif isinstance(sf_hist, (unyt_array, np.ndarray)):
            self.sf_hist = sf_hist  # a numpy array
            self.sf_hist_func = None
            instant_sf = None
        elif sf_hist is None:
            self.sf_hist = None  # we must have been passed a SFZH
            self.sf_hist_func = None
            instant_sf = None
        else:
            raise exceptions.InconsistentArguments(
                f"Unrecognised sf_hist type ({type(sf_hist)}! This should be"
                " either a float, an instance of a SFH function from the "
                "SFH module, or a single float."
            )

        # Store the metallicity distribution we've been given, either...
        if issubclass(type(metal_dist), ZDistCommon):
            self.metal_dist_func = metal_dist  # a ZDist function
            self.metal_dist = None
            instant_metallicity = None
        elif isinstance(metal_dist, (unyt_quantity, float, np.floating)):
            instant_metallicity = metal_dist  # an instantaneous SFH
            self.metal_dist_func = None
            self.metal_dist = None
        elif isinstance(metal_dist, (unyt_array, np.ndarray)):
            self.metal_dist = metal_dist  # a numpy array
            self.metal_dist_func = None
            instant_metallicity = None
        elif metal_dist is None:
            self.metal_dist = None  # we must have been passed a SFZH
            self.metal_dist_func = None
            instant_metallicity = None
        else:
            raise exceptions.InconsistentArguments(
                f"Unrecognised metal_dist type ({type(metal_dist)}! This "
                "should be either a float, an instance of a ZDist function "
                "from the ZDist module, or a single float."
            )

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
            self.metal_dist = np.sum(self.sfzh, axis=0)

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

        elif (
            len(
                set(self.log10metallicities[:-1] - self.log10metallicities[1:])
            )
            == 1
        ):
            # Regular in logspace
            self.metallicity_grid_type = "log10Z"

        else:
            # Irregular
            self.metallicity_grid_type = None

    def _get_sfzh(self, instant_sf, instant_metallicity):
        """
        Computes the SFZH for all possible combinations of input.

        If functions are passed for sf_hist_func and metal_dist_func then
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
        if instant_sf is not None and not isinstance(
            instant_sf, unyt_quantity
        ):
            instant_sf *= self.ages.units

        # Handle the instantaneous SFH case
        if instant_sf is not None:
            # Create SFH array
            self.sf_hist = np.zeros(self.ages.size)

            # Get the bin
            ia = (np.abs(self.ages - instant_sf)).argmin()
            self.sf_hist[ia] = self.initial_mass

        # A delta function for metallicity is a special case
        # equivalent to instant_metallicity = metal_dist_func.metallicity
        if self.metal_dist_func is not None:
            if self.metal_dist_func.name == "DeltaConstant":
                instant_metallicity = self.metal_dist_func.get_metallicity()

        # Handle the instantaneous ZH case
        if instant_metallicity is not None:
            # Create SFH array
            self.metal_dist = np.zeros(self.metallicities.size)

            # Get the bin
            imetal = (
                np.abs(self.metallicities - instant_metallicity)
            ).argmin()
            self.metal_dist[imetal] = self.initial_mass

        # Calculate SFH from function if necessary
        if self.sf_hist_func is not None and self.sf_hist is None:
            # Set up SFH array
            self.sf_hist = np.zeros(self.ages.size)

            # Loop over age bins calculating the amount of mass in each bin
            min_age = 0
            for ia, age in enumerate(self.ages[:-1]):
                max_age = np.mean([self.ages[ia + 1], self.ages[ia]])
                sf = integrate.quad(
                    self.sf_hist_func.get_sfr, min_age, max_age
                )[0]
                self.sf_hist[ia] = sf
                min_age = max_age

            # Normalise SFH array
            self.sf_hist /= np.sum(self.sf_hist)

            # Multiply by initial stellar mass
            self.sf_hist *= self._initial_mass

        # Calculate SFH from function if necessary
        if self.metal_dist_func is not None and self.metal_dist is None:
            # Set up SFH array
            self.metal_dist = np.zeros(self.metallicities.size)

            # Loop over metallicity bins calculating the amount of mass in
            # each bin
            min_metal = 0
            for imetal, metal in enumerate(self.metallicities[:-1]):
                max_metal = np.mean(
                    [
                        self.metallicities[imetal + 1],
                        self.metallicities[imetal],
                    ]
                )
                sf = integrate.quad(
                    self.metal_dist_func.get_dist_weight, min_metal, max_metal
                )[0]
                self.metal_dist[imetal] = sf
                min_metal = max_metal

            # Normalise ZH array
            self.metal_dist /= np.sum(self.metal_dist)

            # Multiply by initial stellar mass
            self.metal_dist *= self._initial_mass

        # Ensure that by this point we have an array for SFH and ZH
        if self.sf_hist is None or self.metal_dist is None:
            raise exceptions.InconsistentArguments(
                "A method for defining both the SFH and ZH must be provided!\n"
                "For each either an instantaneous"
                " value, a SFH/ZH object, or an array must be passed"
            )

        # Finally, calculate the SFZH grid based on the above calculations
        self.sfzh = self.sf_hist[:, np.newaxis] * self.metal_dist

        # Normalise the SFZH grid
        self.sfzh /= np.sum(self.sfzh)

        # ... and multiply it by the initial mass of stars
        self.sfzh *= self._initial_mass

    def get_mask(self, attr, thresh, op, mask=None):
        """
        Create a mask using a threshold and attribute on which to mask.

        Args:
            attr (str)
                The attribute to derive the mask from.
            thresh (float)
                The threshold value.
            op (str)
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=".
            mask (array)
                Optionally, a mask to combine with the new mask.

        Returns:
            mask (array)
                The mask array.
        """
        # Get the attribute
        attr = getattr(self, attr)

        # Apply the operator
        if op == ">":
            new_mask = attr > thresh
        elif op == "<":
            new_mask = attr < thresh
        elif op == ">=":
            new_mask = attr >= thresh
        elif op == "<=":
            new_mask = attr <= thresh
        elif op == "==":
            new_mask = attr == thresh
        elif op == "!=":
            new_mask = attr != thresh
        else:
            raise exceptions.InconsistentArguments(
                "Masking operation must be '<', '>', '<=', '>=', '==', or "
                f"'!=', not {op}"
            )

        # Broadcast the mask to get a mask for SFZH bins
        if new_mask.size == self.sfzh.shape[0]:
            new_mask = np.outer(
                new_mask, np.ones(self.sfzh.shape[1], dtype=int)
            )
        elif new_mask.size == self.sfzh.shape[1]:
            new_mask = np.outer(
                np.ones(self.sfzh.shape[0], dtype=int), new_mask
            )
        elif new_mask.shape == self.sfzh.shape:
            pass  # nothing to do here
        else:
            raise exceptions.InconsistentArguments(
                "Masking array must be the same shape as the SFZH grid "
                f"or an axis (mask.shape={new_mask.shape}, "
                f"sfzh.shape={self.sfzh.shape})"
            )

        # Combine with the existing mask
        if mask is not None:
            if mask.shape == new_mask.shape:
                new_mask = np.logical_and(new_mask, mask)
            else:
                raise exceptions.InconsistentArguments(
                    "Masking array must be the same shape as the SFZH grid "
                    f"or an axis (mask.shape={new_mask.shape}, "
                    f"sfzh.shape={self.sfzh.shape})"
                )

        return new_mask

    def generate_lnu(
        self,
        grid,
        spectra_name,
        old=None,
        young=None,
        mask=None,
        lam_mask=None,
        fesc=0.0,
        **kwargs,
    ):
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
            mask (array):
                An array to mask the SFZH grid. This can be used to mask
                specific SFZH bins.
            lam_mask (array, bool)
                A mask to apply to the wavelength array of the grid. This
                allows for the extraction of specific wavelength ranges.
            fesc (float)
                The Lyman continuum escape fraction, the fraction of
                ionising photons that entirely escape.

        Returns:
            The Stars's integrated rest frame spectra in erg / s / Hz.
        """
        # Ensure arguments make sense
        if old is not None and young is not None:
            raise ValueError("Cannot provide old and young stars together")

        # Get a mask for non-zero bins in the SFZH
        mask = self.get_mask("sfzh", 0, ">", mask=mask)

        # Make the mask for relevent SFZH bins if we haven't been handed one.
        if mask is not None:
            if old is not None:
                mask = self.get_mask("log10ages", old, ">", mask=mask)
            elif young is not None:
                mask = self.get_mask("log10ages", young, "<=", mask=mask)

        if fesc is None:
            fesc = 0.0

        # Add an extra dimension to enable later summation
        sfzh = np.expand_dims(self.sfzh, axis=2)

        # Get the grid spectra (including any wavelength mask)
        if lam_mask is not None:
            grid_spectra = grid.spectra[spectra_name][..., lam_mask]
        else:
            grid_spectra = grid.spectra[spectra_name]

        # Compute the spectra
        spectra = (1 - fesc) * np.sum(
            grid_spectra[mask] * sfzh[mask],
            axis=0,
        )

        # Apply the wavelength mask if provided
        if lam_mask is not None:
            out_spec = np.zeros(grid.lam.size)
            out_spec[lam_mask] = spectra
            spectra = out_spec

        return spectra

    def generate_line(self, grid, line_id, line_type, fesc, **kwargs):
        """
        Calculate rest frame line luminosity and continuum from an SPS Grid.

        This is a flexible base method which extracts the rest frame line
        luminosity of this stellar population from the SPS grid based on the
        passed arguments.

        Args:
            grid (Grid):
                A Grid object.
            line_id (str):
                A str denoting a line. Doublets can be specified using a
                comma (e.g. 'OIII4363,OIII4959').
            line_type (str):
                The type of line to extract. This must match a key in the Grid.
            fesc (float):
                The Lyman continuum escaped fraction, the fraction of
                ionising photons that entirely escaped.

        Returns:
            Line
                An instance of Line contain this lines wavelenth, luminosity,
                and continuum.
        """
        # Ensure line_id is a string
        if not isinstance(line_id, str):
            raise exceptions.InconsistentArguments("line_id must be a string")

        # Set up a list to hold each individual Line
        lines = []

        # Loop over the ids in this container
        for line_id_ in line_id.split(","):
            # Strip off any whitespace (can be left by split)
            line_id_ = line_id_.strip()

            # Get this line's wavelength
            # TODO: The units here should be extracted from the grid but aren't
            # yet stored.
            lam = grid.line_lams[line_id_] * angstrom

            # Line luminosity erg/s
            lum = (1 - fesc) * np.sum(
                grid.line_lums[line_type][line_id_] * self.sfzh, axis=(0, 1)
            )

            # Continuum at line wavelength, erg/s/Hz
            cont = (1 - fesc) * np.sum(
                grid.line_conts[line_type][line_id_] * self.sfzh, axis=(0, 1)
            )

            # Append this lines values to the containers
            lines.append(
                Line(
                    line_id=line_id_,
                    wavelength=lam,
                    luminosity=lum * erg / s,
                    continuum=cont * erg / s / Hz,
                )
            )

        # Don't init another line if there was only 1 in the first place
        if len(lines) == 1:
            return lines[0]
        else:
            return Line(combine_lines=lines)

    def calculate_median_age(self):
        """
        Calculate the median age of the stellar population.
        """
        return weighted_median(self.ages, self.sf_hist) * self.ages.units

    def calculate_mean_age(self):
        """
        Calculate the mean age of the stellar population.
        """
        return weighted_mean(self.ages, self.sf_hist)

    def calculate_mean_metallicity(self):
        """
        Calculate the mean metallicity of the stellar population.
        """
        return weighted_mean(self.metallicities, self.metal_dist)

    def __add__(self, other_stars):
        """
        Add two Stars instances together.

        In simple terms this sums the SFZH grids of both Stars instances.

        This will only work for Stars objects with the same SFZH grid axes.

        Args:
            other_stars (parametric.Stars)
                The other instance of Stars to add to this one.
        """

        if np.all(self.log10ages == other_stars.log10ages) and np.all(
            self.metallicities == other_stars.metallicities
        ):
            new_sfzh = self.sfzh + other_stars.sfzh

        else:
            raise exceptions.InconsistentAddition(
                "SFZH must be the same shape"
            )

        return Stars(self.log10ages, self.metallicities, sfzh=new_sfzh)

    def __radd__(self, other_stars):
        """
        Overloads "reflected" addition to allow two Stars instances to be added
        together when in reverse order, i.e. second_stars + self.

        This will only work for Stars objects with the same SFZH grid axes.

        Args:
            other_stars (parametric.Stars)
                The other instance of Stars to add to this one.
        """

        if np.all(self.log10ages == other_stars.log10ages) and np.all(
            self.metallicities == other_stars.metallicities
        ):
            new_sfzh = self.sfzh + other_stars.sfzh

        else:
            raise exceptions.InconsistentAddition(
                "SFZH must be the same shape"
            )

        return Stars(self.log10ages, self.metallicities, sfzh=new_sfzh)

    @accepts(lum=erg / s / Hz)
    def scale_mass_by_luminosity(self, lum, scale_filter, spectra_type):
        """
        Scale the mass of the stellar population to match a luminosity in a
        specific filter.

        NOTE: This will overwrite the initial mass attribute.

        Args:
            lum (unyt_quantity)
                The desried luminosity in scale_filter.
            scale_filter (Filter)
                The filter in which lum is measured.
            spectra_type (str)
                The spectra key with which to do this scaling, e.g. "incident"
                or "emergent".

        Raises
            MissingSpectraType
                If the requested spectra doesn't exist an error is thrown.
        """

        # Check we have the spectra
        if spectra_type not in self.spectra:
            raise exceptions.MissingSpectraType(
                f"The requested spectra type ({spectra_type}) does not exist"
                " in this stellar population. Have you called the "
                "corresponding spectra method?"
            )

        # Calculate the current luminosity in scale_filter
        sed = self.spectra[spectra_type]
        current_lum = (
            scale_filter.apply_filter(sed.lnu, nu=sed.nu) * sed.lnu.units
        )

        # Calculate the conversion ratio between the requested and current
        # luminosity
        conversion = lum / current_lum

        # Apply conversion to the masses
        self._initial_mass *= conversion

        # Apply the conversion to all spectra
        for key in self.spectra:
            self.spectra[key]._lnu *= conversion
            if self.spectra[key]._fnu is not None:
                self.spectra[key]._fnu *= conversion

        # Apply correction to the SFZH
        self.sfzh *= conversion

    @accepts(flux=nJy)
    def scale_mass_by_flux(self, flux, scale_filter, spectra_type):
        """
        Scale the mass of the stellar population to match a flux in a
        specific filter.

        NOTE: This will overwrite the initial mass attribute.

        Args:
            flux (unyt_quantity)
                The desried flux in scale_filter.
            scale_filter (Filter)
                The filter in which flux is measured.
            spectra_type (str)
                The spectra key with which to do this scaling, e.g. "incident"
                or "emergent".

        Raises
            MissingSpectraType
                If the requested spectra doesn't exist an error is thrown.
        """

        # Check we have the spectra
        if spectra_type not in self.spectra:
            raise exceptions.MissingSpectraType(
                f"The requested spectra type ({spectra_type}) does not exist"
                " in this stellar population. Have you called the "
                "corresponding spectra method?"
            )

        # Get the sed object
        sed = self.spectra[spectra_type]

        # Ensure we have a flux
        if sed.fnu is None:
            raise exceptions.MissingSpectraType(
                "{spectra_type} does not have a flux! Make sure to"
                " run Sed.get_fnu or Galaxy.get_observed_spectra"
            )

        # Calculate the current flux in scale_filter
        current_flux = (
            scale_filter.apply_filter(sed.fnu, nu=sed.obsnu) * sed.fnu.units
        )

        # Calculate the conversion ratio between the requested and current
        # flux
        conversion = flux / current_flux

        # Apply conversion to the masses
        self._initial_mass *= conversion

        # Apply the conversion to all spectra
        for key in self.spectra:
            self.spectra[key]._lnu *= conversion
            if self.spectra[key]._fnu is not None:
                self.spectra[key]._fnu *= conversion

        # Apply correction to the SFZH
        self.sfzh *= conversion

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
            self.metal_dist / np.max(self.metal_dist),
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
            y = self.sf_hist_func.get_sfr(10**x)
            haxx.plot(x, y / np.max(y))

        # Set plot limits
        haxy.set_xlim([0.0, 1.2])
        haxy.set_ylim(*self.log10metallicities_lims)
        haxx.set_ylim([0.0, 1.2])
        haxx.set_xlim(self.log10ages_lims)

        # Set labels
        ax.set_xlabel(r"$\log_{10}(\mathrm{age}/\mathrm{yr})$")
        ax.set_ylabel(r"$\log_{10}Z$")

        # Set the limits so all axes line up
        ax.set_ylim(*self.log10metallicities_lims)
        ax.set_xlim(*self.log10ages_lims)

        # Shall we show it?
        if show:
            plt.show()

        return fig, ax

    def _prepare_sed_args(self, *args, **kwargs):
        """Prepare arguments for SED generation."""
        raise exceptions.NotImplementedError(
            "Parametric stars don't currently require arg preparation"
        )

    def _prepare_line_args(self, *args, **kwargs):
        """Prepare arguments for line generation."""
        raise exceptions.NotImplementedError(
            "Parametric stars don't currently require arg preparation"
        )
