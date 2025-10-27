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
from scipy.interpolate import RegularGridInterpolator
from unyt import Hz, Msun, erg, nJy, s, unyt_array, unyt_quantity, yr

from synthesizer import exceptions
from synthesizer.components.stellar import StarsComponent
from synthesizer.grid import Grid
from synthesizer.parametric.metal_dist import Common as ZDistCommon
from synthesizer.parametric.sf_hist import Common as SFHCommon
from synthesizer.units import Quantity, accepts
from synthesizer.utils.plt import single_histxy
from synthesizer.utils.stats import weighted_mean, weighted_median


class Stars(StarsComponent):
    """The parametric stellar population object.

    This class holds a binned star formation and metal enrichment history
    describing the age and metallicity of the stellar population, an
    optional morphology model describing the distribution of those stars,
    and various other important attributes for defining a parametric
    stellar population.

    Attributes:
        ages (np.ndarray of float):
            The array of ages defining the age axis of the SFZH.
        metallicities (np.ndarray of float):
            The array of metallicitities defining the metallicity axes of
            the SFZH.
        initial_mass (unyt_quantity/float)
            The total initial stellar mass.
        morphology (morphology.* e.g. Sersic2D)
            An instance of one of the morphology classes describing the
            stellar population's morphology. This can be any of the family
            of morphology classes from synthesizer.morphology.
        sfzh (np.ndarray of float):
            An array describing the binned SFZH. If provided all following
            arguments are ignored.
        sf_hist (np.ndarray of float):
            An array describing the star formation history.
        metal_dist (np.ndarray of float):
            An array describing the metallicity distribution.
        sf_hist_func (SFH.*)
            An instance of one of the child classes of SFH. This will be
            used to calculate sf_hist and takes precedence over a passed
            sf_hist if both are present.
        metal_dist_func (ZH.*)
            An instance of one of the child classes of ZH. This will be
            used to calculate metal_dist and takes precedence over a
            passed metal_dist if both are present.
        instant_sf (float):
            An age at which to compute an instantaneous SFH, i.e. all
            stellar mass populating a single SFH bin.
        instant_metallicity (float):
            A metallicity at which to compute an instantaneous ZH, i.e. all
            stellar populating a single ZH bin.
        log10ages_lims (array_like_float)
            The log10(age) limits of the SFZH grid.
        metallicities_lims (np.ndarray of float):
            The metallicity limits of the SFZH grid.
        log10metallicities_lims (np.ndarray of float):
            The log10(metallicity) limits of the SFZH grid.
        metallicity_grid_type (str):
            The type of gridding for the metallicity axis. Either:
                - Regular linear ("Z")
                - Regular logspace ("log10Z")
                - Irregular (None)
    """

    # Define quantities
    initial_mass = Quantity("mass")

    @accepts(initial_mass=Msun.in_base("galactic"))
    def __init__(
        self,
        log10ages,
        metallicities,
        initial_mass=None,
        morphology=None,
        sfzh=None,
        sf_hist=None,
        metal_dist=None,
        fesc=None,
        fesc_ly_alpha=None,
        **kwargs,
    ):
        """Initialise the parametric stellar population.

        Can either be instantiated by:
        - Passing a SFZH grid explictly.
        - Passing instant_sf and instant_metallicity to get an instantaneous
          SFZH.
        - Passing functions that describe the SFH and ZH.
        - Passing arrays that describe the SFH and ZH.
        - Passing any combination of SFH and ZH instant values, arrays
          or functions.

        Args:
            log10ages (np.ndarray of float):
                The array of ages defining the log10(age) axis of the SFZH.
            metallicities (np.ndarray of float):
                The array of metallicitities defining the metallicity axies of
                the SFZH.
            initial_mass (unyt_quantity/float):
                The total initial stellar mass. If provided the SFZH grid will
                be rescaled to obey this total mass.
            morphology (morphology.* e.g. Sersic2D):
                An instance of one of the morphology classes describing the
                stellar population's morphology. This can be any of the family
                of morphology classes from synthesizer.morphology.
            sfzh (np.ndarray of float):
                An array describing the binned SFZH. If provided all following
                arguments are ignored.
            sf_hist (float/unyt_quantity/np.ndarray of float/SFH.*):
                Either:
                    - An age at which to compute an instantaneous SFH, i.e. all
                      stellar mass populating a single SFH bin.
                    - An array describing the star formation history.
                    - An instance of one of the child classes of SFH. This
                      will be used to calculate an array describing the SFH.
            metal_dist (float/unyt_quantity/np.ndarray of float/ZDist.*):
                Either:
                    - A metallicity at which to compute an instantaneous
                      ZH, i.e. all stellar mass populating a single Z bin.
                    - An array describing the metallity distribution.
                    - An instance of one of the child classes of ZH. This
                      will be used to calculate an array describing the
                      metallicity distribution.
            fesc (float):
                The escape fraction of incident radiation from the stars.
            fesc_ly_alpha (float):
                The escape fraction of Ly-alpha radiation from the stars.
            **kwargs (dict):
                Arbitrary keyword arguments to be set as attributes on the
                Stars instance.
        """
        # Instantiate the parent
        StarsComponent.__init__(
            self,
            10**log10ages * yr,
            metallicities,
            _star_type="parametric",
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
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

            # It's somewhat nonsensical to have both an SFZH grid and
            # set the initial mass, but if the user has lets rescale the SFZH
            # to obey their initial mass request
            if self.initial_mass is not None:
                # Normalise the SFZH grid
                self.sfzh /= np.sum(self.sfzh)

                # ... and multiply it by the initial mass of stars
                self.sfzh *= self._initial_mass
            else:
                # Otherwise calculate the total initial mass
                self._initial_mass = np.sum(self.sfzh)

            # Project the SFZH to get the 1D SFH
            self.sf_hist = np.sum(self.sfzh, axis=1)

            # Project the SFZH to get the 1D ZH
            self.metal_dist = np.sum(self.sfzh, axis=0)

        else:
            # Set up the array ready for the calculation
            self.sfzh = np.zeros((len(log10ages), len(metallicities)))

            # Compute the SFZH grid
            self._get_sfzh(instant_sf, instant_metallicity)

        if np.any(~np.isfinite(self.sfzh)):
            raise exceptions.InconsistentArguments(
                "SFZH grid contains NaN or Inf values! "
                "Please check the input parameters."
            )

        # Attach the morphology model
        self.morphology = morphology

        # Check if metallicities are uniformly binned in log10metallicity or
        # linear metallicity or not at all (e.g. BPASS)
        if (
            len(np.unique(self.metallicities[:-1] - self.metallicities[1:]))
            == 1
        ):
            # Regular linearly
            self.metallicity_grid_type = "Z"

        elif (
            len(
                np.unique(
                    self.log10metallicities[:-1] - self.log10metallicities[1:]
                )
            )
            == 1
        ):
            # Regular in logspace
            self.metallicity_grid_type = "log10Z"

        else:
            # Irregular
            self.metallicity_grid_type = None

    @accepts(instant_sf=yr)
    def _get_sfzh(self, instant_sf, instant_metallicity):
        """Compute the SFZH for all possible combinations of input.

        If functions are passed for sf_hist_func and metal_dist_func then
        the SFH and ZH arrays are computed first.

        Args:
            instant_sf (unyt_quantity/float):
                An age at which to compute an instantaneous SFH, i.e. all
                stellar mass populating a single SFH bin. Note, this must
                be the age itself, not the log10(age).
            instant_metallicity (float):
                A metallicity at which to compute an instantaneous ZH, i.e. all
                stellar populating a single ZH bin. Note, this must be the
                metallicity itself, not the log10(metallicity).
        """
        # Hide imports to avoid cyclic imports
        from synthesizer.particle import Stars as ParticleStars

        # If no units assume unit system
        if instant_sf is not None and not isinstance(
            instant_sf, unyt_quantity
        ):
            instant_sf *= self.ages.units

        # A delta function for metallicity is a special case
        # equivalent to instant_metallicity = metal_dist_func.metallicity
        if self.metal_dist_func is not None:
            if self.metal_dist_func.name == "DeltaConstant":
                instant_metallicity = self.metal_dist_func.get_metallicity()

        # If both are instantaneous then we can do the whole SFZH in one go
        if instant_sf is not None and instant_metallicity is not None:
            inst_stars = ParticleStars(
                initial_masses=np.array([self._initial_mass]) * Msun,
                ages=np.array([instant_sf.to("yr").value]) * yr,
                metallicities=np.array([instant_metallicity]),
            )

            # Compute the SFZH grid
            self.sfzh = inst_stars.get_sfzh(
                self.log10ages,
                self.metallicities,
                grid_assignment_method="cic",
            ).sfzh

            # Compute the SFH and ZH arrays
            self.sf_hist = np.sum(self.sfzh, axis=1)
            self.metal_dist = np.sum(self.sfzh, axis=0)

            return

        # Handle the instantaneous SFH case
        elif instant_sf is not None and instant_metallicity is None:
            inst_stars = ParticleStars(
                initial_masses=np.array([self._initial_mass]) * Msun,
                ages=np.array([instant_sf.to("yr").value]) * yr,
                metallicities=np.array([0]),  # this is a dummy value
            )

            # Create SFH array
            self.sf_hist = inst_stars.get_sfh(self.log10ages)

        # Handle the instantaneous ZH case
        elif instant_metallicity is not None and instant_sf is None:
            inst_stars = ParticleStars(
                initial_masses=np.array([self._initial_mass]) * Msun,
                ages=np.array([0]) * yr,  # this is a dummy value
                metallicities=np.array([instant_metallicity]),
            )

            # Create metal distribution array
            self.metal_dist = inst_stars.get_metal_dist(self.metallicities)

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

        # Normalise the SFZH grid if needs be
        if self.initial_mass is not None:
            self.sfzh /= np.sum(self.sfzh)

            # ... and multiply it by the initial mass of stars
            self.sfzh *= self._initial_mass
        else:
            # Otherwise calculate the total initial mass
            self.initial_mass = np.sum(self.sfzh) * Msun

    def get_mask(self, attr, thresh, op, mask=None):
        """Create a mask using a threshold and attribute on which to mask.

        Args:
            attr (str):
                The attribute to derive the mask from.
            thresh (float):
                The threshold value.
            op (str):
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=".
            mask (np.ndarray):
                Optionally, a mask to combine with the new mask.

        Returns:
            mask (np.ndarray):
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
                new_mask, np.ones(self.sfzh.shape[1], dtype=bool)
            )
        elif new_mask.size == self.sfzh.shape[1]:
            new_mask = np.outer(
                np.ones(self.sfzh.shape[0], dtype=bool), new_mask
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

    def calculate_median_age(self):
        """Calculate the median age of the stellar population."""
        return weighted_median(self.ages, self.sf_hist) * self.ages.units

    def calculate_mean_age(self):
        """Calculate the mean age of the stellar population."""
        return weighted_mean(self.ages, self.sf_hist)

    def calculate_mean_metallicity(self):
        """Calculate the mean metallicity of the stellar population."""
        return weighted_mean(self.metallicities, self.metal_dist)

    def __add__(self, other_stars):
        """Add two Stars instances together.

        In simple terms this sums the SFZH grids of both Stars instances.

        This will only work for Stars objects with the same SFZH grid axes.

        Args:
            other_stars (parametric.Stars):
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
        """Add two Stars instances together (reflected addition).

        Overloads "reflected" addition to allow two Stars instances to be added
        together when in reverse order, i.e. second_stars + self.

        This will only work for Stars objects with the same SFZH grid axes.

        Args:
            other_stars (parametric.Stars):
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
        """Scale the stellar mass to match a luminosity in a specific filter.

        NOTE: This will overwrite the initial mass attribute.

        Args:
            lum (unyt_quantity):
                The desried luminosity in scale_filter.
            scale_filter (Filter):
                The filter in which lum is measured.
            spectra_type (str):
                The spectra key with which to do this scaling, e.g. "incident"
                or "emergent".

        Raises:
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
        """Scale the stellar mass to match a flux in a specific filter.

        NOTE: This will overwrite the initial mass attribute.

        Args:
            flux (unyt_quantity):
                The desried flux in scale_filter.
            scale_filter (Filter):
                The filter in which flux is measured.
            spectra_type (str):
                The spectra key with which to do this scaling, e.g. "incident"
                or "emergent".

        Raises:
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

    def get_sfzh(
        self,
        log10ages,
        metallicities,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """Generate the binned SFZH history of this stellar component.

        In the parametric case this will resample the existing SFZH onto the
        desired grid. For a particle based component the binned SFZH is
        calculated by binning the particles onto the desired grid defined by
        the input log10ages and metallicities.


        For a particle based galaxy the binned SFZH produced by this method
        is equivalent to the weights used to extract spectra from the grid.

        Args:
            log10ages (np.ndarray of float):
                The log10 ages of the desired SFZH.
            metallicities (np.ndarray of float):
                The metallicities of the desired SFZH.
            grid_assignment_method (str):
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or their uppercase equivalents (CIC, NGP).
                Defaults to cic. (particle only)
            nthreads (int):
                The number of threads to use in the computation. If set to -1
                all available threads will be used. (particle only)

        Returns:
            numpy.ndarray:
                Numpy array of containing the SFZH.
        """
        # Prepare an interpolator based on the existing SFZH
        interp = RegularGridInterpolator(
            (self.log10ages, self.metallicities),
            self.sfzh,
            bounds_error=False,
            fill_value=0.0,
        )

        # Build a mesh containing the new grid points
        age_mesh, metal_mesh = np.meshgrid(
            log10ages, metallicities, indexing="ij"
        )

        # Interpolate the SFZH onto the new grid
        points = np.column_stack([age_mesh.ravel(), metal_mesh.ravel()])
        new_values = interp(points)  # shape is (N,)

        # Reshape interpolated values onto the new grid shape
        new_sfzh = new_values.reshape(len(log10ages), len(metallicities))

        return Stars(
            log10ages,
            metallicities,
            sfzh=new_sfzh,
            initial_mass=self.initial_mass,
        )

    def plot_sfzh(
        self,
        show=True,
    ):
        """Plot the binned SZFH.

        Args:
            show (bool):
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
        metal_dist = np.sum(self.sfzh, axis=0)
        haxy.fill_betweenx(
            self.log10metallicities,
            metal_dist / np.max(metal_dist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Add binned SF_HIST to top of the plot
        sf_hist = np.sum(self.sfzh, axis=1)
        haxx.fill_between(
            self.log10ages,
            sf_hist / np.max(sf_hist),
            step="mid",
            color="k",
            alpha=0.3,
        )

        # Set plot limits
        haxy.set_xlim([0.0, 1.2])
        haxy.set_ylim(self.log10metallicities[0], self.log10metallicities[-1])
        haxx.set_ylim([0.0, 1.2])
        haxx.set_xlim(self.log10ages[0], self.log10ages[-1])

        # Set labels
        ax.set_xlabel(r"$\log_{10}(\mathrm{age}/\mathrm{yr})$")
        ax.set_ylabel(r"$\log_{10}(Z)$")

        # Set the limits so all axes line up
        ax.set_ylim(self.log10metallicities[0], self.log10metallicities[-1])
        ax.set_xlim(self.log10ages[0], self.log10ages[-1])

        # Shall we show it?
        if show:
            plt.show()

        return fig, ax

    def get_sfh(self):
        """Get the star formation history of the stellar population.

        Returns:
            unyt_array:
                The star formation history of the stellar population.
        """
        return self.sf_hist

    @property
    def sfh(self):
        """Alias for get_sfh."""
        return self.get_sfh()

    def plot_sfh(
        self,
        xlimits=(),
        ylimits=(),
        show=True,
    ):
        """Plot the star formation history of the stellar population.

        Args:
            xlimits (tuple):
                The limits of the x-axis.
            ylimits (tuple):
                The limits of the y-axis.
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.step(self.log10ages, self.sf_hist, where="mid", color="blue")
        ax.fill_between(
            self.log10ages,
            self.sf_hist,
            step="mid",
            color="blue",
            alpha=0.5,
        )

        ax.set_xlabel(r"$\log_{10}(\mathrm{age}/\mathrm{yr})$")
        ax.set_ylabel(r"SFH / M$_\odot$")

        if show:
            plt.show()

        return fig, ax

    def get_metal_dist(self):
        """Get the metallicity distribution of the stellar population.

        Returns:
            unyt_array:
                The metallicity distribution of the stellar population.
        """
        return self.metal_dist

    def plot_metal_dist(
        self,
        xlimits=(),
        ylimits=(),
        show=True,
    ):
        """Plot the metallicity distribution of the stellar population.

        Args:
            xlimits (tuple):
                The limits of the x-axis.
            ylimits (tuple):
                The limits of the y-axis.
            show (bool):
                Should we invoke plt.show()?

        Returns:
            fig
                The Figure object contain the plot axes.
            ax
                The Axes object containing the plotted data.
        """
        fig, ax = plt.subplots()
        ax.semilogy()
        ax.step(self.metallicities, self.metal_dist, where="mid", color="red")
        ax.fill_between(
            self.metallicities,
            self.metal_dist,
            step="mid",
            color="red",
            alpha=0.5,
        )

        ax.set_xlabel(r"$Z$")
        ax.set_ylabel(r"Z_D / M$_\odot$")

        # Apply limits if provided
        if len(ylimits) > 0:
            ax.set_ylim(ylimits)
        if len(xlimits) > 0:
            ax.set_xlim(xlimits)

        if show:
            plt.show()

        return fig, ax

    def get_weighted_attr(self, attr):
        """Get a weighted attribute of the stellar population.

        Args:
            attr (str):
                The attribute to get.

        Returns:
            unyt_quantity:
                The weighted attribute.
        """
        # For now we need to raise an error if this is not an axis of the
        # SFZH grid
        if "age" not in attr and "metal" not in attr:
            raise exceptions.InconsistentArguments(
                "The attribute must be an axis of the SFZH grid"
            )

        # Get the attribute and the weights
        if "age" in attr:
            weight = self.sf_hist
        else:
            weight = self.metal_dist
        attr = getattr(self, attr)

        return weighted_mean(attr, weight)

    def calculate_average_sfr(self, t_range: tuple = (0, 1e8)):
        """Calculate the average SFR over a given age range.

        This method assumes that stars form at discrete time points given by
        `self.ages`, with the mass in `self.get_sfh()`.
        To calculate a continuous rate, it treats this mass as having
        formed uniformly in bins constructed around each age point.

        Args:
            t_range (tuple[float, float]):
                The lookback age limits (t_start, t_end) in years over which
                to calculate the average SFR.

        Returns:
            unyt_quantity:
                The average SFR over the specified time range in Msun/yr.
        """
        # --- Input Validation and Setup ---
        sfh_mass = np.asarray(self.get_sfh())
        age_points = np.asarray(self.ages)

        if sfh_mass.size != age_points.size:
            raise ValueError(
                "Mass array and age points array must have the same size."
            )

        if age_points.size == 0:
            return unyt_quantity(0, units="Msun/yr")

        # Support unyt quantities in t_range
        t_start, t_end = t_range
        if hasattr(t_start, "to"):
            t_start = t_start.to("yr").value
        if hasattr(t_end, "to"):
            t_end = t_end.to("yr").value
        # Ensure consistent ordering
        order = np.argsort(age_points)
        age_points = age_points[order]
        sfh_mass = sfh_mass[order]

        if t_start >= t_end:
            raise ValueError("Start of t_range must be less than its end.")

        # --- Construct Bins from Age Points ---
        if age_points.size == 1:
            # For a single point, assume the bin is centered on it,
            # starting from 0.
            age_edges = np.array([0, 2 * age_points[0]])
        else:
            # Bin edges are the midpoints between age points.
            internal_edges = (age_points[:-1] + age_points[1:]) / 2.0
            # Extrapolate the first and last edges to define the outer bounds.
            first_edge = age_points[0] - (age_points[1] - age_points[0]) / 2.0
            last_edge = (
                age_points[-1] + (age_points[-1] - age_points[-2]) / 2.0
            )
            age_edges = np.concatenate(
                ([first_edge], internal_edges, [last_edge])
            )

        # Ensure the first edge isn't negative for lookback time.
        age_edges[0] = max(0, age_edges[0])

        bin_starts = age_edges[:-1]
        bin_ends = age_edges[1:]
        bin_widths = bin_ends - bin_starts

        rates = np.divide(
            sfh_mass,
            bin_widths,
            out=np.zeros_like(sfh_mass, dtype=float),
            where=(bin_widths > 0),
        )

        overlap_starts = np.maximum(bin_starts, t_start)
        overlap_ends = np.minimum(bin_ends, t_end)
        overlap_durations = np.maximum(0, overlap_ends - overlap_starts)

        total_mass_in_range = np.sum(rates * overlap_durations) * Msun
        range_duration = (t_end - t_start) * yr
        average_sfr = total_mass_in_range / range_duration

        return average_sfr.to("Msun/yr")

    def calculate_surviving_mass(self, grid: Grid):
        """Calculate the surviving mass of the stellar population.

        This is the total mass of stars that have survived to the present day
        given the star formation and metal enrichment history.

        Args:
            grid (Grid):
                The grid to use for calculating the surviving mass.
                This is used to get the stellar fraction at each SFZH bin.

        Returns:
            unyt_quantity: The total surviving mass of the stellar
            population in Msun.
        """
        surviving_mass = np.sum(self.sfzh * grid.stellar_fraction)

        return surviving_mass * Msun

    def get_ionising_photon_luminosity(
        self,
        grid: Grid,
        ion: str = "HI",
    ) -> float:
        """Calculate the ionising photon luminosity from the grid.

        Args:
            grid (object, Grid):
                The SPS Grid object from which to extract spectra.
            ion (str):
                The ion for which to calculate the ionising photon luminosity.
                Must be a recognised ion in the grid's
                log10_specific_ionising_lum dictionary.

        Returns:
             The ionising photon luminosity summed over the grid dimensions.
        """
        if ion not in grid.log10_specific_ionising_lum:
            raise exceptions.MissingGridProperty(
                f"The provided grid does not contain {ion} "
                "ionising luminosities"
            )
        return np.sum(
            10 ** grid.log10_specific_ionising_lum[ion] * self.sfzh,
            axis=(0, 1),
        )
