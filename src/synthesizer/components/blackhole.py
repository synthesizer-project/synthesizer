""" A module for holding blackhole emission models.

The class defined here should never be instantiated directly, there are only
ever instantiated by the parametric/particle child classes.
"""
from copy import deepcopy
import numpy as np
from unyt import c, rad, deg, unyt_quantity

from synthesizer import exceptions
from synthesizer.blackhole_emission_models import Template
from synthesizer.sed import Sed, plot_spectra
from synthesizer.units import Quantity


class BlackholesComponent:
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly, instead it provides the common
    functionality and attributes used by the child parametric and particle
    BlackHole/s classes.

    Attributes:
        spectra (dict, Sed)
            A dictionary containing black hole spectra.
        mass (array-like, float)
            The mass of each blackhole.
        accretion_rate (array-like, float)
            The accretion rate of each blackhole.
        epsilon (array-like, float)
            The radiative efficiency of the blackhole.
        accretion_rate_eddington (array-like, float)
            The accretion rate expressed as a fraction of the Eddington
            accretion rate.
        inclination (array-like, float)
            The inclination of the blackhole disc.
        spin (array-like, float)
            The dimensionless spin of the blackhole.
        bolometric_luminosity (array-like, float)
            The bolometric luminosity of the blackhole.
        metallicity (array-like, float)
            The metallicity of the blackhole which is assumed for the line
            emitting regions.
    """

    # Define class level Quantity attributes
    accretion_rate = Quantity()
    inclination = Quantity()
    bolometric_luminosity = Quantity()
    eddington_luminosity = Quantity()
    bb_temperature = Quantity()
    mass = Quantity()

    def __init__(
        self,
        mass=None,
        accretion_rate=None,
        epsilon=0.1,
        accretion_rate_eddington=None,
        inclination=None,
        spin=None,
        bolometric_luminosity=None,
        metallicity=None,
        **kwargs,
    ):
        """
        Initialise the BlackholeComponent. Where they're not provided missing
        quantities are automatically calcualted. Only some quantities are
        needed for each emission model.

        Args:
            mass (array-like, float)
                The mass of each blackhole.
            accretion_rate (array-like, float)
                The accretion rate of each blackhole.
            epsilon (array-like, float)
                The radiative efficiency of the blackhole.
            accretion_rate_eddington (array-like, float)
                The accretion rate expressed as a fraction of the Eddington
                accretion rate.
            inclination (array-like, float)
                The inclination of the blackhole disc.
            spin (array-like, float)
                The dimensionless spin of the blackhole.
            bolometric_luminosity (array-like, float)
                The bolometric luminosity of the blackhole.
            metallicity (array-like, float)
                The metallicity of the blackhole which is assumed for the line
                emitting regions.
            kwargs (dict)
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
        """

        # Initialise spectra
        self.spectra = {}

        # Save the arguments as attributes
        self.mass = mass
        self.accretion_rate = accretion_rate
        self.epsilon = epsilon
        self.accretion_rate_eddington = accretion_rate_eddington
        self.inclination = inclination
        self.spin = spin
        self.bolometric_luminosity = bolometric_luminosity
        self.metallicity = metallicity

        # Set any of the extra attribute provided as kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        # Check to make sure that both accretion rate and bolometric luminosity
        # haven't been provided because that could be confusing.
        if (self.accretion_rate is not None) and (
            self.bolometric_luminosity is not None
        ):
            raise exceptions.InconsistentArguments(
                """Both accretion rate and bolometric luminosity provided but
                that is confusing. Provide one or the other!"""
            )

        if (self.accretion_rate_eddington is not None) and (
            self.bolometric_luminosity is not None
        ):
            raise exceptions.InconsistentArguments(
                """Both accretion rate (in terms of Eddington) and bolometric
                luminosity provided but that is confusing. Provide one or
                the other!"""
            )

        # If mass, accretion_rate, and epsilon provided calculate the
        # bolometric luminosity.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_bolometric_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # big bump temperature.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_bb_temperature()

        # If mass calculate the Eddington luminosity.
        if self.mass is not None:
            self.calculate_eddington_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # Eddington ratio.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_eddington_ratio()

        # If mass, accretion_rate, and epsilon provided calculate the
        # accretion rate in units of the Eddington accretion rate. This is the
        # bolometric_luminosity / eddington_luminosity.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_accretion_rate_eddington()

        # If inclination, calculate the cosine of the inclination, required by
        # some models (e.g. AGNSED).
        if self.inclination is not None:
            self.cosine_inclination = np.cos(self.inclination.to("radian").value)

    def _prepare_sed_args(self, *args, **kwargs):
        """
        This method is a prototype for generating the arguments for spectra
        generation from AGN grids. It is redefined on the child classes to
        handle the different attributes of parametric and particle cases.
        """
        raise Warning(
            (
                "_prepare_sed_args should be overloaded by child classes:\n"
                "`particle.BlackHoles`\n"
                "`parametric.BlackHole`\n"
                "You should not be seeing this!!!"
            )
        )

    def generate_lnu(
        self,
        emission_model,
        grid,
        spectra_name,
        line_region,
        fesc=0.0,
        mask=None,
        verbose=False,
        grid_assignment_method="cic",
    ):
        """
        Generate the integrated rest frame spectra for a given grid key
        spectra.

        Args:
            emission_model (synthesizer.blackhole_emission_models.*)
                An instance of a blackhole emission model.
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of emission that escapes unattenuated from
                the birth cloud (defaults to 0.0).
            spectra_name (string)
                The name of the target spectra inside the grid file
                (e.g. "incident", "transmitted", "nebular").
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
        """
        # Ensure we have a key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{spectra_name}'"
            )

        # If the mask is False (parametric case) or contains only
        # 0 (particle case) just return an array of zeros
        if isinstance(mask, bool) and not mask:
            return np.zeros(grid.lam)
        if mask is not None and np.sum(mask) == 0:
            return np.zeros(grid.lam)

        from ..extensions.integrated_spectra import compute_integrated_sed

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            emission_model,
            grid,
            fesc=fesc,
            spectra_type=spectra_name,
            line_region=line_region,
            mask=mask,
            grid_assignment_method=grid_assignment_method.lower(),
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        return compute_integrated_sed(*args)

    def __str__(self):
        """Function to print a basic summary of the BlackHoles object.

        Returns a string containing the total mass formed and lists of the
        available SEDs, lines, and images.

        Returns
            str
                Summary string containing the total mass formed and lists of the
                available SEDs, lines, and images.
        """

        # Define the width to print within
        width = 80
        pstr = ""
        pstr += "-" * width + "\n"
        pstr += "SUMMARY OF BLACKHOLE".center(width + 4) + "\n"
        # pstr += get_centred_art(Art.blackhole, width) + "\n"

        pstr += f"Number of blackholes: {self.mass.size} \n"

        for attribute_id in [
            "mass",
            "accretion_rate",
            "accretion_rate_eddington",
            "bolometric_luminosity",
            "eddington_ratio",
            "bb_temperature",
            "eddington_luminosity",
            "spin",
            "epsilon",
            "inclination",
            "cosine_inclination",
        ]:
            attr = getattr(self, attribute_id, None)
            if attr is not None:
                attr = np.round(attr, 3)
                pstr += f"{attribute_id}: {attr} \n"

        return pstr

    def calculate_bolometric_luminosity(self):
        """
        Calculate the black hole bolometric luminosity. This is by itself
        useful but also used for some emission models.

        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        self.bolometric_luminosity = self.epsilon * self.accretion_rate * c**2

        return self.bolometric_luminosity

    def calculate_eddington_luminosity(self):
        """
        Calculate the eddington luminosity of the black hole.

        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        # Note: the factor 1.257E38 comes from:
        # 4*pi*G*mp*c*Msun/sigma_thompson
        self.eddington_luminosity = 1.257e38 * self._mass

        return self.eddington_luminosity

    def calculate_eddington_ratio(self):
        """
        Calculate the eddington ratio of the black hole.

        Returns
            unyt_array
                The black hole eddington ratio
        """

        self.eddington_ratio = self.bolometric_luminosity / self.eddington_luminosity

        return self.eddington_ratio

    def calculate_bb_temperature(self):
        """
        Calculate the black hole big bump temperature. This is used for the
        cloudy disc model.

        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        # Calculate the big bump temperature
        self.bb_temperature = (
            2.24e9 * self._accretion_rate ** (1 / 4) * self._mass**-0.5
        )

        return self.bb_temperature

    def calculate_accretion_rate_eddington(self):
        """
        Calculate the black hole accretion in units of the Eddington rate.

        Returns
            unyt_array
                The black hole accretion rate in units of the Eddington rate.
        """

        self.accretion_rate_eddington = (
            self._bolometric_luminosity / self._eddington_luminosity
        )

        return self.accretion_rate_eddington

    def _get_spectra_disc(
        self,
        emission_model,
        mask,
        verbose,
        grid_assignment_method,
    ):
        """
        Generate the disc spectra, updating the parameters if required.

        Args:
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN).
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.

        Returns:
            dict, Sed
                A dictionary of Sed instances including the escaping and
                transmitted disc emission.
        """

        # Get the wavelength array
        lam = emission_model.grid["nlr"].lam

        # Calculate the incident spectra. It doesn't matter which spectra we
        # use here since we're just using the incident. Note: this assumes the
        # NLR and BLR are not overlapping.
        self.spectra["disc_incident"] = Sed(
            lam,
            self.generate_lnu(
                emission_model,
                emission_model.grid["nlr"],
                spectra_name="incident",
                line_region="nlr",
                fesc=0.0,
                mask=mask,
                verbose=verbose,
                grid_assignment_method=grid_assignment_method,
            ),
        )

        # calculate the transmitted spectra
        nlr_spectra = self.generate_lnu(
            emission_model,
            emission_model.grid["nlr"],
            spectra_name="transmitted",
            line_region="nlr",
            fesc=emission_model.covering_fraction_nlr,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        blr_spectra = self.generate_lnu(
            emission_model,
            emission_model.grid["blr"],
            spectra_name="transmitted",
            line_region="blr",
            fesc=emission_model.covering_fraction_blr,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        self.spectra["disc_transmitted"] = Sed(lam, nlr_spectra + blr_spectra)

        # calculate the escaping spectra.
        self.spectra["disc_escaped"] = Sed(
            lam,
            (
                1
                - emission_model.covering_fraction_blr
                - emission_model.covering_fraction_nlr
            )
            * self.spectra["disc_incident"].lnu,
        )

        # calculate the total spectra, the sum of escaping and transmitted
        self.spectra["disc"] = Sed(
            lam,
            self.spectra["disc_transmitted"]._lnu + self.spectra["disc_escaped"]._lnu,
        )

        return self.spectra["disc"]

    def _get_spectra_lr(
        self,
        emission_model,
        mask,
        line_region,
        verbose,
        grid_assignment_method,
    ):
        """
        Generate the spectra of a generic line region.

        Args
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN).
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.

        Returns:
            Sed
                The NLR spectra
        """

        # In the Unified AGN model the NLR/BLR is illuminated by the isotropic
        # disc emisison hence the need to replace this parameter if it exists.
        # Not all models require an inclination though.
        prev_cosine_inclincation = self.cosine_inclination
        self.cosine_inclination = 0.5

        # Get the nebular spectra of the line region
        spec = self.generate_lnu(
            emission_model,
            emission_model.grid[line_region],
            spectra_name="nebular",
            line_region=line_region,
            fesc=0.0,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        sed = Sed(
            emission_model.grid[line_region].lam,
            getattr(emission_model, f"covering_fraction_{line_region}") * spec,
        )

        # Reset the previously held inclination
        self.cosine_inclination = prev_cosine_inclincation

        return sed

    def _get_spectra_torus(
        self,
        emission_model,
    ):
        """
        Generate the torus emission Sed.

        Args:
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN).

        Returns:
            Sed
                The torus spectra
        """

        # In the Unified AGN model the torus is illuminated by the isotropic
        # disc emisison hence the need to replace this parameter if it exists.
        # Not all models require an inclination though.
        prev_cosine_inclincation = self.cosine_inclination
        self.cosine_inclination = 0.5

        # Get the disc emission
        disc_spectra = self.spectra["disc_incident"]

        # calculate the bolometric dust lunminosity as the difference between
        # the intrinsic and attenuated
        torus_bolometric_luminosity = (
            emission_model.theta_torus / (90 * deg)
        ) * disc_spectra.measure_bolometric_luminosity()

        # create torus spectra
        sed = emission_model.torus_emission_model.get_spectra(disc_spectra.lam)

        # this is normalised to a bolometric luminosity of 1 so we need to
        # scale by the bolometric luminosity.

        sed._lnu *= torus_bolometric_luminosity.value

        # Reset the previously held inclination
        self.cosine_inclination = prev_cosine_inclincation

        return sed

    def get_spectra_intrinsic(
        self,
        emission_model,
        verbose=False,
        grid_assignment_method="cic",
    ):
        """
        Generate intrinsic blackhole spectra for a given emission_model.

        NOTE: any emission model parameters (excluding those fixed on
              the emission model) will be temporaily inherited from this
              from the object and reset after spectra creation.

        Args:
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN)
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.

        Returns:
            dict, Sed
                A dictionary of Sed instances including the intrinsic emission.
        """

        # Early exit if the emission model is a Template, for this we just
        # return the template scaled by bolometric luminosity
        if isinstance(emission_model, Template):
            self.spectra["intrinsic"] = emission_model.get_spectra(
                self.bolometric_luminosity
            )
            return self.spectra

        # Temporarily have the emission model adopt any vairable parameters
        # from this BlackHole/BlackHoles
        used_varaibles = []
        for param in emission_model.variable_parameters:
            # Skip any parameters that don't exist on the black hole component
            if getattr(self, param, None) is None:
                continue

            # Remember the previous values to be returned after getting the
            # spectra
            used_varaibles.append((param, getattr(emission_model, param, None)))

            # Set the passed value
            setattr(emission_model, param, getattr(self, param, None))

        # Check if we have all the required parameters, if not raise an
        # exception and tell the user which are missing. Bolometric luminosity
        # is not strictly required.
        missing_params = []
        for param in emission_model.parameters:
            if (
                param == "bolometric_luminosity"
                or param in emission_model.required_parameters
            ):
                continue
            if getattr(emission_model, param, None) is None:
                missing_params.append(param)
        if len(missing_params) > 0:
            raise exceptions.MissingArgument(
                f"Values not set for these parameters: {missing_params}"
            )
        # Determine the inclination from the cosine_inclination
        inclination = np.arccos(self.cosine_inclination) * rad

        # If the inclination is too high (edge) on we don't see the disc, only
        # the NLR and the torus. Create a mask to pass to the generation
        # method
        mask = inclination < ((90 * deg) - emission_model.theta_torus)

        # Get the disc and BLR spectra
        self._get_spectra_disc(
            emission_model=emission_model,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        self.spectra["blr"] = self._get_spectra_lr(
            emission_model=emission_model,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
            line_region="blr",
        )

        self.spectra["nlr"] = self._get_spectra_lr(
            emission_model=emission_model,
            verbose=verbose,
            mask=None,
            grid_assignment_method=grid_assignment_method,
            line_region="nlr",
        )
        self.spectra["torus"] = self._get_spectra_torus(
            emission_model=emission_model,
        )

        # Calculate the emergent spectra as the sum of the components.
        # Note: the choice of "intrinsic" is to align with the Pacman model
        # which reserves "total" and "emergent" to include dust.
        self.spectra["intrinsic"] = (
            self.spectra["disc"]
            + self.spectra["blr"]
            + self.spectra["nlr"]
            + self.spectra["torus"]
        )

        # Since we're using a coarse grid it might be necessary to rescale
        # the spectra to the bolometric luminosity. This is requested when
        # the emission model is called from a parametric or particle blackhole.
        if isinstance(self.bolometric_luminosity, float):
            scaling = (
                self.bolometric_luminosity
                / self.spectra["intrinsic"].measure_bolometric_luminosity()
            )
            for spectra_id, spectra in self.spectra.items():
                self.spectra[spectra_id] = spectra * scaling
        elif self.bolometric_luminosity is not None:
            scaling = (
                np.sum(self.bolometric_luminosity)
                / self.spectra["intrinsic"].measure_bolometric_luminosity()
            )
            for spectra_id, spectra in self.spectra.items():
                self.spectra[spectra_id] = spectra * scaling

        # Unset any of the fixed parameters we had to inherit
        for param, val in used_varaibles:
            setattr(emission_model, param, val)

        return self.spectra

    def get_spectra_attenuated(
        self,
        emission_model,
        verbose=False,
        grid_assignment_method="cic",
        tau_v=None,
        dust_curve=None,
        dust_emission_model=None,
    ):
        """
        Generate blackhole spectra for a given emission_model including
        dust attenuation and potentially emission.

        Args:
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN)
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            tau_v (float)
                The v-band optical depth.
            dust_curve (object)
                A synthesizer dust.attenuation.AttenuationLaw instance.
            dust_emission_model (object)
                A synthesizer dust.emission.DustEmission instance.

        Returns:
            dict, Sed
                A dictionary of Sed instances including the intrinsic and
                attenuated emission.
        """

        # Generate the intrinsic spectra
        self.get_spectra_intrinsic(
            emission_model=emission_model,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )

        # If dust attenuation is provided then calcualate additional spectra
        if dust_curve is not None and tau_v is not None:
            intrinsic = self.spectra["intrinsic"]
            self.spectra["emergent"] = intrinsic.apply_attenuation(
                tau_v, dust_curve=dust_curve
            )

            # If a dust emission model is also provided then calculate the
            # dust spectrum and total emission.
            if dust_emission_model is not None:
                # ISM dust heated by old stars.
                dust_bolometric_luminosity = (
                    self.spectra["intrinsic"].bolometric_luminosity
                    - self.spectra["emergent"].bolometric_luminosity
                )

                # Calculate normalised dust emission spectrum
                self.spectra["dust"] = dust_emission_model.get_spectra(
                    self.spectra["emergent"].lam
                )

                # Scale the dust spectra by the dust_bolometric_luminosity.
                self.spectra["dust"]._lnu *= dust_bolometric_luminosity.value

                # Calculate total spectrum
                self.spectra["total"] = self.spectra["emergent"] + self.spectra["dust"]

        elif (dust_curve is not None) or (tau_v is not None):
            raise exceptions.MissingArgument(
                "To enable dust attenuation both 'dust_curve' and "
                "'tau_v' need to be provided."
            )

        return self.spectra

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
                will be calculated with the lower limit set to 1000 (100) times less
                than the peak of the spectrum for rest_frame (observed) spectra.
            xlimits (tuple)
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple)
                Tuple with size 2 defining the figure size.
            kwargs (dict)
                arguments to the `sed.plot_spectra` method called from this wrapper

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
