"""A module for holding blackhole emission models.

The class defined here should never be instantiated directly, there are only
ever instantiated by the parametric/particle child classes.
"""

import numpy as np
from unyt import c, deg, rad, unyt_array

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
        self.particle_spectra = {}
        self.spectra = {}

        # Intialise the photometry dictionaries
        self.photo_luminosities = {}
        self.photo_fluxes = {}

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
            self.cosine_inclination = np.cos(
                self.inclination.to("radian").value
            )

    def __str__(self):
        """Function to print a basic summary of the BlackHoles object.

        Returns a string containing the total mass formed and lists of the
        available SEDs, lines, and images.

        Returns
            str
                Summary string containing the total mass formed and lists
                of the available SEDs, lines, and images.
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

        self.eddington_ratio = (
            self._bolometric_luminosity / self._eddington_luminosity
        )

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

    def _prepare_sed_args(
        self,
        emission_model,
        grid,
        fesc,
        spectra_type,
        line_region,
        mask,
        grid_assignment_method,
    ):
        """
        A method to prepare the arguments for SED computation with the C
        functions.

        Args:
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN).
            grid (Grid)
                The SPS grid object to extract spectra from.
            fesc (float)
                The escape fraction.
            spectra_type (str)
                The type of spectra to extract from the Grid. This must match a
                type of spectra stored in the Grid.
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """

        # Handle the case where mask is None
        if mask is None:
            mask = np.ones(self.nbh, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(getattr(grid, axis), dtype=np.float64)
            for axis in grid.axes
        ]
        props = []
        for axis in grid.axes:
            # Source parameters from the appropriate place
            if axis in emission_model.required_parameters:
                # Parameters that need to be provided from the black hole
                # (these already exclude any parameters fixed on the
                # emission model)
                props.append(getattr(self, axis))

            elif (
                axis in emission_model.variable_parameters
                and getattr(self, axis, None) is not None
            ):
                # Variable parameters defined on the black hole (not including
                # line region parameters)
                props.append(getattr(self, axis))

            elif (
                f"{axis} {line_region}" in emission_model.variable_parameters
                and getattr(self, axis + "_" + line_region, None) is not None
            ):
                # Variable line region parameters defined on the black hole
                props.append(getattr(self, axis + "_" + line_region))

            elif getattr(emission_model, axis, None) is not None:
                # Parameters required from the emission_model (not including
                # line region parameters)
                props.append(getattr(emission_model, axis))

            elif (
                getattr(emission_model, axis + "_" + line_region, None)
                is not None
            ):
                # Line region parameters required from the emission_model
                props.append(getattr(emission_model, axis + "_" + line_region))

            else:
                raise exceptions.MissingArgument(
                    f"{axis} can't be sourced from emission_model or self."
                )

        # Calculate npart from the mask
        npart = np.sum(mask)

        # Remove units from any unyt_arrays
        props = [
            prop.value if isinstance(prop, unyt_array) else prop
            for prop in props
        ]

        # Ensure any parameters inherited from the emission model have
        # as many values as particles
        for ind, prop in enumerate(props):
            if isinstance(prop, float):
                props[ind] = np.full(self.nbh, prop)
            elif prop.size == 1:
                props[ind] = np.full(self.nbh, prop)

        # Apply the mask to each property and make contiguous
        props = [
            np.ascontiguousarray(prop[mask], dtype=np.float64)
            for prop in props
        ]

        # For black holes mass is a grid parameter but we still need to
        # multiply by mass in the extensions so just multiply by 1
        mass = np.ones(npart, dtype=np.float64)

        # Make sure we get the wavelength index of the grid array
        nlam = np.int32(grid.spectra[spectra_type].shape[-1])

        # Get the grid spctra
        grid_spectra = np.ascontiguousarray(
            grid.spectra[spectra_type],
            dtype=np.float64,
        )

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)
        grid_dims[ind + 1] = nlam

        # If fesc isn't an array make it one
        if not isinstance(fesc, np.ndarray):
            fesc = np.ascontiguousarray(np.full(npart, fesc))

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        props = tuple(props)

        return (
            grid_spectra,
            grid_props,
            props,
            mass,
            fesc,
            grid_dims,
            len(grid_props),
            np.int32(npart),
            nlam,
            grid_assignment_method,
        )

    def generate_particle_lnu(
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

        from ..extensions.particle_spectra import compute_particle_seds

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
        masked_spec = compute_particle_seds(*args)

        # If there's no mask we're done
        if mask is None:
            return masked_spec

        # If we have a mask we need to account for the zeroed spectra
        spec = np.zeros((self.nbh, masked_spec.shape[-1]))
        spec[mask] = masked_spec
        return spec

    def _get_particle_spectra_disc(
        self,
        emission_model,
        mask,
        verbose,
        grid_assignment_method,
    ):
        """
        Generate the disc spectra, updating the parameters if required
        for each particle.

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
                transmitted disc emission of each particle.
        """

        # Get the wavelength array
        lam = emission_model.grid["nlr"].lam

        # Calculate the incident spectra. It doesn't matter which spectra we
        # use here since we're just using the incident. Note: this assumes the
        # NLR and BLR are not overlapping.

        # The istropic incident disc emission, which is used for the torus,
        # uses the isotropic incident emission so let's calculate that first.
        # To do this we want to temporarily set the cosine_inclination to 0.5
        # and ignore the mask.
        prev_cosine_inclincation = self.cosine_inclination
        self.cosine_inclination = 0.5

        self.particle_spectra["disc_incident_isotropic"] = Sed(
            lam,
            self.generate_particle_lnu(
                emission_model,
                emission_model.grid["nlr"],
                spectra_name="incident",
                line_region="nlr",
                fesc=0.0,
                mask=None,
                verbose=verbose,
                grid_assignment_method=grid_assignment_method,
            ),
        )

        # Reset the cosine_inclination to the original value.
        self.cosine_inclination = prev_cosine_inclincation

        self.particle_spectra["disc_incident_masked"] = Sed(
            lam,
            self.generate_particle_lnu(
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

        self.particle_spectra["disc_incident"] = Sed(
            lam,
            self.generate_particle_lnu(
                emission_model,
                emission_model.grid["nlr"],
                spectra_name="incident",
                line_region="nlr",
                fesc=0.0,
                mask=None,
                verbose=verbose,
                grid_assignment_method=grid_assignment_method,
            ),
        )

        # calculate the transmitted spectra
        nlr_spectra = self.generate_particle_lnu(
            emission_model,
            emission_model.grid["nlr"],
            spectra_name="transmitted",
            line_region="nlr",
            fesc=(1 - emission_model.covering_fraction_nlr),
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        blr_spectra = self.generate_particle_lnu(
            emission_model,
            emission_model.grid["blr"],
            spectra_name="transmitted",
            line_region="blr",
            fesc=(1 - emission_model.covering_fraction_blr),
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        self.particle_spectra["disc_transmitted"] = Sed(
            lam, nlr_spectra + blr_spectra
        )

        # calculate the escaping spectra.
        self.particle_spectra["disc_escaped"] = (
            1
            - emission_model.covering_fraction_blr
            - emission_model.covering_fraction_nlr
        ) * self.particle_spectra["disc_incident_masked"]

        # calculate the total spectra, the sum of escaping and transmitted
        self.particle_spectra["disc"] = (
            self.particle_spectra["disc_transmitted"]
            + self.particle_spectra["disc_escaped"]
        )

        return self.particle_spectra["disc"]

    def _get_particle_spectra_lr(
        self,
        emission_model,
        mask,
        line_region,
        verbose,
        grid_assignment_method,
    ):
        """
        Generate the spectra of a generic line region for each particle.

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
        self.cosine_inclination = np.full(self.nbh, 0.5)

        # Get the nebular spectra of the line region
        spec = self.generate_particle_lnu(
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

    def _get_particle_spectra_torus(
        self,
        emission_model,
    ):
        """
        Generate the torus emission Sed for each particle.

        Args:
            emission_model (blackhole_emission_models.*)
                Any instance of a blackhole emission model (e.g. Template
                or UnifiedAGN).

        Returns:
            Sed
                The torus spectra of each particle.
        """

        # Get the disc emission
        disc_spectra = self.particle_spectra["disc_incident_isotropic"]

        # Calculate the bolometric dust lunminosity as the difference between
        # the intrinsic and attenuated
        torus_bolometric_luminosity = (
            emission_model.theta_torus / (90 * deg)
        ) * disc_spectra.measure_bolometric_luminosity()

        # Create torus spectra
        torus_sed = emission_model.torus_emission_model.get_spectra(
            disc_spectra.lam
        )

        # This is normalised to a bolometric luminosity of 1 so we need to
        # scale by the bolometric luminosity and create a spectra per particle.
        spec = np.zeros((self.nbh, torus_sed.lam.size))
        for i in range(self.nbh):
            spec[i, :] = torus_sed._lnu * torus_bolometric_luminosity[i].value
        torus_sed.lnu = spec

        return torus_sed

    def get_particle_spectra_intrinsic(
        self,
        emission_model,
        verbose=False,
        grid_assignment_method="cic",
    ):
        """
        Generate intrinsic blackhole spectra for a given emission_model
        for all particles.

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
                A dictionary of Sed instances including the intrinsic emission
                of each particle.
        """

        # Early exit if the emission model is a Template, for this we just
        # return the template scaled by bolometric luminosity
        if isinstance(emission_model, Template):
            self.particle_spectra["intrinsic"] = np.full(
                self.nbh,
                emission_model.get_spectra(self.bolometric_luminosity),
            )
            return self.particle_spectra

        # Temporarily have the emission model adopt any vairable parameters
        # from this BlackHole/BlackHoles
        used_varaibles = []
        for param in emission_model.variable_parameters:
            # Skip any parameters that don't exist on the black hole component
            if getattr(self, param, None) is None:
                continue

            # Remember the previous values to be returned after getting the
            # spectra
            used_varaibles.append(
                (param, getattr(emission_model, param, None))
            )

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
        self._get_particle_spectra_disc(
            emission_model=emission_model,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        self.particle_spectra["blr"] = self._get_particle_spectra_lr(
            emission_model=emission_model,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
            line_region="blr",
        )

        self.particle_spectra["nlr"] = self._get_particle_spectra_lr(
            emission_model=emission_model,
            verbose=verbose,
            mask=None,
            grid_assignment_method=grid_assignment_method,
            line_region="nlr",
        )
        self.particle_spectra["torus"] = self._get_particle_spectra_torus(
            emission_model=emission_model,
        )

        # Calculate the emergent spectra as the sum of the components.
        # Note: the choice of "intrinsic" is to align with the Pacman model
        # which reserves "total" and "emergent" to include dust.
        self.particle_spectra["intrinsic"] = (
            self.particle_spectra["disc"]
            + self.particle_spectra["blr"]
            + self.particle_spectra["nlr"]
            + self.particle_spectra["torus"]
        )

        # Since we're using a coarse grid it might be necessary to rescale
        # the spectra to the bolometric luminosity. This is requested when
        # the emission model is called from a parametric or particle blackhole.
        if self.bolometric_luminosity is not None:
            scaling = (
                self.bolometric_luminosity
                / self.particle_spectra[
                    "intrinsic"
                ].measure_bolometric_luminosity()
            )
            for spectra_id, spectra in self.particle_spectra.items():
                for i in range(self.nbh):
                    self.particle_spectra[spectra_id]._lnu[i] = (
                        scaling[i] * spectra._lnu[i, :]
                    )

        # Reset any values the emission model inherited
        for param, val in used_varaibles:
            setattr(emission_model, param, val)

        return self.particle_spectra

    def get_spectra_intrinsic(
        self,
        emission_model,
        verbose=False,
        grid_assignment_method="cic",
    ):
        """
        Generate integrated (summed) intrinsic blackhole spectra for a given
        emission_model over all particles.

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
                A dictionary of Sed instances including the integrated
                intrinsic emission.
        """

        particle_spectra = self.get_particle_spectra_intrinsic(
            emission_model,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )

        self.spectra = {
            spectra_id: sed.sum()
            for spectra_id, sed in particle_spectra.items()
        }

        return self.spectra

    def get_particle_spectra_attenuated(
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
        dust attenuation and potentially emission for all particles.

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
                attenuated emission of each particle.
        """

        # Generate the intrinsic spectra
        self.get_particle_spectra_intrinsic(
            emission_model=emission_model,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )

        # If dust attenuation is provided then calcualate additional spectra
        if dust_curve is not None and tau_v is not None:
            intrinsic = self.particle_spectra["intrinsic"]
            self.particle_spectra["emergent"] = intrinsic.apply_attenuation(
                tau_v, dust_curve=dust_curve
            )

            # If a dust emission model is also provided then calculate the
            # dust spectrum and total emission.
            if dust_emission_model is not None:
                # ISM dust heated by old stars.
                dust_bolometric_luminosity = (
                    self.particle_spectra["intrinsic"].bolometric_luminosity
                    - self.particle_spectra["emergent"].bolometric_luminosity
                )

                # expand dimensions
                dust_bolometric_luminosity = np.expand_dims(
                    dust_bolometric_luminosity, self.nbh
                )

                # Calculate normalised dust emission spectrum
                dust_lnu = dust_emission_model.get_spectra(
                    self.particle_spectra["emergent"].lam
                )._lnu

                # tile and repeat this spectra
                dust_lnu = np.tile(dust_lnu, self.nbh).reshape(
                    self.nbh, len(self.particle_spectra["emergent"].lam)
                )

                # define dust spectra
                self.particle_spectra["dust"] = Sed(
                    self.particle_spectra["emergent"]._lam,
                    lnu=dust_bolometric_luminosity.value * dust_lnu,
                )

                # define total spectra
                self.particle_spectra["total"] = (
                    self.particle_spectra["emergent"]
                    + self.particle_spectra["dust"]
                )

        elif (dust_curve is not None) or (tau_v is not None):
            raise exceptions.MissingArgument(
                "To enable dust attenuation both 'dust_curve' and "
                "'tau_v' need to be provided."
            )

        return self.particle_spectra

    def get_spectra_attenuated(
        self,
        emission_model,
        verbose=False,
        grid_assignment_method="cic",
        tau_v=None,
        dust_curve=None,
        dust_emission_model=None,
    ):
        particle_spectra = self.get_particle_spectra_attenuated(
            emission_model,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
            tau_v=tau_v,
            dust_curve=dust_curve,
            dust_emission_model=dust_emission_model,
        )

        self.spectra = {
            spectra_id: sed.sum()
            for spectra_id, sed in particle_spectra.items()
        }

        return self.spectra

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
                will be calculated with the lower limit set to 1000 (100)
                times less than the peak of the spectrum for rest_frame
                (observed) spectra.
            xlimits (tuple)
                The limits to apply to the x axis. If not provided the optimal
                limits are found based on the ylimits.
            figsize (tuple)
                Tuple with size 2 defining the figure size.
            kwargs (dict)
                arguments to the `sed.plot_spectra` method called from this
                wrapper

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
