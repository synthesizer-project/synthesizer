"""
A generic blackholes class currently holding (ultimately) various blackhole
emission models.

Example Usage:

    UnifiedAGN(
        disc_model='test_grid_agn',
        photoionisation_model='cloudy_c17.03',
        grid_dir=grid_dir
)
"""

import numpy as np
from unyt import Hz, K, cm, deg, km, rad, s, unyt_array

from synthesizer import exceptions
from synthesizer.dust.emission import Greybody
from synthesizer.grid import Grid
from synthesizer.sed import Sed
from synthesizer.units import Quantity


class Template:
    """
    Use a template for the emission model.

    The template is simply scaled by bolometric luminosity.

    Attributes:
        sed (Sed)
            The template spectra for the AGN.
        normalisation (unyt_quantity)
            The normalisation for the spectra. In reality this is the
            bolometric luminosity.
    """

    def __init__(self, filename=None, lam=None, lnu=None):
        """
        Initialise the Template.

        Args:
            filename (str)
                The filename (including full path) to a file containing the
                template. The file should contain two columns with wavelength
                and luminosity (lnu).
            lam (array)
                Wavelength array.
            lnu (array)
                Luminosity array.

        """

        # Ensure we have been given units
        if lam is not None and not isinstance(lam, unyt_array):
            raise exceptions.MissingUnits("lam must be provided with units")
        if lnu is not None and not isinstance(lnu, unyt_array):
            raise exceptions.MissingUnits("lam must be provided with units")

        if filename:
            raise exceptions.UnimplementedFunctionality(
                "Not yet implemented! Feel free to implement and raise a "
                "pull request. Guidance for contributing can be found at "
                "https://github.com/flaresimulations/synthesizer/blob/main/"
                "docs/CONTRIBUTING.md"
            )

        if lam is not None and lnu is not None:
            # initialise a synthesizer Sed object
            self.sed = Sed(lam=lam, lnu=lnu)

            # normalise
            # TODO: add a method to Sed that does this.
            self.normalisation = self.sed.measure_bolometric_luminosity()
            self.sed.lnu /= self.normalisation.value

        else:
            raise exceptions.MissingArgument(
                "Either a filename or both lam and lnu must be provided!"
            )

    def get_spectra(self, bolometric_luminosity):
        """

        Calculating the blackhole spectra. This is done by simply scaling the
        normalised template by the bolometric luminosity

        Args:
            bolometric_luminosity (float)
                The bolometric luminosity of the blackhole(s) for scaling.

        """

        # Ensure we have units for safety
        if bolometric_luminosity is not None and not isinstance(
            bolometric_luminosity, unyt_array
        ):
            raise exceptions.MissingUnits(
                "bolometric luminosity must be provided with units"
            )

        return {
            "intrinsic": bolometric_luminosity.to(
                self.sed.lnu.units * Hz
            ).value
            * self.sed,
        }


class UnifiedAGN:
    """
    The Unified AGN model.
    This combines a disc model, along with modelling of the NLR, BLR,
    and torus.

    Attributes:
        disc_model (str)
            The disc_model to be used. The current test model is the AGNSED
            model.
        photoionisation_model (str)
            The photoionisation model to be used. Normally this would be
            e.g. "cloudy_c17.03", for the test_grid this is "" (i.e.
            empty string).
        grid_dir (str)
            The path to the grid directory.
        bolometric_luminosity (float)
            This is needed to rescale the spectra since the grid is likely
            to be coarse. Defaults to None since it should be provided.
        metallicity (float)
            The metallicity of the NLR and BLR gas. Defaults to None since
            it should be provided.
        ionisation_parameter_blr (float)
            The  ionisation parameter in the BLR. Default value
            is 0.1.
        hydrogen_density_blr (unyt.unit_object.Unit)
            The  hydrogen density in the BLR. Default value
            is 1E9/cm**3.
        covering_fraction_blr (float)
            The covering fraction of the BLR. Default value is 0.1.
        velocity_dispersion_blr (unyt.unit_object.Unit)
            The velocity disperson of the BLR. Default value is
            2000*km/s.
        ionisation_parameter_nlr (float)
            The ionisation parameter in the BLR. Default value
            is 0.01.
        hydrogen_density_nlr (unyt.unit_object.Unit)
            The hydrogen density in the NLR. Default value
            is 1E4/cm**3.
        covering_fraction_nlr (float)
            The covering fraction of the NLR. Default value is 0.1.
        velocity_dispersion_nlr (unyt.unit_object.Unit)
            The velocity disperson of the NLR. Default value is
            500*km/s.
        theta_torus (float)
            The opening angle of the torus component. Default value is
            10*deg.
        torus_emission_model (synthesizer.dust.emission)
            The torus emission model. A synthesizer dust emission model.
            Default is a Greybody with T=1000*K and emissivity=1.6.
        unified_parameters (list)
            A list of black hole parameters which are not specific to the NLR,
            BLR, or torus.
        fixed_parameters (list)
            A list of black hole parameters which have been fixed by the user.
        grid (dict, Grid)
            A dictionary containing the grid objects for the NLR and BLR.
        grid_parameters (array-like)
            The axes of the Grid.
        disc_parameters (list)
            A list of black hole parameters related to the disc.
        torus_parameters (list)
            A list of black hole parameters related to the torus.
        parameters (list)
            A list containing all the parameters of the black hole.
        variable_parameters (list)
            A list of parameters not fixed by the user. There will take the
            default values for the model in use.
        required_parameters (list)
            A list of parameters that must be passed the spectra methods or
            inherited from black hole objects.
        available_spectra (list)
            A list of the spectra types computed for a black hole.
    """

    # Store default values at the class level (this is so we can consider
    # values set by the user in the arguments as truly "fixed" parameters)
    default_params = {
        "ionisation_parameter_blr": 0.1,
        "hydrogen_density_blr": 1e9 / cm**3,
        "covering_fraction_blr": 0.1,
        "velocity_dispersion_blr": 2000 * km / s,
        "ionisation_parameter_nlr": 0.01,
        "hydrogen_density_nlr": 1e4 / cm**3,
        "covering_fraction_nlr": 0.1,
        "velocity_dispersion_nlr": 500 * km / s,
        "theta_torus": 10 * deg,
        "torus_emission_model": Greybody(100 * K, 1.5),
        "bolometric_luminosity": None,  # this is only used for scaling
    }

    # Define Quantities (these are temporarily used when making spectra and
    # guarantee the unit system is respected)
    mass = Quantity()
    bolometric_luminosity = Quantity()

    def __init__(
        self,
        disc_model,
        photoionisation_model,
        grid_dir,
        bolometric_luminosity=None,
        metallicity=None,
        ionisation_parameter_blr=None,
        hydrogen_density_blr=None,
        covering_fraction_blr=None,
        velocity_dispersion_blr=None,
        ionisation_parameter_nlr=None,
        hydrogen_density_nlr=None,
        covering_fraction_nlr=None,
        velocity_dispersion_nlr=None,
        theta_torus=None,
        torus_emission_model=None,
        verbose=False,
    ):
        """
        Intialise the UnifiedAGN emission model.

        Not all agruments must be specfied. Any not specified by the user will
        be assumed based on the models being employed or the default arguments
        defined above, depending on which is appropriate.

        Args:
            disc_model (str)
                The disc_model to be used. The current test model is the AGNSED
                model.
            photoionisation_model (str)
                The photoionisation model to be used. Normally this would be
                e.g. "cloudy_c17.03", for the test_grid this is "" (i.e.
                empty string).
            grid_dir (str)
                The path to the grid directory.
            bolometric_luminosity (float)
                This is needed to rescale the spectra since the grid is likely
                to be coarse. Defaults to None since it should be provided.
            metallicity (float)
                The metallicity of the NLR and BLR gas. Defaults to None since
                it should be provided.
            ionisation_parameter_blr (float)
                The  ionisation parameter in the BLR. Default value
                is 0.1.
            hydrogen_density_blr (unyt.unit_object.Unit)
                The  hydrogen density in the BLR. Default value
                is 1E9/cm**3.
            covering_fraction_blr (float)
                The covering fraction of the BLR. Default value is 0.1.
            velocity_dispersion_blr (unyt.unit_object.Unit)
                The velocity disperson of the BLR. Default value is
                2000*km/s.
            ionisation_parameter_nlr (float)
                The ionisation parameter in the BLR. Default value
                is 0.01.
            hydrogen_density_nlr (unyt.unit_object.Unit)
                The hydrogen density in the NLR. Default value
                is 1E4/cm**3.
            covering_fraction_nlr (float)
                The covering fraction of the NLR. Default value is 0.1.
            velocity_dispersion_nlr (unyt.unit_object.Unit)
                The velocity disperson of the NLR. Default value is
                500*km/s.
            theta_torus (float)
                The opening angle of the torus component. Default value is
                10*deg.
            torus_emission_model (synthesizer.dust.emission)
                The torus emission model. A synthesizer dust emission model.
                Default is a Greybody with T=1000*K and emissivity=1.6.
            verbose (bool)
                Are we talking?
        """

        # Create a dictionary of all the arguments
        args = {
            "bolometric_luminosity": bolometric_luminosity,
            "metallicity": metallicity,
            "ionisation_parameter_blr": ionisation_parameter_blr,
            "hydrogen_density_blr": hydrogen_density_blr,
            "covering_fraction_blr": covering_fraction_blr,
            "velocity_dispersion_blr": velocity_dispersion_blr,
            "ionisation_parameter_nlr": ionisation_parameter_nlr,
            "hydrogen_density_nlr": hydrogen_density_nlr,
            "covering_fraction_nlr": covering_fraction_nlr,
            "velocity_dispersion_nlr": velocity_dispersion_nlr,
            "theta_torus": theta_torus,
            "torus_emission_model": torus_emission_model,
        }

        # Save model and directory as attributes.
        self.disc_model = disc_model
        self.photoionsation_model = photoionisation_model
        self.grid_dir = grid_dir

        # These are the unified model parameters, i.e. all the non-disc and
        # non-torus parameters that are needed.
        self.unified_parameters = [
            "bolometric_luminosity",
            "metallicity",
            "ionisation_parameter_blr",
            "hydrogen_density_blr",
            "covering_fraction_blr",
            "velocity_dispersion_blr",
            "ionisation_parameter_nlr",
            "hydrogen_density_nlr",
            "covering_fraction_nlr",
            "velocity_dispersion_nlr",
            "theta_torus",
            "cosine_inclination",
        ]

        # Set "unified" attributes
        self.bolometric_luminosity = bolometric_luminosity
        self.metallicity = metallicity
        self.cosine_inclination = None

        # Set BLR attributes
        self.ionisation_parameter_blr = ionisation_parameter_blr
        self.hydrogen_density_blr = hydrogen_density_blr
        self.covering_fraction_blr = covering_fraction_blr
        self.velocity_dispersion_blr = velocity_dispersion_blr

        # Set NLR attributes
        self.ionisation_parameter_nlr = ionisation_parameter_nlr
        self.hydrogen_density_nlr = hydrogen_density_nlr
        self.covering_fraction_nlr = covering_fraction_nlr
        self.velocity_dispersion_nlr = velocity_dispersion_nlr

        # Set torus attribues
        self.theta_torus = theta_torus
        self.torus_emission_model = torus_emission_model

        # Create list of fixed parameters to ensure they are honoured when
        # using this model to generate spectra
        self.fixed_parameters = []
        for param, val in args.items():
            if val is not None:
                self.fixed_parameters.append(param)

        # Adopt default values for attributes that have not been fixed
        header_printed = False
        for param, val in args.items():
            if val is None and param in self.default_params:
                setattr(self, param, self.default_params[param])
                if verbose and param != "bolometric_luminosity":
                    if not header_printed:
                        print("Defaults Used:")
                        header_printed = True
                    print(f"    {param} = {self.default_params[param]}")

        # We can have a default for bolometric luminosity but it should never
        # be considered fixed
        if "bolometric_luminosity" in self.fixed_parameters:
            self.fixed_parameters.remove("bolometric_luminosity")

        # Open NLR and BLR grids and store them in a dict
        self.grid = {}
        for line_region in ["nlr", "blr"]:
            self.grid[line_region] = Grid(
                grid_name=f"{disc_model}{photoionisation_model}-{line_region}",
                grid_dir=grid_dir,
                read_lines=False,
            )

        # Get axes from the grid, this allows us to obtain the disc parameters.
        self.grid_parameters = self.grid["nlr"].axes[:]

        # Get disc parameters by removing line region parameters
        # TODO: replace this by saving something in the Grid file.
        self.disc_parameters = []
        for parameter in self.grid_parameters:
            if parameter not in [
                "metallicity",
                "ionisation_parameter",
                "hydrogen_density",
            ]:
                self.disc_parameters.append(parameter)

        # Define the torus parameters
        # TODO: need to extract a list of torus model parameters.
        self.torus_parameters = ["theta_torus", "torus_emission_model"]

        # Get a list of all parameters.
        self.parameters = list(
            set(
                self.disc_parameters
                + self.unified_parameters
                + list(self.default_params.keys())
                + self.torus_parameters
            )
        )

        # Define a list of parameters that must be passed or inherited
        self.required_parameters = []
        for param in self.parameters:
            if getattr(self, param, None) is None:
                self.required_parameters.append(param)

        # Get a list of the parameters which are not fixed and need to be
        # provided. This is used by the components.blackholes to know what
        # needs to be provided
        self.variable_parameters = list(
            set(self.parameters) - set(self.fixed_parameters)
        )

        # List of spectra that can be created
        self.available_spectra = [
            "disc_incident_isotropic",
            "disc_incident",
            "disc_escape",
            "disc_transmitted",
            "disc",
            "nlr",
            "blr",
            "torus",
            "intrinsic",
        ]

        self.spectra = {}

    def _generate_spectra_disc(
        self,
        component,
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
        lam = self.grid["nlr"].lam

        # Calculate the incident spectra. It doesn't matter which spectra we
        # use here since we're just using the incident. Note: this assumes the
        # NLR and BLR are not overlapping.

        # The istropic incident disc emission, which is used for the torus,
        # uses the isotropic incident emission so let's calculate that first.
        # To do this we want to temporarily set the cosine_inclination to 0.5
        # and ignore the mask.
        prev_cosine_inclincation = self.cosine_inclination
        self.cosine_inclination = 0.5

        self.spectra["disc_incident_isotropic"] = Sed(
            lam,
            component.generate_particle_lnu(
                self,
                self.grid["nlr"],
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

        self.spectra["disc_incident_masked"] = Sed(
            lam,
            component.generate_particle_lnu(
                self,
                self.grid["nlr"],
                spectra_name="incident",
                line_region="nlr",
                fesc=0.0,
                mask=mask,
                verbose=verbose,
                grid_assignment_method=grid_assignment_method,
            ),
        )

        self.spectra["disc_incident"] = Sed(
            lam,
            component.generate_particle_lnu(
                self,
                self.grid["nlr"],
                spectra_name="incident",
                line_region="nlr",
                fesc=0.0,
                mask=None,
                verbose=verbose,
                grid_assignment_method=grid_assignment_method,
            ),
        )

        # calculate the transmitted spectra
        nlr_spectra = component.generate_particle_lnu(
            self,
            self.grid["nlr"],
            spectra_name="transmitted",
            line_region="nlr",
            fesc=(1 - self.covering_fraction_nlr),
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        blr_spectra = component.generate_particle_lnu(
            self,
            self.grid["blr"],
            spectra_name="transmitted",
            line_region="blr",
            fesc=(1 - self.covering_fraction_blr),
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        self.spectra["disc_transmitted"] = Sed(lam, nlr_spectra + blr_spectra)

        # calculate the escaping spectra.
        self.spectra["disc_escaped"] = (
            1 - self.covering_fraction_blr - self.covering_fraction_nlr
        ) * self.spectra["disc_incident_masked"]

        # calculate the total spectra, the sum of escaping and transmitted
        self.spectra["disc"] = (
            self.spectra["disc_transmitted"] + self.spectra["disc_escaped"]
        )

        return self.spectra["disc"]

    def _generate_spectra_lr(
        self,
        component,
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
        self.cosine_inclination = np.full(component.nbh, 0.5)

        # Get the nebular spectra of the line region
        spec = component.generate_particle_lnu(
            self,
            self.grid[line_region],
            spectra_name="nebular",
            line_region=line_region,
            fesc=0.0,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )
        sed = Sed(
            self.grid[line_region].lam,
            getattr(self, f"covering_fraction_{line_region}") * spec,
        )

        # Reset the previously held inclination
        self.cosine_inclination = prev_cosine_inclincation

        return sed

    def _generate_spectra_torus(
        self,
        component,
        verbose=True,
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
        disc_spectra = self.spectra["disc_incident_isotropic"]

        # Calculate the bolometric dust lunminosity as the difference between
        # the intrinsic and attenuated
        torus_bolometric_luminosity = (
            self.theta_torus / (90 * deg)
        ) * disc_spectra.measure_bolometric_luminosity()

        # Create torus spectra
        torus_sed = self.torus_emission_model.get_spectra(disc_spectra.lam)

        # This is normalised to a bolometric luminosity of 1 so we need to
        # scale by the bolometric luminosity and create a spectra per particle.
        spec = np.zeros((component.nbh, torus_sed.lam.size))
        for i in range(component.nbh):
            spec[i, :] = torus_sed._lnu * torus_bolometric_luminosity[i].value
        torus_sed.lnu = spec

        return torus_sed

    def get_spectra(
        self, component, verbose=False, grid_assignment_method="cic"
    ):
        # Temporarily have the emission model adopt any vairable parameters
        # from this BlackHole/BlackHoles
        used_variables = []
        for param in self.variable_parameters:
            # Skip any parameters that don't exist on the black hole component
            if getattr(component, param, None) is None:
                continue

            # Remember the previous values to be returned after getting the
            # spectra
            used_variables.append((param, getattr(self, param, None)))

            # Set the passed value
            setattr(self, param, getattr(component, param, None))

        # Check if we have all the required parameters, if not raise an
        # exception and tell the user which are missing. Bolometric luminosity
        # is not strictly required.
        missing_params = []
        for param in self.parameters:
            print(param, getattr(self, param, None))

            if (
                param == "bolometric_luminosity"
                or param in self.required_parameters
            ):
                continue
            if getattr(self, param, None) is None:
                missing_params.append(param)
        if len(missing_params) > 0:
            raise exceptions.MissingArgument(
                f"Values not set for these parameters: {missing_params}"
            )

        # Determine the inclination from the cosine_inclination
        inclination = np.arccos(component.cosine_inclination) * rad

        # If the inclination is too high (edge) on we don't see the disc, only
        # the NLR and the torus. Create a mask to pass to the generation
        # method
        mask = inclination < ((90 * deg) - self.theta_torus)

        # Get the disc and BLR spectra
        self._generate_spectra_disc(
            component,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
        )

        self.spectra["blr"] = self._generate_spectra_lr(
            component,
            mask=mask,
            verbose=verbose,
            grid_assignment_method=grid_assignment_method,
            line_region="blr",
        )

        # mask is None, because we always see the NLR
        self.spectra["nlr"] = self._generate_spectra_lr(
            component,
            verbose=verbose,
            mask=None,
            grid_assignment_method=grid_assignment_method,
            line_region="nlr",
        )

        self.spectra["torus"] = self._generate_spectra_torus(
            component,
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
        # the spectra to the bolometric luminosity. The only SED that tracks
        # bolometric luminosity is the disc_incident_isotropic so this is used
        # for normalisation.
        if self.bolometric_luminosity is not None:
            scaling = (
                component.bolometric_luminosity
                / self.spectra[
                    "disc_incident_isotropic"
                ].measure_bolometric_luminosity()
            )

            print(scaling)

            for spectra_id, spectra in self.spectra.items():
                for i in range(component.nbh):
                    print(scaling[i])
                    self.spectra[spectra_id]._lnu[i] = (
                        scaling[i] * spectra._lnu[i, :]
                    )

        # Reset any values the emission model inherited
        for param, val in used_variables:
            setattr(self, param, val)

        return self.spectra
