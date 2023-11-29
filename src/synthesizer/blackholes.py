"""
A generic blackholes class currently holding (ultimately) various blackhole 
emission models.

TODO: probably a better home for this.

Example Usage:

    UnifiedAGN(
        disc_model='test_grid_agn',
        photoionisation_model='cloudy_c17.03',
        grid_dir=grid_dir
)
"""

import numpy as np
from unyt import deg, km, cm, s, K, rad
from synthesizer.dust.emission import Greybody
from synthesizer.grid import Grid
from synthesizer.sed import Sed
from synthesizer.exceptions import MissingArgument, UnimplementedFunctionality


class Template:

    """
    Use a template for the emission model.

    The template is simply scaled by bolometric luminosity.

    Attributes:
        sed (Sed)
            The template spectra for the AGN.
        normalisation (unyt_quantity)
            The normalisation for the spectra. In reality this is the bolometric
            luminosity.
    """

    def __init__(self, filename=None, lam=None, lnu=None):
        """

        Arguments:
            filename (str)
                The filename (including full path) to a file containing the
                template. The file should contain two columns with wavelength
                and luminosity (lnu).
            lam (array)
                Wavelength array.
            lnu (array)
                Luminosity array.

        """

        if filename:
            raise UnimplementedFunctionality(
                "Not yet implemented! Feel free to implement and raise a "
                "pull request. Guidance for contributing can be found at "
                "https://github.com/flaresimulations/synthesizer/blob/main/"
                "docs/CONTRIBUTING.md"
            )

        if lam and lnu:
            # initialise a synthesizer Sed object
            self.sed = Sed(lam=lam, lnu=lnu)

            # normalise
            # TODO: add a method to Sed that does this.
            self.normalisation = self.sed.measure_bolometric_luminosity()
            self.sed.lnu /= self.normalisation

    def get_spectra(self, bolometric_luminosity):
        """

        Calculating the blackhole spectra. This is done by simply scaling the
        normalised template by the bolometric luminosity

        Arguments:
            bolometric_luminosity (float)
                The bolometric luminosity of the blackhole(s).

        """

        return {"total": self.normalisation * self.sed}


class UnifiedAGN:

    """
    The Unified AGN model.
    This combines a disc model, along with modelling of the NLR, BLR, and torus.

    Attributes:
        disc_model (string)
            The disc model being used.
        photoionisation_model (string)
            The photoionisation model being used.
        grid_dir (string)
            The filepath to the grid file.
        torus_emission_model (synthesizer.dust.emission)
            The dust emission object describing the emission of the torus.
        unified_parameters (list)
            A list of black hole parameters which are not specific to the NLR,
            BLR, or torus.
        fixed_parameters_dict (dict)
            A dictionary containing the parameter values fixed by the user.
        fixed_parameters (list)
            A list of black hole parameters which have been fixed by the user.
        grid (dict, Grid)
            A dictionary containing the grid objects for the NLR and BLR.
        grid_parameters (array-like)
            The axes of the Grid.
        disc_parameters (list)
            A list of black hole parameters related to the disc.
        parameters (list)
            A list containing all the parameters of the black hole.
        variable_parameters (list)
            A list of parameters not fixed by the user. There will take the
            default values for the model in use.
        available_spectra (list)
            A list of the spectra types computed for a black hole.
    """

    def __init__(
        self,
        disc_model,
        photoionisation_model,
        grid_dir,
        bolometric_luminosity=None,
        metallicity=None,
        ionisation_parameter_blr=0.1,
        hydrogen_density_blr=1e9 / cm**3,
        covering_fraction_blr=0.1,
        velocity_dispersion_blr=2000 * km / s,
        ionisation_parameter_nlr=0.01,
        hydrogen_density_nlr=1e4 / cm**3,
        covering_fraction_nlr=0.1,
        velocity_dispersion_nlr=500 * km / s,
        theta_torus=10 * deg,
        torus_emission_model=Greybody(1000 * K, 1.5),
    ):
        """
        Arguments:

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
            **kwargs
                Addition parameters for the specific disc model.

        """

        # Create a dictionary of all the arguments
        args = {
            "disc_model": disc_model,
            "photoionisation_model": photoionisation_model,
            "grid_dir": grid_dir,
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

        # Save the torus model
        self.torus_emission_model = torus_emission_model

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
        ]

        # Create dictionary of fixed parameters based on what was handed
        self.fixed_parameters_dict = {}
        for parameter in self.unified_parameters:
            if args[parameter] is not None:
                self.fixed_parameters_dict[parameter] = args[parameter]

        # Create a list of the fixed_parameters for convenience
        self.fixed_parameters = list(self.fixed_parameters_dict.keys())

        # Open NLR and BLR grids
        # TODO: replace this by a single file.
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

        # TODO: need to extract a list of torus model parameters.
        # self.torus_parameters = []

        # Get a list of all parameters.
        # TODO: need to add torus parameters
        self.parameters = list(set(self.disc_parameters + self.unified_parameters))

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
            "total",
        ]

    def _get_grid_point(self, parameter_dict, line_region):
        """
        Private method used to get the closest grid point for the specified
        parameters.

        Arguments
            parameter_dict (dict)
                A dictionary of all the parameters.
            line_region (str)
                The specific line region of interest ('nlr' or 'dlr').

        Returns
            grid_point (tuple)
                The closest grid point
        """

        # make a list of the grid parameter values to use
        grid_parameter_values = []
        for parameter in self.grid_parameters:
            # for hydrogen_density and ionisation_paramter the parameters are
            # labelled e.g. 'hydrogen_density_nlr'.
            if parameter in ["hydrogen_density", "ionisation_parameter"]:
                parameter = parameter + "_" + line_region
            grid_parameter_values.append(parameter_dict[parameter])

        return self.grid[line_region].get_grid_point(grid_parameter_values)

    def _get_spectra_disc(self, parameter_dict):
        """
        Generate the disc spectra, updating the parameters if required.

        Arguments
            parameter_dict (dict)
                A dictionary of all the parameters.

        Returns
            (dict, synthesizer.sed.Sed)
                A dictionary of Sed instances including the escaping and
                transmitted disc emission.
        """

        # output dictionary
        spectra = {}

        # get grid points
        nlr_grid_point = self._get_grid_point(parameter_dict, "nlr")
        blr_grid_point = self._get_grid_point(parameter_dict, "blr")

        # calculate the incident spectra. It doesn't matter which spectra we
        # use here since we're just using the incident. Note: this assumes the
        # NLR and BLR are not overlapping.
        spectra["disc_incident"] = self.grid["nlr"].get_spectra(
            nlr_grid_point, spectra_id="incident"
        )

        # calculate the transmitted spectra
        nlr_spectra = self.grid["nlr"].get_spectra(
            nlr_grid_point, spectra_id="transmitted"
        )
        blr_spectra = self.grid["blr"].get_spectra(
            blr_grid_point, spectra_id="transmitted"
        )
        spectra["disc_transmitted"] = (
            parameter_dict["covering_fraction_nlr"] * nlr_spectra
            + parameter_dict["covering_fraction_blr"] * blr_spectra
        )

        # calculate the escaping spectra.
        spectra["disc_escaped"] = (
            1
            - parameter_dict["covering_fraction_blr"]
            - parameter_dict["covering_fraction_nlr"]
        ) * spectra["disc_incident"]

        # calculate the total spectra, the sum of escaping and transmitted
        spectra["disc"] = spectra["disc_transmitted"] + spectra["disc_escaped"]

        return spectra

    def _get_spectra_lr(self, parameter_dict, line_region):
        """
        Generate the spectra of a generic line region.

        Arguments
            parameter_dict (dict)
                A dictionary of all the parameters.
            line_region (str)
                The specific line region, i.e. 'nlr' or 'blr'.

        Returns:
            synthesizer.sed.Sed
                The NLR spectra
        """

        # In the Unified AGN model the NLR/BLR is illuminated by the isotropic
        # disc emisison hence the need to replace this parameter if it exists.
        # Not all models require an inclination though.
        if "cosine_inclination" in parameter_dict.keys():
            parameter_dict["cosine_inclination"] = 0.5

        # get the grid point
        grid_point = self._get_grid_point(parameter_dict, line_region)

        # get the nebular spectra of the line region
        sed = parameter_dict[f"covering_fraction_{line_region}"] * self.grid[
            line_region
        ].get_spectra(grid_point, spectra_id="nebular")

        return sed

    def _get_spectra_torus(self, parameter_dict):
        """
        Generate the torus emission Sed.

        Arguments
            parameter_dict (dict)
                A dictionary of all the parameters.

        Returns
            spectra (dict, synthesizer.sed.Sed)
                Dictionary of synthesizer Sed instances.
        """

        # In the Unified AGN model the torus is illuminated by the isotropic
        # disc emisison hence the need to replace this parameter if it exists.
        # Not all models require an inclination though.
        if "cosine_inclination" in parameter_dict.keys():
            parameter_dict["cosine_inclination"] = 0.5

        # Get NLR grid points
        nlr_grid_point = self._get_grid_point(parameter_dict, "nlr")

        # Calcualte the disc emission, since this is incident it doesn't matter
        # if we use the NLR or BLR grid as long as we use the correct grid
        # point.
        disc_spectra = self.grid["nlr"].get_spectra(
            nlr_grid_point, spectra_id="incident"
        )

        # calculate the bolometric dust lunminosity as the difference between
        # the intrinsic and attenuated
        torus_bolometric_luminosity = (
            parameter_dict["theta_torus"] / (90 * deg)
        ) * disc_spectra.measure_bolometric_luminosity()

        # get the spectrum and normalise it properly
        lnu = torus_bolometric_luminosity.to(
            "erg/s"
        ).value * self.torus_emission_model.lnu(disc_spectra.lam)

        # create new Sed object containing torus spectra and return it
        return Sed(disc_spectra.lam, lnu=lnu)

    def get_spectra(self, spectra_ids=None, **kwargs):
        """
        Generate the spectra, updating the parameters if required.

        Returns
            spectra (dict, synthesizer.sed.Sed)
                Dictionary of synthesizer Sed instances.
        """

        # Create a parameter dict by combining the fixed and provided
        # parameters
        parameter_dict = self.fixed_parameters_dict | kwargs

        # Check if we have all the required parameters, if not raise an
        # exception and tell the user which are missing. Bolometric luminosity
        # is not strictly required.
        if set(list(parameter_dict.keys()) + ["bolometric_luminosity"]) != set(
            self.parameters
        ):
            missing_parameters = set(self.parameters) - set(parameter_dict.keys())
            raise MissingArgument(
                f"These parameters: {missing_parameters} \
                                  are missing!"
            )

        # Generate the spectra of the disc

        # Determine the inclination from the cosine_inclination
        inclination = np.arccos(parameter_dict["cosine_inclination"]) * rad

        # If the inclination is too high (edge) on we don't see the disc, only
        # the NLR and the torus.
        if inclination < ((90 * deg) - parameter_dict["theta_torus"]):
            spectra = self._get_spectra_disc(parameter_dict)
            spectra["blr"] = self._get_spectra_lr(parameter_dict, "blr")
        else:
            spectra = {}

        spectra["nlr"] = self._get_spectra_lr(parameter_dict, "nlr")
        spectra["torus"] = self._get_spectra_torus(parameter_dict)

        # If we don't see the BLR and disc still generate spectra but set them to zero
        if inclination >= ((90 * deg) - parameter_dict["theta_torus"]):
            for spectra_id in [
                "blr",
                "disc_transmitted",
                "disc_incident",
                "disc_escape",
                "disc",
            ]:
                spectra[spectra_id] = Sed(lam=spectra["nlr"].lam)

        # Calculate the total spectra as the sum of the components
        spectra["total"] = (
            spectra["disc"] + spectra["blr"] + spectra["nlr"] + spectra["torus"]
        )

        # Since we're using a coarse grid it might be necessary to rescale
        # the spectra to the bolometric luminosity. This is requested when
        # the emission model is called from a parametric or particle blackhole.
        if "bolometric_luminosity" in parameter_dict.keys():
            scaling = (
                parameter_dict["bolometric_luminosity"]
                / spectra["total"].measure_bolometric_luminosity()
            )
            for spectra_id, spectra_ in spectra.items():
                spectra[spectra_id] = spectra_ * scaling

        # If spectra_ids are not provided use the full list
        if spectra_ids is None:
            return parameter_dict, spectra
        else:
            return parameter_dict, {
                spectra_id: spectra[spectra_ids] for spectra_id in spectra_ids
            }
