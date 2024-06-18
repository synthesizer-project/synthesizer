"""A submodule containing the definition of the Unified AGN model.

This module contains the definition of the Unified AGN model that can be used
to generate spectra from components or as a foundation to work from when
creating more complex models.

Example usage:
    # Create the Unified AGN model
    model = UnifiedAGN(
        nlr_grid=nlr_grid,
        blr_grid=blr_grid,
        covering_fraction_nlr=0.5,
        covering_fraction_blr=0.5,
        torus_emission_model=torus_emission_model,
    )

    # Generate a spectra
    spectra = black_holes.get_spectra(model)
"""

from unyt import deg

from synthesizer.emission_models.base_model import EmissionModel


class UnifiedAGN(EmissionModel):
    """
    An emission model that defines the Unified AGN model.

    The UnifiedAGN model includes a disc, nlr, blr and torus component and
    combines these components taking into accountgeometry of the disc
    and torus.

    Attributes:
        disc_incident_isotropic (EmissionModel):
            The disc emission model assuming isotropic emission.
        disc_incident (EmissionModel):
            The disc emission model accounting for the geometry but unmasked.
        nlr_transmitted (EmissionModel):
            The NLR transmitted emission
        blr_transmitted (EmissionModel):
            The BLR transmitted emission
        disc_transmitted (EmissionModel):
            The disc transmitted emission
        disc_escaped (EmissionModel):
            The disc escaped emission
        disc (EmissionModel):
            The disc emission model
        nlr (EmissionModel):
            The NLR emission model
        blr (EmissionModel):
            The BLR emission model
        torus (EmissionModel):
            The torus emission model
    """

    def __init__(
        self,
        nlr_grid,
        blr_grid,
        covering_fraction_nlr,
        covering_fraction_blr,
        torus_emission_model,
    ):
        """
        Initialize the UnifiedAGN model.

        Args:
            nlr_grid (synthesizer.grid.Grid): The grid for the NLR.
            blr_grid (synthesizer.grid.Grid): The grid for the BLR.
            covering_fraction_nlr (float): The covering fraction of the NLR.
            covering_fraction_blr (float): The covering fraction of the BLR.
            torus_emission_model (synthesizer.dust.DustEmissionModel): The dust
                emission model to use for the torus.
        """
        # Get the incident istropic disc emission model
        self.disc_incident_isotropic = self._make_disc_incident_isotropic(
            nlr_grid
        )

        # Get the incident model accounting for the geometry but unmasked
        self.disc_incident = self._make_disc_incident(nlr_grid)

        # Get the transmitted disc emission models
        (
            self.nlr_transmitted,
            self.blr_transmitted,
            self.disc_transmitted,
        ) = self._make_disc_transmitted(
            nlr_grid,
            blr_grid,
            covering_fraction_nlr,
            covering_fraction_blr,
        )

        # Get the escaped disc emission model
        self.disc_escaped = self._make_disc_escaped(
            nlr_grid,
            covering_fraction_nlr,
            covering_fraction_blr,
        )

        # Get the disc emission model
        self.disc = self._make_disc()

        # Get the line regions
        self.nlr, self.blr = self._make_line_regions(nlr_grid, blr_grid)

        # Get the torus emission model
        self.torus = self._make_torus(torus_emission_model)

        # Create the final model
        EmissionModel.__init__(
            self,
            label="intrinsic",
            combine=(
                self.disc,
                self.nlr,
                self.blr,
                self.torus,
            ),
            related_models=(
                self.disc_incident_isotropic,
                self.disc_incident,
            ),
        )

    def _make_disc_incident_isotropic(self, grid):
        """Make the disc spectra assuming isotropic emission."""
        model = EmissionModel(
            grid=grid,
            label="disc_incident_isotropic",
            extract="incident",
            fixed_parameters={"inclination": 60 * deg},
        )

        return model

    def _make_disc_incident(self, grid):
        """Make the disc spectra."""
        model = EmissionModel(
            grid=grid,
            label="disc_incident",
            extract="incident",
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
        )

        return model

    def _make_disc_transmitted(
        self,
        nlr_grid,
        blr_grid,
        covering_fraction_nlr,
        covering_fraction_blr,
    ):
        """Make the disc transmitted spectra."""
        # Make the line regions
        nlr = EmissionModel(
            grid=nlr_grid,
            label="nlr_transmitted",
            extract="transmitted",
            fesc=covering_fraction_nlr,
        )
        blr = EmissionModel(
            grid=blr_grid,
            label="blr_transmitted",
            extract="transmitted",
            fesc=covering_fraction_blr,
        )

        # Combine the models
        model = EmissionModel(
            label="disc_transmitted",
            combine=(nlr, blr),
        )

        return nlr, blr, model

    def _make_disc_escaped(
        self,
        grid,
        covering_fraction_nlr,
        covering_fraction_blr,
    ):
        """
        Make the disc escaped spectra.

        This model is the mirror of the transmitted with the reverse of the
        covering fraction being applied to the disc incident model.
        """
        model = EmissionModel(
            grid=grid,
            label="disc_escaped",
            extract="incident",
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            fesc=1.0 - covering_fraction_nlr - covering_fraction_blr,
        )

        return model

    def _make_disc(self):
        """Make the disc spectra."""
        return EmissionModel(
            label="disc",
            combine=(self.disc_transmitted, self.disc_escaped),
        )

    def _make_line_regions(self, nlr_grid, blr_grid):
        """Make the line regions."""
        # Make the line regions with fixed inclination
        nlr = EmissionModel(
            grid=nlr_grid,
            label="nlr",
            extract="transmitted",
            fixed_parameters={"inclination": 60 * deg},
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
        )
        blr = EmissionModel(
            grid=nlr_grid,
            label="blr",
            extract="transmitted",
            fixed_parameters={"inclination": 60 * deg},
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
        )
        return nlr, blr

    def _make_torus(self, torus_emission_model):
        """Make the torus spectra."""
        return EmissionModel(
            label="torus",
            generator=torus_emission_model,
            scale_by="torus_fraction",
        )
