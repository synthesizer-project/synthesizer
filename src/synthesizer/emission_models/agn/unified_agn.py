"""A submodule containing the definition of the Unified AGN model.

This module contains the definition of the Unified AGN model that can be used
to generate spectra from components or as a foundation to work from when
creating more complex models.

Example usage::

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

from synthesizer import exceptions
from synthesizer.emission_models.base_model import BlackHoleEmissionModel


class UnifiedAGN(BlackHoleEmissionModel):
    """
    An emission model that defines the Unified AGN model.

    The UnifiedAGN model includes a disc, nlr, blr and torus component and
    combines these components taking into accountgeometry of the disc
    and torus.

    Attributes:
        disc_incident_isotropic (BlackHoleEmissionModel):
            The disc emission model assuming isotropic emission.
        disc_incident (BlackHoleEmissionModel):
            The disc emission model accounting for the geometry but unmasked.
        nlr_transmitted (BlackHoleEmissionModel):
            The NLR transmitted emission
        blr_transmitted (BlackHoleEmissionModel):
            The BLR transmitted emission
        disc_transmitted (BlackHoleEmissionModel):
            The disc transmitted emission
        disc_escaped (BlackHoleEmissionModel):
            The disc escaped emission
        disc (BlackHoleEmissionModel):
            The disc emission model
        nlr (BlackHoleEmissionModel):
            The NLR emission model
        blr (BlackHoleEmissionModel):
            The BLR emission model
        torus (BlackHoleEmissionModel):
            The torus emission model
    """

    def __init__(
        self,
        nlr_grid,
        blr_grid,
        covering_fraction_nlr,
        covering_fraction_blr,
        torus_emission_model,
        label="intrinsic",
        **kwargs,
    ):
        """
        Initialize the UnifiedAGN model.

        Args:
            nlr_grid (synthesizer.grid.Grid): The grid for the NLR.
            blr_grid (synthesizer.grid.Grid): The grid for the BLR.
            covering_fraction_nlr (float): The covering fraction of the NLR.
            covering_fraction_blr (float): The covering fraction of the BLR.
            torus_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use for the torus.
            label (str): The label for the model.
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        # Ensure the sum of covering fractions is less than 1
        if covering_fraction_nlr + covering_fraction_blr > 1:
            raise exceptions.InconsistentArguments(
                "The sum of the covering fractions must be less than 1."
            )

        # Get the incident istropic disc emission model
        self.disc_incident_isotropic = self._make_disc_incident_isotropic(
            nlr_grid,
            **kwargs,
        )

        # Get the incident model accounting for the geometry but unmasked
        self.disc_incident = self._make_disc_incident(nlr_grid, **kwargs)

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
            **kwargs,
        )

        # Get the escaped disc emission model
        self.disc_escaped = self._make_disc_escaped(
            nlr_grid,
            covering_fraction_nlr,
            covering_fraction_blr,
            **kwargs,
        )

        # Get the disc emission model
        self.disc = self._make_disc(**kwargs)

        # Get the line regions
        self.nlr, self.blr = self._make_line_regions(
            nlr_grid,
            blr_grid,
            covering_fraction_nlr,
            covering_fraction_blr,
            **kwargs,
        )

        # Get the torus emission model
        self.torus = self._make_torus(torus_emission_model, **kwargs)

        # Create the final model
        BlackHoleEmissionModel.__init__(
            self,
            label=label,
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
            **kwargs,
        )

    def _make_disc_incident_isotropic(self, grid, **kwargs):
        """Make the disc spectra assuming isotropic emission."""
        model = BlackHoleEmissionModel(
            grid=grid,
            label="disc_incident_isotropic",
            extract="incident",
            cosine_inclination=0.5,
            **kwargs,
        )

        return model

    def _make_disc_incident(self, grid, **kwargs):
        """Make the disc spectra."""
        model = BlackHoleEmissionModel(
            grid=grid,
            label="disc_incident",
            extract="incident",
            **kwargs,
        )

        return model

    def _make_disc_transmitted(
        self,
        nlr_grid,
        blr_grid,
        covering_fraction_nlr,
        covering_fraction_blr,
        **kwargs,
    ):
        """Make the disc transmitted spectra."""
        # Make the line regions
        nlr = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="disc_transmitted_nlr",
            extract="transmitted",
            fesc=1 - covering_fraction_nlr,
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            **kwargs,
        )
        blr = BlackHoleEmissionModel(
            grid=blr_grid,
            label="disc_transmitted_blr",
            extract="transmitted",
            fesc=1 - covering_fraction_blr,
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            **kwargs,
        )

        # Combine the models
        model = BlackHoleEmissionModel(
            label="disc_transmitted",
            combine=(nlr, blr),
            **kwargs,
        )

        return nlr, blr, model

    def _make_disc_escaped(
        self,
        grid,
        covering_fraction_nlr,
        covering_fraction_blr,
        **kwargs,
    ):
        """
        Make the disc escaped spectra.

        This model is the mirror of the transmitted with the reverse of the
        covering fraction being applied to the disc incident model.
        """
        model = BlackHoleEmissionModel(
            grid=grid,
            label="disc_escaped",
            extract="incident",
            fesc=(covering_fraction_nlr + covering_fraction_blr),
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            **kwargs,
        )

        return model

    def _make_disc(self, **kwargs):
        """Make the disc spectra."""
        return BlackHoleEmissionModel(
            label="disc",
            combine=(self.disc_transmitted, self.disc_escaped),
            **kwargs,
        )

    def _make_line_regions(
        self,
        nlr_grid,
        blr_grid,
        covering_fraction_nlr,
        covering_fraction_blr,
        **kwargs,
    ):
        """Make the line regions."""
        # Make the line regions with fixed inclination
        nlr = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="nlr",
            extract="nebular",
            fesc=1 - covering_fraction_nlr,
            cosine_inclination=0.5,
            **kwargs,
        )
        blr = BlackHoleEmissionModel(
            grid=blr_grid,
            label="blr",
            extract="nebular",
            fesc=1 - covering_fraction_blr,
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            cosine_inclination=0.5,
            **kwargs,
        )
        return nlr, blr

    def _make_torus(self, torus_emission_model, **kwargs):
        """Make the torus spectra."""
        return BlackHoleEmissionModel(
            label="torus",
            generator=torus_emission_model,
            lum_intrinsic_model=self.disc_incident_isotropic,
            scale_by="torus_fraction",
            **kwargs,
        )
