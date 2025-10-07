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

from synthesizer.emission_models.base_model import BlackHoleEmissionModel
from synthesizer.emission_models.transformers import (
    CoveringFraction,
    EscapingFraction,
)
from synthesizer.exceptions import UnimplementedFunctionality


class UnifiedAGN(BlackHoleEmissionModel):
    """An emission model that defines the Unified AGN model.

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
        torus_emission_model,
        covering_fraction_nlr="covering_fraction_nlr",
        covering_fraction_blr="covering_fraction_blr",
        covered_fraction="covered_fraction",
        disc_transmission="combined",
        label="intrinsic",
        **kwargs,
    ):
        """Initialize the UnifiedAGN model.

        Args:
            nlr_grid (synthesizer.grid.Grid): The grid for the NLR.
            blr_grid (synthesizer.grid.Grid): The grid for the BLR.
            torus_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use for the torus.
            covering_fraction_nlr (float): The covering fraction of the NLR.
            covering_fraction_blr (float): The covering fraction of the BLR.
            covered_fraction (float): The covering fraction of the disc.
            disc_transmission (str): The disc transmission model.
            label (str): The label for the model.
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        # Get the incident istropic disc emission model
        self.disc_incident_isotropic = self._make_disc_incident_isotropic(
            nlr_grid,
            **kwargs,
        )

        # Get the incident disc emission model
        self.disc_incident = self._make_disc_incident(
            nlr_grid,
            **kwargs,
        )

        # Get the emission transmitted through the BLR and NLR
        (self.disc_transmitted_nlr, self.disc_transmitted_blr) = (
            self._make_disc_transmitted_lr(
                nlr_grid,
                blr_grid,
                **kwargs,
            )
        )

        print(self.disc_transmitted_nlr)

        # Get the averaged disc emission
        (
            self.disc_averaged,
            self.disc_averaged_without_torus,
        ) = self._make_disc_averaged(
            covering_fraction_nlr,
            covering_fraction_blr,
            **kwargs,
        )

        # Get the transmitted disc emission models
        self.disc_transmitted = self._make_disc_transmitted(
            nlr_grid,
            blr_grid,
            covering_fraction_nlr,
            covering_fraction_blr,
            disc_transmission,
            **kwargs,
        )

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
                self.disc_averaged,
                self.disc_averaged_without_torus,
                self.disc,
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
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        return model

    def _make_disc_incident(
        self,
        grid,
        **kwargs,
    ):
        """Make the disc spectra."""
        model = BlackHoleEmissionModel(
            grid=grid,
            label="disc_incident",
            extract="incident",
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        return model

    def _make_disc_transmitted_lr(
        self,
        nlr_grid,
        blr_grid,
        **kwargs,
    ):
        """Calculate the disc spectrum transmitted through the line regions.

        Args:
            nlr_grid (synthesizer.grid.Grid):
                The grid for the NLR.
            blr_grid (synthesizer.grid.Grid):
                The grid for the BLR.
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        disc_transmitted_nlr = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="disc_transmitted_nlr",
            extract="transmitted",
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            **kwargs,
        )

        disc_transmitted_blr = BlackHoleEmissionModel(
            grid=blr_grid,
            label="disc_transmitted_blr",
            extract="transmitted",
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        return disc_transmitted_nlr, disc_transmitted_blr

    def _make_disc_transmitted(
        self,
        nlr_grid,
        blr_grid,
        covering_fraction_nlr,
        covering_fraction_blr,
        disc_transmission,
        **kwargs,
    ):
        """Calculate the observed disc spectrum.

        There are a few options here that are set by the disc_transmission
        keyword. Either the disc emission escapes, goes through the NLR, goes
        through the BLR, or is blocked entirely by the torus. These can be set
        direction so that they apply to all blackholes or the keyword random
        can be given. In the random case each blackhole is assigned a random
        option based on the relative escape fractions of the NLR and BLR.

        Args:
            nlr_grid (synthesizer.grid.Grid): The grid for the NLR.
            blr_grid (synthesizer.grid.Grid): The grid for the BLR.
            torus_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use for the torus.
            covering_fraction_nlr (float): The covering fraction of the NLR.
            covering_fraction_blr (float): The covering fraction of the BLR.
            covered_fraction (float): The covering fraction of the disc.
            disc_transmission (str): The disc transmission model.
            label (str): The label for the model.
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        # If disc_transmission == 'none' the emission seen by the observer is
        # simply the incident emission. This step also accounts for the torus.
        if disc_transmission == "none":
            disc_transmitted = BlackHoleEmissionModel(
                label="disc_transmitted",
                combine=(self.disc_incident,),
                mask_attr="_torus_edgeon_cond",
                mask_thresh=90 * deg,
                mask_op="<",
                **kwargs,
            )
        # If disc_transmission == 'nlr' the emission seen by the observer is
        # is the spectrum transmitted through the NLR. This step also accounts
        # for the torus.
        if disc_transmission == "nlr":
            disc_transmitted = BlackHoleEmissionModel(
                label="disc_transmitted",
                combine=(self.disc_transmitted_nlr,),
                mask_attr="_torus_edgeon_cond",
                mask_thresh=90 * deg,
                mask_op="<",
                **kwargs,
            )

        # If disc_transmission == 'blr' the emission seen by the observer is
        # is the spectrum transmitted through the BLR. This step also accounts
        # for the torus.
        if disc_transmission == "blr":
            disc_transmitted = BlackHoleEmissionModel(
                label="disc_transmitted",
                combine=(self.disc_transmitted_blr,),
                mask_attr="_torus_edgeon_cond",
                mask_thresh=90 * deg,
                mask_op="<",
                **kwargs,
            )

        # If disc_transmission == 'combined' the emission seen by the observer
        # includes contributions from all line of sight. This is effectively
        # the disc_averaged without including the torus but then masked for
        # the torus.
        if disc_transmission == "combined":
            disc_transmitted = BlackHoleEmissionModel(
                label="disc_transmitted",
                combine=(self.disc_averaged_without_torus,),
                mask_attr="_torus_edgeon_cond",
                mask_thresh=90 * deg,
                mask_op="<",
                **kwargs,
            )

        # If disc_transmission == 'random' the emission seen by the observer is
        # chosen at random for each blackhole using covering fractions.
        if disc_transmission == "random":
            raise UnimplementedFunctionality(
                "random disc transmission not yet implemented"
            )

        return disc_transmitted

    def _make_disc(self, **kwargs):
        """Now is effectively just an alias to disc_transmitted."""
        return BlackHoleEmissionModel(
            label="disc",
            combine=(self.disc_transmitted,),
            **kwargs,
        )

    def _make_disc_averaged(
        self,
        covering_fraction_nlr,
        covering_fraction_blr,
        **kwargs,
    ):
        """Calculate the isotropic (inclination averaged) disc spectrum."""
        disc_escaped_isotropic = BlackHoleEmissionModel(
            label="disc_escaped_isotropic",
            apply_to=self.disc_incident,
            transformer=EscapingFraction(
                covering_attrs=(
                    "covering_fraction_blr",
                    "covering_fraction_nlr",
                )
            ),
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        disc_transmitted_nlr_isotropic = BlackHoleEmissionModel(
            label="disc_transmitted_nlr_isotropic",
            apply_to=self.disc_transmitted_nlr,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_nlr",)
            ),
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            **kwargs,
        )

        disc_transmitted_blr_isotropic = BlackHoleEmissionModel(
            label="disc_transmitted_blr_isotropic",
            apply_to=self.disc_transmitted_blr,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_blr",)
            ),
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        # Combine the models
        disc_averaged_without_torus = BlackHoleEmissionModel(
            label="disc_averaged_without_torus",
            combine=(
                disc_escaped_isotropic,
                disc_transmitted_nlr_isotropic,
                disc_transmitted_blr_isotropic,
            ),
            **kwargs,
        )

        # Now adjust for the torus
        disc_averaged = BlackHoleEmissionModel(
            label="disc_averaged",
            apply_to=disc_averaged_without_torus,
            transformer=CoveringFraction(
                covering_attrs=(
                    "covering_fraction_nlr",
                    "covering_fraction_blr",
                )
            ),
            fesc="torus_fraction",
            **kwargs,
        )

        return disc_averaged, disc_averaged_without_torus

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
        full_nlr = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="full_reprocessed_nlr",
            extract="nebular",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            **kwargs,
        )
        full_blr = BlackHoleEmissionModel(
            grid=blr_grid,
            label="full_reprocessed_blr",
            extract="nebular",
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        # Applying covering fractions
        nlr = BlackHoleEmissionModel(
            label="nlr",
            apply_to=full_nlr,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_nlr",)
            ),
            fesc=covering_fraction_nlr,
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            **kwargs,
        )
        blr = BlackHoleEmissionModel(
            label="blr",
            apply_to=full_blr,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_blr",)
            ),
            fesc=covering_fraction_blr,
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
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
