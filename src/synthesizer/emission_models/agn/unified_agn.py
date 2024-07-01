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

import numpy as np
from unyt import deg

from synthesizer.emission_models.base_model import BlackHoleEmissionModel
from synthesizer.sed import Sed


def scale_by_incident_isotropic(emission, emitters, model):
    """
    Scale the emission by the incident isotropic disc model.

    Args:
        emission (dict, Sed/LineCollection): The emission to scale.
        emitters (dict): The emitters used to generate the emission.
        model (UnifiedAGN): The Unified AGN model.
    """
    # Handle lines and spectra differently
    if isinstance(emission[list(emission.keys())[0]], Sed):
        # Get the isotropic bolometric luminosity
        isotropic_bolo_lum = emission[
            "disc_incident_isotropic"
        ]._bolometric_luminosity

        # Scale the spectra
        for key, spectra in emission.items():
            # Get the model
            this_model = model[key]

            # Skip the emitter if it's not a black hole
            if this_model.emitter not in "blackhole":
                continue

            # Get the emitter
            emitter = emitters[this_model.emitter]

            # Get the scaling
            scaling = emitter._bolometric_luminosity / isotropic_bolo_lum

            # Scale the spectra
            if scaling.ndim == 0:
                spectra._lnu *= scaling
            elif scaling.ndim == 1 and spectra.ndim == 1:
                spectra._lnu = scaling * np.expand_dims(spectra._lnu, axis=0)
                spectra._lnu = np.sum(spectra._lnu, axis=0)

    else:
        # Loop over the different emissions
        for key, emission in emission.items():
            # Get the model
            this_model = model[key]

            # Skip the emitter if it's not a black hole
            if this_model.emitter not in "blackhole":
                continue

            # Get the emitter
            emitter = emitters[this_model.emitter]

            # Loop over the lines
            for line_id, line in emission[key]:
                # Get the continuum of the isotropic disc emission
                isotropic_continuum = line._continuum

                # Get the scaling
                scaling = emitter._bolometric_luminosity / isotropic_continuum

                # Scale the line
                line._luminosity *= scaling
                line._continuum *= scaling

    return emission


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
        """
        # Get the incident istropic disc emission model
        self.disc_incident_isotropic = self._make_disc_incident_isotropic(
            nlr_grid,
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
            post_processing=(scale_by_incident_isotropic,),
        )

    def _make_disc_incident_isotropic(self, grid):
        """Make the disc spectra assuming isotropic emission."""
        model = BlackHoleEmissionModel(
            grid=grid,
            label="disc_incident_isotropic",
            extract="incident",
            fixed_parameters={"cosine_inclination": 0.5},
            scale_by="bolometric_luminosity",
        )

        return model

    def _make_disc_incident(self, grid):
        """Make the disc spectra."""
        model = BlackHoleEmissionModel(
            grid=grid,
            label="disc_incident",
            extract="incident",
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            scale_by="bolometric_luminosity",
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
        nlr = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="disc_transmitted_nlr",
            extract="transmitted",
            fesc=covering_fraction_nlr,
            scale_by="bolometric_luminosity",
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
        )
        blr = BlackHoleEmissionModel(
            grid=blr_grid,
            label="disc_transmitted_blr",
            extract="transmitted",
            fesc=covering_fraction_blr,
            scale_by="bolometric_luminosity",
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
        )

        # Combine the models
        model = BlackHoleEmissionModel(
            label="disc_transmitted",
            combine=(nlr, blr),
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
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
        model = BlackHoleEmissionModel(
            grid=grid,
            label="disc_escaped",
            extract="incident",
            fesc=1.0 - covering_fraction_nlr - covering_fraction_blr,
            scale_by="bolometric_luminosity",
        )

        return model

    def _make_disc(self):
        """Make the disc spectra."""
        return BlackHoleEmissionModel(
            label="disc",
            combine=(self.disc_transmitted, self.disc_escaped),
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
        )

    def _make_line_regions(self, nlr_grid, blr_grid):
        """Make the line regions."""
        # Make the line regions with fixed inclination
        nlr = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="nlr",
            extract="nebular",
            fixed_parameters={"cosine_inclination": 0.5},
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            scale_by="bolometric_luminosity",
        )
        blr = BlackHoleEmissionModel(
            grid=blr_grid,
            label="blr",
            extract="nebular",
            fixed_parameters={"cosine_inclination": 0.5},
            mask_attr="_torus_edgeon_cond",
            mask_thresh=90 * deg,
            mask_op="<",
            scale_by="bolometric_luminosity",
        )
        return nlr, blr

    def _make_torus(self, torus_emission_model):
        """Make the torus spectra."""
        return BlackHoleEmissionModel(
            label="torus",
            generator=torus_emission_model,
            lum_intrinsic_model=self.disc_incident_isotropic,
            scale_by=("torus_fraction",),
        )
