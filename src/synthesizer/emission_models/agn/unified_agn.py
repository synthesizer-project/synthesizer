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
from synthesizer.emission_models.models import (
    AttenuatedEmission,
    DustEmission,
)
from synthesizer.emission_models.transformers import (
    CoveringFraction,
    EscapingFraction,
)
from synthesizer.emission_models.utils import ParameterFunction
from synthesizer.exceptions import (
    InconsistentParameter,
)


def torus_edgeon_condition(inclination, theta_torus):
    """When this is > 90 deg the torus obscures the disc.

    We will wrap this function in a ParameterFunction to use for masking
    within the UnifiedAGN model.

    Args:
        inclination (unyt_array):
            The inclination of the black hole.
        theta_torus (unyt_array):
            The torus opening angle.
    """
    return inclination + theta_torus


# Wrap the torus_edgeon_condition in a ParameterFunction
torus_edgeon_handler = ParameterFunction(
    func=torus_edgeon_condition,
    sets="torus_edgeon_cond",
    func_args=("inclination", "theta_torus"),
)


class UnifiedAGNIntrinsic(BlackHoleEmissionModel):
    """An emission model that defines the Unified AGN model with no dust.

    The UnifiedAGN model includes a disc, nlr, blr and torus component and
    combines these components taking into account geometry of the disc
    and torus. This model does not include diffuse dust emission.

    Attributes:
        disc_incident_isotropic (BlackHoleEmissionModel):
            The disc emission model assuming isotropic emission.
        disc_incident (BlackHoleEmissionModel):
            The disc emission model accounting for the geometry but unmasked.
        disc_averaged (BlackHoleEmissionModel):
            The inclination averaged observed disc emission.
        disc_averaged_without_torus (BlackHoleEmissionModel):
            The inclination averaged observed disc ignoring the torus.
        disc_transmitted_nlr (BlackHoleEmissionModel):
            The disc spectrum transmitted through the NLR
        disc_transmitted_blr (BlackHoleEmissionModel):
            The disc spectrum transmitted through the BLR
        disc_transmitted (BlackHoleEmissionModel):
            The disc transmitted emission
        disc_transmitted_weighted_combination (BlackHoleEmissionModel):
            The disc transmitted weighted combination emission
        disc (BlackHoleEmissionModel):
            The disc emission model
        nlr (BlackHoleEmissionModel):
            The NLR emission model
        blr (BlackHoleEmissionModel):
            The BLR emission model
        line_regions (BlackHoleEmissionModel):
            The combined BLR and NLR emission
        nlr_continuum (BlackHoleEmissionModel):
            The NLR continuum emission
        blr_continuum (BlackHoleEmissionModel):
            The BLR continuum emission
        torus (BlackHoleEmissionModel):
            The torus emission model
    """

    def __init__(
        self,
        nlr_grid,
        blr_grid,
        torus_emission_model,
        disc_transmission="random",
        label="intrinsic",
        **kwargs,
    ):
        """Initialize the UnifiedAGN model.

        Args:
            nlr_grid (synthesizer.grid.Grid):
                The grid for the NLR.
            blr_grid (synthesizer.grid.Grid):
                The grid for the BLR.
            torus_emission_model (synthesizer.dust.EmissionModel):
                The dust emission model to use for the torus.
            disc_transmission (str):
                The disc transmission model.
            label (str):
                The label for the resulting spectra. Defaults to "intrinsic".
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        # Check that certain parameters are not provided as they would produce
        # inconsistent results.
        for arg_to_check in ["inclination", "theta_torus"]:
            if arg_to_check in kwargs.keys():
                raise exceptions.InconsistentArguments(
                    f"{arg_to_check} must be set on the blackhole object, "
                    "not passed as a keyword argument to the emission model."
                )

        # Get the incident isotropic disc emission model
        self.disc_incident_isotropic = self._make_disc_incident_isotropic(
            nlr_grid,
            **kwargs,
        )

        # Get the incident disc emission model
        self.disc_incident, self.disc_incident_masked = (
            self._make_disc_incident(
                nlr_grid,
                **kwargs,
            )
        )

        # Get the emission transmitted through the BLR and NLR
        (
            self.disc_transmitted_nlr_full,
            self.disc_transmitted_blr_full,
            self.disc_transmitted_nlr_isotropic_full,
            self.disc_transmitted_blr_isotropic_full,
        ) = self._make_disc_transmitted_lr_full(
            nlr_grid,
            blr_grid,
            **kwargs,
        )

        # Get the averaged disc spectrum with and without the torus
        (
            self.disc_averaged,
            self.disc_averaged_without_torus,
        ) = self._make_disc_averaged(
            **kwargs,
        )

        # Get the weighted_combination transmitted disc spectrum
        self.disc_transmitted_weighted_combination = (
            self._make_disc_transmitted_weighted_combination(
                **kwargs,
            )
        )

        # Get the transmitted disc emission spectrum
        self.disc_transmitted = self._make_disc_transmitted(
            disc_transmission,
            **kwargs,
        )

        # Create the disc spectrum
        self.disc = self._make_disc(**kwargs)

        # Get the line region spectra
        self.nlr, self.nlr_continuum, self.blr, self.blr_continuum = (
            self._make_line_regions(
                nlr_grid,
                blr_grid,
                **kwargs,
            )
        )

        # Combined line region spectrum.
        self.line_regions = BlackHoleEmissionModel(
            label="line_regions",
            combine=(
                self.nlr,
                self.blr,
            ),
            **kwargs,
        )

        # Get the torus emission model
        self.torus = self._make_torus(torus_emission_model, **kwargs)

        return BlackHoleEmissionModel.__init__(
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
                self.disc_transmitted,
                self.disc_transmitted_weighted_combination,
                self.disc,
                self.line_regions,
                self.nlr_continuum,
                self.blr_continuum,
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
        # unmasked
        disc_incident = BlackHoleEmissionModel(
            grid=grid,
            label="disc_incident",
            extract="incident",
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        # masked for the torus
        disc_incident_masked = BlackHoleEmissionModel(
            grid=grid,
            label="disc_incident_masked",
            extract="incident",
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            mask_attr="torus_edgeon_cond",
            torus_edgeon_cond=torus_edgeon_handler,
            mask_thresh=90 * deg,
            mask_op="<",
            **kwargs,
        )

        return disc_incident, disc_incident_masked

    def _make_disc_transmitted_lr_full(
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
        # Calculate the "full" spectrum transmitted through the line regions.
        # This masks for the torus but does not yet include the covering
        # fraction.
        disc_transmitted_nlr_full = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="disc_transmitted_nlr_full",
            extract="transmitted",
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            mask_attr="torus_edgeon_cond",
            torus_edgeon_cond=torus_edgeon_handler,
            mask_thresh=90 * deg,
            mask_op="<",
            **kwargs,
        )

        disc_transmitted_blr_full = BlackHoleEmissionModel(
            grid=blr_grid,
            label="disc_transmitted_blr_full",
            extract="transmitted",
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            mask_attr="torus_edgeon_cond",
            torus_edgeon_cond=torus_edgeon_handler,
            mask_thresh=90 * deg,
            mask_op="<",
            **kwargs,
        )

        # Calculate the istropic spectrum transmitted through each of the line
        # regions. This does not include masking of the spectrum.
        disc_transmitted_nlr_isotropic_full = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="disc_transmitted_nlr_isotropic_full",
            extract="transmitted",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            **kwargs,
        )

        disc_transmitted_blr_isotropic_full = BlackHoleEmissionModel(
            grid=blr_grid,
            label="disc_transmitted_blr_isotropic_full",
            extract="transmitted",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        return (
            disc_transmitted_nlr_full,
            disc_transmitted_blr_full,
            disc_transmitted_nlr_isotropic_full,
            disc_transmitted_blr_isotropic_full,
        )

    def _make_disc_transmitted_weighted_combination(
        self,
        **kwargs,
    ):
        """Calculate the weighted_combination disc spectrum.

        Note: when the viewing angle (inlination) meets the torus criteria
        it is always blocked.
        """
        # Now calculate the disc_escaped emission using this transmission
        # fraction.
        disc_escaped_weighted = BlackHoleEmissionModel(
            label="disc_escaped_weighted",
            apply_to=self.disc_incident_masked,
            transformer=CoveringFraction(covering_attrs=("escape_fraction",)),
            **kwargs,
        )

        # Now calculate the disc_transmitted_nlr emission using this
        # transmission fraction.
        disc_transmitted_nlr_weighted = BlackHoleEmissionModel(
            label="disc_transmitted_nlr_weighted",
            apply_to=self.disc_transmitted_nlr_full,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_nlr",)
            ),
            **kwargs,
        )

        # Now calculate the disc_transmitted_blr emission using this
        # transmission fraction.
        disc_transmitted_blr_weighted = BlackHoleEmissionModel(
            label="disc_transmitted_blr_weighted",
            apply_to=self.disc_transmitted_blr_full,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_blr",)
            ),
            **kwargs,
        )

        # Now combine the three different components to produce the total.
        disc_transmitted_averaged = BlackHoleEmissionModel(
            label="disc_transmitted_weighted_combination",
            combine=(
                disc_escaped_weighted,
                disc_transmitted_nlr_weighted,
                disc_transmitted_blr_weighted,
            ),
            **kwargs,
        )

        return disc_transmitted_averaged

    def _make_disc_transmitted(
        self,
        disc_transmission,
        **kwargs,
    ):
        """Calculate the observed disc spectrum.

        For an individual blackhole there are four options. Either the disc
        emission escapes (disc_transmission='none'), is transmitted through
        the NLR (disc_transmission='nlr'), is transmitted through the BLR
        (disc_transmission='blr'), or is the weighted combination
        (disc_transmission='weighted_combination').

        The latter scenario is always calculated but is not used to calculate
        the disc_transmitted spectrum unless explicitly asked for. At the
        initialisation of the blackhole object one of the other three
        scenarios is randomly assigned based on the nlr and blr covering
        fractions. The default behaviour of UnifiedAGN is to use these
        randomly assigned scenarios. However, by providing the
        disc_transmission keyword argument to UnifiedAGN we can overide this
        and force all blackholes to adopt the same transmission scenario.

        Note: when the viewing angle (inlination) meets the torus criteria
        it is always blocked.

        Args:
            disc_transmission (str): The disc transmission sceanrio.
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        if disc_transmission in ["escaped", "none", "nlr", "blr", "random"]:
            # If disc_transmission == 'none' the emission seen by the observer
            # is simply the incident emission. This step also accounts for the
            # torus.
            if disc_transmission in ["none", "escaped"]:
                transmission_fraction_escape = 1.0
                transmission_fraction_nlr = 0.0
                transmission_fraction_blr = 0.0

            # If disc_transmission == 'nlr' the emission seen by the observer
            # is the spectrum transmitted through the NLR. This step also
            # accounts for the torus.
            elif disc_transmission == "nlr":
                transmission_fraction_escape = 0.0
                transmission_fraction_nlr = 1.0
                transmission_fraction_blr = 0.0

            # If disc_transmission == 'blr' the emission seen by the observer
            # is the spectrum transmitted through the BLR. This step also
            # accounts for the torus.
            elif disc_transmission == "blr":
                transmission_fraction_escape = 0.0
                transmission_fraction_nlr = 0.0
                transmission_fraction_blr = 1.0

            # If disc_transmission == 'random' the emission seen by the
            # observer is chosen at random for each blackhole using covering
            # fractions. This is only possible if the transmission fractions
            # have been set on the component.
            elif disc_transmission == "random":
                transmission_fraction_escape = "transmission_fraction_escape"
                transmission_fraction_nlr = "transmission_fraction_nlr"
                transmission_fraction_blr = "transmission_fraction_blr"

            # Now calculate the disc_escaped emission using this transmission
            # fraction.
            self.disc_escaped = BlackHoleEmissionModel(
                label="disc_escaped",
                apply_to=self.disc_incident_masked,
                transformer=CoveringFraction(
                    covering_attrs=("transmission_fraction_escape",)
                ),
                transmission_fraction_escape=transmission_fraction_escape,
                **kwargs,
            )

            # Now calculate the disc_transmitted_nlr emission using this
            # transmission fraction.
            self.disc_transmitted_nlr = BlackHoleEmissionModel(
                label="disc_transmitted_nlr",
                apply_to=self.disc_transmitted_nlr_full,
                transformer=CoveringFraction(
                    covering_attrs=("transmission_fraction_nlr",)
                ),
                transmission_fraction_nlr=transmission_fraction_nlr,
                **kwargs,
            )

            # Now calculate the disc_transmitted_blr emission using this
            # transmission fraction.
            self.disc_transmitted_blr = BlackHoleEmissionModel(
                label="disc_transmitted_blr",
                apply_to=self.disc_transmitted_blr_full,
                transformer=CoveringFraction(
                    covering_attrs=("transmission_fraction_blr",)
                ),
                transmission_fraction_blr=transmission_fraction_blr,
                **kwargs,
            )

            # Now combine the three different components to produce the total.
            disc_transmitted = BlackHoleEmissionModel(
                label="disc_transmitted",
                combine=(
                    self.disc_escaped,
                    self.disc_transmitted_nlr,
                    self.disc_transmitted_blr,
                ),
                **kwargs,
            )

        # If weighted_combination is selected the transmitted is simply the
        # combination of the three scenarios weighted by the respective
        # covering fractions.
        elif disc_transmission == "weighted_combination":
            disc_transmitted = BlackHoleEmissionModel(
                label="disc_transmitted",
                combine=(self.disc_transmitted_weighted_combination,),
                **kwargs,
            )

        # Otherwise raise exception.
        else:
            raise InconsistentParameter("disc_transmission not recognised")

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
        **kwargs,
    ):
        """Calculate the isotropic (inclination averaged) disc spectrum."""
        # Calculate the total amount of disc emission which escapes the BLR
        # and NLR, ignoring the torus
        disc_escaped_isotropic = BlackHoleEmissionModel(
            label="disc_escaped_isotropic",
            apply_to=self.disc_incident_isotropic,
            transformer=EscapingFraction(
                covering_attrs=(
                    "covering_fraction_blr",
                    "covering_fraction_nlr",
                )
            ),
            **kwargs,
        )

        # Calculate the total amount of disc emission which is transmitted
        # through the NLR ignoring the torus
        disc_transmitted_nlr_isotropic = BlackHoleEmissionModel(
            label="disc_transmitted_nlr_isotropic",
            apply_to=self.disc_transmitted_nlr_isotropic_full,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_nlr",)
            ),
            **kwargs,
        )

        # Calculate the total amount of disc emission which is transmitted
        # through the BLR ignoring the torus
        disc_transmitted_blr_isotropic = BlackHoleEmissionModel(
            label="disc_transmitted_blr_isotropic",
            apply_to=self.disc_transmitted_blr_isotropic_full,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_blr",)
            ),
            **kwargs,
        )

        # Combine these three
        disc_averaged_without_torus = BlackHoleEmissionModel(
            label="disc_averaged_without_torus",
            combine=(
                disc_escaped_isotropic,
                disc_transmitted_nlr_isotropic,
                disc_transmitted_blr_isotropic,
            ),
            **kwargs,
        )

        # Now adjust for the torus. This is essentially the averaged light
        # received from the disc.
        disc_averaged = BlackHoleEmissionModel(
            label="disc_averaged",
            apply_to=disc_averaged_without_torus,
            transformer=EscapingFraction(covering_attrs=("torus_fraction",)),
            **kwargs,
        )

        return disc_averaged, disc_averaged_without_torus

    def _make_line_regions(
        self,
        nlr_grid,
        blr_grid,
        **kwargs,
    ):
        """Make the line regions.

        These use the nebular spectra in the relevant grids but utilise the
        isotropic emission (cosine_inclination=0.5) instead of taking account
        of the observer inclination since the BLR are illuminated by a wider
        range of lines-of-sight.
        """
        # Extract the NLR spectra from the grid. Here cosine_inclination=0.5
        # because we are using the isotropic disc emission. No masking is
        # applied for the NLR because it always assumed to be visible.
        full_nlr = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="full_reprocessed_nlr",
            extract="nebular",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            **kwargs,
        )

        # Extract the BLR spectra from the grid. Here cosine_inclination=0.5
        # because we are using the isotropic disc emission. Masking is
        # applied for the BLR because it assumed to be blocked by the torus.
        full_blr = BlackHoleEmissionModel(
            grid=blr_grid,
            label="full_reprocessed_blr",
            extract="nebular",
            mask_attr="torus_edgeon_cond",
            torus_edgeon_cond=torus_edgeon_handler,
            mask_thresh=90 * deg,
            mask_op="<",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        # As above but for the continuum emission alone
        full_nlr_continuum = BlackHoleEmissionModel(
            grid=nlr_grid,
            label="full_continuum_nlr",
            extract="nebular_continuum",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_nlr",
            ionisation_parameter="ionisation_parameter_nlr",
            **kwargs,
        )
        full_blr_continuum = BlackHoleEmissionModel(
            grid=blr_grid,
            label="full_continuum_blr",
            extract="nebular_continuum",
            mask_attr="torus_edgeon_cond",
            torus_edgeon_cond=torus_edgeon_handler,
            mask_thresh=90 * deg,
            mask_op="<",
            cosine_inclination=0.5,
            hydrogen_density="hydrogen_density_blr",
            ionisation_parameter="ionisation_parameter_blr",
            **kwargs,
        )

        # Now apply the relevant covering fractions to the different spectra
        nlr = BlackHoleEmissionModel(
            label="nlr",
            apply_to=full_nlr,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_nlr",)
            ),
            **kwargs,
        )
        blr = BlackHoleEmissionModel(
            label="blr",
            apply_to=full_blr,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_blr",)
            ),
            **kwargs,
        )
        nlr_continuum = BlackHoleEmissionModel(
            label="nlr_continuum",
            apply_to=full_nlr_continuum,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_nlr",)
            ),
            **kwargs,
        )
        blr_continuum = BlackHoleEmissionModel(
            label="blr_continuum",
            apply_to=full_blr_continuum,
            transformer=CoveringFraction(
                covering_attrs=("covering_fraction_blr",)
            ),
            **kwargs,
        )

        return nlr, nlr_continuum, blr, blr_continuum

    def _make_torus(self, torus_emission_model, **kwargs):
        """Make the torus spectra."""
        return BlackHoleEmissionModel(
            label="torus",
            generator=torus_emission_model,
            lum_intrinsic_model=self.disc_incident_isotropic,
            scale_by="torus_fraction",
            **kwargs,
        )


class UnifiedAGNWithDiffuseDustAttenuation(BlackHoleEmissionModel):
    """An emission model that defines the Unified AGN model.

    The UnifiedAGN model includes a disc, nlr, blr and torus component and
    combines these components taking into account geometry of the disc
    and torus. This variant includes dust attenuation.

    Attributes:
        intrinsic (BlackHoleEmissionModel):
            The intrinsic emission
    """

    def __init__(
        self,
        nlr_grid,
        blr_grid,
        torus_emission_model,
        diffuse_dust_curve,
        disc_transmission="random",
        label="attenuated",
        tau_v="tau_v",
        **kwargs,
    ):
        """Initialize the UnifiedAGN model.

        Args:
            nlr_grid (synthesizer.grid.Grid):
                The grid for the NLR.
            blr_grid (synthesizer.grid.Grid):
                The grid for the BLR.
            torus_emission_model (synthesizer.dust.EmissionModel):
                The dust emission model to use for the torus.
            diffuse_dust_curve (synthesizer.emission_models.attenuation):
                The dust attenuation curve for diffuse dust.
            disc_transmission (str):
                The disc transmission model.
            label (str):
                The label for the resulting spectra. This defaults to
                "attenuated".
            tau_v (str):
                The attribute on the emitter to use for tau_v
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        # Calculate the intrinsic emission
        self.intrinsic = UnifiedAGNIntrinsic(
            nlr_grid,
            blr_grid,
            torus_emission_model,
            disc_transmission=disc_transmission,
            **kwargs,
        )

        # Include attenuation from diffuse dust
        AttenuatedEmission.__init__(
            self,
            dust_curve=diffuse_dust_curve,
            apply_to=self.intrinsic,
            tau_v=tau_v,
            emitter="blackhole",
            label=label,
            **kwargs,
        )


class UnifiedAGNWithDiffuseDustAttenuationAndEmission(BlackHoleEmissionModel):
    """An emission model that defines the Unified AGN model.

    The UnifiedAGN model includes a disc, nlr, blr and torus component and
    combines these components taking into account geometry of the disc
    and torus. This variant includes dust attenuation and emission.

    Attributes:
        intrinsic (BlackHoleEmissionModel):
            The intrinsic emission
        attenuated (BlackHoleEmissionModel):
            The attenuated emission
        diffuse_dust_emission (BlackHoleEmissionModel):
            The diffuse dust emission
    """

    def __init__(
        self,
        nlr_grid,
        blr_grid,
        torus_emission_model,
        diffuse_dust_curve,
        diffuse_dust_emission_model,
        tau_v="tau_v",
        disc_transmission="random",
        label="total",
        **kwargs,
    ):
        """Initialize the UnifiedAGN model.

        Args:
            nlr_grid (synthesizer.grid.Grid):
                The grid for the NLR.
            blr_grid (synthesizer.grid.Grid):
                The grid for the BLR.
            torus_emission_model (synthesizer.dust.EmissionModel):
                The dust emission model to use for the torus.
            disc_transmission (str):
                The disc transmission model.
            diffuse_dust_curve (synthesizer.emission_models.attenuation):
                The dust attenuation curve for diffuse dust.
            diffuse_dust_emission_model:
                The diffuse dust emission model.
            tau_v (str):
                The attribute on the emitter to use for tau_v
            label (str):
                The label for the resulting spectra. When dust attenuation and
                emission is included this defaults to "total" otherwise it
                defaults to "intrinsic".
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        # Calculate the intrinsic emission
        self.intrinsic = UnifiedAGNIntrinsic(
            nlr_grid,
            blr_grid,
            torus_emission_model,
            disc_transmission=disc_transmission,
            **kwargs,
        )

        # Include attenuation from diffuse dust
        self.attenuated = AttenuatedEmission(
            dust_curve=diffuse_dust_curve,
            apply_to=self.intrinsic,
            tau_v=tau_v,
            emitter="blackhole",
            label="attenuated",
            **kwargs,
        )

        # Add diffuse dust emission
        self.diffuse_dust_emission = DustEmission(
            dust_emission_model=diffuse_dust_emission_model,
            dust_lum_intrinsic=self.intrinsic,
            dust_lum_attenuated=self.attenuated,
            emitter="blackhole",
            label="diffuse_dust_emission",
            **kwargs,
        )

        # Finally make the total model, this is attenuated +
        # diffuse_dust_emission
        BlackHoleEmissionModel.__init__(
            self,
            label=label,
            combine=(self.attenuated, self.diffuse_dust_emission),
            related_models=(
                self.intrinsic,
                self.attenuated,
                self.diffuse_dust_emission,
            ),
            **kwargs,
        )


class UnifiedAGN(BlackHoleEmissionModel):
    """An emission model that defines the Unified AGN model.

    The UnifiedAGN model includes a disc, nlr, blr and torus component and
    combines these components taking into account geometry of the disc
    and torus. This variant includes dust attenuation.
    """

    def __new__(
        cls,
        nlr_grid,
        blr_grid,
        torus_emission_model,
        disc_transmission="random",
        diffuse_dust_curve=None,
        diffuse_dust_emission_model=None,
        label=None,
        **kwargs,
    ):
        """Initialize the UnifiedAGN model.

        Args:
            nlr_grid (synthesizer.grid.Grid):
                The grid for the NLR.
            blr_grid (synthesizer.grid.Grid):
                The grid for the BLR.
            torus_emission_model (synthesizer.dust.EmissionModel):
                The dust emission model to use for the torus.
            disc_transmission (str):
                The disc transmission model.
            diffuse_dust_curve (synthesizer.emission_models.attenuation):
                The dust attenuation curve for diffuse dust.
            diffuse_dust_emission_model:
                The diffuse dust emission model.
            label (str):
                The label for the resulting spectra. When dust attenuation and
                emission is included this defaults to "total" otherwise it
                defaults to "intrinsic".
            **kwargs: Any additional keyword arguments to pass to the
                BlackHoleEmissionModel.
        """
        # Validate that dust emission model is not provided without a curve
        if diffuse_dust_emission_model and not diffuse_dust_curve:
            raise exceptions.InconsistentArguments(
                "diffuse_dust_emission_model requires diffuse_dust_curve to "
                "be specified."
            )

        # If diffuse_dust_curve and diffuse_dust_emission_model provided then
        # include these
        if diffuse_dust_curve and diffuse_dust_emission_model:
            if label is None:
                label = "total"
            return UnifiedAGNWithDiffuseDustAttenuationAndEmission(
                nlr_grid,
                blr_grid,
                torus_emission_model,
                diffuse_dust_curve,
                diffuse_dust_emission_model,
                disc_transmission=disc_transmission,
                label=label,
                **kwargs,
            )

        # Otherwise if only the diffuse_dust_curve is provided:

        elif diffuse_dust_curve:
            if label is None:
                label = "attenuated"
            return UnifiedAGNWithDiffuseDustAttenuation(
                nlr_grid,
                blr_grid,
                torus_emission_model,
                diffuse_dust_curve,
                disc_transmission=disc_transmission,
                label=label,
                **kwargs,
            )

        # Otherwise return the intrinsic emission
        else:
            if label is None:
                label = "intrinsic"
            return UnifiedAGNIntrinsic(
                nlr_grid,
                blr_grid,
                torus_emission_model,
                disc_transmission=disc_transmission,
                label=label,
                **kwargs,
            )
