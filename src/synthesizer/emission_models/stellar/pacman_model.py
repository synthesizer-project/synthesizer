"""A module defining the Pacman emission models.

This module defines the PacmanEmission and BimodalPacmanEmission classes which
are used to define the emission models for the Pacman model. Both these models
combine various differen spectra together to produce a final total emission
spectrum.

The PacmanEmission model is used to define the emission model for a single
population of stars. Including both intrinsic and attenuate emission, and
if a dust emission model is given also dust emission. It includes the option
to include escaped emission for a given escape fraction, and if a lyman alpha
escape fraction is given, a more sophisticated nebular emission model is used,
including line and nebuluar continuum emission.

The BimodalPacmanEmission model is similar to the PacmanEmission model but
splits the emission into a young and old population.

The Charlot & Fall (2000) model if a special of the BimodalPacmanEmission
model and is also included. This model is identical to the
BimodalPacmanEmission model but with a fixed age pivot of 10^7 Myr and no
escaped emission.

Example::

    To create a PacmanEmission model for a grid with a V-band optical depth of
    0.1 and a dust curve, one would do the following:

    dust_curve = PowerLaw(...)
    model = PacmanEmission(grid, 0.1, dust_curve)

    To create a CharlotFall2000 model, you can use the following code:

    tau_v_ism = 0.33
    tau_v_birth = 0.67
    dust_curve_ism = PowerLaw(...)
    dust_curve_birth = PowerLaw(...)
    age_pivot = 7 * dimensionless
    dust_emission_ism = BlackBody(...)
    dust_emission_birth = GreyBody(...)
    model = CharlotFall2000(
        grid,
        tau_v_ism,
        tau_v_birth,
        dust_curve_ism,
        dust_curve_birth,
        age_pivot,
        dust_emission_ism,
        dust_emission_birth,
    )
"""

from unyt import dimensionless

from synthesizer.emission_models.attenuation import Calzetti2000, PowerLaw
from synthesizer.emission_models.base_model import (
    EmissionModel,
    StellarEmissionModel,
)
from synthesizer.emission_models.models import (
    AttenuatedEmission,
    DustEmission,
)
from synthesizer.emission_models.stellar.models import (
    EmergentEmission,
    IncidentEmission,
    NebularContinuumEmission,
    NebularEmission,
    NebularLineEmission,
    ReprocessedEmission,
    TransmittedEmission,
)


class PacmanEmissionNoEscapedNoDust(StellarEmissionModel):
    """A class defining the Pacman model without escape fraction.

    This model defines both intrinsic and attenuated steller emission without
    dust emission. If a lyman alpha escape fraction is given, a more
    sophisticated nebular emission model is used, including line and
    nebuluar continuum emission with the amount of lyman alpha emission
    scaled by the escape fraction.

    This model will produce
        - incident: the stellar emission incident onto the ISM.
        - nebular: the stellar emission from nebulae.
        - transmitted: the stellar emission transmitted through the ISM.
        - reprocessed: the stellar emission reprocessed by the ISM.
        - attenuated: the intrinsic emission attenuated by dust.
    """

    def __init__(
        self,
        grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        **kwargs,
    ):
        """Initialize the PacmanEmissionNoEscapeNoDust model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v (float):
                The V-band optical depth.
            dust_curve (synthesizer.emission_models.Transformer):
                The assumed dust curve. Defaults to `PowerLaw`, with
                default parameters.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction.
            label (str):
                The label for the total emission model. If `None` this will
                be set to "attenuated".
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The PacmanEmission style models require a reprocessed grid."
            )

        # Create the models we need
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            label="transmitted",
            fesc=0.0,  # No escape fraction
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            label="nebular_line",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="nebular_continuum",
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            label="reprocessed",
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )

        # Make the attenuated emission model
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label="attenuated" if label is None else label,
            apply_to=reprocessed,
            dust_curve=dust_curve,
            tau_v=tau_v,
            related_models=(incident,),
            **kwargs,
        )


class PacmanEmissionNoEscapedWithDust(EmissionModel):
    """A class defining the Pacman model with escape fraction + dust emission.

    This model defines both intrinsic and attenuated steller emission with
    dust emission. If a lyman alpha escape fraction is given, a more
    sophisticated nebular emission model is used, including line and
    nebuluar continuum emission with the amount of lyman alpha emission
    scaled by the escape fraction.

    This model will produce:
        - incident: the stellar emission incident onto the ISM.
        - nebular: the stellar emission from nebulae.
        - transmitted: the stellar emission transmitted through the ISM.
        - reprocessed: the stellar emission reprocessed by the ISM.
        - attenuated: the intrinsic emission attenuated by dust.
        - dust_emission: the emission from dust.
        - total: the final total combined emission.
    """

    def __init__(
        self,
        grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        dust_emission=None,
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        stellar_dust=True,
        **kwargs,
    ):
        """Initialize the PacmanEmissionNoEscapeWithDust model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v (float):
                The V-band optical depth.
            dust_curve (synthesizer.emission_models.Transformer):
                The assumed dust curve. Defaults to `PowerLaw`, with
                default parameters.
            dust_emission (synthesizer.dust.EmissionModel): The dust
                emission.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction.
            label (str):
                The label for the total emission model. If `None` this will
                be set to "total" or "emergent" if dust_emission is `None`.
            stellar_dust (bool):
                If `True`, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The PacmanEmission style models require a reprocessed grid."
            )

        # Create the models we need
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            label="transmitted",
            fesc=0.0,  # No escape fraction
            **kwargs,
        )
        nebular_line = NebularLineEmission(
            grid=grid,
            label="nebular_line",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="nebular_continuum",
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            label="reprocessed",
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            tau_v=tau_v,
            **kwargs,
        )
        dust_emission_model = DustEmission(
            label="dust_emission",
            dust_emission_model=dust_emission,
            dust_lum_intrinsic=reprocessed,
            dust_lum_attenuated=attenuated,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )

        # Finally make the TotalEmission model, this is
        # dust_emission + attenuated
        EmissionModel.__init__(
            self,
            grid=grid,
            label="total" if label is None else label,
            combine=(dust_emission_model, attenuated),
            related_models=(incident,),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )


class PacmanEmissionWithEscapedNoDust(StellarEmissionModel):
    """A class defining the Pacman model with fesc and no dust emission.

    This model defines both intrinsic and attenuated steller emission without
    dust emission. If a lyman alpha escape fraction is given, a more
    sophisticated nebular emission model is used, including line and
    nebuluar continuum emission with the amount of lyman alpha emission
    scaled by the escape fraction.

    This model will produce:
        - incident: the stellar emission incident onto the ISM.
        - nebular: the stellar emission from nebulae.
        - transmitted: the stellar emission transmitted through the ISM.
        - reprocessed: the stellar emission reprocessed by the ISM.
        - intrinsic: the intrinsic emission which is reprocessed + escaped.
        - escaped: the incident emission that completely escapes the ISM.
        - attenuated: the intrinsic emission attenuated by dust.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission (i.e. attenuated + escaped).
    """

    def __init__(
        self,
        grid,
        tau_v="tau_v",
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        **kwargs,
    ):
        """Initialize the PacmanEmissionWithEscapeNoDust model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v (float):
                The V-band optical depth.
            fesc (float):
                The escape fraction.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction.
            label (str):
                The label for the total emission model. If `None` this will
                be set to "emergent".
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The PacmanEmission style models require a reprocessed grid."
            )

        # Ensure we have a non-zero escape fraction
        if fesc == 0.0 or fesc is None:
            raise ValueError(
                "The PacmanEmissionWithEscapeNoDust model requires a non-zero "
                "escape fraction."
            )

        # Create the models we need
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            label="transmitted",
            fesc=fesc,
            incident=incident,
            **kwargs,
        )
        escaped = transmitted["escaped"]
        nebular_line = NebularLineEmission(
            grid=grid,
            label="nebular_line",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="nebular_continuum",
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            label="reprocessed",
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(reprocessed, escaped),
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            dust_curve=PowerLaw(),
            apply_to=reprocessed,
            emitter="stellar",
            tau_v=tau_v,
            **kwargs,
        )
        # Finally make the emergent model, this is attenuated + escaped
        StellarEmissionModel.__init__(
            self,
            label="emergent",
            grid=grid,
            combine=(attenuated, escaped),
            related_models=(intrinsic,),
            **kwargs,
        )


class PacmanEmissionWithEscapedWithDust(StellarEmissionModel):
    """A class defining the Pacman model with fesc and dust emission.

    This model defines both intrinsic and attenuated steller emission with
    dust emission. If a lyman alpha escape fraction is given, a more
    sophisticated nebular emission model is used, including line and
    nebuluar continuum emission with the amount of lyman alpha emission
    scaled by the escape fraction.

    This model will produce:
        - incident: the stellar emission incident onto the ISM.
        - nebular: the stellar emission from nebulae.
        - transmitted: the stellar emission transmitted through the ISM.
        - reprocessed: the stellar emission reprocessed by the ISM.
        - intrinsic: the intrinsic emission which is reprocessed + escaped.
        - escaped: the incident emission that completely escapes the ISM.
        - attenuated: the intrinsic emission attenuated by dust.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission (i.e. attenuated + escaped).
        - dust_emission: the emission from dust.
        - total: the final total combined emission.
    """

    def __init__(
        self,
        grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        dust_emission=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        stellar_dust=True,
        **kwargs,
    ):
        """Initialize the PacmanEmissionWithEscapeWithDust model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v (float):
                The V-band optical depth.
            dust_curve (synthesizer.emission_models.Transformer):
                The assumed dust curve. Defaults to `PowerLaw`, with
                default parameters.
            dust_emission (synthesizer.dust.EmissionModel): The dust
                emission.
            fesc (float):
                The escape fraction.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction.
            label (str):
                The label for the total emission model. If `None` this will
                be set to "total" or "emergent" if dust_emission is `None`.
            stellar_dust (bool):
                If `True`, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The PacmanEmission style models require a reprocessed grid."
            )

        # Ensure we have a non-zero escape fraction
        if fesc == 0.0 or fesc is None:
            raise ValueError(
                "The PacmanEmissionWithEscapeWithDust model requires a "
                "non-zero escape fraction."
            )

        # Create the models we need
        incident = IncidentEmission(
            grid=grid,
            label="incident",
            **kwargs,
        )
        transmitted = TransmittedEmission(
            grid=grid,
            label="transmitted",
            fesc=fesc,
            incident=incident,
            **kwargs,
        )
        escaped = transmitted["escaped"]
        nebular_line = NebularLineEmission(
            grid=grid,
            label="nebular_line",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="nebular_continuum",
            **kwargs,
        )
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            nebular_line=nebular_line,
            nebular_continuum=nebular_continuum,
            **kwargs,
        )
        reprocessed = ReprocessedEmission(
            grid=grid,
            label="reprocessed",
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(reprocessed, escaped),
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            dust_curve=dust_curve,
            apply_to=reprocessed,
            emitter="stellar",
            tau_v=tau_v,
            **kwargs,
        )
        emergent = EmergentEmission(
            grid=grid,
            label="emergent",
            attenuated=attenuated,
            escaped=escaped,
            **kwargs,
        )
        dust_emission_model = DustEmission(
            label="dust_emission",
            dust_emission_model=dust_emission,
            dust_lum_intrinsic=reprocessed,
            dust_lum_attenuated=attenuated,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )

        # Finally make the TotalEmission model, this is dust_emission +
        # emergent
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label="total" if label is None else label,
            combine=(dust_emission_model, emergent),
            related_models=(intrinsic,),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )


class PacmanEmission:
    """A class defining the Pacman model.

    This model defines both intrinsic and attenuated steller emission with or
    without dust emission. It also includes the option to include escaped
    emission for a given escape fraction. If a lyman alpha escape fraction is
    given, a more sophisticated nebular emission model is used, including line
    and nebuluar continuum emission.

    This model will always produce:
        - incident: the stellar emission incident onto the ISM.
        - nebular: the stellar emission from nebulae.
        - transmitted: the stellar emission transmitted through the ISM.
        - reprocessed: the stellar emission reprocessed by the ISM.
        - attenuated: the intrinsic emission attenuated by dust.

    If an escape fraction is given, it will also produce:
        - intrinsic: the intrinsic emission which is reprocessed + escaped.
        - escaped: the incident emission that completely escapes the ISM.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission (i.e. attenuated + escaped).

    If a dust emission model is given, it will also produce:
        - dust_emission: the emission from dust.
        - total: the final total combined emission.

    """

    def __new__(
        cls,
        grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        dust_emission=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        stellar_dust=True,
        **kwargs,
    ):
        """Get a PacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v (float):
                The V-band optical depth.
            dust_curve (synthesizer.emission_models.Transformer):
                The assumed dust curve. Defaults to `PowerLaw`, with
                default parameters.
            dust_emission (synthesizer.dust.EmissionModel): The dust
                emission.
            fesc (float):
                The escape fraction.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction.
            label (str):
                The label for the total emission model. If `None` this will
                be set to "total" (when dust emission is included),
                "attenuated" (when dust emission and an escape fraction are
                not included), and "emergent" (when dust emission is not
                included and an escape fraction is included).
            stellar_dust (bool):
                If `True`, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Are we ignoring the escape fraction?
        if fesc == 0.0 or fesc is None:
            # Do we have a dust emission model?
            if dust_emission is None:
                # No dust emission, no escape fraction, so we can use the
                # PacmanEmissionNoEscapeNoDust model
                return PacmanEmissionNoEscapedNoDust(
                    grid=grid,
                    tau_v=tau_v,
                    dust_curve=dust_curve,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    **kwargs,
                )
            else:
                # We have dust emission, no escape fraction, so we can use the
                # PacmanEmissionNoEscapeWithDust model
                return PacmanEmissionNoEscapedWithDust(
                    grid=grid,
                    tau_v=tau_v,
                    dust_curve=dust_curve,
                    dust_emission=dust_emission,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    stellar_dust=stellar_dust,
                    **kwargs,
                )
        # Ok, we have an escape fraction
        else:
            # Do we have a dust emission model?
            if dust_emission is None:
                # No dust emission, so we can use the
                # PacmanEmissionWithEscapeNoDust model
                return PacmanEmissionWithEscapedNoDust(
                    grid=grid,
                    tau_v=tau_v,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    **kwargs,
                )
            else:
                # We have dust emission, so we can use the
                # PacmanEmissionWithEscapeWithDust model
                return PacmanEmissionWithEscapedWithDust(
                    grid=grid,
                    tau_v=tau_v,
                    dust_curve=dust_curve,
                    dust_emission=dust_emission,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    stellar_dust=stellar_dust,
                    **kwargs,
                )


class BimodalPacmanEmissionNoEscapedNoDust(StellarEmissionModel):
    """A class defining the Bimodal Pacman model without fesc or dust emission.

    This model defines both intrinsic and attenuated stellar emission without
    dust emission, split into young and old populations. If a lyman alpha
    escape fraction is given, a more sophisticated nebular emission model is
    used, including line and nebular continuum emission with the amount of
    lyman alpha emission scaled by the escape fraction.

    This model will produce:
        - young_incident: the stellar emission incident onto the ISM for the
            young population.
        - old_incident: the stellar emission incident onto the ISM for the old
            population.
        - incident: the stellar emission incident onto the ISM for the
            combined population.
        - young_transmitted: the stellar emission transmitted through the ISM
            for the young population.
        - old_transmitted: the stellar emission transmitted through the ISM for
            the old population.
        - transmitted: the stellar emission transmitted through the ISM for the
            combined population.
        - young_nebular: the stellar emission from nebulae for the young
            population.
        - old_nebular: the stellar emission from nebulae for the old
            population.
        - nebular: the stellar emission from nebulae for the combined
            population.
        - young_reprocessed: the stellar emission reprocessed by the ISM for
            the young population.
        - old_reprocessed: the stellar emission reprocessed by the ISM for the
            old population.
        - reprocessed: the stellar emission reprocessed by the ISM for the
            combined population.
        - young_intrinsic: the intrinsic emission for the young population.
        - old_intrinsic: the intrinsic emission for the old population.
        - intrinsic: the intrinsic emission for the combined population.
        - young_attenuated_nebular: the nebular emission attenuated by dust
            for the young population.
        - young_attenuated_ism: the intrinsic emission attenuated by dust for
            the young population.
        - young_attenuated: the intrinsic emission attenuated by dust for the
            young population.
        - old_attenuated: the intrinsic emission attenuated by dust for the old
            population.
        - attenuated: the intrinsic emission attenuated by dust for the
            combined population.
    """

    def __init__(
        self,
        grid,
        tau_v_ism="tau_v_ism",
        tau_v_birth="tau_v_birth",
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        age_pivot=7 * dimensionless,
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        **kwargs,
    ):
        """Initialize the BimodalPacmanEmissionNoEscapeNoDust model.

        Args:
            grid(synthesizer.grid.Grid):
                The grid object.
            tau_v_ism(float):
                The V-band optical depth for the ISM.
            tau_v_birth(float):
                The V-band optical depth for the nebular.
            dust_curve_ism(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the ISM. Defaults to `Calzetti2000`
                with default parameters.
            dust_curve_birth(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the nebular. Defaults to
                `Calzetti2000` with default parameters.
            age_pivot(unyt.unyt_quantity):
                The age pivot between young and old populations,
                expressed in terms of log10(age) in Myr.
            fesc_ly_alpha(float):
                The Lyman alpha escape fraction.
            label(str):
                The label for the total emission model. If `None` this will
                be set to "attenuated".
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The BimodalPacmanEmission style models require a"
                " reprocessed grid."
            )

        # Create the incident models
        young_incident = IncidentEmission(
            grid=grid,
            label="young_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_incident = IncidentEmission(
            grid=grid,
            label="old_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        incident = StellarEmissionModel(
            label="incident",
            combine=(young_incident, old_incident),
            **kwargs,
        )

        # Create the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc=0.0,
            **kwargs,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc=0.0,
            **kwargs,
        )
        transmitted = StellarEmissionModel(
            label="transmitted",
            combine=(young_transmitted, old_transmitted),
            **kwargs,
        )

        # Create the nebular models
        young_nebular_line = NebularLineEmission(
            grid=grid,
            label="young_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        old_nebular_line = NebularLineEmission(
            grid=grid,
            label="old_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        young_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="young_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="old_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            nebular_line=young_nebular_line,
            nebular_continuum=young_nebular_continuum,
            **kwargs,
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            nebular_line=old_nebular_line,
            nebular_continuum=old_nebular_continuum,
            **kwargs,
        )
        nebular = StellarEmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
            **kwargs,
        )

        # Create the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            nebular=young_nebular,
            transmitted=young_transmitted,
            **kwargs,
        )
        old_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_reprocessed",
            nebular=old_nebular,
            transmitted=old_transmitted,
            **kwargs,
        )
        reprocessed = StellarEmissionModel(
            label="reprocessed",
            combine=(young_reprocessed, old_reprocessed),
            **kwargs,
        )

        # Create the intrinsic models
        young_intrinsic = StellarEmissionModel(
            label="young_intrinsic",
            combine=(young_reprocessed,),
            **kwargs,
        )
        old_intrinsic = StellarEmissionModel(
            label="old_intrinsic",
            combine=(old_reprocessed,),
            **kwargs,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
            **kwargs,
        )

        # Create the attenuated models
        young_attenuated_nebular = AttenuatedEmission(
            label="young_attenuated_nebular",
            tau_v=tau_v_birth,
            dust_curve=dust_curve_birth,
            apply_to=young_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=young_attenuated_nebular,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated = young_attenuated_ism
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=old_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        attenuated = StellarEmissionModel(
            label="attenuated",
            combine=(young_attenuated, old_attenuated),
            **kwargs,
        )

        # Store all models as related
        related_models = (
            young_incident,
            old_incident,
            incident,
            young_transmitted,
            old_transmitted,
            transmitted,
            young_nebular,
            old_nebular,
            nebular,
            young_reprocessed,
            old_reprocessed,
            reprocessed,
            young_intrinsic,
            old_intrinsic,
            intrinsic,
            young_attenuated_nebular,
            young_attenuated_ism,
            old_attenuated,
            attenuated,
        )

        # Make the final attenuated emission model
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label="attenuated" if label is None else label,
            combine=(young_attenuated, old_attenuated),
            related_models=related_models,
            **kwargs,
        )


class BimodalPacmanEmissionNoEscapedWithDust(EmissionModel):
    """A class defining the Bimodal Pacman model without fesc + dust emission.

    This model defines both intrinsic and attenuated stellar emission with
    dust emission, split into young and old populations. If a lyman alpha
    escape fraction is given, a more sophisticated nebular emission model is
    used, including line and nebular continuum emission with the amount of
    lyman alpha emission scaled by the escape fraction.

    This model will produce:
        - young_incident: the stellar emission incident onto the ISM for the
            young population.
        - old_incident: the stellar emission incident onto the ISM for the old
            population.
        - incident: the stellar emission incident onto the ISM for the
            combined population.
        - young_transmitted: the stellar emission transmitted through the ISM
            for the young population.
        - old_transmitted: the stellar emission transmitted through the ISM for
            the old population.
        - transmitted: the stellar emission transmitted through the ISM for the
            combined population.
        - young_nebular: the stellar emission from nebulae for the young
            population.
        - old_nebular: the stellar emission from nebulae for the old
            population.
        - nebular: the stellar emission from nebulae for the combined
            population.
        - young_reprocessed: the stellar emission reprocessed by the ISM for
            the young population.
        - old_reprocessed: the stellar emission reprocessed by the ISM for the
            old population.
        - reprocessed: the stellar emission reprocessed by the ISM for the
            combined population.
        - young_intrinsic: the intrinsic emission for the young population.
        - old_intrinsic: the intrinsic emission for the old population.
        - intrinsic: the intrinsic emission for the combined population.
        - young_attenuated_nebular: the nebular emission attenuated by dust
            for the young population.
        - young_attenuated_ism: the intrinsic emission attenuated by dust for
            the young population.
        - young_attenuated: the intrinsic emission attenuated by dust for the
            young population.
        - old_attenuated: the intrinsic emission attenuated by dust for the old
            population.
        - attenuated: the intrinsic emission attenuated by dust for the
            combined population.
        - young_emergent: the emission which emerges from the young stellar
            population.
        - old_emergent: the emission which emerges from the old stellar
            population.
        - emergent: the emission which emerges from the stellar population.
        - young_dust_emission_birth: the emission from dust for the young
            population.
        - young_dust_emission_ism: the emission from dust for the young
            population.
        - young_dust_emission: the emission from dust for the young population.
        - old_dust_emission: the emission from dust for the old population.
        - dust_emission: the emission from dust for the combined population.
        - young_total: the final total combined emission for the young
            population.
        - old_total: the final total combined emission for the old population.
        - total: the final total combined emission for the combined population.
    """

    def __init__(
        self,
        grid,
        tau_v_ism="tau_v_ism",
        tau_v_birth="tau_v_birth",
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        age_pivot=7 * dimensionless,
        dust_emission_ism=None,
        dust_emission_birth=None,
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        stellar_dust=True,
        **kwargs,
    ):
        """Initialize the BimodalPacmanEmissionNoEscapeWithDust model.

        Args:
            grid(synthesizer.grid.Grid):
                The grid object.
            tau_v_ism(float):
                The V-band optical depth for the ISM.
            tau_v_birth(float):
                The V-band optical depth for the nebular.
            dust_curve_ism(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the ISM. Defaults to `Calzetti2000`
                with default parameters.
            dust_curve_birth(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the nebular. Defaults to
                `Calzetti2000` with default parameters.
            age_pivot(unyt.unyt_quantity):
                The age pivot between young and old populations,
                expressed in terms of log10(age) in Myr.
            dust_emission_ism(synthesizer.dust.EmissionModel):
                The dust emission model for the ISM.
            dust_emission_birth(synthesizer.dust.EmissionModel):
                The dust emission model for the nebular.
            fesc_ly_alpha(float):
                The Lyman alpha escape fraction.
            label(str):
                The label for the total emission model. If `None` this will
                be set to "total".
            stellar_dust(bool):
                If `True`, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The BimodalPacmanEmission style models require a "
                "reprocessed grid."
            )

        # Create the incident models
        young_incident = IncidentEmission(
            grid=grid,
            label="young_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_incident = IncidentEmission(
            grid=grid,
            label="old_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        incident = StellarEmissionModel(
            label="incident",
            combine=(young_incident, old_incident),
            **kwargs,
        )

        # Create the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc=0.0,
            **kwargs,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc=0.0,
            **kwargs,
        )
        transmitted = StellarEmissionModel(
            label="transmitted",
            combine=(young_transmitted, old_transmitted),
            **kwargs,
        )

        # Create the nebular models
        young_nebular_line = NebularLineEmission(
            grid=grid,
            label="young_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        old_nebular_line = NebularLineEmission(
            grid=grid,
            label="old_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        young_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="young_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="old_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            nebular_line=young_nebular_line,
            nebular_continuum=young_nebular_continuum,
            **kwargs,
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            nebular_line=old_nebular_line,
            nebular_continuum=old_nebular_continuum,
            **kwargs,
        )
        nebular = StellarEmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
            **kwargs,
        )

        # Create the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            nebular=young_nebular,
            transmitted=young_transmitted,
            **kwargs,
        )
        old_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_reprocessed",
            nebular=old_nebular,
            transmitted=old_transmitted,
            **kwargs,
        )
        reprocessed = StellarEmissionModel(
            label="reprocessed",
            combine=(young_reprocessed, old_reprocessed),
            **kwargs,
        )

        # Create the intrinsic models
        young_intrinsic = StellarEmissionModel(
            label="young_intrinsic",
            combine=(young_reprocessed,),
            **kwargs,
        )
        old_intrinsic = StellarEmissionModel(
            label="old_intrinsic",
            combine=(old_reprocessed,),
            **kwargs,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
            **kwargs,
        )

        # Create the attenuated models
        young_attenuated_nebular = AttenuatedEmission(
            label="young_attenuated_nebular",
            tau_v=tau_v_birth,
            dust_curve=dust_curve_birth,
            apply_to=young_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=young_attenuated_nebular,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated = young_attenuated_ism
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=old_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        attenuated = StellarEmissionModel(
            label="attenuated",
            combine=(young_attenuated, old_attenuated),
            **kwargs,
        )

        # Create the emergent models (same as attenuated for no escape)
        young_emergent = young_attenuated
        old_emergent = old_attenuated
        emergent = attenuated

        # Create the dust emission models
        young_dust_emission_birth = DustEmission(
            label="young_dust_emission_birth",
            dust_emission_model=dust_emission_birth,
            dust_lum_intrinsic=young_intrinsic,
            dust_lum_attenuated=young_attenuated_nebular,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        young_dust_emission_ism = DustEmission(
            label="young_dust_emission_ism",
            dust_emission_model=dust_emission_ism,
            dust_lum_intrinsic=young_intrinsic,
            dust_lum_attenuated=young_attenuated_ism,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        young_dust_emission = EmissionModel(
            label="young_dust_emission",
            combine=(young_dust_emission_birth, young_dust_emission_ism),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        old_dust_emission = DustEmission(
            label="old_dust_emission",
            dust_emission_model=dust_emission_ism,
            dust_lum_intrinsic=old_intrinsic,
            dust_lum_attenuated=old_attenuated,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        dust_emission = EmissionModel(
            label="dust_emission",
            combine=(young_dust_emission, old_dust_emission),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )

        # Create the total models
        young_total = EmissionModel(
            label="young_total",
            combine=(young_dust_emission, young_emergent),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        old_total = EmissionModel(
            label="old_total",
            combine=(old_dust_emission, old_emergent),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )

        # Store all models as related
        related_models = (
            young_incident,
            old_incident,
            incident,
            young_transmitted,
            old_transmitted,
            transmitted,
            young_nebular,
            old_nebular,
            nebular,
            young_reprocessed,
            old_reprocessed,
            reprocessed,
            young_intrinsic,
            old_intrinsic,
            intrinsic,
            young_attenuated_nebular,
            young_attenuated_ism,
            young_attenuated,
            old_attenuated,
            attenuated,
            young_emergent,
            old_emergent,
            emergent,
            young_dust_emission_birth,
            young_dust_emission_ism,
            young_dust_emission,
            old_dust_emission,
            dust_emission,
            young_total,
            old_total,
        )

        # Finally make the TotalEmission model
        EmissionModel.__init__(
            self,
            grid=grid,
            label="total" if label is None else label,
            combine=(young_total, old_total),
            related_models=related_models,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )


class BimodalPacmanEmissionWithEscapedNoDust(StellarEmissionModel):
    """A class defining the BimodalPacman model with fesc but no dust emission.

    This model defines both intrinsic and attenuated stellar emission without
    dust emission, split into young and old populations. If a lyman alpha
    escape fraction is given, a more sophisticated nebular emission model is
    used, including line and nebular continuum emission with the amount of
    lyman alpha emission scaled by the escape fraction.

    This model will produce:
        - young_incident: the stellar emission incident onto the ISM for the
            young population.
        - old_incident: the stellar emission incident onto the ISM for the old
            population.
        - incident: the stellar emission incident onto the ISM for the
            combined population.
        - young_transmitted: the stellar emission transmitted through the ISM
            for the young population.
        - old_transmitted: the stellar emission transmitted through the ISM for
            the old population.
        - transmitted: the stellar emission transmitted through the ISM for the
            combined population.
        - young_escaped: the incident emission that completely escapes the ISM
            for the young population.
        - old_escaped: the incident emission that completely escapes the ISM
            for the old population.
        - escaped: the incident emission that completely escapes the ISM for
            the combined population.
        - young_nebular: the stellar emission from nebulae for the young
            population.
        - old_nebular: the stellar emission from nebulae for the old
            population.
        - nebular: the stellar emission from nebulae for the combined
            population.
        - young_reprocessed: the stellar emission reprocessed by the ISM for
            the young population.
        - old_reprocessed: the stellar emission reprocessed by the ISM for the
            old population.
        - reprocessed: the stellar emission reprocessed by the ISM for the
            combined population.
        - young_intrinsic: the intrinsic emission for the young population.
        - old_intrinsic: the intrinsic emission for the old population.
        - intrinsic: the intrinsic emission for the combined population.
        - young_attenuated_nebular: the nebular emission attenuated by dust
            for the young population.
        - young_attenuated_ism: the intrinsic emission attenuated by dust for
            the young population.
        - young_attenuated: the intrinsic emission attenuated by dust for the
            young population.
        - old_attenuated: the intrinsic emission attenuated by dust for the old
            population.
        - attenuated: the intrinsic emission attenuated by dust for the
            combined population.
        - young_emergent: the emission which emerges from the young stellar
            population, including any escaped emission.
        - old_emergent: the emission which emerges from the old stellar
            population, including any escaped emission.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission.
    """

    def __init__(
        self,
        grid,
        tau_v_ism="tau_v_ism",
        tau_v_birth="tau_v_birth",
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        age_pivot=7 * dimensionless,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        **kwargs,
    ):
        """Initialize the BimodalPacmanEmissionWithEscapeNoDust model.

        Args:
            grid(synthesizer.grid.Grid):
                The grid object.
            tau_v_ism(float):
                The V-band optical depth for the ISM.
            tau_v_birth(float):
                The V-band optical depth for the nebular.
            dust_curve_ism(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the ISM. Defaults to `Calzetti2000`
                with default parameters.
            dust_curve_birth(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the nebular. Defaults to
                `Calzetti2000` with default parameters.
            age_pivot(unyt.unyt_quantity):
                The age pivot between young and old populations,
                expressed in terms of log10(age) in Myr.
            fesc(float):
                The escape fraction.
            fesc_ly_alpha(float):
                The Lyman alpha escape fraction.
            label(str):
                The label for the total emission model. If `None` this will
                be set to "emergent".
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The BimodalPacmanEmission style models require a "
                "reprocessed grid."
            )

        # Ensure we have a non-zero escape fraction
        if fesc == 0.0 or fesc is None:
            raise ValueError(
                "The BimodalPacmanEmissionWithEscapeNoDust model requires a "
                "non-zero escape fraction."
            )

        # Create the incident models
        young_incident = IncidentEmission(
            grid=grid,
            label="young_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_incident = IncidentEmission(
            grid=grid,
            label="old_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        incident = StellarEmissionModel(
            label="incident",
            combine=(young_incident, old_incident),
            **kwargs,
        )

        # Create the transmitted models with escape fraction
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc=fesc,
            incident=young_incident,
            escaped_label="young_escaped",
            **kwargs,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc=fesc,
            incident=old_incident,
            escaped_label="old_escaped",
            **kwargs,
        )
        transmitted = StellarEmissionModel(
            label="transmitted",
            combine=(young_transmitted, old_transmitted),
            **kwargs,
        )

        # Extract escaped models
        young_escaped = young_transmitted["young_escaped"]
        old_escaped = old_transmitted["old_escaped"]
        escaped = StellarEmissionModel(
            label="escaped",
            combine=(young_escaped, old_escaped),
            **kwargs,
        )

        # Create the nebular models
        young_nebular_line = NebularLineEmission(
            grid=grid,
            label="young_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        old_nebular_line = NebularLineEmission(
            grid=grid,
            label="old_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        young_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="young_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="old_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            nebular_line=young_nebular_line,
            nebular_continuum=young_nebular_continuum,
            **kwargs,
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            nebular_line=old_nebular_line,
            nebular_continuum=old_nebular_continuum,
            **kwargs,
        )
        nebular = StellarEmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
            **kwargs,
        )

        # Create the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            nebular=young_nebular,
            transmitted=young_transmitted,
            **kwargs,
        )
        old_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_reprocessed",
            nebular=old_nebular,
            transmitted=old_transmitted,
            **kwargs,
        )
        reprocessed = StellarEmissionModel(
            label="reprocessed",
            combine=(young_reprocessed, old_reprocessed),
            **kwargs,
        )

        # Create the intrinsic models (reprocessed + escaped)
        young_intrinsic = StellarEmissionModel(
            label="young_intrinsic",
            combine=(young_reprocessed, young_escaped),
            **kwargs,
        )
        old_intrinsic = StellarEmissionModel(
            label="old_intrinsic",
            combine=(old_reprocessed, old_escaped),
            **kwargs,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
            **kwargs,
        )

        # Create the attenuated models
        young_attenuated_nebular = AttenuatedEmission(
            label="young_attenuated_nebular",
            tau_v=tau_v_birth,
            dust_curve=dust_curve_birth,
            apply_to=young_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=young_attenuated_nebular,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated = young_attenuated_ism
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=old_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        attenuated = StellarEmissionModel(
            label="attenuated",
            combine=(young_attenuated, old_attenuated),
            **kwargs,
        )

        # Create the emergent models (attenuated + escaped)
        young_emergent = StellarEmissionModel(
            label="young_emergent",
            combine=(young_attenuated, young_escaped),
            **kwargs,
        )
        old_emergent = StellarEmissionModel(
            label="old_emergent",
            combine=(old_attenuated, old_escaped),
            **kwargs,
        )
        emergent = StellarEmissionModel(
            label="emergent",
            combine=(young_emergent, old_emergent),
            **kwargs,
        )

        # Store all models as related
        related_models = (
            young_incident,
            old_incident,
            incident,
            young_transmitted,
            old_transmitted,
            transmitted,
            young_escaped,
            old_escaped,
            escaped,
            young_nebular,
            old_nebular,
            nebular,
            young_reprocessed,
            old_reprocessed,
            reprocessed,
            young_intrinsic,
            old_intrinsic,
            intrinsic,
            young_attenuated_nebular,
            young_attenuated_ism,
            young_attenuated,
            old_attenuated,
            attenuated,
            young_emergent,
            old_emergent,
            emergent,
        )

        # Make the final emergent emission model
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label="emergent" if label is None else label,
            combine=(young_emergent, old_emergent),
            related_models=related_models,
            **kwargs,
        )


class BimodalPacmanEmissionWithEscapedWithDust(StellarEmissionModel):
    """A class defining the Bimodal Pacman model with fesc and dust emission.

    This model defines both intrinsic and attenuated stellar emission with
    dust emission, split into young and old populations. If a lyman alpha
    escape fraction is given, a more sophisticated nebular emission model is
    used, including line and nebular continuum emission with the amount of
    lyman alpha emission scaled by the escape fraction.

    This model will produce:
        - young_incident: the stellar emission incident onto the ISM for the
            young population.
        - old_incident: the stellar emission incident onto the ISM for the old
            population.
        - incident: the stellar emission incident onto the ISM for the
            combined population.
        - young_transmitted: the stellar emission transmitted through the ISM
            for the young population.
        - old_transmitted: the stellar emission transmitted through the ISM for
            the old population.
        - transmitted: the stellar emission transmitted through the ISM for the
            combined population.
        - young_escaped: the incident emission that completely escapes the ISM
            for the young population.
        - old_escaped: the incident emission that completely escapes the ISM
            for the old population.
        - escaped: the incident emission that completely escapes the ISM for
            the combined population.
        - young_nebular: the stellar emission from nebulae for the young
            population.
        - old_nebular: the stellar emission from nebulae for the old
            population.
        - nebular: the stellar emission from nebulae for the combined
            population.
        - young_reprocessed: the stellar emission reprocessed by the ISM for
            the young population.
        - old_reprocessed: the stellar emission reprocessed by the ISM for the
            old population.
        - reprocessed: the stellar emission reprocessed by the ISM for the
            combined population.
        - young_intrinsic: the intrinsic emission for the young population.
        - old_intrinsic: the intrinsic emission for the old population.
        - intrinsic: the intrinsic emission for the combined population.
        - young_attenuated_nebular: the nebular emission attenuated by dust
            for the young population.
        - young_attenuated_ism: the intrinsic emission attenuated by dust for
            the young population.
        - young_attenuated: the intrinsic emission attenuated by dust for the
            young population.
        - old_attenuated: the intrinsic emission attenuated by dust for the old
            population.
        - attenuated: the intrinsic emission attenuated by dust for the
            combined population.
        - young_emergent: the emission which emerges from the young stellar
            population, including any escaped emission.
        - old_emergent: the emission which emerges from the old stellar
            population, including any escaped emission.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission.
        - young_dust_emission_birth: the emission from dust for the young
            population.
        - young_dust_emission_ism: the emission from dust for the young
            population.
        - young_dust_emission: the emission from dust for the young population.
        - old_dust_emission: the emission from dust for the old population.
        - dust_emission: the emission from dust for the combined population.
        - young_total: the final total combined emission for the young
            population.
        - old_total: the final total combined emission for the old population.
        - total: the final total combined emission for the combined population.
    """

    def __init__(
        self,
        grid,
        tau_v_ism="tau_v_ism",
        tau_v_birth="tau_v_birth",
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        age_pivot=7 * dimensionless,
        dust_emission_ism=None,
        dust_emission_birth=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        stellar_dust=True,
        **kwargs,
    ):
        """Initialize the BimodalPacmanEmissionWithEscapeWithDust model.

        Args:
            grid(synthesizer.grid.Grid):
                The grid object.
            tau_v_ism(float):
                The V-band optical depth for the ISM.
            tau_v_birth(float):
                The V-band optical depth for the nebular.
            dust_curve_ism(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the ISM. Defaults to `Calzetti2000`
                with default parameters.
            dust_curve_birth(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the nebular. Defaults to
                `Calzetti2000` with default parameters.
            age_pivot(unyt.unyt_quantity):
                The age pivot between young and old populations,
                expressed in terms of log10(age) in Myr.
            dust_emission_ism(synthesizer.dust.EmissionModel):
                The dust emission model for the ISM.
            dust_emission_birth(synthesizer.dust.EmissionModel):
                The dust emission model for the nebular.
            fesc(float):
                The escape fraction.
            fesc_ly_alpha(float):
                The Lyman alpha escape fraction.
            label(str):
                The label for the total emission model. If `None` this will
                be set to "total".
            stellar_dust(bool):
                If `True`, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Ensure the grid has been processed
        if not grid.reprocessed:
            raise ValueError(
                "The BimodalPacmanEmission style models require a"
                " reprocessed grid."
            )

        # Ensure we have a non-zero escape fraction
        if fesc == 0.0 or fesc is None:
            raise ValueError(
                "The BimodalPacmanEmissionWithEscapeWithDust model requires a "
                "non-zero escape fraction."
            )

        # Create the incident models
        young_incident = IncidentEmission(
            grid=grid,
            label="young_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_incident = IncidentEmission(
            grid=grid,
            label="old_incident",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        incident = StellarEmissionModel(
            label="incident",
            combine=(young_incident, old_incident),
            **kwargs,
        )

        # Create the transmitted models with escape fraction
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc=fesc,
            incident=young_incident,
            **kwargs,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc=fesc,
            incident=old_incident,
            **kwargs,
        )
        transmitted = StellarEmissionModel(
            label="transmitted",
            combine=(young_transmitted, old_transmitted),
            **kwargs,
        )

        # Extract escaped models
        young_escaped = young_transmitted["escaped"]
        old_escaped = old_transmitted["escaped"]
        escaped = StellarEmissionModel(
            label="escaped",
            combine=(young_escaped, old_escaped),
            **kwargs,
        )

        # Create the nebular models
        young_nebular_line = NebularLineEmission(
            grid=grid,
            label="young_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        old_nebular_line = NebularLineEmission(
            grid=grid,
            label="old_nebular_line",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        young_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="young_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_nebular_continuum = NebularContinuumEmission(
            grid=grid,
            label="old_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
            **kwargs,
        )
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            nebular_line=young_nebular_line,
            nebular_continuum=young_nebular_continuum,
            **kwargs,
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            nebular_line=old_nebular_line,
            nebular_continuum=old_nebular_continuum,
            **kwargs,
        )
        nebular = StellarEmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
            **kwargs,
        )

        # Create the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            nebular=young_nebular,
            transmitted=young_transmitted,
            **kwargs,
        )
        old_reprocessed = ReprocessedEmission(
            grid=grid,
            label="old_reprocessed",
            nebular=old_nebular,
            transmitted=old_transmitted,
            **kwargs,
        )
        reprocessed = StellarEmissionModel(
            label="reprocessed",
            combine=(young_reprocessed, old_reprocessed),
            **kwargs,
        )

        # Create the intrinsic models (reprocessed + escaped)
        young_intrinsic = StellarEmissionModel(
            label="young_intrinsic",
            combine=(young_reprocessed, young_escaped),
            **kwargs,
        )
        old_intrinsic = StellarEmissionModel(
            label="old_intrinsic",
            combine=(old_reprocessed, old_escaped),
            **kwargs,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
            **kwargs,
        )

        # Create the attenuated models
        young_attenuated_nebular = AttenuatedEmission(
            label="young_attenuated_nebular",
            tau_v=tau_v_birth,
            dust_curve=dust_curve_birth,
            apply_to=young_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=young_attenuated_nebular,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated = young_attenuated_ism
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_to=old_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        attenuated = StellarEmissionModel(
            label="attenuated",
            combine=(young_attenuated, old_attenuated),
            **kwargs,
        )

        # Create the emergent models (attenuated + escaped)
        young_emergent = EmergentEmission(
            grid=grid,
            label="young_emergent",
            attenuated=young_attenuated,
            escaped=young_escaped,
            **kwargs,
        )
        old_emergent = EmergentEmission(
            grid=grid,
            label="old_emergent",
            attenuated=old_attenuated,
            escaped=old_escaped,
            **kwargs,
        )
        emergent = StellarEmissionModel(
            label="emergent",
            combine=(young_emergent, old_emergent),
            **kwargs,
        )

        # Create the dust emission models
        young_dust_emission_birth = DustEmission(
            label="young_dust_emission_birth",
            dust_emission_model=dust_emission_birth,
            dust_lum_intrinsic=young_intrinsic,
            dust_lum_attenuated=young_attenuated_nebular,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        young_dust_emission_ism = DustEmission(
            label="young_dust_emission_ism",
            dust_emission_model=dust_emission_ism,
            dust_lum_intrinsic=young_intrinsic,
            dust_lum_attenuated=young_attenuated_ism,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        young_dust_emission = EmissionModel(
            label="young_dust_emission",
            combine=(young_dust_emission_birth, young_dust_emission_ism),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        old_dust_emission = DustEmission(
            label="old_dust_emission",
            dust_emission_model=dust_emission_ism,
            dust_lum_intrinsic=old_intrinsic,
            dust_lum_attenuated=old_attenuated,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        dust_emission = EmissionModel(
            label="dust_emission",
            combine=(young_dust_emission, old_dust_emission),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )

        # Create the total models
        young_total = EmissionModel(
            label="young_total",
            combine=(young_dust_emission, young_emergent),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )
        old_total = EmissionModel(
            label="old_total",
            combine=(old_dust_emission, old_emergent),
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )

        # Store all models as related
        related_models = (
            young_incident,
            old_incident,
            incident,
            young_transmitted,
            old_transmitted,
            transmitted,
            young_escaped,
            old_escaped,
            escaped,
            young_nebular,
            old_nebular,
            nebular,
            young_reprocessed,
            old_reprocessed,
            reprocessed,
            young_intrinsic,
            old_intrinsic,
            intrinsic,
            young_attenuated_nebular,
            young_attenuated_ism,
            young_attenuated,
            old_attenuated,
            attenuated,
            young_emergent,
            old_emergent,
            emergent,
            young_dust_emission_birth,
            young_dust_emission_ism,
            young_dust_emission,
            old_dust_emission,
            dust_emission,
            young_total,
            old_total,
        )

        # Finally make the TotalEmission model
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label="total" if label is None else label,
            combine=(young_total, old_total),
            related_models=related_models,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )


class BimodalPacmanEmission:
    """A class defining the Bimodal Pacman model.

    This model defines both intrinsic and attenuated stellar emission with or
    without dust emission. It also includes the option to include escaped
    emission for a given escape fraction. If a lyman alpha escape fraction is
    given, a more sophisticated nebular emission model is used, including line
    and nebular continuum emission.

    All spectra produced have a young, old and combined component. The split
    between young and old is by default 10 ^ 7 Myr but can be changed with the
    age_pivot argument.

    This model will always produce:
        - young_incident: the stellar emission incident onto the ISM for the
            young population.
        - old_incident: the stellar emission incident onto the ISM for the old
            population.
        - incident: the stellar emission incident onto the ISM for the
            combined population.
        - young_attenuated_nebular: the nebular emission attenuated by dust
            for the young population.
        - young_attenuated_ism: the intrinsic emission attenuated by dust for
            the young population.
        - young_attenuated: the intrinsic emission attenuated by dust for the
            young population.
        - old_attenuated: the intrinsic emission attenuated by dust for the old
            population.
        - young_transmitted: the stellar emission transmitted through the ISM
            for the young population.
        - old_transmitted: the stellar emission transmitted through the ISM for
            the old population.
        - transmitted: the stellar emission transmitted through the ISM for the
            combined population.
        - young_nebular: the stellar emission from nebulae for the young
            population.
        - old_nebular: the stellar emission from nebulae for the old
            population.
        - nebular: the stellar emission from nebulae for the combined
            population.
        - young_reprocessed: the stellar emission reprocessed by the ISM for
            the young population.
        - old_reprocessed: the stellar emission reprocessed by the ISM for the
            old population.
        - reprocessed: the stellar emission reprocessed by the ISM for the
            combined population.

    If an escape fraction is given, the following additional models will be
    produced:
        - young_intrinsic: the intrinsic emission for the young population.
        - old_intrinsic: the intrinsic emission for the old population.
        - intrinsic: the intrinsic emission for the combined population.
        - young_escaped: the incident emission that completely escapes the ISM
            for the young population.
        - old_escaped: the incident emission that completely escapes the ISM
            for the old population.
        - escaped: the incident emission that completely escapes the ISM for
            the combined population.
        - young_emergent: the emission which emerges from the stellar
            population, including any escaped emission for the young
            population.
        - old_emergent: the emission which emerges from the stellar population,
            including any escaped emission for the old population.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission for the combined population.

    if dust_emission is not None:
        - young_dust_emission_birth: the emission from dust for the young
            population.
        - young_dust_emission_ism: the emission from dust for the young
            population.
        - young_dust_emission: the emission from dust for the young population.
        - old_dust_emission: the emission from dust for the old population.
        - dust_emission: the emission from dust for the combined population.
        - young_total: the final total combined emission for the young
            population.
        - old_total: the final total combined emission for the old population.
        - total: the final total combined emission for the combined population.
    """

    def __new__(
        cls,
        grid,
        tau_v_ism="tau_v_ism",
        tau_v_birth="tau_v_birth",
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        age_pivot=7 * dimensionless,
        dust_emission_ism=None,
        dust_emission_birth=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label=None,
        stellar_dust=True,
        **kwargs,
    ):
        """Get a BimodalPacmanEmission model.

        Args:
            grid(synthesizer.grid.Grid):
                The grid object.
            tau_v_ism(float):
                The V-band optical depth for the ISM.
            tau_v_birth(float):
                The V-band optical depth for the nebular.
            dust_curve_ism(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the ISM. Defaults to `Calzetti2000`
                with default parameters.
            dust_curve_birth(synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the nebular. Defaults to
                `Calzetti2000` with default parameters.
            age_pivot(unyt.unyt_quantity):
                The age pivot between young and old populations,
                expressed in terms of log10(age) in Myr.
            dust_emission_ism(synthesizer.dust.EmissionModel):
                The dust emission model for the ISM.
            dust_emission_birth(synthesizer.dust.EmissionModel):
                The dust emission model for the nebular.
            fesc(float):
                The escape fraction.
            fesc_ly_alpha(float):
                The Lyman alpha escape fraction.
            label(str):
                The label for the total emission model. If `None` this will
                be set to "total" or "emergent" if dust_emission is `None`.
            stellar_dust(bool):
                If `True`, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            **kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Are we ignoring the escape fraction?
        if fesc == 0.0 or fesc is None:
            # Do we have dust emission models?
            if dust_emission_ism is None or dust_emission_birth is None:
                # No dust emission, no escape fraction, so we can use the
                # BimodalPacmanEmissionNoEscapeNoDust model
                return BimodalPacmanEmissionNoEscapedNoDust(
                    grid=grid,
                    tau_v_ism=tau_v_ism,
                    tau_v_birth=tau_v_birth,
                    dust_curve_ism=dust_curve_ism,
                    dust_curve_birth=dust_curve_birth,
                    age_pivot=age_pivot,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    **kwargs,
                )
            else:
                # We have dust emission, no escape fraction, so we can use the
                # BimodalPacmanEmissionNoEscapeWithDust model
                return BimodalPacmanEmissionNoEscapedWithDust(
                    grid=grid,
                    tau_v_ism=tau_v_ism,
                    tau_v_birth=tau_v_birth,
                    dust_curve_ism=dust_curve_ism,
                    dust_curve_birth=dust_curve_birth,
                    age_pivot=age_pivot,
                    dust_emission_ism=dust_emission_ism,
                    dust_emission_birth=dust_emission_birth,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    stellar_dust=stellar_dust,
                    **kwargs,
                )
        # Ok, we have an escape fraction
        else:
            # Do we have dust emission models?
            if dust_emission_ism is None or dust_emission_birth is None:
                # No dust emission, so we can use the
                # BimodalPacmanEmissionWithEscapeNoDust model
                return BimodalPacmanEmissionWithEscapedNoDust(
                    grid=grid,
                    tau_v_ism=tau_v_ism,
                    tau_v_birth=tau_v_birth,
                    dust_curve_ism=dust_curve_ism,
                    dust_curve_birth=dust_curve_birth,
                    age_pivot=age_pivot,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    **kwargs,
                )
            else:
                # We have dust emission, so we can use the
                # BimodalPacmanEmissionWithEscapeWithDust model
                return BimodalPacmanEmissionWithEscapedWithDust(
                    grid=grid,
                    tau_v_ism=tau_v_ism,
                    tau_v_birth=tau_v_birth,
                    dust_curve_ism=dust_curve_ism,
                    dust_curve_birth=dust_curve_birth,
                    age_pivot=age_pivot,
                    dust_emission_ism=dust_emission_ism,
                    dust_emission_birth=dust_emission_birth,
                    fesc=fesc,
                    fesc_ly_alpha=fesc_ly_alpha,
                    label=label,
                    stellar_dust=stellar_dust,
                    **kwargs,
                )


class CharlotFall2000(BimodalPacmanEmission):
    """The Charlot & Fall(2000) emission model.

    This emission model is based on the Charlot & Fall(2000) model, which
    describes the emission from a galaxy as a combination of emission from a
    young stellar population and an old stellar population. The dust
    attenuation for each population can be different, and dust emission can be
    optionally included.

    This model is a simplified version of the BimodalPacmanEmission model, so
    in reality is just a wrapper around that model. The only difference is that
    there is no option to specify an escape fraction.

    Attributes:
        grid(synthesizer.grid.Grid): The grid object.
        tau_v_ism(float): The V-band optical depth for the ISM.
        tau_v_birth(float): The V-band optical depth for the nebular.
        dust_curve_ism(AttenuationLaw):
            The dust curve for the ISM.
        dust_curve_birth(AttenuationLaw): The dust curve for the
            nebular.
        age_pivot(unyt.unyt_quantity): The age pivot between young and old
            populations, expressed in terms of log10(age) in Myr.
        dust_emission_ism(synthesizer.dust.EmissionModel): The dust
            emission model for the ISM.
        dust_emission_birth(synthesizer.dust.EmissionModel): The dust
            emission model for the nebular.
    """

    def __init__(
        self,
        grid,
        tau_v_ism,
        tau_v_birth,
        age_pivot=7 * dimensionless,
        dust_curve_ism=Calzetti2000(),
        dust_curve_birth=Calzetti2000(),
        dust_emission_ism=None,
        dust_emission_birth=None,
        label=None,
        stellar_dust=True,
        **kwargs,
    ):
        """Initialize the PacmanEmission model.

        Args:
            grid(synthesizer.grid.Grid):
                The grid object.
            tau_v_ism(float):
                The V-band optical depth for the ISM.
            tau_v_birth(float):
                The V-band optical depth for the nebular.
            age_pivot(unyt.unyt_quantity):
                The age pivot between young and old populations, expressed
                in terms of log10(age) in Myr. Defaults to 10 ^ 7 Myr.
            dust_curve_ism(AttenuationLaw):
                The dust curve for the ISM. Defaults to `Calzetti2000`
                with fiducial parameters.
            dust_curve_birth(AttenuationLaw):
                The dust curve for the nebular. Defaults to `Calzetti2000`
                with fiducial parameters.
            dust_emission_ism(synthesizer.dust.EmissionModel):
                The dust emission model for the ISM.
            dust_emission_birth(synthesizer.dust.EmissionModel):
                The dust emission model for the nebular.
            label(str):
                The label for the total emission model. If None this will
                be set to "total" or "emergent" if dust_emission is `None`.
            stellar_dust(bool):
                If `True`, the dust emission will be treated as stellar
                emission, otherwise it will be treated as galaxy emission.
            kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Call the parent constructor to intialise the model
        BimodalPacmanEmission.__init__(
            self,
            grid,
            tau_v_ism=tau_v_ism,
            tau_v_birth=tau_v_birth,
            dust_curve_ism=dust_curve_ism,
            dust_curve_birth=dust_curve_birth,
            age_pivot=age_pivot,
            dust_emission_ism=dust_emission_ism,
            dust_emission_birth=dust_emission_birth,
            label=label,
            stellar_dust=stellar_dust,
            **kwargs,
        )


class ScreenEmission(PacmanEmission):
    """The ScreenEmission model.

    This emission model is a simple dust screen model, where the dust is
    assumed to be in a screen in front of the stars. The dust curve and
    emission model can be specified, but the escape fraction is always zero.

    This model is a simplified version of the PacmanEmission model, so in
    reality is just a wrapper around that model. The only difference is that
    fesc and fesc_ly_alpha are zero by definition.

    Attributes:
        grid(synthesizer.grid.Grid): The grid object.
        dust_curve(AttenuationLaw): The dust curve.
        dust_emission(synthesizer.dust.EmissionModel): The dust emission
            model.
    """

    def __init__(
        self,
        grid,
        tau_v,
        dust_curve=Calzetti2000(),
        dust_emission=None,
        label=None,
        fesc=None,
        fesc_ly_alpha=None,
        **kwargs,
    ):
        """Initialize the ScreenEmission model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v (float):
                The V-band optical depth for the dust screen.
            dust_curve (AttenuationLaw):
                The assumed dust curve. Defaults to `Calzetti2000` with
                default parameters.
            dust_emission (synthesizer.dust.EmissionModel):
                The dust emission model.
            label (str):
                The label for the total emission model. If None this will
                be set to "total" or "emergent" if dust_emission is None.
            fesc (float):
                The escape fraction. This is always zero for this model.
            fesc_ly_alpha (float):
                The Lyman alpha escape fraction. This is always zero for
                this model.
            kwargs:
                Additional keyword arguments to pass to the models.
        """
        # Call the parent constructor to intialise the model
        PacmanEmission.__init__(
            self,
            grid=grid,
            tau_v=tau_v,
            dust_curve=dust_curve,
            dust_emission=dust_emission,
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            label=label,
            **kwargs,
        )
