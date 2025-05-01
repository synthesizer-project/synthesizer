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
    IncidentEmission,
    NebularContinuumEmission,
    NebularEmission,
    NebularLineEmission,
    TransmittedEmission,
)
from synthesizer.emission_models.transformers import (
    EscapedFraction,
    ProcessedFraction,
)


class PacmanEmission(StellarEmissionModel):
    """A class defining the Pacman model.

    This model defines both intrinsic and attenuated steller emission with or
    without dust emission. It also includes the option to include escaped
    emission for a given escape fraction. If a lyman alpha escape fraction is
    given, a more sophisticated nebular emission model is used, including line
    and nebuluar continuum emission.

    This model will always produce:
        - incident: the stellar emission incident onto the ISM.
        - intrinsic: the intrinsic emission (when grid.reprocessed is False
            this is the same as the incident emission).
        - attenuated: the intrinsic emission attenuated by dust.
        - escaped: the incident emission that completely escapes the ISM.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission.

    if grid.reprocessed is True:
        - transmitted: the stellar emission transmitted through the ISM.
        - nebular: the stellar emisison from nebulae.
        - reprocessed: the stellar emission reprocessed by the ISM.

    if dust_emission is not None:
        - dust_emission: the emission from dust.
        - total: the final total combined emission.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object
        dust_curve (AttenuationLaw): The dust curve
        dust_emission (synthesizer.dust.EmissionModel): The dust emission
        fesc (float): The escape fraction
        fesc_ly_alpha (float): The Lyman alpha escape fraction
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
        """Initialize the PacmanEmission model.

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
        # Attach the grid
        self._grid = grid

        # Attach the dust properties
        self._tau_v = tau_v
        self._dust_curve = dust_curve

        # Attach the dust emission properties
        self._dust_emission_model = dust_emission

        # Attach the escape fraction properties
        self._fesc = fesc
        self._fesc_ly_alpha = fesc_ly_alpha

        # Are we using a grid with reprocessing?
        self.grid_reprocessed = grid.reprocessed

        # Make the child emission models
        self.incident = self._make_incident(**kwargs)
        self.transmitted = self._make_transmitted(**kwargs)
        self.escaped = self.transmitted["escaped"]
        self.nebular = self._make_nebular(**kwargs)
        self.reprocessed = self._make_reprocessed(**kwargs)
        if not self.grid_reprocessed:
            self.intrinsic = self._make_intrinsic_no_reprocessing(**kwargs)
        else:
            self.intrinsic = self._make_intrinsic_reprocessed(**kwargs)
        self.attenuated = self._make_attenuated(**kwargs)
        if self._dust_emission_model is not None:
            self.emergent = self._make_emergent(**kwargs)
            self.dust_emission = self._make_dust_emission(
                stellar_dust, **kwargs
            )

        # Finally make the TotalEmission model, this is
        # dust_emission + emergent if dust_emission is not None, otherwise it
        # is just emergent
        self._make_total(label, stellar_dust, **kwargs)

    def _make_incident(self, **kwargs):
        """Make the incident emission model.

        This will ignore any escape fraction given the stellar emission
        incident onto the nebular, ism and dust components.

        Returns:
            StellarEmissionModel:
                - incident
        """
        return IncidentEmission(grid=self._grid, label="incident", **kwargs)

    def _make_transmitted(self, **kwargs):
        """Make the transmitted emission model.

        If the grid has not been reprocessed, this will return None for all
        components because the transmitted spectra won't exist.

        Returns:
            StellarEmissionModel:
                - transmitted
        """
        # No spectra if grid hasn't been reprocessed
        if not self.grid_reprocessed:
            return None

        return TransmittedEmission(
            grid=self._grid,
            label="transmitted",
            fesc=self._fesc,
            **kwargs,
        )

    def _make_nebular(self, **kwargs):
        # No spectra if grid hasn't been reprocessed
        if not self.grid_reprocessed:
            return None

        return NebularEmission(
            grid=self._grid,
            label="nebular",
            fesc_ly_alpha=self._fesc_ly_alpha,
            **kwargs,
        )

    def _make_reprocessed(self, **kwargs):
        # No spectra if grid hasn't been reprocessed
        if not self.grid_reprocessed:
            return None

        return StellarEmissionModel(
            label="reprocessed",
            combine=(self.transmitted, self.nebular),
            **kwargs,
        )

    def _make_intrinsic_no_reprocessing(self, **kwargs):
        """Make the intrinsic emission model for an un-reprocessed grid.

        This will produce the exact same emission as the incident emission but
        unlikely the incident emission, the intrinsic emission will be
        take into account an escape fraction.

        Returns:
            StellarEmissionModel:
                - intrinsic
        """
        return IncidentEmission(
            grid=self._grid,
            label="intrinsic",
            **kwargs,
        )

    def _make_intrinsic_reprocessed(self, **kwargs):
        # If fesc is zero then the intrinsic emission is the same as the
        # reprocessed emission
        if self._fesc == 0.0:
            return StellarEmissionModel(
                label="intrinsic",
                combine=(self.nebular, self.transmitted),
                **kwargs,
            )

        # Otherwise, intrinsic = reprocessed + escaped
        return StellarEmissionModel(
            label="intrinsic",
            combine=(self.reprocessed, self.escaped),
            **kwargs,
        )

    def _make_attenuated(self, **kwargs):
        return AttenuatedEmission(
            label="attenuated",
            dust_curve=self._dust_curve,
            apply_to=self.reprocessed,
            emitter="stellar",
            tau_v=self._tau_v,
            **kwargs,
        )

    def _make_emergent(self, **kwargs):
        # If fesc is zero the emergent spectra is just the attenuated spectra
        return StellarEmissionModel(
            label="emergent",
            combine=(self.attenuated, self.escaped),
            **kwargs,
        )

    def _make_dust_emission(self, stellar_dust, **kwargs):
        return DustEmission(
            label="dust_emission",
            dust_emission_model=self._dust_emission_model,
            dust_lum_intrinsic=self.intrinsic,
            dust_lum_attenuated=self.attenuated,
            emitter="galaxy" if not stellar_dust else "stellar",
            **kwargs,
        )

    def _make_total(self, label, stellar_dust, **kwargs):
        if self._dust_emission_model is not None:
            # Define the related models
            related_models = [
                self.incident,
                self.transmitted,
                self.escaped,
                self.nebular,
                self.reprocessed,
                self.intrinsic,
                self.attenuated,
                self.emergent,
                self.dust_emission,
            ]

            # Remove any None models
            related_models = [
                m
                for m in related_models
                if m is not None and not isinstance(m, tuple)
            ]

            # Call the parent constructor with everything we've made
            EmissionModel.__init__(
                self,
                grid=self._grid,
                label="total" if label is None else label,
                combine=(self.dust_emission, self.emergent),
                related_models=related_models,
                emitter="galaxy" if not stellar_dust else "stellar",
                **kwargs,
            )
        else:
            # OK, total = emergent so we need to handle whether
            # emergent = attenuated + escaped or just attenuated
            # Define the related models
            related_models = [
                self.incident,
                self.transmitted,
                self.nebular,
                self.reprocessed,
                self.intrinsic,
                self.attenuated,
            ]

            # Remove any None models
            related_models = [m for m in related_models if m is not None]

            StellarEmissionModel.__init__(
                self,
                grid=self._grid,
                label="emergent" if label is None else label,
                dust_curve=self._dust_curve,
                apply_to=self.intrinsic,
                tau_v=self._tau_v,
                related_models=related_models,
                **kwargs,
            )


class BimodalPacmanEmission(StellarEmissionModel):
    """A class defining the Pacman model split into young and old populations.

    This model defines both intrinsic and attenuated steller emission with or
    without dust emission. It also includes the option to include escaped
    emission for a given escape fraction. If a lyman alpha escape fraction is
    given, a more sophisticated nebular emission model is used, including line
    and nebuluar continuum emission.

    All spectra produced have a young, old and combined component. The split
    between young and old is by default 10^7 Myr but can be changed with the
    age_pivot argument.

    This model will always produce:
        - young_incident: the stellar emission incident onto the ISM for the
            young population.
        - old_incident: the stellar emission incident onto the ISM for the old
            population.
        - incident: the stellar emission incident onto the ISM for the
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

    if grid.reprocessed is True:
        - young_transmitted: the stellar emission transmitted through the ISM
            for the young population.
        - old_transmitted: the stellar emission transmitted through the ISM for
            the old population.
        - transmitted: the stellar emission transmitted through the ISM for the
            combined population.
        - young_nebular: the stellar emisison from nebulae for the young
            population.
        - old_nebular: the stellar emisison from nebulae for the old
            population.
        - nebular: the stellar emisison from nebulae for the combined
            population.
        - young_reprocessed: the stellar emission reprocessed by the ISM for
            the young population.
        - old_reprocessed: the stellar emission reprocessed by the ISM for the
            old population.
        - reprocessed: the stellar emission reprocessed by the ISM for the
            combined population.
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

    Attributes:
        grid (synthesizer.grid.Grid): The grid object.
        tau_v_ism (float): The V-band optical depth for the ISM.
        tau_v_birth (float): The V-band optical depth for the nebular.
        dust_curve_ism (AttenuationLaw): The dust curve for the
            ISM.
        dust_curve_birth (AttenuationLaw): The dust curve for the
            nebular.
        age_pivot (unyt.unyt_quantity): The age pivot between young and old
            populations, expressed in terms of log10(age) in Myr.
        dust_emission_ism (synthesizer.dust.EmissionModel): The dust
            emission for the ISM.
        dust_emission_birth (synthesizer.dust.EmissionModel): The dust
            emission for the nebular.
        fesc (float): The escape fraction.
        fesc_ly_alpha (float): The Lyman alpha escape fraction.

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
        """Initialize the PacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v_ism (float):
                The V-band optical depth for the ISM.
            tau_v_birth (float):
                The V-band optical depth for the nebular.
            dust_curve_ism (synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the ISM. Defaults to `Calzetti2000`
                with default parameters.
            dust_curve_birth (synthesizer.Transformer.AttenuationLaw):
                The assumed dust curve for the nebular. Defaults to
                `Calzetti2000` with default parameters.
            age_pivot (unyt.unyt_quantity):
                The age pivot between young and old populations,
                expressed in terms of log10(age) in Myr.
            dust_emission_ism (synthesizer.dust.EmissionModel):
                The dust emission model for the ISM.
            dust_emission_birth (synthesizer.dust.EmissionModel):
                The dust emission model for the nebular.
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
        # Attach the grid
        self._grid = grid

        # Attach the dust properties
        self.tau_v_ism = tau_v_ism
        self.tau_v_birth = tau_v_birth
        self._dust_curve_ism = dust_curve_ism
        self._dust_curve_birth = dust_curve_birth

        # Attach the age pivot
        self.age_pivot = age_pivot

        # Attach the dust emission properties
        self.dust_emission_ism = dust_emission_ism
        self.dust_emission_birth = dust_emission_birth

        # Attach the escape fraction properties
        self._fesc = fesc
        self._fesc_ly_alpha = fesc_ly_alpha

        # Are we using a grid with reprocessing?
        self.grid_reprocessed = grid.reprocessed

        # Make the child emission models
        (
            self.young_incident,
            self.old_incident,
            self.incident,
        ) = self._make_incident(**kwargs)
        (
            self.young_transmitted,
            self.old_transmitted,
            self.transmitted,
            self.young_escaped,
            self.old_escaped,
            self.escaped,
        ) = self._make_transmitted(**kwargs)
        (
            self.young_nebular,
            self.old_nebular,
            self.nebular,
        ) = self._make_nebular(**kwargs)
        (
            self.young_reprocessed,
            self.old_reprocessed,
            self.reprocessed,
        ) = self._make_reprocessed(**kwargs)
        if not self.grid_reprocessed:
            (
                self.young_intrinsic,
                self.old_intrinsic,
                self.intrinsic,
            ) = self._make_intrinsic_no_reprocessing(**kwargs)
        else:
            (
                self.young_intrinsic,
                self.old_intrinsic,
                self.intrinsic,
            ) = self._make_intrinsic_reprocessed(**kwargs)
        (
            self.young_attenuated_nebular,
            self.young_attenuated_ism,
            self.young_attenuated,
            self.old_attenuated,
            self.attenuated,
        ) = self._make_attenuated(**kwargs)
        if (
            self.dust_emission_ism is not None
            and self.dust_emission_birth is not None
        ):
            (
                self.young_emergent,
                self.old_emergent,
                self.emergent,
            ) = self._make_emergent(**kwargs)
            (
                self.young_dust_emission_birth,
                self.young_dust_emission_ism,
                self.young_dust_emission,
                self.old_dust_emission,
                self.dust_emission,
            ) = self._make_dust_emission(stellar_dust, **kwargs)

        # Finally make the TotalEmission model, this is
        # dust_emission + emergent if dust_emission is not None, otherwise it
        # is just emergent
        self._make_total(label, stellar_dust, **kwargs)

    def _make_incident(self, **kwargs):
        """Make the incident emission model.

        This will ignore any escape fraction given the stellar emission
        incident onto the nebular, ism and dust components.

        Returns:
            StellarEmissionModel:
                - young_incident
                - old_incident
                - incident
        """
        young_incident = IncidentEmission(
            grid=self._grid,
            label="young_incident",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_incident = IncidentEmission(
            grid=self._grid,
            label="old_incident",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            **kwargs,
        )
        incident = StellarEmissionModel(
            label="incident",
            combine=(young_incident, old_incident),
            **kwargs,
        )

        return young_incident, old_incident, incident

    def _make_transmitted(self, **kwargs):
        """Make the transmitted emission model.

        If the grid has not been reprocessed, this will return None for all
        components because the transmitted spectra won't exist.

        This will also generate the escaped emission models.

        Returns:
            StellarEmissionModel:
                - young_transmitted
                - old_transmitted
                - transmitted
        """
        # No spectra if grid hasn't been reprocessed
        if not self.grid_reprocessed:
            return None, None, None

        full_young_transmitted = StellarEmissionModel(
            grid=self._grid,
            label="full_young_transmitted",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            extract="transmitted",
            **kwargs,
        )
        full_old_transmitted = StellarEmissionModel(
            grid=self._grid,
            label="full_old_transmitted",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            extract="transmitted",
            **kwargs,
        )

        young_transmitted = StellarEmissionModel(
            label="young_transmitted",
            apply_to=full_young_transmitted,
            transformer=ProcessedFraction(),
            fesc=self._fesc,
            **kwargs,
        )
        old_transmitted = StellarEmissionModel(
            label="old_transmitted",
            apply_to=full_old_transmitted,
            transformer=ProcessedFraction(),
            fesc=self._fesc,
            **kwargs,
        )

        young_escaped = StellarEmissionModel(
            label="young_escaped",
            apply_to=full_young_transmitted,
            transformer=EscapedFraction(),
            fesc=self._fesc,
            **kwargs,
        )
        old_escaped = StellarEmissionModel(
            label="old_escaped",
            apply_to=full_old_transmitted,
            transformer=EscapedFraction(),
            fesc=self._fesc,
            **kwargs,
        )

        transmitted = StellarEmissionModel(
            label="transmitted",
            combine=(young_transmitted, old_transmitted),
            **kwargs,
        )
        escaped = StellarEmissionModel(
            label="escaped",
            combine=(young_escaped, old_escaped),
            **kwargs,
        )

        return (
            young_transmitted,
            old_transmitted,
            transmitted,
            young_escaped,
            old_escaped,
            escaped,
        )

    def _make_nebular(self, **kwargs):
        # No spectra if grid hasn't been reprocessed
        if not self.grid_reprocessed:
            return None, None, None

        # Get the line emission
        young_neb_line = NebularLineEmission(
            grid=self._grid,
            label="young_linecont",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc_ly_alpha=self._fesc_ly_alpha,
            **kwargs,
        )
        old_neb_line = NebularLineEmission(
            grid=self._grid,
            label="old_linecont",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc_ly_alpha=self._fesc_ly_alpha,
            **kwargs,
        )

        # Get the nebular continuum emission
        young_neb_cont = NebularContinuumEmission(
            grid=self._grid,
            label="young_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            **kwargs,
        )
        old_neb_cont = NebularContinuumEmission(
            grid=self._grid,
            label="old_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            **kwargs,
        )

        young_nebular = NebularEmission(
            grid=self._grid,
            label="young_nebular",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc_ly_alpha=self._fesc_ly_alpha,
            nebular_line=young_neb_line,
            nebular_continuum=young_neb_cont,
            **kwargs,
        )
        old_nebular = NebularEmission(
            grid=self._grid,
            label="old_nebular",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
            fesc_ly_alpha=self._fesc_ly_alpha,
            nebular_line=old_neb_line,
            nebular_continuum=old_neb_cont,
            **kwargs,
        )
        nebular = StellarEmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
            **kwargs,
        )

        return young_nebular, old_nebular, nebular

    def _make_reprocessed(self, **kwargs):
        # No spectra if grid hasn't been reprocessed
        if not self.grid_reprocessed:
            return None, None, None

        young_reprocessed = StellarEmissionModel(
            label="young_reprocessed",
            combine=(self.young_transmitted, self.young_nebular),
            **kwargs,
        )
        old_reprocessed = StellarEmissionModel(
            label="old_reprocessed",
            combine=(self.old_transmitted, self.old_nebular),
            **kwargs,
        )
        reprocessed = StellarEmissionModel(
            label="reprocessed",
            combine=(young_reprocessed, old_reprocessed),
            **kwargs,
        )

        return young_reprocessed, old_reprocessed, reprocessed

    def _make_intrinsic_no_reprocessing(self, **kwargs):
        """Make the intrinsic emission model for an un-reprocessed grid.

        This will produce the exact same emission as the incident emission but
        unlike the incident emission, the intrinsic emission will take into
        account an escape fraction.

        Returns:
            StellarEmissionModel:
                - young_intrinsic
                - old_intrinsic
                - intrinsic
        """
        young_intrinsic = IncidentEmission(
            grid=self._grid,
            label="young_intrinsic",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc=self._fesc,
            **kwargs,
        )
        old_intrinsic = IncidentEmission(
            grid=self._grid,
            label="old_intrinsic",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
            **kwargs,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
            **kwargs,
        )

        return young_intrinsic, old_intrinsic, intrinsic

    def _make_intrinsic_reprocessed(self, **kwargs):
        """Make the intrinsic emission model for a reprocessed grid.

        Returns:
            StellarEmissionModel:
                - young_intrinsic
                - old_intrinsic
                - intrinsic
        """
        # If fesc is zero then the intrinsic emission is the same as the
        # reprocessed emission
        if self._fesc == 0.0:
            young_intrinsic = StellarEmissionModel(
                label="young_intrinsic",
                combine=(self.young_nebular, self.young_transmitted),
                **kwargs,
            )
            old_intrinsic = StellarEmissionModel(
                label="old_intrinsic",
                combine=(self.old_nebular, self.old_transmitted),
                **kwargs,
            )
            intrinsic = StellarEmissionModel(
                label="intrinsic",
                combine=(young_intrinsic, old_intrinsic),
                **kwargs,
            )
        else:
            # Otherwise, intrinsic = reprocessed + escaped
            young_intrinsic = StellarEmissionModel(
                label="young_intrinsic",
                combine=(self.young_reprocessed, self.young_escaped),
                **kwargs,
            )
            old_intrinsic = StellarEmissionModel(
                label="old_intrinsic",
                combine=(self.old_reprocessed, self.old_escaped),
                **kwargs,
            )
            intrinsic = StellarEmissionModel(
                label="intrinsic",
                combine=(young_intrinsic, old_intrinsic),
                **kwargs,
            )

        return young_intrinsic, old_intrinsic, intrinsic

    def _make_attenuated(self, **kwargs):
        """Make the attenuated emission model.

        Returns:
            StellarEmissionModel:
                - young_attenuated_nebular
                - young_attenuated_ism
                - young_attenuated
                - old_attenuated
                - attenuated
        """
        young_attenuated_nebular = AttenuatedEmission(
            label="young_attenuated_nebular",
            tau_v=self.tau_v_birth,
            dust_curve=self._dust_curve_birth,
            apply_to=self.young_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_to=self.young_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        young_attenuated = AttenuatedEmission(
            label="young_attenuated",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_to=young_attenuated_nebular,
            emitter="stellar",
            **kwargs,
        )
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_to=self.old_reprocessed,
            emitter="stellar",
            **kwargs,
        )
        attenuated = StellarEmissionModel(
            label="attenuated",
            combine=(young_attenuated, old_attenuated),
            **kwargs,
        )

        return (
            young_attenuated_nebular,
            young_attenuated_ism,
            young_attenuated,
            old_attenuated,
            attenuated,
        )

    def _make_emergent(self, **kwargs):
        # If fesc is zero the emergent spectra is just the attenuated spectra
        if self._fesc == 0.0:
            young_emergent = AttenuatedEmission(
                label="young_emergent",
                tau_v=self.tau_v_ism,
                dust_curve=self._dust_curve_ism,
                apply_to=self.young_attenuated_nebular,
                emitter="stellar",
                **kwargs,
            )
            old_emergent = AttenuatedEmission(
                label="old_emergent",
                tau_v=self.tau_v_ism,
                dust_curve=self._dust_curve_ism,
                apply_to=self.old_intrinsic,
                emitter="stellar",
                **kwargs,
            )
            emergent = StellarEmissionModel(
                label="emergent",
                combine=(young_emergent, old_emergent),
                **kwargs,
            )
        else:
            # Otherwise, emergent = attenuated + escaped
            young_emergent = StellarEmissionModel(
                label="young_emergent",
                combine=(self.young_attenuated, self.young_escaped),
                **kwargs,
            )
            old_emergent = StellarEmissionModel(
                label="old_emergent",
                combine=(self.old_attenuated, self.old_escaped),
                **kwargs,
            )
            emergent = StellarEmissionModel(
                label="emergent",
                combine=(young_emergent, old_emergent),
                **kwargs,
            )

        return young_emergent, old_emergent, emergent

    def _make_dust_emission(self, stellar_dust, **kwargs):
        young_dust_emission_birth = DustEmission(
            label="young_dust_emission_birth",
            dust_emission_model=self.dust_emission_birth,
            dust_lum_intrinsic=self.young_intrinsic,
            dust_lum_attenuated=self.young_attenuated_nebular,
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            emitter="stellar" if stellar_dust else "galaxy",
            **kwargs,
        )
        young_dust_emission_ism = DustEmission(
            label="young_dust_emission_ism",
            dust_emission_model=self.dust_emission_ism,
            dust_lum_intrinsic=self.young_intrinsic,
            dust_lum_attenuated=self.young_attenuated_ism,
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            emitter="stellar" if stellar_dust else "galaxy",
            **kwargs,
        )
        young_dust_emission = EmissionModel(
            label="young_dust_emission",
            combine=(young_dust_emission_birth, young_dust_emission_ism),
            emitter="stellar" if stellar_dust else "galaxy",
            **kwargs,
        )
        old_dust_emission = DustEmission(
            label="old_dust_emission",
            dust_emission_model=self.dust_emission_ism,
            dust_lum_intrinsic=self.old_intrinsic,
            dust_lum_attenuated=self.old_attenuated,
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            emitter="stellar" if stellar_dust else "galaxy",
            **kwargs,
        )
        dust_emission = EmissionModel(
            label="dust_emission",
            combine=(young_dust_emission, old_dust_emission),
            emitter="stellar" if stellar_dust else "galaxy",
            **kwargs,
        )

        return (
            young_dust_emission_birth,
            young_dust_emission_ism,
            young_dust_emission,
            old_dust_emission,
            dust_emission,
        )

    def _make_total(self, label, stellar_dust, **kwargs):
        if (
            self.dust_emission_ism is not None
            and self.dust_emission_birth is not None
        ):
            # Get the young and old total emission
            young_total = EmissionModel(
                label="young_total",
                combine=(self.young_dust_emission, self.young_emergent),
                emitter="stellar" if stellar_dust else "galaxy",
                **kwargs,
            )
            old_total = EmissionModel(
                label="old_total",
                combine=(self.old_dust_emission, self.old_emergent),
                emitter="stellar" if stellar_dust else "galaxy",
                **kwargs,
            )

            # Define the related models
            related_models = [
                self.young_incident,
                self.old_incident,
                self.incident,
                self.young_transmitted,
                self.old_transmitted,
                self.transmitted,
                self.young_escaped,
                self.old_escaped,
                self.escaped,
                self.young_nebular,
                self.old_nebular,
                self.nebular,
                self.young_reprocessed,
                self.old_reprocessed,
                self.reprocessed,
                self.young_intrinsic,
                self.old_intrinsic,
                self.intrinsic,
                self.young_attenuated_nebular,
                self.young_attenuated_ism,
                self.young_attenuated,
                self.old_attenuated,
                self.attenuated,
                self.young_emergent,
                self.old_emergent,
                self.emergent,
                self.young_dust_emission_birth,
                self.young_dust_emission_ism,
                self.young_dust_emission,
                self.old_dust_emission,
                self.dust_emission,
            ]

            # Remove any None models
            related_models = [m for m in related_models if m is not None]

            # Call the parent constructor with everything we've made
            EmissionModel.__init__(
                self,
                grid=self._grid,
                label="total" if label is None else label,
                combine=(young_total, old_total),
                related_models=related_models,
                emitter="stellar" if stellar_dust else "galaxy",
                **kwargs,
            )
        else:
            # OK, total = emergent so we need to handle whether
            # emergent = attenuated + escaped or just attenuated
            if self._fesc == 0.0:
                # Get the young and old emergent emission
                young_total = AttenuatedEmission(
                    label="young_emergent",
                    tau_v=self.tau_v_ism,
                    dust_curve=self._dust_curve_ism,
                    apply_to=self.young_intrinsic,
                    emitter="stellar",
                    **kwargs,
                )
                old_total = AttenuatedEmission(
                    label="old_emergent",
                    tau_v=self.tau_v_ism,
                    dust_curve=self._dust_curve_ism,
                    apply_to=self.old_intrinsic,
                    emitter="stellar",
                    **kwargs,
                )

                # Define the related models
                related_models = [
                    self.young_incident,
                    self.old_incident,
                    self.incident,
                    self.young_transmitted,
                    self.old_transmitted,
                    self.transmitted,
                    self.young_nebular,
                    self.old_nebular,
                    self.nebular,
                    self.young_reprocessed,
                    self.old_reprocessed,
                    self.reprocessed,
                    self.young_intrinsic,
                    self.old_intrinsic,
                    self.intrinsic,
                    self.young_attenuated_nebular,
                    self.young_attenuated_ism,
                    self.young_attenuated,
                    self.old_attenuated,
                    self.attenuated,
                ]

                # Remove any None models
                related_models = [m for m in related_models if m is not None]

                # Call the parent constructor with everything we've made
                StellarEmissionModel.__init__(
                    self,
                    grid=self._grid,
                    label="emergent" if label is None else label,
                    combine=(young_total, old_total),
                    related_models=related_models,
                    **kwargs,
                )

            else:
                # Otherwise, emergent = attenuated + escaped

                # Get the young and old emergent emission
                young_total = StellarEmissionModel(
                    label="young_emergent",
                    combine=(self.young_attenuated, self.young_escaped),
                    **kwargs,
                )
                old_total = StellarEmissionModel(
                    label="old_emergent",
                    combine=(self.old_attenuated, self.old_escaped),
                    **kwargs,
                )

                # Define the related models
                related_models = [
                    self.young_incident,
                    self.old_incident,
                    self.incident,
                    self.young_transmitted,
                    self.old_transmitted,
                    self.transmitted,
                    self.young_escaped,
                    self.old_escaped,
                    self.escaped,
                    self.young_nebular,
                    self.old_nebular,
                    self.nebular,
                    self.young_reprocessed,
                    self.old_reprocessed,
                    self.reprocessed,
                    self.young_intrinsic,
                    self.old_intrinsic,
                    self.intrinsic,
                    self.young_attenuated_nebular,
                    self.young_attenuated_ism,
                    self.young_attenuated,
                    self.old_attenuated,
                    self.attenuated,
                ]

                # Remove any None models
                related_models = [m for m in related_models if m is not None]

                # Call the parent constructor with everything we've made
                StellarEmissionModel.__init__(
                    self,
                    grid=self._grid,
                    label="emergent" if label is None else label,
                    combine=(young_total, old_total),
                    related_models=related_models,
                    **kwargs,
                )


class CharlotFall2000(BimodalPacmanEmission):
    """The Charlot & Fall (2000) emission model.

    This emission model is based on the Charlot & Fall (2000) model, which
    describes the emission from a galaxy as a combination of emission from a
    young stellar population and an old stellar population. The dust
    attenuation for each population can be different, and dust emission can be
    optionally included.

    This model is a simplified version of the BimodalPacmanEmission model, so
    in reality is just a wrapper around that model. The only difference is that
    there is no option to specify an escape fraction.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object.
        tau_v_ism (float): The V-band optical depth for the ISM.
        tau_v_birth (float): The V-band optical depth for the nebular.
        dust_curve_ism (AttenuationLaw):
            The dust curve for the ISM.
        dust_curve_birth (AttenuationLaw): The dust curve for the
            nebular.
        age_pivot (unyt.unyt_quantity): The age pivot between young and old
            populations, expressed in terms of log10(age) in Myr.
        dust_emission_ism (synthesizer.dust.EmissionModel): The dust
            emission model for the ISM.
        dust_emission_birth (synthesizer.dust.EmissionModel): The dust
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
            grid (synthesizer.grid.Grid):
                The grid object.
            tau_v_ism (float):
                The V-band optical depth for the ISM.
            tau_v_birth (float):
                The V-band optical depth for the nebular.
            age_pivot (unyt.unyt_quantity):
                The age pivot between young and old populations, expressed
                in terms of log10(age) in Myr. Defaults to 10^7 Myr.
            dust_curve_ism (AttenuationLaw):
                The dust curve for the ISM. Defaults to `Calzetti2000`
                with fiducial parameters.
            dust_curve_birth (AttenuationLaw):
                The dust curve for the nebular. Defaults to `Calzetti2000`
                with fiducial parameters.
            dust_emission_ism (synthesizer.dust.EmissionModel):
                The dust emission model for the ISM.
            dust_emission_birth (synthesizer.dust.EmissionModel):
                The dust emission model for the nebular.
            label (str):
                The label for the total emission model. If None this will
                be set to "total" or "emergent" if dust_emission is `None`.
            stellar_dust (bool):
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
        grid (synthesizer.grid.Grid): The grid object.
        dust_curve (AttenuationLaw): The dust curve.
        dust_emission (synthesizer.dust.EmissionModel): The dust emission
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
            grid,
            tau_v,
            dust_curve,
            dust_emission,
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            label=label,
            **kwargs,
        )
