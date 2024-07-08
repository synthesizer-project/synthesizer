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

    tau_v_ism = 0.5
    tau_v_nebular = 0.5
    dust_curve_ism = PowerLaw(...)
    dust_curve_nebular = PowerLaw(...)
    age_pivot = 7 * dimensionless
    dust_emission_ism = BlackBody(...)
    dust_emission_nebular = GreyBody(...)
    model = CharlotFall2000(
        grid,
        tau_v_ism,
        tau_v_nebular,
        dust_curve_ism,
        dust_curve_nebular,
        age_pivot,
        dust_emission_ism,
        dust_emission_nebular,
    )
"""

from unyt import dimensionless

from synthesizer.emission_models.base_model import StellarEmissionModel
from synthesizer.emission_models.models import (
    AttenuatedEmission,
    DustEmission,
)
from synthesizer.emission_models.stellar.models import (
    EscapedEmission,
    IncidentEmission,
    LineContinuumEmission,
    NebularContinuumEmission,
    NebularEmission,
    TransmittedEmission,
)


class PacmanEmission(StellarEmissionModel):
    """
    A class defining the Pacman model.

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

    if grid.reprocessed is True:
        - transmitted: the stellar emission transmitted through the ISM.
        - nebular: the stellar emisison from nebulae.
        - reprocessed: the stellar emission reprocessed by the ISM.

    if fesc > 0.0:
        - escaped: the incident emission that completely escapes the ISM.
        - emergent: the emission which emerges from the stellar population,
            including any escaped emission.


    if dust_emission is not None:
        - dust_emission: the emission from dust.
        - total: the final total combined emission.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object
        tau_v (float): The V-band optical depth
        dust_curve (synthesizer.dust.DustCurve): The dust curve
        dust_emission (synthesizer.dust.EmissionModel): The dust emission
        fesc (float): The escape fraction
        fesc_ly_alpha (float): The Lyman alpha escape fraction
    """

    def __init__(
        self,
        grid,
        tau_v,
        dust_curve,
        dust_emission=None,
        fesc=0.0,
        fesc_ly_alpha=1.0,
        label=None,
    ):
        """
        Initialize the PacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid): The grid object.
            tau_v (float): The V-band optical depth.
            dust_curve (synthesizer.dust.DustCurve): The dust curve.
            dust_emission (synthesizer.dust.EmissionModel): The dust
                emission.
            fesc (float): The escape fraction.
            fesc_ly_alpha (float): The Lyman alpha escape fraction.
            label (str): The label for the total emission model. If None
                this will be set to "total" or "emergent" if dust_emission is
                None.
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
        self.reprocessed = grid.reprocessed

        # Make the child emission models
        self.incident = self._make_incident()
        self.transmitted = self._make_transmitted()
        self.escaped = self._make_escaped()  # only if fesc > 0.0
        self.nebular = self._make_nebular()
        self.reprocessed = self._make_reprocessed()
        if not self.reprocessed:
            self.intrinsic = self._make_intrinsic_no_reprocessing()
        else:
            self.intrinsic = self._make_intrinsic_reprocessed()
        self.attenuated = self._make_attenuated()
        if self._dust_emission_model is not None:
            self.emergent = self._make_emergent()
            self.dust_emission = self._make_dust_emission()

        # Finally make the TotalEmission model, this is
        # dust_emission + emergent if dust_emission is not None, otherwise it
        # is just emergent
        self._make_total(label)

    def _make_incident(self):
        """
        Make the incident emission model.

        This will ignore any escape fraction given the stellar emission
        incident onto the nebular, ism and dust components.

        Returns:
            StellarEmissionModel:
                - incident
        """
        return IncidentEmission(grid=self._grid, label="incident")

    def _make_transmitted(self):
        """
        Make the transmitted emission model.

        If the grid has not been reprocessed, this will return None for all
        components because the transmitted spectra won't exist.

        Returns:
            StellarEmissionModel:
                - transmitted
        """
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        return TransmittedEmission(
            grid=self._grid, label="transmitted", fesc=self._fesc
        )

    def _make_escaped(self):
        """
        Make the escaped emission model.

        Escaped emission is the mirror of the transmitted emission. It is the
        fraction of the stellar emission that escapes the galaxy and is not
        transmitted through the ISM.

        If fesc=0.0 there is no escaped emission, and this will return None
        for all models.

        If the grid has not been reprocessed, this will return None for all
        components because the transmitted spectra won't exist.

        Returns:
            StellarEmissionModel:
                - escaped
        """
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        # No escaped emission if fesc is zero
        if self._fesc == 0.0:
            return None, None, None

        return EscapedEmission(
            grid=self._grid, label="escaped", fesc=self._fesc
        )

    def _make_nebular(self):
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        return NebularEmission(
            grid=self._grid,
            label="nebular",
            fesc=self._fesc,
            fesc_ly_alpha=self._fesc_ly_alpha,
        )

    def _make_reprocessed(self):
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        return StellarEmissionModel(
            label="reprocessed",
            combine=(self.transmitted, self.nebular),
        )

    def _make_intrinsic_no_reprocessing(self):
        """
        Make the intrinsic emission model for an un-reprocessed grid.

        This will produce the exact same emission as the incident emission but
        unlikely the incident emission, the intrinsic emission will be
        take into account an escape fraction.

        Returns:
            StellarEmissionModel:
                - intrinsic
        """
        return IncidentEmission(
            grid=self._grid, label="intrinsic", fesc=self._fesc
        )

    def _make_intrinsic_reprocessed(self):
        # If fesc is zero then the intrinsic emission is the same as the
        # reprocessed emission
        if self._fesc == 0.0:
            return StellarEmissionModel(
                label="intrinsic",
                combine=(self.reprocessed, self.transmitted),
            )

        # Otherwise, intrinsic = reprocessed + escaped
        return StellarEmissionModel(
            label="intrinsic",
            combine=(self.reprocessed, self.escaped),
        )

    def _make_attenuated(self):
        return AttenuatedEmission(
            label="attenuated",
            tau_v=self._tau_v,
            dust_curve=self._dust_curve,
            apply_dust_to=self.intrinsic,
            emitter="stellar",
        )

    def _make_emergent(self):
        # If fesc is zero the emergent spectra is just the attenuated spectra
        if self._fesc == 0.0:
            return AttenuatedEmission(
                label="emergent",
                tau_v=self._tau_v,
                dust_curve=self._dust_curve,
                apply_dust_to=self.intrinsic,
                emitter="stellar",
            )
        else:
            # Otherwise, emergent = attenuated + escaped
            return StellarEmissionModel(
                label="emergent",
                combine=(self.attenuated, self.escaped),
            )

    def _make_dust_emission(self):
        return DustEmission(
            label="dust_emission",
            dust_emission_model=self._dust_emission_model,
            dust_lum_intrinsic=self.incident,
            dust_lum_attenuated=self.attenuated,
            emitter="stellar",
        )

    def _make_total(self, label):
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
            StellarEmissionModel.__init__(
                self,
                grid=self._grid,
                label="total" if label is None else label,
                combine=(self.dust_emission, self.emergent),
                related_models=related_models,
            )
        else:
            # OK, total = emergent so we need to handle whether
            # emergent = attenuated + escaped or just attenuated
            if self._fesc == 0.0:
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
                    tau_v=self._tau_v,
                    dust_curve=self._dust_curve,
                    apply_dust_to=self.intrinsic,
                    related_models=related_models,
                )
            else:
                # Otherwise, emergent = attenuated + escaped

                # Define the related models
                related_models = [
                    self.incident,
                    self.transmitted,
                    self.escaped,
                    self.nebular,
                    self.reprocessed,
                    self.intrinsic,
                    self.attenuated,
                ]

                # Remove any None models
                related_models = [m for m in related_models if m is not None]

                # Call the parent constructor with everything we've made
                StellarEmissionModel.__init__(
                    self,
                    grid=self._grid,
                    label="emergent" if label is None else label,
                    combine=(self.attenuated, self.escaped),
                )


class BimodalPacmanEmission(StellarEmissionModel):
    """
    A class defining the Pacman model split into young and old populations.

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

    if fesc > 0.0:
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
        - young_dust_emission_nebular: the emission from dust for the young
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
        tau_v_nebular (float): The V-band optical depth for the nebular.
        dust_curve_ism (synthesizer.dust.DustCurve): The dust curve for the
            ISM.
        dust_curve_nebular (synthesizer.dust.DustCurve): The dust curve for the
            nebular.
        age_pivot (unyt.unyt_quantity): The age pivot between young and old
            populations, expressed in terms of log10(age) in Myr.
        dust_emission_ism (synthesizer.dust.EmissionModel): The dust
            emission for the ISM.
        dust_emission_nebular (synthesizer.dust.EmissionModel): The dust
            emission for the nebular.
        fesc (float): The escape fraction.
        fesc_ly_alpha (float): The Lyman alpha escape fraction.

    """

    def __init__(
        self,
        grid,
        tau_v_ism,
        tau_v_nebular,
        dust_curve_ism,
        dust_curve_nebular,
        age_pivot=7 * dimensionless,
        dust_emission_ism=None,
        dust_emission_nebular=None,
        fesc=0.0,
        fesc_ly_alpha=1.0,
        label=None,
    ):
        """
        Initialize the PacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid): The grid object.
            tau_v_ism (float): The V-band optical depth for the ISM.
            tau_v_nebular (float): The V-band optical depth for the nebular.
            dust_curve_ism (synthesizer.dust.DustCurve): The dust curve for the
                ISM.
            dust_curve_nebular (synthesizer.dust.DustCurve): The dust curve for
                the nebular.
            age_pivot (unyt.unyt_quantity): The age pivot between young and old
                populations, expressed in terms of log10(age) in Myr.
            dust_emission_ism (synthesizer.dust.EmissionModel): The dust
                emission model for the ISM.
            dust_emission_nebular (synthesizer.dust.EmissionModel): The
                dust emission model for the nebular.
            fesc (float): The escape fraction.
            fesc_ly_alpha (float): The Lyman alpha escape fraction.
            label (str): The label for the total emission model. If None
                this will be set to "total" or "emergent" if dust_emission is
                None.
        """
        # Attach the grid
        self._grid = grid

        # Attach the dust properties
        self.tau_v_ism = tau_v_ism
        self.tau_v_nebular = tau_v_nebular
        self._dust_curve_ism = dust_curve_ism
        self._dust_curve_nebular = dust_curve_nebular

        # Attach the age pivot
        self.age_pivot = age_pivot

        # Attach the dust emission properties
        self.dust_emission_ism = dust_emission_ism
        self.dust_emission_nebular = dust_emission_nebular

        # Attach the escape fraction properties
        self._fesc = fesc
        self._fesc_ly_alpha = fesc_ly_alpha

        # Are we using a grid with reprocessing?
        self.reprocessed = grid.reprocessed

        # Make the child emission models
        (
            self.young_incident,
            self.old_incident,
            self.incident,
        ) = self._make_incident()
        (
            self.young_transmitted,
            self.old_transmitted,
            self.transmitted,
        ) = self._make_transmitted()
        (
            self.young_escaped,
            self.old_escaped,
            self.escaped,
        ) = self._make_escaped()  # only if fesc > 0.0
        (
            self.young_nebular,
            self.old_nebular,
            self.nebular,
        ) = self._make_nebular()
        (
            self.young_reprocessed,
            self.old_reprocessed,
            self.reprocessed,
        ) = self._make_reprocessed()
        if not self.reprocessed:
            (
                self.young_intrinsic,
                self.old_intrinsic,
                self.intrinsic,
            ) = self._make_intrinsic_no_reprocessing()
        else:
            (
                self.young_intrinsic,
                self.old_intrinsic,
                self.intrinsic,
            ) = self._make_intrinsic_reprocessed()
        (
            self.young_attenuated_nebular,
            self.young_attenuated_ism,
            self.young_attenuated,
            self.old_attenuated,
            self.attenuated,
        ) = self._make_attenuated()
        if (
            self.dust_emission_ism is not None
            and self.dust_emission_nebular is not None
        ):
            (
                self.young_emergent,
                self.old_emergent,
                self.emergent,
            ) = self._make_emergent()
            (
                self.young_dust_emission_nebular,
                self.young_dust_emission_ism,
                self.young_dust_emission,
                self.old_dust_emission,
                self.dust_emission,
            ) = self._make_dust_emission()

        # Finally make the TotalEmission model, this is
        # dust_emission + emergent if dust_emission is not None, otherwise it
        # is just emergent
        self._make_total(label)

    def _make_incident(self):
        """
        Make the incident emission model.

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
        )
        old_incident = IncidentEmission(
            grid=self._grid,
            label="old_incident",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
        )
        incident = StellarEmissionModel(
            label="incident",
            combine=(young_incident, old_incident),
        )

        return young_incident, old_incident, incident

    def _make_transmitted(self):
        """
        Make the transmitted emission model.

        If the grid has not been reprocessed, this will return None for all
        components because the transmitted spectra won't exist.

        Returns:
            StellarEmissionModel:
                - young_transmitted
                - old_transmitted
                - transmitted
        """
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        young_transmitted = TransmittedEmission(
            grid=self._grid,
            label="young_transmitted",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc=self._fesc,
        )
        old_transmitted = TransmittedEmission(
            grid=self._grid,
            label="old_transmitted",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
        )
        transmitted = StellarEmissionModel(
            label="transmitted",
            combine=(young_transmitted, old_transmitted),
        )

        return young_transmitted, old_transmitted, transmitted

    def _make_escaped(self):
        """
        Make the escaped emission model.

        Escaped emission is the mirror of the transmitted emission. It is the
        fraction of the stellar emission that escapes the galaxy and is not
        transmitted through the ISM.

        If fesc=0.0 there is no escaped emission, and this will return None
        for all models.

        If the grid has not been reprocessed, this will return None for all
        components because the transmitted spectra won't exist.

        Returns:
            StellarEmissionModel:
                - young_escaped
                - old_escaped
                - escaped
        """
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        # No escaped emission if fesc is zero
        if self._fesc == 0.0:
            return None, None, None

        young_escaped = EscapedEmission(
            grid=self._grid,
            label="young_escaped",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc=self._fesc,
        )
        old_escaped = EscapedEmission(
            grid=self._grid,
            label="old_escaped",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
        )
        escaped = StellarEmissionModel(
            label="escaped",
            combine=(young_escaped, old_escaped),
        )

        return young_escaped, old_escaped, escaped

    def _make_nebular(self):
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        # Get the line continuum emission
        young_line_cont = LineContinuumEmission(
            grid=self._grid,
            label="young_linecont",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc=self._fesc_ly_alpha,
        )
        old_line_cont = LineContinuumEmission(
            grid=self._grid,
            label="old_linecont",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc_ly_alpha,
        )

        # Get the nebular continuum emission
        young_neb_cont = NebularContinuumEmission(
            grid=self._grid,
            label="young_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc=self._fesc,
        )
        old_neb_cont = NebularContinuumEmission(
            grid=self._grid,
            label="old_nebular_continuum",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
        )

        young_nebular = NebularEmission(
            grid=self._grid,
            label="young_nebular",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc=self._fesc,
            fesc_ly_alpha=self._fesc_ly_alpha,
            linecont=young_line_cont,
            nebular_continuum=young_neb_cont,
        )
        old_nebular = NebularEmission(
            grid=self._grid,
            label="old_nebular",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
            fesc_ly_alpha=self._fesc_ly_alpha,
            linecont=old_line_cont,
            nebular_continuum=old_neb_cont,
        )
        nebular = StellarEmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
        )

        return young_nebular, old_nebular, nebular

    def _make_reprocessed(self):
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        young_reprocessed = StellarEmissionModel(
            label="young_reprocessed",
            combine=(self.young_transmitted, self.young_nebular),
        )
        old_reprocessed = StellarEmissionModel(
            label="old_reprocessed",
            combine=(self.old_transmitted, self.old_nebular),
        )
        reprocessed = StellarEmissionModel(
            label="reprocessed",
            combine=(young_reprocessed, old_reprocessed),
        )

        return young_reprocessed, old_reprocessed, reprocessed

    def _make_intrinsic_no_reprocessing(self):
        """
        Make the intrinsic emission model for an un-reprocessed grid.

        This will produce the exact same emission as the incident emission but
        unlikely the incident emission, the intrinsic emission will be
        take into account an escape fraction.

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
        )
        old_intrinsic = IncidentEmission(
            grid=self._grid,
            label="old_intrinsic",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
        )
        intrinsic = StellarEmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
        )

        return young_intrinsic, old_intrinsic, intrinsic

    def _make_intrinsic_reprocessed(self):
        # If fesc is zero then the intrinsic emission is the same as the
        # reprocessed emission
        if self._fesc == 0.0:
            young_intrinsic = StellarEmissionModel(
                label="young_intrinsic",
                combine=(self.young_nebular, self.young_transmitted),
            )
            old_intrinsic = StellarEmissionModel(
                label="old_intrinsic",
                combine=(self.old_nebular, self.old_transmitted),
            )
            intrinsic = StellarEmissionModel(
                label="intrinsic",
                combine=(young_intrinsic, old_intrinsic),
            )
        else:
            # Otherwise, intrinsic = reprocessed + escaped
            young_intrinsic = StellarEmissionModel(
                label="young_intrinsic",
                combine=(self.young_reprocessed, self.young_escaped),
            )
            old_intrinsic = StellarEmissionModel(
                label="old_intrinsic",
                combine=(self.old_reprocessed, self.old_escaped),
            )
            intrinsic = StellarEmissionModel(
                label="intrinsic",
                combine=(young_intrinsic, old_intrinsic),
            )

        return young_intrinsic, old_intrinsic, intrinsic

    def _make_attenuated(self):
        young_attenuated_nebular = AttenuatedEmission(
            label="young_attenuated_nebular",
            tau_v=self.tau_v_nebular,
            dust_curve=self._dust_curve_nebular,
            apply_dust_to=self.young_intrinsic,
            emitter="stellar",
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_dust_to=self.young_intrinsic,
            emitter="stellar",
        )
        young_attenuated = AttenuatedEmission(
            label="young_attenuated",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_dust_to=young_attenuated_nebular,
            emitter="stellar",
        )
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_dust_to=self.old_intrinsic,
            emitter="stellar",
        )
        attenuated = StellarEmissionModel(
            label="attenuated",
            combine=(young_attenuated, old_attenuated),
        )

        return (
            young_attenuated_nebular,
            young_attenuated_ism,
            young_attenuated,
            old_attenuated,
            attenuated,
        )

    def _make_emergent(self):
        # If fesc is zero the emergent spectra is just the attenuated spectra
        if self._fesc == 0.0:
            young_emergent = AttenuatedEmission(
                label="young_emergent",
                tau_v=self.tau_v_ism,
                dust_curve=self._dust_curve_ism,
                apply_dust_to=self.young_attenuated_nebular,
                emitter="stellar",
            )
            old_emergent = AttenuatedEmission(
                label="old_emergent",
                tau_v=self.tau_v_ism,
                dust_curve=self._dust_curve_ism,
                apply_dust_to=self.old_intrinsic,
                emitter="stellar",
            )
            emergent = StellarEmissionModel(
                label="emergent",
                combine=(young_emergent, old_emergent),
            )
        else:
            # Otherwise, emergent = attenuated + escaped
            young_emergent = StellarEmissionModel(
                label="young_emergent",
                combine=(self.young_attenuated, self.young_escaped),
            )
            old_emergent = StellarEmissionModel(
                label="old_emergent",
                combine=(self.old_attenuated, self.old_escaped),
            )
            emergent = StellarEmissionModel(
                label="emergent",
                combine=(young_emergent, old_emergent),
            )

        return young_emergent, old_emergent, emergent

    def _make_dust_emission(self):
        young_dust_emission_nebular = DustEmission(
            label="young_dust_emission_nebular",
            dust_emission_model=self.dust_emission_nebular,
            dust_lum_intrinsic=self.young_incident,
            dust_lum_attenuated=self.young_attenuated_nebular,
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            emitter="stellar",
        )
        young_dust_emission_ism = DustEmission(
            label="young_dust_emission_ism",
            dust_emission_model=self.dust_emission_ism,
            dust_lum_intrinsic=self.young_incident,
            dust_lum_attenuated=self.young_attenuated_ism,
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            emitter="stellar",
        )
        young_dust_emission = StellarEmissionModel(
            label="young_dust_emission",
            combine=(young_dust_emission_nebular, young_dust_emission_ism),
        )
        old_dust_emission = DustEmission(
            label="old_dust_emission",
            dust_emission_model=self.dust_emission_ism,
            dust_lum_intrinsic=self.old_incident,
            dust_lum_attenuated=self.old_attenuated,
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            emitter="stellar",
        )
        dust_emission = StellarEmissionModel(
            label="dust_emission",
            combine=(young_dust_emission, old_dust_emission),
        )

        return (
            young_dust_emission_nebular,
            young_dust_emission_ism,
            young_dust_emission,
            old_dust_emission,
            dust_emission,
        )

    def _make_total(self, label):
        if (
            self.dust_emission_ism is not None
            and self.dust_emission_nebular is not None
        ):
            # Get the young and old total emission
            young_total = StellarEmissionModel(
                label="young_total",
                combine=(self.young_dust_emission, self.young_emergent),
            )
            old_total = StellarEmissionModel(
                label="old_total",
                combine=(self.old_dust_emission, self.old_emergent),
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
                self.young_dust_emission_nebular,
                self.young_dust_emission_ism,
                self.young_dust_emission,
                self.old_dust_emission,
                self.dust_emission,
            ]

            # Remove any None models
            related_models = [m for m in related_models if m is not None]

            # Call the parent constructor with everything we've made
            StellarEmissionModel.__init__(
                self,
                grid=self._grid,
                label="total" if label is None else label,
                combine=(young_total, old_total),
                related_models=related_models,
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
                    apply_dust_to=self.young_intrinsic,
                    emitter="stellar",
                )
                old_total = AttenuatedEmission(
                    label="old_emergent",
                    tau_v=self.tau_v_ism,
                    dust_curve=self._dust_curve_ism,
                    apply_dust_to=self.old_intrinsic,
                    emitter="stellar",
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
                )

            else:
                # Otherwise, emergent = attenuated + escaped

                # Get the young and old emergent emission
                young_total = StellarEmissionModel(
                    label="young_emergent",
                    combine=(self.young_attenuated, self.young_escaped),
                )
                old_total = StellarEmissionModel(
                    label="old_emergent",
                    combine=(self.old_attenuated, self.old_escaped),
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
                )


class CharlotFall2000(BimodalPacmanEmission):
    """
    The Charlot & Fall (2000) emission model.

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
        tau_v_nebular (float): The V-band optical depth for the nebular.
        dust_curve_ism (synthesizer.dust.DustCurve): The dust curve for the
            ISM.
        dust_curve_nebular (synthesizer.dust.DustCurve): The dust curve for the
            nebular.
        age_pivot (unyt.unyt_quantity): The age pivot between young and old
            populations, expressed in terms of log10(age) in Myr.
        dust_emission_ism (synthesizer.dust.EmissionModel): The dust
            emission model for the ISM.
        dust_emission_nebular (synthesizer.dust.EmissionModel): The dust
            emission model for the nebular.
    """

    def __init__(
        self,
        grid,
        tau_v_ism,
        tau_v_nebular,
        dust_curve_ism,
        dust_curve_nebular,
        age_pivot=7 * dimensionless,
        dust_emission_ism=None,
        dust_emission_nebular=None,
        label=None,
    ):
        """
        Initialize the PacmanEmission model.

        Args:
            grid (synthesizer.grid.Grid): The grid object.
            tau_v_ism (float): The V-band optical depth for the ISM.
            tau_v_nebular (float): The V-band optical depth for the nebular.
            dust_curve_ism (synthesizer.dust.DustCurve): The dust curve for the
                ISM.
            dust_curve_nebular (synthesizer.dust.DustCurve): The dust curve for
                the nebular.
            age_pivot (unyt.unyt_quantity): The age pivot between young and old
                populations, expressed in terms of log10(age) in Myr.
            dust_emission_ism (synthesizer.dust.EmissionModel): The dust
                emission model for the ISM.
            dust_emission_nebular (synthesizer.dust.EmissionModel): The
                dust emission model for the nebular.
        """
        # Call the parent constructor to intialise the model
        BimodalPacmanEmission.__init__(
            self,
            grid,
            tau_v_ism,
            tau_v_nebular,
            dust_curve_ism,
            dust_curve_nebular,
            age_pivot,
            dust_emission_ism,
            dust_emission_nebular,
            label=label,
        )
