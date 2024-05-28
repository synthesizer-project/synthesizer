""" """

from unyt import dimensionless

from synthesizer.emission_models import (
    AttenuatedEmission,
    DustEmission,
    EmissionModel,
    EscapedEmission,
    IncidentEmission,
    NebularEmission,
    TransmittedEmission,
)


class PacmanEmission(EmissionModel):
    def __init__(
        self,
        grid,
        tau_v=None,
        dust_curve=None,
        dust_emission=None,
        fesc=0.0,
        fesc_ly_alpha=1.0,
    ):
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
        self._make_total()

    def _make_incident(self):
        """
        Make the incident emission model.

        This will ignore any escape fraction given the stellar emission
        incident onto the nebular, ism and dust components.

        Returns:
            EmissionModel:
                - incident
        """
        return IncidentEmission(grid=self._grid, label="incident")

    def _make_transmitted(self):
        """
        Make the transmitted emission model.

        If the grid has not been reprocessed, this will return None for all
        components because the transmitted spectra won't exist.

        Returns:
            EmissionModel:
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
            EmissionModel:
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

        return EmissionModel(
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
            EmissionModel:
                - intrinsic
        """
        return IncidentEmission(
            grid=self._grid, label="intrinsic", fesc=self._fesc
        )

    def _make_intrinsic_reprocessed(self):
        # If fesc is zero then the intrinsic emission is the same as the
        # reprocessed emission
        if self._fesc == 0.0:
            return EmissionModel(
                label="intrinsic",
                combine=(self.reprocessed, self.transmitted),
            )

        # Otherwise, intrinsic = reprocessed + escaped
        return EmissionModel(
            label="intrinsic",
            combine=(self.reprocessed, self.escaped),
        )

    def _make_attenuated(self):
        return AttenuatedEmission(
            label="attenuated",
            tau_v=self._tau_v,
            dust_curve=self._dust_curve,
            apply_dust_to=self.intrinsic,
        )

    def _make_emergent(self):
        # If fesc is zero the emergent spectra is just the attenuated spectra
        if self._fesc == 0.0:
            return AttenuatedEmission(
                label="emergent",
                tau_v=self._tau_v,
                dust_curve=self._dust_curve,
                apply_dust_to=self.intrinsic,
            )
        else:
            # Otherwise, emergent = attenuated + escaped
            return EmissionModel(
                label="emergent",
                combine=(self.attenuated, self.escaped),
            )

    def _make_dust_emission(self):
        return DustEmission(
            label="dust_emission",
            dust_emission_model=self._dust_emission_model,
            dust_lum_intrinsic=self.incident,
            dust_lum_attenuated=self.attenuated,
        )

    def _make_total(self):
        if self._dust_emission_model is not None:
            EmissionModel.__init__(
                self,
                grid=self._grid,
                label="total",
                combine=(self.dust_emission, self.emergent),
            )
        else:
            # OK, total = emergent so we need to handle whether
            # emergent = attenuated + escaped or just attenuated
            if self._fesc == 0.0:
                return EmissionModel.__init__(
                    self,
                    grid=self._grid,
                    label="emergent",
                    tau_v=self._tau_v,
                    dust_curve=self._dust_curve,
                    apply_dust_to=self.intrinsic,
                )
            else:
                # Otherwise, emergent = attenuated + escaped
                return EmissionModel.__init__(
                    self,
                    grid=self._grid,
                    label="emergent",
                    combine=(self.attenuated, self.escaped),
                )


class BimodalPacmanEmission(EmissionModel):
    def __init__(
        self,
        grid,
        tau_v_ism=None,
        tau_v_nebular=None,
        dust_curve_ism=None,
        dust_curve_nebular=None,
        age_pivot=7 * dimensionless,
        dust_emission_ism=None,
        dust_emission_nebular=None,
        fesc=0.0,
        fesc_ly_alpha=1.0,
    ):
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
        self._make_total()

    def _make_incident(self):
        """
        Make the incident emission model.

        This will ignore any escape fraction given the stellar emission
        incident onto the nebular, ism and dust components.

        Returns:
            EmissionModel:
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
        incident = EmissionModel(
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
            EmissionModel:
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
        transmitted = EmissionModel(
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
            EmissionModel:
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
        escaped = EmissionModel(
            label="escaped",
            combine=(young_escaped, old_escaped),
        )

        return young_escaped, old_escaped, escaped

    def _make_nebular(self):
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        young_nebular = NebularEmission(
            grid=self._grid,
            label="young_nebular",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
            fesc=self._fesc,
            fesc_ly_alpha=self._fesc_ly_alpha,
        )
        old_nebular = NebularEmission(
            grid=self._grid,
            label="old_nebular",
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op=">=",
            fesc=self._fesc,
            fesc_ly_alpha=self._fesc_ly_alpha,
        )
        nebular = EmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
        )

        return young_nebular, old_nebular, nebular

    def _make_reprocessed(self):
        # No spectra if grid hasn't been reprocessed
        if not self.reprocessed:
            return None, None, None

        young_reprocessed = EmissionModel(
            label="young_reprocessed",
            combine=(self.young_transmitted, self.young_nebular),
        )
        old_reprocessed = EmissionModel(
            label="old_reprocessed",
            combine=(self.old_transmitted, self.old_nebular),
        )
        reprocessed = EmissionModel(
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
            EmissionModel:
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
        intrinsic = EmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
        )

        return young_intrinsic, old_intrinsic, intrinsic

    def _make_intrinsic_reprocessed(self):
        # If fesc is zero then the intrinsic emission is the same as the
        # reprocessed emission
        if self._fesc == 0.0:
            young_intrinsic = EmissionModel(
                label="young_intrinsic",
                combine=(self.young_reprocessed, self.young_transmitted),
            )
            old_intrinsic = EmissionModel(
                label="old_intrinsic",
                combine=(self.old_repocessed, self.old_transmitted),
            )
            intrinsic = EmissionModel(
                label="intrinsic",
                combine=(self.young_intrinsic, self.old_intrinsic),
            )
        else:
            # Otherwise, intrinsic = reprocessed + escaped
            young_intrinsic = EmissionModel(
                label="young_intrinsic",
                combine=(self.young_reprocessed, self.young_escaped),
            )
            old_intrinsic = EmissionModel(
                label="old_intrinsic",
                combine=(self.old_reprocessed, self.old_escaped),
            )
            intrinsic = EmissionModel(
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
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_dust_to=self.young_intrinsic,
        )
        young_attenuated = AttenuatedEmission(
            label="young_attenuated",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_dust_to=young_attenuated_nebular,
        )
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=self.tau_v_ism,
            dust_curve=self._dust_curve_ism,
            apply_dust_to=self.old_intrinsic,
        )
        attenuated = EmissionModel(
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
            )
            old_emergent = AttenuatedEmission(
                label="old_emergent",
                tau_v=self.tau_v_ism,
                dust_curve=self._dust_curve_ism,
                apply_dust_to=self.old_intrinsic,
            )
            emergent = EmissionModel(
                label="emergent",
                combine=(young_emergent, old_emergent),
            )
        else:
            # Otherwise, emergent = attenuated + escaped
            young_emergent = EmissionModel(
                label="young_emergent",
                combine=(self.young_attenuated, self.young_escaped),
            )
            old_emergent = EmissionModel(
                label="old_emergent",
                combine=(self.old_attenuated, self.old_escaped),
            )
            emergent = EmissionModel(
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
        )
        young_dust_emission_ism = DustEmission(
            label="young_dust_emission_ism",
            dust_emission_model=self.dust_emission_ism,
            dust_lum_intrinsic=self.young_incident,
            dust_lum_attenuated=self.young_attenuated_ism,
            mask_attr="log10ages",
            mask_thresh=self.age_pivot,
            mask_op="<",
        )
        young_dust_emission = EmissionModel(
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
        )
        dust_emission = EmissionModel(
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

    def _make_total(self):
        if (
            self.dust_emission_ism is not None
            and self.dust_emission_nebular is not None
        ):
            young_total = EmissionModel(
                label="young_total",
                combine=(self.young_dust_emission, self.young_emergent),
            )
            old_total = EmissionModel(
                label="old_total",
                combine=(self.old_dust_emission, self.old_emergent),
            )
            EmissionModel.__init__(
                self,
                grid=self._grid,
                label="total",
                combine=(young_total, old_total),
                related_models=[
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
                ],
            )
        else:
            # OK, total = emergent so we need to handle whether
            # emergent = attenuated + escaped or just attenuated
            if self._fesc == 0.0:
                young_total = AttenuatedEmission(
                    label="young_emergent",
                    tau_v=self.tau_v_ism,
                    dust_curve=self._dust_curve_ism,
                    apply_dust_to=self.young_intrinsic,
                )
                old_total = AttenuatedEmission(
                    label="old_emergent",
                    tau_v=self.tau_v_ism,
                    dust_curve=self._dust_curve_ism,
                    apply_dust_to=self.old_intrinsic,
                )
                EmissionModel.__init__(
                    self,
                    grid=self._grid,
                    label="emergent",
                    combine=(young_total, old_total),
                )
            else:
                # Otherwise, emergent = attenuated + escaped
                young_total = EmissionModel(
                    label="young_emergent",
                    combine=(self.young_attenuated, self.young_escaped),
                )
                old_total = EmissionModel(
                    label="old_emergent",
                    combine=(self.old_attenuated, self.old_escaped),
                )
                EmissionModel.__init__(
                    self,
                    grid=self._grid,
                    label="emergent",
                    combine=(young_total, old_total),
                )
