"""A submodule containing the Charlot & Fall (2000) emission model."""

from synthesizer.emission_models import (
    AttenuatedEmission,
    DustEmission,
    EmissionModel,
    NebularEmission,
    TransmittedEmission,
)


class CharlotFall2000(EmissionModel):
    def __init__(
        self,
        grid,
        tau_v_ism,
        tau_v_nebular,
        dust_curve_ism,
        dust_curve_nebular,
        age_pivot,
        dust_emission_ism=None,
        dust_emission_nebular=None,
    ):
        """TODO"""
        # We need to start by making all the child models we'll need

        # # Incident (split by age)
        # young_incident = IncidentEmission(
        #     grid=grid,
        #     label="young_incident",
        #     mask_attr="log10ages",
        #     mask_thresh=age_pivot,
        #     mask_op="<",
        # )
        # old_incident = IncidentEmission(
        #     grid=grid,
        #     label="old_incident",
        #     mask_attr="log10ages",
        #     mask_thresh=age_pivot,
        #     mask_op=">=",
        # )
        # incident = EmissionModel(
        #     label="incident",
        #     combine=(young_incident, old_incident),
        # )

        # Transmitted (split by age)
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
        )
        transmitted = EmissionModel(
            label="transmitted",
            combine=(young_transmitted, old_transmitted),
        )

        # Nebular (split by age)
        young_nebular = NebularEmission(
            grid=grid,
            label="young_nebular",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op="<",
        )
        old_nebular = NebularEmission(
            grid=grid,
            label="old_nebular",
            mask_attr="log10ages",
            mask_thresh=age_pivot,
            mask_op=">=",
        )
        nebular = EmissionModel(
            label="nebular",
            combine=(young_nebular, old_nebular),
        )

        # Reprocessed (split by age)
        young_reprocessed = EmissionModel(
            label="young_reprocessed",
            combine=(young_transmitted, young_nebular),
        )
        old_reprocessed = EmissionModel(
            label="old_reprocessed",
            combine=(old_transmitted, old_nebular),
        )
        reprocessed = EmissionModel(
            label="reprocessed",
            combine=(young_reprocessed, old_reprocessed),
        )

        # Intrinsic models (for Charlot & Fall 2000 these are the same
        # as the reprocessed models)
        young_intrinsic = EmissionModel(
            label="young_intrinsic",
            combine=(young_transmitted, young_nebular),
        )
        old_intrinsic = EmissionModel(
            label="old_intrinsic",
            combine=(old_transmitted, old_nebular),
        )
        intrinsic = EmissionModel(
            label="intrinsic",
            combine=(young_intrinsic, old_intrinsic),
        )

        # Attenuated models (for Charlot & Fall 2000 we have different
        # attenuation curves for ISM and nebular dust)
        young_attenuated_nebular = AttenuatedEmission(
            label="young_attenuated_nebular",
            tau_v=tau_v_nebular,
            dust_curve=dust_curve_nebular,
            apply_dust_to=young_intrinsic,
        )
        young_attenuated_ism = AttenuatedEmission(
            label="young_attenuated_ism",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_dust_to=young_intrinsic,
        )
        young_attenuated = AttenuatedEmission(
            label="young_attenuated",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_dust_to=young_attenuated_nebular,
        )
        old_attenuated = AttenuatedEmission(
            label="old_attenuated",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_dust_to=old_intrinsic,
        )
        attenuated = EmissionModel(
            label="attenuated",
            combine=(young_attenuated, old_attenuated),
        )

        # Emergent spectra (split by age, for Charlot & Fall 2000 fesc is 0
        # so the emergent spectra are the same as the attenuated spectra)
        young_emergent = AttenuatedEmission(
            label="young_emergent",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_dust_to=young_attenuated_nebular,
        )
        old_emergent = AttenuatedEmission(
            label="old_emergent",
            tau_v=tau_v_ism,
            dust_curve=dust_curve_ism,
            apply_dust_to=old_intrinsic,
        )
        emergent = EmissionModel(
            label="emergent",
            combine=(young_emergent, old_emergent),
        )

        # Dust Emission (split by age)
        if dust_emission_nebular is not None and dust_emission_ism is not None:
            young_dust_emission_nebular = DustEmission(
                label="young_dust_emission_nebular",
                dust_emission_model=dust_emission_nebular,
                dust_lum_intrinsic=young_intrinsic,
                dust_lum_attenuated=young_attenuated_nebular,
                mask_attr="log10ages",
                mask_thresh=age_pivot,
                mask_op="<",
            )
            young_dust_emission_ism = DustEmission(
                label="young_dust_emission_ism",
                dust_emission_model=dust_emission_ism,
                dust_lum_intrinsic=young_intrinsic,
                dust_lum_attenuated=young_attenuated_ism,
                mask_attr="log10ages",
                mask_thresh=age_pivot,
                mask_op="<",
            )
            young_dust_emission = EmissionModel(
                label="young_dust_emission",
                combine=(young_dust_emission_nebular, young_dust_emission_ism),
            )
            old_dust_emission = DustEmission(
                label="old_dust_emission",
                dust_emission_model=dust_emission_ism,
                dust_lum_intrinsic=old_intrinsic,
                dust_lum_attenuated=old_attenuated,
                mask_attr="log10ages",
                mask_thresh=age_pivot,
                mask_op=">=",
            )
            dust_emission = EmissionModel(
                label="dust_emission",
                combine=(young_dust_emission, old_dust_emission),
            )

            # Construct the total (split by age)
            young_total = EmissionModel(
                label="young_total",
                combine=(young_emergent, young_dust_emission),
            )
            old_total = EmissionModel(
                label="old_total",
                combine=(old_emergent, old_dust_emission),
            )

            # We have all the components, construct the final model
            EmissionModel.__init__(
                self,
                label="total",
                combine=(young_total, old_total),
                related_models=(
                    # incident,
                    transmitted,
                    nebular,
                    intrinsic,
                    reprocessed,
                    attenuated,
                    emergent,
                    dust_emission,
                ),
            )

        elif dust_emission_ism is None and dust_emission_nebular is None:
            # We have no dust emission, construct the final model which is
            # the equivalent of emergent
            EmissionModel.__init__(
                self,
                label="emergent",
                combine=(young_emergent, old_emergent),
                related_models=(
                    # incident,
                    transmitted,
                    nebular,
                    reprocessed,
                    intrinsic,
                    attenuated,
                    emergent,
                ),
            )

        else:
            raise ValueError(
                "Both dust_emission_ism and dust_emission_nebular must be set"
                " or neither."
            )
