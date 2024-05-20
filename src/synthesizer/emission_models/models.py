"""A submodule containing the definitions of common simple emisison models.

This module contains the definitions of simple emission models that can be
used "out of the box" to generate spectra from components or as a foundation
to work from when creating more complex models.

Example usage:
    # Create a simple emission model
    model = TotalEmission(
        grid=grid,
        dust_curve=dust_curve,
        tau_v=tau_v,
        dust_emission_model=dust_emission_model,
        fesc=0.0,
    )

    # Generate the spectra
    spectra = stars.get_spectra(model)

"""

from synthesizer.emission_models.base_model import EmissionModel


class IncidentEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="incident",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            fesc=fesc,
        )


class LineContinuumEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="linecont",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="linecont",
            fesc=fesc,
        )


class TransmittedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="transmitted",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=fesc,
        )


class EscapedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="escaped",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=(1 - fesc),
        )


class NebularContinuumEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="nebular_continuum",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="nebular_continuum",
            fesc=fesc,
        )


class NebularEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="nebular",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                LineContinuumEmission(grid=grid, fesc=fesc),
                NebularContinuumEmission(grid=grid, fesc=fesc),
            ),
            fesc=fesc,
        )


class ReprocessedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="reprocessed",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                NebularEmission(grid=grid, fesc=fesc),
                TransmittedEmission(grid=grid, fesc=fesc),
            ),
            fesc=fesc,
        )


class AttenuatedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        dust_curve,
        apply_dust_to,
        tau_v,
        label="attenuated",
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            dust_curve=dust_curve,
            apply_dust_to=apply_dust_to,
            tau_v=tau_v,
        )


class EmergentEmission(EmissionModel):
    def __init__(
        self,
        grid,
        dust_curve,
        apply_dust_to,
        tau_v,
        fesc=0.0,
        label="emergent",
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                AttenuatedEmission(
                    grid=grid,
                    dust_curve=dust_curve,
                    apply_dust_to=apply_dust_to,
                    tau_v=tau_v,
                ),
                EscapedEmission(grid=grid, fesc=fesc),
            ),
            fesc=fesc,
        )


class DustEmission(EmissionModel):
    def __init__(
        self,
        dust_emission_model,
        label="dust_emission",
    ):
        EmissionModel.__init__(
            self,
            grid=None,
            label=label,
            dust_emission_model=dust_emission_model,
        )


class TotalEmission(EmissionModel):
    def __init__(
        self,
        grid,
        dust_curve,
        tau_v,
        dust_emission_model=None,
        label="total",
        fesc=0.0,
    ):
        # If a dust emission model has been passed then we need combine
        if dust_emission_model is not None:
            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(
                    EmergentEmission(
                        grid=grid,
                        dust_curve=dust_curve,
                        tau_v=tau_v,
                        fesc=fesc,
                        apply_dust_to=ReprocessedEmission(
                            grid=grid,
                            fesc=fesc,
                        ),
                    ),
                    DustEmission(dust_emission_model=dust_emission_model),
                ),
                fesc=fesc,
            )
        else:
            # Otherwise, total == emergent
            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(
                    AttenuatedEmission(
                        grid=grid,
                        dust_curve=dust_curve,
                        apply_dust_to=ReprocessedEmission(
                            grid=grid,
                            fesc=fesc,
                        ),
                        tau_v=tau_v,
                    ),
                    EscapedEmission(grid=grid, fesc=fesc),
                ),
                fesc=fesc,
            )
