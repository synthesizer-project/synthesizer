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
    """
    An emission model that extracts the incident radiation field.

    This defines an extraction of key "incident" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        label="incident",
        fesc=0.0,
    ):
        """
        Initialise the IncidentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            fesc=fesc,
        )


class LineContinuumEmission(EmissionModel):
    """
    An emission model that extracts the line continuum emission.

    This defines an extraction of key "linecont" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        label="linecont",
        fesc=0.0,
    ):
        """
        Initialise the LineContinuumEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="linecont",
            fesc=fesc,
        )


class TransmittedEmission(EmissionModel):
    """
    An emission model that extracts the transmitted radiation field.

    This defines an extraction of key "transmitted" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        label="transmitted",
        fesc=0.0,
    ):
        """
        Initialise the TransmittedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=fesc,
        )


class EscapedEmission(EmissionModel):
    """
    An emission model that extracts the escaped radiation field.

    This defines an extraction of key "escaped" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Note: The escaped model is the "mirror" to the transmitted model. What
    is transmitted is not escaped and vice versa. Therefore
    EscapedEmission.fesc = 1 - TransmittedEmission.fesc. This will be
    checked to be true at the time of spectra generation.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        label="escaped",
        fesc=0.0,
    ):
        """
        Initialise the EscapedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission. (Note that,
                          1-fesc will be used during generation).
        """
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=(1 - fesc),
        )


class NebularContinuumEmission(EmissionModel):
    """
    An emission model that extracts the nebular continuum emission.

    This defines an extraction of key "nebular_continuum" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        label="nebular_continuum",
        fesc=0.0,
    ):
        """
        Initialise the NebularContinuumEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="nebular_continuum",
            fesc=fesc,
        )


class NebularEmission(EmissionModel):
    """
    An emission model that combines the nebular emission.

    This defines a combination of the nebular continuum and line emission
    components.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        combine (list): The emission models to combine.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        label="nebular",
        fesc=0.0,
    ):
        """
        Initialise the NebularEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
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
    """
    An emission model that combines the reprocessed emission.

    This defines a combination of the nebular and transmitted components.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        combine (list): The emission models to combine.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        label="reprocessed",
        fesc=0.0,
    ):
        """
        Initialise the ReprocessedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
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
    """
    An emission model that defines the attenuated emission.

    This defines the attenuation of the reprocessed emission by dust.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
        apply_dust_to (EmissionModel): The emission model to apply the dust to.
        tau_v (float): The optical depth of the dust.
        label (str): The label for this emission model.
    """

    def __init__(
        self,
        grid,
        dust_curve,
        apply_dust_to,
        tau_v,
        label="attenuated",
    ):
        """
        Initialise the AttenuatedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
            apply_dust_to (EmissionModel): The model to apply the dust to.
            tau_v (float): The optical depth of the dust.
            label (str): The label for this emission model.
        """
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            dust_curve=dust_curve,
            apply_dust_to=apply_dust_to,
            tau_v=tau_v,
        )


class EmergentEmission(EmissionModel):
    """
    An emission model that defines the emergent emission.

    This defines combination of the attenuated and escaped emission components
    to produce the emergent emission.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
        apply_dust_to (EmissionModel): The emission model to apply the dust to.
        tau_v (float): The optical depth of the dust.
        fesc (float): The escape fraction of the emission.
        label (str): The label for this emission model.
    """

    def __init__(
        self,
        grid,
        dust_curve,
        apply_dust_to,
        tau_v,
        fesc=0.0,
        label="emergent",
    ):
        """
        Initialise the EmergentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
            apply_dust_to (EmissionModel): The model to apply the dust to.
            tau_v (float): The optical depth of the dust.
            fesc (float): The escape fraction of the emission.
            label (str): The label for this emission model.
        """
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
                EscapedEmission(grid=grid, fesc=1 - apply_dust_to.fesc),
            ),
            fesc=fesc,
        )


class DustEmission(EmissionModel):
    """
    An emission model that defines the dust emission.

    This defines the dust emission model to use.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        dust_emission_model (synthesizer.dust.DustEmissionModel): The dust
            emission model to use.
        label (str): The label for this emission model.
    """

    def __init__(
        self,
        dust_emission_model,
        label="dust_emission",
    ):
        """
        Initialise the DustEmission object.

        Args:
            dust_emission_model (synthesizer.dust.DustEmissionModel): The dust
                emission model to use.
            label (str): The label for this emission model.
        """
        EmissionModel.__init__(
            self,
            grid=None,
            label=label,
            dust_emission_model=dust_emission_model,
        )


class TotalEmission(EmissionModel):
    """
    An emission model that defines the total emission.

    This defines the combination of the emergent and dust emission components
    to produce the total emission.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
        tau_v (float): The optical depth of the dust.
        dust_emission_model (synthesizer.dust.DustEmissionModel): The dust
            emission model to use.
        label (str): The label for this emission model.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        dust_curve,
        tau_v,
        dust_emission_model=None,
        label="total",
        fesc=0.0,
    ):
        """
        Initialise the TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
            tau_v (float): The optical depth of the dust.
            dust_emission_model (synthesizer.dust.DustEmissionModel): The dust
                emission model to use.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
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
