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

    def __init__(self, grid, label="incident", fesc=0.0, **kwargs):
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
            **kwargs,
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
        **kwargs,
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
            **kwargs,
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

    def __init__(self, grid, label="transmitted", fesc=0.0, **kwargs):
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
            **kwargs,
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
        **kwargs,
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
            **kwargs,
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
        **kwargs,
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
            **kwargs,
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
        fesc_ly_alpha=1.0,
        linecont=None,
        nebular_continuum=None,
        **kwargs,
    ):
        """
        Initialise the NebularEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            feac_ly_alpha (float): The escape fraction of Lyman-alpha.
            linecont (EmissionModel): The line continuum model to use, if None
                then one will be created. Only used if fesc_ly_alpha < 1.0.
            nebular_continuum (EmissionModel): The nebular continuum model to
                use, if None then one will be created. Only used if
                fesc_ly_alpha < 1.0.
        """
        # If we have a Lyman-alpha escape fraction then we need to combine
        # the Lyman-alpha line emission with the nebular continuum emission
        # and the rest of the line emission
        if fesc_ly_alpha < 1.0:
            # Make a line continuum model if we need one
            if linecont is None:
                linecont = LineContinuumEmission(grid=grid, fesc=fesc_ly_alpha)

            # Make a nebular continuum model if we need one
            if nebular_continuum is None:
                nebular_continuum = NebularContinuumEmission(
                    grid=grid, fesc=fesc
                )

            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(linecont, nebular_continuum),
                fesc=fesc,
                **kwargs,
            )
        else:
            # Otherwise, we just need the nebular emission from the grid
            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                extract="nebular",
                fesc=fesc,
                **kwargs,
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
        fesc_ly_alpha=1.0,
        nebular=None,
        transmitted=None,
        **kwargs,
    ):
        """
        Initialise the ReprocessedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            nebular (EmissionModel): The nebular model to use, if None then one
                will be created.
            transmitted (EmissionModel): The transmitted model to use, if None
                then one will be created.
        """
        # Make a nebular model if we need one
        if nebular is None:
            nebular = NebularEmission(
                grid=grid,
                fesc=fesc,
                fesc_ly_alpha=fesc_ly_alpha,
            )

        # Make a transmitted model if we need one
        if transmitted is None:
            transmitted = TransmittedEmission(grid=grid, fesc=fesc)

        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(nebular, transmitted),
            fesc=fesc,
            **kwargs,
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
        dust_curve,
        apply_dust_to,
        tau_v,
        label="attenuated",
        **kwargs,
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
            label=label,
            dust_curve=dust_curve,
            apply_dust_to=apply_dust_to,
            tau_v=tau_v,
            **kwargs,
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
        dust_curve=None,
        apply_dust_to=None,
        tau_v=None,
        fesc=0.0,
        label="emergent",
        attenuated=None,
        escaped=None,
        **kwargs,
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
            attenuated (EmissionModel): The attenuated model to use, if None
                then one will be created.
            escaped (EmissionModel): The escaped model to use, if None then one
                will be created.
        """
        # Make an attenuated model if we need one
        if attenuated is None:
            attenuated = AttenuatedEmission(
                grid=grid,
                dust_curve=dust_curve,
                apply_dust_to=apply_dust_to,
                tau_v=tau_v,
            )

        # Make an escaped model if we need one
        if escaped is None:
            escaped = EscapedEmission(grid=grid, fesc=fesc)

        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(attenuated, escaped),
            fesc=fesc,
            **kwargs,
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
        dust_lum_intrinsic=None,
        dust_lum_attenuated=None,
        label="dust_emission",
        **kwargs,
    ):
        """
        Initialise the DustEmission object.

        Args:
            dust_emission_model (synthesizer.dust.DustEmissionModel): The dust
                emission model to use.
            dust_lum_intrinsic (EmissionModel): The intrinsic spectra to use
                when calculating dust luminosity.
            dust_lum_attenuated (EmissionModel): The attenuated spectra to use
                when calculating dust luminosity.
            label (str): The label for this emission model.
        """
        EmissionModel.__init__(
            self,
            label=label,
            dust_emission_model=dust_emission_model,
            dust_lum_intrinsic=dust_lum_intrinsic,
            dust_lum_attenuated=dust_lum_attenuated,
            **kwargs,
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
        fesc_ly_alpha=1.0,
        **kwargs,
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
        # Set up models we need to link
        escaped = EscapedEmission(grid=grid, fesc=fesc, **kwargs)
        nebular = NebularEmission(
            grid=grid,
            fesc=fesc,
            fesc_ly_alpha=fesc_ly_alpha,
            **kwargs,
        )
        transmitted = TransmittedEmission(grid=grid, fesc=fesc, **kwargs)
        reprocessed = ReprocessedEmission(
            grid=grid,
            fesc=fesc,
            nebular=nebular,
            transmitted=transmitted,
            **kwargs,
        )
        attenuated = AttenuatedEmission(
            grid=grid,
            dust_curve=dust_curve,
            apply_dust_to=reprocessed,
            tau_v=tau_v,
            **kwargs,
        )

        # If a dust emission model has been passed then we need combine
        if dust_emission_model is not None:
            emergent = EmergentEmission(
                grid=grid,
                fesc=fesc,
                attenuated=attenuated,
                escaped=escaped,
                **kwargs,
            )
            dust_emission = DustEmission(
                dust_emission_model=dust_emission_model,
                dust_lum_intrinsic=reprocessed,
                dust_lum_attenuated=attenuated,
                **kwargs,
            )

            # Make the total emission model
            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(
                    emergent,
                    dust_emission,
                ),
                fesc=fesc,
                **kwargs,
            )
        else:
            # Otherwise, total == emergent
            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(attenuated, escaped),
                fesc=fesc,
                **kwargs,
            )
