"""A submodule containing the definitions of common stellar emission models.

This module contains the definitions of commoon stellar emission models that
can be used "out of the box" to generate spectra from components or as a
foundation to work from when creating more complex models.

Example usage::

    # Create a simple emission model
    model = TotalEmission(
        grid=grid,
        dust_curve=dust_curve,
        dust_emission_model=dust_emission_model,
        fesc=0.0,
    )

    # Generate the spectra
    spectra = stars.get_spectra(model)

"""

import numpy as np
from unyt import Angstrom

from synthesizer.emission_models.base_model import StellarEmissionModel
from synthesizer.emission_models.models import AttenuatedEmission, DustEmission


class IncidentEmission(StellarEmissionModel):
    """
    An emission model that extracts the incident radiation field.

    This defines an extraction of key "incident" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
    """

    def __init__(self, grid, label="incident", **kwargs):
        """
        Initialise the IncidentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
        """
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            **kwargs,
        )


class NebularLineEmission(StellarEmissionModel):
    """
    An emission model for the nebular line emission.

    This defines the luminosity contribution of the lines to the total
    nebular output.

    This is a child of the EmissionModel class; for a full description of the
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
        label="nebular_line",
        fesc=0.0,
        fesc_ly_alpha=1.0,
        **kwargs,
    ):
        """
        Initialise the NebularLineEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
        """
        if fesc_ly_alpha < 1.0:
            # Get the lyman alpha wavelength elements and create a mask for
            # the line
            lyman_alpha_ind = np.argmin(np.abs(grid.lam - 1216.0 * Angstrom))
            lyman_alpha_mask = np.zeros(len(grid.lam), dtype=bool)
            lyman_alpha_mask[lyman_alpha_ind] = True

            # Since the spectra may have been resampled, we may have split the
            # lyman-alpha line into two. Therefore, we need to mask the
            # surrounding elements as well.
            if lyman_alpha_ind > 0:
                lyman_alpha_mask[lyman_alpha_ind - 1] = True
            if lyman_alpha_ind < len(grid.lam) - 1:
                lyman_alpha_mask[lyman_alpha_ind + 1] = True

            # Create the child models
            linecont_no_lyman = StellarEmissionModel(
                grid=grid,
                label=label + "_no_lyman",
                extract="linecont",
                fesc=fesc,
                lam_mask=~lyman_alpha_mask,
                save=False,
                **kwargs,
            )

            # For lyman-alpha, we reduce the overall luminosity by fesc,
            # then the remaining luminosity by fesc_ly_alpha
            lyman_alpha = StellarEmissionModel(
                label=label + "_lyman_alpha",
                extract="linecont",
                grid=grid,
                fesc=(1 - (1 - fesc) * fesc_ly_alpha),
                lam_mask=lyman_alpha_mask,
                save=False,
                **kwargs,
            )

            # Instantiate the combination model
            StellarEmissionModel.__init__(
                self,
                label=label,
                combine=(linecont_no_lyman, lyman_alpha),
                **kwargs,
            )
        else:
            StellarEmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                extract="linecont",
                fesc=fesc,
                **kwargs,
            )


class TransmittedEmission(StellarEmissionModel):
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
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=fesc,
            **kwargs,
        )


class EscapedEmission(StellarEmissionModel):
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
        fesc (float): The escape fraction of the emission. Note that this will
                        be set to 1 - fesc during generation, i.e. it is the
                        fraction that does not escape that should be passed.
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
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=(1 - fesc),
            **kwargs,
        )


class NebularContinuumEmission(StellarEmissionModel):
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
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="nebular_continuum",
            fesc=fesc,
            **kwargs,
        )


class NebularEmission(StellarEmissionModel):
    """
    An emission model that combines the nebular emissions.

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
        nebular_line=None,
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
            nebular_line (EmissionModel): The nebular line model to use, if
                None then one will be created. Only used if
                fesc_ly_alpha < 1.0.
            nebular_continuum (EmissionModel): The nebular continuum model to
                use, if None then one will be created. Only used if
                fesc_ly_alpha < 1.0.
        """
        # If we have a Lyman-alpha escape fraction then calculate the
        # updated line emission and combine with the nebular continuum.
        if fesc_ly_alpha < 1.0:
            # Make a nebular line model if we need one
            if nebular_line is None:
                nebular_line = NebularLineEmission(
                    grid=grid,
                    fesc_ly_alpha=fesc_ly_alpha,
                    fesc=fesc,
                    **kwargs,
                )

            # Make a nebular continuum model if we need one
            if nebular_continuum is None:
                nebular_continuum = NebularContinuumEmission(
                    grid=grid,
                    fesc=fesc,
                    **kwargs,
                )

            StellarEmissionModel.__init__(
                self,
                label=label,
                combine=(nebular_line, nebular_continuum),
                **kwargs,
            )

        else:
            # Otherwise, we just get the nebular emission directly
            # from the grid
            StellarEmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                extract="nebular",
                fesc=fesc,
                **kwargs,
            )


class IntrinsicEmission(StellarEmissionModel):
    """
    An emission model that defines the intrinsic emission.

    This defines a combination of the reprocessed and escaped emission.

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
        fesc,
        label="intrinsic",
        fesc_ly_alpha=1.0,
        escaped=None,
        reprocessed=None,
        **kwargs,
    ):
        """
        Initialise the IntrinsicEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            escaped (EmissionModel): The escaped model to use, if None then one
                will be created.
            reprocessed (EmissionModel): The reprocessed model to use, if None
                then one will be created.
        """
        # Ensure what we've been asked for makes sense
        if fesc == 0.0:
            raise ValueError(
                "Intrinsic emission model requires an escape fraction > 0.0"
            )

        # Make an escaped model if we need one
        if escaped is None:
            escaped = EscapedEmission(
                grid=grid,
                fesc=fesc,
                **kwargs,
            )

        # Make a reprocessed model if we need one
        if reprocessed is None:
            reprocessed = ReprocessedEmission(
                grid=grid,
                fesc=fesc,
                fesc_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )

        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(escaped, reprocessed),
            fesc=fesc,
            **kwargs,
        )


class ReprocessedEmission(StellarEmissionModel):
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
                **kwargs,
            )

        # Make a transmitted model if we need one
        if transmitted is None:
            transmitted = TransmittedEmission(
                grid=grid,
                fesc=fesc,
                **kwargs,
            )

        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(nebular, transmitted),
            fesc=fesc,
            **kwargs,
        )


class EmergentEmission(StellarEmissionModel):
    """
    An emission model that defines the emergent emission.

    This defines combination of the attenuated and escaped emission components
    to produce the emergent emission.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
        apply_to (EmissionModel): The emission model to apply the dust to.
        fesc (float): The escape fraction of the emission.
        label (str): The label for this emission model.
    """

    def __init__(
        self,
        grid,
        dust_curve=None,
        apply_to=None,
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
            apply_to (EmissionModel): The model to apply the dust to.
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
                apply_to=apply_to,
                emitter="stellar",
                **kwargs,
            )

        # Make an escaped model if we need one
        if escaped is None:
            escaped = EscapedEmission(grid=grid, fesc=fesc, **kwargs)

        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(attenuated, escaped),
            fesc=fesc,
            **kwargs,
        )


class TotalEmission(StellarEmissionModel):
    """
    An emission model that defines the total emission.

    This defines the combination of the emergent and dust emission components
    to produce the total emission.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        dust_curve (synthesizer.dust.DustCurve): The dust curve to use.
        dust_emission_model (synthesizer.dust.EmissionModel): The dust
            emission model to use.
        label (str): The label for this emission model.
        fesc (float): The escape fraction of the emission.
    """

    def __init__(
        self,
        grid,
        dust_curve,
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
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
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
            apply_to=reprocessed,
            emitter="stellar",
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
                emitter="stellar",
                **kwargs,
            )

            # Make the total emission model
            StellarEmissionModel.__init__(
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
            StellarEmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(attenuated, escaped),
                fesc=fesc,
                **kwargs,
            )
