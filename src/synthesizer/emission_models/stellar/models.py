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

from synthesizer import exceptions
from synthesizer.emission_models.base_model import StellarEmissionModel
from synthesizer.emission_models.models import AttenuatedEmission, DustEmission
from synthesizer.emission_models.transformers import (
    EscapedFraction,
    ProcessedFraction,
)


class IncidentEmission(StellarEmissionModel):
    """An emission model that extracts the incident radiation field.

    This defines an extraction of key "incident" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
    """

    def __init__(self, grid, label="incident", **kwargs):
        """Initialise the IncidentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            **kwargs,
        )


class NebularLineEmission(StellarEmissionModel):
    """An emission model for the nebular line emission.

    This defines the luminosity contribution of the lines to the total
    nebular output.

    This is a child of the EmissionModel class; for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
    """

    def __init__(
        self,
        grid,
        label="nebular_line",
        fesc_ly_alpha="fesc_ly_alpha",
        fesc="fesc",
        **kwargs,
    ):
        """Initialise the NebularLineEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            **kwargs: Additional keyword arguments.
        """
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

        # For lyman-alpha, we reduce the overall luminosity by fesc,
        # then the remaining luminosity by fesc_ly_alpha
        lyman_alpha_no_fesc = StellarEmissionModel(
            label=label + "_no_fesc",
            extract="linecont",
            grid=grid,
            save=False,
            **kwargs,
        )

        # We can't define fesc and fesc_ly_alpha in the same model if
        # the user tried tell them they can't
        if "fesc" in kwargs:
            raise exceptions.InconsistentArguments(
                "Cannot define fesc and fesc_ly_alpha in the same model. "
                "Please use another Transformation on NebularLineEmission "
                "to apply your fesc."
            )

        # Instantiate the combination model
        StellarEmissionModel.__init__(
            self,
            label=label,
            apply_to=lyman_alpha_no_fesc,
            transformer=EscapedFraction(fesc_attrs=("fesc_ly_alpha",)),
            fesc_ly_alpha=fesc_ly_alpha,
            lam_mask=lyman_alpha_mask,
            **kwargs,
        )


class TransmittedEmission(StellarEmissionModel):
    """An emission model that extracts the transmitted radiation field.

    This defines 3 models when fesc > 0.0:
      - An extraction of the key "transmitted" from a SPS grid.
      - A transformed emission, for the transmitted radiation field accounting
        for the escape fraction.
      - A transformed emission, for the escaped radiation field accounting for
        the escape fraction.

    If fesc = 0.0 then there will only be the extraction model.

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
        fesc="fesc",
        related_models=(),
        **kwargs,
    ):
        """Initialise the TransmittedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            related_models (list): A list of related models to combine with.
                This is used to combine the escaped and transmitted emission
                models.
            **kwargs: Additional keyword arguments.
        """
        # Define the transmitted extraction model
        full_transmitted = StellarEmissionModel(
            grid=grid,
            label="full_" + label,
            extract="transmitted",
            **kwargs,
        )

        # Get the escaped emission
        escaped = StellarEmissionModel(
            label="escaped",
            grid=grid,
            apply_to=full_transmitted,
            transformer=EscapedFraction(),
            fesc=fesc,
            **kwargs,
        )

        # Combine any extra related_models
        related_models = (escaped,) + tuple(related_models)

        # Get the transmitted emission (accounting for fesc)
        StellarEmissionModel.__init__(
            self,
            label=label,
            apply_to=full_transmitted,
            transformer=ProcessedFraction(),
            fesc=fesc,
            related_models=related_models,
            **kwargs,
        )


class NebularContinuumEmission(StellarEmissionModel):
    """An emission model that extracts the nebular continuum emission.

    This defines an extraction of key "nebular_continuum" from a SPS grid.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        extract (str): The key to extract from the grid.
    """

    def __init__(
        self,
        grid,
        label="nebular_continuum",
        **kwargs,
    ):
        """Initialise the NebularContinuumEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            **kwargs: Additional keyword arguments.
        """
        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="nebular_continuum",
            **kwargs,
        )


class NebularEmission(StellarEmissionModel):
    """An emission model that combines the nebular emissions.

    This defines a combination of the nebular continuum and line emission
    components.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        combine (list): The emission models to combine.
    """

    def __init__(
        self,
        grid,
        label="nebular",
        fesc_ly_alpha="fesc_ly_alpha",
        fesc="fesc",
        nebular_line=None,
        nebular_continuum=None,
        **kwargs,
    ):
        """Initialise the NebularEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            fesc (float): The escape fraction of the emission.
            nebular_line (EmissionModel): The nebular line model to use, if
                None then one will be created. Only used if
                fesc_ly_alpha < 1.0.
            nebular_continuum (EmissionModel): The nebular continuum model to
                use, if None then one will be created. Only used if
                fesc_ly_alpha < 1.0.
            **kwargs: Additional keyword arguments.
        """
        # If we have a Lyman-alpha escape fraction then calculate the
        # updated line emission and combine with the nebular continuum.
        # Make a nebular line model if we need one
        if nebular_line is None:
            nebular_line = NebularLineEmission(
                grid=grid,
                fesc_ly_alpha=fesc_ly_alpha,
                label=label + "_line",
                **kwargs,
            )

        # Make a nebular continuum model if we need one
        if nebular_continuum is None:
            nebular_continuum = NebularContinuumEmission(
                grid=grid,
                label=label + "_continuum",
                **kwargs,
            )

        # Combined nebular emission
        combined_nebular = StellarEmissionModel(
            label=label + "_combined",
            combine=(nebular_line, nebular_continuum),
            save=False,
            **kwargs,
        )

        StellarEmissionModel.__init__(
            self,
            label=label,
            apply_to=combined_nebular,
            transformer=ProcessedFraction(),
            fesc=fesc,
            **kwargs,
        )


class ReprocessedEmission(StellarEmissionModel):
    """An emission model that combines the reprocessed emission.

    This defines a combination of the nebular and transmitted components.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        combine (list): The emission models to combine.
    """

    def __init__(
        self,
        grid,
        label="reprocessed",
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        nebular=None,
        transmitted=None,
        **kwargs,
    ):
        """Initialise the ReprocessedEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            nebular (EmissionModel): The nebular model to use, if None then one
                will be created.
            transmitted (EmissionModel): The transmitted model to use, if None
                then one will be created.
            **kwargs: Additional keyword arguments.
        """
        # Make a nebular model if we need one
        if nebular is None:
            nebular = NebularEmission(
                grid=grid,
                fesc_ly_alpha=fesc_ly_alpha,
                fesc=fesc,
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
            **kwargs,
        )


class IntrinsicEmission(StellarEmissionModel):
    """An emission model that defines the intrinsic emission.

    This defines a combination of the reprocessed and escaped emission.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        label (str): The label for this emission model.
        combine (list): The emission models to combine.
    """

    def __init__(
        self,
        grid,
        label="intrinsic",
        fesc_ly_alpha="fesc_ly_alpha",
        reprocessed=None,
        **kwargs,
    ):
        """Initialise the IntrinsicEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            label (str): The label for this emission model.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            reprocessed (EmissionModel): The reprocessed model to use, if None
                then one will be created.
            **kwargs: Additional keyword arguments.
        """
        # Make a reprocessed model if we need one
        if reprocessed is None:
            reprocessed = ReprocessedEmission(
                grid=grid,
                fesc_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )

        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(reprocessed["escaped"], reprocessed),
            **kwargs,
        )


class EmergentEmission(StellarEmissionModel):
    """An emission model that defines the emergent emission.

    This defines combination of the attenuated and escaped emission components
    to produce the emergent emission.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        dust_curve (AttenuationLaw): The dust curve to use.
        apply_to (EmissionModel): The emission model to apply the dust to.
        label (str): The label for this emission model.
    """

    def __init__(
        self,
        grid,
        dust_curve=None,
        apply_to=None,
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        label="emergent",
        attenuated=None,
        escaped=None,
        **kwargs,
    ):
        """Initialise the EmergentEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            apply_to (EmissionModel): The model to apply the dust to.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            label (str): The label for this emission model.
            attenuated (EmissionModel): The attenuated model to use, if None
                then one will be created.
            escaped (EmissionModel): The escaped model to use, if None then one
                will be created.
            **kwargs: Additional keyword arguments.
        """
        # If apply_to is None then we need to make a model to apply to
        if apply_to is None and attenuated is None:
            apply_to = ReprocessedEmission(
                grid=grid,
                fesc=fesc,
                feac_ly_alpha=fesc_ly_alpha,
                **kwargs,
            )

        # Make an attenuated model if we need one
        if attenuated is None:
            attenuated = AttenuatedEmission(
                grid=grid,
                dust_curve=dust_curve,
                apply_to=apply_to,
                emitter="stellar",
                **kwargs,
            )

        # Do we have an escaped model?
        if escaped is None:
            escaped = attenuated["escaped"]

        StellarEmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(attenuated, escaped),
            fesc=fesc,
            **kwargs,
        )


class TotalEmission(StellarEmissionModel):
    """An emission model that defines the total emission.

    This defines the combination of the emergent and dust emission components
    to produce the total emission.

    This is a child of the EmissionModel class for a full description
    of the parameters see the EmissionModel class.

    Attributes:
        grid (synthesizer.grid.Grid): The grid object to extract from.
        dust_curve (AttenuationLaw): The dust curve to use.
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
        fesc="fesc",
        fesc_ly_alpha="fesc_ly_alpha",
        **kwargs,
    ):
        """Initialise the TotalEmission object.

        Args:
            grid (synthesizer.grid.Grid): The grid object to extract from.
            dust_curve (AttenuationLaw): The dust curve to use.
            dust_emission_model (synthesizer.dust.EmissionModel): The dust
                emission model to use.
            label (str): The label for this emission model.
            fesc (float): The escape fraction of the emission.
            fesc_ly_alpha (float): The escape fraction of Lyman-alpha.
            **kwargs: Additional keyword arguments.
        """
        # Set up models we need to link
        nebular = NebularEmission(
            grid=grid,
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
        escaped = transmitted["escaped"]

        # If a dust emission model has been passed then we need combine
        if dust_emission_model is not None:
            emergent = EmergentEmission(
                grid=grid,
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
                **kwargs,
            )
        else:
            # Otherwise, total == emergent
            StellarEmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(attenuated, escaped),
                **kwargs,
            )
