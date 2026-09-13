"""A submodule containing the definitions of common emission models.

This module contains the definitions of commoon emission models that
can be used "out of the box" to generate spectra from components or as a
foundation to work from when creating more complex models.

Example usage::

    # Create a simple emission model
    model = DustEmission(
        dust_emission_model=BlackBody(50 * K),
    )

    # Generate the spectra
    spectra = stars.get_spectra(model)

"""

from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.parameters import ParameterList


def _each_generator(generator):
    """Iterate over the generators a generator argument stands for.

    A generator argument is usually one generator, but it can also be a
    declaration that the model should be varied over several of them, in which
    case whatever is being set up has to be set up on each.

    Args:
        generator (Generator/ParameterList):
            The generator, or a declaration listing several.

    Returns:
        list:
            The generators.
    """
    if isinstance(generator, ParameterList):
        return list(generator.values)

    return [generator]


class DustEmission(EmissionModel):
    """An emission model that defines the dust emission.

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
        emitter,
        label="dust_emission",
        dust_lum_intrinsic=None,
        dust_lum_attenuated=None,
        grid=None,
        **kwargs,
    ):
        """Initialise the DustEmission object.

        Args:
            dust_emission_model (synthesizer.dust.DustEmissionModel): The dust
                emission model to use.
            emitter (string): The emitter this model is associated with.
            dust_lum_intrinsic (EmissionModel): The intrinsic spectra to use
                when calculating dust luminosity.
            dust_lum_attenuated (EmissionModel): The attenuated spectra to use
                when calculating dust luminosity.
            label (str): The label for this emission model.
            grid (synthesizer.grid.Grid): The grid object.
            **kwargs (dict): Additional keyword arguments to pass to the
                parent class.
        """
        # Attach both models if they are not None to the dust emission
        # generator. The generator can also be a declaration that it should be
        # varied over several generators, in which case each of them needs the
        # same wiring.
        if dust_lum_intrinsic is not None and dust_lum_attenuated is not None:
            for generator in _each_generator(dust_emission_model):
                generator.set_energy_balance(
                    dust_lum_intrinsic, dust_lum_attenuated
                )
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            generator=dust_emission_model,
            emitter=emitter,
            **kwargs,
        )


class AttenuatedEmission(EmissionModel):
    """An emission model that defines the attenuated emission.

    This defines the attenuation of the reprocessed emission by dust.

    This is a child of the EmissionModel class for a full description of the
    parameters see the EmissionModel class.

    Attributes:
        dust_curve (synthesizer.AttenuationLaw): The dust curve to use.
        apply_to (EmissionModel): The emission model to apply the dust to.
        label (str): The label for this emission model.
    """

    def __init__(
        self,
        dust_curve,
        apply_to,
        emitter,
        label="attenuated",
        grid=None,
        **kwargs,
    ):
        """Initialise the AttenuatedEmission object.

        Args:
            dust_curve (AttenuationLaw): The dust curve to use.
            apply_to (EmissionModel): The model to apply the dust to.
            emitter (string): The emitter this model is associated with.
            label (str): The label for this emission model.
            grid (synthesizer.grid.Grid): The grid object.
            **kwargs (dict): Additional keyword arguments to pass to the
                parent class.
        """
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            dust_curve=dust_curve,
            apply_to=apply_to,
            emitter=emitter,
            **kwargs,
        )


class TemplateEmission(EmissionModel):
    """An emission model that uses a template for emission extraction.

    This is a child of the EmisisonModel class, for a full description of the
    parameters see the EmissionModel class.
    """

    def __init__(self, template, emitter, label="template", **kwargs):
        """Initialise the TemplateEmission model.

        Args:
            template (Template):
                The template object containing the AGN emission.
            emitter (str):
                The emitter this model is associated with.
            label (str):
                The label for the model.
            **kwargs (dict):
                Additional keyword arguments to pass to the parent class.

        """
        EmissionModel.__init__(
            self,
            label=label,
            generator=template,
            emitter=emitter,
            **kwargs,
        )
