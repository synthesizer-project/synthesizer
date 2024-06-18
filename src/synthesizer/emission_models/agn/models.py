"""A submodule containing the definitions of common agn emisison models.

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

from unyt import Hz, unyt_array

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.sed import Sed


class Template(EmissionModel):
    """
    Use a template for the emission model.

    The model is different to all other emission models in that it scales a
    template by bolometric luminosity.

    Attributes:
        sed (Sed)
            The template spectra for the AGN.
        normalisation (unyt_quantity)
            The normalisation for the spectra. In reality this is the
            bolometric luminosity.
    """

    def __init__(
        self,
        label="template",
        filename=None,
        lam=None,
        lnu=None,
        fesc=0.0,
        **kwargs,
    ):
        """
        Initialise the Template.

        Args:
            label (str)
                The label for the model.
            filename (str)
                The filename (including full path) to a file containing the
                template. The file should contain two columns with wavelength
                and luminosity (lnu).
            lam (array)
                Wavelength array.
            lnu (array)
                Luminosity array.
            fesc (float)
                The escape fraction of the AGN.
            **kwargs

        """
        EmissionModel.__init__(
            self,
            grid=None,
            label=label,
            fesc=fesc,
            **kwargs,
        )

        # Ensure we have been given units
        if lam is not None and not isinstance(lam, unyt_array):
            raise exceptions.MissingUnits("lam must be provided with units")
        if lnu is not None and not isinstance(lnu, unyt_array):
            raise exceptions.MissingUnits("lam must be provided with units")

        if filename:
            raise exceptions.UnimplementedFunctionality(
                "Not yet implemented! Feel free to implement and raise a "
                "pull request. Guidance for contributing can be found at "
                "https://github.com/flaresimulations/synthesizer/blob/main/"
                "docs/CONTRIBUTING.md"
            )

        if lam is not None and lnu is not None:
            # initialise a synthesizer Sed object
            self.sed = Sed(lam=lam, lnu=lnu)

            # normalise
            # TODO: add a method to Sed that does this.
            self.normalisation = self.sed.measure_bolometric_luminosity()
            self.sed.lnu /= self.normalisation.value

        else:
            raise exceptions.MissingArgument(
                "Either a filename or both lam and lnu must be provided!"
            )

    def _scale_template(self, bolometric_luminosity):
        """
        Calculate the blackhole spectra by scaling the template.

        Args:
            bolometric_luminosity (float)
                The bolometric luminosity of the blackhole(s) for scaling.

        """
        # Ensure we have units for safety
        if bolometric_luminosity is not None and not isinstance(
            bolometric_luminosity, unyt_array
        ):
            raise exceptions.MissingUnits(
                "bolometric luminosity must be provided with units"
            )

        return (
            bolometric_luminosity.to(self.sed.lnu.units * Hz).value
            * self.sed
            * (1 - self.fesc)
        )
