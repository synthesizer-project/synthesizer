"""A submodule containing the Generator class.

A Generator dictates the process of generating emissions (spectra and lines)
from input parameters, a functional form, and/or a input emission. It provides
a framework that can be extended to create any generator that simply returns
a spectrum or set of lines.

The main interface to a Generator is the `generate_lnu` and `generate_lines`
methods. These will be called by the `EmissionModel` to generate the
emission spectra and lines, respectively when an `EmissionModel` is
set up with a `Generator`.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from synthesizer import exceptions
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer.components import Component
from synthesizer.emission_models import EmissionModel
from synthesizer.emission_models.utils import get_params
from synthesizer.emissions import LineCollection, Sed


class Generator(ABC):
    """An abstract base class for emission generators."""

    def __init__(self, required_params: Tuple = ()) -> None:
        """Initialize the Generator.

        Args:
            required_params (tuple, optional):
                The name of any required parameters needed by the generator
                when generating emissions. These should either be available
                from an emitter, an input emission (Sed/LineCollection), or
                be overridden in the EmissionModel itself. If they are
                missing an exception will be raised.
        """
        # Store the parameters this generator will need
        self._required_params = required_params

    def _extract_params(
        self,
        model: EmissionModel | None,
        emission: Sed | LineCollection | None,
        emitter: Component | BaseGalaxy | None,
    ) -> dict[str, Any]:
        """Extract the required parameters for the generation.

        This method should look for the required parameters in
        model.fixed_parameters, on the emission, and on the emitter (in
        that order of importance). If any of the required parameters are
        missing an exception will be raised.

        Args:
            model (EmissionModel):
                The emission model generating the emission.
            emission (Line/Sed):
                The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.

        Returns:
            dict
                A dictionary containing the required parameters.
        """
        # Extract the parameters (Missing parameters will return None)
        params = get_params(self._required_params, model, emission, emitter)

        # Check if any of the required parameters are missing
        missing_params = [
            param for param, value in params.items() if value is None
        ]
        if len(missing_params) > 0:
            missing_strs = [f"'{s}'" for s in missing_params]
            raise exceptions.MissingAttribute(
                f"{', '.join(missing_strs)} can't be "
                "found on the EmissionModel, emission (Sed/LineCollection), "
                f"or emitter (Stars/BlackHoles/Galaxy) "
                f"(required by {self.__class__.__name__})"
            )

        return params

    @abstractmethod
    def generate_lnu(self, *args, **kwargs):
        """Generate the rest frame spectrum.

        This method should be implemented by subclasses to generate the
        lnu spectrum based on the provided parameters.
        """
        pass

    @abstractmethod
    def generate_lines(self, *args, **kwargs):
        """Generate the rest frame line luminosity and continuum.

        This method should be implemented by subclasses to generate the
        emission lines based on the provided parameters.
        """
        pass
