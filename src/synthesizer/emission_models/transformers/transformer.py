"""A module defining the Transformer abstract base class.

A Transformer defines an object which takes some form of emission and
transforms it in some way. This could include attenuation, scaling,
or any other transformation that can be applied to emission (either lines
or a spectra). For now we only implement and use attenuation child classes.

Any Transformer class should inherit from this class and implement the
required methods.
"""

from abc import ABC, abstractmethod

from synthesizer import exceptions
from synthesizer.emission_models.utils import get_params


class Transformer(ABC):
    """An abstract base class defining the Transformer interface.

    A Transformer defines an object which takes some form of emission and
    transforms it in some way. This could include attenuation, scaling,
    or any other transformation that can be applied to emission (either lines
    or a spectra). For now we only implement and use attenuation child classes.

    Any Transformer class should inherit from this class and implement the
    required methods.

    Everything associated to this class is private and should not be accessed
    by the user. The child classes can present a public interface to the user
    which makes more sense in the specific context of the transformation. We
    only need to define these methods for use behind the scenes when generating
    emissions with an EmissionModel.

    Attributes:
        _required_params (tuple):
            The name of any required parameters needed by the transformer
            when transforming an emission. These should either be
            available from an emitter or from the EmissionModel itself.
            If they are missing an exception will be raised.
    """

    def __init__(self, required_params=()):
        """Initialise the Transformer.

        Args:
            required_params (tuple, optional):
                The name of any required parameters needed by the transformer
                when transforming an emission. These should either be
                available from an emitter or from the EmissionModel itself.
                If they are missing an exception will be raised.
        """
        # Store the parameters this transformer will need
        self._required_params = required_params

    @abstractmethod
    def _transform(self, emission, emitter, model, *args, **kwargs):
        """Transform the emission in some way.

        This method must look for the required parameters in
        model.fixed_parameters, on the emission, and on the emitter (in
        that order of importance). If should then apply the transformation
        using whatever methods are required using the parameters found.

        For example, get_transmission on a dust attenuation transformer would
        require the optical depth (tau_v) and the wavelength grid. The former
        of these could be found as an override on the model, or as an
        attribute on the emitter. The wavelength grid will instead be found
        on the emission object itself. These are then passed to the
        get_transmission method which returns the transmission curve. This
        method should then apply the transmission curve to the emission,
        producing a new emission object which is returned.

        Args:
            emission (Line/Sed):
                The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.
            *args (tuple):
                Any additional arguments required for the transformation.
            **kwargs (dict):
                Any additional keyword arguments required for the
                transformation.

        Returns:
            Emission
                The transformed emission.
        """
        pass

    def _extract_params(self, model, emission, emitter):
        """Extract the required parameters for the transformation.

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
