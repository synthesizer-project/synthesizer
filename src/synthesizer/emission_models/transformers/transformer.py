"""A module defining the Transformer abstract base class.

A Transformer defines an object which takes some form of emission and
transforms it in some way. This could include attenuation, scaling,
or any other transformation that can be applied to emission (either lines
or a spectra). For now we only implement and use attenuation child classes.

Any Transformer class should inherit from this class and implement the
required methods.
"""

from abc import ABC, abstractmethod


class Transformer(ABC):
    """
    An abstract base class defining the Transformer interface.

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
        _transformer_type (str)
            The type of transformer. This is used to identify the
            transformer when applying it to an emission.
    """

    def __init__(self, required_params=()):
        """
        Initialise the Transformer.

        Args:
            required_params (tuple, optional)
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

        Args:
            emission (Line/Sed)
                The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy)
                The object emitting the emission.
            model (EmissionModel)
                The emission model generating the emission.
            args
                Any additional arguments required for the transformation.
            kwargs
                Any additional keyword arguments required for the
                transformation.

        Returns:
            Emission
                The transformed emission.
        """
        pass
