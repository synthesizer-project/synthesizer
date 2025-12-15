"""A submodule defining the base class for emission generators."""

from abc import ABC, abstractmethod

from synthesizer import exceptions
from synthesizer.emission_models.utils import get_params


class Generator(ABC):
    """An abstract base class defining the Generator interface.

    A Generator defines an object which generates emissions based on
    some input parameters. This could include generation of spectra/lines
    based on component properties or existing emissions.

    Any Generator class should inherit from this class and implement the
    required methods.

    Everything associated to this class is private and should not be accessed
    by the user. The child classes can present a public interface to the user
    which makes more sense in the specific context of the generation. We only
    need to define these methods for use behind the scenes when generating
    emissions with an EmissionModel.

    Attributes:
        _required_params (tuple):
            The name of any required parameters needed by the generator
            when generating an emission. These should either be
            available from an emitter or from the EmissionModel itself.
            If they are missing an exception will be raised.
        _required_emissions (tuple):
            The name of any required emissions needed by the generator
            when generating an emission. These should be available from
            the emitter in either "spectra", "particle_spectra", "lines",
            or "particle_lines" (which is dictated by the current method
            called). If they are missing an exception will be raised.
    """

    def __init__(self, required_params=(), required_emissions=()):
        """Initialise the Generator.

        Args:
            required_params (tuple, optional):
                The name of any required parameters needed by the generator
                when generating an emission. These should either be
                available from an emitter or from the EmissionModel itself.
                If they are missing an exception will be raised.
            required_emissions (tuple, optional):
                The name of any required emissions needed by the generator
                when generating an emission. These should be available from
                the emitter in either "spectra", "particle_spectra", "lines",
                or "particle_lines" (which is dictated by the current method
                called). If they are missing an exception will be raised.
        """
        # Store the parameters this generator will need
        self._required_params = required_params
        self._required_emissions = required_emissions

    @abstractmethod
    def _generate_spectra(self, lams, emitter, model, *args, **kwargs):
        """Generate spectra based on the emitter and model.

        This method must look for the required parameters in
        model.fixed_parameters, on the emitter and on the generator itself (in
        that order of importance). If any of the required parameters are
        missing an exception will be raised. This can be achieved by using the
        base class's _extract_params method.

        For example, a generator that generates a blackbody spectrum based on
        component temperature would need to look for the temperature parameter
        on the emitter or model and then use this to generate the spectrum.

        If an input emission is needed then the same behaviour should be
        implemented using the _extract_spectra method.

        Args:
            lams (unyt_array):
                The wavelengths at which to generate the spectra.
            emitter (Emitter):
                The emitter for which to generate the spectra.
            model (EmissionModel):
                The emission model calling this generator.
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            spectra (dict):
                A dictionary of generated spectra.
        """
        raise NotImplementedError(
            "_generate_spectra method must be implemented in child classes."
        )

    @abstractmethod
    def _generate_lines(self, lams, emitter, model, *args, **kwargs):
        """Generate lines based on the emitter and model.

        This method must look for the required parameters in
        model.fixed_parameters, on the emitter and on the generator itself (in
        that order of importance). If any of the required parameters are
        missing an exception will be raised. This can be achieved by using the
        base class's _extract_params method.

        For example, a generator that generates emission lines based on
        component metallicity would need to look for the metallicity parameter
        on the emitter or model and then use this to generate the lines.

        If an input emission is needed then the same behaviour should be
        implemented using the _extract_lines method.

        Args:
            lams (unyt_array):
                The wavelengths at which to generate the lines.
            emitter (Emitter):
                The emitter for which to generate the lines.
            model (EmissionModel):
                The emission model calling this generator.
            *args:
                Additional positional arguments.
            **kwargs:
                Additional keyword arguments.

        Returns:
            lines (dict):
                A dictionary of generated lines.
        """
        raise NotImplementedError(
            "_generate_lines method must be implemented in child classes."
        )

    def _extract_params(self, model, emitter):
        """Extract the required parameters for the generation.

        This method should look for the required parameters in
        model.fixed_parameters, on the emitter and on the generator itself (in
        that order of importance). If any of the required parameters are
        missing an exception will be raised.

        Args:
            model (EmissionModel):
                The emission model generating the emission.
            emitter (Emitter):
                The object emitting the emission.

        Returns:
            dict
                A dictionary containing the required parameters.
        """
        # Extract the parameters (Missing parameters will return None)
        params = get_params(
            self._required_params,
            model,
            None,
            emitter,
            obj=self,
        )

        # Check if any of the required parameters are missing
        missing_params = [
            param for param, value in params.items() if value is None
        ]
        if len(missing_params) > 0:
            missing_strs = [f"'{s}'" for s in missing_params]
            raise exceptions.MissingAttribute(
                f"{', '.join(missing_strs)} can't be "
                "found on the EmissionModel or emitter "
                f"(required by {self.__class__.__name__})"
            )

        return params

    def _extract_emissions(self, emission_dict, per_particle):
        """Extract the required emissions for the generation.

        This method should look for the required emissions in
        emitter.spectra, or emitter.particle_spectra (depending on whether the
        model has per_particle set to True or False). If any of the required
        emissions are missing an exception will be raised.

        Args:
            emission_dict (dict):
                The dictionary containing all emissions generated so far. These
                can be spectra, particle_spectra, lines, or particle_lines.
            per_particle (bool):
                Whether to extract from particle_spectra or spectra.

        Returns:
            dict
                A dictionary containing the required emissions.
        """
        # Extract the emissions (Missing emissions will return None)
        emissions = {}
        for emission_name in self._required_emissions:
            emissions[emission_name] = emission_dict.get(emission_name, None)

        # Check if any of the required emissions are missing
        missing_emissions = [
            name for name, value in emissions.items() if value is None
        ]
        if len(missing_emissions) > 0:
            missing_strs = [f"'{s}'" for s in missing_emissions]
            raise exceptions.MissingAttribute(
                f"{', '.join(missing_strs)} can't be "
                "found on the emitter "
                f"(required by {self.__class__.__name__})"
            )

        return emissions
