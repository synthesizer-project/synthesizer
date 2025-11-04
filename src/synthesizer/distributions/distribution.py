"""A submodule defining the base distribution class.

These distribution classes define mathematical distributions which can be
used for sampling values in various contexts within the synthesizer framework.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from unyt import Unit

from synthesizer import exceptions


class Distribution(ABC):
    """An abstract base class for probability distributions.

    This class defines the interface for probability distributions that can be
    sampled from. Subclasses must implement the `sample` method to provide
    an interface for generating random samples according to the specific
    distribution.

    Methods:
        sample(size: int) -> list:
            Generate random samples from the distribution.

    Example:
        >>> class NormalDistribution(Distribution):
        ...     def __init__(self, mean: float, stddev: float):
        ...         super().__init__(
        ...             units=Unit("unitless"),
        ...             required_parameters=("mean", "stddev"),
        ...         )
        ...         self.mean = mean
        ...         self.stddev = stddev
        ...     def distribution(self, x):
        ...         # Define the normal distribution function here
        ...         pass

        >>> normal_dist = NormalDistribution(mean=0, stddev=1)
        >>> samples = normal_dist.sample(size=1000)

        >>> # Sample with parameters extracted from an object
        >>> normal_dist = NormalDistribution(mean=None, stddev=None)
        >>> stars = Stars(..., mean=10, stddev=3)
        >>> extracted_params = normal_dist.extract_parameters(stars)
        >>> samples = normal_dist.sample(size=1000, **extracted_params)

        >>> # Mix fixed and extracted parameters
        >>> normal_dist_fixed = NormalDistribution(mean=0, stddev=None)
        >>> stars = Stars(..., stddev=2)
        >>> extracted_params = normal_dist_fixed.extract_parameters(stars)
        >>> samples = normal_dist_fixed.sample(size=1000, **extracted_params)

        >>> # Extracting parameters from multiple objects
        >>> normal_dist_multi = NormalDistribution(mean=None, stddev=None)
        >>> stars = Stars(..., mean=5)
        >>> m = EmissionModel(..., stddev=1)
        >>> extracted_params = normal_dist_multi.extract_parameters(stars, m)

        >>> # Extracting using list of objects with priority
        >>> normal_dist = NormalDistribution(mean=None, stddev=None)
        >>> s = Stars(..., mean=5)
        >>> g = Gas(..., mean=10, stddev=2)
        >>> extracted_params = normal_dist_priority.extract_parameters(s, g)
        >>> samples = normal_dist.sample(size=1000, **extracted_params)
    """

    def __init__(
        self,
        units: Unit,
        required_parameters: Tuple[str],
    ) -> None:
        """Initialise the base Distribution class.

        Args:
            units (unyt.Unit):
                The unyt units associated with the distribution's samples.
            required_parameters (Tuple[str]):
                A tuple of parameters that the distribution requires. This
                ensures that any caller provides these parameters when
                sampling but can also be used to extract parameters from an
                emitter object.
        """
        # Attach the distribution properties
        self.units = units
        self.required_parameters = required_parameters

    @abstractmethod
    def distribution(self, **kwargs):
        """Define the probability distribution function.

        This method should be implemented by subclasses to define the
        specific probability distribution function.

        Args:
            **kwargs:
                Keyword arguments required by the distribution.

        Returns:
            The (unitless) value of the probability distribution function.
        """
        pass

    def sample(
        self,
        size: int,
        **kwargs,
    ) -> list:
        """Generate random samples from the distribution.

        This method generates random samples according to the defined
        probability distribution function.

        Args:
            size (int):
                The number of samples to generate.
            **kwargs:
                Keyword arguments required by the distribution.

        Returns:
            unyt_array/np.ndarray:
                An array of random samples drawn from the distribution,
                with the appropriate units (or as an array if unitless).
        """
        # If we have fixed parameters, override the kwargs with them
        for param, value in self.fixed_parameters.items():
            kwargs[param] = value

        # Ensure all required parameters are provided
        for param in self.required_parameters:
            if param not in kwargs or kwargs[param] is None:
                raise exceptions.MissingParameter(
                    f"Missing required parameter '{param}' for sampling "
                    f"from distribution."
                )

        # We have what we need, call the distribution function
        return self.distribution(size, **kwargs)

    def extract_parameters(self, *objs) -> dict:
        """Extract required parameters from an object or provided parameters.

        This method attempts to extract the required parameters for the
        distribution from the provided object. If a parameter is not found
        in the object, it falls back to the provided parameters.

        Note that sample will raise any error for missing parameters, here we
        just want to return None.


        Args:
            *objs:
                A list of objects to extract parameters from. The method will
                prioritize earlier objects in the list.

        Returns:
            dict:
                A dictionary of extracted parameters.
        """
        # Define the container for the extracted parameters
        extracted_params = {}

        # Loop over the required parameters
        for param in self.required_parameters:
            # Try to extract from the provided objects
            for obj in objs:
                if obj is not None and hasattr(obj, param):
                    extracted_params[param] = getattr(obj, param)
                    break
            else:
                extracted_params[param] = None

        return extracted_params
