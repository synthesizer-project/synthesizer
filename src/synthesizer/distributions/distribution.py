"""A submodule defining the base distribution class.

These distribution classes define mathematical distributions which can be
used for sampling values in various contexts within the synthesizer framework.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from unyt import Unit, dimensionless, unyt_array

from synthesizer import exceptions


class Distribution(ABC):
    """An abstract base class for probability distributions.

    This class defines the interface for probability distributions that can be
    sampled from. Subclasses must implement the `sample` method to provide
    an interface for generating random samples according to the specific
    distribution.

    Attributes:
        units (unyt.Unit):
            The unyt units associated with the distribution's samples.
        required_parameters (Tuple[str]):
            A tuple of parameters that the distribution requires. Ensures
            that these parameters are provided before sampling.

    Methods:
        _distribution() -> float:
            Define the probability distribution function, returning a single
            sample value.
        sample(nsamples: int) -> Union[unyt_array, np.ndarray]:
            Generate random samples from the distribution.
        plot(fig: Optional[plt.Figure] = None,
             ax: Optional[plt.Axes] = None,
             show: bool = False, **kwargs) -> Tuple[plt.Figure, plt.Axes]:
            Plot the distribution function.
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
                A tuple of parameters that the distribution requires. Ensures
                that these parameters are provided before sampling.
        """
        # Attach the distribution properties
        self.units = units
        self.required_parameters = required_parameters

    @abstractmethod
    def _distribution(self) -> float:
        """Define the probability distribution function.

        This method should be implemented by subclasses to define the
        specific probability distribution function.

        Returns:
            The (unitless) value of the probability distribution function
            generated using the Distribution's attributes.
        """
        pass

    def sample(
        self,
        nsamples: int,
    ) -> Union[unyt_array, np.ndarray]:
        """Generate random samples from the distribution.

        This method generates random samples according to the defined
        probability distribution function.

        Args:
            nsamples (int):
                The number of samples to generate.
            **kwargs:
                Keyword arguments required by the distribution.

        Returns:
            unyt_array/np.ndarray:
                An array of random samples drawn from the distribution,
                with the appropriate units (or as an array if unitless).
        """
        # Ensure all required parameters are provided
        for param in self.required_parameters:
            if getattr(self, param, None) is None:
                raise exceptions.MissingParameter(
                    f"Missing required parameter '{param}' for sampling "
                    f"from distribution."
                )

        # We have what we need, call the distribution function and attach
        # units if appropriate
        if self.units == dimensionless:
            return np.array([self.distribution() for _ in range(nsamples)])
        return unyt_array(
            [self.distribution() for _ in range(nsamples)],
            self.units,
        )

    @abstractmethod
    def plot(
        self,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
        show: bool = False,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the distribution function.

        This method generates a plot of the distribution function over a
        specified range.

        Args:
            fig (matplotlib.figure.Figure, optional):
                The figure to plot on. If None, a new figure is created.
            ax (matplotlib.axes.Axes, optional):
                The axes to plot on. If None, new axes are created.
            show (bool, optional):
                Whether to display the plot immediately. Defaults to False.
            **kwargs:
                Additional keyword arguments to pass to the plot call.

        Returns:
            fig, ax:
                The figure and axes containing the plot.
        """
        pass
