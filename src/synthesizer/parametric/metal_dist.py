"""A submodule for creating and manipulating metallicity distributions.

NOTE: This module is imported as ZDist in parametric.__init__ enabling the
      syntax shown below.

Example usage:

    from synthesizer.parametric import ZDist

    print(ZDist.parametrisations)

    metal_dist = ZDist.DeltaConstant(...)
    metal_dist = ZDist.Normal(...)
    metal_dist = ZDist.LogNormal(...)

Any parameter can instead be a function of the stars' age, called with the
age (a unyt_quantity or unyt_array in yr), e.g. for an enrichment history:

    metal_dist = ZDist.DeltaConstant(metallicity=lambda age: ...)

    metal_dist.get_dist_weight(metals)

"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf

from synthesizer import exceptions

# The number of points the numerical CDF integrates the distribution over
NUMERICAL_CDF_POINTS = 8192

# Define a list of the available parametrisations
parametrisations = (
    "DeltaConstant",
    "Normal",
    "LogNormal",
)


class Common:
    """The parent class for all ZDist parametrisations.

    Attributes:
        name (str):
           The name of this ZDist. This is set by the child and encodes
           the type of the metallicity distribution. Possible values are
           defined in parametrisations above.
        parameters (dict):
            A dictionary containing the parameters of the model.
    """

    def __init__(self, name, **kwargs):
        """Initialise the parent.

        Args:
            name (str):
                The name of this ZDist. This is set by the child and encodes
                the type of the metallicity distribution. Possible values are
                defined in parametrisations above.
            **kwargs (dict):
                A dictionary containing the parameters of the model.
        """
        # Set the name string
        self.name = name

        # Store the model parameters (defined as kwargs)
        self.parameters = kwargs

    @property
    def depends_on_age(self):
        """Whether any parameter is a function of the stars' age.

        Returns:
            bool:
                True if any parameter is callable.
        """
        return any(callable(value) for value in self.parameters.values())

    def at_age(self, age):
        """Get this distribution with its parameters evaluated at an age.

        Args:
            age (unyt_quantity):
                The age of the stars.

        Returns:
            Common:
                A distribution of the same type with fixed parameters.
        """
        return type(self)(
            **{
                name: value(age) if callable(value) else value
                for name, value in self.parameters.items()
            }
        )

    def __str__(self):
        """Print basic summary of the metallicity distribution."""
        pstr = ""
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF PARAMETERISED METAL ENRICHMENT HISTORY" + "\n"
        pstr += str(self.__class__) + "\n"
        for parameter_name, parameter_value in self.parameters.items():
            pstr += f"{parameter_name}: {parameter_value}" + "\n"
        pstr += "-" * 10 + "\n"
        return pstr

    def _weight(self, metal):
        """Prototype for child defined distribution functions."""
        raise exceptions.UnimplementedFunctionality(
            "This should never be called from the parent."
            " How did you get here!?"
        )

    def get_dist_weight(self, metal):
        """Return the weight of the bin/s defined by metal.

        Args:
            metal (float/np.ndarray of float):
                The metallicity bin/s to evaluate.

        Returns:
            float:
                The weight at metal or each value in metal.
        """
        # If we have been handed an array we need to loop
        if isinstance(metal, (np.ndarray, list)):
            return np.array([self._weight(z) for z in metal])

        return self._weight(metal)

    def get_cdf(self, metals):
        """Get the weight between a metallicity of zero and each metallicity.

        Children with a closed form override this, the default integrates
        the distribution numerically on a dense linear grid.

        Args:
            metals (np.ndarray of float):
                The (linear) metallicities at which to evaluate the CDF.

        Returns:
            np.ndarray of float:
                The integrated weight up to each metallicity.
        """
        metals = np.asarray(metals, dtype=np.float64)
        top = np.max(metals, initial=0.0)
        if top <= 0.0:
            return np.zeros_like(metals)
        fine = np.unique(
            np.concatenate(
                (np.linspace(0.0, top, NUMERICAL_CDF_POINTS), metals.ravel())
            )
        )
        weights = np.asarray(self.get_dist_weight(fine), dtype=np.float64)
        cdf = cumulative_trapezoid(weights, x=fine, initial=0.0)
        return np.interp(metals, fine, cdf)

    def get_bin_weights(self, edges):
        """Get the weight of each metallicity bin.

        Args:
            edges (np.ndarray of float):
                The (linear) metallicity bin edges.

        Returns:
            np.ndarray of float:
                The integrated weight in each bin.
        """
        return np.diff(self.get_cdf(edges))


class DeltaConstant(Common):
    """A single metallicity "distribution".

    Attributes:
        metallicity_ (float):
            The single (linear) metallicity for all stellar mass in the
            distribution.
        log10metallicity_ (float):
            The log base 10 of metallicity.
    """

    def __init__(self, metallicity=None, log10metallicity=None):
        """Initialise the metallicity distribution and parent.

        Either metallicity or log10metallicity must be provided.

        Args:
            metallicity (float):
                The single (linear) metallicity for all stellar mass in the
                distribution.
            log10metallicity (float):
                The log base 10 of metallicity.
        """
        # We need one metallicity definition
        if metallicity is None and log10metallicity is None:
            raise exceptions.InconsistentArguments(
                "Either metallicity or log10metallicity must be provided."
            )

        # Instantiate the parent
        Common.__init__(
            self,
            name="DeltaConstant",
            metallicity=metallicity,
            log10metallicity=log10metallicity,
        )

        # Handled wether we've been passed logged or linear metallicity and
        # set the attributes accordingly (a function of age is evaluated
        # when the population is binned, see Common.at_age)
        if self.depends_on_age:
            self.metallicity_ = None
            self.log10metallicity_ = None
        elif metallicity is not None:
            self.metallicity_ = metallicity
            self.log10metallicity_ = np.log10(self.metallicity_)
        else:
            self.log10metallicity_ = log10metallicity
            self.metallicity_ = 10**self.log10metallicity_

    def get_metallicities(self, ages):
        """Get the metallicity of stars formed at each of a set of ages.

        Args:
            ages (unyt_array):
                The ages of the stars (any shape).

        Returns:
            np.ndarray of float:
                The (linear) metallicity at each age.
        """
        metal = self.parameters["metallicity"]
        log10metal = self.parameters["log10metallicity"]
        if metal is None:
            value = log10metal(ages) if callable(log10metal) else log10metal
            value = 10 ** _to_float(value)
        else:
            value = _to_float(metal(ages) if callable(metal) else metal)
        return np.broadcast_to(value, np.shape(ages))

    def get_metallicity(self, *args):
        """Return the single metallicity.

        NOTE: DeltaConstant is a special case where get_dist_weight is not
        applicable because a single metallicity must always be rather than a
        weight.

        Returns:
            float
                The metallicity.
        """
        # Check for bad behaviour
        if len(args) > 0:
            raise exceptions.InconsistentArguments(
                "A DeltaConstant metallicity distribution takes no arguments."
                " It can only return a single value and should not be passed"
                "any arguments."
            )

        return self.metallicity_

    def get_log10_metallicity(self, *args):
        """Return the log base 10 of the single metallicity.

        NOTE: DeltaConstant is a special case where get_dist_weight is not
        applicable because a single metallicity must always be rather than a
        weight.

        Returns:
            float
                The log10(metallicity).
        """
        # Check for bad behaviour
        if len(args) > 0:
            raise exceptions.InconsistentArguments(
                "A DeltaConstant metallicity distribution takes no arguments."
                " It can only return a single value and should not be passed"
                "any arguments."
            )

        return self.log10metallicity_


class Normal(Common):
    """A normally distributed metallicity distribution.

    Attributes:
        mean (float):
            The mean of the normal distribution.
        sigma (float):
            The standard deviation of the normal distribution.
    """

    def __init__(self, mean, sigma):
        """Initialise the metallicity distribution and parent.

        Either metallicity or log10metallicity must be provided.

        Args:
            mean (float):
                The mean of the normal distribution.
            sigma (float):
                The standard deviation of the normal distribution.
        """
        # Instantiate the parent
        Common.__init__(
            self,
            name="Normal",
            mean=mean,
            sigma=sigma,
        )

        # Define this models parameters
        self.mean = mean
        self.sigma = sigma

    def _weight(self, metal):
        """Return the distribution at a metallicity.

        Args:
            metal (float):
                The (linear) metallicity at which to evaluate the distribution.

        Returns:
            float
                The weight of the metallicity distribution at metal.
        """
        norm = 1 / (self.sigma * np.sqrt(2 * np.pi))
        exponent = ((metal - self.mean) / self.sigma) ** 2
        return norm * np.exp(-0.5 * exponent)

    def get_cdf(self, metals):
        """Get the weight between a metallicity of zero and each metallicity.

        Args:
            metals (np.ndarray of float):
                The (linear) metallicities at which to evaluate the CDF.

        Returns:
            np.ndarray of float:
                The integrated weight up to each metallicity.
        """
        metals = np.maximum(np.asarray(metals, dtype=np.float64), 0.0)
        mean = float(self.mean)
        scale = float(self.sigma) * np.sqrt(2.0)
        return 0.5 * (erf((metals - mean) / scale) - erf(-mean / scale))


class LogNormal(Common):
    """A log-normal metallicity distribution.

    Attributes:
        log10metallicity (float):
            The mean of log10(metallicity), i.e. the median metallicity.
        sigma (float):
            The standard deviation of log10(metallicity) in dex.
    """

    def __init__(self, log10metallicity, sigma):
        """Initialise the metallicity distribution and parent.

        Args:
            log10metallicity (float):
                The mean of log10(metallicity), i.e. the median metallicity.
            sigma (float):
                The standard deviation of log10(metallicity) in dex.
        """
        Common.__init__(
            self,
            name="LogNormal",
            log10metallicity=log10metallicity,
            sigma=sigma,
        )
        self.log10metallicity = log10metallicity
        self.sigma = sigma

    def _weight(self, metal):
        """Return the distribution (per unit linear metallicity).

        Args:
            metal (float):
                The (linear) metallicity at which to evaluate the distribution.

        Returns:
            float
                The weight of the metallicity distribution at metal.
        """
        if metal <= 0:
            return 0.0
        x = (np.log10(metal) - self.log10metallicity) / self.sigma
        return np.exp(-0.5 * x**2) / (
            metal * np.log(10) * self.sigma * np.sqrt(2 * np.pi)
        )

    def get_cdf(self, metals):
        """Get the weight between a metallicity of zero and each metallicity.

        Args:
            metals (np.ndarray of float):
                The (linear) metallicities at which to evaluate the CDF.

        Returns:
            np.ndarray of float:
                The integrated weight up to each metallicity.
        """
        metals = np.asarray(metals, dtype=np.float64)
        with np.errstate(divide="ignore"):
            x = (np.log10(metals) - float(self.log10metallicity)) / (
                float(self.sigma) * np.sqrt(2.0)
            )
        return np.where(metals > 0, 0.5 * (1.0 + erf(x)), 0.0)


def _to_float(value):
    """Convert a (possibly dimensionless unyt) value to floats.

    Args:
        value (float/np.ndarray/unyt_array):
            The value.

    Returns:
        np.ndarray of float:
            The value without units.
    """
    if hasattr(value, "units"):
        value = value.to("dimensionless").ndview
    return np.asarray(value, dtype=np.float64)
