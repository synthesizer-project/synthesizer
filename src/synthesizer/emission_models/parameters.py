"""A submodule containing the parameter types used by emission models.

Emission model parameters can be passed as simple values, as strings (which
point at an attribute to read from the emitter), or as one of the parameter
types defined in this module. The parameter types defined here describe
parameters which need some extra machinery before a value can be used:

- ParameterFunction: a value computed at run time from other parameters.
- ParameterList: a list of values, each of which produces a variant model.
- ParameterDistribution: a distribution which is sampled to produce a list of
  values, each of which produces a variant model. This is an abstract base
  class, use one of its flavours (e.g. ParameterUniformDist).

Example usage::

    def compute_metallicity(mass, age, fixed_param):
        return (mass * 0.01) + (age * 0.001) + fixed_param

    param_func = ParameterFunction(
        func=compute_metallicity,
        sets="metallicity",
        func_args=["mass", "age", "fixed_param"],
    )

    model = EmissionModel(
        label="custom_model",
        grid=grid,
        fixed_param=0.02,
        metallicity=param_func,
    )

    # Vary the escape fraction over 3 values. This produces nothing until
    # expand_models is called, at which point the model (and everything
    # depending on it) is duplicated once per value.
    varied = EmissionModel(
        label="transmitted",
        grid=grid,
        extract="transmitted",
        fesc=ParameterList([0.0, 0.1, 0.5], label_modifier="fesc_%.2f"),
    )
    expanded = varied.expand_models()
"""

from __future__ import annotations

import inspect

import numpy as np

from synthesizer import exceptions


class ParameterFunction:
    """A class for wrapping functions that compute parameters for emitters.

    This class can be used to wrap functions which take emitter attributes
    as inputs and return a computed parameter value or array of values. This
    class is designed as a dependency injection mechanism to be passed to
    EmissionModel arguments that require dynamic parameter computation from
    an emitter. As such, this is mostly designed for internal use within the
    Synthesizer package, but it can also be used by an experienced user to
    create custom parameter functions.

    Any function wrapped by this class must:
        - Follow this signature: func(**kwargs) -> value
        - Return a single value or numpy/unyt array of values. If an array is
          returned, it must be the same shape as arrays on the emitter (i.e.
          nstar in length for per star properties etc.).
        - Have kwargs which are either attributes of the emitter object or
          fixed parameters on an EmissionModel.
        - Have kwargs which are all defined in the "func_args" list
          (set during initialization).

    Example:
        def compute_metallicity(mass, age, fixed_param):
            # Compute metallicity based on mass, age, and a fixed parameter
            return (mass * 0.01) + (age * 0.001) + fixed_param

        param_func = ParameterFunction(
            func=compute_metallicity,
            func_args=['mass', 'age', 'fixed_param']
        )

        # Define an emission model that fixes 'fixed_param' to 0.02
        model = EmissionModel(
            label='custom_model',
            fixed_param=0.02,
            grid=grid,
            metallicity_param=param_func,
        )

        # Later... call get spectra on an emitter which will use the function
        # to compute metallicity dynamically.
        emitter.get_spectra(model)

        # And you can see the cached value to was used
        print(emitter.model_param_cache['custom_model']['metallicity_param'])
    """

    def __init__(self, func: callable, sets: str, func_args: list) -> None:
        """Initialize the function wrapper.

        This will attach the function and set the list of argument names
        that the function takes ready for later extraction.

        Args:
            func (callable):
                The function to wrap.
            sets (str):
                A string indicating the attribute on the emitter that this
                function sets.
            func_args (list):
                A list of argument names that the function takes. These must
                correspond to attributes on the emitter or fixed parameters
                on an EmissionModel.

        Raises:
            ValueError:
                If func is not callable.
        """
        if not callable(func):
            raise ValueError("func must be a callable function.")

        self.func = func
        self.func_args = func_args
        self.sets = sets

        # Ensure the function signature matches the func_args
        sig = inspect.signature(func)
        for arg in func_args:
            if arg not in sig.parameters:
                raise exceptions.InconsistentArguments(
                    f"Found func_arg '{arg}' on ParameterFunction which is "
                    "not an argument of the wrapped function "
                    f"'{func.__name__}'."
                )
        for param in sig.parameters:
            if param not in func_args:
                raise exceptions.InconsistentArguments(
                    f"Found argument '{param}' on the wrapped function "
                    f"'{func.__name__}' which is not in the func_args list "
                    "of the ParameterFunction."
                )

    def __call__(self, model, emission, emitter, obj=None):
        """Call the wrapped function with parameters extracted from objects.

        This will extract the required parameters from the model, emission,
        emitter, or optional object and call the wrapped function with those
        parameters.

        Args:
            model (EmissionModel):
                The model object.
            emission (Sed/LineCollection):
                The emission object.
            emitter (Stars/Gas/Galaxy):
                The emitter object.
            obj (object, optional):
                An optional additional object to look for parameters on last.

        Returns:
            value:
                The value returned by the wrapped function.
        """
        # Import the parameter machinery here to avoid a circular import at
        # module load time (utils imports this module for its isinstance
        # checks).
        from synthesizer.emission_models.utils import cache_param, get_param

        # Extract the required parameters
        func_kwargs = {}
        for arg in self.func_args:
            func_kwargs[arg] = get_param(
                arg,
                model,
                emission,
                emitter,
                obj,
            )

        # Call the function with the extracted parameters
        try:
            val = self.func(**func_kwargs)
        except Exception as e:
            raise exceptions.ParameterFunctionError(
                f"Error calling ParameterFunction "
                f"'{self.func.__name__}': {str(e)}"
            ) from e

        # Cache the computed value on the emitter for later use (if provided)
        if emitter is not None:
            cache_param(
                param=self.sets,
                emitter=emitter,
                model_label=model.label,
                value=val,
            )

        return val

    def __repr__(self):
        """Return a string representation of the ParameterFunction.

        Returns:
            str:
                A string representation of the ParameterFunction.
        """
        return (
            f"ParameterFunction({self.func.__name__}, "
            f"sets='{self.sets}', "
            f"args={self.func_args})"
        )


class ParameterList:
    """A list of parameter values, each producing a variant emission model.

    Unlike the other parameter types, a ParameterList does not describe a
    single value to use during an emission calculation. Instead it declares
    that the model should be varied over a set of values. This variation is
    only realised when EmissionModel.expand_models is called, at which point
    the model carrying the list (and every model which depends on it) is
    duplicated once per value, with each duplicate labelled using the
    label_modifier.

    Because a ParameterList is never a usable value, encountering one during
    an emission calculation is an error (get_param will complain and tell the
    user to call expand_models first).

    Attributes:
        values (list):
            The values to vary the parameter over.
        label_modifier (str):
            A printf style format string used to construct the label suffix
            for each variant model, e.g. "fesc_%.2f" will produce suffixes
            like "_fesc_0.10" (expand_models prepends the underscore).
    """

    def __init__(self, values, label_modifier):
        """Initialise the parameter list.

        Args:
            values (list/tuple/ndarray):
                The values to vary the parameter over. Must contain at least
                one value.
            label_modifier (str):
                A printf style format string used to construct the label
                suffix for each variant model (e.g. "fesc_%.2f").

        Raises:
            InconsistentArguments
                If no values are given, if the label_modifier is not a format
                string, or if two values produce the same label suffix.
        """
        # Store the values, normalising containers to a list but leaving the
        # values themselves (which may carry units) untouched
        self.values = list(values)

        # We need something to vary over
        if len(self.values) == 0:
            raise exceptions.InconsistentArguments(
                "A ParameterList must contain at least one value."
            )

        # The label modifier must actually include the value, otherwise every
        # variant would collide
        if not isinstance(label_modifier, str) or "%" not in label_modifier:
            raise exceptions.InconsistentArguments(
                "label_modifier must be a printf style format string "
                "including the value (e.g. 'fesc_%.2f'), got "
                f"{label_modifier}."
            )

        self.label_modifier = label_modifier

        # Eagerly check the suffixes are unique, this is much friendlier than
        # discovering the collision part way through an expansion. Note that
        # labels are the keys emissions are stored under, so they cannot be
        # allowed to collide.
        suffixes = [self.suffix(value) for value in self.values]
        if len(set(suffixes)) != len(suffixes):
            # Report only the collisions themselves, listing every value is
            # useless for a large number of values
            duplicates = sorted(
                {suffix for suffix in suffixes if suffixes.count(suffix) > 1}
            )
            raise exceptions.InconsistentArguments(
                f"The label_modifier '{label_modifier}' does not produce a "
                f"unique label for every value ({len(suffixes)} values gave "
                f"{len(set(suffixes))} unique labels). Colliding labels: "
                f"{duplicates[:5]}{'...' if len(duplicates) > 5 else ''}. Use "
                "a format string with more precision."
            )

    def suffix(self, value):
        """Return the label suffix for a single value.

        Args:
            value:
                The value to construct a suffix for.

        Returns:
            str:
                The label suffix, including the leading underscore.
        """
        # Strip any units before formatting, printf formatting of a unyt
        # quantity would otherwise include the unit string
        if hasattr(value, "value") and not isinstance(value, str):
            value = value.value

        return "_" + (self.label_modifier % value)

    def __len__(self):
        """Return the number of values in the list."""
        return len(self.values)

    def __iter__(self):
        """Iterate over the values in the list."""
        return iter(self.values)

    def __repr__(self):
        """Return a string representation of the ParameterList."""
        return (
            f"{self.__class__.__name__}({self.values}, "
            f"label_modifier='{self.label_modifier}')"
        )


class ParameterDistribution:
    """A distribution of parameter values, sampled to produce variant models.

    This is the base class for the distribution flavours (e.g.
    ParameterUniformDist, ParameterNormalDist). It behaves exactly like a
    ParameterList except that the values are sampled from a distribution
    rather than stated explicitly.

    Sampling happens once, at the start of expansion, via realise(). This
    matters: if the distribution were sampled per variant then a model
    downstream of another varied model would draw different values in each
    branch, rather than sharing one set of samples.

    Note that this samples one scalar per variant model. It does NOT sample a
    value per particle. Use a ParameterFunction for per particle
    stochasticity.

    Attributes:
        n (int):
            The number of values to sample.
        seed (int):
            The seed for the random number generator. Pass a seed for a
            reproducible set of samples.
        label_modifier (str):
            A printf style format string used to construct the label suffix
            for each variant model.
    """

    def __init__(self, n, seed=None, label_modifier=None):
        """Initialise the distribution.

        Args:
            n (int):
                The number of values to sample from the distribution.
            seed (int):
                The seed for the random number generator. Pass a seed for a
                reproducible set of samples.
            label_modifier (str):
                A printf style format string used to construct the label
                suffix for each variant model (e.g. "fesc_%.2f"). Required.

        Raises:
            InconsistentArguments
                If n is not a positive integer or the label_modifier is
                missing or not a format string.
        """
        # We need to sample at least once
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise exceptions.InconsistentArguments(
                f"n must be a positive integer, got {n}."
            )

        # The label modifier must actually include the value, otherwise every
        # variant would collide
        if not isinstance(label_modifier, str) or "%" not in label_modifier:
            raise exceptions.InconsistentArguments(
                "label_modifier must be a printf style format string "
                "including the value (e.g. 'fesc_%.2f'), got "
                f"{label_modifier}."
            )

        self.n = int(n)
        self.seed = seed
        self.label_modifier = label_modifier

    def _sample(self, rng, n):
        """Sample n values from the distribution.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int):
                The number of values to sample.

        Returns:
            list:
                The sampled values.
        """
        raise exceptions.UnimplementedFunctionality(
            f"{self.__class__.__name__} does not implement _sample. Use one "
            "of the ParameterDistribution flavours (e.g. "
            "ParameterUniformDist) or define _sample on your own subclass."
        )

    def realise(self):
        """Sample the distribution and return the values as a ParameterList.

        This is called once at the start of an expansion, converting every
        distribution in the tree into a concrete list of values before any
        models are duplicated.

        Returns:
            ParameterList:
                The sampled values, ready to be expanded.
        """
        rng = np.random.default_rng(self.seed)
        return ParameterList(
            self._sample(rng, self.n),
            label_modifier=self.label_modifier,
        )

    def __len__(self):
        """Return the number of values which will be sampled."""
        return self.n

    def __repr__(self):
        """Return a string representation of the ParameterDistribution."""
        return (
            f"{self.__class__.__name__}(n={self.n}, seed={self.seed}, "
            f"label_modifier='{self.label_modifier}')"
        )


class ParameterUniformDist(ParameterDistribution):
    """A uniform distribution of parameter values.

    Attributes:
        low:
            The lower bound of the distribution (inclusive).
        high:
            The upper bound of the distribution (exclusive).
    """

    def __init__(self, low, high, n, seed=None, label_modifier=None):
        """Initialise the uniform distribution.

        Args:
            low:
                The lower bound of the distribution (inclusive).
            high:
                The upper bound of the distribution (exclusive).
            n (int):
                The number of values to sample.
            seed (int):
                The seed for the random number generator.
            label_modifier (str):
                A printf style format string used to construct the label
                suffix for each variant model.

        Raises:
            InconsistentArguments
                If low is not less than high.
        """
        ParameterDistribution.__init__(
            self,
            n=n,
            seed=seed,
            label_modifier=label_modifier,
        )

        if low >= high:
            raise exceptions.InconsistentArguments(
                f"low must be less than high (got low={low}, high={high})."
            )

        self.low = low
        self.high = high

    def _sample(self, rng, n):
        """Sample n values uniformly between low and high.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int):
                The number of values to sample.

        Returns:
            list:
                The sampled values.
        """
        return list(rng.uniform(self.low, self.high, n))

    def __repr__(self):
        """Return a string representation of the ParameterUniformDist."""
        return (
            f"{self.__class__.__name__}(low={self.low}, high={self.high}, "
            f"n={self.n}, seed={self.seed}, "
            f"label_modifier='{self.label_modifier}')"
        )


class ParameterLogUniformDist(ParameterUniformDist):
    """A log uniform distribution of parameter values.

    Values are sampled uniformly in log10 space between low and high, which
    are given in linear space.
    """

    def _sample(self, rng, n):
        """Sample n values log uniformly between low and high.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int):
                The number of values to sample.

        Returns:
            list:
                The sampled values, in linear space.
        """
        if self.low <= 0:
            raise exceptions.InconsistentArguments(
                "low must be greater than zero for a log uniform "
                f"distribution (got low={self.low})."
            )

        return list(
            10
            ** rng.uniform(
                np.log10(self.low),
                np.log10(self.high),
                n,
            )
        )


class ParameterNormalDist(ParameterDistribution):
    """A normal distribution of parameter values.

    Attributes:
        mean:
            The mean of the distribution.
        sigma:
            The standard deviation of the distribution.
    """

    def __init__(self, mean, sigma, n, seed=None, label_modifier=None):
        """Initialise the normal distribution.

        Args:
            mean:
                The mean of the distribution.
            sigma:
                The standard deviation of the distribution.
            n (int):
                The number of values to sample.
            seed (int):
                The seed for the random number generator.
            label_modifier (str):
                A printf style format string used to construct the label
                suffix for each variant model.

        Raises:
            InconsistentArguments
                If sigma is not positive.
        """
        ParameterDistribution.__init__(
            self,
            n=n,
            seed=seed,
            label_modifier=label_modifier,
        )

        if sigma <= 0:
            raise exceptions.InconsistentArguments(
                f"sigma must be greater than zero (got sigma={sigma})."
            )

        self.mean = mean
        self.sigma = sigma

    def _sample(self, rng, n):
        """Sample n values from the normal distribution.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int):
                The number of values to sample.

        Returns:
            list:
                The sampled values.
        """
        return list(rng.normal(self.mean, self.sigma, n))

    def __repr__(self):
        """Return a string representation of the ParameterNormalDist."""
        return (
            f"{self.__class__.__name__}(mean={self.mean}, "
            f"sigma={self.sigma}, n={self.n}, seed={self.seed}, "
            f"label_modifier='{self.label_modifier}')"
        )


class ParameterLogNormalDist(ParameterNormalDist):
    """A log normal distribution of parameter values.

    Values are sampled from a normal distribution in log10 space, where mean
    and sigma are given in log10 space, and returned in linear space.
    """

    def _sample(self, rng, n):
        """Sample n values from the log normal distribution.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int):
                The number of values to sample.

        Returns:
            list:
                The sampled values, in linear space.
        """
        return list(10 ** rng.normal(self.mean, self.sigma, n))


# The parameter types which declare a variation and thus require an expansion
# before an emission can be generated
VARIATION_TYPES = (ParameterList, ParameterDistribution)


def find_variations(model):
    """Return the variation markers attached to a model.

    Args:
        model (EmissionModel):
            The model to inspect.

    Returns:
        dict:
            A dictionary of the form {<param_name>: <marker>} containing every
            ParameterList/ParameterDistribution in the model's fixed
            parameters.
    """
    return {
        param: value
        for param, value in model.fixed_parameters.items()
        if isinstance(value, VARIATION_TYPES)
    }
