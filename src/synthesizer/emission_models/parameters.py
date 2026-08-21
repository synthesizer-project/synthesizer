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

import copy
import inspect

import numpy as np
from unyt.exceptions import UnitConversionError

from synthesizer import exceptions
from synthesizer.synth_warnings import warn


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
            func (callable): The function to wrap.
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
            model (EmissionModel): The model object.
            emission (Sed/LineCollection): The emission object.
            emitter (Stars/Gas/Galaxy): The emitter object.
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
    duplicated once per value, with each duplicate labelled to say which value
    produced it.

    Those labels are how the resulting emissions are addressed, so every value
    needs one. There are two ways to provide them, and exactly one must be
    used:

    - label_modifier, a printf style format string applied to each value. This
      is the natural choice for numbers: "fesc_%.2f" gives "_fesc_0.10".
    - labels, one name per value. This is the choice for anything a format
      string cannot sensibly render, such as a list of dust curves or of
      ParameterFunctions, whose repr would make for an unusable label.

    Because a ParameterList is never a usable value, encountering one during
    an emission calculation is an error (get_param will complain and tell the
    user to call expand_models first).

    Attributes:
        values (list): The values to vary the parameter over.
        label_modifier (str):
            The format string used to name each variant, or None if explicit
            labels were given instead.
        labels (list):
            The name given to each variant, or None if a format string was
            given instead.
        suffixes (list):
            The suffix appended to a model's label for each value, including
            the leading underscore.
    """

    def __init__(self, values, label_modifier=None, labels=None):
        """Initialise the parameter list.

        Args:
            values (list/tuple/ndarray):
                The values to vary the parameter over. Must contain at least
                one value.
            label_modifier (str):
                A printf style format string used to name each variant (e.g.
                "fesc_%.2f"). Mutually exclusive with labels.
            labels (list):
                A name for each value, used when a format string cannot
                sensibly render them. Mutually exclusive with label_modifier.

        Raises:
            InconsistentArguments
                If no values are given, if neither or both of the naming
                arguments are given, if they are malformed, or if two values
                end up with the same label.
        """
        # Store the values, normalising containers to a list but leaving the
        # values themselves (which may carry units) untouched
        self.values = list(values)

        # We need something to vary over
        if len(self.values) == 0:
            raise exceptions.InconsistentArguments(
                "A ParameterList must contain at least one value."
            )

        # Exactly one way of naming the variants
        if (label_modifier is None) == (labels is None):
            raise exceptions.InconsistentArguments(
                "A ParameterList needs exactly one of label_modifier (a "
                "format string applied to each value, e.g. 'fesc_%.2f') or "
                "labels (a name for each value, for values a format string "
                f"cannot render). Got label_modifier={label_modifier} and "
                f"labels={labels}."
            )

        self.label_modifier = label_modifier
        self.labels = list(labels) if labels is not None else None

        if self.labels is not None:
            self.suffixes = self._suffixes_from_labels()
        else:
            self.suffixes = self._suffixes_from_modifier()

        # Eagerly check the suffixes are unique, which is much friendlier than
        # discovering the collision part way through an expansion. Note that
        # labels are the keys emissions are stored under, so they cannot be
        # allowed to collide.
        if len(set(self.suffixes)) != len(self.suffixes):
            # Report only the collisions themselves, listing every value is
            # useless for a large number of values
            duplicates = sorted(
                {
                    suffix
                    for suffix in self.suffixes
                    if self.suffixes.count(suffix) > 1
                }
            )
            raise exceptions.InconsistentArguments(
                f"{len(self.suffixes)} values gave "
                f"{len(set(self.suffixes))} unique labels. Colliding labels: "
                f"{duplicates[:5]}{'...' if len(duplicates) > 5 else ''}. "
                + (
                    "Use a format string with more precision."
                    if self.label_modifier is not None
                    else "Every label must be distinct."
                )
            )

    def _suffixes_from_labels(self):
        """Return the label suffix for each value, from the given names.

        Returns:
            list:
                The suffixes, including the leading underscore.

        Raises:
            InconsistentArguments
                If the labels don't match the values, or aren't usable names.
        """
        if len(self.labels) != len(self.values):
            raise exceptions.InconsistentArguments(
                f"Got {len(self.labels)} labels for {len(self.values)} "
                "values. A ParameterList needs one label per value."
            )

        for label in self.labels:
            if not isinstance(label, str) or len(label) == 0:
                raise exceptions.InconsistentArguments(
                    f"Labels must be non-empty strings, got {label!r}."
                )

        return ["_" + label for label in self.labels]

    def _suffixes_from_modifier(self):
        """Return the label suffix for each value, from the format string.

        Returns:
            list:
                The suffixes, including the leading underscore.

        Raises:
            InconsistentArguments
                If the format string doesn't include the value, or cannot
                render it.
        """
        # The modifier must actually include the value, otherwise every variant
        # would collide
        if (
            not isinstance(self.label_modifier, str)
            or "%" not in self.label_modifier
        ):
            raise exceptions.InconsistentArguments(
                "label_modifier must be a printf style format string "
                "including the value (e.g. 'fesc_%.2f'), got "
                f"{self.label_modifier}."
            )

        try:
            return [self.suffix(value) for value in self.values]
        except TypeError as e:
            raise exceptions.InconsistentArguments(
                f"The label_modifier '{self.label_modifier}' cannot render "
                f"these values ({e}). Pass labels instead, naming each value, "
                "which is what anything a format string can't render needs."
            ) from None

    def _convert_values(self, convert):
        """Return this list with a check or conversion applied to its values.

        A declaration can be passed to an argument whose units are checked
        when the object is built, e.g. the central wavelength of a dust curve.
        The check belongs on the values rather than on the declaration holding
        them, so it is applied here.

        The labels are left exactly as they were, so naming a variant after
        the value the user wrote does not change because that value was
        converted into different units.

        Args:
            convert (callable):
                Called with each value, returning the value to keep.

        Returns:
            ParameterList:
                A copy of this list holding the converted values.
        """
        converted = copy.copy(self)
        converted.values = [convert(value) for value in self.values]

        return converted

    def items(self):
        """Iterate over the values and the suffix each one produces.

        Returns:
            zip:
                Pairs of (value, suffix).
        """
        return zip(self.values, self.suffixes)

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
        naming = (
            f"label_modifier='{self.label_modifier}'"
            if self.label_modifier is not None
            else f"labels={self.labels}"
        )

        return f"{self.__class__.__name__}({self.values}, {naming})"


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
        n (int): The number of values to sample.
        seed (int):
            The seed for the random number generator. Pass a seed for a
            reproducible set of samples.
        label_modifier (str):
            A printf style format string used to construct the label suffix
            for each variant model.
        units (unyt.Unit):
            The units the sampled values carry, or None for a dimensionless
            parameter.
    """

    # The attributes describing this distribution which are in the same space
    # as the values it samples, and can therefore be stated as quantities
    # rather than as magnitudes
    _value_attrs = ()

    def __init__(
        self,
        n,
        seed=None,
        label_modifier=None,
        labels=None,
        units=None,
    ):
        """Initialise the distribution.

        Args:
            n (int): The number of values to sample from the distribution.
            seed (int):
                The seed for the random number generator. Pass a seed for a
                reproducible set of samples.
            label_modifier (str):
                A printf style format string used to name each variant (e.g.
                "fesc_%.2f"). Mutually exclusive with labels.
            labels (list):
                A name for each of the n samples. Mutually exclusive with
                label_modifier.
            units (unyt.Unit):
                The units the sampled values carry. Pass this when the
                parameter needs units, or state the distribution with
                quantities and leave this alone.

        Raises:
            InconsistentArguments
                If n is not a positive integer, or the variants are not named
                by exactly one of the two arguments.
        """
        # We need to sample at least once
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise exceptions.InconsistentArguments(
                f"n must be a positive integer, got {n}."
            )

        # Exactly one way of naming the variants. The list this realises into
        # does the detailed checking, but catching it here points at the
        # declaration the user actually wrote.
        if (label_modifier is None) == (labels is None):
            raise exceptions.InconsistentArguments(
                "A ParameterDistribution needs exactly one of label_modifier "
                "(a format string applied to each sample) or labels (a name "
                f"for each sample). Got label_modifier={label_modifier} and "
                f"labels={labels}."
            )

        # A label modifier renders the sampled values, so without a seed the
        # labels themselves change from run to run, and with them the keys the
        # emissions are stored under and the group names in any file written
        # from them
        if seed is None and label_modifier is not None:
            warn(
                f"A {type(self).__name__} with no seed will sample different "
                "values every run, so the labels naming its variants will "
                "change too. Pass a seed for labels which stay put."
            )

        self.n = int(n)
        self.seed = seed
        self.label_modifier = label_modifier
        self.labels = list(labels) if labels is not None else None
        self.units = getattr(units, "units", units)

    def _sample(self, rng, n):
        """Sample n values from the distribution.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int): The number of values to sample.

        Returns:
            list:
                The sampled values.
        """
        raise exceptions.UnimplementedFunctionality(
            f"{self.__class__.__name__} does not implement _sample. Use one "
            "of the ParameterDistribution flavours (e.g. "
            "ParameterUniformDist) or define _sample on your own subclass."
        )

    def _adopt_units(self):
        """Reduce the values describing this distribution to magnitudes.

        A distribution can be stated either as magnitudes with the units given
        separately, or as quantities carrying their own units. Both end up as
        magnitudes plus the units the samples will carry, which is what the
        generators need since they strip units when they sample.

        Called by each flavour once it has stored the values describing it.

        Raises:
            InconsistentArguments
                If the values disagree with each other, or with the units the
                distribution was given.
        """
        # Either every value states its units or none of them do. A mixture
        # would mean guessing what the bare ones were measured in.
        carried = {
            attr: hasattr(getattr(self, attr), "units")
            for attr in self._value_attrs
        }
        if len(set(carried.values())) > 1:
            with_units = sorted(a for a, has in carried.items() if has)
            without = sorted(a for a, has in carried.items() if not has)
            raise exceptions.InconsistentArguments(
                f"The values describing a {self.__class__.__name__} must "
                "either all carry units or all be magnitudes with the units "
                f"given by the units argument (got {with_units} with units "
                f"and {without} without)."
            )

        for attr in self._value_attrs:
            value = getattr(self, attr)

            # A magnitude, which the units given (if any) describe
            if not hasattr(value, "units"):
                continue

            # A quantity, so it states the units itself. The first one to do
            # so sets them and the rest are measured against it.
            if self.units is None:
                self.units = value.units

            try:
                setattr(self, attr, float(value.to(self.units).value))
            except UnitConversionError:
                raise exceptions.InconsistentArguments(
                    f"{attr} on a {self.__class__.__name__} is in "
                    f"{value.units}, which does not agree with the "
                    f"{self.units} the rest of the distribution is in."
                ) from None

    def _convert_values(self, convert):
        """Return this distribution with a check applied to its units.

        A distribution states the units of its samples once, so a check on the
        values it will produce is a check on those units. Passing one of them
        through the check both validates it and converts it, and the values
        describing the distribution follow that conversion so the samples land
        where they did before.

        Args:
            convert (callable):
                Called with a value, returning the value to keep.

        Returns:
            ParameterDistribution:
                A copy of this distribution in the checked units.
        """
        converted = copy.copy(self)

        # One of the sampled values, checked and converted. A dimensionless
        # distribution has no units to offer, which is where a distribution
        # passed to an argument needing units is caught.
        one = convert(1.0 if self.units is None else 1.0 * self.units)

        converted.units = getattr(one, "units", None)
        scaling = getattr(one, "value", one)
        for attr in self._value_attrs:
            setattr(converted, attr, getattr(self, attr) * scaling)

        return converted

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
        samples = self._sample(rng, self.n)

        # The generators sample magnitudes, so put the units back on
        if self.units is not None:
            samples = [sample * self.units for sample in samples]

        return ParameterList(
            samples,
            label_modifier=self.label_modifier,
            labels=self.labels,
        )

    def __len__(self):
        """Return the number of values which will be sampled."""
        return self.n

    def __repr__(self):
        """Return a string representation of the ParameterDistribution."""
        return (
            f"{self.__class__.__name__}(n={self.n}, seed={self.seed}, "
            f"units={self.units}, label_modifier='{self.label_modifier}')"
        )


class ParameterUniformDist(ParameterDistribution):
    """A uniform distribution of parameter values.

    Attributes:
        low:
            The lower bound of the distribution (inclusive).
        high:
            The upper bound of the distribution (exclusive).
    """

    # low and high are in the same space as the sampled values
    _value_attrs = ("low", "high")

    def __init__(
        self,
        low,
        high,
        n,
        seed=None,
        label_modifier=None,
        labels=None,
        units=None,
    ):
        """Initialise the uniform distribution.

        Args:
            low:
                The lower bound of the distribution (inclusive).
            high:
                The upper bound of the distribution (exclusive).
            n (int): The number of values to sample.
            seed (int): The seed for the random number generator.
            label_modifier (str):
                A printf style format string used to name each variant.
            labels (list): A name for each sample, instead of a format string.
            units (unyt.Unit):
                The units the samples carry, if the bounds are magnitudes.

        Raises:
            InconsistentArguments
                If low is not less than high.
        """
        ParameterDistribution.__init__(
            self,
            n=n,
            seed=seed,
            label_modifier=label_modifier,
            labels=labels,
            units=units,
        )

        self.low = low
        self.high = high

        # Reduce the bounds to magnitudes before comparing them, so bounds
        # stated in different units are compared in the same ones
        self._adopt_units()

        if self.low >= self.high:
            raise exceptions.InconsistentArguments(
                f"low must be less than high (got low={self.low}, "
                f"high={self.high})."
            )

    def _sample(self, rng, n):
        """Sample n values uniformly between low and high.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int): The number of values to sample.

        Returns:
            list:
                The sampled values.
        """
        return list(rng.uniform(self.low, self.high, n))

    def __repr__(self):
        """Return a string representation of the ParameterUniformDist."""
        return (
            f"{self.__class__.__name__}(low={self.low}, high={self.high}, "
            f"n={self.n}, seed={self.seed}, units={self.units}, "
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
            n (int): The number of values to sample.

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

    # mean and sigma are in the same space as the sampled values
    _value_attrs = ("mean", "sigma")

    def __init__(
        self,
        mean,
        sigma,
        n,
        seed=None,
        label_modifier=None,
        labels=None,
        units=None,
    ):
        """Initialise the normal distribution.

        Args:
            mean:
                The mean of the distribution.
            sigma:
                The standard deviation of the distribution.
            n (int): The number of values to sample.
            seed (int): The seed for the random number generator.
            label_modifier (str):
                A printf style format string used to name each variant.
            labels (list): A name for each sample, instead of a format string.
            units (unyt.Unit):
                The units the samples carry, if mean and sigma are magnitudes.

        Raises:
            InconsistentArguments
                If sigma is not positive.
        """
        ParameterDistribution.__init__(
            self,
            n=n,
            seed=seed,
            label_modifier=label_modifier,
            labels=labels,
            units=units,
        )

        self.mean = mean
        self.sigma = sigma

        self._adopt_units()

        if self.sigma <= 0:
            raise exceptions.InconsistentArguments(
                f"sigma must be greater than zero (got sigma={self.sigma})."
            )

    def _sample(self, rng, n):
        """Sample n values from the normal distribution.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int): The number of values to sample.

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
            f"units={self.units}, label_modifier='{self.label_modifier}')"
        )


class ParameterLogNormalDist(ParameterNormalDist):
    """A log normal distribution of parameter values.

    Values are sampled from a normal distribution in log10 space, where mean
    and sigma are given in log10 space, and returned in linear space.

    Since mean and sigma are in log10 space they cannot carry the units of a
    sample. Give those with the units argument.
    """

    # mean and sigma are in log10 space rather than in the space of the
    # sampled values, so they are always magnitudes
    _value_attrs = ()

    def _sample(self, rng, n):
        """Sample n values from the log normal distribution.

        Args:
            rng (numpy.random.Generator):
                The random number generator to sample with.
            n (int): The number of values to sample.

        Returns:
            list:
                The sampled values, in linear space.
        """
        return list(10 ** rng.normal(self.mean, self.sigma, n))


# The parameter types which declare a variation and thus require an expansion
# before an emission can be generated
VARIATION_TYPES = (ParameterList, ParameterDistribution)


# The attributes a variation can be declared on which are not fixed
# parameters. A transformer or a generator is an object rather than a value, so
# it is held directly on the model rather than in its fixed parameters, but
# varying one is just as reasonable as varying a number.
VARIATION_ATTRS = ("transformer", "generator")


def find_variations(model):
    """Return the variation markers attached to a model.

    A variation can be declared in three places: on a fixed parameter, on the
    transformer or generator the model uses, or on an argument of that
    transformer or generator. The last of these is how the arguments of a dust
    curve or a dust emission model are varied, e.g. the slope of a PowerLaw,
    which belong to the curve rather than to the model.

    All three are reported keyed by the name the expansion should set, so it
    does not have to care which is which. An argument of a transformer or
    generator is keyed by a dotted name, e.g. "transformer.slope".

    Args:
        model (EmissionModel): The model to inspect.

    Returns:
        dict:
            A dictionary of the form {<name>: <marker>} containing every
            ParameterList and ParameterDistribution declared on the model.
    """
    found = {
        param: value
        for param, value in model.fixed_parameters.items()
        if isinstance(value, VARIATION_TYPES)
    }

    for attr in VARIATION_ATTRS:
        value = getattr(model, f"_{attr}", None)

        # The transformer or generator itself is varied
        if isinstance(value, VARIATION_TYPES):
            found[attr] = value
            continue

        # One of its arguments is varied. Every attribute is considered, not
        # only the parameters it declares as required, so that arguments which
        # are never looked up on the model (a reference wavelength, a flag)
        # can be varied too.
        if value is not None:
            for name, arg in vars(value).items():
                if isinstance(arg, VARIATION_TYPES):
                    found[f"{attr}.{name}"] = arg

    return found


def set_variation_value(model, name, value):
    """Set the value a variation resolved to on a model.

    Args:
        model (EmissionModel): The model to set it on.
        name (str):
            The name the variation was declared under, as reported by
            find_variations.
        value:
            The value this variant uses.
    """
    # An argument of a transformer or a generator. The object is copied before
    # the argument is set on it, because one transformer or generator can be
    # shared by several models and setting the argument in place would change
    # it for all of them.
    if "." in name:
        attr, arg = name.split(".", 1)
        varied = copy.copy(getattr(model, f"_{attr}"))
        setattr(varied, arg, value)
        setattr(model, f"_{attr}", varied)

    # A transformer or a generator lives on the model itself; everything else
    # is a fixed parameter
    elif name in VARIATION_ATTRS:
        setattr(model, f"_{name}", value)
    else:
        model.fixed_parameters[name] = value
