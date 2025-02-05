"""A module for dynamically returning attributes with and without units.

The Units class below acts as a container for the unit system.

The Quantity is a descriptor object which uses the Units class to attach units
to attributes of a class. The Quantity descriptor can be used to attach units
to class attributes.

Example defintion:

    class Foo:

        bar = Quantity("spatial")

        def __init__(self, bar):
            self.bar = bar

Example usage:

    foo = Foo(bar)

    bar_with_units = foo.bar
    bar_no_units = foo._bar

"""

import os
from functools import wraps

import yaml
from unyt import (
    Unit,
    dimensionless,
    unyt_array,
    unyt_quantity,
)
from unyt.exceptions import UnitConversionError

from synthesizer import exceptions
from synthesizer.warnings import warn

# Define the path to your YAML file
FILE_PATH = os.path.join(os.path.dirname(__file__), "default_units.yml")


def _load_and_convert_unit_categories() -> dict:
    """
    Load the default unit system from a YAML file.

    This loads all the strings stored in the YAML file and converts them into
    unyt Unit objects.

    One thing to note is this process will treat Msun as a first class unit not
    a compound unit in the galactic base system. This is because unyt does not
    support compound units in the base system, but means we don't need to
    worry about converting between the two base systems.

    Returns:
        dict
            A dictionary of unyt Unit objects
    """
    # Load the yaml file
    data: dict
    with open(FILE_PATH, "r") as f:
        data = yaml.safe_load(f)

    # Extract the unit categories dictionary
    unit_categories: dict = data["UnitCategories"]

    # Convert the string units to unyt Unit objects
    converted: dict = {
        key: Unit(value["unit"]) for key, value in unit_categories.items()
    }

    return converted


# Get the default units system (this can be modified by the user).
# NOTE: This module-level variable will be initialized only once on import
UNIT_CATEGORIES = _load_and_convert_unit_categories()


class DefaultUnits:
    """
    The DefaultUnits class is a container for the default unit system.

    This class is used to store the default unit system for Synthesizer. It
    contains all the unit categories defined in the default unit system.

    Attributes:
        ... (unyt.unit_object.Unit)
            The unit for each category defined in the default unit system.
    """

    def __init__(self):
        """
        Initialise the default unit system.

        This will extract all the unit categories from the previously loaded
        YAML file and attach them as attributes to the DefaultUnits object.
        """
        for key, unit in UNIT_CATEGORIES.items():
            setattr(self, key, unit)

    def __getitem__(self, name):
        """Get a unit from the default unit system."""
        if hasattr(self, name):
            return getattr(self, name)
        raise KeyError(f"Unit category {name} not found.")

    def __setitem__(self, name, value):
        """Set a unit in the default unit system."""
        setattr(self, name, value)

    def items(self):
        """Return the items of the default unit system."""
        return UNIT_CATEGORIES.items()

    def keys(self):
        """Return the keys of the default unit system."""
        return UNIT_CATEGORIES.keys()

    def values(self):
        """Return the values of the default unit system."""
        return UNIT_CATEGORIES.values()

    def __iter__(self):
        """Iterate over the default unit system."""
        return iter(UNIT_CATEGORIES)

    def __len__(self):
        """Return the length of the default unit system."""
        return len(UNIT_CATEGORIES)

    def __type__(self):
        """Return the type of the default unit system."""
        return type(UNIT_CATEGORIES)

    def __str__(self):
        """
        Return a string representation of the default unit system.

        Returns:
            table (str)
                A string representation of the LineCollection object.
        """
        # Local import to avoid cyclic imports
        from synthesizer.utils import TableFormatter

        # Intialise the table formatter
        formatter = TableFormatter(self)

        return (
            formatter.get_table("Default Units")
            .replace("Attribute", "Category ")
            .replace("Value", "Unit ")
        )


# Instantiate the default unit system
default_units = DefaultUnits()


class UnitSingleton(type):
    """
    A metaclass used to ensure singleton behaviour for the Units class.

    A singleton design pattern is used to ensure that only one instance of the
    class can exist at any one time.
    """

    # Define a private dictionary to store instances of UnitSingleton
    _instances = {}

    def __call__(cls, new_units=None, force=False):
        """
        Make an instance of the child class or return the original.

        When a new instance is made, this method is called.

        Unless forced to redefine Units (highly inadvisable), the original
        instance is returned giving it a new reference to the original
        instance.

        If a new unit system is passed and one already exists and warning is
        printed and the original is returned.

        Returns:
            Units
                A new instance of Units if one does not exist (or a new one
                is forced), or the first instance of Units if one does exist.
        """
        # Are we forcing an update?... I hope not
        if force:
            cls._instances[cls] = super(UnitSingleton, cls).__call__(
                new_units, force
            )

        # Print a warning if an instance exists and arguments have been passed
        elif cls in cls._instances and new_units is not None:
            warn(
                "Units are already set. \nAny modified units will "
                "not take effect. \nUnits should be configured before "
                "running anything else... \nbut you could (and "
                "shouldn't) force it: Units(new_units_dict, force=True)."
            )

        # If we don't already have an instance the dictionary will be empty
        if cls not in cls._instances:
            cls._instances[cls] = super(UnitSingleton, cls).__call__(
                new_units, force
            )

        return cls._instances[cls]


class Units(metaclass=UnitSingleton):
    """
    Holds the definition of the internal unit system using unyt.

    Units is a Singleton, meaning there can only ever be one. Each time a new
    instance is instantiated the original will be returned. This enforces a
    consistent unit system is used in a single top level namespace.

    All default attributes are hardcoded but these can be modified by
    instantiating the original Units instance with a dictionary of units of
    the form {"variable": unyt.unit}. This must be done before any calculations
    have been performed, changing the unit system will not retroactively
    convert computed quantities! In fact, if any quantities have been
    calculated the original default Units object will have already been
    instantiated, thus the default Units will be returned regardless
    of the modifications dictionary due to the rules of a Singleton
    metaclass. The user can force an update but BE WARNED this is
    dangerous and should be avoided.

    Attributes:
        ... (unyt.unit_object.Unit)
            The unit for each category defined in the default unit system or
            any modifications made by the user.
    """

    def __init__(self, units=None, force=False):
        """
        Intialise the Units object.

        Args:
            units (dict)
                A dictionary containing any modifications to the default unit
                system. This can either modify the unit categories
                defined in the default unit system, e.g.:

                    units = {"wavelength": microns,
                             "smoothing_lengths": kpc,
                             "lam": m}

                Or, if desired, individual attributes can be modified
                explicitly, e.g.:

                    units = {"coordinates": kpc,
                             "smoothing_lengths": kpc,
                             "lam": m}
            force (bool)
                A flag for whether to force an update of the Units object.
        """
        # Define a dictionary to hold the unit system. We'll use this if we
        # need to dump the current unit system to the default units yaml file
        self._units = {}

        # First off we need to attach the default unit system
        # to the Units object
        for key, unit in default_units.items():
            setattr(self, key, unit)
            self._units[key] = unit

        # Do we have any modifications to the default unit system
        if units is not None:
            print("Redefining unit system:")

            # Loop over new units
            for key in units:
                print("%s:" % key, units[key])

                # If we are modifying an existing unit makes sure it is
                # compatible with the default unit system (we can't do this
                # for new units as we don't know what they are but other
                # errors down stream will soon alert the user to their mistake)
                if hasattr(self, key):
                    if getattr(self, key).is_equivalent(units[key]):
                        raise exceptions.IncorrectUnits(
                            f"Unit {units[key]} for {key} is not "
                            "compatible with the expected units "
                            f"of {getattr(self, key)}."
                        )

                # Set the new unit
                setattr(self, key, units[key])
                self._units[key] = units[key]

    def __str__(self):
        """
        Return a string representation of the default unit system.

        Returns:
            table (str)
                A string representation of the LineCollection object.
        """
        # Local import to avoid cyclic imports
        from synthesizer.utils import TableFormatter

        # Intialise the table formatter
        formatter = TableFormatter(self)

        return (
            formatter.get_table("Unit System")
            .replace("Attribute", "Category ")
            .replace("Value", "Unit ")
        )

    def overwrite_defaults_yaml(self):
        """
        Permenantly overwrite the default unit system with the current one.

        This method is used to overwrite the default unit system with the
        current one. This is to be used when the user wants to permenantly
        modify the default unit system with the current one.
        """
        #
        # If we haven't already made a copy of the original default units
        # yaml file then do so now
        original_path = os.path.join(
            os.path.dirname(__file__), "original_units.yml"
        )
        original_units = {}
        original_units["UnitCategories"] = UNIT_CATEGORIES
        if not os.path.exists(original_path):
            with open(original_path, "w") as f:
                yaml.dump(original_units, f)

        # Write the current unit system to the default units yaml file
        new_units = {}
        new_units["UnitCategories"] = self._units
        with open(FILE_PATH, "w") as f:
            yaml.dump(new_units, f)

    def reset_defaults_yaml(self):
        """
        Reset the default unit system to the original one.

        This will overwrite the default_units.yml file with the
        original_units.yml file.
        """
        # Check the original units file exists
        original_path = os.path.join(
            os.path.dirname(__file__), "original_units.yml"
        )
        if not os.path.exists(original_path):
            raise FileNotFoundError("Original units file not found.")

        # Copy the original units file to the default units file
        with open(original_path, "r") as f:
            original_units = yaml.safe_load(f)
        with open(FILE_PATH, "w") as f:
            yaml.dump(original_units, f)

        # Reload the default unit system
        global UNIT_CATEGORIES
        UNIT_CATEGORIES = _load_and_convert_unit_categories()

        # Reset the Units object
        self.__init__(force=True)


class Quantity:
    """
    A decriptor class controlling dynamicly associated attribute units.

    Provides the ability to associate attribute values on an object with unyt
    units defined in the global unit system (Units).

    Attributes:
        unit (unyt.unit_object.Unit)
            The unit for this Quantity from the global unit system.
        public_name (str)
            The name of the class variable containing Quantity. Used the user
            wants values with a unit returned.
        private_name (str)
            The name of the class variable with a leading underscore. Used the
            mostly internally for (or when the user wants) values without a
            unit returned.
    """

    def __init__(self, category):
        """
        Initialise the Quantity.

        This will extract the unit from the global unit system based on the
        passed category. Note that this unit can be overriden if the user
        specified a unit override for the attribute associated with this
        Quantity.

        Args:
            category (str)
                The category of the attribute. This is used to get the unit
                from the global unit system.
        """
        # Get the unit based on the category passed at initialisation. This
        # can be overriden in __set_name__ if the user set a specific unit for
        # the attribute associated with this Quantity.
        self.unit = getattr(Units(), category)

    def __set_name__(self, owner, name):
        """
        Store the name of the class variable when it is assigned a Quantity.

        When a class variable is assigned a Quantity() this method is called
        extracting the name of the class variable, assigning it to attributes
        for use when returning values with or without units.
        """
        self.public_name = name
        self.private_name = "_" + name

        # Do we have a unit override for this attribute?
        if hasattr(Units(), name):
            self.unit = getattr(Units(), name)

    def __get__(self, obj, type=None):
        """
        Return the value of the attribute with units.

        When referencing an attribute with its public_name this method is
        called. It handles the returning of the values stored in the
        private_name variable with units.

        The value is stored under the private_name variable on the instance
        of the class. If we instead used the private name directly we would
        bypass the Quantity descriptor and return the value without units.

        If the value is None then None is returned regardless.

        Returns:
            unyt_array/unyt_quantity/None
                The value with units attached or None if value is None.
        """
        value = getattr(obj, self.private_name)

        # If we have an uninitialised attribute avoid the multiplying NoneType
        # error and just return None
        if value is None:
            return None

        return value * self.unit

    def __set__(self, obj, value):
        """
        Set the value of the attribute with units.

        When setting a Quantity variable this method is called, firstly the
        value is converted to the expected units. Once converted the value is
        stored on the instance of the class under the private_name variable.

        Args:
            obj (arbitrary)
                The object contain the Quantity attribute that we are storing
                value in.
            value (array-like/float/int)
                The value to store in the attribute.
        """
        # Do we need to perform a unit conversion? If not we assume value
        # is already in the default unit system
        if isinstance(value, (unyt_quantity, unyt_array)):
            if value.units != self.unit and value.units != dimensionless:
                value = value.to(self.unit).value
            else:
                value = value.value

        # Set the attribute
        setattr(obj, self.private_name, value)


def has_units(x):
    """
    Check whether the passed variable has units.

    This will check the argument is a unyt_quanity or unyt_array.

    Args:
        x (generic variable)
            The variables to check.

    Returns:
        bool
            True if the variable has units, False otherwise.
    """
    # Do the check
    if isinstance(x, (unyt_array, unyt_quantity)):
        return True

    return False


def _check_arg(units, name, value):
    """
    Check the units of an argument.

    This function is used to check the units of an argument passed to
    a function. If the units are missing or incompatible an error will be
    raised. If the units don't match the defined units in units then the values
    will be converted to the correct units.

    Args:
        units (dict)
            The dictionary of units defined in the accepts decorator.
        name (str)
            The name of the argument.
        value (generic variable)
            The value of the argument.

    Returns:
        generic variable
            The value of the argument with the correct units.

    Raises:
        MissingUnits
            If the argument is missing units.
        IncorrectUnits
            If the argument has incompatible units.
    """
    # Early exit if the argument isn't in the units dictionary
    if name not in units:
        return value

    # If the argument is None just skip it, its an optional argument that
    # hasn't been passed... or the user has somehow managed to pass None
    # which is sufficently weird to cause an obvious error elsewhere
    if value is None:
        return None

    # Handle the unyt_array/unyt_quantity cases
    if isinstance(value, (unyt_array, unyt_quantity)):
        # We know we have units but are they compatible?
        if value.units != units[name]:
            try:
                return value.to(units[name])
            except UnitConversionError:
                raise exceptions.IncorrectUnits(
                    f"{name} passed with incompatible units. "
                    f"Expected {units[name]} (or equivalent) but "
                    f"got {value.units}."
                )
        else:
            # Otherwise the value is in the expected units
            return value

    # Handle the list/tuple case
    elif isinstance(value, (list, tuple)):
        # Ensure the value is mutable
        converted = list(value)

        # Loop over the elements of the argument checking
        # they have units and those units are compatible
        for j, v in enumerate(value):
            # Are we missing units on the passed argument?
            if not has_units(v):
                raise exceptions.MissingUnits(
                    f"{name} is missing units! Expected"
                    f"to be in {units[name]} "
                    "(or equivalent)."
                )

            # Convert to the expected units
            elif v.units != units[name]:
                try:
                    converted[j] = _check_arg(units, name, v)
                except UnitConversionError:
                    raise exceptions.IncorrectUnits(
                        f"{name}@{j} passed with "
                        "incompatible units. "
                        f"Expected {units[name][j]}"
                        " (or equivalent) but "
                        f"got {v.units}."
                    )
            else:
                # Otherwise the value is in the expected units
                converted[j] = v

        return converted

    # If None of these were true then we haven't got units.
    raise exceptions.MissingUnits(
        f"{name} is missing units! Expected to "
        f"be in {units[name]} (or equivalent)."
    )


def accepts(**units):
    """
    Check arguments passed to the wrapped function have compatible units.

    This decorator will cross check any of the arguments passed to the wrapped
    function with the units defined in this decorators kwargs. If units are
    not compatible or are missing an error will be raised. If the units don't
    match the defined units in units then the values will be converted to the
    correct units.

    This is inspired by the accepts decorator in the unyt package, but includes
    Synthesizer specific errors and conversion functionality.

    Args:
        **units
            The keyword arguments defined with this decorator. Each takes the
            form of argument=unit_for_argument. In reality this is a
            dictionary of the form {"variable": unyt.unit}.

    Returns:
        function
            The wrapped function.
    """

    def check_accepts(func):
        """
        Check arguments passed to the wrapped function have compatible units.

        Args:
            func (function)
                The function to be wrapped.

        Returns:
            function
                The wrapped function.
        """
        arg_names = func.__code__.co_varnames

        @wraps(func)
        def wrapped(*args, **kwargs):
            """
            Handle all the arguments passed to the wrapped function.

            Args:
                *args
                    The arguments passed to the wrapped function.
                **kwargs
                    The keyword arguments passed to the wrapped function.

            Returns:
                The result of the wrapped function.
            """
            # Convert the positional arguments to a list (it must be mutable
            # for what comes next)
            args = list(args)

            # Check the positional arguments
            for i, (name, value) in enumerate(zip(arg_names, args)):
                args[i] = _check_arg(units, name, value)

            # Check the keyword arguments
            for name, value in kwargs.items():
                kwargs[name] = _check_arg(units, name, value)

            return func(*args, **kwargs)

        return wrapped

    return check_accepts
