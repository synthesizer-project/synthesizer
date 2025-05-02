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
import shutil
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
from synthesizer.synth_warnings import warn

# Define the path to your YAML file
FILE_PATH = os.path.join(os.path.dirname(__file__), "default_units.yml")


def _load_and_convert_unit_categories() -> dict:
    """Load the default unit system from a YAML file.

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


def unit_is_compatible(value, unit):
    """Check if two values have compatible units.

    This function checks that a unyt_quantity or unyt_array or another Unit is
    compatible with a unit, i.e. it has the same dimensions.
    If they are not compatible, it raises an exception.

    This could also be done wrapping a conversion attempt in a try/except
    block but this is more efficient as it avoids the overhead of
    unyt's conversion system.

    I might have missed a method in unyt for this but I couldn't find one.

    Args:
        value (unyt_quantity/unyt_array/Unit):
            The value to check.
        unit (Unit):
            The unit to check against.

    Returns:
        bool
            True if the values have compatible units, False otherwise.
    """
    # Handle the unyt_array/unyt_quantity cases
    if isinstance(value, (unyt_quantity, unyt_array)):
        return value.units.dimensions == unit.dimensions

    # Handle the Unit case
    elif isinstance(value, Unit):
        return value.dimensions == unit.dimensions

    # If we get here then we didn't get two unyt quantities or arrays
    raise exceptions.InconsistentArguments(
        "Can only check values with units for compatibility, "
        f"not {type(value)} and {type(unit)}."
    )


class DefaultUnits:
    """The DefaultUnits class is a container for the default unit system.

    This class is used to store the default unit system for Synthesizer. It
    contains all the unit categories defined in the default unit system.

    Attributes:
        ... (unyt.unit_object.Unit)
            The unit for each category defined in the default unit system.
    """

    def __init__(self):
        """Initialise the default unit system.

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
        """Return a string representation of the default unit system.

        Returns:
            table (str):
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
    """A metaclass used to ensure singleton behaviour for the Units class.

    A singleton design pattern is used to ensure that only one instance of the
    class can exist at any one time.
    """

    # Define a private dictionary to store instances of UnitSingleton
    _instances = {}

    def __call__(cls, new_units=None, force=False):
        """Make an instance of the child class or return the original.

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
                "Units are already set. Any modified units will "
                "not take effect. Units should be configured before "
                "running anything else... but you could (and "
                "shouldn't) force it: Units(new_units_dict, force=True)."
            )

        # If we don't already have an instance the dictionary will be empty
        if cls not in cls._instances:
            cls._instances[cls] = super(UnitSingleton, cls).__call__(
                new_units, force
            )

        return cls._instances[cls]


class Units(metaclass=UnitSingleton):
    """Holds the definition of the internal unit system using unyt.

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
        """Intialise the Units object.

        Args:
            units (dict):
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
            force (bool):
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
                    if getattr(self, key).dimensions == units[key]:
                        raise exceptions.IncorrectUnits(
                            f"Unit {units[key]} for {key} is not "
                            "compatible with the expected units "
                            f"of {getattr(self, key)}."
                        )

                # Set the new unit
                setattr(self, key, units[key])
                self._units[key] = units[key]

    def __str__(self):
        """Return a string representation of the default unit system.

        Returns:
            table (str):
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

    def _preserve_orig_units(self):
        """Write out the original unit system to a yaml file.

        This makes sure we can always reverse the unit system back to the
        original state.
        """
        # Get the original units file path
        original_path = os.path.join(
            os.path.dirname(__file__), "original_units.yml"
        )

        # If the original file already exists then we don't need to do anything
        if os.path.exists(original_path):
            return

        # Make a copy of the original units file
        shutil.copy(FILE_PATH, original_path)

        print(f"Original unit system has been preserved at {original_path}.")

    def overwrite_defaults_yaml(self):
        """Permenantly overwrite the default unit system with the current one.

        This method is used to overwrite the default unit system with the
        current one. This is to be used when the user wants to permenantly
        modify the default unit system with the current one.
        """
        # If we haven't already made a copy of the original default units
        # yaml file then do so now
        self._preserve_orig_units()

        # Contstruct the dictionary to write out
        new_units = {}
        new_units["UnitCategories"] = {}
        for key, unit in self._units.items():
            new_units["UnitCategories"][key] = {"unit": str(unit)}

        # Write the current unit system to the default units yaml file
        with open(FILE_PATH, "w") as f:
            yaml.dump(new_units, f)

        print(f"Default unit system has been updated at {FILE_PATH}.")

    def reset_defaults_yaml(self):
        """Reset the default unit system to the original one.

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
        shutil.copy(original_path, FILE_PATH)

        # Remove the original units file since we don't need it anymore
        os.remove(original_path)

        # Reload the default unit system
        global UNIT_CATEGORIES
        UNIT_CATEGORIES = _load_and_convert_unit_categories()

        # Remove all units from the Units object
        for key in self._units:
            delattr(self, key)

        # Reset the Units object
        self.__init__(force=True)

        print(f"Default unit system has been reset to {FILE_PATH}.")


class Quantity:
    """A decriptor class controlling dynamicly associated attribute units.

    Provides the ability to associate attribute values on an object with unyt
    units defined in the global unit system (Units).

    Attributes:
        unit (unyt.unit_object.Unit)
            The unit for this Quantity from the global unit system.
        public_name (str):
            The name of the class variable containing Quantity. Used the user
            wants values with a unit returned.
        private_name (str):
            The name of the class variable with a leading underscore. Used the
            mostly internally for (or when the user wants) values without a
            unit returned.
    """

    def __init__(self, category):
        """Initialise the Quantity.

        This will extract the unit from the global unit system based on the
        passed category. Note that this unit can be overriden if the user
        specified a unit override for the attribute associated with this
        Quantity.

        Args:
            category (str):
                The category of the attribute. This is used to get the unit
                from the global unit system.
        """
        # Get the unit based on the category passed at initialisation. This
        # can be overriden in __set_name__ if the user set a specific unit for
        # the attribute associated with this Quantity.
        self.unit = getattr(Units(), category)

    def __set_name__(self, owner, name):
        """Store the name of the class variable when it is assigned a Quantity.

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
        """Return the value of the attribute with units.

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
        """Set the value of the attribute with units.

        When setting a Quantity variable this method is called, firstly the
        value is converted to the expected units. Once converted the value is
        stored on the instance of the class under the private_name variable.

        Args:
            obj (Any):
                The object contain the Quantity attribute that we are storing
                value in.
            value (array-like/float/int):
                The value to store in the attribute.
        """
        # Do we need to perform a unit conversion? If not we assume value
        # is already in the default unit system
        if isinstance(value, (unyt_quantity, unyt_array)):
            if value.units != self.unit and value.units != dimensionless:
                value = unyt_to_ndview(value, self.unit)
            else:
                value = value.ndview

        # Set the attribute
        setattr(obj, self.private_name, value)


def has_units(x):
    """Check whether the passed variable has units.

    This will check the argument is a unyt_quanity or unyt_array.

    Args:
        x (generic variable):
            The variables to check.

    Returns:
        bool
            True if the variable has units, False otherwise.
    """
    # Do the check
    if isinstance(x, (unyt_array, unyt_quantity)):
        return True

    return False


def unyt_to_ndview(arr, unit=None):
    """Extract the underlying data from a `unyt_array` or `unyt_quantity`.

    An ndview is a pointer to the underlying data of a `unyt_array` or
    `unyt_quantity`.

    This is a helper function to enable the extraction of the underlying data
    from a `unyt_array` or `unyt_quantity` WITHOUT making a copy of the data.
    This is possible with the `ndview` property on a `unyt_array` or
    `unyt_quantity`, however, this is not implemented with an inplace unit
    conversion.

    This function can either be used to extract the underlying data in the
    existing units, or to convert inplace to a new unit and then return the
    view (an operation not implemented in unyt to date, as far as I can tell).

    Args:
        arr (unyt_array/unyt_quantity): The unyt_array or unyt_quantity to
            extract the data from.
        unit (unyt.unit_object.Unit): The unit to convert to. If None, the
            existing unit is used. If the unit is not compatible with the
            existing unit, an error will be raised.

    Returns:
        np.ndarray: The underlying data as a numpy array WITHOUT doing a copy.

    Raises:
        UnitConversionError: If the unit is not compatible with the existing
            unit.
    """
    # If we don't have a unit then just return the ndview
    if unit is None:
        return arr.ndview

    # If the units are the same then just return the ndview
    if arr.units == unit:
        return arr.ndview

    # Ok, we need to do a conversion. We'll do this inplace and then
    # return the ndview
    # NOTE: for some reason this method of conversion can lead to very small
    # precision differences vs the to, to_value (etc.) methods. In reality
    # these diffences are negligable but they can lead to exact comparisons
    # failing. This is fine as long as np.isclose/np.allclose is used to check
    # for equality.
    arr.convert_to_units(unit)
    return arr.ndview


def _raise_or_convert(expected_unit, name, value):
    """Ensure we have been passed compatible units and convert if needed.

    Args:
        expected_unit (unyt.Unit/list of unyt.Unit):
            The expected unit for the value.
        name (str):
            The name of the variable being checked (only used for error
            messages).
        value (Any):
            The value to check.

    Returns:
        Any:
            The value with the expected unit.
    """
    # Handle the unyt_array/unyt_quantity cases
    if isinstance(value, (unyt_array, unyt_quantity)):
        # We know we have units but are they compatible?
        if value.units != expected_unit:
            try:
                value.convert_to_units(expected_unit)
            except UnitConversionError:
                raise exceptions.IncorrectUnits(
                    f"{name} passed with incompatible units. "
                    f"Expected {expected_unit} (or equivalent) but "
                    f"got {value.units}."
                )
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
                    f"to be in {expected_unit} "
                    "(or equivalent)."
                )

            # Convert to the expected units
            elif v.units != expected_unit:
                try:
                    converted[j] = _raise_or_convert(expected_unit, name, v)
                except UnitConversionError:
                    raise exceptions.IncorrectUnits(
                        f"{name}@{j} passed with "
                        "incompatible units. "
                        f"Expected {expected_unit[j]}"
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
        f"be in {expected_unit} (or equivalent)."
    )


def _check_arg(units, name, value):
    """Check the units of an argument.

    This function is used to check the units of an argument passed to
    a function. If the units are missing or incompatible an error will be
    raised. If the units don't match the defined units in units then the values
    will be converted to the correct units.

    Args:
        units (dict):
            The dictionary of units defined in the accepts decorator.
        name (str):
            The name of the argument.
        value (generic variable):
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

    # Unpack the units from the units dictionary
    expected_units = units[name]

    # We have two cases now, either we have a single unit and the check is
    # trivial or we have a list of units and we need to check each one
    if isinstance(expected_units, (list, tuple)):
        for i, unit in enumerate(expected_units):
            # Try each unit conversion and capture the error to raise a
            # more informative error message for this situation
            try:
                return _raise_or_convert(unit, name, value)
            except (UnitConversionError, exceptions.IncorrectUnits):
                continue  # we'll raise below

        # If we get here then none of the units worked so raise an error
        raise exceptions.IncorrectUnits(
            f"{name} passed with incompatible units. "
            f"Expected any of {expected_units} (or equivalent)."
        )

    else:
        return _raise_or_convert(expected_units, name, value)


def accepts(**units):
    """Check arguments passed to the wrapped function have compatible units.

    This decorator will cross check any of the arguments passed to the wrapped
    function with the units defined in this decorators kwargs. If units are
    not compatible or are missing an error will be raised. If the units don't
    match the defined units in units then the values will be converted to the
    correct units.

    This is inspired by the accepts decorator in the unyt package, but includes
    Synthesizer specific errors and conversion functionality.

    Args:
        **units (dict):
            The keyword arguments defined with this decorator. Each takes the
            form of argument=unit_for_argument. In reality this is a
            dictionary of the form {"variable": unyt.unit}.

    Returns:
        function
            The wrapped function.
    """

    def check_accepts(func):
        """Check arguments have compatible units.

        This will check the arguments passed to the wrapped function have
        compatible units. If the units are missing or incompatible an error
        will be raised. If the units don't match the units passed to the
        accepts decorator in units then the values will be converted
        to the correct units.

        Args:
            func (function): The function to be wrapped.

        Returns:
            function: The wrapped function.
        """
        arg_names = func.__code__.co_varnames

        @wraps(func)
        def wrapped(*args, **kwargs):
            """Handle all the arguments passed to the wrapped function.

            Args:
                *args:
                    The arguments passed to the wrapped function.
                **kwargs:
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
