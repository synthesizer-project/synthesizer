"""A module for dynamically returning attributes with and without units.

The Units class below acts as a container of unit definitions for various
attributes spread throughout Synthesizer.

The Quantity is the object that defines all attributes with attached units. Its
a helper class which enables the optional return of units. The Quantity
is a descriptor which is defined in the class body with the unit category
passed as an argument.

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


# Get the default units system we have on file (this can be modified by the
# user.
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
    A metaclass used to ensure singleton behaviour of Units.

    i.e. there can only ever be a single instance of a class in a namespace.

    Adapted from:
    https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python
    """

    # Define a private dictionary to store instances of UnitSingleton
    _instances = {}

    def __call__(cls, new_units=None, force=False):
        """
        When a new instance is made (calling class), this method is called.

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
        lam (unyt.unit_object.Unit)
            Rest frame wavelength unit.
        obslam (unyt.unit_object.Unit)
            Observer frame wavelength unit.
        wavelength (unyt.unit_object.Unit)
            Alias for rest frame wavelength unit.
        wavelengths (unyt.unit_object.Unit)
            Alias for rest frame wavelength unit.

        nu (unyt.unit_object.Unit)
            Rest frame frequency unit.
        obsnu (unyt.unit_object.Unit)
            Observer frame frequency unit.
        nuz (unyt.unit_object.Unit)
            Observer frame frequency unit.

        luminosity (unyt.unit_object.Unit)
            Luminosity unit.
        lnu (unyt.unit_object.Unit)
            Rest frame spectral luminosity density (in terms of frequency)
            unit.
        llam (unyt.unit_object.Unit)
            Rest frame spectral luminosity density (in terms of wavelength)
            unit.
        continuum (unyt.unit_object.Unit)
            Continuum level of an emission line unit.

        fnu (unyt.unit_object.Unit)
            Spectral flux density (in terms of frequency) unit.
        flam (unyt.unit_object.Unit)
            Spectral flux density (in terms of wavelength) unit.
        flux (unyt.unit_object.Unit)
            "Rest frame" Spectral flux density (at 10 pc) unit.

        photo_lnu (unyt.unit_object.Unit)
            Rest frame photometry unit.
        photo_fnu (unyt.unit_object.Unit)
            Observer frame photometry unit.

        ew (unyt.unit_object.Unit)
            Equivalent width unit.

        coordinates (unyt.unit_object.Unit)
            Particle coordinate unit.
        centre (unyt.unit_object.Unit)
            Galaxy/particle distribution centre unit.
        radii (unyt.unit_object.Unit)
            Particle radii unit.
        smoothing_lengths (unyt.unit_object.Unit)
            Particle smoothing length unit.
        softening_length (unyt.unit_object.Unit)
            Particle gravitational softening length unit.

        velocities (unyt.unit_object.Unit)
            Particle velocity unit.

        masses (unyt.unit_object.Unit)
            Particle masses unit.
        initial_masses (unyt.unit_object.Unit)
            Stellar particle initial mass unit.
        initial_mass (unyt.unit_object.Unit)
            Stellar population initial mass unit.
        current_masses (unyt.unit_object.Unit)
            Stellar particle current mass unit.
        dust_masses (unyt.unit_object.Unit)
            Gas particle dust masses unit.

        ages (unyt.unit_object.Unit)
            Stellar particle age unit.

        accretion_rate (unyt.unit_object.Unit)
            Black hole accretion rate unit.
        bolometric_luminosity (unyt.unit_object.Unit)
            Bolometric luminositiy unit.
        bolometric_luminosities (unyt.unit_object.Unit)
            Bolometric luminositiy unit.
        bb_temperature (unyt.unit_object.Unit)
            Black hole big bump temperature unit.
        bb_temperatures (unyt.unit_object.Unit)
            Black hole big bump temperature unit.
        inclination (unyt.unit_object.Unit)
            Black hole inclination unit.
        inclinations (unyt.unit_object.Unit)
            Black hole inclination unit.

        resolution (unyt.unit_object.Unit)
            Image resolution unit.
        fov (unyt.unit_object.Unit)
            Field of View unit.
        orig_resolution (unyt.unit_object.Unit)
            Original resolution (for resampling) unit.

        softening_lengths (unyt.unit_object.Unit)
            Particle gravitational softening length unit.
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
        # First off we need to attach the default unit system
        # to the Units object
        for key, unit in default_units.items():
            setattr(self, key, unit)

        # # Wavelengths
        # self.lam = Angstrom  # rest frame wavelengths
        # self.obslam = Angstrom  # observer frame wavelengths
        # # vacuum rest frame wavelengths alias
        # self.vacuum_wavelength = Angstrom
        # self.wavelength = Angstrom  # rest frame wavelengths alias
        # self.wavelengths = Angstrom  # rest frame wavelengths alias
        # self.original_lam = Angstrom  # SVO filter wavelengths
        # self.lam_min = Angstrom  # filter minimum wavelength
        # self.lam_max = Angstrom  # filter maximum wavelength
        # self.lam_eff = Angstrom  # filter effective wavelength
        # self.lam_fwhm = Angstrom  # filter FWHM
        # self.mean_lams = Angstrom  # filter collection mean wavelenghts
        # self.pivot_lams = Angstrom  # filter collection pivot wavelengths
        #
        # # Frequencies
        # self.nu = Hz  # rest frame frequencies
        # self.nuz = Hz  # rest frame frequencies
        # self.obsnu = Hz  # observer frame frequencies
        # self.original_nu = Hz  # SVO filter wavelengths
        #
        # # Luminosities
        # self.luminosity = erg / s  # luminosity
        # self.luminosities = erg / s
        # self.bolometric_luminosity = erg / s
        # self.bolometric_luminosities = erg / s
        # self.eddington_luminosity = erg / s
        # self.eddington_luminosities = erg / s
        # self.lnu = erg / s / Hz  # spectral luminosity density
        # self.llam = erg / s / Angstrom  # spectral luminosity density
        # self.flam = erg / s / Angstrom / cm**2  # spectral flux density
        # self.continuum = erg / s / Hz  # continuum level of an emission line
        #
        # # Fluxes
        # self.fnu = nJy  # rest frame flux
        # self.flux = erg / s / cm**2  # rest frame "flux" at 10 pc
        #
        # # Photometry
        # self.photo_lnu = erg / s / Hz  # rest frame photometry
        # self.photo_fnu = erg / s / cm**2 / Hz  # observer frame photometry
        #
        # # Equivalent width
        # self.equivalent_width = Angstrom
        #
        # # Spatial quantities
        # self.coordinates = Mpc
        # self.centre = Mpc
        # self.radii = Mpc
        # self.smoothing_lengths = Mpc
        # self.softening_length = Mpc
        #
        # # Velocities
        # self.velocities = km / s
        #
        # # Masses
        # self.mass = Msun.in_base("galactic")
        # self.masses = Msun.in_base("galactic")
        # self.initial_masses = Msun.in_base(
        #     "galactic"
        # )  # initial mass of stellar particles
        # self.initial_mass = Msun.in_base(
        #     "galactic"
        # )  # initial mass of stellar population
        # self.current_masses = Msun.in_base(
        #     "galactic"
        # )  # current mass of stellar particles
        # self.dust_masses = Msun.in_base(
        #     "galactic"
        # )  # current dust mass of gas particles
        #
        # # Time quantities
        # self.ages = yr  # Stellar ages
        #
        # # Black holes quantities
        # self.accretion_rate = Msun.in_base("galactic") / yr
        # self.accretion_rates = Msun.in_base("galactic") / yr
        # self.bb_temperature = K
        # self.bb_temperatures = K
        # self.inclination = deg
        # self.inclinations = deg
        #
        # # Imaging quantities
        # self.resolution = Mpc
        # self.fov = Mpc
        # self.orig_resolution = Mpc
        #
        # # Gravitational softening lengths
        # self.softening_lengths = Mpc
        #
        # Do we have any modifications to the default unit system
        if units is not None:
            print("Redefining unit system:")
            for key in units:
                print("%s:" % key, units[key])
                setattr(self, key, units[key])

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


class Quantity:
    """
    Provides the ability to associate attribute values on an object with unyt
    units defined in the global unit system held in (Units).

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
        When referencing an attribute with its public_name this method is
        called. It handles the returning of the values stored in the
        private_name variable with units.

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
        When setting a Quantity variable this method is called, firstly hiding
        the private name that stores the value array itself and secondily
        applying any necessary unit conversions.

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
