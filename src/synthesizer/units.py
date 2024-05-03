"""A module for dynamically returning attributes with and without units.

The Units class below acts as a container of unit definitions for various
attributes spread throughout Synthesizer.

The Quantity is the object that defines all attributes with attached units. Its
a helper class which enables the optional return of units.

Example defintion:

    class Foo:

        bar = Quantity()

        def __init__(self, bar):
            self.bar = bar

Example usage:

    foo = Foo(bar)

    bar_with_units = foo.bar
    bar_no_units = foo._bar

"""

from unyt import (
    Angstrom,
    Hz,
    K,
    Mpc,
    Msun,
    cm,
    deg,
    dimensionless,
    erg,
    km,
    nJy,
    s,
    unyt_array,
    unyt_quantity,
    yr,
)

# Define an importable dictionary with the default unit system
default_units = {
    "lam": Angstrom,
    "obslam": Angstrom,
    "wavelength": Angstrom,
    "vacuum_wavelength": Angstrom,
    "original_lam": Angstrom,
    "lam_min": Angstrom,
    "lam_max": Angstrom,
    "lam_eff": Angstrom,
    "lam_fwhm": Angstrom,
    "mean_lams": Angstrom,
    "pivot_lams": Angstrom,
    "nu": Hz,
    "obsnu": Hz,
    "nuz": Hz,
    "original_nu": Hz,
    "luminosity": erg / s,
    "luminosities": erg / s,
    "bolometric_luminosity": erg / s,
    "bolometric_luminosities": erg / s,
    "lnu": erg / s / Hz,
    "llam": erg / s / Angstrom,
    "continuum": erg / s / Hz,
    "flux": erg / s / cm**2,
    "fnu": nJy,
    "flam": erg / s / Angstrom / cm**2,
    "equivalent_width": Angstrom,
    "coordinates": Mpc,
    "smoothing_lengths": Mpc,
    "softening_length": Mpc,
    "velocities": km / s,
    "mass": Msun.in_base("galactic"),
    "masses": Msun.in_base("galactic"),
    "initial_masses": Msun.in_base("galactic"),
    "initial_mass": Msun.in_base("galactic"),
    "current_masses": Msun.in_base("galactic"),
    "dust_masses": Msun.in_base("galactic"),
    "ages": yr,
    "accretion_rate": Msun.in_base("galactic") / yr,
    "accretion_rates": Msun.in_base("galactic") / yr,
    "bb_temperature": K,
    "bb_temperatures": K,
    "inclination": deg,
    "inclinations": deg,
    "resolution": Mpc,
    "fov": Mpc,
    "orig_resolution": Mpc,
    "centre": Mpc,
    "photo_luminosities": erg / s / Hz,
    "photo_fluxes": erg / s / cm**2 / Hz,
}


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
            print(
                "WARNING! Units are already set. \nAny modified units will "
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

        photo_luminosities (unyt.unit_object.Unit)
            Rest frame photometry unit.
        photo_fluxes (unyt.unit_object.Unit)
            Observer frame photometry unit.

        ew (unyt.unit_object.Unit)
            Equivalent width unit.

        coordinates (unyt.unit_object.Unit)
            Particle coordinate unit.
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
        centre (unyt.unit_object.Unit)
            Centre of the image unit.
    """

    def __init__(self, units=None, force=False):
        """
        Intialise the Units object.

        Args:
            units (dict)
                A dictionary containing any modifications to the default unit
                system. This dictionary must be of the form:

                    units = {"coordinates": kpc,
                             "smoothing_lengths": kpc,
                             "lam": m}
            force (bool)
                A flag for whether to force an update of the Units object.
        """

        # First define all possible units with their defaults

        # Wavelengths
        self.lam = Angstrom  # rest frame wavelengths
        self.obslam = Angstrom  # observer frame wavelengths
        # vacuum rest frame wavelengths alias
        self.vacuum_wavelength = Angstrom
        self.wavelength = Angstrom  # rest frame wavelengths alias
        self.original_lam = Angstrom  # SVO filter wavelengths
        self.lam_min = Angstrom  # filter minimum wavelength
        self.lam_max = Angstrom  # filter maximum wavelength
        self.lam_eff = Angstrom  # filter effective wavelength
        self.lam_fwhm = Angstrom  # filter FWHM
        self.mean_lams = Angstrom  # filter collection mean wavelenghts
        self.pivot_lams = Angstrom  # filter collection pivot wavelengths

        # Frequencies
        self.nu = Hz  # rest frame frequencies
        self.nuz = Hz  # rest frame frequencies
        self.obsnu = Hz  # observer frame frequencies
        self.original_nu = Hz  # SVO filter wavelengths

        # Luminosities
        self.luminosity = erg / s  # luminosity
        self.luminosities = erg / s
        self.bolometric_luminosity = erg / s
        self.bolometric_luminosities = erg / s
        self.eddington_luminosity = erg / s
        self.eddington_luminosities = erg / s
        self.lnu = erg / s / Hz  # spectral luminosity density
        self.llam = erg / s / Angstrom  # spectral luminosity density
        self.flam = erg / s / Angstrom / cm**2  # spectral flux density
        self.continuum = erg / s / Hz  # continuum level of an emission line

        # Fluxes
        self.fnu = nJy  # rest frame flux
        self.flux = erg / s / cm**2  # rest frame "flux" at 10 pc

        # Photometry
        self.photo_luminosities = erg / s / Hz  # rest frame photometry
        self.photo_fluxes = erg / s / cm**2 / Hz  # observer frame photometry

        # Equivalent width
        self.equivalent_width = Angstrom

        # Spatial quantities
        self.coordinates = Mpc
        self.smoothing_lengths = Mpc
        self.softening_length = Mpc

        # Velocities
        self.velocities = km / s

        # Masses
        self.mass = Msun.in_base("galactic")
        self.masses = Msun.in_base("galactic")
        self.initial_masses = Msun.in_base(
            "galactic"
        )  # initial mass of stellar particles
        self.initial_mass = Msun.in_base(
            "galactic"
        )  # initial mass of stellar population
        self.current_masses = Msun.in_base(
            "galactic"
        )  # current mass of stellar particles
        self.dust_masses = Msun.in_base(
            "galactic"
        )  # current dust mass of gas particles

        # Time quantities
        self.ages = yr  # Stellar ages

        # Black holes quantities
        self.accretion_rate = Msun.in_base("galactic") / yr
        self.accretion_rates = Msun.in_base("galactic") / yr
        self.bb_temperature = K
        self.bb_temperatures = K
        self.inclination = deg
        self.inclinations = deg

        # Imaging quantities
        self.resolution = Mpc
        self.fov = Mpc
        self.orig_resolution = Mpc
        self.centre = Mpc

        # Do we have any modifications to the default unit system
        if units is not None:
            print("Redefining unit system:")
            for key in units:
                print("%s:" % key, units[key])
                setattr(self, key, units[key])

    def __str__(self):
        """
        Enables the printing of the current unit system.
        """
        out_str = "Unit System: \n"
        for key in default_units:
            out_str += (
                "%s: ".ljust(22 - len(key)) % key
                + getattr(self, key).__str__()
                + "\n"
            )

        return out_str


class Quantity:
    """
    Provides the ability to associate attribute values on an object with unyt
    units defined in the global unit system held in (Units).

    Attributes:
        units (Units)
            The global unit system.
        public_name (str)
            The name of the class variable containing Quantity. Used the user
            wants values with a unit returned.
        private_name (str)
            The name of the class variable with a leading underscore. Used the
            mostly internally for (or when the user wants) values without a
            unit returned.
    """

    def __init__(self):
        """
        Initialise the Quantity. This gets the unit system and then perfroms
        any necessary conversions if handed a unyt_array/unyt_quantity. If a n
        ormal array is passed it assumes this is already in the appropriate
        units.
        """

        # Attach the unit system
        self.units = Units()

    def __set_name__(self, owner, name):
        """
        When a class variable is assigned a Quantity() this method is called
        extracting the name of the class variable, assigning it to attributes
        for use when returning values with or without units.
        """
        self.public_name = name
        self.private_name = "_" + name

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
        unit = getattr(self.units, self.public_name)

        # If we have an uninitialised attribute avoid the multiplying NoneType
        # error and just return None
        if value is None:
            return None

        return value * unit

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
            if (
                value.units != getattr(self.units, self.public_name)
                and value.units != dimensionless
            ):
                value = value.to(getattr(self.units, self.public_name)).value
            else:
                value = value.value

        # Set the attribute
        setattr(obj, self.private_name, value)
