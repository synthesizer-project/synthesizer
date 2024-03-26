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

from typing import Any, Dict, Optional, Type, Union

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
from unyt.unit_object import Unit

# Define an importable dictionary with the default unit system
default_units: Dict[str, Unit] = {
    "lam": Angstrom,
    "obslam": Angstrom,
    "wavelength": Angstrom,
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
    _instances: Dict[Type, "Units"] = {}

    def __call__(
        cls,
        new_units: Optional[Dict[str, Unit]] = None,
        force: bool = False,
    ) -> "Units":
        """
        When a new instance is made (calling class), this method is called.

        Unless forced to redefine Units (highly inadvisable), the original
        instance is returned giving it a new reference to the original
        instance.

        If a new unit system is passed and one already exists and warning is
        printed and the original is returned.

        Returns:
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
        lam: Rest frame wavelength.
        obslam: Observer frame wavelength.
        wavelength: Alias for rest frame wavelength.

        nu: Rest frame frequency.
        obsnu: Observer frame frequency.
        nuz: Observer frame frequency.

        luminosity: Luminosity.
        lnu: Rest frame spectral luminosity density (in terms of frequency).
        llam: Rest frame spectral luminosity density (in terms of wavelength).
        continuum: Continuum level of an emission line.

        fnu: Spectral flux density (in terms of frequency).
        flam: Spectral flux density (in terms of wavelength).
        flux: "Rest frame" Spectral flux density (at 10 pc).

        photo_luminosities: Rest frame photometry.
        photo_fluxes: Observer frame photometry.

        ew: Equivalent width.

        coordinates: Particle coordinate.
        smoothing_lengths: Particle smoothing length.
        softening_length: Particle gravitational softening length.

        velocities: Particle velocity.

        masses: Particle masses.
        initial_masses: Stellar particle initial mass.
        initial_mass: Stellar population initial mass.
        current_masses: Stellar particle current mass.
        dust_masses: Gas particle dust masses.

        ages: Stellar particle age.

        accretion_rate: Black hole accretion rate.
        bolometric_luminosity: Bolometric luminositiy.
        bolometric_luminosities: Bolometric luminositiy.
        bb_temperature: Black hole big bump temperature.
        bb_temperatures: Black hole big bump temperature.
        inclination: Black hole inclination.
        inclinations: Black hole inclination.

        resolution: Image resolution.
        fov: Field of View.
        orig_resolution: Original resolution (for resampling).
        centre: Centre of the image.
    """

    lam: Unit
    obslam: Unit
    wavelength: Unit
    original_lam: Unit
    lam_min: Unit
    lam_max: Unit
    lam_eff: Unit
    lam_fwhm: Unit
    mean_lams: Unit
    pivot_lams: Unit
    nu: Unit
    obsnu: Unit
    nuz: Unit
    original_nu: Unit
    luminosity: Unit
    luminosities: Unit
    bolometric_luminosity: Unit
    bolometric_luminosities: Unit
    lnu: Unit
    llam: Unit
    continuum: Unit
    flux: Unit
    fnu: Unit
    flam: Unit
    equivalent_width: Unit
    coordinates: Unit
    smoothing_lengths: Unit
    softening_length: Unit
    velocities: Unit
    mass: Unit
    masses: Unit
    initial_masses: Unit
    initial_mass: Unit
    current_masses: Unit
    dust_masses: Unit
    ages: Unit
    accretion_rate: Unit
    accretion_rates: Unit
    bb_temperature: Unit
    bb_temperatures: Unit
    inclination: Unit
    inclinations: Unit
    resolution: Unit
    fov: Unit
    orig_resolution: Unit
    centre: Unit
    photo_luminosities: Unit
    photo_fluxes: Unit

    def __init__(
        self,
        units: Optional[Dict[str, Unit]] = None,
        force: bool = False,
    ) -> None:
        """
        Intialise the Units object.

        Args:
            units: A dictionary containing any modifications to the default
                   unit system. This dictionary must be of the form:

                       units = {"coordinates": kpc,
                                "smoothing_lengths": kpc,
                                "lam": m}

            force: A flag for whether to force an update of the Units object.
        """

        # First define all possible units with their defaults

        # Wavelengths
        self.lam = Angstrom  # rest frame wavelengths
        self.obslam = Angstrom  # observer frame wavelengths
        self.wavelength = Angstrom  # rest frame wavelengths alais
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

    def __str__(self) -> str:
        """
        Enables the printing of the current unit system.
        """
        out_str: str = "Unit System: \n"
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
        units: The global unit system.
        public_name: The name of the class variable containing Quantity. Used
                     when the user wants values with a unit returned.
        private_name: The name of the class variable with a leading underscore.
                      Used mostly internally for (or when the user wants)
                      values without a unit returned.
    """

    units: Units
    public_name: str
    private_name: str

    def __init__(self) -> None:
        """
        Initialise the Quantity. This gets the unit system and then perfroms
        any necessary conversions if handed a unyt_array/unyt_quantity. If a n
        ormal array is passed it assumes this is already in the appropriate
        units.
        """

        # Attach the unit system
        self.units = Units()

    def __set_name__(self, owner: Type, name: str) -> None:
        """
        When a class variable is assigned a Quantity() this method is called
        extracting the name of the class variable, assigning it to attributes
        for use when returning values with or without units.
        """
        self.public_name = name
        self.private_name = "_" + name

    def __get__(
        self,
        obj: Any,
        type: Optional[Type] = None,
    ) -> Union[unyt_array, unyt_quantity, None]:
        """
        When referencing an attribute with its public_name this method is
        called. It handles the returning of the values stored in the
        private_name variable with units.

        If the value is None then None is returned regardless.

        Args:
            obj: The object containing the Quantity attribute that we are
                 returning the value of.
            type: The type of the object containing the Quantity attribute.

        Returns:
            The value with units attached or None if value is None.
        """
        value: Any = getattr(obj, self.private_name)
        unit: Unit = getattr(self.units, self.public_name)

        # If we have an uninitialised attribute avoid the multiplying NoneType
        # error and just return None
        if value is None:
            return None

        return value * unit

    def __set__(
        self,
        obj: Any,
        value: Union[unyt_quantity, unyt_array, Any],
    ) -> None:
        """
        When setting a Quantity variable this method is called, firstly hiding
        the private name that stores the value array itself and secondily
        applying any necessary unit conversions.

        Args:
            obj: The object contain the Quantity attribute that we are storing
                 value in.
            value: The value to store in the attribute.
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
