"""A module for holding blackhole emission models.

The class defined here should never be instantiated directly, there are only
ever instantiated by the parametric/particle child classes.
BlackholesComponent is a child class of Component.
"""

import numpy as np
from unyt import Hz, Msun, angstrom, c, cm, deg, erg, km, s, yr

from synthesizer import exceptions
from synthesizer.components.component import Component
from synthesizer.line import Line
from synthesizer.synth_warnings import warn
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter


class BlackholesComponent(Component):
    """
    The parent class for stellar components of a galaxy.

    This class contains the attributes and spectra creation methods which are
    common to both parametric and particle stellar components.

    This should never be instantiated directly, instead it provides the common
    functionality and attributes used by the child parametric and particle
    BlackHole/s classes.

    Attributes:
        spectra (dict, Sed)
            A dictionary containing black hole spectra.
        mass (array-like, float)
            The mass of each blackhole.
        accretion_rate (array-like, float)
            The accretion rate of each blackhole.
        epsilon (array-like, float)
            The radiative efficiency of the blackhole.
        accretion_rate_eddington (array-like, float)
            The accretion rate expressed as a fraction of the Eddington
            accretion rate.
        inclination (array-like, float)
            The inclination of the blackhole disc.
        spin (array-like, float)
            The dimensionless spin of the blackhole.
        bolometric_luminosity (array-like, float)
            The bolometric luminosity of the blackhole.
        metallicity (array-like, float)
            The metallicity of the blackhole which is assumed for the line
            emitting regions.

    Attributes (For EmissionModels):
        ionisation_parameter_blr (array-like, float)
            The ionisation parameter of the broad line region.
        hydrogen_density_blr (array-like, float)
            The hydrogen density of the broad line region.
        covering_fraction_blr (array-like, float)
            The covering fraction of the broad line region (effectively
            the escape fraction).
        velocity_dispersion_blr (array-like, float)
            The velocity dispersion of the broad line region.
        ionisation_parameter_nlr (array-like, float)
            The ionisation parameter of the narrow line region.
        hydrogen_density_nlr (array-like, float)
            The hydrogen density of the narrow line region.
        covering_fraction_nlr (array-like, float)
            The covering fraction of the narrow line region (effectively
            the escape fraction).
        velocity_dispersion_nlr (array-like, float)
            The velocity dispersion of the narrow line region.
        theta_torus (array-like, float)
            The angle of the torus.
        torus_fraction (array-like, float)
            The fraction of the torus angle to 90 degrees.
    """

    # Define class level Quantity attributes
    accretion_rate = Quantity("mass_rate")
    inclination = Quantity("angle")
    bolometric_luminosity = Quantity("luminosity")
    eddington_luminosity = Quantity("luminosity")
    bb_temperature = Quantity("temperature")
    mass = Quantity("mass")

    @accepts(
        mass=Msun.in_base("galactic"),
        accretion_rate=Msun.in_base("galactic") / yr,
        accretion_rate_eddington=Msun.in_base("galactic") / yr,
        inclination=deg,
        bolometric_luminosity=erg / s,
        hydrogen_density_blr=cm**-3,
        hydrogen_density_nlr=cm**-3,
        velocity_dispersion_blr=km / s,
        velocity_dispersion_nlr=km / s,
        theta_torus=deg,
    )
    def __init__(
        self,
        fesc,
        mass=None,
        accretion_rate=None,
        epsilon=0.1,
        accretion_rate_eddington=None,
        inclination=0.0 * deg,
        spin=None,
        bolometric_luminosity=None,
        metallicity=None,
        ionisation_parameter_blr=0.1,
        hydrogen_density_blr=1e9 / cm**3,
        covering_fraction_blr=0.1,
        velocity_dispersion_blr=2000 * km / s,
        ionisation_parameter_nlr=0.01,
        hydrogen_density_nlr=1e4 / cm**3,
        covering_fraction_nlr=0.1,
        velocity_dispersion_nlr=500 * km / s,
        theta_torus=10 * deg,
        **kwargs,
    ):
        """
        Initialize a BlackholesComponent instance with properties for black hole emission models.
        
        Missing quantities are computed automatically when sufficient parameters are provided.
        Do not supply both accretion_rate and bolometric_luminosity (or accretion_rate_eddington and
        bolometric_luminosity) to avoid ambiguity.
        
        Args:
            fesc: Escape fraction for the component.
            mass: Masses of the black holes.
            accretion_rate: Accretion rates of the black holes.
            epsilon: Radiative efficiency of the black holes.
            accretion_rate_eddington: Accretion rates expressed as fractions of the Eddington rate.
            inclination: Inclination angles of the black hole discs.
            spin: Dimensionless spins of the black holes.
            bolometric_luminosity: Bolometric luminosities of the black holes.
            metallicity: Metallicities for the line emitting regions.
            ionisation_parameter_blr: Ionisation parameters for the broad line region.
            hydrogen_density_blr: Hydrogen densities in the broad line region.
            covering_fraction_blr: Covering fractions in the broad line region.
            velocity_dispersion_blr: Velocity dispersions in the broad line region.
            ionisation_parameter_nlr: Ionisation parameters for the narrow line region.
            hydrogen_density_nlr: Hydrogen densities in the narrow line region.
            covering_fraction_nlr: Covering fractions in the narrow line region.
            velocity_dispersion_nlr: Velocity dispersions in the narrow line region.
            theta_torus: Angles of the torus.
            kwargs: Additional parameters for the emission models.
        
        Raises:
            InconsistentArguments: If both accretion_rate and bolometric_luminosity are provided, or if
                both accretion_rate_eddington and bolometric_luminosity are provided.
        """
        # Initialise the parent class
        Component.__init__(self, "BlackHoles", fesc, **kwargs)

        # Save the black hole properties
        self.mass = mass
        self.accretion_rate = accretion_rate
        self.epsilon = epsilon
        self.accretion_rate_eddington = accretion_rate_eddington
        self.spin = spin
        self.bolometric_luminosity = bolometric_luminosity
        self.metallicity = metallicity

        # Below we attach all the possible attributes that could be needed by
        # the emission models.

        # Set BLR attributes
        self.ionisation_parameter_blr = ionisation_parameter_blr
        self.hydrogen_density_blr = hydrogen_density_blr
        self.covering_fraction_blr = covering_fraction_blr
        self.velocity_dispersion_blr = velocity_dispersion_blr

        # Set NLR attributes
        self.ionisation_parameter_nlr = ionisation_parameter_nlr
        self.hydrogen_density_nlr = hydrogen_density_nlr
        self.covering_fraction_nlr = covering_fraction_nlr
        self.velocity_dispersion_nlr = velocity_dispersion_nlr

        # The inclination of the black hole disc
        self.inclination = (
            inclination if inclination is not None else 0.0 * deg
        )

        # The angle of the torus
        self.theta_torus = theta_torus
        self.torus_fraction = (self.theta_torus / (90 * deg)).value
        self._torus_edgeon_cond = self.inclination + self.theta_torus

        # Check to make sure that both accretion rate and bolometric luminosity
        # haven't been provided because that could be confusing.
        if (self.accretion_rate is not None) and (
            self.bolometric_luminosity is not None
        ):
            raise exceptions.InconsistentArguments(
                """Both accretion rate and bolometric luminosity provided but
                that is confusing. Provide one or the other!"""
            )

        if (self.accretion_rate_eddington is not None) and (
            self.bolometric_luminosity is not None
        ):
            raise exceptions.InconsistentArguments(
                """Both accretion rate (in terms of Eddington) and bolometric
                luminosity provided but that is confusing. Provide one or
                the other!"""
            )

        # If mass, accretion_rate, and epsilon provided calculate the
        # bolometric luminosity.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_bolometric_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # big bump temperature.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_bb_temperature()

        # If mass calculate the Eddington luminosity.
        if self.mass is not None:
            self.calculate_eddington_luminosity()

        # If mass, accretion_rate, and epsilon provided calculate the
        # Eddington ratio.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_eddington_ratio()

        # If mass, accretion_rate, and epsilon provided calculate the
        # accretion rate in units of the Eddington accretion rate. This is the
        # bolometric_luminosity / eddington_luminosity.
        if (
            self.mass is not None
            and self.accretion_rate is not None
            and self.epsilon is not None
        ):
            self.calculate_accretion_rate_eddington()

        # If inclination, calculate the cosine of the inclination, required by
        # some models (e.g. AGNSED).
        if self.inclination is not None:
            self.cosine_inclination = np.cos(
                self.inclination.to("radian").value
            )

    def generate_line(
        self,
        grid,
        line_id,
        line_type,
        mask=None,
        method="cic",
        nthreads=0,
        verbose=False,
    ):
        """
        Calculate rest frame emission line properties from an AGN grid.
        
        This method computes the integrated emission line luminosity and continuum for the
        current black hole component using grid-based interpolation. It accepts a comma-separated
        string of emission line identifiers and returns a single Line object when one line is specified,
        or a combined Line object when multiple lines are provided. If no particles are present, or if
        a provided mask excludes all particles, the method returns a Line with zero luminosity and
        continuum and issues a warning.
        
        Args:
            grid: A Grid object containing AGN spectral data.
            line_id (str): A comma-separated string of emission line identifiers.
            line_type (str): The spectral line type to extract; must match an entry in the grid.
            mask: Optional array to select a subset of particles.
            method (str): The grid interpolation method ("cic" for cloud-in-cell or "ngp" for nearest grid point).
            nthreads (int): Number of threads for computation (-1 uses all available threads).
            verbose (bool): Flag to enable verbose output (currently not used).
        
        Returns:
            Line: A Line object encapsulating the wavelength, luminosity, and continuum. If multiple
            lines are specified, the returned object combines individual lines.
        
        Raises:
            InconsistentArguments: If 'line_id' is not provided as a comma-separated string.
        """
        from synthesizer.extensions.integrated_line import (
            compute_integrated_line,
        )

        # Ensure line_id is a string
        if not isinstance(line_id, str):
            raise exceptions.InconsistentArguments("line_id must be a string")

        # If we have have 0 particles (regardless of mask) just return a line
        # containing zeros
        if hasattr(self, "nbh") and self.nbh == 0:
            return Line(
                combine_lines=[
                    Line(
                        line_id=line_id_,
                        wavelength=grid.line_lams[line_id_] * angstrom,
                        luminosity=0.0 * erg / s,
                        continuum=0.0 * erg / s / Hz,
                    )
                    for line_id_ in line_id.split(",")
                ]
            )

        # Ensure and warn that the masking hasn't removed everything
        if mask is not None and np.sum(mask) == 0:
            warn("Age mask has filtered out all particles")

            return Line(
                combine_lines=[
                    Line(
                        line_id=line_id_,
                        wavelength=grid.line_lams[line_id_] * angstrom,
                        luminosity=0.0 * erg / s,
                        continuum=0.0 * erg / s / Hz,
                    )
                    for line_id_ in line_id.split(",")
                ]
            )

        # Set up a list to hold each individual Line
        lines = []

        # Loop over the ids in this container
        for line_id_ in line_id.split(","):
            # Strip off any whitespace (can be left by split)
            line_id_ = line_id_.strip()

            # Get this line's wavelength
            # TODO: The units here should be extracted from the grid but aren't
            # yet stored.
            lam = grid.line_lams[line_id_] * angstrom

            # Get the luminosity and continuum
            lum, cont = compute_integrated_line(
                *self._prepare_line_args(
                    grid,
                    line_id_,
                    line_type,
                    mask=mask,
                    grid_assignment_method=method,
                    nthreads=nthreads,
                )
            )

            # Append this lines values to the containers
            lines.append(
                Line(
                    line_id=line_id_,
                    wavelength=lam,
                    luminosity=lum * erg / s,
                    continuum=cont * erg / s / Hz,
                )
            )

        # Don't init another line if there was only 1 in the first place
        if len(lines) == 1:
            return lines[0]
        else:
            return Line(combine_lines=lines)

    def calculate_bolometric_luminosity(self):
        """
        Calculate the black hole bolometric luminosity. This is by itself
        useful but also used for some emission models.

        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        self.bolometric_luminosity = self.epsilon * self.accretion_rate * c**2

        return self.bolometric_luminosity

    def calculate_eddington_luminosity(self):
        """
        Calculate the eddington luminosity of the black hole.

        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        # Note: the factor 1.257E38 comes from:
        # 4*pi*G*mp*c*Msun/sigma_thompson
        self.eddington_luminosity = 1.257e38 * self._mass

        return self.eddington_luminosity

    def calculate_eddington_ratio(self):
        """
        Calculate the eddington ratio of the black hole.

        Returns
            unyt_array
                The black hole eddington ratio
        """

        self.eddington_ratio = (
            self._bolometric_luminosity / self._eddington_luminosity
        )

        return self.eddington_ratio

    def calculate_bb_temperature(self):
        """
        Calculate the black hole big bump temperature. This is used for the
        cloudy disc model.

        Returns
            unyt_array
                The black hole bolometric luminosity
        """

        # Calculate the big bump temperature
        self.bb_temperature = (
            2.24e9 * self._accretion_rate ** (1 / 4) * self._mass**-0.5
        )

        return self.bb_temperature

    def calculate_accretion_rate_eddington(self):
        """
        Calculate the black hole accretion in units of the Eddington rate.

        Returns
            unyt_array
                The black hole accretion rate in units of the Eddington rate.
        """

        self.accretion_rate_eddington = (
            self._bolometric_luminosity / self._eddington_luminosity
        )

        return self.accretion_rate_eddington

    def __str__(self):
        """
        Return a string representation of the particle object.

        Returns:
            table (str)
                A string representation of the particle object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("Black Holes")
