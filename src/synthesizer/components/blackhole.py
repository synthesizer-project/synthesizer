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
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter
from synthesizer.warnings import warn


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
    accretion_rate = Quantity()
    inclination = Quantity()
    bolometric_luminosity = Quantity()
    eddington_luminosity = Quantity()
    bb_temperature = Quantity()
    mass = Quantity()

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
        Initialise the BlackholeComponent.

        Where they're not provided missing quantities are automatically
        calcualted. Not all parameters need to be set for every emission model.

        Args:
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
            ionisation_parameter_blr (array-like, float)
                The ionisation parameter of the broadline region.
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
            kwargs (dict)
                Any other parameter for the emission models can be provided as
                kwargs.
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

    def generate_lnu(
        self,
        grid,
        spectra_name,
        fesc=None,
        mask=None,
        lam_mask=None,
        verbose=False,
        grid_assignment_method="cic",
        nthreads=0,
        vel_shift=False,
    ):
        """
        Generate integrated rest frame spectra for a given key.

        Args:
            emission_model (synthesizer.blackhole_emission_models.*)
                An instance of a blackhole emission model.
            grid (obj):
                Spectral grid object.
            fesc (float):
                Fraction of emission that escapes unattenuated from
                the birth cloud (defaults to 0.0).
            spectra_name (string)
                The name of the target spectra inside the grid file
                (e.g. "incident", "transmitted", "nebular").
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            lam_mask (array, bool)
                A mask to apply to the wavelength array of the grid. This
                allows for the extraction of specific wavelength ranges.
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.
            vel_shift (bool)
                Flags whether to apply doppler shift to the spectrum.
            c (float)
                Speed of light
        """
        # Ensure we have a key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{spectra_name}'"
            )

        # If we have have 0 particles (regardless of mask) just return an
        # array of zeros
        if hasattr(self, "nbh") and self.nbh == 0:
            return np.zeros(len(grid.lam))

        # If the mask is False (parametric case) or contains only
        # 0 (particle case) just return an array of zeros
        if isinstance(mask, bool) and not mask:
            return np.zeros(len(grid.lam))
        if mask is not None and np.sum(mask) == 0:
            return np.zeros(len(grid.lam))

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            grid,
            fesc=fesc,
            spectra_type=spectra_name,
            mask=mask,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
            vel_shift=vel_shift,
            lam_mask=lam_mask,
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        if vel_shift:
            from ..extensions.particle_spectra import (
                compute_part_sed_with_vel_shift,
            )

            spec = np.sum(compute_part_sed_with_vel_shift(*args), axis=0)
        else:
            from ..extensions.integrated_spectra import compute_integrated_sed

            spec = compute_integrated_sed(*args)

        # If we had a wavelength mask we need to make sure we return a spectra
        # compatible with the original wavelength array.
        if lam_mask is not None:
            out_spec = np.zeros(len(grid.lam))
            out_spec[lam_mask] = spec
            spec = out_spec

        return spec

    def generate_line(
        self,
        grid,
        line_id,
        line_type,
        fesc,
        mask=None,
        method="cic",
        nthreads=0,
        verbose=False,
    ):
        """
        Calculate rest frame line luminosity and continuum from an AGN Grid.

        This is a flexible base method which extracts the rest frame line
        luminosity of this stellar population from the AGN grid based on the
        passed arguments.

        Args:
            grid (Grid):
                A Grid object.
            line_id (list/str):
                A list of line_ids or a str denoting a single line.
                Doublets can be specified as a nested list or using a
                comma (e.g. 'OIII4363,OIII4959').
            line_type (str)
                The type of line to extract from the grid. Must match the
                spectra/line type in the grid file.
            fesc (float/array-like, float)
                Fraction of AGN emission that escapes unattenuated from
                the birth cloud. Can either be a single value
                or an value per star (defaults to 0.0).
            mask (array)
                A mask to apply to the particles (only applicable to particle)
            method (str)
                The method to use for the interpolation. Options are:
                'cic' - Cloud in cell
                'ngp' - Nearest grid point
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.

        Returns:
            Line
                An instance of Line contain this lines wavelenth, luminosity,
                and continuum.
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
                    fesc,
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
