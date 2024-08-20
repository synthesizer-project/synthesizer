"""A module for working with arrays of black holes.

Contains the BlackHoles class for use with particle based systems. This houses
all the data detailing collections of black hole particles. Each property is
stored in (N_bh, ) shaped arrays for efficiency.

When instantiate a BlackHoles object a myriad of extra optional properties can
be set by providing them as keyword arguments.

Example usages:

    bhs = BlackHoles(masses, metallicities,
                     redshift=redshift, accretion_rate=accretion_rate, ...)
"""

import os

import numpy as np
from unyt import Hz, angstrom, cm, deg, erg, km, rad, s, unyt_array

from synthesizer import exceptions
from synthesizer.components import BlackholesComponent
from synthesizer.line import Line
from synthesizer.particle.particles import Particles
from synthesizer.units import Quantity
from synthesizer.utils import TableFormatter, value_to_array
from synthesizer.warnings import warn


class BlackHoles(Particles, BlackholesComponent):
    """
    The base BlackHoles class. This contains all data a collection of black
    holes could contain. It inherits from the base Particles class holding
    attributes and methods common to all particle types.

    The BlackHoles class can be handed to methods elsewhere to pass information
    about the stars needed in other computations. For example a Galaxy object
    can be initialised with a BlackHoles object for use with any of the Galaxy
    helper methods.

    Note that due to the many possible operations, this class has a large
    number ofoptional attributes which are set to None if not provided.

    Attributes:
        nbh (int)
            The number of black hole particles in the object.
        smoothing_lengths (array-like, float)
            The smoothing length describing the black holes neighbour kernel.
        particle_spectra (dict)
            A dictionary of Sed objects containing any of the generated
            particle spectra.
    """

    # Define the allowed attributes
    attrs = [
        "_masses",
        "_coordinates",
        "_velocities",
        "metallicities",
        "nparticles",
        "redshift",
        "_accretion_rate",
        "_bb_temperature",
        "_bolometric_luminosity",
        "_softening_lengths",
        "_smoothing_lengths",
        "nbh",
    ]

    # Define quantities
    smoothing_lengths = Quantity()

    def __init__(
        self,
        masses,
        accretion_rates,
        epsilons=0.1,
        inclinations=None,
        spins=None,
        metallicities=None,
        redshift=None,
        coordinates=None,
        velocities=None,
        softening_length=None,
        smoothing_lengths=None,
        centre=None,
        ionisation_parameter_blr=0.1,
        hydrogen_density_blr=1e9 / cm**3,
        covering_fraction_blr=0.1,
        velocity_dispersion_blr=2000 * km / s,
        ionisation_parameter_nlr=0.01,
        hydrogen_density_nlr=1e4 / cm**3,
        covering_fraction_nlr=0.1,
        velocity_dispersion_nlr=500 * km / s,
        theta_torus=10 * deg,
        tau_v=None,
        **kwargs,
    ):
        """
        Intialise the Stars instance. The first two arguments are always
        required. All other arguments are optional attributes applicable
        in different situations.

        Args:
            masses (array-like, float)
                The mass of each particle in Msun.
            metallicities (array-like, float)
                The metallicity of the region surrounding the/each black hole.
            epsilons (array-like, float)
                The radiative efficiency. By default set to 0.1.
            inclination (array-like, float)
                The inclination of the blackhole. Necessary for many emission
                models.
            redshift (float)
                The redshift/s of the black hole particles.
            accretion_rates (array-like, float)
                The accretion rate of the/each black hole in Msun/yr.
            coordinates (array-like, float)
                The 3D positions of the particles.
            velocities (array-like, float)
                The 3D velocities of the particles.
            softening_length (float)
                The physical gravitational softening length.
            smoothing_lengths (array-like, float)
                The smoothing length describing the black holes neighbour
                kernel.
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
            tau_v (array-like, float)
                The optical depth of the dust model.
            kwargs (dict)
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
        """

        # Handle singular values being passed (arrays are just returned)
        masses = value_to_array(masses)
        accretion_rates = value_to_array(accretion_rates)
        epsilons = value_to_array(epsilons)
        inclinations = value_to_array(inclinations)
        spins = value_to_array(spins)
        metallicities = value_to_array(metallicities)
        smoothing_lengths = value_to_array(smoothing_lengths)

        # Instantiate parents
        Particles.__init__(
            self,
            coordinates=coordinates,
            velocities=velocities,
            masses=masses,
            redshift=redshift,
            softening_length=softening_length,
            nparticles=len(masses),
            centre=centre,
            tau_v=tau_v,
            name="Black Holes",
        )
        BlackholesComponent.__init__(
            self,
            mass=masses,
            accretion_rate=accretion_rates,
            epsilon=epsilons,
            inclination=inclinations,
            spin=spins,
            metallicity=metallicities,
            ionisation_parameter_blr=ionisation_parameter_blr,
            hydrogen_density_blr=hydrogen_density_blr,
            covering_fraction_blr=covering_fraction_blr,
            velocity_dispersion_blr=velocity_dispersion_blr,
            ionisation_parameter_nlr=ionisation_parameter_nlr,
            hydrogen_density_nlr=hydrogen_density_nlr,
            covering_fraction_nlr=covering_fraction_nlr,
            velocity_dispersion_nlr=velocity_dispersion_nlr,
            theta_torus=theta_torus,
            **kwargs,
        )

        # Set a frontfacing clone of the number of particles with clearer
        # naming
        self.nbh = self.nparticles

        # Make pointers to the singular black hole attributes for consistency
        # in the backend
        for singular, plural in [
            ("mass", "masses"),
            ("accretion_rate", "accretion_rates"),
            ("metallicity", "metallicities"),
            ("spin", "spins"),
            ("inclination", "inclinations"),
            ("epsilon", "epsilons"),
            ("bb_temperature", "bb_temperatures"),
            ("bolometric_luminosity", "bolometric_luminosities"),
            ("accretion_rate_eddington", "accretion_rates_eddington"),
            ("epsilon", "epsilons"),
            ("eddington_ratio", "eddington_ratios"),
        ]:
            setattr(self, plural, getattr(self, singular))

        # Set the smoothing lengths
        self.smoothing_lengths = smoothing_lengths

        # Check the arguments we've been given
        self._check_bh_args()

    def _check_bh_args(self):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """
        # Need an early exit if we have no black holes since any
        # multidimensional  attributes will trigger the error below erroneously
        if self.nbh == 0:
            return

        # Ensure all arrays are the expected length
        for key in self.attrs:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent black hole array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

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

    def calculate_random_inclination(self):
        """
        Calculate random inclinations to blackholes.
        """

        self.inclination = (
            np.random.uniform(low=0.0, high=np.pi / 2.0, size=self.nbh) * rad
        )

        self.cosine_inclination = np.cos(self.inclination.to("rad").value)

    def _prepare_sed_args(
        self,
        grid,
        fesc,
        spectra_type,
        mask,
        grid_assignment_method,
        nthreads,
    ):
        """
        A method to prepare the arguments for SED computation with the C
        functions.

        Args:
            grid (Grid)
                The SPS grid object to extract spectra from.
            fesc (float)
                The escape fraction.
            spectra_type (str)
                The type of spectra to extract from the Grid. This must match a
                type of spectra stored in the Grid.
            mask(array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use for the computation. If -1 then
                all available threads are used.

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """
        # Which line region is this for?
        if "nlr" in grid.grid_name:
            line_region = "nlr"
        elif "blr" in grid.grid_name:
            line_region = "blr"
        else:
            raise exceptions.InconsistentArguments(
                "Grid used for blackholes does not appear to be for"
                " a line region (nlr or blr)."
            )

        # Handle the case where mask is None
        if mask is None:
            mask = np.ones(self.nbh, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(getattr(grid, axis), dtype=np.float64)
            for axis in grid.axes
        ]
        props = []
        for axis in grid.axes:
            # Parameters that need to be provided from the black hole
            prop = getattr(self, axis, None)

            # We might be trying to get a Quanitity, in which case we need
            # a leading _
            if prop is None:
                prop = getattr(self, f"_{axis}", None)

            # We might be missing a line region suffix, if prop is
            # None we need to try again with the suffix
            if prop is None:
                prop = getattr(self, f"{axis}_{line_region}", None)

            # We could also be tripped up by plurals (TODO: stop this from
            # happening!)
            elif prop is None and axis == "mass":
                prop = getattr(self, "masses", None)
            elif prop is None and axis == "accretion_rate":
                prop = getattr(self, "accretion_rates", None)
            elif prop is None and axis == "metallicity":
                prop = getattr(self, "metallicities", None)

            # If we still have None here then our blackhole component doesn't
            # have the required parameter
            if prop is None:
                raise exceptions.InconsistentArguments(
                    f"Could not find {axis} or {axis}_{line_region} "
                    f"on {type(self)}"
                )

            props.append(prop)

        # Calculate npart from the mask
        npart = np.sum(mask)

        # Remove units from any unyt_arrays
        props = [
            prop.value if isinstance(prop, unyt_array) else prop
            for prop in props
        ]

        # Ensure any parameters inherited from the emission model have
        # as many values as particles
        for ind, prop in enumerate(props):
            if isinstance(prop, float):
                props[ind] = np.full(self.nbh, prop)
            elif prop.size == 1:
                props[ind] = np.full(self.nbh, prop)

        # Apply the mask to each property and make contiguous
        props = [
            np.ascontiguousarray(prop[mask], dtype=np.float64)
            for prop in props
        ]

        # For black holes mass is a grid parameter but we still need to
        # multiply by mass in the extensions so just multiply by 1
        mass = np.ones(npart, dtype=np.float64)

        # Make sure we get the wavelength index of the grid array
        nlam = np.int32(grid.spectra[spectra_type].shape[-1])

        # Get the grid spctra
        grid_spectra = np.ascontiguousarray(
            grid.spectra[spectra_type],
            dtype=np.float64,
        )

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)
        grid_dims[ind + 1] = nlam

        # If fesc isn't an array make it one
        if not isinstance(fesc, np.ndarray):
            fesc = np.ascontiguousarray(np.full(npart, fesc))

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        props = tuple(props)

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_spectra,
            grid_props,
            props,
            mass,
            fesc,
            grid_dims,
            len(grid_props),
            np.int32(npart),
            nlam,
            grid_assignment_method,
            nthreads,
        )

    def _prepare_line_args(
        self,
        grid,
        line_id,
        line_type,
        fesc,
        mask,
        grid_assignment_method,
        nthreads,
    ):
        """
        Generate the arguments for the C extension to compute lines.

        Args:
            grid (Grid)
                The AGN grid object to extract lines from.
            line_id (str)
                The id of the line to extract.
            line_type (str)
                The type of line to extract from the grid. Must match the
                spectra/line type in the grid file.
            fesc (float/array-like, float)
                Fraction of stellar emission that escapes unattenuated from
                the birth cloud. Can either be a single value
                or an value per star (defaults to 0.0).
            mask (bool)
                A mask to be applied to the stars. Spectra will only be
                computed and returned for stars with True in the mask.
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.
        """
        # Which line region is this for?
        if "nlr" in grid.grid_name:
            line_region = "nlr"
        elif "blr" in grid.grid_name:
            line_region = "blr"
        else:
            raise exceptions.InconsistentArguments(
                "Grid used for blackholes does not appear to be for"
                " a line region (nlr or blr)."
            )

        # Handle the case where mask is None
        if mask is None:
            mask = np.ones(self.nbh, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(getattr(grid, axis), dtype=np.float64)
            for axis in grid.axes
        ]
        props = []
        for axis in grid.axes:
            # Parameters that need to be provided from the black hole
            prop = getattr(self, axis, None)

            # We might be trying to get a Quanitity, in which case we need
            # a leading _
            if prop is None:
                prop = getattr(self, f"_{axis}", None)

            # We might be missing a line region suffix, if prop is
            # None we need to try again with the suffix
            if prop is None:
                prop = getattr(self, f"{axis}_{line_region}", None)

            # We could also be tripped up by plurals (TODO: stop this from
            # happening!)
            elif prop is None and axis == "mass":
                prop = getattr(self, "masses", None)
            elif prop is None and axis == "accretion_rate":
                prop = getattr(self, "accretion_rates", None)
            elif prop is None and axis == "metallicity":
                prop = getattr(self, "metallicities", None)

            # If we still have None here then our blackhole component doesn't
            # have the required parameter
            if prop is None:
                raise exceptions.InconsistentArguments(
                    f"Could not find {axis} or {axis}_{line_region} "
                    f"on {type(self)}"
                )

            props.append(prop)

        # Calculate npart from the mask
        npart = np.sum(mask)

        # Remove units from any unyt_arrays
        props = [
            prop.value if isinstance(prop, unyt_array) else prop
            for prop in props
        ]

        # Ensure any parameters inherited from the emission model have
        # as many values as particles
        for ind, prop in enumerate(props):
            if isinstance(prop, float):
                props[ind] = np.full(self.nbh, prop)
            elif prop.size == 1:
                props[ind] = np.full(self.nbh, prop)

        # Apply the mask to each property and make contiguous
        props = [
            np.ascontiguousarray(prop[mask], dtype=np.float64)
            for prop in props
        ]

        # For black holes mass is a grid parameter but we still need to
        # multiply by mass in the extensions so just multiply by 1
        part_mass = np.ones(npart, dtype=np.float64)

        # Make sure we set the number of particles to the size of the mask
        npart = np.int32(np.sum(mask))

        # Get the line grid and continuum
        grid_line = np.ascontiguousarray(
            grid.line_lums[line_type][line_id],
            np.float64,
        )
        grid_continuum = np.ascontiguousarray(
            grid.line_conts[line_type][line_id],
            np.float64,
        )

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)

        # If fesc isn't an array make it one
        if not isinstance(fesc, np.ndarray):
            fesc = np.ascontiguousarray(np.full(npart, fesc))

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(props)

        # If nthreads is -1 then use all available threads
        if nthreads == -1:
            nthreads = os.cpu_count()

        return (
            grid_line,
            grid_continuum,
            grid_props,
            part_props,
            part_mass,
            fesc,
            grid_dims,
            len(grid_props),
            npart,
            grid_assignment_method,
            nthreads,
        )

    def generate_particle_lnu(
        self,
        grid,
        spectra_name,
        fesc=0.0,
        mask=None,
        verbose=False,
        grid_assignment_method="cic",
        nthreads=0,
    ):
        """
        Generate per particle rest frame spectra for a given key.

        Args:
            grid (obj):
                Spectral grid object.
            spectra_name (string)
                The name of the target spectra inside the grid file
                (e.g. "incident", "transmitted", "nebular").
            fesc (float):
                Fraction of emission that escapes unattenuated from
                the birth cloud (defaults to 0.0).
            mask (array-like, bool)
                If not None this mask will be applied to the inputs to the
                spectra creation.
            verbose (bool)
                Are we talking?
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use for the computation. If -1 then
                all available threads are used.
        """
        # Ensure we have a key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{spectra_name}'"
            )

        # If we have no black holes return zeros
        if self.nbh == 0:
            return np.zeros((self.nbh, len(grid.lam)))

        from ..extensions.particle_spectra import compute_particle_seds

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            grid,
            fesc=fesc,
            spectra_type=spectra_name,
            mask=mask,
            grid_assignment_method=grid_assignment_method.lower(),
            nthreads=nthreads,
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        masked_spec = compute_particle_seds(*args)

        # If there's no mask we're done
        if mask is None:
            return masked_spec

        # If we have a mask we need to account for the zeroed spectra
        spec = np.zeros((self.nbh, masked_spec.shape[-1]))
        spec[mask] = masked_spec
        return spec

    def generate_particle_line(
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
        luminosity of this blackhole population based on the
        passed arguments and calculate the luminosity and continuum for
        each individual particle.

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
                Fraction of blackhole emission that escapes unattenuated from
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
        from synthesizer.extensions.particle_line import (
            compute_particle_line,
        )

        # Ensure line_id is a string
        if not isinstance(line_id, str):
            raise exceptions.InconsistentArguments("line_id must be a string")

        # If we have no black holes return zeros
        if self.nbh == 0:
            return Line(
                *[
                    Line(
                        line_id=line_id_,
                        wavelength=grid.line_lams[line_id_] * angstrom,
                        luminosity=np.zeros(self.nparticles) * erg / s,
                        continuum=np.zeros(self.nparticles) * erg / s / Hz,
                    )
                    for line_id_ in line_id.split(",")
                ]
            )

        # Ensure and warn that the masking hasn't removed everything
        if mask is not None and np.sum(mask) == 0:
            warn("Age mask has filtered out all particles")

            return Line(
                *[
                    Line(
                        line_id=line_id_,
                        wavelength=grid.line_lams[line_id_] * angstrom,
                        luminosity=np.zeros(self.nparticles) * erg / s,
                        continuum=np.zeros(self.nparticles) * erg / s / Hz,
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
            _lum, _cont = compute_particle_line(
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

            # Account for the mask
            if mask is not None:
                lum = np.zeros(self.nparticles)
                cont = np.zeros(self.nparticles)
                lum[mask] = _lum
                cont[mask] = _cont
            else:
                lum = _lum
                cont = _cont

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
            return Line(*lines)

    def get_particle_spectra(
        self,
        emission_model,
        dust_curves=None,
        tau_v=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate blackhole spectra as described by the emission model.

        Args:
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool)
                Are we talking?
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of spectra which can be attached to the
                appropriate spectra attribute of the component
                (spectra/particle_spectra)
        """
        # Get the spectra
        spectra = emission_model._get_spectra(
            emitters={"blackhole": self},
            per_particle=True,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=covering_fraction,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the spectra dictionary
        self.particle_spectra.update(spectra)

        return self.particle_spectra[emission_model.label]

    def get_particle_lines(
        self,
        line_ids,
        emission_model,
        dust_curves=None,
        tau_v=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        **kwargs,
    ):
        """
        Generate stellar lines as described by the emission model.

        Args:
            line_ids (list):
                A list of line_ids. Doublets can be specified as a nested list
                or using a comma (e.g. 'OIII4363,OIII4959').
            emission_model (EmissionModel):
                The emission model to use.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                      should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or {<label>: str(<attribute>)} to use an attribute
                      of the component as the optical depth.
            fesc (dict):
                An override to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or {<label>: str(<attribute>)} to use an
                      attribute of the component as the escape fraction.
            mask (dict):
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form {<label>: {"attr": attr,
                      "thresh": thresh, "op": op}} to add a specific mask to
                      a particular model.
            verbose (bool)
                Are we talking?
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            LineCollection
                A LineCollection object containing the lines defined by the
                root model.
        """
        # Get the lines
        lines = emission_model._get_lines(
            line_ids=line_ids,
            emitters={"blackhole": self},
            per_particle=True,
            dust_curves=dust_curves,
            tau_v=tau_v,
            covering_fraction=covering_fraction,
            mask=mask,
            verbose=verbose,
            **kwargs,
        )

        # Update the lines dictionary
        self.particle_lines.update(lines)

        return self.particle_lines[emission_model.label]
