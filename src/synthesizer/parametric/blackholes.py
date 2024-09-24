"""A module for working with a parametric black holes.

Contains the BlackHole class for use with parametric systems. This houses
all the attributes and functionality related to parametric black holes.

Example usages::

    bhs = BlackHole(
        bolometric_luminosity,
        mass,
        accretion_rate,
        epsilon,
        inclination,
        spin,
        metallicity,
        offset,
    )
"""

import os

import numpy as np
from unyt import Msun, cm, deg, erg, km, kpc, s, unyt_array, yr

from synthesizer import exceptions
from synthesizer.components import BlackholesComponent
from synthesizer.parametric.morphology import PointSource
from synthesizer.units import accepts
from synthesizer.utils import TableFormatter


class BlackHole(BlackholesComponent):
    """
    The base parametric BlackHole class.

    Attributes:
        morphology (PointSource)
            An instance of the PointSource morphology that describes the
            location of this blackhole
    """

    @accepts(
        mass=Msun.in_base("galactic"),
        accretion_rate=Msun.in_base("galactic") / yr,
        inclination=deg,
        offset=kpc,
        bolometric_luminosity=erg / s,
        hydrogen_density_blr=1 / cm**3,
        hydrogen_density_nlr=1 / cm**3,
        velocity_dispersion_blr=km / s,
        velocity_dispersion_nlr=km / s,
        theta_torus=deg,
    )
    def __init__(
        self,
        mass=None,
        accretion_rate=None,
        epsilon=0.1,
        inclination=None,
        spin=None,
        offset=np.array([0.0, 0.0]) * kpc,
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
        Intialise the Stars instance. The first two arguments are always
        required. All other arguments are optional attributes applicable
        in different situations.

        Args:
            mass (float)
                The mass of each particle in Msun.
            accretion_rate (float)
                The accretion rate of the/each black hole in Msun/yr.
            epsilon (float)
                The radiative efficiency. By default set to 0.1.
            inclination (float)
                The inclination of the blackhole. Necessary for some disc
                models.
            spin (float)
                The spin of the blackhole. Necessary for some disc models.
            offset (unyt_array)
                The (x,y) offsets of the blackhole relative to the centre of
                the image. Units can be length or angle but should be
                consistent with the scene.
            bolometric_luminosity (float)
                The bolometric luminosity
            metallicity (float)
                The metallicity of the region surrounding the/each black hole.
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
            kwargs (dict)
                Any parameter for the emission models can be provided as kwargs
                here to override the defaults of the emission models.
        """
        # Initialise base class
        BlackholesComponent.__init__(
            self,
            bolometric_luminosity=bolometric_luminosity,
            mass=mass,
            accretion_rate=accretion_rate,
            epsilon=epsilon,
            inclination=inclination,
            spin=spin,
            metallicity=metallicity,
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
        # Initialise morphology using the in-built point-source class
        self.morphology = PointSource(offset=offset)

    def get_mask(self, attr, thresh, op, mask=None):
        """
        Create a mask using a threshold and attribute on which to mask.

        Args:
            attr (str)
                The attribute to derive the mask from.
            thresh (float)
                The threshold value.
            op (str)
                The operation to apply. Can be '<', '>', '<=', '>=', "==",
                or "!=".
            mask (array)
                Optionally, a mask to combine with the new mask.

        Returns:
            mask (array)
                The mask array.
        """
        # Get the attribute
        attr = getattr(self, attr)

        # Apply the operator
        if op == ">":
            new_mask = attr > thresh
        elif op == "<":
            new_mask = attr < thresh
        elif op == ">=":
            new_mask = attr >= thresh
        elif op == "<=":
            new_mask = attr <= thresh
        elif op == "==":
            new_mask = attr == thresh
        elif op == "!=":
            new_mask = attr != thresh
        else:
            raise exceptions.InconsistentArguments(
                "Masking operation must be '<', '>', '<=', '>=', '==', or "
                f"'!=', not {op}"
            )

        # Combine with the existing mask
        if mask is not None:
            new_mask = np.logical_and(new_mask, mask)

        return new_mask

    def _prepare_sed_args(
        self,
        grid,
        fesc,
        spectra_type,
        grid_assignment_method,
        nthreads,
        **kwargs,
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
            grid_assignment_method (string)
                The type of method used to assign particles to a SPS grid
                point. Allowed methods are cic (cloud in cell) or nearest
                grid point (ngp) or there uppercase equivalents (CIC, NGP).
                Defaults to cic.
            nthreads (int)
                The number of threads to use in the C extension. If -1 then
                all available threads are used.
            kwargs (dict)
                Any other arguments. Mainly unused and here for consistency
                with particle version of this method which does have extra
                arguments.

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

        # Remove units from any unyt_arrays and make contiguous
        props = [
            prop.value if isinstance(prop, unyt_array) else prop
            for prop in props
        ]
        props = [
            np.ascontiguousarray(prop, dtype=np.float64) for prop in props
        ]

        # For black holes mass is a grid parameter but we still need to
        # multiply by mass in the extensions so just multiply by 1
        mass = np.ones(1, dtype=np.float64)

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
            np.array([fesc]),
            grid_dims,
            len(grid_props),
            np.int32(1),
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
                The type of line to extract from the grid. This must match a
                type of line stored in the grid.
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
