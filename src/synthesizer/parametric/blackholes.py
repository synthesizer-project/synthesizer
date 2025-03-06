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
from synthesizer.components.blackhole import BlackholesComponent
from synthesizer.parametric.morphology import PointSource
from synthesizer.units import accepts


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
        fesc=None,
        **kwargs,
    ):
        """
        Initialize a BlackHole instance.
        
        Initializes a parametric black hole with physical properties such as mass, accretion rate,
        radiative efficiency, inclination, and spin. Configurable parameters for the broad and narrow
        line regions and the torus angle allow for tailored emission models. By default, the black hole
        is represented as a single particle (nparticles=1 and nbh=1) and its morphology is set as a
        point source using the specified offset.
        
        Args:
            mass (float, optional): Mass of each particle in solar masses.
            accretion_rate (float, optional): Accretion rate in solar masses per year.
            epsilon (float, optional): Radiative efficiency (default: 0.1).
            inclination (float, optional): Inclination angle, required for certain disc models.
            spin (float, optional): Black hole spin, required for some disc models.
            offset (unyt_array, optional): (x, y) offset relative to the image center 
                (default: np.array([0.0, 0.0]) * kpc).
            bolometric_luminosity (float, optional): Bolometric luminosity.
            metallicity (float, optional): Metallicity of the surrounding region.
            ionisation_parameter_blr (float or array-like, optional): Ionisation parameter of the broad line region (default: 0.1).
            hydrogen_density_blr (float or array-like, optional): Hydrogen density of the broad line region (default: 1e9 / cm**3).
            covering_fraction_blr (float or array-like, optional): Covering fraction of the broad line region (default: 0.1).
            velocity_dispersion_blr (float or array-like, optional): Velocity dispersion in the broad line region (default: 2000 * km / s).
            ionisation_parameter_nlr (float or array-like, optional): Ionisation parameter of the narrow line region (default: 0.01).
            hydrogen_density_nlr (float or array-like, optional): Hydrogen density of the narrow line region (default: 1e4 / cm**3).
            covering_fraction_nlr (float or array-like, optional): Covering fraction of the narrow line region (default: 0.1).
            velocity_dispersion_nlr (float or array-like, optional): Velocity dispersion in the narrow line region (default: 500 * km / s).
            theta_torus (float or array-like, optional): Angular size of the torus (default: 10 * deg).
            fesc (float or array-like, optional): Escape fraction; if None, defaults to 0.0.
            **kwargs: Additional keyword arguments for overriding emission model defaults.
        """
        # Initialise base class
        BlackholesComponent.__init__(
            self,
            fesc=fesc,
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

        # By default a parametric black hole will explictily have 1 "particle",
        # set this here so that the downstream extraction can access the
        # attribute.
        self.nparticles = 1
        self.nbh = 1

        # Initialise morphology using the in-built point-source class
        self.morphology = PointSource(offset=offset)

    def get_mask(self, attr, thresh, op, mask=None):
        """
        Generate a boolean mask by comparing an attribute to a threshold.
        
        This method retrieves a specified attribute from the instance and computes a boolean
        mask by comparing its value to a given threshold using the provided operator. Supported
        operators are '<', '>', '<=', '>=', '==', and '!='. If an optional mask is supplied, the
        resulting mask is combined with it via a logical AND operation.
        
        Args:
            attr (str): Name of the attribute used for creating the mask.
            thresh (float): Threshold value for the comparison.
            op (str): Comparison operator; must be one of '<', '>', '<=', '>=', '==', or '!='.
            mask (numpy.ndarray, optional): Existing boolean mask to combine with.
        
        Raises:
            InconsistentArguments: If the provided operator is not one of the supported options.
        
        Returns:
            numpy.ndarray: A boolean mask indicating where the attribute meets the threshold condition.
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
        Prepare and format input arguments for the C extension that computes spectral lines.
        
        This function determines the line region (narrow or broad) from the grid name,
        establishes a default mask for a singular black hole if none is provided, extracts
        and normalizes the required black hole properties, and scales them to match the
        number of particles. It then constructs and returns a tuple containing the grid
        line and continuum arrays, contiguous property arrays from both the grid and the
        black hole, the bolometric luminosity, an escape fraction array, grid dimensions,
        the count of grid properties, the number of particles, the grid assignment method,
        and the number of threads to use.
        
        Args:
            grid: AGN grid object containing spectral line data.
            line_id (str): Identifier of the line to extract.
            line_type (str): Category of the line (e.g. blr or nlr) matching a type in the grid.
            fesc: Escape fraction of stellar emission, as a single float or an array.
            mask: Boolean array or None. If None, a default mask for a singular black hole is created.
            grid_assignment_method (str): Method for assigning particles to grid points (e.g., 'cic' or 'ngp').
            nthreads (int): Number of threads to use; if -1, uses all available CPU threads.
        
        Returns:
            Tuple containing:
                grid_line: Array of grid line luminosities.
                grid_continuum: Array of grid continuum values.
                grid_props: Tuple of contiguous grid property arrays.
                part_props: Tuple of contiguous, masked particle property arrays.
                bol_lum: Bolometric luminosity as a float.
                fesc: Contiguous array of escape fraction values.
                grid_dims: Array of grid dimensions.
                num_grid_props: Count of grid property arrays.
                npart: Number of particles determined from the mask.
                grid_assignment_method: The method used for grid assignment.
                nthreads: Number of threads to be used.
        
        Raises:
            InconsistentArguments: If the grid does not indicate a recognized line region or
                if a required black hole property for the specified line region is missing.
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

        # Handle the case where mask is None, we need to make a mask of size
        # 1 since a parametric blackhole is always singular
        if mask is None:
            mask = np.ones(1, dtype=bool)

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

        # For black holes the grid Sed are normalised to 1.0 so we need to
        # scale by the bolometric luminosity.
        bol_lum = self.bolometric_luminosity.value

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
            bol_lum,
            fesc,
            grid_dims,
            len(grid_props),
            npart,
            grid_assignment_method,
            nthreads,
        )
