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
import numpy as np
from unyt import rad

from synthesizer.particle.particles import Particles
from synthesizer.components import BlackholesComponent
from synthesizer.blackhole_emission_model import Template
from synthesizer import exceptions
from synthesizer.units import Quantity
from synthesizer.utils import value_to_array


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
    number of optional attributes which are set to None if not provided.

    Attributes:
        nbh (int)
            The number of black hole particles in the object.
        smoothing_lengths (array-like, float)
            The smoothing length describing the black holes neighbour kernel.
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
        "_bol_luminosity",
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
            accretion_rate (array-like, float)
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
        )
        BlackholesComponent.__init__(
            self,
            mass=masses,
            accretion_rate=accretion_rates,
            epsilon=epsilons,
            inclination=inclinations,
            spin=spins,
            metallicity=metallicities,
        )

        # Set a frontfacing clone of the number of particles with clearer
        # naming
        self.nbh = self.nparticles

        # Check the arguments we've been given
        self._check_bh_args()

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

    def _check_bh_args(self):
        """
        Sanitizes the inputs ensuring all arguments agree and are compatible.

        Raises:
            InconsistentArguments
                If any arguments are incompatible or not as expected an error
                is thrown.
        """

        # Ensure all arrays are the expected length
        for key in self.attrs:
            attr = getattr(self, key)
            if isinstance(attr, np.ndarray):
                if attr.shape[0] != self.nparticles:
                    raise exceptions.InconsistentArguments(
                        "Inconsistent black hole array sizes! (nparticles=%d, "
                        "%s=%d)" % (self.nparticles, key, attr.shape[0])
                    )

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
        grid_assignment_method,
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

        Returns:
            tuple
                A tuple of all the arguments required by the C extension.
        """

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(getattr(grid, axis), dtype=np.float64)
            for axis in grid.axes
        ]
        props = [
            np.ascontiguousarray(getattr(self, axis), dtype=np.float64)
            for axis in grid.axes
        ]

        # For black holes mass is a grid parameter but we still need to
        # multiply by mass in the extensions so just multiply by 1
        mass = np.ones(self.nbh, dtype=np.float64)

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

        return (
            grid_spectra,
            grid_props,
            props,
            mass,
            fesc,
            grid_dims,
            len(grid_props),
            np.int32(self.nbh),
            nlam,
            grid_assignment_method,
        )

    def generate_lnu(
        self,
        emission_model,
        spectra_name,
        agn_region,
        fesc=0.0,
        verbose=False,
        grid_assignment_method="cic",
        bolometric_luminosity=None,
    ):
        """
        Generate the integrated rest frame spectra for a given grid key
        spectra.
        """
        # Handle the special Template case where a spectra is scaled
        # and returned
        if isinstance(emission_model, Template):
            return emission_model.get_spectra(bolometric_luminosity)

        # Ensure we have a key in the grid. If not error.
        if spectra_name not in list(emission_model.grid[agn_region].spectra.keys()):
            raise exceptions.MissingSpectraType(
                f"The Grid does not contain the key '{spectra_name}'"
            )

        from ..extensions.integrated_spectra import compute_integrated_sed

        # Prepare the arguments for the C function.
        args = self._prepare_sed_args(
            emission_model.grid[agn_region],
            fesc=fesc,
            spectra_type=spectra_name,
            grid_assignment_method=grid_assignment_method.lower(),
        )

        # Get the integrated spectra in grid units (erg / s / Hz)
        return compute_integrated_sed(*args)

    def _generate_particle_lnu(
        self,
    ):
        """
        Get the particle spectra from the grid using either a CIC or NGP
        method.
        """
        pass
