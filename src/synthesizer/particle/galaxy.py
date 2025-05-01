"""A module containing all the funtionality for Particle based galaxies.

Like it's parametric variant this module contains the Galaxy object definition
from which all galaxy focused functionality can be performed. This variant uses
Particle objects, which can either be derived from simulation data or generated
from parametric models. A Galaxy can contain Stars, Gas, and / or BlackHoles.

Despite its name a Particle based Galaxy can be used for any collection of
particles to enable certain functionality (e.g. imaging of a galaxy group, or
spectra for all particles in a simulation.)

Example usage:

    galaxy = Galaxy(stars, gas, black_holes, ...)
    galaxy.stars.get_spectra(...)

"""

import copy

import numpy as np
from scipy.spatial import cKDTree
from unyt import Mpc, Msun, Myr, rad, unyt_quantity

from synthesizer import exceptions
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer.extensions.timers import tic, toc
from synthesizer.imaging import Image, SpectralCube
from synthesizer.parametric.stars import Stars as ParametricStars
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars
from synthesizer.synth_warnings import deprecated, warn
from synthesizer.units import accepts
from synthesizer.utils.geometry import get_rotation_matrix


class Galaxy(BaseGalaxy):
    """The Particle Galaxy class.

    When working with particles this object provides interfaces for calculating
    spectra, galaxy properties and images. A galaxy can be composed of any
    combination of particle.Stars, particle.Gas, or
    particle.BlackHoles objects.

    Attributes:
        stars (object, Stars):
            An instance of Stars containing the stellar particle data.
        gas (object, Gas):
            An instance of Gas containing the gas particle data.
        black_holes (object, BlackHoles):
            An instance of BlackHoles containing the black hole particle
            data.
        redshift (float):
            The redshift of the galaxy.
        centre (unyt_array of float):
            The centre of the galaxy particles. Can be defined in a number
            of ways (e.g. centre of mass, centre of potential, etc.)
        galaxy_type (str):
            A string describing the type of galaxy. This is set to "Particle"
            for this class.
    """

    @accepts(centre=Mpc)
    def __init__(
        self,
        name="particle galaxy",
        stars=None,
        gas=None,
        black_holes=None,
        redshift=None,
        centre=None,
        **kwargs,
    ):
        """Initialise a particle based Galaxy.

        Args:
            name (str):
                A name to identify the galaxy. Only used for external
                labelling, has no internal use.
            stars (object, Stars/Stars):
                An instance of Stars containing the stellar particle data
            gas (object, Gas):
                An instance of Gas containing the gas particle data.
            black_holes (object, BlackHoles):
                An instance of BlackHoles containing the black hole particle
                data.
            redshift (float):
                The redshift of the galaxy.
            centre (float):
                Centre of the galaxy particles. Can be defined in a number
                of ways (e.g. centre of mass)
            **kwargs (dict):
                Arbitrary keyword arguments.

        Raises:
            InconsistentArguments
        """
        # Check we haven't been given a SFZH
        if isinstance(stars, ParametricStars):
            raise exceptions.InconsistentArguments(
                "Parametric Stars passed instead of particle based Stars "
                "object. Did you mean synthesizer.parametric.Galaxy "
                "instead?"
            )

        # Set the type of galaxy
        self.galaxy_type = "Particle"

        # Instantiate the parent (load stars and gas below)
        BaseGalaxy.__init__(
            self,
            stars=stars,
            gas=gas,
            black_holes=black_holes,
            redshift=redshift,
            centre=centre,
        )

        # Define a name for this galaxy
        self.name = name

        # If we have them, record how many stellar / gas particles there are
        if self.stars:
            self.calculate_integrated_stellar_properties()

        if self.gas:
            self.calculate_integrated_gas_properties()

        # Attach any additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def calculate_integrated_stellar_properties(self):
        """Calculate integrated stellar properties."""
        # Define integrated properties of this galaxy
        if self.stars.current_masses is not None:
            self.stellar_mass = np.sum(self.stars.current_masses)

            if self.stars.ages is not None:
                self.stellar_mass_weighted_age = (
                    np.sum(self.stars.ages * self.stars.current_masses)
                    / self.stellar_mass
                )
            else:
                self.stellar_mass_weighted_age = None
                warn(
                    "Ages of stars not provided, "
                    "setting stellar_mass_weighted_age to `None`"
                )
        else:
            self.stellar_mass_weighted_age = None
            warn(
                "Current mass of stars not provided, "
                "setting stellar_mass_weighted_age to `None`"
            )

    def calculate_integrated_gas_properties(self):
        """Calculate integrated gas properties."""
        # Define integrated properties of this galaxy
        if self.gas.masses is not None:
            self.gas_mass = np.sum(self.gas.masses)

            # mass weighted gas phase metallicity
            self.mass_weighted_gas_metallicity = (
                np.sum(self.gas.masses * self.gas.metallicities)
                / self.gas_mass
            )
        else:
            self.mass_weighted_gas_metallicity = None
            warn(
                "Mass of gas particles not provided, "
                "setting mass_weighted_gas_metallicity to `None`"
            )

        if self.gas.star_forming is not None:
            mask = self.gas.star_forming
            if np.sum(mask) == 0:
                self.sf_gas_mass = 0.0
                self.sf_gas_metallicity = 0.0
            else:
                self.sf_gas_mass = np.sum(self.gas.masses[mask])

                # mass weighted gas phase metallicity
                self.sf_gas_metallicity = (
                    np.sum(
                        self.gas.masses[mask] * self.gas.metallicities[mask]
                    )
                    / self.sf_gas_mass
                )
        else:
            self.sf_gas_mass = None
            self.sf_gas_metallicity = None
            warn(
                "Star forming gas particle mask not provided, "
                "setting sf_gas_mass and sf_gas_metallicity to `None`"
            )

    @accepts(initial_masses=Msun.in_base("galactic"), ages=Myr)
    def load_stars(
        self,
        initial_masses=None,
        ages=None,
        metallicities=None,
        stars=None,
        **kwargs,
    ):
        """Load arrays for star properties into a `Stars`  object.

        This will populate the stars attribute with the instantiated Stars
        object.

        Args:
            initial_masses (unyt_array of float):
                Initial stellar particle masses (mass at birth), Msol
            ages (unyt_array of float):
                Star particle age, Myr
            metallicities (unyt_array of float):
                Star particle metallicity (total metal fraction)
            stars (Stars):
                A pre-existing stars particle object to use. Defaults to None.
            **kwargs (dict):
                Arbitrary keyword arguments.

        Returns:
            None
        """
        if stars is not None:
            # Add Stars particle object to this galaxy
            self.stars = stars
        else:
            # If nothing has been provided, just set to None and return
            if (
                (initial_masses is None)
                | (ages is None)
                | (metallicities is None)
            ):
                warn(
                    "In `load_stars`: one of either `initial_masses`"
                    ", `ages` or `metallicities` is not provided, setting "
                    "`stars` object to `None`"
                )
                self.stars = None
                return None
            else:
                # Create a new Stars object from particle arrays
                self.stars = Stars(
                    initial_masses, ages, metallicities, **kwargs
                )

        self.calculate_integrated_stellar_properties()

        # Assign additional galaxy-level properties
        self.stars.redshift = self.redshift
        if self.centre is not None:
            self.stars.centre = self.centre

    @accepts(masses=Msun.in_base("galactic"))
    def load_gas(
        self,
        masses=None,
        metallicities=None,
        gas=None,
        **kwargs,
    ):
        """Load arrays for gas particle properties into a `Gas` object.

        This will populate the gas attribute with the instantiated Gas object.

        Args:
            masses (unyt_array of float):
                Gas particle masses.
            metallicities (unyt_array of float):
                Gas particle metallicities (total metal fraction).
            gas (Gas):
                A pre-existing gas particle object to use. Defaults to None.
            **kwargs (dict):
                Arbitrary keyword arguments.
        """
        if gas is not None:
            # Add Gas particle object to this galaxy
            self.gas = gas
        else:
            # If nothing has been provided, just set to None and return
            if (masses is None) | (metallicities is None):
                warn(
                    "In `load_gas`: one of either `masses`"
                    " or `metallicities` is not provided, setting "
                    "`gas` object to `None`"
                )
                self.gas = None
                return None
            else:
                # Create a new `gas` object from particle arrays
                self.gas = Gas(masses, metallicities, **kwargs)

        self.calculate_integrated_gas_properties()

        # Assign additional galaxy-level properties
        self.gas.redshift = self.redshift
        if self.centre is not None:
            self.gas.centre = self.centre

    def calculate_black_hole_metallicity(self, default_metallicity=0.012):
        """Calculate the metallicity of the region surrounding a black hole.

        This is defined as the mass weighted average metallicity of all gas
        particles whose SPH kernels intersect the black holes position.

        Args:
            default_metallicity (float):
                The metallicity value used when no gas particles are in range
                of the black hole. The default is solar metallcity.
        """
        # Ensure we actually have Gas and black holes
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "Calculating the metallicity of the region surrounding the "
                "black hole requires a Galaxy to be intialised with a Gas "
                "object!"
            )
        if self.black_holes is None:
            raise exceptions.InconsistentArguments(
                "This Galaxy does not have a black holes object!"
            )

        # Construct a KD-Tree to efficiently get all gas particles which
        # intersect the black hole
        tree = cKDTree(self.gas._coordinates)

        # Query the tree for gas particles in range of each black hole, here
        # we use the maximum smoothing length to get all possible intersections
        # without calculating the distance for every gas particle.
        inds = tree.query_ball_point(
            self.black_holes._coordinates, r=self.gas._smoothing_lengths.max()
        )

        # Loop over black holes
        metallicities = np.zeros(self.black_holes.nbh)
        for ind, gas_in_range in enumerate(inds):
            # Handle black holes with no neighbouring gas
            if len(gas_in_range) == 0:
                metallicities[ind] = default_metallicity

            # Calculate the separation between the black hole and gas particles
            sep = (
                self.gas._coordinates[gas_in_range, :]
                - self.black_holes._coordinates[ind, :]
            )

            dists = np.sqrt(sep[:, 0] ** 2 + sep[:, 1] ** 2 + sep[:, 2] ** 2)

            # Get only the gas particles with smoothing lengths that intersect
            okinds = dists < self.gas._smoothing_lengths[gas_in_range]
            gas_in_range = np.array(gas_in_range, dtype=int)[okinds]

            # The above operation can remove all gas neighbours...
            if len(gas_in_range) == 0:
                metallicities[ind] = default_metallicity
                continue

            # Calculate the mass weight metallicity of this black holes region
            metallicities[ind] = np.average(
                self.gas.metallicities[gas_in_range],
                weights=self.gas._masses[gas_in_range],
            )

        # Assign the metallicity we have found
        self.black_holes.metallicities = metallicities

    def integrate_particle_spectra(self):
        """Integrate all particle spectra on any attached components."""
        # Handle stellar spectra
        if self.stars is not None:
            self.stars.integrate_particle_spectra()

        # Handle black hole spectra
        if self.black_holes is not None:
            self.black_holes.integrate_particle_spectra()

        # Handle gas spectra
        if self.gas is not None:
            # Nothing to do here... YET
            pass

    def get_stellar_los_tau_v(
        self,
        kappa,
        kernel,
        tau_v_attr="tau_v",
        mask=None,
        threshold=1,
        force_loop=0,
        min_count=100,
        nthreads=1,
    ):
        """Calculate the LOS optical depth for each star particle.

        This will calculate the optical depth for each star particle based on
        the gas particle distribution. The stars are considered to interact
        with a gas particle if gas_z > star_z and the star postion is within
        the SPH kernel of the gas particle.

        Note: the resulting tau_vs will be associated to the stars object at
        self.stars.tau_v.

        Args:
            kappa (float):
                The dust opacity in units of Msun / pc**2.
            kernel (np.ndarray of float):
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            tau_v_attr (str):
                The attribute to store the tau_v values in the stars object.
                Defaults to "tau_v".
            mask (bool):
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float):
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool):
                By default (False) the C function will only loop over nearby
                gas particles to search for contributions to the LOS surface
                density. This forces the loop over *all* gas particles.
            min_count (int):
                The minimum number of particles in a leaf cell of the tree
                used to search for gas particles. Can be used to tune the
                performance of the tree search in extreme cases. If there are
                fewer particles in a leaf cell than this value, the search
                will be performed with a brute force loop.
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.
        """
        start = tic()

        # Ensure we have stars and gas
        if self.stars is None:
            raise exceptions.InconsistentArguments(
                "No Stars object has been provided! We can't calculate line "
                "of sight dust attenuation without a Stars object containing "
                "the stellar particles!"
            )
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "No Gas object has been provided! We can't calculate line of "
                "sight dust attenuation without a Gas object containing the "
                "dust!"
            )

        # Compute the dust surface densities
        los_dustsds = self.stars.get_los_column_density(
            self.gas,
            "dust_masses",
            kernel,
            mask=mask,
            threshold=threshold,
            force_loop=force_loop,
            min_count=min_count,
            nthreads=nthreads,
        )  # Msun / Mpc**2

        los_dustsds /= (1e6) ** 2  # Msun / pc**2

        # Finalise the calculation
        tau_v = kappa * los_dustsds

        # Apply the mask if provided
        if mask is not None:
            tau_vs = np.zeros(self.stars.nparticles)
            tau_vs[mask] = tau_v
        else:
            tau_vs = tau_v

        # Store the result in self.stars
        setattr(self.stars, tau_v_attr, tau_vs)

        toc("Calculating stellar LOS tau_v", start)

        return tau_v

    def get_black_hole_los_tau_v(
        self,
        kappa,
        kernel,
        tau_v_attr="tau_v",
        mask=None,
        threshold=1,
        force_loop=0,
        min_count=100,
        nthreads=1,
    ):
        """Calculate the LOS optical depth for each black hole particle.

        This will calculate the optical depth for each black hole particle
        based on the gas particle distribution. The black holes are considered
        to interact with a gas particle if gas_z > black_hole_z and the black
        hole postion is within the SPH kernel of the gas particle.

        Note: the resulting tau_vs will be associated to the black_holes object
        at self.black_holes.tau_v.

        Args:
            kappa (float):
                The dust opacity in units of Msun / pc**2.
            kernel (np.ndarray of float):
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            tau_v_attr (str):
                The attribute to store the tau_v values in the black_holes
                object. Defaults to "tau_v".
            mask (np.ndarray of bool):
                A mask to be applied to the black holes. Surface densities will
                only be computed and returned for black holes with True in the
                mask.
            threshold (float):
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool):
                By default (False) the C function will only loop over nearby
                gas particles to search for contributions to the LOS surface
                density. This forces the loop over *all* gas particles.
            min_count (int):
                The minimum number of particles in a leaf cell of the tree
                used to search for gas particles. Can be used to tune the
                performance of the tree search in extreme cases. If there are
                fewer particles in a leaf cell than this value, the search
                will be performed with a brute force loop.
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.
        """
        start = tic()

        # Ensure we have black holes and gas
        if self.black_holes is None:
            raise exceptions.InconsistentArguments(
                "No BlackHoles object has been provided! We can't calculate "
                "line of sight dust attenuation without a BlackHoles object "
                "containing the black hole particles!"
            )
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "No Gas object has been provided! We can't calculate line of "
                "sight dust attenuation without a Gas object containing the "
                "dust!"
            )

        # Compute the dust surface densities
        los_dustsds = self.black_holes.get_los_column_density(
            self.gas,
            "dust_masses",
            kernel,
            mask=mask,
            threshold=threshold,
            force_loop=force_loop,
            min_count=min_count,
            nthreads=nthreads,
        )

        los_dustsds /= (1e6) ** 2  # Msun / pc**2

        # Finalise the calculation
        tau_v = kappa * los_dustsds

        # Apply the mask if provided
        if mask is not None:
            tau_vs = np.zeros(self.black_holes.nbh)
            tau_vs[mask] = tau_v
        else:
            tau_vs = tau_v

        # Store the result in self.black_holes
        setattr(self.black_holes, "tau_v", tau_vs)

        toc("Calculating black hole LOS tau_v", start)

        return tau_v

    @deprecated()
    def calculate_los_tau_v(
        self,
        kappa,
        kernel,
        tau_v_attr="tau_v",
        mask=None,
        threshold=1,
        force_loop=0,
        min_count=100,
        nthreads=1,
    ):
        """Calculate the LOS optical depth for each star particle.

        This will calculate the optical depth for each star particle based on
        the gas particle distribution. The stars are considered to interact
        with a gas particle if gas_z > star_z and the star postion is within
        the SPH kernel of the gas particle.

        Note: the resulting tau_vs will be associated to the stars object at
        self.stars.tau_v.

        Args:
            kappa (float):
                The dust opacity in units of Msun / pc**2.
            kernel (np.ndarray of float):
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            tau_v_attr (str):
                The attribute to store the tau_v values in the stars object.
                Defaults to "tau_v".
            mask (bool):
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float):
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool):
                By default (False) the C function will only loop over nearby
                gas particles to search for contributions to the LOS surface
                density. This forces the loop over *all* gas particles.
            min_count (int):
                The minimum number of particles in a leaf cell of the tree
                used to search for gas particles. Can be used to tune the
                performance of the tree search in extreme cases. If there are
                fewer particles in a leaf cell than this value, the search
                will be performed with a brute force loop.
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.
        """
        start = tic()

        # Ensure we have stars and gas
        if self.stars is None:
            raise exceptions.InconsistentArguments(
                "No Stars object has been provided! We can't calculate line "
                "of sight dust attenuation without a Stars object containing "
                "the stellar particles!"
            )
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "No Gas object has been provided! We can't calculate line of "
                "sight dust attenuation without a Gas object containing the "
                "dust!"
            )

        # Compute the dust surface densities
        los_dustsds = self.stars.get_los_column_density(
            self.gas,
            "dust_masses",
            kernel,
            mask=mask,
            threshold=threshold,
            force_loop=force_loop,
            min_count=min_count,
            nthreads=nthreads,
        )  # Msun / Mpc**2

        los_dustsds /= (1e6) ** 2  # Msun / pc**2

        # Finalise the calculation
        tau_v = kappa * los_dustsds

        # Apply the mask if provided
        if mask is not None:
            tau_vs = np.zeros(self.stars.nparticles)
            tau_vs[mask] = tau_v
        else:
            tau_vs = tau_v

        # Store the result in self.stars
        setattr(self.stars, tau_v_attr, tau_vs)

        toc("Calculating LOS tau_v", start)

        return tau_v

    def calculate_dust_screen_gamma(
        self,
        gamma_min=0.01,
        gamma_max=1.8,
        beta=0.1,
        Z_norm=0.035,
        sf_gas_metallicity=None,
        sf_gas_mass=None,
        stellar_mass=None,
    ):
        """Calculate the optical depth gamma parameter.

        Gamma is a parametrisation for controlling the optical depth
        due to dust dependent on the mass and metallicity of star forming
        gas.

        gamma = gamma_max - (gamma_max - gamma_min) / C

        C = 1 + (Z_SF / Z_MW) * (M_SF / M_star) * (1 / beta)

        gamma_max and gamma_min set the upper and lower bounds to which gamma
        asymptotically approaches where the star forming gas mass is high (low)
        and the star forming gas metallicity is high (low), respectively.

        Z_SF is the star forming gas metallicity, Z_MW is the Milky
        Way value (defaults to value from Zahid+14), M_SF is the star forming
        gas mass, M_star is the stellar mass, and beta is a normalisation
        value.

        The gamma array can be used directly in attenuation methods.

        Zahid+14:
        https://iopscience.iop.org/article/10.1088/0004-637X/791/2/130

        Args:
            gamma_min (float):
                Lower limit of the gamma parameter.
            gamma_max (float):
                Upper limit of the gamma parameter.
            beta (float):
                Normalisation value, default 0.1
            Z_norm (float):
                Metallicity normsalition value, defaults to Zahid+14
                value for the Milky Way (0.035)
            sf_gas_metallicity (array):
                Custom star forming gas metallicity array. If None,
                defaults to value attached to this galaxy object.
            sf_gas_mass (array):
                Custom star forming gas mass array, units Msun. If
                None, defaults to value attached to this galaxy object.
            stellar_mass (array):
                Custom stellar mass array, units Msun. If None,
                defaults to value attached to this galaxy object.

        Returns:
            gamma (array):
                Dust attentuation scaling parameter for this galaxy
        """
        if sf_gas_metallicity is None:
            if self.sf_gas_metallicity is None:
                raise ValueError("No sf_gas_metallicity provided")
            else:
                sf_gas_metallicity = self.sf_gas_metallicity

        if sf_gas_mass is None:
            if self.sf_gas_mass is None:
                raise ValueError("No sf_gas_mass provided")
            else:
                sf_gas_mass = self.sf_gas_mass.value  # Msun

        if stellar_mass is None:
            if self.stellar_mass is None:
                raise ValueError("No stellar_mass provided")
            else:
                stellar_mass = self.stellar_mass.value  # Msun

        if sf_gas_mass == 0.0:
            gamma = gamma_min
        elif stellar_mass == 0.0:
            gamma = gamma_min
        else:
            C = 1 + (sf_gas_metallicity / Z_norm) * (
                sf_gas_mass / stellar_mass
            ) * (1.0 / beta)
            gamma = gamma_max - (gamma_max - gamma_min) / C

        return gamma

    @accepts(stellar_mass_weighted_age=Myr)
    def calculate_dust_to_metal_vijayan19(
        self,
        stellar_mass_weighted_age=None,
        ism_metallicity=None,
    ):
        """Calculate the dust to metal ratio from stellar age and metallicity.

        This uses a fitting function for the dust-to-metals ratio based on
        galaxy properties, from L-GALAXIES dust modeling.

        Vijayan+19: https://arxiv.org/abs/1904.02196

        Note this will recalculate the dust masses based on the new dust-to-
        metal ratio.

        Args:
            stellar_mass_weighted_age (float):
                Mass weighted age of stars in Myr. Defaults to None,
                and uses value provided on this galaxy object (in Gyr)
            ism_metallicity (float):
                Mass weighted gas-phase metallicity. Defaults to None,
                and uses value provided on this galaxy object
                (dimensionless)
        """
        # Ensure we have what we need for the calculation
        if stellar_mass_weighted_age is None:
            if self.stellar_mass_weighted_age is None:
                raise ValueError("No stellar_mass_weighted_age provided")
            else:
                # Formula uses Age in Gyr while the supplied Age is in Myr
                stellar_mass_weighted_age = (
                    self.stellar_mass_weighted_age.value / 1e6
                )  # Myr

        if ism_metallicity is None:
            if self.mass_weighted_gas_metallicity is None:
                raise ValueError("No mass_weighted_gas_metallicity provided")
            else:
                ism_metallicity = self.mass_weighted_gas_metallicity

        # Fixed parameters from Vijayan+21
        D0, D1, alpha, beta, gamma = 0.008, 0.329, 0.017, -1.337, 2.122
        tau = 5e-5 / (D0 * ism_metallicity)
        dtm = D0 + (D1 - D0) * (
            1.0
            - np.exp(
                -alpha
                * (ism_metallicity**beta)
                * ((stellar_mass_weighted_age / (1e3 * tau)) ** gamma)
            )
        )
        if np.isnan(dtm) or np.isinf(dtm):
            dtm = 0.0

        # Save under gas properties
        self.gas.dust_to_metal_ratio = dtm

        # We need to recalculate the dust masses so things don't end up
        # inconsistent (dust_masses are automatically calculated at
        # intialisation). If the user handed dust masses and then called this
        # function, they will be overwritten and it will be confusing but
        # that's so unlikely and they'll work out when they see this comment.
        self.gas.dust_masses = (
            self.gas.masses
            * self.gas.metallicities
            * self.gas.dust_to_metal_ratio
        )

        return dtm

    def get_map_stellar_mass(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
    ):
        """Make a mass map, either with or without smoothing.

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The stellar mass image.
        """
        # Instantiate the Image object.
        img = Image(
            resolution=resolution,
            fov=fov,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_img_hist(
                signal=self.stars.current_masses,
                coordinates=self.stars.centered_coordinates,
            )

        elif img_type == "smoothed":
            # Compute image
            img.get_img_smoothed(
                signal=self.stars.current_masses,
                coordinates=self.stars.centered_coordinates,
                smoothing_lengths=self.stars.smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        return img

    def get_map_gas_mass(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
    ):
        """Make a mass map, either with or without smoothing.

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The gas mass image.
        """
        # Instantiate the Image object.
        img = Image(
            resolution=resolution,
            fov=fov,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_img_hist(
                signal=self.gas.masses,
                coordinates=self.gas.centered_coordinates,
            )

        elif img_type == "smoothed":
            # Compute image
            img.get_img_smoothed(
                signal=self.gas.masses,
                coordinates=self.gas.centered_coordinates,
                smoothing_lengths=self.gas.smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        return img

    def get_map_stellar_age(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
    ):
        """Make an age map, either with or without smoothing.

        The age in a pixel is the initial mass weighted average age in that
        pixel.

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The stellar age image.
        """
        # Instantiate the Image object.
        weighted_img = Image(
            resolution=resolution,
            fov=fov,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            weighted_img.get_img_hist(
                signal=self.stars.ages,
                normalisation=self.stars.initial_masses,
                coordinates=self.stars.centered_coordinates,
            )

        elif img_type == "smoothed":
            # Compute image
            weighted_img.get_img_smoothed(
                signal=self.stars.ages,
                normalisation=self.stars.initial_masses,
                coordinates=self.stars.centered_coordinates,
                smoothing_lengths=self.stars.smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        return weighted_img

    def get_map_stellar_metal_mass(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
    ):
        """Make a stellar metal mass map, either with or without smoothing.

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The stellar metal mass image.
        """
        # Instantiate the Image object.
        img = Image(
            resolution=resolution,
            fov=fov,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_img_hist(
                signal=self.stars.metallicities * self.stars.masses,
                coordinates=self.stars.centered_coordinates,
            )

        elif img_type == "smoothed":
            # Compute image
            img.get_img_smoothed(
                signal=self.stars.metallicities * self.stars.masses,
                coordinates=self.stars.centered_coordinates,
                smoothing_lengths=self.stars.smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        return img

    def get_map_gas_metal_mass(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
    ):
        """Make a gas metal mass map, either with or without smoothing.

        TODO: make dust map!

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The gas metal mass image.
        """
        # Instantiate the Image object.
        img = Image(
            resolution=resolution,
            fov=fov,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_img_hist(
                signal=self.gas.metallicities * self.gas.masses,
                coordinates=self.gas.centered_coordinates,
            )

        elif img_type == "smoothed":
            # Compute image
            img.get_img_smoothed(
                signal=self.gas.metallicities * self.gas.masses,
                coordinates=self.gas.centered_coordinates,
                smoothing_lengths=self.gas.smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        return img

    def get_map_stellar_metallicity(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
    ):
        """Make a stellar metallicity map, either with or without smoothing.

        The metallicity in a pixel is the mass weighted average metallicity in
        that pixel.

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The stellar metallicity image.
        """
        # Make the weighted image
        weighted_img = self.get_map_stellar_metal_mass(
            resolution,
            fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
        )

        # Make the mass image
        mass_img = self.get_map_stellar_mass(
            resolution, fov, img_type, kernel, kernel_threshold, nthreads
        )

        # Divide out the mass contribution, handling zero contribution pixels
        img = weighted_img.arr
        img[img > 0] /= mass_img.arr[mass_img.arr > 0]
        img *= self.stars.ages.units

        return Image(
            resolution=resolution,
            fov=fov,
            img=img,
        )

    def get_map_gas_metallicity(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        nthreads=1,
    ):
        """Make a gas metallicity map, either with or without smoothing.

        The metallicity in a pixel is the mass weighted average metallicity in
        that pixel.

        TODO: make dust map!

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The gas metallicity image.
        """
        # Make the weighted image
        weighted_img = self.get_map_gas_metal_mass(
            resolution,
            fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            nthreads=nthreads,
        )

        # Make the mass image
        mass_img = self.get_map_gas_mass(
            resolution, fov, img_type, kernel, kernel_threshold, nthreads
        )

        # Divide out the mass contribution, handling zero contribution pixels
        img = weighted_img.arr
        img[img > 0] /= mass_img.arr[mass_img.arr > 0]
        img *= self.stars.ages.units

        return Image(
            resolution=resolution,
            fov=fov,
            img=img,
        )

    def get_map_sfr(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        age_bin=100 * Myr,
        nthreads=1,
    ):
        """Make a star formation rate map, either with or without smoothing.

        Only stars younger than age_bin are included in the map. This is
        calculated by computing the initial mass map for stars in the age bin
        and then dividing by the size of the age bin.

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            age_bin (unyt_quantity/float):
                The size of the age bin used to calculate the star formation
                rate. If supplied without units, the unit system is assumed.
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The SFR image.
        """
        # Convert the age bin if necessary
        if isinstance(age_bin, unyt_quantity):
            if age_bin.units != self.stars.ages.units:
                age_bin = age_bin.to(self.stars.ages.units)
        else:
            age_bin *= self.stars.ages.units

        # Get the mask for stellar particles in the age bin
        mask = self.stars.ages < age_bin

        #  Warn if we have stars to plot in this bin
        if self.stars.ages[mask].size == 0:
            warn("The SFR is 0! (there are 0 stars in the age bin)")

        # Instantiate the Image object.
        img = Image(resolution=resolution, fov=fov)

        # Make the initial mass map, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_img_hist(
                signal=self.stars.initial_masses[mask],
                coordinates=self.stars.centered_coordinates[mask, :],
            )

        elif img_type == "smoothed":
            # Compute image
            img.get_img_smoothed(
                signal=self.stars.initial_masses[mask],
                coordinates=self.stars.centered_coordinates[mask, :],
                smoothing_lengths=self.stars.smoothing_lengths[mask],
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        # Convert the initial mass map to SFR
        img.arr /= age_bin.value
        img.units = img.units / age_bin.units

        return img

    def get_map_ssfr(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
        age_bin=100 * Myr,
        nthreads=1,
    ):
        """Make a specific star formation rate map.

        Only stars younger than age_bin are included in the map. This is
        calculated by computing the initial mass map for stars in the age bin
        and then dividing by the size of the age bin and stellar mass of
        the galaxy.

        Args:
            resolution (float):
                The size of a pixel.
            fov (float):
                The width of the image in image coordinates.
            img_type (str):
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            age_bin (unyt_quantity/float):
                The size of the age bin used to calculate the star formation
                rate. If supplied without units, the unit system is assumed.
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            Image: The sSFR image.
        """
        # Convert the age bin if necessary
        if isinstance(age_bin, unyt_quantity):
            if age_bin.units != self.stars.ages.units:
                age_bin = age_bin.to(self.stars.ages.units)
        else:
            age_bin *= self.stars.ages.units

        # Get the mask for stellar particles in the age bin
        mask = self.stars.ages < age_bin

        #  Warn if we have stars to plot in this bin
        if self.stars.ages[mask].size == 0:
            warn("The SFR is 0! (there are 0 stars in the age bin)")

        # Instantiate the Image object.
        img = Image(resolution=resolution, fov=fov)

        # Make the initial mass map, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_img_hist(
                signal=self.stars.initial_masses[mask],
                coordinates=self.stars.centered_coordinates[mask, :],
                normalisation=self.stars.current_masses[mask] / age_bin,
            )

        elif img_type == "smoothed":
            # Compute image
            img.get_img_smoothed(
                signal=self.stars.initial_masses[mask],
                coordinates=self.stars.centered_coordinates[mask, :],
                smoothing_lengths=self.stars.smoothing_lengths[mask],
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
                normalisation=self.stars.current_masses[mask] / age_bin,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        return img

    def get_data_cube(
        self,
        resolution,
        fov,
        lam,
        cube_type="hist",
        stellar_spectra=None,
        blackhole_spectra=None,
        kernel=None,
        kernel_threshold=1,
        quantity="lnu",
        nthreads=1,
    ):
        """Make a SpectralCube from an Sed held by this galaxy.

        Data cubes are calculated by smoothing spectra over the component
        morphology. The Sed used is defined by <component>_spectra.

        If multiple components are requested they will be combined into a
        single output data cube.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (unyt_quantity, float):
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov (unyt_quantity, float):
                The width of the image in image coordinates.
            lam (unyt_array, float):
                The wavelength array to use for the data cube.
            cube_type (str):
                The type of data cube to make. Either "smoothed" to smooth
                particle spectra over a kernel or "hist" to sort particle
                spectra into individual spaxels.
            stellar_spectra (str):
                The stellar spectra key to make into a data cube.
            blackhole_spectra (str):
                The black hole spectra key to make into a data cube.
            kernel (np.ndarray of float):
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float):
                The kernel's impact parameter threshold (by default 1).
            quantity (str):
                The Sed attribute/quantity to sort into the data cube, i.e.
                "lnu", "llam", "luminosity", "fnu", "flam" or "flux".
            nthreads (int):
                The number of threads to use in the tree search. Default is 1.

        Returns:
            SpectralCube:
                The spectral data cube object containing the derived
                data cube.
        """
        start = tic()

        # Make sure we have an image to make
        if stellar_spectra is None and blackhole_spectra is None:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(stellar_spectra or blackhole_spectra)!"
                " What component/s do you want a data cube of?"
            )

        # Make stellar image if requested
        if stellar_spectra is not None:
            # Instantiate the Image colection ready to make the image.
            stellar_cube = SpectralCube(
                resolution=resolution, fov=fov, lam=lam
            )

            # Make the image using the requested method
            if cube_type == "hist":
                stellar_cube.get_data_cube_hist(
                    sed=self.stars.particle_spectra[stellar_spectra],
                    coordinates=self.stars.centered_coordinates,
                    quantity=quantity,
                )
            else:
                stellar_cube.get_data_cube_smoothed(
                    sed=self.stars.particle_spectra[stellar_spectra],
                    coordinates=self.stars.centered_coordinates,
                    smoothing_lengths=self.stars.smoothing_lengths,
                    kernel=kernel,
                    kernel_threshold=kernel_threshold,
                    quantity=quantity,
                    nthreads=nthreads,
                )

        # Make blackhole image if requested
        if blackhole_spectra is not None:
            # Instantiate the Image colection ready to make the image.
            blackhole_cube = SpectralCube(
                resolution=resolution, fov=fov, lam=lam
            )

            # Make the image using the requested method
            if cube_type == "hist":
                blackhole_cube.get_data_cube_hist(
                    sed=self.blackhole.particle_spectra[blackhole_spectra],
                    coordinates=self.blackhole.centered_coordinates,
                    quantity=quantity,
                )
            else:
                blackhole_cube.get_data_cube_smoothed(
                    sed=self.blackhole.particle_spectra[blackhole_spectra],
                    coordinates=self.blackhole.centered_coordinates,
                    smoothing_lengths=self.blackhole.smoothing_lengths,
                    kernel=kernel,
                    kernel_threshold=kernel_threshold,
                    quantity=quantity,
                    nthreads=nthreads,
                )

        # Return the images, combining if there are multiple components
        if stellar_spectra is not None and blackhole_spectra is not None:
            toc("Computing stellar and blackhole spectral data cubes", start)
            return stellar_cube + blackhole_cube
        elif stellar_spectra is not None:
            toc("Computing stellar spectral data cube", start)
            return stellar_cube
        toc("Computing blackhole spectral data cube", start)
        return blackhole_cube

    @accepts(phi=rad, theta=rad)
    def rotate_particles(
        self,
        phi=0 * rad,
        theta=0 * rad,
        rot_matrix=None,
        inplace=True,
    ):
        """Rotate coordinates.

        This method can either use angles or a provided rotation matrix.

        When using angles phi is the rotation around the z-axis while theta
        is the rotation around the y-axis.

        This can both be done in place or return a new instance, by default
        this will be done in place.

        Args:
            phi (unyt_quantity):
                The angle in radians to rotate around the z-axis. If rot_matrix
                is defined this will be ignored.
            theta (unyt_quantity):
                The angle in radians to rotate around the y-axis. If rot_matrix
                is defined this will be ignored.
            rot_matrix (np.ndarray of float):
                A 3x3 rotation matrix to apply to the coordinates
                instead of phi and theta.
            inplace (bool):
                Whether to perform the rotation in place or return a new
                instance.

        Returns:
            Particles
                A new instance of the particles with the rotated coordinates,
                if inplace is False.
        """
        # Are we rotating in place?
        if inplace:
            gal = self
        else:
            gal = copy.deepcopy(self)

        # Do the stars
        if gal.stars is not None:
            gal.stars.rotate_particles(
                phi=phi,
                theta=theta,
                rot_matrix=rot_matrix,
                inplace=True,
            )

        # Do the gas
        if gal.gas is not None:
            gal.gas.rotate_particles(
                phi=phi,
                theta=theta,
                rot_matrix=rot_matrix,
                inplace=True,
            )

        # Do the black holes
        if gal.black_holes is not None:
            gal.black_holes.rotate_particles(
                phi=phi,
                theta=theta,
                rot_matrix=rot_matrix,
                inplace=True,
            )

        # If we aren't rotating in place we need to return a new instance
        if not inplace:
            return gal
        return

    def rotate_edge_on(self, component="stars", inplace=True):
        """Rotate the particle distribution to edge-on.

        This will rotate the particle distribution such that the angular
        momentum vector is aligned with the y-axis in an image

        Args:
            component (str):
                The component whose angular momentum vector should be used for
                the rotation. Options are "stars", "gas" and "black_holes".
            inplace (bool):
                Whether to perform the rotation in place or return a new
                instance.

        Returns:
            Particles
                A new instance of the particles with rotated coordinates,
                if inplace is False.
        """
        # Get the angular momentum to rotate towards
        angular_momentum = getattr(self, component).angular_momentum

        # Get the rotation matrix to rotate ang_mom_hat to the y-axis
        rot_matrix = get_rotation_matrix(
            angular_momentum,
            np.array([1, 0, 0]),
        )

        # Call the rotate_particles method with the computed angles
        return self.rotate_particles(rot_matrix=rot_matrix, inplace=inplace)

    def rotate_face_on(self, component="stars", inplace=True):
        """Rotate the particle distribution to face-on.

        This will rotate the particle distribution such that the angular
        momentum vector is aligned with the z-axis in an image.

        Args:
            component (str):
                The component whose angular momentum vector should be used for
                the rotation. Options are "stars", "gas" and "black_holes".
            inplace (bool):
                Whether to perform the rotation in place or return a new
                instance.

        Returns:
            Particles
                A new instance of the particles with rotated coordinates,
                if inplace is False.
        """
        # Get the angular momentum to rotate towards
        angular_momentum = getattr(self, component).angular_momentum

        # Get the rotation matrix to rotate ang_mom_hat to the z-axis
        rot_matrix = get_rotation_matrix(
            angular_momentum,
            np.array([0, 0, -1]),
        )

        # Call the rotate_particles method with the computed angles
        return self.rotate_particles(rot_matrix=rot_matrix, inplace=inplace)
