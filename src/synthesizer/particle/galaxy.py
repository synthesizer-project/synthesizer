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

import numpy as np
from scipy.spatial import cKDTree
from unyt import Myr, unyt_quantity

from synthesizer import exceptions
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer.imaging import Image, ImageCollection, SpectralCube
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle import Gas, Stars
from synthesizer.warnings import warn


class Galaxy(BaseGalaxy):
    """The Particle based Galaxy object.

    When working with particles this object provides interfaces for calculating
    spectra, galaxy properties and images. A galaxy can be composed of any
    combination of particle.Stars, particle.Gas, or
    particle.BlackHoles objects.

    Attributes:

    """

    attrs = [
        "stars",
        "gas",
        "sf_gas_metallicity",
        "sf_gas_mass",
        "gas_mass",
    ]

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
        """Initialise a particle based Galaxy with objects derived from
           Particles.

        Args:
            name (str)
                A name to identify the galaxy. Only used for external
                labelling, has no internal use.
            stars (object, Stars/Stars)
                An instance of Stars containing the stellar particle data
            gas (object, Gas)
                An instance of Gas containing the gas particle data.
            black_holes (object, BlackHoles)
                An instance of BlackHoles containing the black hole particle
                data.
            redshift (float)
                The redshift of the galaxy.
            centre (float)
                Centre of the galaxy particles. Can be defined in a number
                of ways (e.g. centre of mass)
            **kwargs
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
            stars=None,
            gas=None,
            black_holes=black_holes,
            redshift=redshift,
            centre=centre,
        )

        # Manually load stars and gas at particle level
        self.load_stars(stars=stars)
        self.load_gas(gas=gas)

        # Define a name for this galaxy
        self.name = name

        # If we have them, record how many stellar / gas particles there are
        if self.stars:
            self.calculate_integrated_stellar_properties()

        if self.gas:
            self.calculate_integrated_gas_properties()

        # Ensure all attributes are initialised to None
        for attr in Galaxy.attrs:
            try:
                getattr(self, attr)
            except AttributeError:
                setattr(self, attr, None)

        # Attach any additional keyword arguments
        for key, value in kwargs.items():
            setattr(self, key, value)

    def calculate_integrated_stellar_properties(self):
        """
        Calculate integrated stellar properties
        """

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
        """
        Calculate integrated gas properties
        """

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

    def load_stars(
        self,
        initial_masses=None,
        ages=None,
        metallicities=None,
        stars=None,
        **kwargs,
    ):
        """
        Load arrays for star properties into a `Stars`  object.

        This will populate the stars attribute with the instantiated Stars
        object.

        Args:
            initial_masses (array_like, float)
                Initial stellar particle masses (mass at birth), Msol
            ages (array_like, float)
                Star particle age, Myr
            metallicities (array_like, float)
                Star particle metallicity (total metal fraction)
            stars (stars particle object)
                A pre-existing stars particle object to use. Defaults to None.
            **kwargs
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
        self.stars.centre = self.centre

    def load_gas(
        self,
        masses=None,
        metallicities=None,
        gas=None,
        **kwargs,
    ):
        """
        Load arrays for gas particle properties into a `Gas` object.

        This will populate the gas attribute with the instantiated Gas object.

        Args:
            masses : array_like (float)
                gas particle masses, Msol
            metallicities : array_like (float)
                gas particle metallicity (total metal fraction)
            gas (gas particle object)
                A pre-existing gas particle object to use. Defaults to None.
            **kwargs
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
        self.gas.centre = self.centre

    def calculate_black_hole_metallicity(self, default_metallicity=0.012):
        """
        Calculates the metallicity of the region surrounding a black hole. This
        is defined as the mass weighted average metallicity of all gas
        particles whose SPH kernels intersect the black holes position.

        Args:
            default_metallicity (float)
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

    def _prepare_los_args(self, kernel, mask, threshold, force_loop):
        """
        A method to prepare the arguments for line of sight metal surface
        density computation with the C function.

        Args:
            kernel (array_like, float)
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            mask (bool)
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float)
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
        """

        # If we have no gas, throw an error
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "No Gas object has been provided! We can't calculate line of "
                "sight dust attenuation without a Gas object containing the "
                "dust!"
            )

        # Ensure we actually have the properties needed
        if self.stars.coordinates is None:
            raise exceptions.InconsistentArguments(
                "Star object is missing coordinates!"
            )
        if self.gas.coordinates is None:
            raise exceptions.InconsistentArguments(
                "Gas object is missing coordinates!"
            )
        if self.gas.smoothing_lengths is None:
            raise exceptions.InconsistentArguments(
                "Gas object is missing smoothing lengths!"
            )
        if self.gas.metallicities is None:
            raise exceptions.InconsistentArguments(
                "Gas object is missing metallicities!"
            )
        if self.gas.masses is None:
            raise exceptions.InconsistentArguments(
                "Gas object is missing masses!"
            )
        if self.gas.dust_to_metal_ratio is None:
            raise exceptions.InconsistentArguments(
                "Gas object is missing DTMs (dust_to_metal_ratio)!"
            )

        # Set up the kernel inputs to the C function.
        kernel = np.ascontiguousarray(kernel, dtype=np.float64)
        kdim = kernel.size

        # Set up the stellar inputs to the C function.
        star_pos = np.ascontiguousarray(
            self.stars._coordinates[mask, :], dtype=np.float64
        )
        nstar = self.stars._coordinates[mask, :].shape[0]

        # Set up the gas inputs to the C function.
        gas_pos = np.ascontiguousarray(self.gas._coordinates, dtype=np.float64)
        gas_sml = np.ascontiguousarray(
            self.gas._smoothing_lengths, dtype=np.float64
        )
        gas_met = np.ascontiguousarray(
            self.gas.metallicities, dtype=np.float64
        )
        gas_mass = np.ascontiguousarray(self.gas._masses, dtype=np.float64)
        if isinstance(self.gas.dust_to_metal_ratio, float):
            gas_dtm = np.ascontiguousarray(
                np.full_like(gas_mass, self.gas.dust_to_metal_ratio),
                dtype=np.float64,
            )
        else:
            gas_dtm = np.ascontiguousarray(
                self.gas.dust_to_metal_ratio, dtype=np.float64
            )
        ngas = gas_mass.size

        return (
            kernel,
            star_pos,
            gas_pos,
            gas_sml,
            gas_met,
            gas_mass,
            gas_dtm,
            nstar,
            ngas,
            kdim,
            threshold,
            np.max(gas_sml),
            force_loop,
        )

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

    def calculate_los_tau_v(
        self,
        kappa,
        kernel,
        mask=None,
        threshold=1,
        force_loop=0,
    ):
        """
        Calculate tau_v for each star particle based on the distribution of
        stellar and gas particles.

        Note: the resulting tau_vs will be associated to the stars object at
        self.stars.tau_v.

        Args:
            kappa (float)
                ...
            kernel (array_like/float)
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value
                of impact parameter q via kernel[int(k*q)]. Note, this can be
                an arbitrary kernel.
            mask (bool)
                A mask to be applied to the stars. Surface densities will only
                be computed and returned for stars with True in the mask.
            threshold (float)
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
            force_loop (bool)
                By default (False) the C function will only loop over nearby
                gas particles to search for contributions to the LOS surface
                density. This forces the loop over *all* gas particles.
        """

        from ..extensions.los import compute_dust_surface_dens

        # If we don't have a mask make a fake one for consistency
        if mask is None:
            mask = np.ones(self.stars.nparticles, dtype=bool)

        # Prepare the arguments
        args = self._prepare_los_args(kernel, mask, threshold, force_loop)

        # Compute the dust surface densities
        los_dustsds = compute_dust_surface_dens(*args)  # Msun / Mpc**2

        los_dustsds /= (1e6) ** 2  # Msun / pc**2

        # Finalise the calculation
        tau_v = kappa * los_dustsds

        # Store the result in self.stars
        if self.stars.tau_v is None:
            self.stars.tau_v = np.zeros(self.stars.nparticles)
        self.stars.tau_v[mask] = tau_v

        return tau_v

    def screen_dust_gamma_parameter(
        self,
        gamma_min=0.01,
        gamma_max=1.8,
        beta=0.1,
        Z_norm=0.035,
        sf_gas_metallicity=None,
        sf_gas_mass=None,
        stellar_mass=None,
    ):
        """
        Calculate the gamma parameter, controlling the optical depth
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

    def dust_to_metal_vijayan19(
        self, stellar_mass_weighted_age=None, ism_metallicity=None
    ):
        """
        Fitting function for the dust-to-metals ratio based on
        galaxy properties, from L-GALAXIES dust modeling.

        Vijayan+19: https://arxiv.org/abs/1904.02196

        Args:
            stellar_mass_weighted_age (float)
                Mass weighted age of stars in Myr. Defaults to None,
                and uses value provided on this galaxy object (in Gyr)
                ism_metallicity (float)
                Mass weighted gas-phase metallicity. Defaults to None,
                and uses value provided on this galaxy object
                (dimensionless)
        """

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

        return dtm

    def get_images_luminosity(
        self,
        resolution,
        fov,
        img_type="hist",
        stellar_photometry=None,
        blackhole_photometry=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Make an ImageCollection from luminosities.

        Images can either be a simple histogram ("hist") or an image with
        particles smoothed over their SPH kernel. The photometry used for these
        images is extracted from the Sed stored on a component under the key
        defined by <component>_spectra.

        Note that black holes will never be smoothed and only produce a
        histogram due to the point source nature of black holes.

        If multiple components are requested they will be combined into a
        single output image.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            img_type : str
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            stellar_photometry (string)
                The stellar spectra key from which to extract photometry
                to use for the image.
            blackhole_photometry (string)
                The black hole spectra key from which to extract photometry
                to use for the image.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Make sure we have an image to make
        if stellar_photometry is None and blackhole_photometry is None:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(stellar_photometry or blackhole_photometry)!"
                " What component do you want images of?"
            )

        # Make stellar image if requested
        if stellar_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            stellar_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Make the image
            if img_type == "hist":
                # Compute the image
                stellar_imgs.get_imgs_hist(
                    photometry=self.stars.particle_spectra[
                        stellar_photometry
                    ].photo_luminosities,
                    coordinates=self.stars.centered_coordinates,
                )

            elif img_type == "smoothed":
                # Compute the image
                stellar_imgs.get_imgs_smoothed(
                    photometry=self.stars.particle_spectra[
                        stellar_photometry
                    ].photo_luminosities,
                    coordinates=self.stars.centered_coordinates,
                    smoothing_lengths=self.stars.smoothing_lengths,
                    kernel=kernel,
                    kernel_threshold=kernel_threshold,
                )

            else:
                raise exceptions.UnknownImageType(
                    "Unknown img_type %s. (Options are 'hist' or "
                    "'smoothed')" % img_type
                )

        # Make blackhole image if requested
        if blackhole_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            blackhole_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Compute the image
            blackhole_imgs.get_imgs_hist(
                photometry=self.black_holes.particle_spectra[
                    blackhole_photometry
                ].photo_luminosities,
                coordinates=self.black_holes.centered_coordinates,
            )

        # Return the images, combining if there are multiple components
        if stellar_photometry is not None and blackhole_photometry is not None:
            return stellar_imgs + blackhole_imgs
        elif stellar_photometry is not None:
            return stellar_imgs
        return blackhole_imgs

    def get_images_flux(
        self,
        resolution,
        fov,
        img_type="hist",
        stellar_photometry=None,
        blackhole_photometry=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Make an ImageCollection from fluxes.

        Images can either be a simple histogram ("hist") or an image with
        particles smoothed over their SPH kernel. The photometry used for these
        images is extracted from the Sed stored on a component under the key
        defined by <component>_spectra.

        Note that black holes will never be smoothed and only produce a
        histogram due to the point source nature of black holes.

        If multiple components are requested they will be combined into a
        single output image.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            img_type : str
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            stellar_photometry (string)
                The stellar spectra key from which to extract photometry
                to use for the image.
            blackhole_photometry (string)
                The black hole spectra key from which to extract photometry
                to use for the image.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image : array-like
                A 2D array containing the image.
        """
        # Make sure we have an image to make
        if stellar_photometry is None and blackhole_photometry is None:
            raise exceptions.InconsistentArguments(
                "At least one spectra type must be provided "
                "(stellar_photometry or blackhole_photometry)!"
                " What component do you want images of?"
            )

        # Make stellar image if requested
        if stellar_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            stellar_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Make the image
            if img_type == "hist":
                # Compute the image
                stellar_imgs.get_imgs_hist(
                    photometry=self.stars.particle_spectra[
                        stellar_photometry
                    ].photo_fluxes,
                    coordinates=self.stars.centered_coordinates,
                )

            elif img_type == "smoothed":
                # Compute the image
                stellar_imgs.get_imgs_smoothed(
                    photometry=self.stars.particle_spectra[
                        stellar_photometry
                    ].photo_fluxes,
                    coordinates=self.stars.centered_coordinates,
                    smoothing_lengths=self.stars.smoothing_lengths,
                    kernel=kernel,
                    kernel_threshold=kernel_threshold,
                )

            else:
                raise exceptions.UnknownImageType(
                    "Unknown img_type %s. (Options are 'hist' or "
                    "'smoothed')" % img_type
                )

        # Make blackhole image if requested
        if blackhole_photometry is not None:
            # Instantiate the Image colection ready to make the image.
            blackhole_imgs = ImageCollection(resolution=resolution, fov=fov)

            # Compute the image
            blackhole_imgs.get_imgs_hist(
                photometry=self.black_holes.particle_spectra[
                    blackhole_photometry
                ].photo_fluxes,
                coordinates=self.black_holes.centered_coordinates,
            )

        # Return the images, combining if there are multiple components
        if stellar_photometry is not None and blackhole_photometry is not None:
            return stellar_imgs + blackhole_imgs
        elif stellar_photometry is not None:
            return stellar_imgs
        return blackhole_imgs

    def get_map_stellar_mass(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Make a mass map, either with or without smoothing.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
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
    ):
        """
        Make a mass map, either with or without smoothing.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
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
    ):
        """
        Make an age map, either with or without smoothing.

        The age in a pixel is the initial mass weighted average age in that
        pixel.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
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
                signal=self.stars.ages * self.stars.initial_masses,
                coordinates=self.stars.centered_coordinates,
            )

        elif img_type == "smoothed":
            # Compute image
            weighted_img.get_img_smoothed(
                signal=self.stars.ages * self.stars.initial_masses,
                coordinates=self.stars.centered_coordinates,
                smoothing_lengths=self.stars.smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
            )

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )

        # Set up the initial mass image
        mass_img = Image(
            resolution=resolution,
            fov=fov,
        )

        # Make the initial mass map
        if img_type == "hist":
            # Compute the image
            mass_img.get_img_hist(
                signal=self.stars.initial_masses,
                coordinates=self.stars.centered_coordinates,
            )

        elif img_type == "smoothed":
            # Compute image
            mass_img.get_img_smoothed(
                signal=self.stars.initial_masses,
                coordinates=self.stars.centered_coordinates,
                smoothing_lengths=self.stars.smoothing_lengths,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
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

    def get_map_stellar_metal_mass(
        self,
        resolution,
        fov,
        img_type="hist",
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Make a stellar metal mass map, either with or without smoothing.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
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
    ):
        """
        Make a gas metal mass map, either with or without smoothing.

        TODO: make dust map!

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
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
    ):
        """
        Make a stellar metallicity map, either with or without smoothing.

        The metallicity in a pixel is the mass weighted average metallicity in
        that pixel.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """
        # Make the weighted image
        weighted_img = self.get_map_stellar_metal_mass(
            resolution,
            fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the mass image
        mass_img = self.get_map_stellar_mass(
            resolution, fov, img_type, kernel, kernel_threshold
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
    ):
        """
        Make a gas metallicity map, either with or without smoothing.

        The metallicity in a pixel is the mass weighted average metallicity in
        that pixel.

        TODO: make dust map!

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """
        # Make the weighted image
        weighted_img = self.get_map_gas_metal_mass(
            resolution,
            fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the mass image
        mass_img = self.get_map_gas_mass(
            resolution, fov, img_type, kernel, kernel_threshold
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
    ):
        """
        Make a SFR map, either with or without smoothing.

        Only stars younger than age_bin are included in the map. This is
        calculated by computing the initial mass map for stars in the age bin
        and then dividing by the size of the age bin.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).
            age_bin (unyt_quantity/float)
                The size of the age bin used to calculate the star formation
                rate. If supplied without units, the unit system is assumed.

        Returns:
            Image
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
    ):
        """
        Make a SFR map, either with or without smoothing.

        Only stars younger than age_bin are included in the map. This is
        calculated by computing the initial mass map for stars in the age bin
        and then dividing by the size of the age bin and stellar mass of
        the galaxy.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).
            age_bin (unyt_quantity/float)
                The size of the age bin used to calculate the star formation
                rate. If supplied without units, the unit system is assumed.

        Returns:
            Image
        """
        # Get the SFR map
        img = self.get_map_sfr(
            resolution=resolution,
            fov=fov,
            img_type=img_type,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            age_bin=age_bin,
        )

        # Convert the SFR map to sSFR
        img.arr /= self.stellar_mass.value
        img.units = img.units / self.stellar_mass.units

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
    ):
        """
        Make a SpectralCube from an Sed held by this galaxy.

        Data cubes are calculated by smoothing spectra over the component
        morphology. The Sed used is defined by <component>_spectra.

        If multiple components are requested they will be combined into a
        single output data cube.

        NOTE: Either npix or fov must be defined.

        Args:
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            lam (unyt_array, float)
                The wavelength array to use for the data cube.
            cube_type (str)
                The type of data cube to make. Either "smoothed" to smooth
                particle spectra over a kernel or "hist" to sort particle
                spectra into individual spaxels.
            stellar_spectra (string)
                The stellar spectra key to make into a data cube.
            blackhole_spectra (string)
                The black hole spectra key to make into a data cube.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions
                module. Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).
            quantity (str):
                The Sed attribute/quantity to sort into the data cube, i.e.
                "lnu", "llam", "luminosity", "fnu", "flam" or "flux".

        Returns:
            SpectralCube
                The spectral data cube object containing the derived
                data cube.
        """
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
                )

        # Return the images, combining if there are multiple components
        if stellar_spectra is not None and blackhole_spectra is not None:
            return stellar_cube + blackhole_cube
        elif stellar_spectra is not None:
            return stellar_cube
        return blackhole_cube
