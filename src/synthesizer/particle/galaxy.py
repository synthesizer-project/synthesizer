""" A module containing all the funtionality for Particle based galaxies.

Like it's parametric variant this module contains the Galaxy object definition
from which all galaxy focused functionality can be performed. This variant uses
Particle objects, which can either be derived from simulation data or generated
from parametric models. A Galaxy can contain Stars, Gas, and / or BlackHoles.

Despite its name a Particle based Galaxy can be used for any collection of
particles to enable certain functionality (e.g. imaging of a galaxy group, or
spectra for all particles in a simulation.)

Example usage:

    galaxy = Galaxy(stars, gas, black_holes, ...)
    galaxystars.get_spectra_incident(...)

"""
import numpy as np
from unyt import kpc, Myr, unyt_quantity
from scipy.spatial import cKDTree

from synthesizer.particle import Stars
from synthesizer.particle import Gas
from synthesizer.sed import Sed
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer import exceptions
from synthesizer.imaging.images import ParticleImage
from synthesizer.parametric import Stars as ParametricStars


class Galaxy(BaseGalaxy):
    """The Particle based Galaxy object.

    When working with particles this object provides interfaces for calculating
    spectra, galaxy properties and images. A galaxy can be composed of any
    combination of particle.Stars, particle.Gas, or particle.BlackHoles objects.

    Attributes:

    """

    __slots__ = [
        "lam",
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
    ):
        """Initialise a particle based Galaxy with objects derived from
           Particles.

        Args:
            name (str)
                A name to identify the galaxy. Only used for external labelling,
                has no internal use.
            stars (object, Stars/Stars)
                An instance of Stars containing the stellar particle data
            gas (object, Gas)
                An instance of Gas containing the gas particle data.
            black_holes (object, BlackHoles)
                An instance of BlackHoles containing the black hole particle
                data.
            redshift (float)
                The redshift of the galaxy.

        Raises:
            InconsistentArguments
        """

        # Check we haven't been given a SFZH
        if isinstance(stars, ParametricStars):
            raise exceptions.InconsistentArguments(
                "Parametric Stars passed instead of particle based Stars object."
                " Did you mean synthesizer.parametric.Galaxy instead?"
            )

        # Set the type of galaxy
        self.galaxy_type = "Particle"

        # Instantiate the parent
        BaseGalaxy.__init__(
            self,
            stars=stars,
            gas=gas,
            black_holes=black_holes,
            redshift=redshift,
        )

        # Define a name for this galaxy
        self.name = name

        # If we have them, record how many stellar / gas particles there are
        if self.stars:
            self.calculate_integrated_stellar_properties()

        if self.gas:
            self.calculate_integrated_gas_properties()

        # Ensure all attributes are initialised to None
        for attr in Galaxy.__slots__:
            try:
                getattr(self, attr)
            except AttributeError:
                setattr(self, attr, None)

    def calculate_integrated_stellar_properties(self):
        """
        Calculate integrated stellar properties
        """

        # Define integrated properties of this galaxy
        if self.stars.current_masses is not None:
            self.stellar_mass = np.sum(self.stars.current_masses)

    def calculate_integrated_gas_properties(self):
        """
        Calculate integrated gas properties
        """

        # Define integrated properties of this galaxy
        if self.gas.masses is not None:
            self.gas_mass = np.sum(self.gas.masses)

        if self.gas.star_forming is not None:
            mask = self.gas.star_forming
            if np.sum(mask) == 0:
                self.sf_gas_mass = 0.0
                self.sf_gas_metallicity = 0.0
            else:
                self.sf_gas_mass = np.sum(self.gas.masses[mask])

                # mass weighted gas phase metallicity
                self.sf_gas_metallicity = (
                    np.sum(self.gas.masses[mask] * self.gas.metallicities[mask])
                    / self.sf_gas_mass
                )

    def load_stars(self, initial_masses, ages, metals, **kwargs):
        """
        Load arrays for star properties into a `Stars`  object,
        and attach to this galaxy object

        Args:
        initial_masses : array_like (float)
            initial stellar particle masses (mass at birth), Msol
        ages : array_like (float)
            star particle age, Myr
        metals : array_like (float)
            star particle metallicity (total metal fraction)
        **kwargs

        Returns:
        None

        # TODO: this should be able to take a pre-existing stars object!
        """
        self.stars = Stars(initial_masses, ages, metals, **kwargs)
        self.calculate_integrated_stellar_properties()

        # Assign the redshift
        self.stars.redshift = self.redshift

    def load_gas(self, masses, metals, **kwargs):
        """
        Load arrays for gas particle properties into a `Gas` object,
        and attach to this galaxy object

        Args:
            masses : array_like (float)
                gas particle masses, Msol
            metals : array_like (float)
                gas particle metallicity (total metal fraction)
        **kwargs

        Returns:
        None

        # TODO: this should be able to take a pre-existing stars object!
        """
        self.gas = Gas(masses, metals, **kwargs)
        self.calculate_integrated_gas_properties()

    def calculate_black_hole_metallicity(self, default_metallicity=0.012):
        """
        Calculates the metallicity of the region surrounding a black hole. This
        is defined as the mass weighted average metallicity of all gas particles
        whose SPH kernels intersect the black holes position.

        Args:
            default_metallicity (float)
                The metallicity value used when no gas particles are in range
                of the black hole. The default is solar metallcity.
        """

        # Ensure we actually have Gas and black holes
        if self.gas is None:
            raise exceptions.InconsistentArguments(
                "Calculating the metallicity of the region surrounding the black"
                " hole requires a Galaxy to be intialised with a Gas object!"
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
        metals = np.zeros(self.black_holes.nbh)
        for ind, gas_in_range in enumerate(inds):
            # Handle black holes with no neighbouring gas
            if len(gas_in_range) == 0:
                metals[ind] = default_metallicity

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
                metals[ind] = default_metallicity
                continue

            # Calculate the mass weight metallicity of this black holes region
            metals[ind] = np.average(
                self.gas.metallicities[gas_in_range],
                weights=self.gas._masses[gas_in_range],
            )

        # Assign the metallicity we have found
        self.black_holes.metallicities = metals

    def _prepare_los_args(self, kernel, mask, threshold, force_loop):
        """
        A method to prepare the arguments for line of sight metal surface
        density computation with the C function.

        Args:
            kernel (array_like, float)
                A 1D description of the SPH kernel. Values must be in ascending
                order such that a k element array can be indexed for the value of
                impact parameter q via kernel[int(k*q)]. Note, this can be an
                arbitrary kernel.
            mask (bool)
                A mask to be applied to the stars. Surface densities will only be
                computed and returned for stars with True in the mask.
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
            raise exceptions.InconsistentArguments("Gas object is missing coordinates!")
        if self.gas.smoothing_lengths is None:
            raise exceptions.InconsistentArguments(
                "Gas object is missing smoothing lengths!"
            )
        if self.gas.metallicities is None:
            raise exceptions.InconsistentArguments(
                "Gas object is missing metallicities!"
            )
        if self.gas.masses is None:
            raise exceptions.InconsistentArguments("Gas object is missing masses!")
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
        gas_sml = np.ascontiguousarray(self.gas._smoothing_lengths, dtype=np.float64)
        gas_met = np.ascontiguousarray(self.gas.metallicities, dtype=np.float64)
        gas_mass = np.ascontiguousarray(self.gas._masses, dtype=np.float64)
        if isinstance(self.gas.dust_to_metal_ratio, float):
            gas_dtm = np.ascontiguousarray(
                np.full_like(gas_mass, self.gas.dust_to_metal_ratio), dtype=np.float64
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

    def get_line_los():
        """
        ParticleGalaxy specific method for obtaining the line luminosities
        subject to line of sight attenuation to each star particle.
        """

        pass

    def get_particle_line_intrinsic(self, grid):
        # """
        # Calculate line luminosities from individual young stellar particles

        # Warning: slower than calculating integrated line luminosities,
        # particularly where young particles are resampled, as it does
        # not use vectorisation.

        # Args:
        #     grid (object):
        #         `Grid` object.
        # """
        # age_mask = self.stars.log10ages < grid.max_age
        # lum = np.zeros((np.sum(age_mask), len(grid.lines)))

        # if np.sum(age_mask) == 0:
        #     return lum
        # else:
        #     for i, (mass, age, metal) in enumerate(zip(
        #             self.stars.initial_masses[age_mask],
        #             self.stars.log10ages[age_mask],
        #             self.stars.log10metallicities[age_mask])):

        #         weights_temp = self._calculate_weights(grid, metal, age, mass,
        #                                                young_stars=True)
        #         lum[i] = np.sum(grid.line_luminosities * weights_temp,
        #                         axis=(1, 2))

        # return lum

        pass

    def get_particle_line_attenuated():
        pass

    def get_particle_line_screen():
        pass

    def get_particle_line_los():
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
                order such that a k element array can be indexed for the value of
                impact parameter q via kernel[int(k*q)]. Note, this can be an
                arbitrary kernel.
            mask (bool)
                A mask to be applied to the stars. Surface densities will only be
                computed and returned for stars with True in the mask.
            threshold (float)
                The threshold above which the SPH kernel is 0. This is normally
                at a value of the impact parameter of q = r / h = 1.
        """

        from ..extensions.los import compute_dust_surface_dens

        # If we don't have a mask make a fake one for consistency
        if mask is None:
            mask = np.ones(self.stars.nparticles, dtype=bool)

        # Prepare the arguments
        args = self._prepare_los_args(kernel, mask, threshold, force_loop)

        # Compute the dust surface densities
        los_dustsds = compute_dust_surface_dens(*args)

        # Finalise the calculation
        tau_v = kappa * los_dustsds

        # Store the result in self.stars
        if self.stars.tau_v is None:
            self.stars.tau_v = np.zeros(self.stars.nparticles)
        self.stars.tau_v[mask] = tau_v

        return tau_v

    def screen_dust_gamma_parameter(
        self,
        beta=0.1,
        Z_norm=0.035,
        sf_gas_metallicity=None,
        sf_gas_mass=None,
        stellar_mass=None,
        gamma_min=None,
        gamma_max=None,
    ):
        """
        Calculate the gamma parameter controlling the optical depth
        due to dust dependent on the mass and metallicity of star forming
        gas.

        gamma = (Z_SF / Z_MW) * (M_SF / M_star) * (1 / beta)

        where Z_SF is the star forming gas metallicity, Z_MW is the Milky
        Way value (defaults to value from Zahid+14), M_SF is the star forming gas mass, M_star
        is the stellar mass, and beta is a normalisation value.


        Zahid+14:
        https://iopscience.iop.org/article/10.1088/0004-637X/791/2/130

        Args:
            beta (float):
                normalisation value, default 0.1
            Z_norm (float):
                metallicity normsalition value, defaults to Zahid+14
                value for the Milky Way (0.035)
            sf_gas_metallicity (array):
                custom star forming gas metallicity array. If None,
                defaults to value attached to this galaxy object.
            sf_gas_mass (array):
                custom star forming gas mass array. If None,
                defaults to value attached to this galaxy object.
            stellar_mass (array):
                custom stellar mass array. If None, defaults to value
                attached to this galaxy object.
            gamma_min (float):
                lower limit of the gamma parameter. If None, no lower
                limit implemented. Default = None
            gamma_max (float):
                upper limit of the gamma parameter. If None, no upper
                limit implemented. Default = None

        Returns:
            gamma (array):
                gamma scaling parameter for this galaxy
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
                sf_gas_mass = self.sf_gas_mass

        if stellar_mass is None:
            if self.stellar_mass is None:
                raise ValueError("No stellar_mass provided")
            else:
                stellar_mass = self.stellar_mass

        if sf_gas_mass == 0.0:
            gamma = 0.0
        elif stellar_mass == 0.0:
            gamma = 1.0
        else:
            gamma = (
                (sf_gas_metallicity / Z_norm)
                * (sf_gas_mass / stellar_mass)
                * (1.0 / beta)
            )

        if gamma_min is not None:
            gamma[gamma < gamma_min] = gamma_min

        if gamma_max is not None:
            gamma[gamma > gamma_max] = gamma_max

        return gamma

    def make_images(
        self,
        resolution,
        fov,
        img_type="hist",
        sed=None,
        filters=(),
        pixel_values=None,
        psfs=None,
        depths=None,
        snrs=None,
        aperture=None,
        noises=None,
        rest_frame=True,
        cosmo=None,
        psf_resample_factor=1,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes images, either one or one per filter. This is a generic method
        that will make every sort of image using every possible combination of
        arguments allowed by the ParticleImage class. These methods can be
        either a simple histogram or smoothing particles over a kernel. Either
        of these operations can be done with or without a PSF and noise.

        Args:
            resolution (Quantity, float)
                The size of a pixel.
                (Ignoring any supersampling defined by psf_resample_factor)
            fov : float
                The width of the image in image coordinates.
            img_type : str
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            sed : obj (SED)
                An sed object containing the spectra for this image.
            filters : obj (FilterCollection)
                An imutable collection of Filter objects. If provided images are
                made for each filter.
            pixel_values : array-like (float)
                The values to be sorted/smoothed into pixels. Only needed if an sed
                and filters are not used.
            psfs : dict
                A dictionary containing the psf in each filter where the key is
                each filter code and the value is the psf in that filter.
            depths : dict
                A dictionary containing the depth of an observation in each filter
                where the key is each filter code and the value is the depth in
                that filter.
            aperture : float/dict
                Either a float describing the size of the aperture in which the
                depth is defined or a dictionary containing the size of the depth
                aperture in each filter.
            rest_frame : bool
                Are we making an observation in the rest frame?
            cosmo : obj (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            psf_resample_factor : float
                The factor by which the image should be resampled for robust PSF
                convolution. Note the images after PSF application will be
                downsampled to the native pixel scale.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns
        -------
        Image : array-like
            A 2D array containing the image.
        """

        # Handle a super resolution image
        if psf_resample_factor is not None:
            if psf_resample_factor != 1:
                resolution /= psf_resample_factor

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            sed=sed,
            filters=filters,
            coordinates=self.stars._coordinates,
            smoothing_lengths=self.stars._smoothing_lengths,
            pixel_values=pixel_values,
            rest_frame=rest_frame,
            redshift=self.redshift,
            cosmo=cosmo,
            psfs=psfs,
            depths=depths,
            apertures=aperture,
            snrs=snrs,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

            if psfs is not None:
                # Convolve the image/images
                img.get_psfed_imgs()

                # Downsample to the native resolution if we need to.
                if psf_resample_factor is not None:
                    if psf_resample_factor != 1:
                        img.downsample(1 / psf_resample_factor)

            if depths is not None or noises is not None:
                img.get_noisy_imgs(noises)

            return img

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

            if psfs is not None:
                # Convolve the image/images
                img.get_psfed_imgs()

                # Downsample to the native resolution if we need to.
                if psf_resample_factor is not None:
                    if psf_resample_factor != 1:
                        img.downsample(1 / psf_resample_factor)

            if depths is not None or noises is not None:
                img.get_noisy_imgs(noises)

            return img

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or " "'smoothed')" % img_type
            )

    def make_stellar_mass_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes a mass map, either with or without smoothing.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.stars._coordinates,
            smoothing_lengths=self.stars._smoothing_lengths,
            pixel_values=self.stars._current_masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or " "'smoothed')" % img_type
            )

        return img

    def make_gas_mass_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes a mass map, either with or without smoothing.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.gas._coordinates,
            smoothing_lengths=self.gas._smoothing_lengths,
            pixel_values=self.gas._masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or " "'smoothed')" % img_type
            )

        return img

    def make_stellar_age_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes a age map, either with or without smoothing. The
        age in a pixel is the initial mass weighted average age in that
        pixel.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.stars._coordinates,
            smoothing_lengths=self.stars._smoothing_lengths,
            pixel_values=self.stars._ages * self.stars._initial_masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or " "'smoothed')" % img_type
            )

        # Set up the initial mass image
        mass_img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.stars._coordinates,
            smoothing_lengths=self.stars._smoothing_lengths,
            pixel_values=self.stars._initial_masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the initial mass map
        if img_type == "hist":
            # Compute the image
            mass_img.get_hist_imgs()

        else:
            # Compute image
            mass_img.get_imgs()

            # Divide out the mass contribution to get the mean metallicity
        img.img[img.img > 0] /= mass_img.img[mass_img.img > 0]

        return img

    def make_stellar_metallicity_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes a stellar metallicity map, either with or without smoothing. The
        metallicity in a pixel is the mass weighted average metallicity in that
        pixel.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.stars._coordinates,
            smoothing_lengths=self.stars._smoothing_lengths,
            pixel_values=self.stars.metallicities * self.stars._current_masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                f"Unknown img_type {img_type}. (Options are 'hist' or 'smoothed')"
            )

        # Make the mass image
        mass_img = self.make_stellar_mass_map(
            resolution, fov, img_type, cosmo, kernel, kernel_threshold
        )

        # Divide out the mass contribution to get the mean metallicity
        img.img[img.img > 0] /= mass_img.img[mass_img.img > 0]

        return img

    def make_gas_metallicity_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes a gas metallicity map, either with or without smoothing. The
        metallicity in a pixel is the mass weighted average metallicity in that
        pixel.

        TODO: make dust map!

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.gas._coordinates,
            smoothing_lengths=self.gas._smoothing_lengths,
            pixel_values=self.gas.metallicities * self.gas._masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                f"Unknown img_type {img_type}. (Options are 'hist' or 'smoothed')"
            )

        # Make the mass image
        mass_img = self.make_gas_mass_map(
            resolution, fov, img_type, cosmo, kernel, kernel_threshold
        )

        # Divide out the mass contribution to get the mean metallicity
        img.img[img.img > 0] /= mass_img.img[mass_img.img > 0]

        return img

    def make_stellar_metal_mass_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes a stellar metal mass map, either with or without smoothing.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.stars._coordinates,
            smoothing_lengths=self.stars._smoothing_lengths,
            pixel_values=self.stars.metallicities * self.stars._current_masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                f"Unknown img_type {img_type}. (Options are 'hist' or 'smoothed')"
            )

        return img

    def make_gas_metal_mass_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
    ):
        """
        Makes a gas metal mass map, either with or without smoothing.

        TODO: make dust map!

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).

        Returns:
            Image
        """

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.gas._coordinates,
            smoothing_lengths=self.gas._smoothing_lengths,
            pixel_values=self.gas.metallicities * self.gas._masses,
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                f"Unknown img_type {img_type}. (Options are 'hist' or 'smoothed')"
            )

        return img

    def make_sfr_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
        age_bin=100 * Myr,
    ):
        """
        Makes a SFR map, either with or without smoothing. Only stars younger
        than age_bin are included in the map. This is calculated by computing
        the initial mass map for stars in the age bin and then dividing by the
        size of the age bin.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
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
            print("The SFR is 0! (there are 0 stars in the age bin)")

        # Instantiate the Image object.
        img = ParticleImage(
            resolution=resolution,
            fov=fov,
            coordinates=self.stars._coordinates[mask, :],
            smoothing_lengths=self.stars._smoothing_lengths[mask],
            pixel_values=self.stars._initial_masses[mask],
            redshift=self.redshift,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
        )

        # Make the initial mass map, handling incorrect image types
        if img_type == "hist":
            # Compute the image
            img.get_hist_imgs()

        elif img_type == "smoothed":
            # Compute image
            img.get_imgs()

        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or " "'smoothed')" % img_type
            )

        # Convert the initial mass map to SFR
        img.img /= age_bin

        return img

    def make_ssfr_map(
        self,
        resolution,
        fov,
        img_type="hist",
        cosmo=None,
        kernel=None,
        kernel_threshold=1,
        age_bin=100 * Myr,
    ):
        """
        Makes a SFR map, either with or without smoothing. Only stars younger
        than age_bin are included in the map. This is calculated by computing
        the initial mass map for stars in the age bin and then dividing by the
        size of the age bin.

        Args:
            resolution (float)
                The size of a pixel.
            fov (float)
                The width of the image in image coordinates.
            img_type (str)
                The type of image to be made, either "hist" -> a histogram, or
                "smoothed" -> particles smoothed over a kernel.
            cosmo (astropy.cosmology)
                A cosmology object from astropy, used for cosmological calculations
                when converting rest frame luminosity to flux.
            kernel (array-like, float)
                The values from one of the kernels from the kernel_functions module.
                Only used for smoothed images.
            kernel_threshold (float)
                The kernel's impact parameter threshold (by default 1).
            age_bin (unyt_quantity/float)
                The size of the age bin used to calculate the star formation
                rate. If supplied without units, the unit system is assumed.

        Returns:
            Image
        """

        # Get the SFR map
        img = self.make_sfr_map(
            resolution=resolution,
            fov=fov,
            img_type=img_type,
            cosmo=cosmo,
            kernel=kernel,
            kernel_threshold=kernel_threshold,
            age_bin=age_bin,
        )

        # Convert the SFR map to sSFR
        img.img /= self.stellar_mass

        return img
