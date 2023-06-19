import numpy as np

from ..exceptions import MissingSpectraType
from ..particle.stars import Stars
from ..particle.gas import Gas
from ..sed import Sed
from ..dust import power_law
from .galaxy import BaseGalaxy
from .. import exceptions
from ..imaging.images import ParticleImage


class ParticleGalaxy(BaseGalaxy):

    __slots__ = [
        # "stellar_lum", "stellar_lum_array",
        # "intrinsic_lum", "intrinsic_lum_array",
        "spectra", "spectra_array", "lam",
        "stars", "gas",
        "sf_gas_metallicity", "sf_gas_mass",
        "gas_mass"
    ]

    def __init__(self, name="galaxy", stars=None, gas=None, redshift=None):

        # Define a name for this galaxy
        self.name = name

        # What is the redshift of this galaxy?
        self.redshift = redshift

        # self.stellar_lum = None
        # self.stellar_lum_array = None
        # self.intrinsic_lum = None
        # self.intrinsic_lum_array = None

        self.spectra = {}  # integrated spectra dictionary
        self.spectra_array = {}  # spectra arrays dictionary

        self.stars = stars  # a star object
        self.gas = gas

        # If we have them, record how many stellar particles there are
        if self.stars:
            self.calculate_integrated_stellar_properties()

        if self.gas:
            self.calculate_integrated_gas_properties()

        # Ensure all attributes are intialised to None
        for attr in ParticleGalaxy.__slots__:
            try:
                getattr(self, attr)
            except AttributeError:
                setattr(self, attr, None)

    def calculate_integrated_stellar_properties(self):
        """
        Calculate integrated stellar properties
        """
        self.n_starparticles = self.stars.nparticles

        # Define integrated properties of this galaxy
        if self.stars.current_masses is not None:
            self.stellar_mass = np.sum(self.stars.current_masses)

    def calculate_integrated_gas_properties(self):
        """
        Calculate integrated gas properties
        """
        self.n_gasparticles = self.gas.nparticles

        # Define integrated properties of this galaxy
        if self.gas.masses is not None:
            self.gas_mass = np.sum(self.gas.masses)

        if self.gas.star_forming is not None:
            mask = self.gas.star_forming
            if np.sum(mask) == 0:
                self.sf_gas_mass = 0.
                self.sf_gas_metallicity = 0.
            else:
                self.sf_gas_mass = np.sum(self.gas.masses[mask])

                # mass weighted gas phase metallicity
                self.sf_gas_metallicity = \
                    np.sum(self.gas.masses[mask] *
                           self.gas.metallicities[mask]) / self.sf_gas_mass

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

    def _prepare_args(self, grid, fesc, spectra_type, mask=None):
        """
        A method to prepare the arguments for SED computation with the C
        functions.
        """

        if mask is None:
            mask = np.ones(self.stars.nparticles, dtype=bool)

        # Set up the inputs to the C function.
        grid_props = [
            np.ascontiguousarray(grid.log10ages, dtype=np.float64),
            np.ascontiguousarray(grid.log10metallicities, dtype=np.float64)
        ]
        part_props = [
            np.ascontiguousarray(
                self.stars.log10ages[mask], dtype=np.float64),
            np.ascontiguousarray(
                self.stars.log10metallicities[mask], dtype=np.float64),
        ]
        part_mass = np.ascontiguousarray(
            self.stars.initial_masses[mask], dtype=np.float64)
        npart = np.int32(part_mass.size)
        nlam = np.int32(grid.spectra[spectra_type].shape[-1])

        # Slice the spectral grids and pad them with copies of the edges.
        grid_spectra = np.ascontiguousarray(
            grid.spectra[spectra_type], np.float64)

        # Get the grid dimensions after slicing what we need
        grid_dims = np.zeros(len(grid_props) + 1, dtype=np.int32)
        for ind, g in enumerate(grid_props):
            grid_dims[ind] = len(g)
        grid_dims[ind + 1] = nlam

        # Convert inputs to tuples
        grid_props = tuple(grid_props)
        part_props = tuple(part_props)

        return (grid_spectra, grid_props, part_props, part_mass, fesc,
                grid_dims, len(grid_props), npart, nlam)

        
    def generate_spectra(
        self,
        grid,
        spectra_type,
        fesc=0.0,
        update=True,
        young=False,
        old=False,
        sed_object=True,
        return_wavelength=False
    ):
        """

        TODO: DEPRECATED, left to avoid compatibility issues (for now)

        Calculate spectra from stellar particles.

        Parameters
        ----------
        grid : obj (Grid)
            The SPS grid object sampled by stellar particle to make the SED.
        spectra_type : string
            The spectra type stored in the grid. Will return an error if not
            provided in the grid object
        fesc : float
            The Lyc escape fraction.
        update : bool
            Should we update the Galaxy's attributes?
        young : bool
            Are we masking for only young stars?
        old : bool
            Are we masking for only old stars?
        sed_object: bool
            Flag for whether to retun an Sed object, or the individual
            numpy arrays
        return wavelength : bool
            if sed_object==False, Flag for whether to return grid wavelength

        Returns
        -------
        if sed_object==True:
            Sed object containing integrated intrinsic spectra
        if return_wavelength==True:
            grid.lam: array-like (float)
                The wavelength array associated to grid (N_wavelength).
        if sed_object==False:
            stellar_lum : array-like (float)
                Integrated stellar spectrum.

        Raises
        ------
        InconsistentArguments
            Errors if both a young and old component is requested because these
            directly contradict each other resulting in 0 particles in
            the mask.
        """

        # Ensure we have a total key in the grid. If not error.
        if spectra_type not in list(grid.spectra.keys()):
            raise MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_type
            )
        
        # get particle age masks
        mask = self._get_masks(young, old)

        from ..extensions.csed import compute_integrated_sed

        # Prepare the arguments for the C function.
        args = self._prepare_args(grid, fesc=fesc, 
                                  spectra_type=spectra_type,
                                  mask=mask)

        # Get the integrated spectra in grid units (erg / s / Hz)
        spec = compute_integrated_sed(*args)
        
        # Store the spectra in the galaxy
        self.spectra[spectra_type] = Sed(grid.lam, spec)

        if sed_object:
            return self.spectra[spectra_type]
        else:
            if return_wavelength:
                return grid.lam, spec
            else:
                return spec
            
    def generate_lnu(
                    self,
                    grid,
                    spectra_name,
                    fesc=0.0,
                    young=False,
                    old=False,
                    verbose=False
    ):

        # Ensure we have a total key in the grid. If not error.
        if spectra_name not in list(grid.spectra.keys()):
            raise MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_name
            )
        
        # get particle age masks
        mask = self._get_masks(young, old)

        if np.sum(mask) == 0:
            if verbose:
                print('Age mask has filtered out all particles')

            return np.zeros(len(grid.lam))

        from ..extensions.csed import compute_integrated_sed

        # Prepare the arguments for the C function.
        args = self._prepare_args(grid, fesc=fesc, 
                                  spectra_type=spectra_name,
                                  mask=mask)

        # Get the integrated spectra in grid units (erg / s / Hz)
        spec = compute_integrated_sed(*args)

        return spec
        
        # # Store the spectra in the galaxy
        # self.spectra[spectra_name] = Sed(grid.lam, spec)

        # if sed_object:
        #     return self.spectra[spectra_name]
        # else:
        #     if return_wavelength:
        #         return grid.lam, spec
        #     else:
        #         return spec

    def get_stellar_spectra(self, grid, 
                            update=True, 
                            sed_object=True,
                            young=False,
                            old=False,
                            return_wavelength=False):

        lnu = self.generate_lnu(grid, 'stellar', 
                                young=young,
                                old=old)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['stellar'] = sed

        if sed_object:
            return sed
        else:
            if return_wavelength:
                return grid.lam, lnu
            else:
                return lnu

    def get_nebular_spectra(self, grid, fesc=0.0, 
                            update=True,
                            young=False,
                            old=False,
                            sed_object=True,
                            return_wavelength=False):

        lnu = self.generate_lnu(grid, 'nebular',
                                young=young,
                                old=old)

        lnu *= (1-fesc)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['nebular'] = sed

        if sed_object:
            return sed
        else:
            if return_wavelength:
                return grid.lam, lnu
            else:
                return lnu

    def get_intrinsic_spectra(self, grid, fesc=0.0, 
                              update=True,
                              young=False,
                              old=False,
                              sed_object=True,
                              return_wavelength=False):

        stellar = self.get_stellar_spectra(grid, update=update,
                                           young=young,
                                           old=old)
        
        nebular = self.get_nebular_spectra(grid, fesc, 
                                           update=update,
                                           young=young,
                                           old=old)

        lnu = stellar._lnu + nebular._lnu
        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['intrinsic'] = sed

        if sed_object:
            return sed
        else:
            if return_wavelength:
                return grid.lam, lnu
            else:
                return lnu

    def get_screen_spectra(self, grid, fesc=0.0, tauV=None, 
                           dust_curve=power_law({'slope': -1.}), 
                           update=True,
                           young=False,
                           old=False,
                           sed_object=True,
                           return_wavelength=False):
        
        # --- begin by calculating intrinsic spectra
        intrinsic = self.get_intrinsic_spectra(grid, 
                                               update=update, 
                                               fesc=fesc,
                                               young=young,
                                               old=old)

        if tauV:
            T = dust_curve.attenuate(tauV, grid.lam)
        else:
            T = 1.0

        lnu = T * intrinsic._lnu
        sed = Sed(grid.lam, T * intrinsic._lnu)

        if update:
            self.spectra['attenuated'] = sed

        if sed_object:
            return sed
        else:
            if return_wavelength:
                return grid.lam, lnu
            else:
                return lnu

    def _get_masks(self, young=None, old=None):
        """
        Get masks for which components we are handling, if a sub-component
        has not been requested it's necessarily all particles.
        """

        if young and old:
            raise exceptions.InconsistentParameter(
                "Galaxy sub-component can not be simultaneously young and old"
            )
        if young:
            s = self.stars.log10ages <= np.log10(young)
        elif old:
            s = self.stars.log10ages > np.log10(old)
        else:
            s = np.ones(self.n_starparticles, dtype=bool)

        return s

    def get_CharlotFall_spectra(self, grid, 
                                tauV_ISM, 
                                tauV_BC,
                                alpha_ISM=-0.7, 
                                alpha_BC=-1.3,
                                intrinsic_young=None, 
                                intrinsic_old=None,
                                save_young_and_old=False, 
                                sed_object=True,
                                update=True,
                                return_wavelength=False):
        """
        Calculates dust attenuated spectra assuming the Charlot & Fall (2000)
        dust model. In this model young star particles are embedded in a dusty
        birth cloud, and thus feel more dust attenuation.

        Parameters
        ----------
        grid : obj (Grid)
            The spectral frid
        tauV_ISM: float
            numerical value of dust attenuation due to the ISM in the V-band
        tauV_BC: float
            numerical value of dust attenuation due to the BC in the V-band
        alpha_ISM: float
            slope of the ISM dust curve, -0.7 in MAGPHYS
        alpha_BC: float
            slope of the BC dust curve, -1.3 in MAGPHYS
        save_young_and_old: boolean
            flag specifying whether to save young and old spectra individually
        sed_object: bool
            flag whether to return an SED object
        update: bool
            flag for whether to update the `intrinsic` and `attenuated` spectra
            inside the galaxy object `spectra` dictionary. These are the combined values
            of young and old.
        return_wavelength: bool
            return wavelenght numpy array

        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        # _, stellar_sed_young, intrinsic_sed_young = \
        #         self.generate_intrinsic_spectra(
        #     grid, update=False, young=1E7, integrated=integrated)
        # this does not return an Sed object
        # _, stellar_sed_old, intrinsic_sed_old = \
        #         self.generate_intrinsic_spectra(
        # grid, update=False, old=1E7, integrated=integrated)  # this does not
        # return an Sed object

        intrinsic_sed_young = self.get_intrinsic_spectra(
            grid, update=False, young=1E7)
        intrinsic_sed_old = self.get_intrinsic_spectra(
            grid, update=False, old=1e7)

        # save combined intrinsic spectra 
        if update:
            self.spectra['intrinsic'] = intrinsic_sed_young + intrinsic_sed_old

        if save_young_and_old:

            # if integrated:
            self.spectra['intrinsic_young'] = intrinsic_sed_young
            self.spectra['intrinsic_old'] = intrinsic_sed_old

            # else:
            #     self.spectra_array['intrinsic_young'] = Sed(
            #         grid.lam, intrinsic_sed_young)
            #     self.spectra_array['intrinsic_old'] = Sed(
            #         grid.lam, intrinsic_sed_old)
            #     self.spectra['intrinsic_young'] = Sed(
            #         grid.lam, np.sum(intrinsic_sed_young))
            #     self.spectra['intrinsic_old'] = Sed(
            #         grid.lam, np.sum(intrinsic_sed_old))

        T_ISM = power_law({'slope': alpha_ISM}).attenuate(tauV_ISM, grid.lam)
        T_BC = power_law({'slope': alpha_BC}).attenuate(tauV_BC, grid.lam)

        T_young = T_ISM * T_BC
        T_old = T_ISM

        lnu_young = intrinsic_sed_young.lnu * T_young
        lnu_old = intrinsic_sed_old.lnu * T_old

        if save_young_and_old:

            # if integrated:
            self.spectra['attenuated_young'] = Sed(grid.lam, lnu_young)
            self.spectra['attenuated_old'] = Sed(grid.lam, lnu_old)

            # else:
            #     self.spectra_array['attenuated_young'] = Sed(
            #         grid.lam, sed_young)
            #     self.spectra_array['attenuated_old'] = Sed(grid.lam, sed_old)
            #     self.spectra['attenuated_young'] = Sed(
            #         grid.lam, np.sum(sed_young))
            #     self.spectra['attenuated_old'] = \
            #         Sed(grid.lam, np.sum(sed_old))

        lnu = lnu_young + lnu_old
        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['attenuated'] = sed
        
        if sed_object:
            return sed
        else:
            if return_wavelength:
                return grid.lam, lnu
            else:
                return lnu

    def calculate_los_tauV(self, update=True):
        """
        Calculate tauV for each star particle based on the distribution of
        star/gas particles
        """

        # by default should update self.stars
        
        pass

    def apply_los(self, tauV, 
                  spectra_type,
                  dust_curve=power_law({'slope': -1.}),
                  integrated=True, sed_object=True):
        """
        Generate
        tauV: V-band optical depth for every star particle
        dust_curve: instance of the dust class
        """

        T = np.outer(tauV, dust_curve.T(self.lam))

        # need exception
        # if not self.intrinsic_lum_array:
        #     print('Must generate spectra for individual star particles')

        # these two should have the same shape so should work?
        sed = self.spectra_array[spectra_type] * T
        self.spectra_array['attenuated'] = Sed(self.lam, sed)
        self.spectra['attenuated'] = Sed(self.lam, np.sum(sed, axis=0))

        if integrated:
            return self.spectra['attenuated']
        else:
            return self.spectra_array['attenuated']
        
    def generate_particle_spectra(
        self,
        grid,
        spectra_type,
        fesc=0.0,
        update=True,
        young=False,
        old=False,
        sed_object=True,
        return_wavelength=False
    ):
        """
        Calculate intrinsic spectra for all *individual* stellar particles.
        The stellar SED component is always created, the intrinsic SED
        component is only computed if the "total" grid is available from
        the passed grid.

        TODO: need to be able to apply masks to get young and old stars.

        Parameters
        ----------
        grid : obj (Grid)
            The SPS grid object sampled by stellar particle to make the SED.
        fesc : float
            The Lyc escape fraction.
        update : bool
            Should we update the Galaxy's attributes?
        young : bool
            Are we masking for only young stars?
        old : bool
            Are we masking for only old stars?
        sed_object : bool
            Flag for whether to return an Sed object, or the individual
            numpy arrays
        return wavelength : bool
            if sed_object==False, Flag for whether to return grid wavelength

        Returns
        -------
        grid.lam: array-like (float)
            The wavelength array associated to grid (N_wavelength).
        stellar_lum/stellar_lum_array : float/array-like (float)
            Stellar spectrum of each particle (N_part, N_wavelength).
        intrinsic_lum/intrinsic_lum_array :
            Intrinsic spectrum of each particle (N_part, N_wavelength).

        Raises
        ------
        InconsistentArguments
            Errors if both a young and old component is requested because these
            directly contradict each other resulting in 0 particles in
            the mask.
        """

        # Ensure we have a total key in the grid. If not error.
        if spectra_type not in list(grid.spectra.keys()):
            raise MissingSpectraType(
                "The Grid does not contain the key '%s'" % spectra_type
            )

        from ..extensions.csed import compute_particle_seds

        # Prepare the arguments for the C function.
        args = self._prepare_args(grid, fesc=fesc, spectra_type=spectra_type)

        # Get the integrated stellar SED
        spec_arr = compute_particle_seds(*args)

        # Store the spectra in the galaxy
        self.spectra_array[spectra_type] = spec_arr

        if sed_object:
            return Sed(grid.lam, spec_arr)
        else:
            if return_wavelength:
                return grid.lam, spec_arr
            else:
                return spec_arr

    def stellar_particle_line_luminosities(self, grid):
        """
        Calculate line luminosities from individual young stellar particles

        Warning: slower than calculating integrated line luminosities,
        particularly where young particles are resampled, as it does
        not use vectorisation.

        Args
        grid (object)
        """
        age_mask = self.stars.log10ages < grid.max_age
        lum = np.zeros((np.sum(age_mask), len(grid.lines)))

        if np.sum(age_mask) == 0:
            return lum
        else:
            for i, (mass, age, metal) in enumerate(zip(
                    self.stars.initial_masses[age_mask],
                    self.stars.log10ages[age_mask],
                    self.stars.log10metallicities[age_mask])):

                weights_temp = self._calculate_weights(grid, metal, age, mass,
                                                       young_stars=True)
                lum[i] = np.sum(grid.line_luminosities * weights_temp,
                                axis=(1, 2))

            return lum

    def integrated_stellar_line_luminosities(self, grid, save=False,
                                             verbose=False):
        """
        Calculate integrated line luminosities for whole galaxy

        Args
        grid (object)
        save (bool, False) determines if line luminosities dict saved
                           to galaxy object. If false, return dict
        """
        # lum = self.stellar_particle_line_luminosities(grid)

        age_mask = self.stars.log10ages < grid.max_age

        if np.sum(age_mask) > 0:
            weights_temp =\
                self._calculate_weights(
                        grid,
                        self.stars.log10metallicities[age_mask],
                        self.stars.log10ages[age_mask],
                        self.stars.initial_masses[age_mask],
                        young_stars=True
                )

            lum = np.sum(grid.line_luminosities * weights_temp, axis=(1, 2))
        else:
            if verbose:
                print("Warning: no particles below max age limit")
            lum = np.empty([grid.line_luminosities.shape[0]])
            lum[:] = np.nan

        line_lums = {}
        for i, line in enumerate(grid.lines):
            line_lums[line] = lum[i]

        if save:
            self.stellar_line_luminosities = line_lums
        else:
            return line_lums

    def create_stellarmass_hist(self, resolution, npix=None, fov=None):
        """
        Calculate a 2D histogram of the galaxy's mass distribution.

        NOTE: Either npix or fov must be defined.

        Parameters
        ----------
        resolution : float
           The size of a pixel.
        npix : int
            The number of pixels along an axis.
        fov : float
            The width of the image in image coordinates.

        Returns
        -------
        Image : array-like
            A 2D array containing the image.

        """

        # Instantiate the Image object.
        img = ParticleImage(resolution, npix, fov, stars=self.stars,
                            pixel_values=self.stars.initial_masses)

        return img.get_hist_img()

    def screen_dust_gamma_parameter(
        self, beta=0.1, Z14=0.035, sf_gas_metallicity=None,
        sf_gas_mass=None, stellar_mass=None
    ):
        """
        Calculate the gamma parameter controlling the optical depth
        due to dust from integrated galaxy properties

        Args:

        """

        if sf_gas_metallicity is None:
            if self.sf_gas_metallicity is None:
                raise ValueError('No sf_gas_metallicity provided')
            else:
                sf_gas_metallicity = self.sf_gas_metallicity

        if sf_gas_mass is None:
            if self.sf_gas_mass is None:
                raise ValueError('No sf_gas_mass provided')
            else:
                sf_gas_mass = self.sf_gas_mass

        if stellar_mass is None:
            if self.stellar_mass is None:
                raise ValueError('No stellar_mass provided')
            else:
                stellar_mass = self.stellar_mass

        gamma = (sf_gas_metallicity / Z14) *\
                (sf_gas_mass / stellar_mass) *\
                (1. / beta)

        return gamma

    def make_image(self, resolution, fov=None, img_type="hist",
                   sed=None, filters=(), pixel_values=None, psfs=None,
                   depths=None, snrs=None, aperture=None, noises=None,
                   kernel_func=None, rest_frame=True, cosmo=None,
                   psf_resample_factor=1,
                   ):
        """
        Makes images, either one or one per filter. This is a generic method
        that will make every sort of image using every possible combination of
        arguments allowed by the ParticleImage class. These methods can be
        either a simple histogram or smoothing particles over a kernel. Either
        of these operations can be done with or without a PSF and noise.
        NOTE: Either npix or fov must be defined.
        Parameters
        ----------
        resolution : float
           The size of a pixel.
           (Ignoring any supersampling defined by psf_resample_factor)
        npix : int
            The number of pixels along an axis.
        fov : float
            The width of the image in image coordinates.
        img_type : str
            The type of image to be made, either "hist" -> a histogram, or
            "smoothed" -> particles smoothed over a kernel.
        sed : obj (SED)
            An sed object containing the spectra for this image.
        survey : obj (Survey)
            WorkInProgress
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
        kernel_func : function
            A function describing the smoothing kernel that returns a single
            number between 0 and 1. This function can be imported from the
            options in kernel_functions.py or can be user defined. If user
            defined the function must return the kernel value corredsponding
            to the position of a particle with smoothing length h at distance
            r from the centre of the kernel (r/h).
        rest_frame : bool
            Are we making an observation in the rest frame?
        redshift : float
            The redshift of the observation. Used when converting rest frame
            luminosity to flux.
        cosmo : obj (astropy.cosmology)
            A cosmology object from astropy, used for cosmological calculations
            when converting rest frame luminosity to flux.
        igm : obj (Inoue14/Madau96)
            Object containing the absorbtion due to an intergalactic medium.
        psf_resample_factor : float
            The factor by which the image should be resampled for robust PSF
            convolution. Note the images after PSF application will be
            downsampled to the native pixel scale.
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
            stars=self.stars,
            filters=filters,
            pixel_values=pixel_values,
            rest_frame=rest_frame,
            redshift=self.redshift,
            cosmo=cosmo,
            psfs=psfs,
            depths=depths,
            apertures=aperture,
            snrs=snrs,
        )

        # Make the image, handling incorrect image types
        if img_type == "hist":

            # Compute the image
            img.get_hist_img()

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
            img.get_smoothed_img(kernel_func)

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
                "Unknown img_type %s. (Options are 'hist' or "
                "'smoothed')" % img_type
            )
