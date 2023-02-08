import synthesizer.exceptions as exceptions
from ..particle.stars import Stars
from ..sed import Sed
from ..dust import power_law
from .galaxy import BaseGalaxy
from .. import exceptions
from ..weights import calculate_weights
import numpy as np
from ..imaging.images import ParticleImage


class ParticleGalaxy(BaseGalaxy):
    def __init__(self, stars=None, gas=None):
        self.name = 'galaxy'

        self.stellar_lum = None
        self.stellar_lum_array = None
        self.intrinsic_lum = None
        self.intrinsic_lum_array = None

        self.spectra = {}  # integrated spectra dictionary
        self.spectra_array = {}  # spectra arrays dictionary

        self.stars = stars  # a star object
        self.gas = gas


        # If we have them record how many stellar particles there are
        if self.stars:
            self.nparticles = stars.nparticles
    
            # Define integrated properties of this galaxy
            if stars.current_masses is not None:
                self.stellar_mass = np.sum(stars.current_masses)

    # this should be able to take a pre-existing stars object!

    def load_stars(self, initial_masses, ages, metals, **kwargs):
        self.stars = Stars(initial_masses, ages, metals, **kwargs)
        self.nparticles = len(initial_masses)

    # def load_gas(self, masses, metals, **kwargs):
    #

    def calculate_los_tauV(self, update=True):
        """
        Calculate tauV for each star particle based on the distribution of star/gas particles
        """

        # by default should update self.stars
    
    def generate_intrinsic_spectra(
        self,
        grid,
        fesc=0.0,
        update=True,
        young=False,
        old=False,
        sed_object=False,
        return_wavelength=False
    ):
        """
        Calculate intrinsic spectra from stellar particles. The stellar SED
        component is always created, the intrinsic SED component is only
        computed if the "total" grid is available form the passed grid.

        Either an integrated spectra or per-stellar particle SED can be
        requested. If an SED is requested then the integrated spectra is
        automatically calculated anyway.

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
            directly contradict each other resulting in 0 particles in the mask.
        """
    
        # Get masks for which components we are handling, if a sub-component
        # has not been requested it's necessarily all particles.
        s = self._get_masks(young, old)

        # Calculate integrared spectra

        # Calculate the grid weights for all stellar particles
        weights_temp = self._calculate_weights(
            grid,
            self.stars.log10metallicities[s],
            self.stars.log10ages[s],
            self.stars.initial_masses[s],
        )

        # Get the mask for grid cells we need to sum
        non0_inds = np.where(weights_temp > 0)

        # print(np.unique(weights_temp[non0_inds]))

        # Compute integrated stellar sed
        stellar_lum = np.sum(
            grid.spectra["stellar"][non0_inds[0], non0_inds[1], :]
            * weights_temp[non0_inds[0], non0_inds[1], None],
            axis=0,
        )
        
        # print(np.sum(stellar_lum))

        # # TODO: perhaps should also check that fesc is not false
        # if "total" in list(grid.spectra.keys()):

        #     # Compute the integrated intrinsic sed
        #     intrinsic_lum = np.sum(
        #         (1.0 - fesc)
        #         * grid.spectra["total"][non0_inds[0], non0_inds[1], :]
        #         * weights_temp[non0_inds[0], non0_inds[1], None],
        #         axis=0,
        #     )

        # else:

        #     # If no nebular emission the intrinsic emission is simply
        #     # the stellar emission
        #     intrinsic_lum = stellar_lum

        # # Update the SED's attributes
        # if update:
        #     self.stellar_lum = stellar_lum
        #     self.intrinsic_lum = intrinsic_lum
        #     self.spectra["stellar"] = Sed(grid.lam, self.stellar_lum)
        #     self.spectra["intrinsic"] = Sed(grid.lam, self.intrinsic_lum)
        #     self.lam = grid.lam
    
        if sed_object:
            return Sed(grid.lam, stellar_lum)
        else:
            if return_wavelength:
                return grid.lam, stellar_lum
            else:
                return stellar_lum
    
    def generate_intrinsic_particle_spectra(
        self,
        grid,
        fesc=0.0,
        update=True,
        young=False,
        old=False,
        sed_object=False,
        return_wavelength=False
    ):
        """
        Calculate intrinsic spectra for all *individual* stellar particles. 
        The stellar SED component is always created, the intrinsic SED 
        component is only computed if the "total" grid is available form 
        the passed grid.
            
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
            Flag for whether to retun an Sed object, or the individual 
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
            directly contradict each other resulting in 0 particles in the mask.
        """

        s = self._get_masks(young, old)
    
        # Calculate spectra for every particle individually. This is
        # necessary for los calculation anyway.
    
        # # Initialise arrays to store SEDs
        # stellar_lum_array = np.zeros(
        #     (self.nparticles, grid.spectra["stellar"].shape[-1])
        # )
    
        intrinsic_lum_array = np.zeros(
            (self.nparticles, grid.spectra["stellar"].shape[-1])
        )
    
        # Loop over all stellar particles
        for i, (mass, age, metal) in enumerate(
            zip(
                self.stars.initial_masses[s],
                self.stars.log10ages[s],
                self.stars.log10metallicities[s],
            )
        ):
    
            # Calculate the grid weights for this particle
            weights_temp = self._calculate_weights(grid, metal, age, mass)
            non0_inds = np.where(weights_temp > 0)
    
            # Get the mask for grid cells we need to sum
            non0_inds = np.where(weights_temp > 0)
    
            # Compute the stellar sed for this particle
            stellar_lum_array[i] = np.sum(
                grid.spectra["stellar"][non0_inds[0], non0_inds[1], :]
                * weights_temp[non0_inds[0], non0_inds[1], None],
                axis=0,
            )
    
            # # TODO: perhaps should also check that fesc is not false
            # if "total" in list(grid.spectra.keys()):
    
            #     # Calculate the intrinsic sed for this particle
            #     # TODO: I'm not sure this will actually work if fesc is an array
            #     intrinsic_lum_array[i] = np.sum(
            #         (1.0 - fesc)
            #         * grid.spectra["total"][non0_inds[0], non0_inds[1], :]
            #         * weights_temp[non0_inds[0], non0_inds[1], None],
            #         axis=0,
            #     )
    
            # else:
            #     # If no nebular emission the intrinsic emission is simply
            #     # the stellar emission
            #     intrinsic_lum_array[i] = stellar_lum_array[i]
            
        # # Update the SED's attributes
        # if update:
    
        #     # Store the values of the SED in arrays local to Galaxy.
        #     # (These quantities are actually repeated, in the context
        #     # of an SED object below.)
        #     self.stellar_lum_array = stellar_lum_array
        #     self.intrinsic_lum_array = intrinsic_lum_array
    
        #     # Compute the integrated SEDs
        #     self.stellar_lum = np.sum(stellar_lum_array, axis=0)
        #     self.intrinsic_lum = np.sum(intrinsic_lum_array, axis=0)
    
        #     # Create the SED objects and store them in the dictionaries
        #     # TODO: Repititon of above, may want to consolidate
        #     self.spectra["stellar"] = Sed(grid.lam, self.stellar_lum)
        #     self.spectra_array["stellar"] = Sed(
        #         grid.lam, self.stellar_lum_array
        #     )
        #     self.spectra["intrinsic"] = Sed(grid.lam, self.intrinsic_lum)
        #     self.spectra_array["intrinsic"] = Sed(
        #         grid.lam, self.intrinsic_lum_array
        #     )
    
        #     # Store the wavelength array
        #     self.lam = grid.lam
    
        if sed_object:
            return Sed(grid.lam, stellar_lum_array)
        else:
            if return_wavelength:
                return grid.lam, stellar_lum_array
            else:
                return stellar_lum_array

    def _get_masks(self, young, old):
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
            s = np.ones(self.nparticles, dtype=bool)

        return s

    def get_screen(self, tauV, dust_curve=power_law({'slope': -1.}),
                   integrated=True):
        """
        Get Sed object for intrinsic spectrum of individual star particles or
        entire galaxy
        Args
        tauV: numerical value of dust attenuation in the V-band
        dust_curve: instance of dust class
        """

        T = np.exp(-tauV) * dust_curve.T(self.lam)

        # --- always calculate the integrated spectra since this is low overhead
        # compared to doing the sed_array
        sed = Sed(self.lam, self.intrinsic_lum * T)
        self.spectra['attenuated'] = sed
        # self.spectra['T'] = T

        if integrated:
            return sed
        else:
            sed_array = Sed(self.lam, self.intrinsic_lum_array * T)
            self.spectra_array['attenuated'] = sed_array
            return sed_array

    def get_CF00(self, grid, tauV_ISM, tauV_BC, alpha_ISM=-0.7, alpha_BC=-1.3,
                 integrated=True, save_young_and_old=False):
        """
        Calculates dust attenuated spectra assuming the Charlot & Fall (2000) dust model. In this model young star particles
        are embedded in a dusty birth cloud and thus feel more dust attenuation.


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
            flag specifying whether to save young and old

        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        _, stellar_sed_young, intrinsic_sed_young = self.generate_intrinsic_spectra(
            grid, update=False, young=1E7, integrated=integrated)  # this does not return an Sed object
        _, stellar_sed_old, intrinsic_sed_old = self.generate_intrinsic_spectra(
            grid, update=False, old=1E7, integrated=integrated)  # this does not return an Sed object

        if save_young_and_old:

            if integrated:

                self.spectra['intrinsic_young'] = Sed(
                    self.lam, intrinsic_sed_young)
                self.spectra['intrinsic_old'] = Sed(
                    self.lam, intrinsic_sed_old)

            else:

                self.spectra_array['intrinsic_young'] = Sed(
                    self.lam, intrinsic_sed_young)
                self.spectra_array['intrinsic_old'] = Sed(
                    self.lam, intrinsic_sed_old)
                self.spectra['intrinsic_young'] = Sed(
                    self.lam, np.sum(intrinsic_sed_young))
                self.spectra['intrinsic_old'] = Sed(
                    self.lam, np.sum(intrinsic_sed_old))

        T_ISM = power_law({'slope': alpha_ISM}).attenuate(tauV_ISM, grid.lam)
        T_BC = power_law({'slope': alpha_BC}).attenuate(tauV_BC, grid.lam)

        T_young = T_ISM * T_BC
        T_old = T_ISM

        sed_young = intrinsic_sed_young * T_young
        sed_old = intrinsic_sed_old * T_old

        if save_young_and_old:

            if integrated:

                self.spectra['attenuated_young'] = Sed(self.lam, sed_young)
                self.spectra['attenuated_old'] = Sed(self.lam, sed_old)

            else:

                self.spectra_array['attenuated_young'] = Sed(
                    self.lam, sed_young)
                self.spectra_array['attenuated_old'] = Sed(self.lam, sed_old)
                self.spectra['attenuated_young'] = Sed(
                    self.lam, np.sum(sed_young))
                self.spectra['attenuated_old'] = Sed(self.lam, np.sum(sed_old))

        # --- total SED
        sed = Sed(self.lam, sed_young + sed_old)

        if integrated:
            self.spectra['attenuated'] = sed
        else:
            self.spectra_array['attenuated'] = sed
            self.spectra['attenuated'] = np.sum(sed)

        return sed

    def get_los(self, tauV, dust_curve=power_law({'slope': -1.}), integrated=True):
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
        sed = self.intrinsic_lum_array * T
        self.spectra_array['attenuated'] = Sed(self.lam, sed)
        self.spectra['attenuated'] = Sed(self.lam, np.sum(sed, axis=0))

        if integrated:
            return self.spectra['attenuated']
        else:
            return self.spectra_array['attenuated']

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
                self._calculate_weights(grid,
                                        self.stars.log10metallicities[age_mask],
                                        self.stars.log10ages[age_mask],
                                        self.stars.initial_masses[age_mask],
                                        young_stars=True)

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

    def _calculate_weights(self, grid, metals, ages, imasses,
                           young_stars=False):
        """
        Find weights of particles on grid

        Will calculate for particles individually
        """
        in_arr = np.array([ages, metals, imasses], dtype=np.float64).T
        if (not hasattr(metals, '__len__')):  # check it's an array
            in_arr = in_arr[None, :]  # update dimensions if scalar

        if young_stars:  # filter grid object
            return calculate_weights(grid.log10ages[grid.ages <= grid.max_age],
                                     grid.log10metallicities, in_arr)
        else:
            return calculate_weights(grid.log10ages, grid.log10metallicities,
                                     in_arr)

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

    def make_image(self, resolution, npix=None, fov=None, img_type="hist",
                   sed=None, survey=None, filters=(), pixel_values=None,
                   with_psf=False,  with_noise=False, kernel_func=None,
                   rest_frame=True, redshift=None, cosmo=None, igm=None):
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
            An imutable collection of Filter objects. If provided images are made
            for each filter.
        pixel_values : array-like (float)
            The values to be sorted/smoothed into pixels. Only needed if an sed
            and filters are not used.
        with_psf : bool
            Are we applying a PSF? PLACEHOLDER
        with_noise : bool
            Are we adding noise? PLACEHOLDER
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

        Returns
        -------
        Image : array-like
            A 2D array containing the image.

        """

        # Instantiate the Image object.
        img = ParticleImage(resolution=resolution, npix=npix, fov=fov, sed=sed,
                            stars=self.stars, survey=survey, filters=filters,
                            pixel_values=pixel_values, rest_frame=True,
                            redshift=None, cosmo=None, igm=None)
        
        # Make the image, handling incorrect image types
        if img_type == "hist" and not with_psf and not with_noise:
            
            # Compute image
            img.get_hist_img()

            return img
        
        elif img_type == "hist" and with_psf and not with_noise:
            raise exceptions.UnimplementedFunctionality(
                "PSF functionality coming soon."
            )
        elif img_type == "hist" and not with_psf and with_noise:
            raise exceptions.UnimplementedFunctionality(
                "Noise functionality coming soon."
            )
        elif img_type == "hist" and with_psf and with_noise:
            raise exceptions.UnimplementedFunctionality(
                "PSF and noise functionality coming soon."
            )
        elif img_type == "smoothed" and not with_psf and not with_noise:
            
            # Compute image
            img.get_smoothed_img(kernel_func)

            return img
        
        elif img_type == "smoothed" and with_psf and not with_noise:
            raise exceptions.UnimplementedFunctionality(
                "Smothed functionality coming soon."
            )
        elif img_type == "smoothed" and not with_psf and with_noise:
            raise exceptions.UnimplementedFunctionality(
                "Smothed functionality coming soon."
            )
        elif img_type == "smoothed" and with_psf and with_noise:
            raise exceptions.UnimplementedFunctionality(
                "Smothed functionality coming soon."
            )
        else:
            raise exceptions.UnknownImageType(
                "Unknown img_type %s. (Options are 'hist' or 'smoothed')"
            )
            
