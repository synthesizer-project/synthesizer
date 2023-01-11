from .stars import Stars
from ..sed import Sed
from ..dust import power_law
from ..galaxy import BaseGalaxy
from .. import exceptions
from ..weights import calculate_weights
import numpy as np
from ..imaging.images import ParticleImage


class Galaxy(BaseGalaxy):
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

        if self.stars:
            self.nparticles = stars.nparticles

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

    def generate_intrinsic_spectra(self, grid, fesc=0.0, update=True, young=False, old=False, integrated=True):
        """
        Calculate spectra for all individual stellar particles

        Warning: *much* slower than calculating integrated spectra,
        as it does not use vectorisation.

        Returns
        lum_array (array) spectrum for each particle, (N_part, wl)
        fesc (array or float) LyC escape fraction
        """

        if young:
            if old:
                raise exceptions.InconsistentParameter(
                    'Galaxy sub-component can not be simultaneously young and old')
        if young:
            s = self.stars.log10ages <= np.log10(young)
        elif old:
            s = self.stars.log10ages > np.log10(old)
        else:
            s = np.ones(self.nparticles, dtype=bool)

        # just calculate integrared spectra
        if integrated:

            weights_temp = self._calculate_weights(grid,
                                                   self.stars.log10metallicities[s],
                                                   self.stars.log10ages[s],
                                                   self.stars.initial_masses[s])

            stellar_lum = np.sum(grid.spectra['stellar'] * weights_temp[:, :, None],
                                 axis=(0, 1))

            # perhaps should also check that fesc is not false
            if 'total' in list(grid.spectra.keys()):

                intrinsic_lum = np.sum((1.-fesc) * grid.spectra['total'] * weights_temp[:, :, None],
                                       axis=(0, 1))

            else:

                # --- if no nebular emission the intrinsic emission is simply the stellar emission
                intrinsic_lum = stellar_lum

            if update:
                self.stellar_lum = stellar_lum
                self.intrinsic_lum = intrinsic_lum
                self.spectra['stellar'] = Sed(grid.lam, self.stellar_lum)
                self.spectra['intrinsic'] = Sed(grid.lam, self.intrinsic_lum)
                self.lam = grid.lam

            return grid.lam, stellar_lum, intrinsic_lum

        # else calculate spectra for every particle individually. This is necessary for los calculation anyway.
        else:

            stellar_lum_array = np.zeros((self.nparticles,
                                          grid.spectra['stellar'].shape[-1]))

            intrinsic_lum_array = np.zeros((self.nparticles,
                                            grid.spectra['stellar'].shape[-1]))

            for i, (mass, age, metal) in enumerate(zip(
                    self.stars.initial_masses[s],
                    self.stars.log10ages[s],
                    self.stars.log10metallicities[s])):

                weights_temp = self._calculate_weights(grid, metal, age, mass)

                stellar_lum_array[i] = np.sum(grid.spectra['stellar'] * weights_temp[:, :, None],
                                              axis=(0, 1))

                # perhaps should also check that fesc is not false
                if 'total' in list(grid.spectra.keys()):

                    # --- I'm not sure this will actually work if fesc is an array
                    intrinsic_lum_array[i] = np.sum((1.-fesc) * grid.spectra['total'] * weights_temp[:, :, None],
                                                    axis=(0, 1))

                else:

                    # --- if no nebular emission the intrinsic emission is simply the stellar emission
                    intrinsic_lum_array[i] = stellar_lum_array[i]

            if update:

                # --- these quantities are actually repeated, in the context of an SED object below.
                self.stellar_lum_array = stellar_lum_array
                self.intrinsic_lum_array = intrinsic_lum_array
                # --- create integrated stellar SED
                self.stellar_lum = np.sum(stellar_lum_array, axis=0)
                # --- create integrated intrinsic SED
                self.intrinsic_lum = np.sum(intrinsic_lum_array, axis=0)

                # --- repititon of above, may want to consolidate
                self.spectra['stellar'] = Sed(grid.lam, self.stellar_lum)
                self.spectra_array['stellar'] = Sed(
                    grid.lam, self.stellar_lum_array)
                self.spectra['intrinsic'] = Sed(grid.lam, self.intrinsic_lum)
                self.spectra_array['intrinsic'] = Sed(
                    grid.lam, self.intrinsic_lum_array)

                self.lam = grid.lam

            return grid.lam, stellar_lum_array, intrinsic_lum_array

    def get_screen(self, tauV, dust_curve=power_law({'slope': -1.}), integrated=True):
        """
        Get Sed object for intrinsic spectrum of individual star particles or entire galaxy
        Args
        tauV: numerical value of dust attenuation in the V-band
        dust_curve: instance of dust class
        """

        T = np.exp(-tauV) * dust_curve.T(self.lam)

        # --- always calculate the integrated spectra since this is low overhead compared to doing the sed_array
        sed = Sed(self.lam, self.intrinsic_lum * T)
        self.spectra['attenuated'] = sed
        # self.spectra['T'] = T

        if integrated:
            return sed
        else:
            sed_array = Sed(self.lam, self.intrinsic_lum_array * T)
            self.spectra_array['attenuated'] = sed_array
            return sed_array

    def get_CF00(self, grid, tauV_ISM, tauV_BC, alpha_ISM=-0.7, alpha_BC=-1.3, integrated=True, save_young_and_old=False):
        """
        Get Sed object for intrinsic spectrum of individual star particles or entire galaxy
        Args
        grid: grid object
        tauV_ISM: numerical value of dust attenuation due to the ISM in the V-band
        tauV_BC: numerical value of dust attenuation due to the BC in the V-band
        alpha_ISM: slope of the ISM dust curve, -0.7 in MAGPHYS
        alpha_BC: slope of the BC dust curve, -1.3 in MAGPHYS
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

        T_ISM = np.exp(-tauV_ISM) * power_law({'slope': alpha_ISM}).T(self.lam)
        T_BC = np.exp(-tauV_BC) * power_law({'slope': alpha_BC}).T(self.lam)

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
            return calculate_weights(grid.log10ages, grid.log10metallicities, in_arr)

    def create_stellarmass_hist(self, resolution, npix=None, fov=None):
        """
        Calculate a 2D histogram of the galaxies mass distribution.


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
