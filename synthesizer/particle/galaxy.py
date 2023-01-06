import numpy as np

from weights import calculate_weights
from .stars import Stars
from .sed import Sed
from .dust_curves import power_law

class Galaxy:
    def __init__(self):
        self.name = 'galaxy'

    def load_stars(self, masses, ages, metals, **kwargs):
        self.stars = Stars(masses, ages, metals, **kwargs)

    def stellar_spectra(self, grid):
        """
        Calculate spectra for all individual stellar particles

        Warning: *much* slower than calculating integrated spectra,
        as it does not use vectorisation.

        Returns
        lum_array (array) spectrum for each particle, (N_part, wl)
        """
        intrinsic_lum_array = np.zeros((len(self.stars.initial_masses),
                        grid.spectra['stellar'].shape[-1]))

        for i, (mass, age, metal) in enumerate(zip(
                self.stars.initial_masses,
                self.stars.log10ages,
                self.stars.log10metallicities)):

            weights_temp = self._calculate_weights(grid, metal, age, mass)
            intrinsic_lum_array[i] = np.sum(grid.spectra['stellar'] * weights_temp[:, :, None],
                            axis=(0, 1))

        self.intrinsic_lum_array = intrinsic_lum_array
        self.lam = grid.spectra['lam']

        return self.lam, intrinsic_lum_array

    def integrated_stellar_spectra(self, grid):
        """
        Calculate integrated spectrum for whole galaxy

        Args
        grid: grid object
        """
        weights_temp = self._calculate_weights(grid,
                                               self.stars.log10metallicities,
                                               self.stars.log10ages,
                                               self.stars.initial_masses)

        intrinsic_lum = np.sum(grid.spectra['stellar'] * weights_temp[:, :, None],
                       axis=(0, 1))

        self.intrinsic_lum = intrinsic_lum
        self.lam = grid.spectra['lam']


        return self.lam, self.intrinsic_lum



    # This could be split up
    def get_intrinsic(self, integrated = True):

        """
        Get Sed object for intrinsic spectrum of individual star particles or entire galaxy
        """

        if integrated:
            return Sed(self.lam, self.intrinsic_lum)
        else:
            return Sed(self.lam, self.intrinsic_lum_array)


    def get_screen(self, tauV, dust_curve, integrated = True):

        """
        Get Sed object for intrinsic spectrum of individual star particles or entire galaxy
        Args
        tauV: numerical value of dust attenuation in the V-band
        dust_curve: instance of dust curve
        """

        T = np.exp(-tauV) * dust_curve.T(self.lam)

        if integrated:
            lum = self.intrinsic_lum * T
        else:
            lum = self.intrinsic_lum_array * T

        return Sed(self.lam, self.lum)



    def get_CF00(self, tauV_ISM, tauV_BC, alpha_ISM = -0.7, alpha_BC = -1.3, integrated = True):

        """
        Get Sed object for intrinsic spectrum of individual star particles or entire galaxy
        Args
        tauV_ISM: numerical value of dust attenuation due to the ISM in the V-band
        tauV_BC: numerical value of dust attenuation due to the BC in the V-band
        alpha_ISM: slope of the ISM dust curve, -0.7 in MAGPHYS
        alpha_BC: slope of the BC dust curve, -1.3 in MAGPHYS
        """

        if integrated:
            intrinsic_lum_young =
            intrinsic_lum_old =
        else:
            intrinsic_lum_young =
            intrinsic_lum_old =


        T_ISM = power_law({'slope': alpha_ISM}).T(self.lam)
        T_BC = power_law({'slope': alpha_BC}).T(self.lam)

        T_young = T_ISM * T_BC
        T_old = T_ISM

        lum_young = intrinsic_lum_young * T_young
        lum_young = intrinsic_lum_old * T_old


        return Sed(self.lam, self.lum)


    def get_los(self, tau, dust = power_law({'slope': -1.}), integrated = True):


        return Sed(self.lam, self.lum)



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
