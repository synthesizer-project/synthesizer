import numpy as np

from weights import calculate_weights
from .stars import Stars


class Galaxy:
    def __init__(self):
        self.name = 'galaxy'

    def load_stars(self, masses, ages, metals, **kwargs):
        self.stars = Stars(masses, ages, metals, **kwargs)

    def _calculate_weights(self, grid, metals, ages, imasses, young_stars=False):
        """
        Find weights of particles on grid

        Will calculate for particles individually
        """
        in_arr = np.array([metals, ages, imasses], dtype=np.float64)
        if (not hasattr(metals, '__len__')):  # check it's an array
            in_arr = in_arr[None, :]  # update dimensions if scalar
        
        if young_stars:  # filter grid object 
            return calculate_weights(grid.metallicities, 
                                     grid.ages[grid.ages <= grid.max_age], in_arr)
        else:
            return calculate_weights(grid.metallicities, grid.ages, in_arr)

    def stellar_particle_spectra(self, grid):
        """
        Calculate spectra for all individual stellar particles
        
        Warning: *much* slower than calculating integrated spectra,
        as it does not use vectorisation.

        Returns
        lum (array) spectrum for each particle, (N_part, wl)
        """
        lum = np.zeros((len(self.stars.initial_masses), grid.spectra.shape[-1]))
       
        for i, (mass, age, metal) in enumerate(zip(
                self.stars.initial_masses,
                self.stars.log10ages,
                self.stars.log10metallicities)):
                
            weights_temp = self._calculate_weights(grid, metal, age, mass)
            lum[i] = np.sum(grid.spectra * weights_temp[:,:,None], axis=(0,1))

        return lum

    def calculate_stellar_spectrum(self, grid, save=False):
        """
        Calculate integrated spectrum for whole galaxy

        Args
        grid: grid object
        save (bool, False): determines if spectra saved to galaxy object
                            if False, method returns spectrum as array
        """
        weights_temp = self._calculate_weights(grid,
                self.stars.log10metallicities,
                self.stars.log10ages,
                self.stars.initial_masses)

        _spec = np.sum(grid.spectra * weights_temp[:,:,None], axis=(0,1))
        
        # lum = self.stellar_particle_spectra(grid)
        # _spec = np.sum(lum, axis=0)
        if save:
            self.stellar_spectrum = _spec
        else:
            return _spec

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
        if np.sum(age_mask) < 1:
            return np.empty([1, grid.line_luminosities.shape[0]])

        lum = np.zeros((np.sum(age_mask), len(grid.lines)))
       
        for i, (mass, age, metal) in enumerate(zip(
                self.stars.initial_masses[age_mask],
                self.stars.log10ages[age_mask],
                self.stars.log10metallicities[age_mask])):
                
            weights_temp = self._calculate_weights(grid, metal, age, mass, 
                                                   young_stars=True)
            lum[i] = np.sum(grid.line_luminosities * weights_temp, axis=(1,2))

        return lum

    def calculate_stellar_line_luminosities(self, grid, save=False, verbose=False):
        """
        Calculate integrated line luminosities for whole galaxy

        Args
        grid (object)
        save (bool, False) determines if line luminosities dict saved
                           to galaxy object. If false, return dict
        """
        # lum = self.stellar_particle_line_luminosities(grid)
        
        age_mask = self.stars.log10ages < grid.max_age
        if np.sum(age_mask) < 1:
            return np.empty([1, grid.line_luminosities.shape[0]])

        weights_temp = self._calculate_weights(grid,
                self.stars.log10metallicities[age_mask],
                self.stars.log10ages[age_mask],
                self.stars.initial_masses[age_mask], young_stars=True)

        lum = np.sum(grid.line_luminosities * weights_temp, axis=(1,2))
        
        if lum is None:
            if verbose:
                print("Warning: no particles below max age limit")
            
            line_lums = None
        else:
            # lum_arr = np.sum(lum, axis=0)
            line_lums = {}
            for i, line in enumerate(grid.lines):
                line_lums[line] = lum[i]
            
        if save:
            self.stellar_line_luminosities = line_lums
        else:
            return line_lums
