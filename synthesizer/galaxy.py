import numpy as np

from .stars import Stars


class Galaxy:
    def __init__(self):
        self.name = 'galaxy'

    def load_stars(self, masses, ages, metals, **kwargs):
        self.stars = Stars(masses, ages, metals, **kwargs)

    def stellar_particle_spectra(self, grid):
        """
        Calculate spectra for all individual stellar particles

        Returns
        lum (array) spectrum for each particle, (N_part, wl)
        """
        lum = np.zeros((len(self.stars.masses), grid.spectra.shape[-1]))
        for i, (mass, age, metal) in enumerate(zip(
                self.stars.masses,
                self.stars.log10ages,
                self.stars.log10metallicities)):

            ia = self.NGP_assignment(grid, 'ages', age)
            iZ = self.NGP_assignment(grid, 'metallicities', metal)
            # TODO: alternative interpolation schemes

            lum[i] = mass * grid.spectra[ia, iZ]

        return lum

    def calculate_stellar_spectrum(self, grid, save=False):
        """
        Calculate integrated spectrum for whole galaxy

        Args
        grid: grid object
        save (bool, False): determines if spectra saved to galaxy object
                            if False, method returns spectrum as array
        """
        lum = self.stellar_particle_spectra(grid)
        _spec = np.sum(lum, axis=0)
        if save:
            self.stellar_spectrum = _spec
        else:
            return _spec

    def stellar_particle_line_luminosities(self, grid):
        age_mask = self.stars.log10ages < grid.max_age
        if np.sum(age_mask) < 1:
            return np.empty([1, grid.line_luminosities.shape[0]])

        lum = np.zeros((np.sum(age_mask), len(grid.lines)))
        for i, (mass, age, metal) in enumerate(zip(self.stars.masses[age_mask],
                             self.stars.log10ages[age_mask],
                             self.stars.log10metallicities[age_mask])):

            ia = self.NGP_assignment(grid, 'ages', age)
            iZ = self.NGP_assignment(grid, 'metallicities', metal)
            # TODO: alternative interpolation schemes

            lum[i] = mass * grid.line_luminosities[:, iZ, ia]

        return lum

    def calculate_stellar_line_luminosities(self, grid, save=False, verbose=False):
        """
        Calculate integrated line luminosities for whole galaxy

        Args
        grid (object)
        save (bool, False) determines if line luminosities dict saved
                           to galaxy object. If false, return dict
        """
        lum = self.stellar_particle_line_luminosities(grid)
        if lum is None:
            if verbose:
                print("Warning: no particles below max age limit")
            
            line_lums = None
        else:
            lum_arr = np.sum(lum, axis=0)
            line_lums = {}
            for i, line in enumerate(grid.lines):
                line_lums[line] = lum_arr[i]
            
        if save:
            self.stellar_line_luminosities = line_lums
        else:
            return line_lums
        
    def NGP_assignment(self, grid, parameter, value):
        """
        Nearest Grid Point (NGP) assignment
        """
        _arr = eval(f'grid.{parameter}')
        return (np.abs(_arr - value)).argmin()
