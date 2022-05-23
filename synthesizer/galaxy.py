import numpy as np

from .stars import Stars


class Galaxy:
    def __init__(self):
        self.name = 'galaxy'

    def load_stars(self, masses, ages, metals):
        self.stars = Stars(masses, ages, metals)

    def stellar_particle_spectra(self, grid):
        
        l = np.zeros((len(self.stars.masses), grid.spectra.shape[-1]))
        for i, (mass, age, metal) in enumerate(zip(
                self.stars.masses,
                self.stars.log10ages,
                self.stars.log10metallicities)):

            # NGP assignment scheme
            ia = (np.abs(grid.ages - age)).argmin()
            iZ = (np.abs(grid.metallicities - metal)).argmin()

            # TODO: alternative interpolation schemes

            l[i] = mass * grid.spectra[ia, iZ]

        return l

    def calculate_stellar_spectrum(self, grid, save=False):
        l = self.stellar_particle_spectra(grid)
        _spec = np.sum(l, axis=0)
        if save:
            self.stellar_spectrum = _spec
        else:
            return _spec

