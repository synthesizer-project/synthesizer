import numpy as np

from .particles import Particles


class Stars(Particles):
    def __init__(self, masses, ages, metallicities):
        self.masses = masses
        self.ages = ages
        self.metallicities = metallicities

        self.log10ages = np.log10(self.ages)
        self.log10metallicities = np.log10(self.metallicities)
