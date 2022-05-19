from .particles import Particles 

class Stars(Particles):
    def __init__(self, masses, ages, metallicities):
        self.masses = masses
        self.ages = ages
        self.metallicities = metallicities



