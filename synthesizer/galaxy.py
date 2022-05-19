from .stars import Stars

class Galaxy:
    def __init__(self):
        self.name = 'galaxy'

    def load_stars(self, masses, ages, metals):
        self.stars = Stars(masses, ages, metals)

    def stellar_spectra(self):
        # placeholder
        return None
