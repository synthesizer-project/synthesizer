"""
Module containing various reference abundance patterns.
"""

from dataclasses import dataclass

# list of available reference patterns
available_patterns = ["Asplund2009", "GalacticConcordance", "Gutkin2016"]


@dataclass
class Asplund2009:
    def __init__(self):
        """
        The Solar abundance pattern used by Asplund (2009).
        """

        # meta information
        self.ads = (
            "https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/"
            "abstract"
        )
        self.doi = "10.1146/annurev.astro.46.060407.145222"
        self.arxiv = "arXiv:0909.0948"
        self.bibcode = "2009ARA&A..47..481A"

        # total metallicity
        self.metallicity = 0.0134

        # logarthmic abundances, i.e. log10(N_element/N_H)
        self.abundance = {
            "H": 0.0,
            "He": -1.07,
            "Li": -10.95,
            "Be": -10.62,
            "B": -9.3,
            "C": -3.57,
            "N": -4.17,
            "O": -3.31,
            "F": -7.44,
            "Ne": -4.07,
            "Na": -5.07,
            "Mg": -4.40,
            "Al": -5.55,
            "Si": -4.49,
            "P": -6.59,
            "S": -4.88,
            "Cl": -6.5,
            "Ar": -5.60,
            "K": -6.97,
            "Ca": -5.66,
            "Sc": -8.85,
            "Ti": -7.05,
            "V": -8.07,
            "Cr": -6.36,
            "Mn": -6.57,
            "Fe": -4.50,
            "Co": -7.01,
            "Ni": -5.78,
            "Cu": -7.81,
            "Zn": -7.44,
        }


@dataclass
class GalacticConcordance:
    def __init__(self):
        """
        The Galactic Concordance pattern from Nicholls et al.
        """

        # meta information
        self.ads = (
            "https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.4403N/" "abstract"
        )
        self.doi = "https://doi.org/10.1093/mnras/stw3235"
        self.arxiv = None
        self.bibcode = "2017MNRAS.466.4403N"

        # total metallicity
        self.metallicity = 0.015

        # logarthmic abundances, i.e. log10(N_element/N_H)
        self.abundance = {
            "H": 0.0,
            "He": -1.09,
            "Li": -8.722,
            "Be": -10.68,
            "B": -9.193,
            "C": -3.577,
            "N": -4.21,
            "O": -3.24,
            "F": -7.56,
            "Ne": -3.91,
            "Na": -5.79,
            "Mg": -4.44,
            "Al": -5.57,
            "Si": -4.50,
            "P": -6.59,
            "S": -4.88,
            "Cl": -6.75,
            "Ar": -5.60,
            "K": -6.96,
            "Ca": -5.68,
            "Sc": -8.84,
            "Ti": -7.07,
            "V": -8.11,
            "Cr": -6.38,
            "Mn": -6.58,
            "Fe": -4.48,
            "Co": -7.07,
            "Ni": -5.80,
            "Cu": -7.82,
            "Zn": -7.44,
        }


@dataclass
class Gutkin2016:
    def __init__(self):
        """
        The Solar abundance pattern used by Gutkin (2016).
        """

        # meta information
        self.ads = """https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/
        abstract"""
        self.doi = "10.1093/mnras/stw1716"
        self.arxiv = "arXiv:1607.06086"
        self.bibcode = "2016MNRAS.462.1757G"

        # total metallicity
        self.metallicity = 0.01524

        # logarthmic abundances, i.e. log10(N_element/N_H)
        self.abundance = {
            "H": 0.0,
            "He": -1.01,
            "Li": -10.99,
            "Be": -10.63,
            "B": -9.47,
            "C": -3.53,
            "N": -4.32,
            "O": -3.17,
            "F": -7.44,
            "Ne": -4.01,
            "Na": -5.70,
            "Mg": -4.45,
            "Al": -5.56,
            "Si": -4.48,
            "P": -6.57,
            "S": -4.87,
            "Cl": -6.53,
            "Ar": -5.63,
            "K": -6.92,
            "Ca": -5.67,
            "Sc": -8.86,
            "Ti": -7.01,
            "V": -8.03,
            "Cr": -6.36,
            "Mn": -6.64,
            "Fe": -4.51,
            "Co": -7.11,
            "Ni": -5.78,
            "Cu": -7.82,
            "Zn": -7.43,
        }
