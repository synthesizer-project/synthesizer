available_patterns = ["Asplund2009", "Gutkin2016"]


class Asplund2009:

    """
    The Solar abundance pattern used by Asplund (2009).
    """

    # meta information
    ads = """https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/
        abstract"""
    doi = "10.1146/annurev.astro.46.060407.145222"
    arxiv = "arXiv:0909.0948"
    bibcode = "2009ARA&A..47..481A"

    # total metallicity
    metallicity = 0.0134

    # logarthmic abundances, i.e. log10(N_element/N_H)
    abundance = {
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


class Gutkin2016:

    """
    The Solar abundance pattern used by Gutkin (2016).
    """

    # meta information
    ads = """https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/
    abstract"""
    doi = "10.1093/mnras/stw1716"
    arxiv = "arXiv:1607.06086"
    bibcode = "2016MNRAS.462.1757G"

    # total metallicity
    metallicity = 0.01524

    # logarthmic abundances, i.e. log10(N_element/N_H)
    abundance = {
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
