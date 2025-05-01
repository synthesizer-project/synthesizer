"""A module containing various reference abundance patterns."""

from dataclasses import dataclass, field

# list of available reference patterns
available_patterns = ["Asplund2009", "GalacticConcordance", "Gutkin2016"]


@dataclass
class Asplund2009:
    """A class defining the Asplund (2009) solar abundance pattern.

    The Asplund (2009) solar abundance pattern is a widely used reference
    abundance pattern for stars and galaxies. It is based on the solar
    abundances of elements and is often used as a standard for comparing
    the chemical composition of other stars and galaxies. The pattern
    includes the relative abundances of various elements, such as hydrogen,
    helium, carbon, nitrogen, oxygen, and others, in logarithmic form.

    References:
        Asplund, M., Grevesse, N., & Sauval, A. J. (2009). The solar
        abundances of the elements: Solar and galactic chemical evolution.
        Annual Review of Astronomy and Astrophysics, 47(1), 481-522.
        doi:10.1146/annurev.astro.46.060407.145222
        https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract

    Attributes:
        ads (str): The ADS link to the Asplund (2009) paper.
        doi (str): The DOI of the Asplund (2009) paper.
        arxiv (str): The arXiv link to the Asplund (2009) paper.
        bibcode (str): The bibcode of the Asplund (2009) paper.
        metallicity (float): The total metallicity of the pattern.
        abundance (dict): A dictionary containing the logarithmic
            abundances of various elements in the pattern.
    """

    # meta information
    ads: str = (
        "https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract"
    )
    doi: str = "10.1146/annurev.astro.46.060407.145222"
    arxiv: str = "arXiv:0909.0948"
    bibcode: str = "2009ARA&A..47..481A"

    # total metallicity
    metallicity: float = 0.0134

    # logarithmic abundances, i.e. log10(N_element/N_H)
    abundance: dict = field(
        default_factory=lambda: {
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
    )


@dataclass
class GalacticConcordance:
    """A class defining the Galactic Concordance abundance pattern.

    The Galactic Concordance abundance pattern is a reference
    abundance pattern for stars and galaxies, based on the
    Galactic Concordance model. It is used to study the chemical
    evolution of galaxies and the formation of stars. The pattern
    includes the relative abundances of various elements, such as
    hydrogen, helium, carbon, nitrogen, oxygen, and others, in
    logarithmic form.

    References:
        Nicholls, D. C., Dopita, M. A., & Sutherland, R. S. (2017).
        The Galactic Concordance abundance pattern: A new standard
        for the chemical evolution of galaxies. Monthly Notices of
        the Royal Astronomical Society, 466(4), 4403-4415.
        doi:10.1093/mnras/stw3235
        https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.4403N/abstract

    Attributes:
        ads (str): The ADS link to the Galactic Concordance paper.
        doi (str): The DOI of the Galactic Concordance paper.
        arxiv (str): The arXiv link to the Galactic Concordance paper.
        bibcode (str): The bibcode of the Galactic Concordance paper.
        metallicity (float): The total metallicity of the pattern.
        abundance (dict): A dictionary containing the logarithmic
            abundances of various elements in the pattern.
    """

    # meta information
    ads: str = "https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.4403N/abstract"
    doi: str = "https://doi.org/10.1093/mnras/stw3235"
    arxiv: str = "arxiv:1612.03546"
    bibcode: str = "2017MNRAS.466.4403N"

    # total metallicity
    metallicity: float = 0.015

    # logarithmic abundances, i.e. log10(N_element/N_H)
    abundance: dict = field(
        default_factory=lambda: {
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
    )


@dataclass
class Gutkin2016:
    """A class defining the Gutkin (2016) solar abundance pattern.

    The Gutkin (2016) solar abundance pattern is a reference abundance
    pattern for stars and galaxies, based on the solar abundances of
    elements. It is used to study the chemical evolution of galaxies
    and the formation of stars. The pattern includes the relative
    abundances of various elements, such as hydrogen, helium, carbon,
    nitrogen, oxygen, and others, in logarithmic form.

    References:
        Gutkin, J., et al. (2016). The stellar population synthesis
        model of the MPA/JHU group: A new approach to the analysis of
        galaxy spectra. Monthly Notices of the Royal Astronomical
        Society, 462(2), 1757-1774. doi:10.1093/mnras/stw1716
        https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/abstract

    Attributes:
        ads (str): The ADS link to the Gutkin (2016) paper.
        doi (str): The DOI of the Gutkin (2016) paper.
        arxiv (str): The arXiv link to the Gutkin (2016) paper.
        bibcode (str): The bibcode of the Gutkin (2016) paper.
        metallicity (float): The total metallicity of the pattern.
        abundance (dict): A dictionary containing the logarithmic
            abundances of various elements in the pattern.
    """

    doi: str = "10.1093/mnras/stw1716"
    arxiv: str = "arXiv:1607.06086"
    bibcode: str = "2016MNRAS.462.1757G"

    # total metallicity
    metallicity: float = 0.01524

    # logarithmic abundances, i.e. log10(N_element/N_H)
    abundance: dict = field(
        default_factory=lambda: {
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
    )
