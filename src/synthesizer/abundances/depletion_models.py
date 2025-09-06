r"""A module defining depletion model for use in photoionization models.

Module containing various depletion models. Depletion models relate the gas
phase depleted abundances to the total abundances, i.e.:
    (X/H)_{gas,dep} = D_{x}\times (X/H)_{total}
    (X/H)_{dust} = (1-D_{x})\times (X/H)_{total}
"""

import numpy as np

available_patterns = [
    "Jenkins2009_Gunasekera2021",
    "CloudyClassic",
    "Gutkin2016",
]


class Jenkins2009_Gunasekera2021:
    """Implemention of the Jenkins (2009) depletion pattern.

    This is the model that is built into cloudy23 as described by
    Gunasekera (2021). This modification adds in additional elements that
    were not considered by Jenkins (2009).

    In this model the depletion (D_x) is written as:

    D_x = 10**(B_x +A_x (F_* − z_x ))

    where A_x, B_x, and z_X are the fitted parameters for each element while
    f_* (fstar) is the scaling parameter.
    """

    # (a_x, b_x, z_x)
    #
    parameters = {
        # "H": 1.0,
        # "He": 1.0,
        "Li": (-1.136, -0.246, 0.000),
        # "Be": 0.6,
        "B": (-0.849, 0.698, 0.000),
        "C": (-0.101, -0.193, 0.803),
        "N": (0.00, -0.11, 0.55),
        "O": (-0.23, -0.15, 0.60),
        # "F": 0.3,
        # "Ne": 1.0,
        "Na": (2.071, -3.059, 0.000),
        "Mg": (-1.00, -0.80, 0.53),
        "Al": (-3.330, 0.179, 0.000),
        "Si": (-1.14, -0.57, 0.31),
        "P": (-0.95, -0.17, 0.49),
        "S": (-0.88, -0.09, 0.29),
        "Cl": (-1.24, -0.31, 0.61),
        "Ar": (-0.516, -0.133, 0.000),
        "K": (-0.133, -0.859, 0.000),
        "Ca": (-1.822, -1.768, 0.000),
        # "Sc": 0.005,
        "Ti": (-2.05, -1.96, 0.43),
        # "V": 0.006,
        "Cr": (-1.45, -1.51, 0.47),
        "Mn": (-0.86, -1.35, 0.52),
        "Fe": (-1.29, -1.51, 0.44),
        # "Co": 0.01,
        "Ni": (-1.49, -1.83, 0.60),
        "Cu": (-0.71, -1.10, 0.71),
        "Zn": (-0.61, -0.28, 0.56),
    }

    def __init__(self, fstar=0.5, limit=1.0):
        """Initialise the class.

        Args:
            fstar (float):
                The Jenkins (2009) scaling parameter.
            limit (float):
                The maximum depletion value. This is used to limit the
                depletion to a maximum value. This is useful for elements
                that are not depleted in the ISM, e.g. H and He.
        """
        self.depletion = {}
        for element, parameters in self.parameters.items():
            # unpack parameters. Despite convention I've chosen to use
            a_x, b_x, z_x = parameters
            # calculate depletion, including limit

            depletion = np.min([limit, 10 ** (b_x + a_x * (fstar - z_x))])
            self.depletion[element] = depletion


class CloudyClassic:
    """Implementation of the 'cloudy classic' depletion pattern.

    This is the same as the one that is built into cloudy23.
    """

    depletion_ = {
        "H": 1.0,
        "He": 1.0,
        "Li": 0.16,
        "Be": 0.6,
        "B": 0.13,
        "C": 0.4,
        "N": 1.0,
        "O": 0.6,
        "F": 0.3,
        "Ne": 1.0,
        "Na": 0.2,
        "Mg": 0.2,
        "Al": 0.01,
        "Si": 0.03,
        "P": 0.25,
        "S": 1.0,
        "Cl": 0.4,
        "Ar": 1.0,
        "K": 0.3,
        "Ca": 0.0001,
        "Sc": 0.005,
        "Ti": 0.008,
        "V": 0.006,
        "Cr": 0.006,
        "Mn": 0.05,
        "Fe": 0.01,
        "Co": 0.01,
        "Ni": 0.01,
        "Cu": 0.1,
        "Zn": 0.25,
    }

    def __init__(self, scale=1.0):
        """Initialise the cloudy classic depletion pattern.

        Args:
            scale (float):
                Scale factor for the depletion.
        """
        self.depletion = {
            element: scale * depletion
            for element, depletion in self.depletion_.items()
        }


class Gutkin2016:
    """Depletion pattern created for Synthesizer 2024.

    Gutkin+2016 (ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/abstract)

    Note: in previous version we adjusted N ("N": 0.89) following Dopita+2006
    (ui.adsabs.harvard.edu/abs/2013ApJS..208...10D/abstrac) and Dopita+2013
    (ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract)
    """

    # This is the inverse depletion
    depletion_ = {
        "H": 1.0,
        "He": 1.0,
        "Li": 0.16,
        "Be": 0.6,
        "B": 0.13,
        "C": 0.5,
        "N": 1.0,
        "O": 0.7,
        "F": 0.3,
        "Ne": 1.0,
        "Na": 0.25,
        "Mg": 0.2,
        "Al": 0.02,
        "Si": 0.1,
        "P": 0.25,
        "S": 1.0,
        "Cl": 0.5,
        "Ar": 1.0,
        "K": 0.3,
        "Ca": 0.003,
        "Sc": 0.005,
        "Ti": 0.008,
        "V": 0.006,
        "Cr": 0.006,
        "Mn": 0.05,
        "Fe": 0.01,
        "Co": 0.01,
        "Ni": 0.04,
        "Cu": 0.1,
        "Zn": 0.25,
    }

    def __init__(self, scale=1.0):
        """Initialise the Gutkin (2016) depletion pattern.

        Args:
            scale (float):
                Scale factor for the depletion.
        """
        self.depletion = {
            element: scale * depletion
            for element, depletion in self.depletion_.items()
        }
