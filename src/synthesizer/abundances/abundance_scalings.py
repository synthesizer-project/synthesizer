"""A module containing various elemental abundance scalings.

This modules provides methods to scale various elements with metallicity
instead of simply linearly.

Example usage:
    from synthesizer.abundances import abundance_scalings

    # create an instance of the scaling class
    scaling = abundance_scalings.GalacticConcordance()

    # get the Nitrogen abundance for a given metallicity
    nitrogen_abundance = scaling.nitrogen(0.02)

    # get the Carbon abundance for a given metallicity
    carbon_abundance = scaling.carbon(0.02)
"""

import numpy as np

from synthesizer.abundances import (
    elements,
)

# list of available scalings
available_scalings = ["Dopita2006", "GalacticConcordance"]


class Dopita2006:
    """The Dopita (2006) model for abundance scalings.

    This includes scalings for Nitrogen and Carbon.

    References:
        Dopita, M. A., Kewley, L. J., & Heisler, C. A. 2006,
        ApJS, 167, 177
        doi:10.1086/508261
        https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract


    The scaling for Nitrogen is given by:
        log10(N/H) = log10(1.1e-5 * Z + 4.9e-5 * Z^2)

    The scaling for Carbon is given by:
        log10(C/H) = log10(6e-5 * Z + 2e-4 * Z^2)

    where Z is the metallicity (mass fraction in metals).

    The reference metallicity for the model is 0.016.
    The scaling for other elements is not provided in the original paper.

    Attributes:
        ads (str):
            The ADS link to the paper.
        doi (str):
            The DOI of the paper.
        available_elements (list):
            List of available elements for scaling.
        reference_metallicity (float):
            The reference metallicity for the model.
    """

    def __init__(self):
        """Initialise the Dopita2006 scalings."""
        self.ads = (
            "https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract"
        )
        self.doi = "10.1086/508261"
        self.available_elements = ["N", "C"]

        # the reference metallicity for the model
        self.reference_metallicity = 0.016

    def nitrogen(self, metallicity):
        """Scaling function for Nitrogen.

        Args:
            metallicity (float):
                The metallicity (mass fraction in metals)

        Returns:
            abundance (float):
                The logarithmic abundance of Nitrogen relative to Hydrogen.

        """
        # the metallicity scaled to the Dopita (2006) value
        scaled_metallicity = metallicity / self.reference_metallicity

        abundance = np.log10(
            1.1e-5 * scaled_metallicity + 4.9e-5 * (scaled_metallicity) ** 2
        )

        return abundance

    def carbon(self, metallicity):
        """Scaling functions for Carbon.

        Args:
            metallicity (float):
                The metallicity (mass fraction in metals)

        Returns:
            abundance (float):
                The logarithmic abundance of Carbon relative to Hydrogen.

        """
        # the metallicity scaled to the Dopita (2006) value
        scaled_metallicity = metallicity / self.reference_metallicity

        abundance = np.log10(
            6e-5 * scaled_metallicity + 2e-4 * (scaled_metallicity) ** 2
        )

        return abundance


class GalacticConcordance:
    """The "Galactic Concordance" model from Nicholls et al. 2017.

    In addition to providing a reference abundance pattern Nicholls et al.
    2017 provide scalings for a large range of elements.

    For most of these the scaling, relative to Oxygen, is a simple linear
    scaling with plataus at low and high O/H.

    For Nitrogen and Carbon a more complicated scheme is used.

    References:
        Nicholls, D. C., Dopita, M. A., & Sutherland, R. S. 2017,
        MNRAS, 466, 4403
        doi:10.1093/mnras/stw3235
        https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.4403N/abstract

    The scaling for Nitrogen is given by:
        log10(N/H) = log10(10^(-1.732) + 10^(O/H + 2.19))

    The scaling for Carbon is given by:
        log10(C/H) = log10(10^(-0.8) + 10^(O/H + 2.72))

    where O/H is the Oxygen to Hydrogen ratio.

    The reference metallicity for the model is 0.015.
    The scaling for other elements is not provided in the original paper.

    Attributes:
        ads (str):
            The ADS link to the paper.
        doi (str):
            The DOI of the paper.
        available_elements (list):
            List of available elements for scaling.
        reference_metallicity (float):
            The reference metallicity for the model.
        reference_abundance (dict):
            Dictionary of reference abundances.
        scaling_parameters (dict):
            Dictionary of scaling parameters for each element.
    """

    def __init__(self):
        """Initialise the GalacticConcordance scalings."""
        # Meta information
        self.ads = (
            "https://ui.adsabs.harvard.edu/abs/2017MNRAS.466.4403N/abstract"
        )
        self.doi = "https://doi.org/10.1093/mnras/stw3235"
        self.arxiv = None
        self.bibcode = "2017MNRAS.466.4403N"

        # the reference metallicity (mass fraction in metals), i.e. Z
        self.reference_metallicity = 0.015

        # logarthmic abundances, i.e. log10(N_element/N_H)
        self.reference_abundance = {
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

        # the reference oxygen abundance for GalacticConcordance, i.e. [O/H]
        self.reference_oxygen_abundance = self.reference_abundance["O"]

        # these are the scaling parameters used for elements other than CNO
        # and a handful of light elements
        self.scaling_parameters = {
            "F": 0.00,
            "Ne": 0.00,
            "Na": -0.30,
            "Mg": -0.10,
            "Al": -0.10,
            "Si": -0.10,
            "P": -0.50,
            "S": -0.10,
            "Cl": 0.00,
            "Ar": 0.00,
            "K": -0.10,
            "Ca": -0.15,
            "Sc": -0.25,
            "Ti": -0.15,
            "V": -0.50,
            "Cr": -0.50,
            "Mn": -0.50,
            "Fe": -0.50,
            "Co": -0.50,
            "Ni": -0.50,
            "Cu": -0.50,
            "Zn": -0.30,
        }

        # elements that have a predefined method
        self.available_elements = ["N", "C"]
        self.available_elements_names = ["nitrogen", "carbon"]

        # initialise elements dataclass to access dictionary mapping id to name
        element_info = elements.Elements()

        # create a method for every element contained in scaling_parameters
        for element, scaling_parameter in self.scaling_parameters.items():
            # get element name
            element_name = element_info.name[element]

            # create method
            setattr(
                self,
                element_name,
                self.scaling_method_creator(
                    element,
                    scaling_parameter,
                    reference_abundances=self.reference_abundance,
                ),
            )

            # append element to available element list
            self.available_elements.append(element)
            self.available_elements_names.append(element_name)

    class scaling_method_creator:
        """A class for making scaling methods for individual elements."""

        def __init__(
            self,
            element,
            scaling_parameter,
            lower_break=-0.25,
            upper_break=0.5,
            reference_metallicity=0.015,
            reference_abundances={},
        ):
            """Initialiation for a specific element.

            Arguments:
                element (str):
                    The element
                scaling_parameter (float):
                    The scaling parameter, the slope of of the scaling
                    relation.
                lower_break (float):
                    The lower-break (O/H) below which the abundance simply
                    scales with metallicity. By default -0.25.
                upper_break (float):
                    The upper-break (O/H) above which the abundance simply
                    scales with metallicity. By default 0.5.
                reference_metallicity (float):
                    The reference metallicity.
                reference_abundances (dict, float):
                    Dictionary of reference abundances.

            Returns:
                abundance (float):
                    The abundance of the specific element.
            """
            self.element = element
            self.lower_break = lower_break
            self.upper_break = upper_break
            self.scaling_parameter = scaling_parameter

            # should try and inherit these
            self.reference_metallicity = reference_metallicity
            self.reference_abundances = reference_abundances

        def __call__(self, metallicity):
            """Calculate the abundance for a given metallicity.

            Arguments:
                metallicity (float):
                    The metallicity.

            Returns:
                abundance (float):
                    The abundance of the specific element.
            """
            # calculate log10(oxygen to hydrogen) for a given metallicity
            log10xi = np.log10(metallicity / self.reference_metallicity)

            if log10xi < self.lower_break:
                delta_x = self.scaling_parameter
            elif log10xi > self.upper_break:
                delta_x = (
                    self.scaling_parameter / self.lower_break
                ) * self.upper_break
            else:
                delta_x = (self.scaling_parameter / self.lower_break) * log10xi

            abundance = (
                self.reference_abundances[self.element] + delta_x + log10xi
            )

            return abundance

    def oxygen_to_hydrogen(self, metallicity):
        """Calculate the oxygen to hydrogen ratio (O/H) for a metallicity.

        Arguments:
            metallicity (float):
                The metallicity.

        Returns:
            oxygen_to_hydrogen (float):
                The Oxygen to Hydrogen ratio (O/H).
        """
        # calculate oxygen to hydroge for a given metallicity
        return self.reference_oxygen_abundance + np.log10(
            metallicity / self.reference_metallicity
        )

    def nitrogen_to_oxygen(self, oxygen_to_hydrogen):
        """Calculate the Nitrogen to Oxygen abundance for a given (O/H).

        Args:
            oxygen_to_hydrogen (float):
                The log of the Oxygen to Hydrogen ratio, i.e. (O/H).

        Returns:
            nitrogen_to_oxygen (float):
                The logarithmic abundance relative to Oxygen.

        """
        a = -1.732
        b = 2.19
        nitrogen_to_oxygen = np.log10(10**a + 10 ** (oxygen_to_hydrogen + b))

        return nitrogen_to_oxygen

    def nitrogen(self, metallicity):
        """Calculate the Nitrogen abundance (N/H) for a given metallicity.

        Args:
            metallicity (float):
                The metallicity.

        Returns:
            abundance (float):
                The logarithmic Nitrogen abundance, i.e. (N/H).

        """
        # calculate oxygen to hydrogen for a given metallicity
        oxygen_to_hydrogen = self.oxygen_to_hydrogen(metallicity)

        abundance = (
            self.nitrogen_to_oxygen(oxygen_to_hydrogen) + oxygen_to_hydrogen
        )

        return abundance

    def carbon_to_oxygen(self, oxygen_to_hydrogen):
        """Calculate the Carbon to Oxygen abundance for a given (O/H).

        Args:
            oxygen_to_hydrogen (float):
                The log of the Oxygen to Hydrogen ratio, i.e. (O/H).

        Returns:
            carbon_to_oxygen (float):
                The logarithmic abundance relative to Oxygen.

        """
        a = -0.8
        b = 2.72
        carbon_to_oxygen = np.log10(10**a + 10 ** (oxygen_to_hydrogen + b))

        return carbon_to_oxygen

    def carbon(self, metallicity):
        """Calculate the Carbon abundance (C/H) for a given metallicity.

        Args:
            metallicity (float):
                The metallicity.

        Returns:
            abundance (float):
                The logarithmic Carbon abundance, i.e. (C/H).

        """
        # calculate oxygen to hydrogen for a given metallicity
        oxygen_to_hydrogen = self.oxygen_to_hydrogen(metallicity)

        abundance = (
            self.carbon_to_oxygen(oxygen_to_hydrogen) + oxygen_to_hydrogen
        )

        return abundance
