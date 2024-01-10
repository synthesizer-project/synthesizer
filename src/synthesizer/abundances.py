"""A module for creating and manipulating abundance patterns

Abundance patterns describe the relative abundances of elements in a particular
component of a galaxy (e.g. stars, gas, dust). This code is used to define
abundance patterns as a function of metallicity, alpha enhancement, etc.

The main current use of this code is in the creation cloudy input models when
processing SPS incident grids to model nebular emission.

This script is a modified version of
https://github.com/stephenmwilkins/SPS_tools/blob/master/SPS_tools/cloudy/abundances.py

Some notes on (standard) notation:
- [X/H] = log10(N_X/N_H) - log10(N_X/N_H)_sol
"""

from copy import deepcopy
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt

from synthesizer.exceptions import InconsistentParameter


class ScalingFunctions:

    """
    This is a class holds scaling functions for individual elements. In each
    case the function returns the logarthimic abundance relative to Hydrogen
    for a given metallicity.

    Example:

    ScalingFunctions.N.Dopita2006(0.016)

    or

    element_functions = getattr(ScalingFunctions, 'N')
    scaling_function = getattr(element_functions, 'Dopita2006')
    scaling_function(0.016)

    """

    available_scalings = ['Dopita2006']

    class Dopita2006:

        """Scaling functions for Nitrogen."""

        ads = "https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract"
        doi = "10.1086/508261"
        available_elements = ['N', 'C']

        def N(metallicity):
            """

            Args:
                metallicity (float)
                    The metallicity (mass fraction in metals)

            Returns:
                abundance (float)
                    The logarithmic abundance relative to Hydrogen.

            """

            # the metallicity scaled to the Dopita (2006) value
            dopita_solar_metallicity = 0.016
            scaled_metallicity = metallicity / dopita_solar_metallicity

            abundance = np.log10(
                1.1e-5 * scaled_metallicity
                + 4.9e-5 * (scaled_metallicity) ** 2
            )

            return abundance

        def C(metallicity):
            """
            Scaling functions for Carbon.

            Args:
                metallicity (float)
                    The metallicity (mass fraction in metals)

            Returns:
                abundance (float)
                    The logarithmic abundance relative to Hydrogen.

            """

            # the metallicity scaled to the Dopita (2006) value
            dopita_solar_metallicity = 0.016
            scaled_metallicity = metallicity / dopita_solar_metallicity

            abundance = np.log10(
                6e-5 * scaled_metallicity + 2e-4 * (scaled_metallicity) ** 2
            )

            return abundance


class ElementDefinitions:

    """
    A simple class containing various useful lists and dictionaries.

    Attributes:
        non_metals (list, string)
            A list of elements classified as non-metals.
        metals (list, string)
            A list of elements classified as metals.
        all_elements (list, string)
            A list of all elements, functionally the concatenation of metals
            and non-metals.
        alpha_elements (list, string)
            A list of the elements classified as alpha-elements.
        name (dict, string)
            A dictionary holding the full name of each element.
        A (dict, float)
            Atomic mass of each element (in amus).
    """

    non_metals = [
        "H",
        "He",
    ]

    metals = [
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
    ]

    all_elements = metals + non_metals

    alpha_elements = [
        "O",
        "Ne",
        "Mg",
        "Si",
        "S",
        "Ar",
        "Ca",
        "Ti",
    ]  # the alpha process elements

    name = {}
    name["H"] = "Hydrogen"
    name["He"] = "Helium"
    name["Li"] = "Lithium"
    name["Be"] = "Beryllium"
    name["B"] = "Boron"
    name["C"] = "Carbon"
    name["N"] = "Nitrogen"
    name["O"] = "Oxygen"
    name["F"] = "Fluorine"
    name["Ne"] = "Neon"
    name["Na"] = "Sodium"
    name["Mg"] = "Magnesium"
    name["Al"] = "Aluminium"
    name["Si"] = "Silicon"
    name["P"] = "Phosphorus"
    name["S"] = "Sulphur"
    name["Cl"] = "Chlorine"
    name["Ar"] = "Argon"
    name["K"] = "Potassium"
    name["Ca"] = "Calcium"
    name["Sc"] = "Scandium"
    name["Ti"] = "Titanium"
    name["V"] = "Vanadium"
    name["Cr"] = "Chromium"
    name["Mn"] = "Manganese"
    name["Fe"] = "Iron"
    name["Co"] = "Cobalt"
    name["Ni"] = "Nickel"
    name["Cu"] = "Copper"
    name["Zn"] = "Zinc"

    # mass of elements in amus
    A = {}
    A["H"] = 1.008
    A["He"] = 4.003
    A["Li"] = 6.940
    A["Be"] = 9.012
    A["B"] = 10.81
    A["C"] = 12.011
    A["N"] = 14.007
    A["O"] = 15.999
    A["F"] = 18.998
    A["Ne"] = 20.180
    A["Na"] = 22.990
    A["Mg"] = 24.305
    A["Al"] = 26.982
    A["Si"] = 28.085
    A["P"] = 30.973
    A["S"] = 32.06
    A["Cl"] = 35.45
    A["Ar"] = 39.948
    A["K"] = 39.0983
    A["Ca"] = 40.078
    A["Sc"] = 44.955
    A["Ti"] = 47.867
    A["V"] = 50.9415
    A["Cr"] = 51.9961
    A["Mn"] = 54.938
    A["Fe"] = 55.845
    A["Co"] = 58.933
    A["Ni"] = 58.693
    A["Cu"] = 63.546
    A["Zn"] = 65.38


class SolarAbundances:

    """
    A class containing various Solar abundance patterns.
    """

    available_patterns = ['Asplund2009']

    class Asplund2009:

        # meta information
        ads = """https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/
            abstract"""
        doi = '10.1146/annurev.astro.46.060407.145222'
        arxiv = 'arXiv:0909.0948'
        bibcode = '2009ARA&A..47..481A'

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

class DepletionPatterns:

    """
    Class containing various depletion patterns.

    Noble gases aren't depleted (even though there is some
    evidence for Argon depletion
    (see https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.2310G/abstract)

    """

    available_patterns = ['Synthesizer2024']


    class Jenkins2009:

        """
        Implemention of the Jenkins (2009) depletion pattern that is built into
        cloudy23.
        """

        # (AX, BX, zX)
        parameters = {
            # "H": 1.0,
            # "He": 1.0,
            "Li": (-1.136, -0.246, 0.000),
            # "Be": 0.6,
            "B": (-0.101,  -0.193,  0.803),
            "C": (-0.10, -0.19, 0.80),
            "N": (0.00, -0.11, 0.55),
            "O": (-0.23, -0.15, 0.60),
            # "F": 0.3,
            # "Ne": 1.0,
            "Na": (2.071,  -3.059,  0.000),
            "Mg": (-1.00, -0.80, 0.53),
            # "Al": 0.02,
            "Si": (-1.14, -0.57, 0.31),
            "P": (-0.95, -0.17, 0.49),
            "S": (-0.88, -0.09, 0.29),
            "Cl": (-1.24, -0.31, 0.61),
            # "Ar": 1.0,
            # "K": 0.3,
            # "Ca": 0.003,
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


        def __init__(self, f_star=0.5):

            """
            Args:
                f_star (float)
                    Parameter             
            """

            # Dx = 10**(BX +AX (F∗−zX ))
            # This Dx factor then multiplies the ref- erence abundance to produce the post-depletion gas- phase abundances.

            self.depletion = {}

            for element, parameters  in self.parameters.items():
                # unpack parameters
                AX, BX, zX = parameters
                # calculate depletion
                self.depletion[element] = 10**(BX+AX*(f_star-zX))



    class Gutkin2016:

        """
        Depletion pattern created for Synthesizer 2024.

        Gutkin+2016:
            https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/abstract

        Note: in previous version we adjusted N ("N": 0.89) following:
        Dopita+2013:
            https://ui.adsabs.harvard.edu/abs/2013ApJS..208...10D/abstract
        Dopita+2006:
            https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract
        """

        # This is the inverse depletion
        inverse_depletion = {
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

            """
            Args:
                scale (float)
                    Scale factor for the depletion.
            """
            self.depletion = {element: scale*(1-inverse_depletion_)
                              for element, inverse_depletion_ in
                              self.inverse_depletion.items()}
            





class Abundances(ElementDefinitions):

    """ A class for calculating elemental abundances including various
    scaling and depletion on to dust

    """

    def __init__(
        self,
        metallicity=SolarAbundances.Asplund2009.metallicity,
        alpha=0.0,
        abundances=False,
        solar=SolarAbundances.Asplund2009,
        depletion=None,
        depletion_model=None,
        depletion_scale=None,
        dust_to_metal_ratio=None,
    ):
        """
        Initialise an abundance pattern

        Args:
            metallicity (float)
                Mass fraction in metals, default is Solar metallicity.
            alpha (float)
                Enhancement of the alpha elements relative to the solar
                abundance pattern.
            abundances (dict, float/str)
                A dictionary containing the abundances for specific elements or
                functions to calculate them for the specified metallicity.
            solar (object)
                Solar abundance pattern object.
            depletion (dict, float)
                The depletion pattern to use.
            depletion_model (object)
                The depletion model object.
            depletion_scale (float)
                The depletion scale factor. Sometimes this is linear, but for 
                some models (e.g. Jenkins (2009)) it's more complex.
            dust_to_metal_ratio (float)
                the fraction of metals in dust.



        """

        # save all parameters to object
        self.metallicity = metallicity  # mass fraction in metals
        self.alpha = alpha
        
        self.solar = solar

        # depletion on to dust
        self.depletion = depletion
        self.depletion_model = depletion_model
        self.depletion_scale = depletion_scale
        self.dust_to_metal_ratio = dust_to_metal_ratio


        # set depletions to be zero initially
        # self.depletion = {element: 0.0 for element in self.all_elements}

        # Set helium mass fraction following Bressan et al. (2012)
        # 10.1111/j.1365-2966.2012.21948.x
        # https://ui.adsabs.harvard.edu/abs/2012MNRAS.427..127B/abstract
        self.helium_mass_fraction = 0.2485 + 1.7756 * self.metallicity

        # Define mass fraction in hydrogen
        self.hydrogen_mass_fraction = (
            1.0 - self.helium_mass_fraction - self.metallicity
        )

        # logathrimic total abundance of element relative to H
        total = {}

        # hydrogen is by definition 0.0
        total["H"] = 0.0
        total["He"] = np.log10(
            self.helium_mass_fraction
            / self.hydrogen_mass_fraction
            / self.A["He"]
        )

        # Scale elemental abundances from solar abundances based on given
        # metallicity
        for e in self.metals:
            total[e] = self.solar.abundance[e] + np.log10(
                self.metallicity / self.solar.metallicity
            )

        # Scale alpha-element abundances from solar abundances
        for e in self.alpha_elements:
            total[e] += alpha

        # Set holding elements that don't need to be rescaled.
        unscaled_metals = set([])

        # If abundances argument is provided go ahead and set the abundances.
        if abundances:
            # loop over each element in the dictionary
            for element, value in abundances.items():
                # Setting alpha, nitrogen_abundance, or carbon_abundance will
                # result in the metallicity no longer being correct. To account
                # for this we need to rescale the abundances to recover the
                # correct metallicity. However, we don't want to rescale the
                # things we've changed. For this reason, here we record the
                # elements which have changed. See below for the rescaling.
                unscaled_metals.add(element)

                # if value is a float simply set the abundance to this value.
                if isinstance(value, float):
                    total[element] = value

                # if value is a str use this to call the specific function to
                # calculate the abundance from the metallicity.
                elif isinstance(value, str):
                    # get the class holding functions for this element
                    study_functions = getattr(ScalingFunctions, value)

                    # get the specific function request by value
                    scaling_function = getattr(study_functions, element)
                    total[element] = scaling_function(metallicity)

        # Set of the metals to be scaled, see above.
        scaled_metals = set(self.metals) - unscaled_metals

        # Calculate the mass in unscaled, scaled, and non-metals.
        mass_in_unscaled_metals = self.calculate_mass(list(unscaled_metals), a=total)
        mass_in_scaled_metals = self.calculate_mass(list(scaled_metals), a=total)
        mass_in_non_metals = self.calculate_mass(["H", "He"], a=total)

        # Now, calculate the scaling factor. The metallicity is:
        # metallicity = scaling*mass_in_scaled_metals + mass_in_unscaled_metals
        #  / (scaling*mass_in_scaled_metals + mass_in_non_metals +
        # mass_in_unscaled_metals)
        # and so (by rearranging) the scaling factor is:
        scaling = (
            mass_in_unscaled_metals
            - metallicity * mass_in_unscaled_metals
            - metallicity * mass_in_non_metals
        ) / (mass_in_scaled_metals * (metallicity - 1))

        # now apply this scaling
        for i in scaled_metals:
            total[i] += np.log10(scaling)

        # save as attribute
        self.total = total

        # If a depletion pattern or depletion_model is provided then calculate
        # the depletion.
        if (depletion is not False) or (depletion_model is not False):
            self.add_depletion(
                depletion=depletion,
                depletion_model=depletion_model,
                depletion_scale=depletion_scale)

    def add_depletion(self,
                      depletion=False,
                      depletion_model=False,
                      depletion_scale=False):

        """
        Function to add depletion.
        
        
        """

        # Add exception if both a depletion pattern and depletion_model is
        # provided.


        # If provided, calculate depletion pattern by calling the depletion
        # model with the depletion scale.
        if depletion_model:

            # If a depletion_scale is provided use this...
            if self.depletion_scale:
                depletion = depletion_model(depletion_scale).depletion
            # ... otherwise use the default.
            else:
                depletion = depletion_model().depletion

        # apply depletion pattern
        if depletion:

            # deplete the gas and dust
            self.gas = {}
            self.dust = {}
            for element in self.all_elements:

                # if an entry exists for the element apply depletion
                if element in depletion.keys():

                    self.gas[element] = (
                        self.total[element]
                        + np.log10(1.-depletion[element])
                        )

                    if depletion[element] == 0.0:
                        self.dust[element] = -np.inf
                    else:
                        self.dust[element] = (
                            self.total[element]
                            + np.log10(depletion[element])
                            )
                        
                # otherwise assume no depletion
                else:
                    depletion[element] = 0.0
                    self.gas[element] = self.total[element]
                    self.dust[element] = -np.inf
                    
            # calculate mass fraction in metals
            # NOTE: this should be identical to the metallicity.
            self.metal_mass_fraction = self.calculate_mass_fraction(
                self.metals)

            # calculate mass fraction in dust
            self.dust_mass_fraction = self.calculate_mass_fraction(
                self.metals,
                a=self.dust)

            # calculate dust-to-metal ratio and save as an attribute
            self.dust_to_metal_ratio = (self.dust_mass_fraction /
                                        self.metal_mass_fraction)

            # calculate integrated dust abundance
            # this is used by cloudy23 
            self.dust_abundance = self.calculate_integrated_abundance(
                self.metals,
                a=self.dust)

            # Associate parameters with object
            self.depletion = depletion
            self.depletion_scale = depletion_scale
            self.depletion_model = depletion_model


    def __getitem__(self, arg):
        """
        A method to return the logarithmic abundance for a particular element
        relative to H or relative solar.

        Arguments:
            arg (str)
                The element (e.g. "O") or an element, reference element pair
                (e.g. "[O/Fe]").

        Returns:
            (float)
                The abundance relevant to H or relative to Solar when a
                reference element is also provided.
        """

        # default case, just return log10(k/H)
        if arg in self.all_elements:
            return self.total[arg]

        # alternative case, return solar relative abundance [X/Y]
        elif arg[0] == "[":
            element, ref_element = arg[1:-1].split("/")
            return self.solar_relative_abundance(
                element, ref_element=ref_element
            )

    def __str__(self):
        """
        Method to print a basic summary of the Abundances object.

        Returns:
            summary (str)
                String containing summary information.
        """

        # Set up string for printing
        summary = ""

        # Add the content of the summary to the string to be printed
        summary += "-" * 20 + "\n"
        summary += "ABUNDANCE PATTERN SUMMARY\n"
        summary += f"X: {self.hydrogen_mass_fraction:.3f}\n"
        summary += f"Y: {self.helium_mass_fraction:.3f}\n"
        summary += f"Z: {self.metallicity:.3f}\n"
        summary += f"Z/Z_sol: {self.metallicity/self.solar.metallicity:.2g}\n"
        summary += f"alpha: {self.alpha:.3f}\n"
        summary += f"dust-to-metal ratio: {self.dust_to_metal_ratio}\n"
        # summary += (
        #     f"MAX dust-to-metal ratio: {self.max_dust_to_metal_ratio:.3f}\n"
        # )

        summary += "-" * 10 + "\n"
        summary += (
            "element log10(X/H)_total (log10(X/H)+12) [[X/H]]"
            " |depletion| log10(X/H)_gas log10(X/H)_dust \n"
        )
        for ele in self.all_elements:
            summary += (
                f"{self.name[ele]}: {self.total[ele]:.2f} "
                f"({self.total[ele]+12:.2f}) "
                f"[{self.total[ele]-self.solar.abundance[ele]:.2f}] ")

            if self.depletion:
                summary += (
                        f"|{self.depletion[ele]:.2f}| "
                        f"{self.gas[ele]:.2f} "
                        f"{self.dust[ele]:.2f}"
                )
            summary += "\n"
            
        summary += "-" * 20

        return summary

    def calculate_integrated_abundance(self, elements, a=None):
        """
        Method to get the integrated abundance for a collection of elements.

        Args:
            elements (list, str)
                A list of element names.
            a (dict)
                The component to use.

        Returns:
            integrated abundance (float)
                The mass in those elements. Normally this needs to be
                normalised to be useful.
        """

        # if the component is not provided, assume it's the total
        if not a:
            a = self.total

        return np.sum([10 ** (a[i]) for i in elements])

    def calculate_mass(self, elements, a=None):
        """
        Method to get the mass for a collection of elements.

        Args:
            elements (list, str)
                A list of element names.
            a (dict)
                The component to use.

        Returns:
            mass (float)
                The mass in those elements. Normally this needs to be
                normalised to be useful.
        """

        # if the component is not provided, assume it's the total
        if not a:
            a = self.total

        return np.sum([self.A[i] * 10 ** (a[i]) for i in elements])
    
    def calculate_mass_fraction(self, elements, a=None):
        """
        Method to get the mass fraction for a collection of elements.

        Args:
            elements (list, str)
                A list of element names.
            a (dict)
                The component to use.

        Returns:
            mass (float)
                The mass in those elements. Normally this needs to be
                normalised to be useful.
        """

        # calculate the total mass
        total_mass = self.calculate_mass(self.all_elements)

        return self.calculate_mass(elements, a=a)/total_mass


    def solar_relative_abundance(self, element, ref_element="H"):
        """
        A method to return an element's abundance relative to that in the Sun,
        i.e. [X/H] = log10(N_X/N_H) - log10(N_X/N_H)_sol

        Arguments:
            element (str)
                The element of interest.
            ref_element (str)
                The reference element.

        Returns:
            abundance (float)
                The logarithmic relative abundance of an element, relative to
                the sun.

        """
        return (self.total[element] - self.total[ref_element]) - (
            self.solar.abundance[element] - self.solar.abundance[ref_element]
        )

    def get_max_dust_to_metal_ratio(self):
        """
        Method to calculate the maximum dust to metal ratio possible.

        Returns
            (float)
                The maximum dust to metal ratio possible for this depletion and
                abundance pattern.

        """

        dust = 0.0  # mass fraction in dust
        for element in self.metals:
            dust += (
                10 ** self.total[element]
                * self.A[element]
                * (1.0 - self.depletion_pattern.depletion[element])
            )

        return dust / self.metallicity




def plot_abundance_pattern(a, show=False, ylim=None, components=["total"]):
    """
    Funtion to plot a single abundance pattern, but possibly including all
    components.

    Args:
        a (abundances.Abundance)
            Abundance pattern object.
        components (list, str)
            List of components to plot. By default only plot "total".
        show (Bool)
            Toggle whether to show the plot.
        ylim (list/tuple, float)
            Limits of y-axis.
    """

    fig = plt.figure(figsize=(7.0, 4.0))

    left = 0.15
    height = 0.75
    bottom = 0.2
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    colors = cmr.take_cmap_colors("cmr.bubblegum", len(a.all_elements))

    for line, ls, ms in zip(
        components, ["-", "--", "-.", ":"], ["o", "s", "D", "d", "^"]
    ):
        i_ = range(len(a.all_elements))
        a_ = []

        for i, (e, c) in enumerate(zip(a.all_elements, colors)):
            value = getattr(a, line)[e]
            ax.scatter(i, value, color=c, s=40, zorder=2, marker=ms)
            a_.append(value)

        ax.plot(i_, a_, lw=2, ls=ls, c="0.5", label=rf"$\rm {line}$", zorder=1)

    for i, (e, c) in enumerate(zip(a.all_elements, colors)):
        ax.axvline(i, alpha=0.05, lw=1, c="k", zorder=0)

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([-12.0, 0.1])

    ax.legend()
    ax.set_xticks(
        range(len(a.all_elements)), a.name, rotation=90, fontsize=6.0
    )

    ax.set_ylabel(r"$\rm log_{10}(X/H)$")

    if show:
        plt.show()

    return fig, ax


def plot_multiple_abundance_patterns(
    abundance_patterns,
    labels=None,
    show=False,
    ylim=None,
):
    """
    Function to plot multiple abundance patterns.

    Args:
        a (abundances.Abundance)
            Abundance pattern object.
        components (list, str)
            List of components to plot. By default only plot "total".
        show (Bool)
            Toggle whether to show the plot.
        ylim (list/tuple, float)
            Limits of y-axis.
    """

    fig = plt.figure(figsize=(7.0, 4.0))

    left = 0.15
    height = 0.75
    bottom = 0.2
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    a = abundance_patterns[0]

    colors = cmr.take_cmap_colors("cmr.bubblegum", len(a.all_elements))

    if not labels:
        labels = range(len(abundance_patterns))

    for a, label, ls, ms in zip(
        abundance_patterns,
        labels,
        ["-", "--", "-.", ":"],
        ["o", "s", "D", "d", "^"],
    ):
        i_ = range(len(a.all_elements))
        a_ = []

        for i, (e, c) in enumerate(zip(a.all_elements, colors)):
            ax.scatter(i, a.total[e], color=c, s=40, zorder=2, marker=ms)
            a_.append(a.total[e])

        ax.plot(
            i_, a_, lw=2, ls=ls, c="0.5", label=rf"$\rm {label}$", zorder=1
        )

    for i, (e, c) in enumerate(zip(a.all_elements, colors)):
        ax.axvline(i, alpha=0.05, lw=1, c="k", zorder=0)

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([-12.0, 0.1])

    ax.legend()
    ax.set_xticks(
        range(len(a.all_elements)), a.name, rotation=90, fontsize=6.0
    )

    ax.set_ylabel(r"$\rm log_{10}(X/H)$")

    if show:
        plt.show()

    return fig, ax
