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

import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt
import synthesizer.exceptions as exceptions
from synthesizer.abundances import (
    solar_abundance_patterns,
    depletion_models,
    abundance_scalings,
)


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

    all_elements = non_metals + metals

    # the alpha process elements
    alpha_elements = [
        "O",
        "Ne",
        "Mg",
        "Si",
        "S",
        "Ar",
        "Ca",
        "Ti",
    ]

    # Name
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


class Abundances(ElementDefinitions):

    """
    A class for calculating elemental abundances including various
    scaling and depletion on to dust.

    Attributes:
        metallicity (float)
            Mass fraction in metals, default is Solar metallicity.
            Optional initialisation argument. Defaults to Solar metallicity
            assuming Asplund (2009).
        alpha (float)
            Enhancement of the alpha elements relative to the solar
            abundance pattern. Optional initialisation argument. Defaults to
            0.0 (no alpha-enhancement).
        abundances (dict, float/str)
            A dictionary containing the abundances for specific elements or
            functions to calculate them for the specified metallicity. Optional
            initialisation argument. Defaults to None.
        solar (object)
            Solar abundance pattern. Optional initialisation argument.
            Defaults to Asplund (2009) pattern.
        depletion (dict, float)
            The depletion pattern to use. Should not be provided with
            depletion_model. Optional initialisation argument. Defaults to
            None.
        depletion_model (object)
            The depletion model object. Should not be provided with
            depletion. Optional initialisation argument. Defaults to None.
        depletion_scale (float)
            The depletion scale factor. Sometimes this is linear, but for
            some models (e.g. Jenkins (2009)) it's more complex. Optional
            initialisation argument. Defaults to None.
        helium_mass_fraction (float)
            The helium mass fraction (more commonly denoted as "Y").
        hydrogen_mass_fraction (float)
            The hydrogen mass fraction (more commonly denoted as "X").
        total (dict, float)
            The total logarithmic abundance of each element.
        gas (dict, float)
            The logarithmic abundance of each element in the depleted gas
            phase.
        dust (dict, float)
            The logarithmic abundance of each element in the dust phase.
        metal_mass_fraction (float)
            Mass fraction in metals. Since this should be metallicity it is
            redundant but serves as a useful test.
        dust_mass_fraction (float)
            Mass fraction in metals.
        dust_to_metal_ratio (float)
            Dust-to-metal ratio.
    """

    def __init__(
        self,
        metallicity=None,
        alpha=0.0,
        abundances=None,
        solar=solar_abundance_patterns.Asplund2009,
        depletion=None,
        depletion_model=None,
        depletion_scale=None,
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
            solar (class or str)
                Solar abundance pattern object or str defining the class.
            depletion (dict, float)
                The depletion pattern to use. Should not be provided with
                depletion_model.
            depletion_model (class or str)
                The depletion model class or string defining the class.
                Should not be provided with depletion.
            depletion_scale (float)
                The depletion scale factor. Sometimes this is linear, but for
                some models (e.g. Jenkins (2009)) it's more complex.

        """

        # save all arguments to object
        self.metallicity = metallicity  # mass fraction in metals
        self.alpha = alpha
        self.solar = solar
        # depletion on to dust
        self.depletion = depletion
        self.depletion_model = depletion_model
        self.depletion_scale = depletion_scale

        # If depletion model is provided as a string use this to extract the
        # class.
        if isinstance(solar, str):
            if solar in solar_abundance_patterns.available_patterns:
                self.solar = getattr(solar_abundance_patterns, solar)
            else:
                raise exceptions.UnrecognisedOption(
                    """Solar abundance pattern
                not recognised!"""
                )

        # If a metallicity is not provided use the metallicity assumed by the
        # Solar abundance pattern.
        if self.metallicity is None:
            self.metallicity = self.solar.metallicity

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
        if alpha != 0.0:
            # unscaled_metals = self.alpha_elements
            for e in self.alpha_elements:
                total[e] += alpha

        # Set holding elements that don't need to be rescaled.
        unscaled_metals = set([])

        # If abundances argument is provided go ahead and set the abundances.
        if abundances is not None:
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
                    study_functions = getattr(abundance_scalings, value)

                    # get the specific function request by value
                    scaling_function = getattr(study_functions, element)
                    total[element] = scaling_function(metallicity)

        # Set of the metals to be scaled, see above.
        scaled_metals = set(self.metals) - unscaled_metals

        # Calculate the mass in unscaled, scaled, and non-metals.
        mass_in_unscaled_metals = self.calculate_mass(
            list(unscaled_metals), a=total
        )
        mass_in_scaled_metals = self.calculate_mass(
            list(scaled_metals), a=total
        )
        mass_in_non_metals = self.calculate_mass(["H", "He"], a=total)

        # Now, calculate the scaling factor. The metallicity is:
        # metallicity = scaling*mass_in_scaled_metals + mass_in_unscaled_metals
        #  / (scaling*mass_in_scaled_metals + mass_in_non_metals +
        # mass_in_unscaled_metals)
        # and so (by rearranging) the scaling factor is:
        scaling = (
            mass_in_unscaled_metals
            - self.metallicity * mass_in_unscaled_metals
            - self.metallicity * mass_in_non_metals
        ) / (mass_in_scaled_metals * (self.metallicity - 1))

        # now apply this scaling
        for i in scaled_metals:
            total[i] += np.log10(scaling)

        # save as attribute
        self.total = total

        # If a depletion pattern or depletion_model is provided then calculate
        # the depletion.
        if (depletion is not None) or (depletion_model is not None):
            self.add_depletion(
                depletion=depletion,
                depletion_model=depletion_model,
                depletion_scale=depletion_scale,
            )
        else:
            self.gas = self.total
            self.depletion = {element: 1.0 for element in self.all_elements}
            self.dust = {element: -np.inf for element in self.all_elements}
            self.metal_mass_fraction = self.metallicity
            self.dust_mass_fraction = 0.0
            self.dust_to_metal_ratio = 0.0

    def add_depletion(
        self, depletion=None, depletion_model=None, depletion_scale=None
    ):
        """
        Method to add depletion using a provided depletion pattern or model.
        This method creates the following attributes:
            gas (dict, float)
                The logarithmic abundances of the gas, including depletion.
            dust (dict, float)
                The logarithmic abundances of the dust. Set to -np.inf is no
                contribution.
            metal_mass_fraction (float)
                Mass fraction in metals. Since this should be metallicity it is
                redundant but serves as a useful test.
            dust_mass_fraction (float)
                Mass fraction in metals.
            dust_to_metal_ratio (float)
                Dust-to-metal ratio.

        Args:
            depletion (dict, float)
                The depletion pattern to use. Should not be provided with
                depletion_model.
            depletion_model (object)
                The depletion model object. Should not be provided with
                depletion.
            depletion_scale (float)
                The depletion scale factor. Sometimes this is linear, but for
                some models (e.g. Jenkins (2009)) it's more complex.
        """

        # If depletion model is provided as a string use this to extract the
        # class.
        if isinstance(depletion_model, str):
            if depletion_model in depletion_models.available_patterns:
                depletion_model = getattr(depletion_models, depletion_model)
            else:
                raise exceptions.UnrecognisedOption(
                    """Depletion model not
                recognised!"""
                )

        # Raise exception if both a depletion pattern and depletion_model is
        # provided.
        if (depletion is not None) and (depletion_model is not None):
            raise exceptions.InconsistentParameter(
                "Can not provide by a depletion pattern and a depletion model"
            )

        # Raise exception if a depletion_scale is provided by not a
        # depletion_model.
        if (depletion_scale is not None) and (depletion_model is None):
            raise exceptions.InconsistentParameter(
                """If a depletion scale is provided then a depletion model must
                also be provided"""
            )

        # If provided, calculate depletion pattern by calling the depletion
        # model with the depletion scale.
        if depletion_model:
            # If a depletion_scale is provided use this...
            if self.depletion_scale is not None:
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
                    # depletion factors >1.0 are unphysical so cap at 1.0
                    if depletion[element] > 1.0:
                        depletion[element] = 1.0

                    self.gas[element] = np.log10(
                        10 ** self.total[element] * depletion[element]
                    )

                    if depletion[element] == 1.0:
                        self.dust[element] = -np.inf
                    else:
                        self.dust[element] = np.log10(
                            10 ** self.total[element]
                            * (1 - depletion[element])
                        )

                # otherwise assume no depletion
                else:
                    depletion[element] = 1.0
                    self.gas[element] = self.total[element]
                    self.dust[element] = -np.inf

            # calculate mass fraction in metals
            # NOTE: this should be identical to the metallicity.
            self.metal_mass_fraction = self.calculate_mass_fraction(
                self.metals
            )

            # calculate mass fraction in dust
            self.dust_mass_fraction = self.calculate_mass_fraction(
                self.metals, a=self.dust
            )

            # calculate dust-to-metal ratio and save as an attribute
            self.dust_to_metal_ratio = (
                self.dust_mass_fraction / self.metal_mass_fraction
            )

            # calculate integrated dust abundance
            # this is used by cloudy23
            self.dust_abundance = self.calculate_integrated_abundance(
                self.metals, a=self.dust
            )

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
        summary += f"dust mass fraction: {self.dust_mass_fraction}\n"
        summary += f"dust-to-metal ratio: {self.dust_to_metal_ratio}\n"
        summary += "-" * 10 + "\n"

        column_width = 16
        column_format = " ".join(f"{{{i}:<{column_width}}}" for i in range(7))

        column_names = (
            "Element",
            "log10(X/H)",
            "log10(X/H)+12",
            "[X/H]",
            "depletion",
            "log10(X/H)_gas",
            "log10(X/H)_dust",
        )
        summary += column_format.format(*column_names) + "\n"

        for ele in self.all_elements:
            quantities = (
                f"{self.name[ele]}",
                f"{self.total[ele]:.2f}",
                f"{self.total[ele]+12:.2f}",
                f"{self.total[ele]-self.solar.abundance[ele]:.2f}",
                f"{self.depletion[ele]:.2f}",
                f"{self.gas[ele]:.2f}",
                f"{self.dust[ele]:.2f}",
            )
            summary += column_format.format(*quantities) + "\n"

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

        return self.calculate_mass(elements, a=a) / total_mass

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
