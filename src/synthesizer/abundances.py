"""
This script is a modified version of
https://github.com/stephenmwilkins/SPS_tools/blob/master/SPS_tools/cloudy/abundances.py

A note on notation:

[X/H] = log10(N_X/N_H) - log10(N_X/N_H)_sol

[alpha/Fe] = sum()
"""

from copy import deepcopy
import numpy as np
import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .exceptions import InconsistentParameter


class Elements:
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

    all_elements = [
        "H",
        "He",
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

    # Elemental abundances
    # Asplund (2009) Solar, same as GASS (Grevesse et al. (2010)) in cloudy
    # https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract

    # Asplund (2009) Solar - HOWEVER, running metallicity() on the solar abundances below yields 0.0135
    Z_sol = 0.0134

    sol = {}
    # These are log10(N_element/N_H) ratios
    sol["H"] = 0.0
    sol["He"] = -1.07
    sol["Li"] = -10.95
    sol["Be"] = -10.62
    sol["B"] = -9.3
    sol["C"] = -3.57
    sol["N"] = -4.17
    sol["O"] = -3.31
    sol["F"] = -7.44
    sol["Ne"] = -4.07
    sol["Na"] = -5.07
    sol["Mg"] = -4.40
    sol["Al"] = -5.55
    sol["Si"] = -4.49
    sol["P"] = -6.59
    sol["S"] = -4.88
    sol["Cl"] = -6.5
    sol["Ar"] = -5.60
    sol["K"] = -6.97
    sol["Ca"] = -5.66
    sol["Sc"] = -8.85
    sol["Ti"] = -7.05
    sol["V"] = -8.07
    sol["Cr"] = -6.36
    sol["Mn"] = -6.57
    sol["Fe"] = -4.50
    sol["Co"] = -7.01
    sol["Ni"] = -5.78
    sol["Cu"] = -7.81
    sol["Zn"] = -7.44

    # ---------------- default Depletion
    # --- ADOPTED VALUES
    # Gutkin+2016: https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/abstract
    # Dopita+2013: https://ui.adsabs.harvard.edu/abs/2013ApJS..208...10D/abstract
    # Dopita+2006: https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract

    default_depletion = {}
    # Depletion of 1 -> no depletion, while 0 -> fully depleted
    # Noble gases aren't depleted (eventhough there is some eveidence
    # for Argon depletion -> https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.2310G/abstract)
    default_depletion["H"] = 1.0
    default_depletion["He"] = 1.0
    default_depletion["Li"] = 0.16
    default_depletion["Be"] = 0.6
    default_depletion["B"] = 0.13
    default_depletion["C"] = 0.5
    # <----- replaced by Dopita+2013 value, Gutkin+2016 assumes no depletion
    default_depletion["N"] = 0.89
    default_depletion["O"] = 0.7
    default_depletion["F"] = 0.3
    default_depletion["Ne"] = 1.0
    default_depletion["Na"] = 0.25
    default_depletion["Mg"] = 0.2
    default_depletion["Al"] = 0.02
    default_depletion["Si"] = 0.1
    default_depletion["P"] = 0.25
    default_depletion["S"] = 1.0
    default_depletion["Cl"] = 0.5
    default_depletion["Ar"] = 1.0
    default_depletion["K"] = 0.3
    default_depletion["Ca"] = 0.003
    default_depletion["Sc"] = 0.005
    default_depletion["Ti"] = 0.008
    default_depletion["V"] = 0.006
    default_depletion["Cr"] = 0.006
    default_depletion["Mn"] = 0.05
    default_depletion["Fe"] = 0.01
    default_depletion["Co"] = 0.01
    default_depletion["Ni"] = 0.04
    default_depletion["Cu"] = 0.1
    default_depletion["Zn"] = 0.25


class Abundances(Elements):
    def __init__(
        self, Z=Elements.Z_sol, alpha=0.0, C="Dopita2006", N="Dopita2006", dust_to_metal_ratio=False
    ):
        """
        A class for calcualting elemental abundances including various scaling and depletion on to dust

        Arguments
        ----------
        Z : float
            Mass fraction in metals, default is Solar metallicity.
        alpha: float
            Enhancement of the alpha elements
        C: float, str
            log10(C/O) ratio. A str is the name of the function f(Z) to use instead.
        N: float, str
            log10(N/O) ratio. A str is the name of the function f(Z) to use instead.
        dust_to_metal_ratio: float
            the fraction of metals in dust

        Returns
        -------
        """

        # save all parameters to object
        self.Z = Z  # mass fraction in metals
        self.alpha = alpha
        self.C = C
        self.N = N
        self.dust_to_metal_ratio = dust_to_metal_ratio

        # set depletions to be zero
        self.depletion = {element: 0.0 for element in self.all_elements}

        # Set helium abundance following Bressan et al. (2012)
        self.Y = 0.2485 + 1.7756 * self.Z

        # Define mass fraction in hydrogen
        self.X = 1.0 - self.Y - self.Z

        # logathrimic total abundance of element relative to H
        total = {}

        # hydrogen is by definition 0.0
        total["H"] = 0.0
        total["He"] = np.log10(self.Y / self.X / self.A["He"])

        # Scale elemental abundances from solar abundances based on given metallicity
        for e in self.metals:
            total[e] = self.sol[e] + np.log10(Z / self.Z_sol)

        # Scale alpha-element abundances from solar abundances
        for e in self.alpha_elements:
            total[e] += alpha

        # Rescale Nitrogen
        if isinstance(N, float):
            total["N"] += N
        elif isinstance(N, str):
            # if N == 'Feltre2016':
            #     """ apply scaling for secondary Nitrogen production using relation in Feltre16.
            #         this is itself based on Groves (2004).
            #         [N/H] = [O/H](10**1.6 + 10**(2.33+log10 [O/H])) """

            #     OH = (total['O']-total['H']) - (self.sol['O']-self.sol['H'])
            #     NH = OH*(10**1.6 + 10**(2.33+OH))
            #     total['N'] = NH + total['H'] + (self.sol['N']-self.sol['H'])
            #     print(total['N'])

            if N == "Dopita2006":
                """
                Scaling applied to match with our solar metallicity, this done by
                solving the equation to get the adopted solar metallicity.
                """
                Dopita_Z_sol = 0.016
                total["N"] = np.log10(
                    1.1e-5 * (Z / Dopita_Z_sol) + 4.9e-5 * (Z / Dopita_Z_sol) ** 2
                )

        # Rescale Carbon
        if isinstance(C, float):
            total["C"] += C
        elif isinstance(C, str):
            if C == "Dopita2006":
                """
                Scaling applied to match with our solar metallicity, this done by
                solving the equation to get the adopted solar metallicity.
                """
                Dopita_Z_sol = 0.016
                total["C"] = np.log10(
                    6e-5 * (Z / Dopita_Z_sol) + 2e-4 * (Z / Dopita_Z_sol) ** 2
                )

            else:
                print("ERROR: Supplied scaling is not known")

        # rescale abundances to recover correct Z

        correction = np.log10(Z / self.get_metallicity(total))

        for i in self.metals:
            total[i] += correction

        # copy total to be gas
        self.total = total
        self.gas = deepcopy(total)

        # calculate the maximum dust-to-metal ratio possible
        self.max_dust_to_metal_ratio = self.get_max_dust_to_metal_ratio()

        if dust_to_metal_ratio:
            # check that the dust to metal ratio is allowed
            if dust_to_metal_ratio <= self.max_dust_to_metal_ratio:
                # get scaled depletion values, i.e. the fraction of each element which is depleted on to dust
                self.get_depletions()

                # define dust abundances
                self.dust = {}

                # neither Hydrogen or Helium are depleted on to dust
                self.dust["H"] = -99
                self.dust["He"] = -99

                # deplete elements in the gas
                for element in self.metals:
                    self.gas[element] += np.log10(1 - self.depletion[element])

                    # calculate (X/H) contained in dust
                    if self.depletion[element] > 0.0:
                        self.dust[element] = np.log10(
                            self.depletion[element] * 10 ** self.total[element]
                        )
                    else:
                        self.dust[element] = -99

            else:
                # this doesn't work
                InconsistentParameter(
                    f"The dust-to-metal ratio (dust_to_metal_ratio) must be less than the maximum possible ratio ({self.max_dust_to_metal_ratio:.2f})"
                )

        else:
            # if not dust_to_metal_ratio ratio is provided set the dust to be None
            self.dust = {element: -99 for element in self.all_elements}
            self.depletion = {element: 0.0 for element in self.all_elements}

    def __getitem__(self, k):
        """
        Function to return the logarithmic abundance relative to H

        Returns
        -------
        float
            logarthmic abundance.
        """

        # default case, just return log10(k/H)
        if k in self.all_elements:
            return self.total[k]
        # return solar relative abundance [X/Y]
        elif k[0] == "[":
            element, ref_element = k[1:-1].split("/")
            return self.solar_relative_abundance(element, ref_element=ref_element)

    def __str__(self):
        """Function to print a basic summary of the Abundances object.

        Returns a string containing

        Returns
        -------
        str
            Summary string containing summary information.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 20 + "\n"
        pstr += f"ABUNDANCE PATTERN SUMMARY\n"
        pstr += f"X: {self.X:.3f}\n"
        pstr += f"Y: {self.Y:.3f}\n"
        pstr += f"Z: {self.Z:.3f}\n"
        pstr += f"Z/Z_sol: {self.Z/self.Z_sol:.2g}\n"
        pstr += f"alpha: {self.alpha:.3f}\n"
        pstr += f"C: {self.C} (scaling of C/H relative to Solar)\n"
        pstr += f"N: {self.N} (scaling of N/H relative to Solar)\n"
        pstr += f"dust-to-metal ratio: {self.dust_to_metal_ratio}\n"
        pstr += f"MAX dust-to-metal ratio: {self.max_dust_to_metal_ratio:.3f}\n"

        pstr += "-" * 10 + "\n"
        pstr += "element log10(X/H)_total (log10(X/H)+12) [depletion] log10(X/H)_gas log10(X/H)_dust \n"
        for ele in self.all_elements:
            pstr += f"{self.name[ele]}: {self.total[ele]:.2f} ({self.total[ele]+12:.2f}) [{self.depletion[ele]:.2f}] {self.gas[ele]:.2f} {self.dust[ele]:.2f}\n"
        pstr += "-" * 20

        return pstr

    def get_metallicity(self, a=None):
        """
        This function determines the mass fraction of the metals, or the metallicity


        TODO: rewrite this for improved clarity


        :param elements: a dictionary with the absolute elemental abundances

        :return: A single number
        :rtype: float
        """

        if not a:
            a = self.total

        return np.sum([self.A[i] * 10 ** (a[i]) for i in self.metals]) / np.sum(
            [self.A[i] * 10 ** (a[i]) for i in self.all_elements]
        )

    def get_depletions(self):
        """
        This function returns the depletion after scaling using the solar abundances and depletion patterns from the dust-to-metal ratio. This is the fraction of each element that is depleted on to dust.
        """
        for element in self.all_elements:
            self.depletion[element] = (self.dust_to_metal_ratio / self.max_dust_to_metal_ratio) * (
                1.0 - self.default_depletion[element]
            )

        return self.depletion

    def solar_relative_abundance(self, e, ref_element="H"):
        """
        This function returns an element's abundance relative to that in the Sun, i.e. [X/H] = log10(N_X/N_H) - log10(N_X/N_H)_sol
        :param a: the element of interest
        :param a: a dictionary with the absolute elemental abundances
        """

        return (self.total[e] - self.total[ref_element]) - (
            self.sol[e] - self.sol[ref_element]
        )

    def get_max_dust_to_metal_ratio(self):
        """
        This function determine the maximum dust-to-metal ratio
        """

        dust = 0.0  # mass fraction in dust
        for element in self.metals:
            dust += (
                10 ** self.total[element]
                * self.A[element]
                * (1.0 - self.default_depletion[element])
            )

        return dust / self.Z

    def get_dust_to_metal_ratio(self):
        """
        This function measures the dust-to-metal ratio for the calculated depletion
        """

        dust = 0.0  # mass fraction in dust
        for element in self.metals:
            dust += (10 ** (self.total[element]) - 10 ** (self.gas[element])) * self.A[
                element
            ]

        return dust / self.Z


# eventually move these to dedicated plotting module


def plot_abundance_pattern(a, show=False, ylim=None, lines=["total"]):
    """Plot single abundance patterns, but possibly including total, gas and dust"""

    fig = plt.figure(figsize=(7.0, 4.0))

    left = 0.15
    height = 0.75
    bottom = 0.2
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    colors = cmr.take_cmap_colors("cmr.bubblegum", len(a.all_elements))

    for line, ls, ms in zip(lines, ["-", "--", "-.", ":"], ["o", "s", "D", "d", "^"]):
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
    ax.set_xticks(range(len(a.all_elements)), a.name, rotation=90, fontsize=6.0)

    ax.set_ylabel(r"$\rm log_{10}(X/H)$")

    if show:
        plt.show()

    return fig, ax


def plot_multiple_abundance_patterns(
    abundance_patterns, labels=None, show=False, ylim=None
):
    """Plot multiple abundance patterns"""

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
        abundance_patterns, labels, ["-", "--", "-.", ":"], ["o", "s", "D", "d", "^"]
    ):
        i_ = range(len(a.all_elements))
        a_ = []

        for i, (e, c) in enumerate(zip(a.all_elements, colors)):
            ax.scatter(i, a.total[e], color=c, s=40, zorder=2, marker=ms)
            a_.append(a.total[e])

        ax.plot(i_, a_, lw=2, ls=ls, c="0.5", label=rf"$\rm {label}$", zorder=1)

    for i, (e, c) in enumerate(zip(a.all_elements, colors)):
        ax.axvline(i, alpha=0.05, lw=1, c="k", zorder=0)

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([-12.0, 0.1])

    ax.legend()
    ax.set_xticks(range(len(a.all_elements)), a.name, rotation=90, fontsize=6.0)

    ax.set_ylabel(r"$\rm log_{10}(X/H)$")

    if show:
        plt.show()

    return fig, ax
