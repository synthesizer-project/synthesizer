# This script is a modified version of
# https://github.com/stephenmwilkins/SPS_tools/blob/master/SPS_tools/cloudy/abundances.py

# A note on notation


# [X/H] = log10(N_X/N_H) - log10(N_X/N_H)_sol

# [alpha/Fe] = sum()


from copy import deepcopy
import numpy as np
import cmasher as cmr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Elements:

    metals = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
              'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
              'Cu', 'Zn']

    all_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
                         'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                         'Co', 'Ni', 'Cu', 'Zn']

    alpha_elements = ['O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti']  # the alpha process elements

    name = {}
    name['H'] = 'Hydrogen'
    name['He'] = 'Helium'
    name['Li'] = 'Lithium'
    name['Be'] = 'Beryllium'
    name['B'] = 'Boron'
    name['C'] = 'Carbon'
    name['N'] = 'Nitrogen'
    name['O'] = 'Oxygen'
    name['F'] = 'Fluorine'
    name['Ne'] = 'Neon'
    name['Na'] = 'Sodium'
    name['Mg'] = 'Magnesium'
    name['Al'] = 'Aluminium'
    name['Si'] = 'Silicon'
    name['P'] = 'Phosphorus'
    name['S'] = 'Sulphur'
    name['Cl'] = 'Chlorine'
    name['Ar'] = 'Argon'
    name['K'] = 'Potassium'
    name['Ca'] = 'Calcium'
    name['Sc'] = 'Scandium'
    name['Ti'] = 'Titanium'
    name['V'] = 'Vanadium'
    name['Cr'] = 'Chromium'
    name['Mn'] = 'Manganese'
    name['Fe'] = 'Iron'
    name['Co'] = 'Cobalt'
    name['Ni'] = 'Nickel'
    name['Cu'] = 'Copper'
    name['Zn'] = 'Zinc'

    # mass of elements in amus
    A = {}
    A['H'] = 1.008
    A['He'] = 4.003
    A['Li'] = 6.940
    A['Be'] = 9.012
    A['B'] = 10.81
    A['C'] = 12.011
    A['N'] = 14.007
    A['O'] = 15.999
    A['F'] = 18.998
    A['Ne'] = 20.180
    A['Na'] = 22.990
    A['Mg'] = 24.305
    A['Al'] = 26.982
    A['Si'] = 28.085
    A['P'] = 30.973
    A['S'] = 32.06
    A['Cl'] = 35.45
    A['Ar'] = 39.948
    A['K'] = 39.0983
    A['Ca'] = 40.078
    A['Sc'] = 44.955
    A['Ti'] = 47.867
    A['V'] = 50.9415
    A['Cr'] = 51.9961
    A['Mn'] = 54.938
    A['Fe'] = 55.845
    A['Co'] = 58.933
    A['Ni'] = 58.693
    A['Cu'] = 63.546
    A['Zn'] = 65.38

    # Elemental abundances
    # Asplund (2009) Solar, same as GASS (Grevesse et al. (2010)) in cloudy
    # https://ui.adsabs.harvard.edu/abs/2009ARA%26A..47..481A/abstract

    # Asplund (2009) Solar - HOWEVER, running metallicity() on the solar abundances below yields 0.0135
    Z_sol = 0.0134

    sol = {}
    # These are log10(N_element/N_H) ratios
    sol['H'] = 0.0
    sol['He'] = -1.07
    sol['Li'] = -10.95
    sol['Be'] = -10.62
    sol['B'] = -9.3
    sol['C'] = -3.57
    sol['N'] = -4.17
    sol['O'] = -3.31
    sol['F'] = -7.44
    sol['Ne'] = -4.07
    sol['Na'] = -5.07
    sol['Mg'] = -4.40
    sol['Al'] = -5.55
    sol['Si'] = -4.49
    sol['P'] = -6.59
    sol['S'] = -4.88
    sol['Cl'] = -6.5
    sol['Ar'] = -5.60
    sol['K'] = -6.97
    sol['Ca'] = -5.66
    sol['Sc'] = -8.85
    sol['Ti'] = -7.05
    sol['V'] = -8.07
    sol['Cr'] = -6.36
    sol['Mn'] = -6.57
    sol['Fe'] = -4.50
    sol['Co'] = -7.01
    sol['Ni'] = -5.78
    sol['Cu'] = -7.81
    sol['Zn'] = -7.44

    # ---------------- default Depletion
    # --- ADOPTED VALUES
    # Gutkin+2016: https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/abstract
    # Dopita+2013: https://ui.adsabs.harvard.edu/abs/2013ApJS..208...10D/abstract
    # Dopita+2006: https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract

    default_depletion = {}
    # Depletion of 1 -> no depletion, while 0 -> fully depleted
    # Noble gases aren't depleted (eventhough there is some eveidence
    # for Argon depletion -> https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.2310G/abstract)
    default_depletion['H'] = 1.0
    default_depletion['He'] = 1.0
    default_depletion['Li'] = 0.16
    default_depletion['Be'] = 0.6
    default_depletion['B'] = 0.13
    default_depletion['C'] = 0.5
    # <----- replaced by Dopita+2013 value, Gutkin+2016 assumes no depletion
    default_depletion['N'] = 0.89
    default_depletion['O'] = 0.7
    default_depletion['F'] = 0.3
    default_depletion['Ne'] = 1.0
    default_depletion['Na'] = 0.25
    default_depletion['Mg'] = 0.2
    default_depletion['Al'] = 0.02
    default_depletion['Si'] = 0.1
    default_depletion['P'] = 0.25
    default_depletion['S'] = 1.0
    default_depletion['Cl'] = 0.5
    default_depletion['Ar'] = 1.0
    default_depletion['K'] = 0.3
    default_depletion['Ca'] = 0.003
    default_depletion['Sc'] = 0.005
    default_depletion['Ti'] = 0.008
    default_depletion['V'] = 0.006
    default_depletion['Cr'] = 0.006
    default_depletion['Mn'] = 0.05
    default_depletion['Fe'] = 0.01
    default_depletion['Co'] = 0.01
    default_depletion['Ni'] = 0.04
    default_depletion['Cu'] = 0.1
    default_depletion['Zn'] = 0.25


class Abundances(Elements):

    def __init__(self, Z=Elements.Z_sol, alpha=0.0, CO=0.0, d2m=False, scaling='Dopita+2013'):
        """
        This function returns the elemental abundances after removing the depleted amount

        :param Z: float, the total metallicity (includes depletion as well)
        :param alpha: float, log10(alpha-enhancement factor)
        :param CO: float, the abundance of Carbon in CO (not implemented)
        :param d2m: float(?), dust to metal ratio
        :param scaling: string

        :return: dictionary with different elemental abundances as log10(element/H)
        :rtype: float
        """

        self.Z = Z
        self.alpha = alpha
        self.CO = CO
        self.d2m = d2m
        self.scaling = scaling

        # set all depletions to zero initially
        self.depletion = {element: 0.0 for element in self.all_elements}

        # logathrimic abundance of element relative to H
        a = {}

        # hydrogen is by definition 0.0
        a['H'] = 0.0

        # New scaling applied to match the He abundance at Z_sol
        a['He'] = np.log10(0.0737 + 0.0114*(Z / self.Z_sol))

        # Scale elemental abundances from solar abundances based on given metallicity
        for e in self.metals:
            a[e] = self.sol[e] + np.log10(Z / self.Z_sol)

        # Additionally scale alpha-element abundances from solar abundances
        for e in self.alpha_elements:
            a[e] += alpha

        # apply an additional scaling
        if scaling == 'Dopita+2013':
            # Actually from Dopita+2006
            # Scaling applied to match with our solar metallicity, this done by
            # solving the equation to get the adopted solar metallicity
            Z_sol_Dopita = 0.016
            C_fac = self.Z_sol / 1.01973
            N_fac = self.Z_sol / 1.06774

            a['C'] = np.log10(6e-5 * Z / C_fac + 2e-4 * (Z / C_fac)**2)
            a['N'] = np.log10(1.1e-5 * Z / N_fac + 4.9e-5 * (Z / N_fac)**2)

        elif scaling == 'Wilkins+2020':
            a['N'] = -4.47 + np.log10(Z / self.Z_sol + (Z / self.Z_sol)**2)

        # rescale abundances to recover correct Z
        cor = np.log10(Z / self.get_metallicity(a))

        for i in self.metals:
            a[i] += cor

        # calculate max_d2m
        self.a = a

        self.max_d2m = self.get_max_d2m()

        self.a_nodep = deepcopy(a)

        if d2m:

            # get scaled depletion values, i.e. the fraction of each element which is depleted on to dust
            self.get_depletions()

            for element in self.metals:
                self.a[element] += np.log10(self.depletion[element])

    def __getitem__(self, element):
        """
        Function to return the logarithmic abundance relative to H


        Returns
        -------
        float
            logarthmic abundance.
        """

        return self.a[element]

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
        pstr += "-"*20 + "\n"
        pstr += f"ABUNDANCE PATTERN SUMMARY\n"
        pstr += f"Z: {self.Z}\n"
        pstr += f"Z/Z_sol: {self.Z/self.Z_sol:.2g}\n"
        pstr += f"alpha: {self.alpha}\n"
        pstr += f"CO: {self.CO}\n"
        pstr += f"dust-to-metal ratio: {self.d2m}\n"
        pstr += "-"*10 + "\n"
        pstr += "element log10(X/H) (log10(X/H)+12) [depletion]\n"
        for ele in self.all_elements:
            pstr += f"{self.name[ele]}: {self.a[ele]:.2f} ({self.a[ele]+12:.2f}) [{self.depletion[ele]:.2f}]\n"
        pstr += "-"*20

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
            a = self.a

        return np.sum([self.A[i]*10**(a[i]) for i in self.metals]) /\
            np.sum([self.A[i]*10**(a[i])
                    for i in self.all_elements])

    def get_depletions(self):
        """
        This function returns the depletion after scaling using the solar abundances and
        depletion patterns from the dust-to-metal ratio.

        """
        for element in self.all_elements:
            # calculate depletion factor by rescaling the
            self.depletion[element] = 1-((self.d2m/self.max_d2m) *
                                         (1.-self.default_depletion[element]))

        return self.depletion

    def solar_relative_abundance(self, e, ref_element='H'):
        """
        This function returns an element's abundance relative to that in the Sun, i.e. [X/H] = log10(N_X/N_H) - log10(N_X/N_H)_sol
        :param a: the element of interest
        :param a: a dictionary with the absolute elemental abundances
        """

        return (self.a[e]-self.a[ref_element]) - (self.sol[e]-self.sol[ref_element])

    def get_max_d2m(self):
        """
        This function determine the maximum dust-to-metal ratio
        """

        dust = 0.0  # mass fraction in dust
        for element in self.metals:
            dust += 10**self.a[element]*self.A[element]*(1.0-self.default_depletion[element])

        return dust/self.Z

    def get_d2m(self):
        """
        This function measures the dust-to-metal ratio for the calculated depletion
        """

        dust = 0.0  # mass fraction in dust
        for element in self.metals:
            dust += (10**(self.a_nodep[element])-10**(self.a[element]))*self.A[element]

        return dust/self.Z


# eventually move this to dedicated plotting module

def plot_abundance_pattern(abundance_patterns, labels=None, show=False, ylim=None):
    """ Plot multiple abundance patterns """

    fig = plt.figure(figsize=(7., 4.))

    left = 0.15
    height = 0.75
    bottom = 0.2
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    a = abundance_patterns[0]

    colors = cmr.take_cmap_colors('cmr.bubblegum', len(a.all_elements))

    if not labels:
        labels = range(len(abundance_patterns))

    for a, label, ls, ms in zip(abundance_patterns, labels, ['-', '--', '-.', ':'], ['o', 's', 'D', 'd', '^']):

        i_ = range(len(a.all_elements))
        a_ = []

        for i, (e, c) in enumerate(zip(a.all_elements, colors)):
            ax.scatter(i, a.a[e], color=c, s=40, zorder=2, marker=ms)
            a_.append(a.a[e])

        ax.plot(i_, a_, lw=2, ls=ls, c='0.5', label=rf'$\rm {label}$', zorder=1)

    for i, (e, c) in enumerate(zip(a.all_elements, colors)):
        ax.axvline(i, alpha=0.05, lw=1, c='k', zorder=0)

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([-12., 0.1])

    ax.legend()
    ax.set_xticks(range(len(a.all_elements)), a.name, rotation=90, fontsize=6.)

    ax.set_ylabel(r'$\rm X/H$')

    if show:
        plt.show()

    return fig, ax
