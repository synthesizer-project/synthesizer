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



class Abundances:

    metals = ['Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si',
              'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni',
              'Cu', 'Zn']

    all_elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al',
                         'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe',
                         'Co', 'Ni', 'Cu', 'Zn']

    alpha_elements = ['O', 'Ne', 'Mg', 'Si', 'S', 'Ar', 'Ca', 'Ti'] # the alpha process elements

    # Define elements and their abundances
    name = {}
    name['H']   = 'Hydrogen'
    name['He']  = 'Helium'
    name['Li']  = 'Lithium'
    name['Be']  = 'Beryllium'
    name['B']   = 'Boron'
    name['C']   = 'Carbon'
    name['N']   = 'Nitrogen'
    name['O']   = 'Oxygen'
    name['F']   = 'Fluorine'
    name['Ne']  = 'Neon'
    name['Na']  = 'Sodium'
    name['Mg']  = 'Magnesium'
    name['Al']  = 'Aluminium'
    name['Si']  = 'Silicon'
    name['P']   = 'Phosphorus'
    name['S']   = 'Sulphur'
    name['Cl']  = 'Chlorine'
    name['Ar']  = 'Argon'
    name['K']   = 'Potassium'
    name['Ca']  = 'Calcium'
    name['Sc']  = 'Scandium'
    name['Ti']  = 'Titanium'
    name['V']   = 'Vanadium'
    name['Cr']  = 'Chromium'
    name['Mn']  = 'Manganese'
    name['Fe']  = 'Iron'
    name['Co']  = 'Cobalt'
    name['Ni']  = 'Nickel'
    name['Cu']  = 'Copper'
    name['Zn']  = 'Zinc'




    A = {}
    A['H']  = 1.008
    A['He'] = 4.003
    A['Li'] = 6.940
    A['Be'] = 9.012
    A['B']  = 10.81
    A['C']  = 12.011
    A['N']  = 14.007
    A['O']  = 15.999
    A['F']  = 18.998
    A['Ne'] = 20.180
    A['Na'] = 22.990
    A['Mg'] = 24.305
    A['Al'] = 26.982
    A['Si'] = 28.085
    A['P']  = 30.973
    A['S']  = 32.06
    A['Cl'] = 35.45
    A['Ar'] = 39.948
    A['K']  = 39.0983
    A['Ca'] = 40.078
    A['Sc'] = 44.955
    A['Ti'] = 47.867
    A['V']  = 50.9415
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

    Z_sol = 0.0134 # Asplund (2009) Solar - HOWEVER, running metallicity() on the solar abundances below yields 0.0135

    sol = {}
    #These are log10(N_element/N_H) ratios
    sol['H']    = 0.0
    sol['He']   = -1.07
    sol['Li']   = -10.95
    sol['Be']   = -10.62
    sol['B']    = -9.3
    sol['C']    = -3.57
    sol['N']    = -4.17
    sol['O']    = -3.31
    sol['F']    = -7.44
    sol['Ne']   = -4.07
    sol['Na']   = -5.07
    sol['Mg']   = -4.40
    sol['Al']   = -5.55
    sol['Si']   = -4.49
    sol['P']    = -6.59
    sol['S']    = -4.88
    sol['Cl']   = -6.5
    sol['Ar']   = -5.60
    sol['K']    = -6.97
    sol['Ca']   = -5.66
    sol['Sc']   = -8.85
    sol['Ti']   = -7.05
    sol['V']    = -8.07
    sol['Cr']   = -6.36
    sol['Mn']   = -6.57
    sol['Fe']   = -4.50
    sol['Co']   = -7.01
    sol['Ni']   = -5.78
    sol['Cu']   = -7.81
    sol['Zn']   = -7.44


    # ---------------- Depletion
    # --- ADOPTED VALUES
    # Gutkin+2016: https://ui.adsabs.harvard.edu/abs/2016MNRAS.462.1757G/abstract
    # Dopita+2013: https://ui.adsabs.harvard.edu/abs/2013ApJS..208...10D/abstract
    # Dopita+2006: https://ui.adsabs.harvard.edu/abs/2006ApJS..167..177D/abstract

    depsol = {}
    # Depletion of 1 -> no depletion, while 0 -> fully depleted
    # Noble gases aren't depleted (eventhough there is some eveidence
    # for Argon depletion -> https://ui.adsabs.harvard.edu/abs/2022MNRAS.512.2310G/abstract)
    depsol['H']     = 1.0
    depsol['He']    = 1.0
    depsol['Li']    = 0.16
    depsol['Be']    = 0.6
    depsol['B']     = 0.13
    depsol['C']     = 0.5
    depsol['N']     = 0.89 # <----- replaced by Dopita+2013 value, Gutkin+2016 assumes no depletion
    depsol['O']     = 0.7
    depsol['F']     = 0.3
    depsol['Ne']    = 1.0
    depsol['Na']    = 0.25
    depsol['Mg']    = 0.2
    depsol['Al']    = 0.02
    depsol['Si']    = 0.1
    depsol['P']     = 0.25
    depsol['S']     = 1.0
    depsol['Cl']    = 0.5
    depsol['Ar']    = 1.0
    depsol['K']     = 0.3
    depsol['Ca']    = 0.003
    depsol['Sc']    = 0.005
    depsol['Ti']    = 0.008
    depsol['V']     = 0.006
    depsol['Cr']    = 0.006
    depsol['Mn']    = 0.05
    depsol['Fe']    = 0.01
    depsol['Co']    = 0.01
    depsol['Ni']    = 0.04
    depsol['Cu']    = 0.1
    depsol['Zn']    = 0.25

    def metallicity(self, elements):
        """
        This function determines the mass fraction of the metals, or the metallicity

        :param elements: a dictionary with the absolute elemental abundances

        :return: A single number
        :rtype: float
        """
        # TODO: rewrite this for improved clarity

        return np.sum([self.A[i]*10**(elements[i]) for i in self.metals]) /\
                    np.sum([self.A[i]*10**(elements[i])\
                    for i in self.all_elements])



    def generate_abundances(self, Z, alpha=0.0, CO=0.0, d2m=False, scaling='Dopita+2013'):
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

        params = {'Z':Z, 'alpha':alpha, 'CO':CO, 'd2m':d2m, 'scaling': scaling}

        a = {}

        a['H'] = 0.0

        # New scaling applied to match the He abundance at Z_sol
        a['He'] = np.log10(0.0737 + 0.0114*(Z / self.Z_sol))

        for e in self.metals:
            # Scale elemental abundances from solar abundances based on given metallicity
            a[e] = self.sol[e] + np.log10(Z / self.Z_sol)

        for e in self.alpha_elements:
            # Scale alpha-element abundances from solar abundances based on given metallicity
            a[e] += alpha


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
        cor = np.log10(Z / self.metallicity(a))

        for i in self.metals: a[i] += cor


        if d2m:

            dep = self.depletions(d2m)

            a_nodep = deepcopy(a)

            for i in self.metals: a[i] += np.log10(dep[i])

            return AbundancePattern(params, a, a_nodep = a_nodep)

        else:

            return AbundancePattern(params, a)



    def dust_to_metal(self, a, dep):

        """
        This function returns the dust-to-metal ratio from the depleted amount of metals

        :param a: a dictionary with the absolute elemental abundances
        :param dep: a dictionary with non-depleted fraction of the metals

        :return: the dust-to-metal ratio
        :rtype: float
        """
        # TODO: rewrite this for improved clarity

        return np.sum([self.A[i] * (1. - dep[i])*10**a[i] for i in self.metals]) /\
                np.sum([self.A[i]*10**a[i] for i in self.metals])


    def depletions(self, d2m):

        """
        This function returns the depletion after scaling using the solar abundances and
        depletion patterns from the dust-to-metal ratio.

        :param d2m: float, dust-to-metal ratio

        :return: a dictionary of depletion patterns
        :rtype: float
        """

        dep = {}

        for i in self.metals:

            if self.depsol[i] != 1.0:
                # A very crude interpolation, using no depletion,
                # solar depletion patterns and fully depleted metal
                # corresponding to 0, solar and 1 dust-to-metal ratio
                dep[i] = np.interp(d2m, np.array([0.0, self.dust_to_metal(self.sol, self.depsol), 1.0]),
                                   np.array([1.0, self.depsol[i], 0.0]))
            else:
                dep[i] = 1.0

        return dep






class AbundancePattern(Abundances):

    def __init__(self, params, a, a_nodep = None):

        self.params = params
        self.a = a
        self.a_nodep = a_nodep
        self.Z = self.metallicity(self.a)

        self.names = [self.name[e] for e in self.all_elements]


    def solar_relative_abundance(self, e, ref_element = 'H'):

        """
        This function returns an element's abundance relative to that in the Sun, i.e. [X/H] = log10(N_X/N_H) - log10(N_X/N_H)_sol
        :param a: the element of interest
        :param a: a dictionary with the absolute elemental abundances
        """

        return (self.a[e]-self.a[ref_element]) - (self.sol[e]-self.sol[ref_element])


    def plot_abundances(self, show = True):

        """
        Make a nice plot of the abundances
        """


        fig = plt.figure(figsize = (7., 4.))

        left  = 0.15
        height = 0.8
        bottom = 0.2
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        colors = cmr.take_cmap_colors('cmr.bubblegum', len(self.all_elements))

        for i,(e,c) in enumerate(zip(self.all_elements, colors)):
            ax.axvline(i, alpha = 0.05, lw=1, c='k')
            ax.scatter(i, self.a[e], color=c, s=50)
            if e in self.alpha_elements:
                ax.scatter(i, self.a[e], color=c, s=50, lw=2, ec='k')


        ax.set_xticks(range(len(self.all_elements)), self.names, rotation = 90, fontsize = 6.)
        ax.set_ylim([-12., 0.1])
        ax.set_ylabel(r'$\rm X/H$')

        if show: plt.show()

        return fig, ax



def plot_abundance_patterns(abundance_patterns, labels = None, show = True, ylim = None):

    """ Plot multiple abundance patterns """


    fig = plt.figure(figsize = (7., 4.))

    left  = 0.15
    height = 0.75
    bottom = 0.2
    width = 0.8

    ax = fig.add_axes((left, bottom, width, height))

    a = abundance_patterns[0]

    colors = cmr.take_cmap_colors('cmr.bubblegum', len(a.all_elements))

    if not labels: labels = range(len(abundance_patterns))

    for a, label, ls, ms in zip(abundance_patterns, labels, ['-','--','-.',':'], ['o','s','D','d','^']):

        i_ = range(len(a.all_elements))
        a_ = []

        for i,(e,c) in enumerate(zip(a.all_elements, colors)):
            ax.scatter(i, a.a[e], color=c, s=40, zorder = 2, marker = ms)
            a_.append(a.a[e])

        ax.plot(i_,a_, lw = 2, ls = ls, c='0.5', label = rf'$\rm {label}$', zorder = 1)


    for i,(e,c) in enumerate(zip(a.all_elements, colors)):
        ax.axvline(i, alpha = 0.05, lw=1, c='k', zorder = 0)

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim([-12., 0.1])

    ax.legend()
    ax.set_xticks(range(len(a.all_elements)), a.names, rotation = 90, fontsize = 6.)

    ax.set_ylabel(r'$\rm X/H$')

    if show: plt.show()

    return fig, ax
