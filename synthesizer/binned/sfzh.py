

import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

from scipy import integrate

from ..plt import single, single_histxy, mlabel


# def bin_width(log10ages, log10Zs):
#
#     """ The age width of each bin """
#
#     BW = np.zeros((len(log10ages), len(log10Zs)))
#     min_age = 0
#     for ia, log10age in enumerate(log10ages[:-1]):
#         max_age = int(10**np.mean([log10ages[ia+1],log10age])) # years
#         BW[ia,:] = max_age-min_age
#         min_age = max_age
#     return BW



def instant(log10ages, metallicities, log10age, metallicity):

    """ simply returns the SFZH where only bin is populated corresponding to the age and metallicity """
    SFZH = np.zeros((len(log10ages), len(metallicities)))
    ia = (np.abs(log10ages - log10age)).argmin()
    iZ = (np.abs(metallicities - metallicity)).argmin()
    SFZH[ia,iZ] = 1
    return SFZH




class BinnedSFZH:

    """ this is a simple object for holding a binned star formation and metal enrichment history. This can be extended with other methods. """

    def __init__(self, log10ages, metallicities, sfzh, sfh_f = None, Zh_f = None):
        self.log10ages = log10ages
        self.log10ages_lims = [self.log10ages[0], self.log10ages[-1]]
        self.metallicities = metallicities
        self.metallicities_lims = [self.metallicities[0], self.metallicities[-1]]
        self.log10metallicities = np.log10(metallicities)
        self.log10metallicities_lims = [self.log10metallicities[0], self.log10metallicities[-1]]
        self.sfzh = sfzh # 2D star formation and metal enrichment history
        self.sfh = np.sum(self.sfzh, axis=1) # 1D star formation history
        self.Z = np.sum(self.sfzh, axis=0) # metallicity distribution
        self.sfh_f = sfh_f # function used to generate the star formation history if given
        self.Zh_f = Zh_f # function used to generate the metallicity history/distribution if given

        # --- check if metallicities on regular grid in log10metallicity or metallicity or not at all (e.g. BPASS
        if len(set(self.metallicities[:-1]-self.metallicities[1:]))==1:
            self.metallicity_grid = 'Z'
        elif len(set(self.log10metallicities[:-1]-self.log10metallicities[1:]))==1:
            self.metallicity_grid = 'log10Z'
        else:
            self.metallicity_grid = None


    def plot(self, show = True):

        """ Make a nice plots of the binned SZFH """

        fig, ax, haxx, haxy = single_histxy()

        ax.imshow(self.sfzh.T, origin = 'lower', extent = [*self.log10ages_lims, self.log10metallicities[0], self.log10metallicities[-1]], cmap = cmr.sunburst, aspect = 'auto') # this is technically incorrect because metallicity is not on a an actual grid.

        # --- add binned Z to right of the plot
        # haxx.step(log10ages, sfh, where='mid', color='k')
        haxy.fill_betweenx(self.log10metallicities, self.Z/np.max(self.Z), step='mid', color='k', alpha = 0.3)

        # --- add binned SFH to top of the plot
        # haxx.step(log10ages, sfh, where='mid', color='k')
        haxx.fill_between(self.log10ages, self.sfh/np.max(self.sfh), step='mid', color='k', alpha = 0.3)

        # --- add SFR to top of the plot
        if self.sfh_f:
            x = np.linspace(*self.log10ages_lims, 1000)
            y = sfh.sfr(10**x)
            haxx.plot(x, y/np.max(y))

        haxy.set_xlim([0., 1.2])
        haxy.set_ylim(self.log10metallicities_lims)
        haxx.set_ylim([0., 1.2])
        haxx.set_xlim(self.log10ages_lims)

        ax.set_xlabel(mlabel('log_{10}(age/yr)'))
        ax.set_ylabel(mlabel('log_{10}Z'))

        if show: plt.show()

        return fig, ax





class Binned:

    def sfh(ages, sfh_, log10 = False):

        if log10:
            ages = 10**ages

        SFH = np.zeros(len(ages))

        min_age = 0
        for ia, age in enumerate(ages[:-1]):
            max_age = int(np.mean([ages[ia+1],ages[ia]])) # years
            sf = integrate.quad(sfh_.sfr, min_age, max_age)[0]
            SFH[ia] = sf
            min_age = max_age

        # --- normalise
        SFH /= np.sum(SFH)

        return SFH


    def sfzh(log10ages, metallicities, sfh, Zh, stellar_mass = 1.):

        """ return an instance of the BinnedSFZH class """

        ages = 10**log10ages

        sfzh = np.zeros((len(log10ages), len(metallicities)))

        if Zh.dist == 'delta':
            min_age = 0
            for ia, age in enumerate(ages[:-1]):
                max_age = int(np.mean([ages[ia+1],ages[ia]])) # years
                sf = integrate.quad(sfh.sfr, min_age, max_age)[0]
                iZ = (np.abs(metallicities - Zh.Z(age))).argmin()
                sfzh[ia,iZ] = sf
                min_age = max_age

        if Zh.dist == 'dist':
            print('WARNING: NOT YET IMPLEMENTED')


        # --- normalise
        sfzh /= np.sum(sfzh)
        sfzh *= stellar_mass

        return BinnedSFZH(log10ages, metallicities, sfzh)




class ZH:

    class deltaConstant:

        """ return a single metallicity as a function of age. """

        def __init__(self, Z_, log10 = False):
            self.dist = 'delta'

            self.Z_ = Z_
            self.log10Z_ = np.log10(Z_)

        def Z(self, age):
            return self.Z_

        def log10Z(self, age):
            return self.log10Z_





class SFH:

    class Common:

        def sfr(self, age):
            if type(age) == np.float:
                return self.sfr_(age)
            elif type(age) == np.ndarray:
                return np.array([self.sfr_(a) for a in age])


    class Constant(Common):

        def __init__(self, duration):
            self.duration = duration

        def sfr_(self, age):
            if age < self.duration:
                return 1.0
            else:
                return 0.0


    class Exponential(Common):

        def __init__(self, tau):

            self.tau = tau

        def sfr_(self, age):

            return np.exp(-age/self.tau)


    class TruncatedExponential(Common):

        def __init__(self, tau, max_age):

            self.tau = tau
            self.max_age = max_age

        def sfr_(self, age):

            if age < self.max_age:
                return np.exp(-age/self.tau)
            else:
                return 0.0


    class LogNormal(Common):

        def __init__(self, peak_age, tau, max_age):

            self.max_age = max_age
            self.peak_age = peak_age
            self.tpeak = max_age-peak_age
            self.tau = tau
            self.T0 = np.log(self.tpeak)+tau**2


        def sfr_(self, age):

            """ age is lookback time """

            if age < self.max_age:
                return (1./(self.max_age-age))*np.exp(-(np.log(self.max_age-age)-self.T0)**2/(2*self.tau**2))
            else:
                return 0.0
