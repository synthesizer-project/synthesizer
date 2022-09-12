

# --- general
import h5py
import copy
import numpy as np
from scipy import integrate
from unyt import yr



import matplotlib.pyplot as plt
import cmasher as cmr

from . import dust_curves
from .sed import Sed
from .plt import single_histxy, mlabel
from .stats import weighted_median, weighted_mean



class SEDGenerator():

    def __init__(self, grid, SFZH):

        self.grid = grid
        self.sfzh = np.expand_dims(SFZH.sfzh, axis=2) # add an extra dimension to the sfzh to allow the fast summation

        self.lam = self.grid.lam
        self.log10lam = np.log10(self.grid.lam)


        self.spectra = {}

        # --- calculate stellar SED
        self.spectra['stellar'] = Sed(self.lam) # pure stellar emission
        self.spectra['stellar'].lnu = np.sum(self.grid.spectra['stellar'] * self.sfzh, axis =(0,1) ) # calculate pure stellar emission

        self.spectra['intrinsic'] = Sed(self.lam) # nebular + stellar (but no dust)
        self.spectra['attenuated'] = Sed(self.lam) # nebular + stellar (but no dust)

        self.spectra['total'] = copy.deepcopy(self.spectra['stellar']) # nebular + stellar + dust


    def screen(self, tauV = None, dust_curve = 'power_law', dust_parameters = {'slope': -1.}):

        """ in the simple screen model all starlight is equally affected by a screen of gas and dust. By definition fesc = 0.0. """

        self.spectra['intrinsic'].lnu = np.sum(self.grid.spectra['total'] * self.sfzh, axis = (0,1) ) # -- stellar transmitted + nebular

        if tauV:
            tau = tauV * getattr(dust_curves, dust_curve)(params = dust_parameters).tau(self.lam)
            T = np.exp(-tau)
        else:
            T = 1.0

        self.spectra['nebular'] = Sed(self.lam)
        self.spectra['nebular'].lnu = self.spectra['intrinsic'].lnu - self.spectra['stellar'].lnu
        self.spectra['total'].lnu = self.spectra['intrinsic'].lnu * T



    def pacman(self, fesc = 0.0, fesc_LyA = 1.0, tauV = None, dust_curve = 'power_law', dust_parameters = {'slope': -1.}):

        """ in the PACMAN model some fraction (fesc) of the pure stellar emission is assumed to completely escape the galaxy without reprocessing by gas or dust. The rest is assumed to be reprocessed by both gas and a screen of dust. """

        self.spectra['escape'] = Sed(self.lam) # this is the starlight that escapes any reprocessing
        self.spectra['escape'].lnu = fesc * self.spectra['stellar'].lnu

        self.spectra['reprocessed'] = Sed(self.lam) # this is the starlight after reprocessing by gas

        if fesc_LyA<1.0:
            # if Lyman-alpha escape fraction is specified reduce LyA luminosity

            # --- generate contribution of line emission alone and reduce the contribution of Lyman-alpha
            linecont = np.sum(self.grid.spectra['linecont'] * self.sfzh, axis=(0,1))
            idx = self.grid.get_nearest_index(1216., self.grid.lam) # get index of Lyman-alpha
            linecont[idx] *= fesc_LyA # reduce the contribution of Lyman-alpha

            nebular_continuum = np.sum(self.grid.spectra['nebular_continuum'] * self.sfzh, axis=(0,1))
            transmitted = np.sum(self.grid.spectra['transmitted'] * self.sfzh, axis=(0,1))
            self.spectra['reprocessed'].lnu = (1.-fesc) * (linecont + nebular_continuum + transmitted)

        else:
            self.spectra['reprocessed'].lnu = (1.-fesc) * np.sum(self.grid.spectra['total'] * self.sfzh, axis=(0,1))

        self.spectra['intrinsic'].lnu =  self.spectra['escape'].lnu + self.spectra['reprocessed'].lnu # the light before reprocessing by dust

        if tauV:
            tau = tauV * getattr(dust_curves, dust_curve)(params = dust_parameters).tau(self.lam)
            T = np.exp(-tau)
            self.spectra['attenuated'].lnu = self.spectra['escape'].lnu + T*self.spectra['reprocessed'].lnu
            self.spectra['total'].lnu = self.spectra['attenuated'].lnu
        else:
            self.spectra['total'].lnu = self.spectra['escape'].lnu + self.spectra['reprocessed'].lnu



    def CF00_dust(tauV, p = {}):

        """ add Charlot \& Fall (2000) dust """

        print('WARNING: not yet implemented')

    # def get_Q(self, SFZH):
    #     return np.log10(np.sum(10**self.grid['log10Q'] * SFZH, axis=(0,1)))
    #
    # def get_log10Q(self, SFZH):
    #     return self.get_Q(SFZH)



    def Al(self):

        """ Calcualte attenuation as a function of wavelength """

        return -2.5*np.log10(self.total.lnu/self.intrinsic.lnu)

    def A(self,l):

        """ Calculate attenuation at a given wavelength """

        return -2.5*np.log10(np.interp(l, self.total.lam, self.total.lnu)/np.interp(l, self.intrinsic.lam, self.intrinsic.lnu))

    def A1500(self):

        """ Calculate rest-frame FUV attenuation """

        return self.A(1500.)




    def plot_spectra(self, show = True, spectra_to_plot = None):

        """ plots all spectra associated with a galaxy object """

        fig = plt.figure(figsize = (3.5, 5.))

        left  = 0.15
        height = 0.8
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        if type(spectra_to_plot) != list:
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            ax.plot(np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha = 0.8, label = sed_name)

        ax.set_xlim([2.5, 4.2])
        ax.set_ylim([27., 29.5])
        ax.legend(fontsize = 8, labelspacing = 0.0)
        ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
        ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')

        if show: plt.show()

        return fig, ax


    def plot_observed_spectra(self, cosmo, z, fc = None, show = True, spectra_to_plot = None):

        """ plots all spectra associated with a galaxy object """

        fig = plt.figure(figsize = (3.5, 5.))

        left  = 0.15
        height = 0.8
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))
        filter_ax = ax.twinx()

        if type(spectra_to_plot) != list:
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            sed.get_fnu(cosmo, z)
            ax.plot(sed.lamz, sed.fnu, lw=1, alpha = 0.8, label = sed_name)
            print(sed_name)

            if fc:
                sed.get_broadband_fluxes(fc)
                for f, filter in fc.filter.items():
                    wv = filter.pivwv()
                    filter_ax.plot(filter.lam, filter.t)
                    ax.scatter(wv, sed.broadband_fluxes[f])



        ax.set_xlim([5000., 100000.])
        ax.set_ylim([0., 100])
        filter_ax.set_ylim([-1., 5])
        ax.legend(fontsize = 8, labelspacing = 0.0)
        ax.set_xlabel(r'$\rm log_{10}(\lambda_{obs}/\AA)$')
        ax.set_ylabel(r'$\rm log_{10}(f_{\nu}/nJy)$')

        if show: plt.show()

        return fig, ax




















# def rebin_SFZH(sfzh, new_log10ages, new_metallicities):
#
#     """ take a BinnedSFZH object and rebin it on to a new grid. The context is taking a binned SFZH from e.g. a SAM and mapping it on to a new grid e.g. from a particular SPS model """









class BinnedSFZH:

    """ this is a simple class for holding a binned star formation and metal enrichment history. This can be extended with other methods. """

    def __init__(self, log10ages, metallicities, sfzh, sfh_f = None, Zh_f = None):
        self.log10ages = log10ages
        self.ages = 10**log10ages
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

    def calculate_median_age(self):

        """ calculate the median age """

        return weighted_median(self.ages, self.sfh) * yr

    def calculate_mean_age(self):

        """ calculate the mean age """

        return weighted_mean(self.ages, self.sfh) * yr


    def calculate_mean_metallicity(self):

        """ calculate the mean metallicity """

        return weighted_mean(self.metallicities, self.Z)

    def summary(self):

        """ print basic summary of the binned star formation and metal enrichment history """

        print('-'*10)
        print('SUMMARY OF BINNED SFZH')
        print(f'median age: {self.calculate_median_age().to("Myr"):.2f}')
        print(f'mean age: {self.calculate_mean_age().to("Myr"):.2f}')
        print(f'mean metallicity: {self.calculate_mean_metallicity():.4f}')



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









def generate_sfh(ages, sfh_, log10 = False):

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

def generate_instant_sfzh(log10ages, metallicities, log10age, metallicity):

    """ simply returns the SFZH where only bin is populated corresponding to the age and metallicity """

    sfzh = np.zeros((len(log10ages), len(metallicities)))
    ia = (np.abs(log10ages - log10age)).argmin()
    iZ = (np.abs(metallicities - metallicity)).argmin()
    sfzh[ia,iZ] = 1

    return BinnedSFZH(log10ages, metallicities, sfzh)



def generate_sfzh(log10ages, metallicities, sfh, Zh, stellar_mass = 1.):

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

    """ A collection of classes describing the metallicity history (and distribution) """


    class deltaConstant:

        """ return a single metallicity as a function of age. """

        def __init__(self, parameters):

            self.dist = 'delta' # set distribution type
            self.parameters = parameters
            if 'Z' in parameters.keys():
                self.Z_ = parameters['Z']
                self.log10Z_ = np.log10(self.Z_)
            elif 'log10Z' in parameters.keys():
                self.log10Z_ = parameters['log10Z']
                self.Z_ = 10**self.log10Z_

        def Z(self, age):
            return self.Z_

        def log10Z(self, age):
            return self.log10Z_





class SFH:

    """ A collection of classes describing parametric star formation histories """

    class Common:

        def sfr(self, age):
            if type(age) == np.float:
                return self.sfr_(age)
            elif type(age) == np.ndarray:
                return np.array([self.sfr_(a) for a in age])

        def calculate_sfh(self, t_range = [0, 1E10], dt = 1E6):

            """ calcualte the age of a given star formation history """

            t = np.arange(*t_range, dt)
            sfh = self.sfr(t)
            return t, sfh

        def calculate_median_age(self, t_range = [0, 1E10], dt = 1E6):

            """ calcualte the median age of a given star formation history """

            t, sfh = self.calculate_sfh(t_range = t_range, dt = dt)

            return weighted_median(t, sfh) * yr

        def calculate_mean_age(self, t_range = [0, 1E10], dt = 1E6):

            """ calcualte the median age of a given star formation history """

            t, sfh = self.calculate_sfh(t_range = t_range, dt = dt)

            return weighted_mean(t, sfh) * yr

        def calculate_moment(self, n):

            """ calculate the n-th moment of the star formation history """

            print('WARNING: not yet implemnted')
            return


        def summary(self):

            """ print basic summary of the star formation history """

            print('-'*10)
            print('SUMMARY OF PARAMETERISED SFH')
            print(self.__class__)
            for parameter_name, parameter_value in self.parameters.items():
                print(f'{parameter_name}: {parameter_value}')
            print(f'median age: {self.calculate_median_age().to("Myr"):.2f}')
            print(f'mean age: {self.calculate_mean_age().to("Myr"):.2f}')




    class Constant(Common):

        """
        A constant star formation history
            sfr = 1; t<=duration
            sfr = 0; t>duration
        """

        def __init__(self, parameters):
            self.parameters = parameters
            self.duration = self.parameters['duration'].to('yr').value

        def sfr_(self, age):
            if age <= self.duration:
                return 1.0
            else:
                return 0.0


    class Exponential(Common):

        """
        An exponential star formation history
        """

        def __init__(self, parameters):
            self.parameters = parameters
            self.tau, = self.parameters['tau'].to('yr').value

        def sfr_(self, age):

            return np.exp(-age/self.tau)


    class TruncatedExponential(Common):

        """
        A truncated exponential star formation history
        """

        def __init__(self, parameters):
            self.parameters = parameters
            self.tau = self.parameters['tau'].to('yr').value
            self.max_age = self.parameters['max_age'].to('yr').value


        def sfr_(self, age):

            if age < self.max_age:
                return np.exp(-age/self.tau)
            else:
                return 0.0


    class LogNormal(Common):
        """
        A log-normal star formation history
        """

        def __init__(self, parameters):
            self.parameters = parameters
            self.peak_age = self.parameters['peak_age'].to('yr').value
            self.tau = self.parameters['tau']
            self.max_age = self.parameters['max_age'].to('yr').value

            self.tpeak = self.max_age-self.peak_age
            self.T0 = np.log(self.tpeak)+self.tau**2


        def sfr_(self, age):

            """ age is lookback time """

            if age < self.max_age:
                return (1./(self.max_age-age))*np.exp(-(np.log(self.max_age-age)-self.T0)**2/(2*self.tau**2))
            else:
                return 0.0
