

# --- general
import h5py
import copy
import numpy as np
from scipy import integrate
from unyt import yr, erg, Hz, s, cm, angstrom


import matplotlib.pyplot as plt
import cmasher as cmr

from ..dust import power_law
from ..sed import Sed, convert_fnu_to_flam
from ..line import Line
from ..plt import single_histxy, mlabel
from ..stats import weighted_median, weighted_mean



class Galaxy():

    def __init__(self, SFZH):


        self.sfzh = SFZH.sfzh
        self.sfzh_ = np.expand_dims(self.sfzh, axis=2) # add an extra dimension to the sfzh to allow the fast summation
        self.spectra = {} # dictionary holding spectra
        self.lines = {} # dictionary holding lines


    def get_Q(self, grid):
        """ return the ionising photon luminosity (log10Q) for a given SFZH. """

        return np.sum(10**self.grid.log10Q * self.sfzh, axis=(0, 1))




    def generate_lnu(self, grid, spectra_name):

        return np.sum(grid.spectra[spectra_name] * self.sfzh_, axis=(0, 1))  # calculate pure stellar emission



    def get_stellar_spectra(self, grid, update = True):

        """ generate the pure stellar spectra using the provided grid"""

        lnu = self.generate_lnu(grid, 'stellar')

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['stellar'] = sed

        return sed


    def get_nebular_spectra(self, grid, fesc = 0.0, update = True):

        lnu = self.generate_lnu(grid, 'nebular')

        lnu *= (1-fesc)

        sed = Sed(grid.lam, lnu)

        if update:
            self.spectra['nebular'] = sed

        return sed


    def get_intrinsic_spectra(self, grid, fesc = 0.0, update = True):

        """ this generates the intrinsic spectra, i.e. not including dust but including nebular emission. It also generates the stellar and nebular spectra too. """

        stellar = self.get_stellar_spectra(grid, update = update)
        nebular = self.get_nebular_spectra(grid, fesc, update = update)

        sed = Sed(grid.lam, stellar.lnu + nebular.lnu)

        if update:
            self.spectra['intrinsic'] = sed

        return sed



    def get_screen_spectra(self, grid, tauV=None, fesc=0.0, dust_curve = power_law({'slope': -1.}), update = True):

        """
        Similar to get_intrinsic_spectra but applies a dust screen
        """

        # --- begin by calculating intrinsic spectra
        intrinsic = self.get_intrinsic_spectra(grid, fesc, update = update)

        if tauV:
            T = np.exp(-tauV) * dust_curve.T(grid.lam)
        else:
            T = 1.0

        sed = Sed(grid.lam, T * intrinsic.lnu)

        if update:
            self.spectra['attenuated'] = sed

        return sed





    def get_pacman_spectra(self, grid, fesc=0.0, fesc_LyA=1.0, tauV=None, dust_curve = power_law({'slope': -1.}), update = True):

        """ in the PACMAN model some fraction (fesc) of the pure stellar emission is assumed to completely escape the galaxy without reprocessing by gas or dust. The rest is assumed to be reprocessed by both gas and a screen of dust. """


        # --- begin by generating the pure stellar spectra
        stellar = self.get_stellar_spectra(grid, update = update)

        # --- this is the starlight that escapes any reprocessing
        self.spectra['escape'] = Sed(grid.lam, fesc * stellar.lnu)

        # --- this is the starlight after reprocessing by gas
        self.spectra['reprocessed'] = Sed(grid.lam)
        self.spectra['intrinsic'] = Sed(grid.lam)
        self.spectra['attenuated'] = Sed(grid.lam)
        self.spectra['total'] = Sed(grid.lam)

        if fesc_LyA < 1.0:
            # if Lyman-alpha escape fraction is specified reduce LyA luminosity

            # --- generate contribution of line emission alone and reduce the contribution of Lyman-alpha
            linecont = np.sum(grid.spectra['linecont'] * self.sfzh_, axis=(0, 1))
            idx = grid.get_nearest_index(1216., grid.lam)  # get index of Lyman-alpha
            linecont[idx] *= fesc_LyA  # reduce the contribution of Lyman-alpha

            nebular_continuum = np.sum(
                grid.spectra['nebular_continuum'] * self.sfzh_, axis=(0, 1))
            transmitted = np.sum(grid.spectra['transmitted'] * self.sfzh_, axis=(0, 1))
            self.spectra['reprocessed'].lnu = (
                1.-fesc) * (linecont + nebular_continuum + transmitted)

        else:
            self.spectra['reprocessed'].lnu = (
                1.-fesc) * np.sum(grid.spectra['total'] * self.sfzh_, axis=(0, 1))

        self.spectra['intrinsic'].lnu = self.spectra['escape'].lnu + \
            self.spectra['reprocessed'].lnu  # the light before reprocessing by dust

        if tauV:
            T = np.exp(-tauV) * dust_curve.T(grid.lam)
            self.spectra['attenuated'].lnu = self.spectra['escape'].lnu + \
                T*self.spectra['reprocessed'].lnu
            self.spectra['total'].lnu = self.spectra['attenuated'].lnu
        else:
            self.spectra['total'].lnu = self.spectra['escape'].lnu + self.spectra['reprocessed'].lnu

        return self.spectra['total']


    def get_CF00_spectra(tauV, p={}):
        """ add Charlot \& Fall (2000) dust """

        print('WARNING: not yet implemented')


    def Al(self):
        """ Calcualte attenuation as a function of wavelength """

        return -2.5*np.log10(self.total.lnu/self.intrinsic.lnu)

    def A(self, l):
        """ Calculate attenuation at a given wavelength """

        return -2.5*np.log10(np.interp(l, self.total.lam, self.total.lnu)/np.interp(l, self.intrinsic.lam, self.intrinsic.lnu))

    def A1500(self):
        """ Calculate rest-frame FUV attenuation """

        return self.A(1500.)

    def plot_spectra(self, show=True, spectra_to_plot=None):
        """ plots all spectra associated with a galaxy object """

        fig = plt.figure(figsize=(3.5, 5.))

        left = 0.15
        height = 0.8
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        if type(spectra_to_plot) != list:
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            ax.plot(np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=sed_name)

        ax.set_xlim([2.5, 4.2])
        ax.set_ylim([27., 29.5])
        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r'$\rm log_{10}(\lambda/\AA)$')
        ax.set_ylabel(r'$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$')

        if show:
            plt.show()

        return fig, ax

    def plot_observed_spectra(self, cosmo, z, fc=None, show=True, spectra_to_plot=None):
        """ plots all spectra associated with a galaxy object """

        fig = plt.figure(figsize=(3.5, 5.))

        left = 0.15
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
            ax.plot(sed.lamz, sed.fnu, lw=1, alpha=0.8, label=sed_name)
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
        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r'$\rm log_{10}(\lambda_{obs}/\AA)$')
        ax.set_ylabel(r'$\rm log_{10}(f_{\nu}/nJy)$')

        if show:
            plt.show()

        return fig, ax



    def get_intrinsic_line(self, grid, line_id, quantity = False, update = True):

        """ return intrinsic quantities (luminosity, EW) for a single line or line set """

        if type(line_id) is str:
            line_id = [line_id]

        luminosity_ = []
        continuum_ = []
        wavelength_ = []

        for line_id_ in line_id:
            grid_line = grid.lines[line_id_]

            wavelength_.append(grid_line['wavelength'])  # \AA
            continuum_.append(np.sum(grid_line['continuum'] * self.sfzh, axis=(0, 1))) #  continuum at line wavelength, erg/s/Hz
            luminosity_.append(np.sum(grid_line['luminosity'] * self.sfzh, axis=(0, 1)))

        # --- create line object

        line = Line(line_id, wavelength_, continuum_, luminosity_)

        if update:
            self.lines[line.id] = line

        return line


    # def get_intrinsic_line(self, tauV):
    #
    #     return
    #
    # def apply_dust_pacman_lines(self):
    #
    #     return
    #
    # def apply_dust_CF00_lines(self):
    #
    #     return
