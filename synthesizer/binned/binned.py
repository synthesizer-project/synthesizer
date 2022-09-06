

import numpy as np
from ..sed import Sed
import matplotlib.pyplot as plt
import cmasher as cmr

import copy

from . import dust_curves

class empty: pass

import h5py




# def generate_SED(grid, sfzh, fesc = False, dust_model = False, dust_parameters = False):
#
#     sed = SED(grid, sfzh)
#
#     if fesc:
#
#         sed.add_nebular_emission(fesc)
#
#     if dust_model:
#
#         sed.





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
