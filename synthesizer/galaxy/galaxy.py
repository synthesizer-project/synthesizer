
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr


class BaseGalaxy:

    """ a base galaxy class """

    def T(self):
        """ Calcualte transmission as a function of wavelength """

        return self.spectra['attenuated'].lam, self.spectra['attenuated'].lnu/self.spectra['intrinsic'].lnu

    def Al(self):
        """ Calcualte attenuation as a function of wavelength """

        lam, T = self.T()

        return lam, -2.5*np.log10(T)

    def A(self, l):
        """ Calculate attenuation at a given wavelength """

        lam, Al = self.Al()

        return np.interp(l, lam, Al)

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
                for f in fc:
                    wv = f.pivwv()
                    filter_ax.plot(f.lam, f.t)
                    ax.scatter(wv, sed.broadband_fluxes[f.filter_code])

        ax.set_xlim([5000., 100000.])
        ax.set_ylim([0., 100])
        filter_ax.set_ylim([-1., 5])
        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r'$\rm log_{10}(\lambda_{obs}/\AA)$')
        ax.set_ylabel(r'$\rm log_{10}(f_{\nu}/nJy)$')

        if show:
            plt.show()

        return fig, ax
