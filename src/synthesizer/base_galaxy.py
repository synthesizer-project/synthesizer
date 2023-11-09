import numpy as np
import matplotlib.pyplot as plt

from synthesizer.sed import Sed
from synthesizer.dust.attenuation import PowerLaw
from synthesizer import exceptions
from synthesizer.line import Line
from synthesizer.particle import Stars as ParticleStars
from synthesizer.parametric import BinnedSFZH as ParametricStars


class BaseGalaxy:
    """
    The base galaxy class
    """

    def __init__(self, stars, gas, black_holes, **kwargs):
        """
        Instantiate the base Galaxy class.

        This is the parent class of both parametric.Galaxy and particle.Galaxy.

        Note: The stars, gas, and black_holes component objects differ for
        parametric and particle galaxies but are attached at this parent level
        regardless to unify the Galaxy syntax for both cases.

        Args:

        """
        # Add some place holder attributes which are overloaded on the children
        self.spectra = {}

        # Attach the components
        self.stars = stars
        self.gas = gas
        self.black_holes = black_holes

        if not isinstance(self, (ParametricStars, ParticleStars)):
            raise Warning(
                "Instantiating a BaseGalaxy object is not "
                "supported behaviour. Instead, you should "
                "use one of the derived Galaxy classes:\n"
                "`particle.galaxy.Galaxy`\n"
                "`parametric.galaxy.Galaxy`"
            )

    def get_spectra_dust(self, emissionmodel):
        """
        Calculates dust emission spectra using the attenuated and intrinsic
        spectra that have already been generated and an emission model.

        Parameters
        ----------
        emissionmodel : obj
            The spectral frid

        Returns
        -------
        obj (Sed)
             A Sed object containing the dust attenuated spectra
        """

        # use wavelength grid from attenuated spectra
        # NOTE: in future it might be good to allow a custom wavelength grid

        lam = self.spectra["emergent"].lam

        # calculate the bolometric dust lunminosity as the difference between
        # the intrinsic and attenuated

        dust_bolometric_luminosity = (
            self.spectra["intrinsic"].measure_bolometric_luminosity()
            - self.spectra["emergent"].measure_bolometric_luminosity()
        )

        # get the spectrum and normalise it properly
        lnu = dust_bolometric_luminosity.to("erg/s").value * emissionmodel.lnu(lam)

        # create new Sed object containing dust spectra
        sed = Sed(lam, lnu=lnu)

        # associate that with the component's spectra dictionarity
        self.spectra["dust"] = sed
        self.spectra["total"] = self.spectra["dust"] + self.spectra["emergent"]

        return sed

    def get_equivalent_width(self, feature, blue, red, spectra_to_plot=None):
        """
        Gets all equivalent widths associated with a sed object

        Parameters
        ----------
        index: float
            the index to be used in the computation of equivalent width.
        spectra_to_plot: float array
            An empty list of spectra to be populated.

        Returns
        -------
        equivalent_width : float
            The calculated equivalent width at the current index.
        """

        equivalent_width = None

        if not isinstance(spectra_to_plot, list):
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]

            # Compute equivalent width
            equivalent_width = sed.measure_index(feature, blue, red)

        return equivalent_width

    def transmission(self):
        """
        Calculate transmission as a function of wavelength

        Returns:
            transmission (array)
        """

        return (
            self.spectra["attenuated"].lam,
            self.spectra["attenuated"].lnu / self.spectra["intrinsic"].lnu,
        )

    def Al(self):
        """
        Calculate attenuation as a function of wavelength

        Returns:
            attenuation (array)
        """

        lam, transmission = self.transmission()

        return lam, -2.5 * np.log10(transmission)

    def A(self, l):
        """
        Calculate attenuation at a given wavelength

        Returns:
            attenuation (float)
        """

        lam, Al = self.Al()

        return np.interp(l, lam, Al)

    def AV(self):
        """
        Calculate rest-frame FUV attenuation

        Returns:
            attenuation at rest-frame 1500 angstrom (float)
        """

        return self.A(5500.0)

    def A1500(self):
        """
        Calculate rest-frame FUV attenuation

        Returns:
            attenuation at rest-frame 1500 angstrom (float)
        """

        return self.A(1500.0)

    def plot_spectra(
        self, show=False, spectra_to_plot=None, ylimits=("peak", 5), figsize=(3.5, 5)
    ):
        """
        plots all spectra associated with a galaxy object

        Args:
            show (bool):
                flag for whether to show the plot or just return the
                figure and axes
            spectra_to_plot (None, list):
                list of named spectra to plot that are present in
                `galaxy.spectra`
            figsize (tuple):
                tuple with size 2 defining the figure size

        Returns:
            fig (object)
            ax (object)
        """

        fig = plt.figure(figsize=figsize)

        left = 0.15
        height = 0.6
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))

        if not isinstance(spectra_to_plot, list):
            spectra_to_plot = list(self.spectra.keys())

        # only plot FIR if 'total' is plotted otherwise just plot UV-NIR
        if "total" in spectra_to_plot:
            xlim = [2.0, 7.0]
        else:
            xlim = [2.0, 4.5]

        ypeak = -100
        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            ax.plot(
                np.log10(sed.lam), np.log10(sed.lnu), lw=1, alpha=0.8, label=sed_name
            )

            if np.max(np.log10(sed.lnu)) > ypeak:
                ypeak = np.max(np.log10(sed.lnu))

        # ax.set_xlim([2.5, 4.2])

        if ylimits[0] == "peak":
            if ypeak == ypeak:
                ylim = [ypeak - ylimits[1], ypeak + 0.5]
            ax.set_ylim(ylim)

        ax.set_xlim(xlim)

        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r"$\rm log_{10}(\lambda/\AA)$")
        ax.set_ylabel(r"$\rm log_{10}(L_{\nu}/erg\ s^{-1}\ Hz^{-1} M_{\odot}^{-1})$")

        if show:
            plt.show()

        return fig, ax

    def plot_observed_spectra(
        self,
        cosmo,
        z,
        fc=None,
        show=False,
        spectra_to_plot=None,
        figsize=(3.5, 5.0),
        verbose=True,
    ):
        """
        plots all spectra associated with a galaxy object

        Args:

        Returns:
        """

        fig = plt.figure(figsize=figsize)

        left = 0.15
        height = 0.7
        bottom = 0.1
        width = 0.8

        ax = fig.add_axes((left, bottom, width, height))
        filter_ax = ax.twinx()

        if not isinstance(spectra_to_plot, list):
            spectra_to_plot = list(self.spectra.keys())

        for sed_name in spectra_to_plot:
            sed = self.spectra[sed_name]
            sed.get_fnu(cosmo, z)
            ax.plot(sed.obslam, sed.fnu, lw=1, alpha=0.8, label=sed_name)
            print(sed_name)

            if fc:
                sed.get_broadband_fluxes(fc, verbose=verbose)
                for f in fc:
                    wv = f.pivwv()
                    filter_ax.plot(f.lam, f.t)
                    ax.scatter(wv, sed.broadband_fluxes[f.filter_code], zorder=4)

        # ax.set_xlim([5000., 100000.])
        # ax.set_ylim([0., 100])
        filter_ax.set_ylim([3, 0])
        ax.legend(fontsize=8, labelspacing=0.0)
        ax.set_xlabel(r"$\rm log_{10}(\lambda_{obs}/\AA)$")
        ax.set_ylabel(r"$\rm log_{10}(f_{\nu}/nJy)$")

        if show:
            plt.show()

        return fig, ax  # , filter_ax
