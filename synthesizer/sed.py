



import numpy as np

from scipy.integrate import simps
from scipy.stats import linregress

from . import igm



h = 6.626E-34*1E7 # erg/Hz
c = 3.E8 #m/s


class Sed:

    def __init__(self, lam, description = False):

        """ Initialise an empty spectral energy distribution object """

        self.description = description

        self.lam = lam # \AA
        self.lnu = np.zeros(self.lam.shape) # luminosity ers/s/Hz
        self.nu = 3E8/(self.lam/1E10) # Hz


    def return_beta(self, wv = [1500., 2500.]):

        """ Return the UV continuum slope (\beta) based on measurements at two wavelength. """

        f0 = np.interp(wv[0], self.lam, self.lnu)
        f1 = np.interp(wv[1], self.lam, self.lnu)

        return np.log10(f0/f1)/np.log10(wv[0]/wv[1])-2.0

    def return_beta_spec(self, wv = [1250., 3000.]):

        """ Return the UV continuum slope (\beta) based on linear regression to the spectra over a wavelength range. """

        s = (self.lam>wv[0])&(self.lam<wv[1])

        slope, intercept, r, p, se = linregress(np.log10(self.lam[s]), np.log10(self.lnu[s]))

        return slope-2.0


    def get_Lnu(self, F): # broad band luminosity/erg/s/Hz

        self.Lnu = {f: np.trapz(self.lnu * F[f].T, self.lam) / np.trapz(F[f].T, self.lam) for f in F['filters']}


    def get_fnu(self, cosmo, z, igm = igm.madau96):

        """
        Calculate the observed frame spectral energy distribution in nJy
        """

        self.lamz = self.lam * (1. + z)
        self.fnu = 1E23 * 1E9 * self.lnu * (1.+z) / (4 * np.pi * cosmo.luminosity_distance(z).to('cm').value**2) # nJy

        if igm:
            self.fnu *= igm(self.lamz, z)


    def get_Fnu(self, F): # broad band flux/nJy

        self.Fnu = {f: np.trapz(self.fnu * F[f].T, self.lamz) / np.trapz(F[f].T, self.lamz) for f in F['filters']}

        self.Fnu_array = np.array([self.Fnu[f] for f in F['filters']])

    def return_Fnu(self, F): # broad band flux/nJy

        return {f: np.trapz(self.fnu * F[f].T, self.lamz) / np.trapz(F[f].T, self.lamz) for f in F['filters']}


    def return_log10Q(self):
        """
        measure the ionising photon luminosity
        :return:
        """

        llam = self.lnu * c / (self.lam**2*1E-10) # erg s^-1 \AA^-1
        nlam = (llam*self.lam*1E-10)/(h*c) # s^-1 \AA^-1
        s = ((self.lam >= 0) & (self.lam < 912)).nonzero()[0]
        Q = simps(nlam[s], self.lam[s])

        return np.log10(Q)





def rebin(l, f, n): # rebin SED [currently destroys original]

    n_len = int(np.floor(len(l)/n))
    l = l[:n_len*n]
    f = f[:n_len*n]
    nl = np.mean(l.reshape(n_len,n), axis=1)
    nf = np.sum(f.reshape(n_len,n), axis=1)/n

    return nl, nf
