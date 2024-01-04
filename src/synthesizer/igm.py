"""calculates the absorption due to the intergalactic medium
using e.g. Madau et al. formalism."""

import os
import numpy as np
from . import __file__ as filepath


__all__ = ["Inoue14", "Madau96"]


class Inoue14:
    """
    IGM absorption from Inoue et al. (2014)

    Taken from py-eazy.

    Args:
        scale_tau : float
            Parameter multiplied to the IGM :math:`\tau` values (exponential
            in the linear absorption fraction).
            I.e., :math:`f_{\\mathrm{igm}} = e^{-\\mathrm{scale_\tau} \tau}`.
    """

    def __init__(self, scale_tau=1.0):
        self._load_data()
        self.scale_tau = scale_tau
        self.name = "Inoue14"

    def _load_data(self):
        path = os.path.join(os.path.dirname(filepath), "data")
        # print path

        LAF_file = os.path.join(path, "LAFcoeff.txt")
        DLA_file = os.path.join(path, "DLAcoeff.txt")

        data = np.loadtxt(LAF_file, unpack=True)
        ix, lam, ALAF1, ALAF2, ALAF3 = data
        self.lam = lam[:, np.newaxis]
        self.ALAF1 = ALAF1[:, np.newaxis]
        self.ALAF2 = ALAF2[:, np.newaxis]
        self.ALAF3 = ALAF3[:, np.newaxis]

        data = np.loadtxt(DLA_file, unpack=True)
        ix, lam, ADLA1, ADLA2 = data
        self.ADLA1 = ADLA1[:, np.newaxis]
        self.ADLA2 = ADLA2[:, np.newaxis]

        return True

    def tLSLAF(self, zS, lobs):
        """
        Lyman series, Lyman-alpha forest
        """
        z1LAF = 1.2
        z2LAF = 4.7

        l2 = self.lam  # [:, np.newaxis]
        tLSLAF_value = np.zeros_like(lobs * l2).T

        x0 = lobs < l2 * (1 + zS)
        x1 = x0 & (lobs < l2 * (1 + z1LAF))
        x2 = x0 & ((lobs >= l2 * (1 + z1LAF)) & (lobs < l2 * (1 + z2LAF)))
        x3 = x0 & (lobs >= l2 * (1 + z2LAF))

        tLSLAF_value = np.zeros_like(lobs * l2)
        tLSLAF_value[x1] += ((self.ALAF1 / l2**1.2) * lobs**1.2)[x1]
        tLSLAF_value[x2] += ((self.ALAF2 / l2**3.7) * lobs**3.7)[x2]
        tLSLAF_value[x3] += ((self.ALAF3 / l2**5.5) * lobs**5.5)[x3]

        return tLSLAF_value.sum(axis=0)

    def tLSDLA(self, zS, lobs):
        """
        Lyman Series, DLA
        """
        z1DLA = 2.0

        l2 = self.lam  # [:, np.newaxis]
        tLSDLA_value = np.zeros_like(lobs * l2)

        x0 = (lobs < l2 * (1 + zS)) & (lobs < l2 * (1.0 + z1DLA))
        x1 = (lobs < l2 * (1 + zS)) & ~(lobs < l2 * (1.0 + z1DLA))

        tLSDLA_value[x0] += ((self.ADLA1 / l2**2) * lobs**2)[x0]
        tLSDLA_value[x1] += ((self.ADLA2 / l2**3) * lobs**3)[x1]

        return tLSDLA_value.sum(axis=0)

    def tLCDLA(self, zS, lobs):
        """
        Lyman continuum, DLA
        """
        z1DLA = 2.0
        lamL = 911.8

        tLCDLA_value = np.zeros_like(lobs)

        x0 = lobs < lamL * (1.0 + zS)
        if zS < z1DLA:
            tLCDLA_value[x0] = (
                0.2113 * _pow(1.0 + zS, 2) - 0.07661 * _pow(1.0 + zS, 2.3) *
                _pow(lobs[x0] / lamL, (-3e-1)) - 0.1347 *
                _pow(lobs[x0] / lamL, 2)
            )
        else:
            x1 = lobs >= lamL * (1.0 + z1DLA)

            tLCDLA_value[x0 & x1] = (
                0.04696 * _pow(1.0 + zS, 3) - 0.01779 * _pow(1.0 + zS, 3.3) *
                _pow(lobs[x0 & x1] / lamL, (-3e-1)) - 0.02916 *
                _pow(lobs[x0 & x1] / lamL, 3)
            )
            tLCDLA_value[x0 & ~x1] = (
                0.6340 + 0.04696 * _pow(1.0 + zS, 3) - 0.01779 *
                _pow(1.0 + zS, 3.3) * _pow(lobs[x0 & ~x1] / lamL, (-3e-1)) -
                0.1347 * _pow(lobs[x0 & ~x1] / lamL, 2) - 0.2905 *
                _pow(lobs[x0 & ~x1] / lamL, (-3e-1))
            )

        return tLCDLA_value

    def tLCLAF(self, zS, lobs):
        """
        Lyman continuum, LAF
        """
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value = np.zeros_like(lobs)

        x0 = lobs < lamL * (1.0 + zS)

        if zS < z1LAF:
            tLCLAF_value[x0] = 0.3248 * (
                _pow(lobs[x0] / lamL, 1.2) -
                _pow(1.0 + zS, -9e-1) * _pow(lobs[x0] / lamL, 2.1)
            )
        elif zS < z2LAF:
            x1 = lobs >= lamL * (1 + z1LAF)
            tLCLAF_value[x0 & x1] = 2.545e-2 * (
                _pow(1.0 + zS, 1.6) * _pow(lobs[x0 & x1] / lamL, 2.1) -
                _pow(lobs[x0 & x1] / lamL, 3.7)
            )
            tLCLAF_value[x0 & ~x1] = (
                2.545e-2 * _pow(1.0 + zS, 1.6) *
                _pow(lobs[x0 & ~x1] / lamL, 2.1) + 0.3248 *
                _pow(lobs[x0 & ~x1] / lamL, 1.2) - 0.2496 *
                _pow(lobs[x0 & ~x1] / lamL, 2.1)
            )
        else:
            x1 = lobs > lamL * (1.0 + z2LAF)
            x2 = (lobs >= lamL * (1.0 + z1LAF)) & (lobs < lamL * (1.0 + z2LAF))
            x3 = lobs < lamL * (1.0 + z1LAF)

            tLCLAF_value[x0 & x1] = 5.221e-4 * (
                _pow(1.0 + zS, 3.4) * _pow(lobs[x0 & x1] / lamL, 2.1) -
                _pow(lobs[x0 & x1] / lamL, 5.5)
            )
            tLCLAF_value[x0 & x2] = (
                5.221e-4 * _pow(1.0 + zS, 3.4) *
                _pow(lobs[x0 & x2] / lamL, 2.1) +
                0.2182 * _pow(lobs[x0 & x2] / lamL, 2.1) - 2.545e-2 *
                _pow(lobs[x0 & x2] / lamL, 3.7)
            )
            tLCLAF_value[x0 & x3] = (
                5.221e-4 * _pow(1.0 + zS, 3.4) *
                _pow(lobs[x0 & x3] / lamL, 2.1) + 0.3248 *
                _pow(lobs[x0 & x3] / lamL, 1.2) - 3.140e-2 *
                _pow(lobs[x0 & x3] / lamL, 2.1)
            )

        return tLCLAF_value

    def tau(self, z, lobs):
        """Get full Inoue IGM absorption

        Parameters
        ----------
        z : float
            Redshift to evaluate IGM absorption

        lobs : array
            Observed-frame wavelength(s) in Angstroms.

        Returns
        -------
        abs : array
            IGM absorption

        """
        tau_LS = self.tLSLAF(z, lobs) + self.tLSDLA(z, lobs)
        tau_LC = self.tLCLAF(z, lobs) + self.tLCDLA(z, lobs)

        # Upturn at short wavelengths, low-z
        # k = 1./100
        # l0 = 600-6/k
        # clip = lobs/(1+z) < 600.
        # tau_clip = 100*(1-1./(1+np.exp(-k*(lobs/(1+z)-l0))))
        tau_clip = 0.0

        return self.scale_tau * (tau_LC + tau_LS + tau_clip)

    def T(self, z, lobs):
        tau = self.tau(z, lobs)
        t = np.exp(-tau)

        t[t != t] = 0.0  # squash NaNs
        t[t > 1] = 1

        return t


class Madau96:
    def __init__(self):
        self.wvs = [1216.0, 1026.0, 973.0, 950.0]
        self.a = [0.0036, 0.0017, 0.0012, 0.00093]
        self.name = "Madau96"

    def T(self, z, lobs):
        Expteff = np.array([])
        for _l in lobs:
            if _l > self.wvs[0] * (1 + z):
                Expteff = np.append(Expteff, 1)
                continue

            if _l <= self.wvs[-1] * (1 + z) - 1500:
                Expteff = np.append(Expteff, 0)
                continue

            teff = 0
            for i in range(0, len(self.wvs) - 1, 1):
                teff += self.a[i] * (_l / self.wvs[i]) ** 3.46
                if self.wvs[i + 1] * (1 + z) < _l <= self.wvs[i] * (1 + z):
                    Expteff = np.append(Expteff, np.exp(-teff))
                    continue

            if _l <= self.wvs[-1] * (1 + z):
                Expteff = np.append(
                    Expteff,
                    np.exp(
                        -(
                            teff + 0.25 * (_l / self.wvs[-1]) ** 3 *
                            ((1 + z) ** 0.46 - (_l / self.wvs[-1]) ** 0.46) +
                            9.4 * (_l / self.wvs[-1]) ** 1.5 *
                            ((1 + z) ** 0.18 - (_l / self.wvs[-1]) ** 0.18) -
                            0.7 * (_l / self.wvs[-1]) ** 3 * (
                                (_l / self.wvs[-1]) ** (-1.32) -
                                (1 + z) ** (-1.32)
                            ) + 0.023 *
                            ((_l / self.wvs[-1]) ** 1.68 - (1 + z) ** 1.68)
                        )
                    ),
                )
                continue

        return Expteff


def _pow(a, b):
    """C-like power, a**b"""
    return a**b
