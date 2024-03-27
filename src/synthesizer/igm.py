"""A module containing Intergalactic Medium absorption models.

Each class in this module provides tranmission curves defining the absorption
due to the intergalactic medium.

Example usage:

    igm = Inoue14()
    z = 2.0
    lam = np.linspace(1000, 2000, 1000)
    transmission = igm.T(z, lam)
"""

import os
from typing import List

import numpy as np
from numpy.typing import NDArray

from . import __file__ as filepath

__all__: List[str] = ["Inoue14", "Madau96"]


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

    scale_tau: float
    name: str
    lam: NDArray[np.float64]
    ALAF1: NDArray[np.float64]
    ALAF2: NDArray[np.float64]
    ALAF3: NDArray[np.float64]
    ADLA1: NDArray[np.float64]
    ADLA2: NDArray[np.float64]

    def __init__(self, scale_tau: float = 1.0) -> None:
        """
        Initialize the Inoue14 model.

        Args:
            scale_tau: Scaling factor for the optical depth.
        """
        self._load_data()
        self.scale_tau = scale_tau
        self.name = "Inoue14"

    def _load_data(self) -> bool:
        """TODO: doc string"""
        path: str = os.path.join(os.path.dirname(filepath), "data")

        LAF_file: str = os.path.join(path, "LAFcoeff.txt")
        DLA_file: str = os.path.join(path, "DLAcoeff.txt")

        data_laf: NDArray[np.float64] = np.loadtxt(LAF_file, unpack=True)
        _, lam, ALAF1, ALAF2, ALAF3 = data_laf
        self.lam = lam[:, np.newaxis]
        self.ALAF1 = ALAF1[:, np.newaxis]
        self.ALAF2 = ALAF2[:, np.newaxis]
        self.ALAF3 = ALAF3[:, np.newaxis]

        data_dla: NDArray[np.float64] = np.loadtxt(DLA_file, unpack=True)
        _, _, ADLA1, ADLA2 = data_dla
        self.ADLA1 = ADLA1[:, np.newaxis]
        self.ADLA2 = ADLA2[:, np.newaxis]

        return True

    def tLSLAF(
        self,
        zS: float,
        lobs: NDArray[np.float64],
    ) -> np.float64:
        """
        Lyman series, Lyman-alpha forest
        """
        z1LAF: float = 1.2
        z2LAF: float = 4.7

        l2: NDArray[np.float64] = self.lam  # [:, np.newaxis]

        x0: NDArray[np.bool_] = lobs < l2 * (1 + zS)
        x1: NDArray[np.bool_] = x0 & (lobs < l2 * (1 + z1LAF))
        x2: NDArray[np.bool_] = x0 & (
            (lobs >= l2 * (1 + z1LAF)) & (lobs < l2 * (1 + z2LAF))
        )
        x3: NDArray[np.bool_] = x0 & (lobs >= l2 * (1 + z2LAF))

        tLSLAF_value: NDArray[np.float64] = np.zeros_like(lobs * l2)
        tLSLAF_value[x1] += ((self.ALAF1 / l2**1.2) * lobs**1.2)[x1]
        tLSLAF_value[x2] += ((self.ALAF2 / l2**3.7) * lobs**3.7)[x2]
        tLSLAF_value[x3] += ((self.ALAF3 / l2**5.5) * lobs**5.5)[x3]

        tlslaf_sum: np.float64 = tLSLAF_value.sum(axis=0)

        return tlslaf_sum

    def tLSDLA(
        self,
        zS: float,
        lobs: NDArray[np.float64],
    ) -> np.float64:
        """
        Lyman Series, DLA
        """
        z1DLA: float = 2.0

        l2: NDArray[np.float64] = self.lam  # [:, np.newaxis]
        tLSDLA_value: NDArray[np.float64] = np.zeros_like(lobs * l2)

        x0: NDArray[np.bool_] = (lobs < l2 * (1 + zS)) & (
            lobs < l2 * (1.0 + z1DLA)
        )
        x1: NDArray[np.bool_] = (lobs < l2 * (1 + zS)) & ~(
            lobs < l2 * (1.0 + z1DLA)
        )

        tLSDLA_value[x0] += ((self.ADLA1 / l2**2) * lobs**2)[x0]
        tLSDLA_value[x1] += ((self.ADLA2 / l2**3) * lobs**3)[x1]

        tlsdla_sum: np.float64 = tLSDLA_value.sum(axis=0)

        return tlsdla_sum

    def tLCDLA(
        self,
        zS: float,
        lobs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Lyman continuum, DLA
        """
        z1DLA: float = 2.0
        lamL: float = 911.8

        tLCDLA_value: NDArray[np.float64] = np.zeros_like(lobs)

        x0: NDArray[np.bool_] = lobs < lamL * (1.0 + zS)
        if zS < z1DLA:
            tLCDLA_value[x0] = (
                0.2113 * np.power(1.0 + zS, 2)
                - 0.07661
                * np.power(1.0 + zS, 2.3)
                * np.power(lobs[x0] / lamL, (-3e-1))
                - 0.1347 * np.power(lobs[x0] / lamL, 2)
            )
        else:
            x1: NDArray[np.bool_] = lobs >= lamL * (1.0 + z1DLA)

            tLCDLA_value[x0 & x1] = (
                0.04696 * np.power(1.0 + zS, 3)
                - 0.01779
                * np.power(1.0 + zS, 3.3)
                * np.power(lobs[x0 & x1] / lamL, (-3e-1))
                - 0.02916 * np.power(lobs[x0 & x1] / lamL, 3)
            )
            tLCDLA_value[x0 & ~x1] = (
                0.6340
                + 0.04696 * np.power(1.0 + zS, 3)
                - 0.01779
                * np.power(1.0 + zS, 3.3)
                * np.power(lobs[x0 & ~x1] / lamL, (-3e-1))
                - 0.1347 * np.power(lobs[x0 & ~x1] / lamL, 2)
                - 0.2905 * np.power(lobs[x0 & ~x1] / lamL, (-3e-1))
            )

        return tLCDLA_value

    def tLCLAF(
        self,
        zS: float,
        lobs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Lyman continuum, LAF
        """
        z1LAF = 1.2
        z2LAF = 4.7
        lamL = 911.8

        tLCLAF_value: NDArray[np.float64] = np.zeros_like(lobs)

        x0: NDArray[np.bool_] = lobs < lamL * (1.0 + zS)

        if zS < z1LAF:
            tLCLAF_value[x0] = 0.3248 * (
                np.power(lobs[x0] / lamL, 1.2)
                - np.power(1.0 + zS, -9e-1) * np.power(lobs[x0] / lamL, 2.1)
            )
        elif zS < z2LAF:
            x1: NDArray[np.bool_] = lobs >= lamL * (1 + z1LAF)
            tLCLAF_value[x0 & x1] = 2.545e-2 * (
                np.power(1.0 + zS, 1.6) * np.power(lobs[x0 & x1] / lamL, 2.1)
                - np.power(lobs[x0 & x1] / lamL, 3.7)
            )
            tLCLAF_value[x0 & ~x1] = (
                2.545e-2
                * np.power(1.0 + zS, 1.6)
                * np.power(lobs[x0 & ~x1] / lamL, 2.1)
                + 0.3248 * np.power(lobs[x0 & ~x1] / lamL, 1.2)
                - 0.2496 * np.power(lobs[x0 & ~x1] / lamL, 2.1)
            )
        else:
            x1_: NDArray[np.bool_] = lobs > lamL * (1.0 + z2LAF)
            x2: NDArray[np.bool_] = (lobs >= lamL * (1.0 + z1LAF)) & (
                lobs < lamL * (1.0 + z2LAF)
            )
            x3: NDArray[np.bool_] = lobs < lamL * (1.0 + z1LAF)

            tLCLAF_value[x0 & x1_] = 5.221e-4 * (
                np.power(1.0 + zS, 3.4) * np.power(lobs[x0 & x1_] / lamL, 2.1)
                - np.power(lobs[x0 & x1_] / lamL, 5.5)
            )
            tLCLAF_value[x0 & x2] = (
                5.221e-4
                * np.power(1.0 + zS, 3.4)
                * np.power(lobs[x0 & x2] / lamL, 2.1)
                + 0.2182 * np.power(lobs[x0 & x2] / lamL, 2.1)
                - 2.545e-2 * np.power(lobs[x0 & x2] / lamL, 3.7)
            )
            tLCLAF_value[x0 & x3] = (
                5.221e-4
                * np.power(1.0 + zS, 3.4)
                * np.power(lobs[x0 & x3] / lamL, 2.1)
                + 0.3248 * np.power(lobs[x0 & x3] / lamL, 1.2)
                - 3.140e-2 * np.power(lobs[x0 & x3] / lamL, 2.1)
            )

        return tLCLAF_value

    def tau(
        self,
        z: float,
        lobs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Get full Inoue IGM absorption.

        Args:
            z: Redshift to evaluate IGM absorption

            lobs: Observed-frame wavelength(s) in Angstroms.

        Returns:
            IGM absorption

        """
        tau_LS: np.float64 = self.tLSLAF(z, lobs) + self.tLSDLA(z, lobs)
        tau_LC: NDArray[np.float64] = self.tLCLAF(z, lobs) + self.tLCDLA(
            z, lobs
        )

        # Upturn at short wavelengths, low-z
        # k = 1./100
        # l0 = 600-6/k
        # clip = lobs/(1+z) < 600.
        # tau_clip = 100*(1-1./(1+np.exp(-k*(lobs/(1+z)-l0))))
        tau_clip: float = 0.0

        return self.scale_tau * (tau_LC + tau_LS + tau_clip)

    def T(
        self,
        z: float,
        lobs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute the transmission curve.
        """
        tau: NDArray[np.float64] = self.tau(z, lobs)
        t: NDArray[np.float64] = np.exp(-tau)

        t[t != t] = 0.0  # squash NaNs
        t[t > 1] = 1

        return t


class Madau96:
    wvs: List[float]
    a: List[float]
    name: str

    def __init__(self) -> None:
        self.wvs = [1216.0, 1026.0, 973.0, 950.0]
        self.a = [0.0036, 0.0017, 0.0012, 0.00093]
        self.name = "Madau96"

    def T(
        self,
        z: float,
        lobs: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        Expteff: NDArray[np.float64] = np.array([])
        for _l in lobs:
            if _l > self.wvs[0] * (1 + z):
                Expteff = np.append(Expteff, 1)
                continue

            if _l <= self.wvs[-1] * (1 + z) - 1500:
                Expteff = np.append(Expteff, 0)
                continue

            teff: float = 0
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
                            teff
                            + 0.25
                            * (_l / self.wvs[-1]) ** 3
                            * ((1 + z) ** 0.46 - (_l / self.wvs[-1]) ** 0.46)
                            + 9.4
                            * (_l / self.wvs[-1]) ** 1.5
                            * ((1 + z) ** 0.18 - (_l / self.wvs[-1]) ** 0.18)
                            - 0.7
                            * (_l / self.wvs[-1]) ** 3
                            * (
                                (_l / self.wvs[-1]) ** (-1.32)
                                - (1 + z) ** (-1.32)
                            )
                            + 0.023
                            * ((_l / self.wvs[-1]) ** 1.68 - (1 + z) ** 1.68)
                        )
                    ),
                )
                continue

        return Expteff
