"""A module for computing Intergalactic Medium (IGM) absorption.

This module contains classes for computing IGM absorption from Inoue et al.
(2014) and Madau et al. (1996) models.

These are used when observer frame fluxes are computed using Sed.get_fnu(...)
and are not designed to be used directly by the user.
"""

import os

import numpy as np

from . import __file__ as filepath

__all__ = ["Inoue14", "Madau96"]


class Inoue14:
    """
    IGM absorption from Inoue et al. (2014).

    Attributes:
        scale_tau (float): Parameter multiplied to the IGM tau values
                           (exponential in the linear absorption fraction).
                           I.e., f_igm = e^(-scale_tau * tau).
        name (str): Name of the IGM model.
    """

    def __init__(self, scale_tau=1.0):
        """
        Initialize the Inoue14 class.

        Args:
            scale_tau (float): Parameter multiplied to the IGM tau values
                               (exponential in the linear absorption fraction).
                               I.e., f_igm = e^(-scale_tau * tau).
        """
        self.scale_tau = scale_tau
        self.name = "Inoue14"
        self._load_data()

    def _load_data(self):
        """Load the data required for calculations from text files."""
        path = os.path.join(os.path.dirname(filepath), "../../data")

        laf_file = os.path.join(path, "LAFcoeff.txt")
        dla_file = os.path.join(path, "DLAcoeff.txt")

        # Load Lyman-alpha forest coefficients
        laf_data = np.loadtxt(laf_file, unpack=True)
        _, self.lambda_laf, self.a_laf1, self.a_laf2, self.a_laf3 = laf_data

        # Load Damped Lyman-alpha coefficients
        dla_data = np.loadtxt(dla_file, unpack=True)
        _, self.lambda_dla, self.a_dla1, self.a_dla2 = dla_data

        self.lambda_laf = self.lambda_laf[:, np.newaxis]
        self.a_laf1 = self.a_laf1[:, np.newaxis]
        self.a_laf2 = self.a_laf2[:, np.newaxis]
        self.a_laf3 = self.a_laf3[:, np.newaxis]

        self.lambda_dla = self.lambda_dla[:, np.newaxis]
        self.a_dla1 = self.a_dla1[:, np.newaxis]
        self.a_dla2 = self.a_dla2[:, np.newaxis]

        return True

    def t_lyman_series_laf(self, redshift_source, obs_wavelength):
        """
        Lyman series, Lyman-alpha forest.

        Args:
            redshift_source (float): Redshift of the source.
            obs_wavelength (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Lyman-alpha forest absorption values.
        """
        z1_laf = 1.2
        z2_laf = 4.7

        # Initialize absorption array
        lambda_laf = self.lambda_laf
        absorption = np.zeros_like(
            obs_wavelength[:, np.newaxis] * lambda_laf.T
        )

        # Condition checks for different redshift ranges
        condition0 = obs_wavelength[:, np.newaxis] < lambda_laf * (
            1 + redshift_source
        )
        condition1 = condition0 & (
            obs_wavelength[:, np.newaxis] < lambda_laf * (1 + z1_laf)
        )
        condition2 = condition0 & (
            (obs_wavelength[:, np.newaxis] >= lambda_laf * (1 + z1_laf))
            & (obs_wavelength[:, np.newaxis] < lambda_laf * (1 + z2_laf))
        )
        condition3 = condition0 & (
            obs_wavelength[:, np.newaxis] >= lambda_laf * (1 + z2_laf)
        )

        # Calculate absorption values
        absorption[condition1] += (
            (self.a_laf1 / lambda_laf**1.2)
            * obs_wavelength[:, np.newaxis] ** 1.2
        )[condition1]
        absorption[condition2] += (
            (self.a_laf2 / lambda_laf**3.7)
            * obs_wavelength[:, np.newaxis] ** 3.7
        )[condition2]
        absorption[condition3] += (
            (self.a_laf3 / lambda_laf**5.5)
            * obs_wavelength[:, np.newaxis] ** 5.5
        )[condition3]

        return absorption.sum(axis=1)

    def t_lyman_series_dla(self, redshift_source, obs_wavelength):
        """
        Lyman Series, Damped Lyman-alpha absorption.

        Args:
            redshift_source (float): Redshift of the source.
            obs_wavelength (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Damped Lyman-alpha absorption values.
        """
        z1_dla = 2.0

        # Initialize absorption array
        lambda_dla = self.lambda_dla
        absorption = np.zeros_like(
            obs_wavelength[:, np.newaxis] * lambda_dla.T
        )

        # Condition checks for different redshift ranges
        condition0 = (
            obs_wavelength[:, np.newaxis] < lambda_dla * (1 + redshift_source)
        ) & (obs_wavelength[:, np.newaxis] < lambda_dla * (1.0 + z1_dla))
        condition1 = (
            obs_wavelength[:, np.newaxis] < lambda_dla * (1 + redshift_source)
        ) & ~(obs_wavelength[:, np.newaxis] < lambda_dla * (1.0 + z1_dla))

        # Calculate absorption values
        absorption[condition0] += (
            (self.a_dla1 / lambda_dla**2) * obs_wavelength[:, np.newaxis] ** 2
        )[condition0]
        absorption[condition1] += (
            (self.a_dla2 / lambda_dla**3) * obs_wavelength[:, np.newaxis] ** 3
        )[condition1]

        return absorption.sum(axis=1)

    def t_lyman_continuum_dla(self, redshift_source, obs_wavelength):
        """
        Lyman continuum, Damped Lyman-alpha absorption.

        Args:
            redshift_source (float): Redshift of the source.
            obs_wavelength (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Lyman continuum absorption values for DLA.
        """
        z1_dla = 2.0
        lambda_l = 911.8

        absorption = np.zeros_like(obs_wavelength)

        # Condition checks for different redshift ranges
        condition0 = obs_wavelength < lambda_l * (1.0 + redshift_source)
        if redshift_source < z1_dla:
            absorption[condition0] = (
                0.2113 * (1.0 + redshift_source) ** 2
                - 0.07661
                * (1.0 + redshift_source) ** 2.3
                * (obs_wavelength[condition0] / lambda_l) ** -0.3
                - 0.1347 * (obs_wavelength[condition0] / lambda_l) ** 2
            )
        else:
            condition1 = obs_wavelength >= lambda_l * (1.0 + z1_dla)
            absorption[condition0 & condition1] = (
                0.04696 * (1.0 + redshift_source) ** 3
                - 0.01779
                * (1.0 + redshift_source) ** 3.3
                * (obs_wavelength[condition0 & condition1] / lambda_l) ** -0.3
                - 0.02916
                * (obs_wavelength[condition0 & condition1] / lambda_l) ** 3
            )
            absorption[condition0 & ~condition1] = (
                0.6340
                + 0.04696 * (1.0 + redshift_source) ** 3
                - 0.01779
                * (1.0 + redshift_source) ** 3.3
                * (obs_wavelength[condition0 & ~condition1] / lambda_l) ** -0.3
                - 0.1347
                * (obs_wavelength[condition0 & ~condition1] / lambda_l) ** 2
                - 0.2905
                * (obs_wavelength[condition0 & ~condition1] / lambda_l) ** -0.3
            )

        return absorption

    def t_lyman_continuum_laf(self, redshift_source, obs_wavelength):
        """
        Lyman continuum, Lyman-alpha forest.

        Args:
            redshift_source (float): Redshift of the source.
            obs_wavelength (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Lyman continuum absorption values for LAF.
        """
        z1_laf = 1.2
        z2_laf = 4.7
        lambda_l = 911.8

        absorption = np.zeros_like(obs_wavelength)

        # Condition checks for different redshift ranges
        condition0 = obs_wavelength < lambda_l * (1.0 + redshift_source)

        if redshift_source < z1_laf:
            absorption[condition0] = 0.3248 * (
                (obs_wavelength[condition0] / lambda_l) ** 1.2
                - (1.0 + redshift_source) ** -0.9
                * (obs_wavelength[condition0] / lambda_l) ** 2.1
            )
        elif redshift_source < z2_laf:
            condition1 = obs_wavelength >= lambda_l * (1 + z1_laf)
            absorption[condition0 & condition1] = 2.545e-2 * (
                (1.0 + redshift_source) ** 1.6
                * (obs_wavelength[condition0 & condition1] / lambda_l) ** 2.1
                - (obs_wavelength[condition0 & condition1] / lambda_l) ** 3.7
            )
            absorption[condition0 & ~condition1] = (
                2.545e-2
                * (1.0 + redshift_source) ** 1.6
                * (obs_wavelength[condition0 & ~condition1] / lambda_l) ** 2.1
                + 0.3248
                * (obs_wavelength[condition0 & ~condition1] / lambda_l) ** 1.2
                - 0.2496
                * (obs_wavelength[condition0 & ~condition1] / lambda_l) ** 2.1
            )
        else:
            condition1 = obs_wavelength > lambda_l * (1.0 + z2_laf)
            condition2 = (obs_wavelength >= lambda_l * (1.0 + z1_laf)) & (
                obs_wavelength < lambda_l * (1.0 + z2_laf)
            )
            condition3 = obs_wavelength < lambda_l * (1.0 + z1_laf)

            absorption[condition0 & condition1] = 5.221e-4 * (
                (1.0 + redshift_source) ** 3.4
                * (obs_wavelength[condition0 & condition1] / lambda_l) ** 2.1
                - (obs_wavelength[condition0 & condition1] / lambda_l) ** 5.5
            )
            absorption[condition0 & condition2] = (
                5.221e-4
                * (1.0 + redshift_source) ** 3.4
                * (obs_wavelength[condition0 & condition2] / lambda_l) ** 2.1
                + 0.2182
                * (obs_wavelength[condition0 & condition2] / lambda_l) ** 2.1
                - 2.545e-2
                * (obs_wavelength[condition0 & condition2] / lambda_l) ** 3.7
            )
            absorption[condition0 & condition3] = (
                5.221e-4
                * (1.0 + redshift_source) ** 3.4
                * (obs_wavelength[condition0 & condition3] / lambda_l) ** 2.1
                + 0.3248
                * (obs_wavelength[condition0 & condition3] / lambda_l) ** 1.2
                - 3.140e-2
                * (obs_wavelength[condition0 & condition3] / lambda_l) ** 2.1
            )

        return absorption

    def tau(self, redshift, obs_wavelength):
        """
        Get full Inoue IGM absorption.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            obs_wavelength (array): Observed-frame wavelength(s) in Angstroms.

        Returns:
            array: IGM absorption.
        """
        tau_ls = self.t_lyman_series_laf(
            redshift, obs_wavelength
        ) + self.t_lyman_series_dla(redshift, obs_wavelength)
        tau_lc = self.t_lyman_continuum_laf(
            redshift, obs_wavelength
        ) + self.t_lyman_continuum_dla(redshift, obs_wavelength)

        # Upturn at short wavelengths, low-z (currently set to 0)
        tau_clip = 0.0

        return self.scale_tau * (tau_lc + tau_ls + tau_clip)

    def get_transmission(self, redshift, obs_wavelength):
        """
        Get transmission curve.

        Args:
            redshift (float): Redshift to evaluate transmission.
            obs_wavelength (array): Observed-frame wavelength(s) in Angstroms.

        Returns:
            array: Transmission values.
        """
        tau = self.tau(redshift, obs_wavelength)
        transmission = np.exp(-tau)

        # Handle NaNs and ensure transmission values are within [0, 1]
        transmission[np.isnan(transmission)] = 0.0
        transmission[transmission > 1] = 1

        return transmission


class Madau96:
    """
    IGM absorption from Madau et al. (1996).

    Attributes:
        wavelengths (array): Wavelengths of the transmission curve.
        coefficients (array): Coefficients of the transmission curve.
        name (str): Name of the model.
    """

    def __init__(self):
        """Initialize the Madau96 class."""
        self.wavelengths = [1216.0, 1026.0, 973.0, 950.0]
        self.coefficients = [0.0036, 0.0017, 0.0012, 0.00093]
        self.name = "Madau96"

    def get_transmission(self, redshift, obs_wavelength):
        """
        Get transmission curve.

        Args:
            redshift (float): Redshift to evaluate transmission.
            obs_wavelength (array): Observed-frame wavelength(s) in Angstroms.

        Returns:
            array: Transmission values.
        """
        exp_teff = np.array([])
        for wavelength in obs_wavelength:
            if wavelength > self.wavelengths[0] * (1 + redshift):
                exp_teff = np.append(exp_teff, 1)
                continue

            if wavelength <= self.wavelengths[-1] * (1 + redshift) - 1500:
                exp_teff = np.append(exp_teff, 0)
                continue

            teff = 0
            for i in range(len(self.wavelengths) - 1):
                teff += (
                    self.coefficients[i]
                    * (wavelength / self.wavelengths[i]) ** 3.46
                )
                if (
                    self.wavelengths[i + 1] * (1 + redshift)
                    < wavelength
                    <= self.wavelengths[i] * (1 + redshift)
                ):
                    exp_teff = np.append(exp_teff, np.exp(-teff))
                    continue

            if wavelength <= self.wavelengths[-1] * (1 + redshift):
                exp_teff = np.append(
                    exp_teff,
                    np.exp(
                        -(
                            teff
                            + 0.25
                            * (wavelength / self.wavelengths[-1]) ** 3
                            * (
                                (1 + redshift) ** 0.46
                                - (wavelength / self.wavelengths[-1]) ** 0.46
                            )
                            + 9.4
                            * (wavelength / self.wavelengths[-1]) ** 1.5
                            * (
                                (1 + redshift) ** 0.18
                                - (wavelength / self.wavelengths[-1]) ** 0.18
                            )
                            - 0.7
                            * (wavelength / self.wavelengths[-1]) ** 3
                            * (
                                (wavelength / self.wavelengths[-1]) ** -1.32
                                - (1 + redshift) ** -1.32
                            )
                            + 0.023
                            * (
                                (wavelength / self.wavelengths[-1]) ** 1.68
                                - (1 + redshift) ** 1.68
                            )
                        )
                    ),
                )

        return exp_teff
