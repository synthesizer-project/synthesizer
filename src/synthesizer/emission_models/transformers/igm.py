"""A module for computing Intergalactic Medium (IGM) absorption.

This module contains classes for computing IGM absorption from Inoue et al.
(2014) and Madau et al. (1996) models.

These are used when observer frame fluxes are computed using Sed.get_fnu(...)
and are not designed to be used directly by the user.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from unyt import angstrom

from synthesizer.emission_models.transformers.transformer import Transformer
from synthesizer.exceptions import UnimplementedFunctionality
from synthesizer.units import accepts

# Define the path to the data files
filepath = os.path.abspath(__file__)


def sigmoid(x, A, a, c):
    """Sigmoid function centered at x=6.

    Args:
        x (float): Input value (typically redshift).
        A (float): Amplitude parameter.
        a (float): Slope parameter.
        c (float): Offset parameter.

    Returns:
        float: Sigmoid function value.
    """
    return A / (1 + np.exp(-a * (x - 6))) + c


def sigma_a(nu_rest):
    """Lyα absorption cross section for the restframe frequency.

    Based on Totani+06, eqn (1) for CGM damping wing calculation.

    Args:
        nu_rest (float): Rest-frame frequency in Hz.

    Returns:
        float: Lyα absorption cross section at the restframe frequency [cm²].
    """
    Lam_a = 6.255486e8  # Hz
    nu_lya = 2.46607e15  # Hz
    C = 6.9029528e22  # Constant factor [Ang²/s²]

    sig = (
        C
        * (nu_rest / nu_lya) ** 4
        / (
            4 * np.pi**2 * (nu_rest - nu_lya) ** 2
            + Lam_a**2 * (nu_rest / nu_lya) ** 6 / 4
        )
    )

    s = sig * 1e-16  # convert Å⁻² to cm⁻²

    return s

__all__ = ["Inoue14", "Madau96", "Asada24"]


class IGMBase(Transformer):
    """Base class for IGM absorption models.

    Attributes:
        name (str): Name of the model.
    """

    def __init__(self, name):
        """Initialize the IGMBase class."""
        self.name = name

        # Initialize the base class
        Transformer.__init__(self, required_params=("redshift", "obslam"))

    def get_transmission(self, redshift, lam_obs):
        """Compute the IGM transmission.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: IGM transmission.
        """
        raise UnimplementedFunctionality(
            "get_transmission() must be implemented in a subclass."
        )

    def plot_transmission(
        self,
        redshift,
        lam_obs,
        show=False,
        fig=None,
        ax=None,
        figsize=(8, 6),
    ):
        """Plot the IGM transmission.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            lam_obs (array): Observed-frame wavelengths in Angstroms.
            show (bool): Whether to show the plot.
            fig (matplotlib.figure.Figure): Figure to plot on.
            ax (matplotlib.axes.Axes): Axes to plot on.
            figsize (tuple): Figure size.

        Returns:
            tuple: Figure and Axes objects.
        """
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        # Compute the transmission
        transmission = self.get_transmission(redshift, lam_obs)

        # Plot the transmission
        ax.plot(lam_obs, transmission, label=self.name)

        # Set the plot labels
        ax.set_xlabel(r"Wavelength ($\AA$)")
        ax.set_ylabel("Transmission")
        ax.legend()

        if show:
            plt.show()

        return fig, ax

    def _transform(self, emission, emitter, model, mask=None, lam_mask=None):
        """Apply the IGM to either a Line or Sed object.

        Args:
            emission (Line/Sed): The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy): The object emitting the
                emission.
            model (EmissionModel): The emission model generating the emission.
            mask (np.ndarray): The mask to apply to the emission.
            lam_mask (np.ndarray): The wavelength mask to apply to the
                emission.

        Returns:
            Line/Sed: The transformed emission.
        """
        # Extract the required parameters
        params = self._extract_params(model, emission, emitter)

        # Compute the transmission
        transmission = self.get_transmission(
            params["redshift"],
            params["obslam"],
        )

        # Masking is not supported for IGM absorption (it also doesn't really
        # make sense to mask the IGM absorption)
        if mask is not None:
            raise UnimplementedFunctionality(
                "Masking is not supported for IGM absorption. "
                "Are you sure you meant to do this?"
            )

        # Apply the transmission to the emission (here we can use the
        # overloaded multiplication operator regardless of the type of
        # emission object)
        return emission.scale(transmission, lam_mask=lam_mask, mask=mask)


class Inoue14(IGMBase):
    r"""IGM absorption from Inoue et al. (2014).

    Adapted from py-eazy.

    Attributes:
        scale_tau (float): Parameter multiplied to the IGM :math:`\tau` values
            (exponential in the linear absorption fraction). I.e.,
            :math:`f_{\\mathrm{igm}} = e^{-\\mathrm{scale_\tau} \tau}`.
        name (str): Name of the model.
        lam (array): Wavelengths for the model.
        alf1 (array): Coefficients for the Lyman-alpha forest.
        alf2 (array): Coefficients for the Lyman-alpha forest.
        alf3 (array): Coefficients for the Lyman-alpha forest.
        adla1 (array): Coefficients for the Damped Lyman-alpha absorption.
        adla2 (array): Coefficients for the Damped Lyman-alpha absorption.
    """

    def __init__(self, scale_tau=1.0):
        """Initialize the Inoue14 class with a scaling factor for tau."""
        # Initialize the base class
        IGMBase.__init__(self, "Inoue14")

        # Prepare attributes that will be loaded from data files
        self.lam = None
        self.alf1 = None
        self.alf2 = None
        self.alf3 = None
        self.adla1 = None
        self.adla2 = None

        # Load the data
        self._load_data()

        # Set the scale factor for the IGM absorption
        self.scale_tau = scale_tau

    def _load_data(self):
        """Load the coefficient data.

        This will load the Lyman-alpha forest (LAF) and Damped
        Lyman-alpha (DLA) coefficients from the data files.
        """
        data_path = os.path.join(os.path.dirname(filepath), "../../data")

        # Load LAF coefficients
        laf_file = os.path.join(data_path, "LAFcoeff.txt")
        data = np.loadtxt(laf_file, unpack=True)
        _, lam, alf1, alf2, alf3 = data
        self.lam = lam[:, np.newaxis] * angstrom
        self.alf1 = alf1[:, np.newaxis]
        self.alf2 = alf2[:, np.newaxis]
        self.alf3 = alf3[:, np.newaxis]

        # Load DLA coefficients
        dla_file = os.path.join(data_path, "DLAcoeff.txt")
        data = np.loadtxt(dla_file, unpack=True)
        _, lam, adla1, adla2 = data
        self.adla1 = adla1[:, np.newaxis]
        self.adla2 = adla2[:, np.newaxis]

        return True

    @accepts(lam_obs=angstrom)
    def tau_laf(self, redshift, lam_obs):
        """Compute the Lyman series and Lyman-alpha forest optical depth.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to the Lyman-alpha forest.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value
        lam = self.lam.value

        z1_laf = 1.2
        z2_laf = 4.7

        tau_laf_value = np.zeros_like(lam_obs * lam).T

        # Conditions based on observed lam and redshift
        cond0 = lam_obs < lam * (1 + redshift)
        cond1 = cond0 & (lam_obs < lam * (1 + z1_laf))
        cond2 = cond0 & (
            (lam_obs >= lam * (1 + z1_laf)) & (lam_obs < lam * (1 + z2_laf))
        )
        cond3 = cond0 & (lam_obs >= lam * (1 + z2_laf))

        tau_laf_value = np.zeros_like(lam_obs * lam)
        tau_laf_value[cond1] += ((self.alf1 / lam**1.2) * lam_obs**1.2)[cond1]
        tau_laf_value[cond2] += ((self.alf2 / lam**3.7) * lam_obs**3.7)[cond2]
        tau_laf_value[cond3] += ((self.alf3 / lam**5.5) * lam_obs**5.5)[cond3]

        return tau_laf_value.sum(axis=0)

    @accepts(lam_obs=angstrom)
    def tau_dla(self, redshift, lam_obs):
        """Compute the Lyman series and Damped Lyman-alpha (DLA) optical depth.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to DLA.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value
        lam = self.lam.value

        z1_dla = 2.0

        tau_dla_value = np.zeros_like(lam_obs * lam)

        # Conditions based on observed wavelength and redshift
        cond0 = (lam_obs < lam * (1 + redshift)) & (
            lam_obs < lam * (1.0 + z1_dla)
        )
        cond1 = (lam_obs < lam * (1 + redshift)) & ~(
            lam_obs < lam * (1.0 + z1_dla)
        )

        tau_dla_value[cond0] += ((self.adla1 / lam**2) * lam_obs**2)[cond0]
        tau_dla_value[cond1] += ((self.adla2 / lam**3) * lam_obs**3)[cond1]

        return tau_dla_value.sum(axis=0)

    @accepts(lam_obs=angstrom)
    def tau_lc_dla(self, redshift, lam_obs):
        """Compute the Lyman continuum optical depth for DLA.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to Lyman continuum for DLA.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value

        z1_dla = 2.0
        lam_l = 911.8

        tau_lc_dla_value = np.zeros_like(lam_obs)

        cond0 = lam_obs < lam_l * (1.0 + redshift)
        if redshift < z1_dla:
            tau_lc_dla_value[cond0] = (
                0.2113 * (1.0 + redshift) ** 2
                - 0.07661
                * (1.0 + redshift) ** 2.3
                * (lam_obs[cond0] / lam_l) ** (-0.3)
                - 0.1347 * (lam_obs[cond0] / lam_l) ** 2
            )
        else:
            cond1 = lam_obs >= lam_l * (1.0 + z1_dla)

            tau_lc_dla_value[cond0 & cond1] = (
                0.04696 * (1.0 + redshift) ** 3
                - 0.01779
                * (1.0 + redshift) ** 3.3
                * (lam_obs[cond0 & cond1] / lam_l) ** (-0.3)
                - 0.02916 * (lam_obs[cond0 & cond1] / lam_l) ** 3
            )
            tau_lc_dla_value[cond0 & ~cond1] = (
                0.6340
                + 0.04696 * (1.0 + redshift) ** 3
                - 0.01779
                * (1.0 + redshift) ** 3.3
                * (lam_obs[cond0 & ~cond1] / lam_l) ** (-0.3)
                - 0.1347 * (lam_obs[cond0 & ~cond1] / lam_l) ** 2
                - 0.2905 * (lam_obs[cond0 & ~cond1] / lam_l) ** (-0.3)
            )

        return tau_lc_dla_value

    @accepts(lam_obs=angstrom)
    def tau_lc_laf(self, redshift, lam_obs):
        """Compute the Lyman continuum optical depth for LAF.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to Lyman continuum for LAF.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value

        z1_laf = 1.2
        z2_laf = 4.7
        lam_l = 911.8

        tau_lc_laf_value = np.zeros_like(lam_obs)

        cond0 = lam_obs < lam_l * (1.0 + redshift)

        if redshift < z1_laf:
            tau_lc_laf_value[cond0] = 0.3248 * (
                (lam_obs[cond0] / lam_l) ** 1.2
                - (1.0 + redshift) ** -0.9 * (lam_obs[cond0] / lam_l) ** 2.1
            )
        elif redshift < z2_laf:
            cond1 = lam_obs >= lam_l * (1 + z1_laf)
            tau_lc_laf_value[cond0 & cond1] = 2.545e-2 * (
                (1.0 + redshift) ** 1.6
                * (lam_obs[cond0 & cond1] / lam_l) ** 2.1
                - (lam_obs[cond0 & cond1] / lam_l) ** 3.7
            )
            tau_lc_laf_value[cond0 & ~cond1] = (
                2.545e-2
                * (1.0 + redshift) ** 1.6
                * (lam_obs[cond0 & ~cond1] / lam_l) ** 2.1
                + 0.3248 * (lam_obs[cond0 & ~cond1] / lam_l) ** 1.2
                - 0.2496 * (lam_obs[cond0 & ~cond1] / lam_l) ** 2.1
            )
        else:
            cond1 = lam_obs > lam_l * (1.0 + z2_laf)
            cond2 = (lam_obs >= lam_l * (1.0 + z1_laf)) & (
                lam_obs < lam_l * (1.0 + z2_laf)
            )
            cond3 = lam_obs < lam_l * (1.0 + z1_laf)

            tau_lc_laf_value[cond0 & cond1] = 5.221e-4 * (
                (1.0 + redshift) ** 3.4
                * (lam_obs[cond0 & cond1] / lam_l) ** 2.1
                - (lam_obs[cond0 & cond1] / lam_l) ** 5.5
            )
            tau_lc_laf_value[cond0 & cond2] = (
                5.221e-4
                * (1.0 + redshift) ** 3.4
                * (lam_obs[cond0 & cond2] / lam_l) ** 2.1
                + 0.2182 * (lam_obs[cond0 & cond2] / lam_l) ** 2.1
                - 2.545e-2 * (lam_obs[cond0 & cond2] / lam_l) ** 3.7
            )
            tau_lc_laf_value[cond0 & cond3] = (
                5.221e-4
                * (1.0 + redshift) ** 3.4
                * (lam_obs[cond0 & cond3] / lam_l) ** 2.1
                + 0.3248 * (lam_obs[cond0 & cond3] / lam_l) ** 1.2
                - 3.140e-2 * (lam_obs[cond0 & cond3] / lam_l) ** 2.1
            )

        return tau_lc_laf_value

    def tau(self, redshift, lam_obs):
        """Compute the total IGM optical depth.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Total IGM absorption optical depth.
        """
        tau_ls = self.tau_laf(redshift, lam_obs) + self.tau_dla(
            redshift, lam_obs
        )
        tau_lc = self.tau_lc_laf(redshift, lam_obs) + self.tau_lc_dla(
            redshift, lam_obs
        )

        # Upturn at short wavelengths, low-z
        # k = 1./100
        # l0 = 600-6/k
        # clip = lam_obs/(1+redshift) < 600.
        # tau_clip = 100*(1-1./(1+np.exp(-k*(lam_obs/(1+redshift)-l0))))
        tau_clip = 0.0

        return self.scale_tau * (tau_lc + tau_ls + tau_clip)

    def get_transmission(self, redshift, lam_obs):
        """Compute the IGM transmission.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: IGM transmission.
        """
        tau = self.tau(redshift, lam_obs)
        transmission = np.exp(-tau)

        # Handle NaNs and values greater than 1
        transmission[transmission != transmission] = 0.0  # squash NaNs
        transmission[transmission > 1] = 1

        return transmission


class Madau96(IGMBase):
    """IGM absorption from Madau et al. (1996).

    Attributes:
        lams (list): List of wavelengths for the model.
        coefficients (list): List of coefficients for the model.
        name (str): Name of the model.
    """

    def __init__(self):
        """Initialize the Madau96 class."""
        # Initialize the base class
        IGMBase.__init__(self, "Madau96")

        self.lams = [1216.0, 1026.0, 973.0, 950.0]
        self.coefficients = [0.0036, 0.0017, 0.0012, 0.00093]

    @accepts(lam_obs=angstrom)
    def get_transmission(self, redshift, lam_obs):
        """Compute the IGM transmission.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: IGM transmission.
        """
        # Strip off units for the following calculations (we know the units
        # are Angstroms from the decorator)
        lam_obs = lam_obs.value

        exp_teff = np.array([])
        for wl in lam_obs:
            if wl > self.lams[0] * (1 + redshift):
                exp_teff = np.append(exp_teff, 1)
                continue

            if wl <= self.lams[-1] * (1 + redshift) - 1500:
                exp_teff = np.append(exp_teff, 0)
                continue

            teff = 0
            for i in range(0, len(self.lams) - 1, 1):
                teff += self.coefficients[i] * (wl / self.lams[i]) ** 3.46
                if (
                    self.lams[i + 1] * (1 + redshift)
                    < wl
                    <= self.lams[i] * (1 + redshift)
                ):
                    exp_teff = np.append(exp_teff, np.exp(-teff))
                    continue

            if wl <= self.lams[-1] * (1 + redshift):
                exp_teff = np.append(
                    exp_teff,
                    np.exp(
                        -(
                            teff
                            + 0.25
                            * (wl / self.lams[-1]) ** 3
                            * (
                                (1 + redshift) ** 0.46
                                - (wl / self.lams[-1]) ** 0.46
                            )
                            + 9.4
                            * (wl / self.lams[-1]) ** 1.5
                            * (
                                (1 + redshift) ** 0.18
                                - (wl / self.lams[-1]) ** 0.18
                            )
                            - 0.7
                            * (wl / self.lams[-1]) ** 3
                            * (
                                (wl / self.lams[-1]) ** (-1.32)
                                - (1 + redshift) ** (-1.32)
                            )
                            + 0.023
                            * (
                                (wl / self.lams[-1]) ** 1.68
                                - (1 + redshift) ** 1.68
                            )
                        )
                    ),
                )
                continue

        return exp_teff


class Asada24(IGMBase):
    r"""IGM+CGM absorption from Asada et al. (2024, in prep).

    The IGM model is from Inoue+ (2014), with additional Lyα damping
    absorption at z>6 as described in Asada+24. This model includes both
    Intergalactic Medium (IGM) and Circumgalactic Medium (CGM) effects.

    Attributes:
        sigmoid_params (tuple): Parameters that control the redshift evolution
            of the CGM HI gas column density (A, a, c). Default values are
            from Asada et al. (2024): (3.5918, 1.8414, 18.001).
        scale_tau (float): Scalar multiplied to tau_igm. I.e.,
            :math:`f_{\\mathrm{igm}} = e^{-\\mathrm{scale_\\tau} \\tau}`.
        add_cgm (bool): Add the additional LyA damping absorption at z>6 as
            described in Asada+24. If False, the transmission will be
            identical to Inoue+2014.
        name (str): Name of the model.
        lam (array): Wavelengths for the model.
        alf1 (array): Coefficients for the Lyman-alpha forest.
        alf2 (array): Coefficients for the Lyman-alpha forest.
        alf3 (array): Coefficients for the Lyman-alpha forest.
        adla1 (array): Coefficients for the Damped Lyman-alpha absorption.
        adla2 (array): Coefficients for the Damped Lyman-alpha absorption.
    """

    def __init__(
        self,
        sigmoid_params=(3.5918, 1.8414, 18.001),
        scale_tau=1.0,
        add_cgm=True,
    ):
        """Initialize the Asada24 class.

        Args:
            sigmoid_params (tuple): Parameters that control the redshift
                evolution of the CGM HI gas column density (A, a, c).
                Default values are from Asada et al. (2024).
            scale_tau (float): Scalar multiplied to tau_igm.
            add_cgm (bool): Add the additional LyA damping absorption at z>6.
                If False, the transmission will be identical to Inoue+2014.
        """
        # Initialize the base class
        IGMBase.__init__(self, "Asada24")

        # Prepare attributes that will be loaded from data files
        self.lam = None
        self.alf1 = None
        self.alf2 = None
        self.alf3 = None
        self.adla1 = None
        self.adla2 = None

        # Load the data
        self._load_data()

        # Set the parameters
        self.sigmoid_params = sigmoid_params
        self.scale_tau = scale_tau
        self.add_cgm = add_cgm

    def _load_data(self):
        """Load the coefficient data.

        This will load the Lyman-alpha forest (LAF) and Damped
        Lyman-alpha (DLA) coefficients from the data files.
        """
        data_path = os.path.join(os.path.dirname(filepath), "../../data")

        # Load LAF coefficients
        laf_file = os.path.join(data_path, "LAFcoeff.txt")
        data = np.loadtxt(laf_file, unpack=True)
        _, lam, alf1, alf2, alf3 = data
        self.lam = lam[:, np.newaxis] * angstrom
        self.alf1 = alf1[:, np.newaxis]
        self.alf2 = alf2[:, np.newaxis]
        self.alf3 = alf3[:, np.newaxis]

        # Load DLA coefficients
        dla_file = os.path.join(data_path, "DLAcoeff.txt")
        data = np.loadtxt(dla_file, unpack=True)
        _, lam, adla1, adla2 = data
        self.adla1 = adla1[:, np.newaxis]
        self.adla2 = adla2[:, np.newaxis]

        return True

    @accepts(lam_obs=angstrom)
    def tau_laf(self, redshift, lam_obs):
        """Compute the Lyman series and Lyman-alpha forest optical depth.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to the Lyman-alpha forest.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value
        lam = self.lam.value

        z1_laf = 1.2
        z2_laf = 4.7

        tau_laf_value = np.zeros_like(lam_obs * lam).T

        # Conditions based on observed lam and redshift
        cond0 = lam_obs < lam * (1 + redshift)
        cond1 = cond0 & (lam_obs < lam * (1 + z1_laf))
        cond2 = cond0 & (
            (lam_obs >= lam * (1 + z1_laf)) & (lam_obs < lam * (1 + z2_laf))
        )
        cond3 = cond0 & (lam_obs >= lam * (1 + z2_laf))

        tau_laf_value = np.zeros_like(lam_obs * lam)
        tau_laf_value[cond1] += ((self.alf1 / lam**1.2) * lam_obs**1.2)[cond1]
        tau_laf_value[cond2] += ((self.alf2 / lam**3.7) * lam_obs**3.7)[cond2]
        tau_laf_value[cond3] += ((self.alf3 / lam**5.5) * lam_obs**5.5)[cond3]

        return tau_laf_value.sum(axis=0)

    @accepts(lam_obs=angstrom)
    def tau_dla(self, redshift, lam_obs):
        """Compute the Lyman series and Damped Lyman-alpha (DLA) optical depth.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to DLA.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value
        lam = self.lam.value

        z1_dla = 2.0

        tau_dla_value = np.zeros_like(lam_obs * lam)

        # Conditions based on observed wavelength and redshift
        cond0 = (lam_obs < lam * (1 + redshift)) & (
            lam_obs < lam * (1.0 + z1_dla)
        )
        cond1 = (lam_obs < lam * (1 + redshift)) & ~(
            lam_obs < lam * (1.0 + z1_dla)
        )

        tau_dla_value[cond0] += ((self.adla1 / lam**2) * lam_obs**2)[cond0]
        tau_dla_value[cond1] += ((self.adla2 / lam**3) * lam_obs**3)[cond1]

        return tau_dla_value.sum(axis=0)

    @accepts(lam_obs=angstrom)
    def tau_lc_dla(self, redshift, lam_obs):
        """Compute the Lyman continuum optical depth for DLA.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to Lyman continuum for DLA.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value

        z1_dla = 2.0
        lam_l = 911.8

        tau_lc_dla_value = np.zeros_like(lam_obs)

        cond0 = lam_obs < lam_l * (1.0 + redshift)
        if redshift < z1_dla:
            tau_lc_dla_value[cond0] = (
                0.2113 * (1.0 + redshift) ** 2
                - 0.07661
                * (1.0 + redshift) ** 2.3
                * (lam_obs[cond0] / lam_l) ** (-0.3)
                - 0.1347 * (lam_obs[cond0] / lam_l) ** 2
            )
        else:
            cond1 = lam_obs >= lam_l * (1.0 + z1_dla)

            tau_lc_dla_value[cond0 & cond1] = (
                0.04696 * (1.0 + redshift) ** 3
                - 0.01779
                * (1.0 + redshift) ** 3.3
                * (lam_obs[cond0 & cond1] / lam_l) ** (-0.3)
                - 0.02916 * (lam_obs[cond0 & cond1] / lam_l) ** 3
            )
            tau_lc_dla_value[cond0 & ~cond1] = (
                0.6340
                + 0.04696 * (1.0 + redshift) ** 3
                - 0.01779
                * (1.0 + redshift) ** 3.3
                * (lam_obs[cond0 & ~cond1] / lam_l) ** (-0.3)
                - 0.1347 * (lam_obs[cond0 & ~cond1] / lam_l) ** 2
                - 0.2905 * (lam_obs[cond0 & ~cond1] / lam_l) ** (-0.3)
            )

        return tau_lc_dla_value

    @accepts(lam_obs=angstrom)
    def tau_lc_laf(self, redshift, lam_obs):
        """Compute the Lyman continuum optical depth for LAF.

        Args:
            redshift (float): Source redshift.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Optical depth due to Lyman continuum for LAF.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value

        z1_laf = 1.2
        z2_laf = 4.7
        lam_l = 911.8

        tau_lc_laf_value = np.zeros_like(lam_obs)

        cond0 = lam_obs < lam_l * (1.0 + redshift)

        if redshift < z1_laf:
            tau_lc_laf_value[cond0] = 0.3248 * (
                (lam_obs[cond0] / lam_l) ** 1.2
                - (1.0 + redshift) ** (-0.9) * (lam_obs[cond0] / lam_l) ** 2.1
            )
        elif redshift < z2_laf:
            cond1 = lam_obs >= lam_l * (1 + z1_laf)
            tau_lc_laf_value[cond0 & cond1] = 2.545e-2 * (
                (1.0 + redshift) ** 1.6
                * (lam_obs[cond0 & cond1] / lam_l) ** 2.1
                - (lam_obs[cond0 & cond1] / lam_l) ** 3.7
            )
            tau_lc_laf_value[cond0 & ~cond1] = (
                2.545e-2
                * (1.0 + redshift) ** 1.6
                * (lam_obs[cond0 & ~cond1] / lam_l) ** 2.1
                + 0.3248 * (lam_obs[cond0 & ~cond1] / lam_l) ** 1.2
                - 0.2496 * (lam_obs[cond0 & ~cond1] / lam_l) ** 2.1
            )
        else:
            cond1 = lam_obs > lam_l * (1.0 + z2_laf)
            cond2 = (lam_obs >= lam_l * (1.0 + z1_laf)) & (
                lam_obs < lam_l * (1.0 + z2_laf)
            )
            cond3 = lam_obs < lam_l * (1.0 + z1_laf)

            tau_lc_laf_value[cond0 & cond1] = 5.221e-4 * (
                (1.0 + redshift) ** 3.4
                * (lam_obs[cond0 & cond1] / lam_l) ** 2.1
                - (lam_obs[cond0 & cond1] / lam_l) ** 5.5
            )
            tau_lc_laf_value[cond0 & cond2] = (
                5.221e-4
                * (1.0 + redshift) ** 3.4
                * (lam_obs[cond0 & cond2] / lam_l) ** 2.1
                + 0.2182 * (lam_obs[cond0 & cond2] / lam_l) ** 2.1
                - 2.545e-2 * (lam_obs[cond0 & cond2] / lam_l) ** 3.7
            )
            tau_lc_laf_value[cond0 & cond3] = (
                5.221e-4
                * (1.0 + redshift) ** 3.4
                * (lam_obs[cond0 & cond3] / lam_l) ** 2.1
                + 0.3248 * (lam_obs[cond0 & cond3] / lam_l) ** 1.2
                - 3.140e-2 * (lam_obs[cond0 & cond3] / lam_l) ** 2.1
            )

        return tau_lc_laf_value

    def tau_igm(self, redshift, lam_obs):
        """Compute the total IGM optical depth.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: Total IGM absorption optical depth.
        """
        tau_ls = self.tau_laf(redshift, lam_obs) + self.tau_dla(
            redshift, lam_obs
        )
        tau_lc = self.tau_lc_laf(redshift, lam_obs) + self.tau_lc_dla(
            redshift, lam_obs
        )
        tau_clip = 0.0

        return self.scale_tau * (tau_lc + tau_ls + tau_clip)

    def lgNHI_z(self, redshift):
        """HI column density as a function of redshift.

        Calibrated in Asada+ 2024. Only valid at z>=6.

        Args:
            redshift (float): Redshift of the source.

        Returns:
            float: log10(HI column density [cm⁻²]).
        """
        lgN_HI = sigmoid(
            redshift,
            self.sigmoid_params[0],
            self.sigmoid_params[1],
            self.sigmoid_params[2],
        )

        return lgN_HI

    @accepts(lam_obs=angstrom)
    def tau_cgm(self, N_HI, redshift, lam_obs):
        """CGM optical depth given by Totani+06, eqn (1).

        Args:
            N_HI (float): HI column density [cm⁻²].
            redshift (float): Redshift of the source.
            lam_obs (array): Observed-frame wavelength array [Å].

        Returns:
            array: tau_CGM values.
        """
        # Strip units for the following calculations
        lam_obs = lam_obs.value

        lam_rest = lam_obs / (1 + redshift)
        nu_rest = 3e18 / lam_rest  # Convert Å to Hz

        tau = np.zeros(len(lam_obs))

        for i, nu in enumerate(nu_rest):
            tau[i] = N_HI * sigma_a(nu)

        return tau

    def get_transmission(self, redshift, lam_obs):
        """Compute the IGM+CGM transmission.

        Args:
            redshift (float): Redshift to evaluate IGM absorption.
            lam_obs (array): Observed-frame wavelengths in Angstroms.

        Returns:
            array: IGM+CGM transmission.
        """
        # Compute IGM transmission
        tau_igm_val = self.tau_igm(redshift, lam_obs)
        igmz = np.exp(-tau_igm_val)

        # Compute CGM transmission if requested
        if self.add_cgm:
            if redshift < 6:
                tau_C = 0.0
            else:
                NHI = 10 ** (self.lgNHI_z(redshift))
                tau_C = self.tau_cgm(NHI, redshift, lam_obs)

            cgmz = np.exp(-tau_C)
        else:
            cgmz = 1.0

        transmission = igmz * cgmz

        # Handle NaNs and values greater than 1
        transmission[transmission != transmission] = 0.0  # squash NaNs
        transmission[transmission > 1] = 1

        return transmission
