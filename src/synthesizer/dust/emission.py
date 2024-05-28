"""Module containing dust emission functionality"""

from functools import partial

import numpy as np
from scipy import integrate
from scipy.optimize import fsolve
from unyt import (
    Angstrom,
    Hz,
    Lsun,
    Msun,
    accepts,
    c,
    erg,
    h,
    kb,
    s,
    um,
    unyt_array,
    unyt_quantity,
)
from unyt.dimensions import mass as mass_dim
from unyt.dimensions import temperature as temperature_dim

from synthesizer import exceptions
from synthesizer.sed import Sed
from synthesizer.utils import planck
from synthesizer.warnings import warn


class EmissionBase:
    """
    Dust emission base class for holding common methods.

    Attributes:
        temperature (float)
            The temperature of the dust.
    """

    def __init__(self, temperature):
        """
        Initialises the base class for dust emission models.

        Args:
            temperature (float)
                The temperature of the dust.
        """

        self.temperature = temperature

    def _lnu(self, *args):
        """
        A prototype private method used during integration. This should be
        overloaded by child classes!
        """
        raise exceptions.UnimplementedFunctionality(
            "EmissionBase should not be instantiated directly!"
            " Instead use one to child models (Blackbody, Greybody, Casey12)."
        )

    def normalisation(self):
        """
        Provide normalisation of _lnu by integrating the function from 8->1000
        um.
        """
        return integrate.quad(
            self._lnu,
            c / (1000 * um),
            c / (8 * um),
            full_output=False,
            limit=100,
        )[0]

    def get_spectra(self, _lam):
        """
        Returns the normalised lnu for the provided wavelength grid

        Args:
            _lam (float/array-like, float)
                    An array of wavelengths (expected in AA, global unit)

        """
        if isinstance(_lam, (unyt_quantity, unyt_array)):
            lam = _lam
        else:
            lam = _lam * Angstrom

        lnu = (erg / s / Hz) * self._lnu(c / lam).value / self.normalisation()

        sed = Sed(lam=lam, lnu=lnu)

        # normalise the spectrum
        sed._lnu /= sed.measure_bolometric_luminosity().value

        return sed


class Blackbody(EmissionBase):
    """
    A class to generate a blackbody emission spectrum.
    """

    @accepts(temperature=temperature_dim)
    def __init__(self, temperature):
        """
        A function to generate a simple blackbody spectrum.

        Args:
            temperature (unyt_array)
                The temperature of the dust.

        """

        EmissionBase.__init__(self, temperature)

    # @accepts(nu=1/time)
    def _lnu(self, nu):
        """
        Generate unnormalised spectrum for given frequency (nu) grid.

        Args:
            nu (unyt_array)
                The frequency at which to calculate lnu.

        Returns:
            unyt_array
                The unnormalised spectral luminosity density.

        """

        return planck(nu, self.temperature)


class Greybody(EmissionBase):
    """
    A class to generate a greybody emission spectrum.

    Attributes:
        emissivity (float)
            The emissivity of the dust (dimensionless).
    """

    @accepts(temperature=temperature_dim)
    def __init__(self, temperature, emissivity):
        """
        Initialise the dust emission model.

        Args:
            temperature (unyt_array)
                The temperature of the dust.

            emissivity (float)
                The Emissivity (dimensionless).

        """

        EmissionBase.__init__(self, temperature)
        self.emissivity = emissivity

    # @accepts(nu=1/time)
    def _lnu(self, nu):
        """
        Generate unnormalised spectrum for given frequency (nu) grid.

        Args:
            nu (unyt_array)
                The frequencies at which to calculate the spectral luminosity
                density.

        Returns
            lnu (unyt_array)
                The unnormalised spectral luminosity density.

        """

        return nu**self.emissivity * planck(nu, self.temperature)


class Casey12(EmissionBase):
    """
    A class to generate a dust emission spectrum using the Casey (2012) model.
    https://ui.adsabs.harvard.edu/abs/2012MNRAS.425.3094C/abstract

    Attributes:
        emissivity (float)
            The emissivity of the dust (dimensionless).

        alpha (float)
            The power-law slope (dimensionless)  [good value = 2.0].

        n_bb (float)
            Normalisation of the blackbody component [default 1.0].

        lam_0 (float)
            Wavelength where the dust optical depth is unity.

        lam_c (float)
            The power law turnover wavelength.

        n_pl (float)
            The power law normalisation.

    """

    @accepts(temperature=temperature_dim)
    def __init__(
        self, temperature, emissivity, alpha, N_bb=1.0, lam_0=200.0 * um
    ):
        """
        Args:
            lam (unyt_array)
                The wavelengths at which to calculate the emission.

            temperature (unyt_array)
                The temperature of the dust.

            emissivity (float)
                The emissivity (dimensionless) [good value = 1.6].

            alpha (float)
                The power-law slope (dimensionless)  [good value = 2.0].

            n_bb (float)
                Normalisation of the blackbody component [default 1.0].

            lam_0 (float)
                Wavelength where the dust optical depth is unity.
        """

        EmissionBase.__init__(self, temperature)
        self.emissivity = emissivity
        self.alpha = alpha
        self.N_bb = N_bb
        self.lam_0 = lam_0

        # Calculate the power law turnover wavelength
        b1 = 26.68
        b2 = 6.246
        b3 = 0.0001905
        b4 = 0.00007243
        lum = (
            (b1 + b2 * alpha) ** -2
            + (b3 + b4 * alpha) * temperature.to("K").value
        ) ** -1

        self.lam_c = (3.0 / 4.0) * lum * um

        # Calculate normalisation of the power-law term

        # See Casey+2012, Table 1 for the expression
        # Missing factors of lam_c and c in some places

        self.n_pl = (
            self.N_bb
            * (1 - np.exp(-((self.lam_0 / self.lam_c) ** emissivity)))
            * (c / self.lam_c) ** 3
            / (np.exp(h * c / (self.lam_c * kb * temperature)) - 1)
        )

    # @accepts(nu=1/time)
    def _lnu(self, nu):
        """
        Generate unnormalised spectrum for given frequency (nu) grid.

        Args:
            nu (unyt_array)
                The frequencies at which to calculate the spectral luminosity
                density.

        Returns
            lnu (unyt_array)
                The unnormalised spectral luminosity density.

        """

        # Essential, when using scipy.integrate, since
        # the integration limits are passed unitless
        if np.isscalar(nu):
            nu *= Hz

        # Define a function to calcualate the power-law component.
        def _power_law(lam):
            """
            Calcualate the power-law component.

            Args:
                lam (unyt_array)
                    The wavelengths at which to calculate lnu.
            """
            return (
                self.n_pl
                * ((lam / self.lam_c) ** (self.alpha))
                * np.exp(-((lam / self.lam_c) ** 2))
            )

        def _blackbody(lam):
            """
            Calcualate the blackbody component.

            Args:
                lam (unyt_array)
                    The wavelengths at which to calculate lnu.
            """
            return (
                self.N_bb
                * (1 - np.exp(-((self.lam_0 / lam) ** self.emissivity)))
                * (c / lam) ** 3
                / (np.exp((h * c) / (lam * kb * self.temperature)) - 1.0)
            )

        return _power_law(c / nu) + _blackbody(c / nu)


class IR_templates:
    """
    A class to generate a dust emission spectrum using either:
    (i) Draine and Li model (2007) --
    DL07 - https://ui.adsabs.harvard.edu/abs/2007ApJ...657..810D/abstract
    Umax (Maximum radiation field heating the dust) is chosen as 1e7.
    Has less effect where the maximum is on the spectrum
    (ii) Astrodust + PAH model (2023) -- **Not implemented**
    Astrodust - https://ui.adsabs.harvard.edu/abs/2023ApJ...948...55H/abstract

    Attributes:
        mdust (float)
            The mass of dust in the galaxy (Msun).

        template (string)
            The IR template model to be used
            (Currently only Draine and Li 2007 model implemented)

        ldust (float)
            The dust luminosity of the galaxy (integrated from 0 to inf),
            obtained using energy balance here.

        gamma (float)
            Fraction of the dust mass that is associated with the
            power-law part of the starlight intensity distribution.

        qpah (float)
            Fraction of dust mass in the form of PAHs [good value=2.5%]

        umin (float)
            Radiation field heating majority of the dust.

        alpha (float)
            The power law normalisation [good value = 2.].

        p0 (float)
            Power absorbed per unit dust mass in a radiation field
            with U = 1

    """

    @accepts(mdust=mass_dim)
    def __init__(
        self,
        grid,
        mdust,
        ldust=None,
        template="DL07",
        gamma=None,
        qpah=0.025,
        umin=None,
        alpha=2.0,
        p0=125.0,
        verbose=True,
    ):
        self.grid = grid
        self.mdust = mdust
        self.template = template
        self.ldust = ldust
        self.gamma = gamma
        self.qpah = qpah
        self.umin = umin
        self.alpha = alpha
        self.p0 = p0
        self.verbose = verbose

    def dl07(self, grid):
        """
        Draine and Li models
        For simplicity, only MW models are implemented
        (SMC model has only qpah=0.1%)
        These are the extended grids of DL07

        Attributes:
            grid: grid class
        """

        # Define the models parameters
        qpahs = grid.qpah
        umins = grid.umin
        alphas = grid.alpha

        # default Umax=1e7
        umax = 1e7

        if (self.gamma is None) or (self.umin is None) or (self.alpha == 2.0):
            warn(
                "Gamma, Umin or alpha for DL07 model not provided, "
                "using default values"
            )
            warn(
                "Computing required values using Magdis+2012 "
                "stacking results"
            )

            self.u_avg = u_mean_magdis12(
                (self.mdust / Msun).value, (self.ldust / Lsun).value, self.p0
            )

            if self.gamma is None:
                warn("Gamma not provided, choosing default gamma value as 5%")
                self.gamma = 0.05

            func = partial(
                solve_umin, umax=umax, u_avg=self.u_avg, gamma=self.gamma
            )
            self.umin = fsolve(func, [0.1])

        qpah_id = qpahs == qpahs[np.argmin(np.abs(qpahs - self.qpah))]
        umin_id = umins == umins[np.argmin(np.abs(umins - self.umin))]
        alpha_id = alphas == alphas[np.argmin(np.abs(alphas - self.alpha))]

        if np.sum(umin_id) == 0:
            raise exceptions.UnimplementedFunctionality.GridError(
                "No valid model templates found for the given values"
            )

        self.qpah_id = qpah_id
        self.umin_id = umin_id
        self.alpha_id = alpha_id

    def get_spectra(self, _lam, dust_components=False, verbose=True):
        """
        Returns the lnu for the provided wavelength grid

        Arguments:
            _lam (float/array-like, float)
                    An array of wavelengths (expected in AA, global unit)

            dust_components (boolean)
                    If True, returns the constituent dust components

        """

        if self.template == "DL07":
            if verbose:
                print("Using the Draine & Li 2007 dust models")
            self.dl07(self.grid)
        else:
            raise exceptions.UnimplementedFunctionality(
                f"{self.template} not a valid model!"
            )

        if isinstance(_lam, (unyt_quantity, unyt_array)):
            lam = _lam
        else:
            lam = _lam * Angstrom

        # interpret the dust spectra for the given
        # wavelength range
        self.grid.interp_spectra(new_lam=lam)
        lnu_old = (
            (1.0 - self.gamma)
            * self.grid.spectra["diffuse"][self.qpah_id, self.umin_id][0]
            * (self.mdust / Msun).value
        )

        lnu_young = (
            self.gamma
            * self.grid.spectra["pdr"][
                self.qpah_id, self.umin_id, self.alpha_id
            ][0]
            * (self.mdust / Msun).value
        )

        sed_old = Sed(lam=lam, lnu=lnu_old * (erg / s / Hz))
        sed_young = Sed(lam=lam, lnu=lnu_young * (erg / s / Hz))

        # Replace NaNs with zero for wavelength regimes
        # with no values given
        sed_old._lnu[np.isnan(sed_old._lnu)] = 0.0
        sed_young._lnu[np.isnan(sed_young._lnu)] = 0.0

        if dust_components:
            return sed_old, sed_young
        else:
            return sed_old + sed_young


def u_mean_magdis12(mdust, ldust, p0):
    """
    P0 value obtained from stacking analysis in Magdis+12
    For alpha=2.0
    https://ui.adsabs.harvard.edu/abs/2012ApJ...760....6M/abstract
    """

    return ldust / (p0 * mdust)


def u_mean(umin, umax, gamma):
    """
    For fixed alpha=2.0
    """

    return (1.0 - gamma) * umin + gamma * np.log(umax / umin) / (
        umin ** (-1) - umax ** (-1)
    )


def solve_umin(umin, umax, u_avg, gamma):
    """
    For fixed alpha=2.0
    """

    return u_mean(umin, umax, gamma) - u_avg
