"""Module containing dust emission functionality"""

from functools import partial
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
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
from synthesizer.grid import Grid
from synthesizer.sed import Sed
from synthesizer.utils import planck
from synthesizer.warnings import warn


class EmissionBase:
    """
    Dust emission base class for holding common methods.

    Attributes:
        temperature (float)
            The temperature of the dust.
        cmb_factor (float)
            The multiplicative factor to account for
            CMB heating at high-redshift
    """

    temperature: Optional[Union[unyt_quantity, float]]
    cmb_factor: float

    def __init__(
        self,
        temperature: Optional[Union[unyt_quantity, float]] = None,
        cmb_factor: float = 1,
    ) -> None:
        """
        Initialises the base class for dust emission models.

        Args:
            temperature (float)
                The temperature of the dust.
            cmb_factor (float)
                The multiplicative factor to account for
                CMB heating at high-redshift
        """

        self.temperature = temperature
        self.cmb_factor = cmb_factor

    def _lnu(
        self, *args: Optional[Union[unyt_array, NDArray[np.float64]]]
    ) -> Optional[Union[unyt_array, NDArray[np.float64]]]:
        """
        A prototype private method used during integration. This should be
        overloaded by child classes!
        """
        raise exceptions.UnimplementedFunctionality(
            "EmissionBase should not be instantiated directly!"
            " Instead use one to child models (Blackbody, Greybody, Casey12)."
        )

    def normalisation(self) -> float:
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

    def apply_cmb_heating(self, emissivity: float, z: float) -> None:
        """
        Returns the factor by which the CMB boosts the
        infrared luminosity
        (See implementation in da Cunha+2013)

        Args:
            emissivity (float)
                The emissivity index in the FIR (no unit)
            z (float)
                The redshift of the galaxy
        """

        # temperature of CMB at z=0
        _T_cmb_0 = 2.73
        _T_cmb_z = _T_cmb_0 * (1 + z)
        _exp_factor = 4.0 + emissivity

        _temperature = (
            self.temperature**_exp_factor
            + _T_cmb_z**_exp_factor
            - _T_cmb_0**_exp_factor
        ) ** (1 / _exp_factor)

        cmb_factor: float = (_temperature / self.temperature) ** (
            4 + emissivity
        )

        self.cmb_factor = cmb_factor

    def get_spectra(self, _lam: Union[NDArray[np.float64], unyt_array]) -> Sed:
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
        sed._lnu /= sed.measure_bolometric_luminosity().value  # type: ignore

        # apply heating due to CMB, if applicable
        sed._lnu *= self.cmb_factor

        return sed


class Blackbody(EmissionBase):
    """
    A class to generate a blackbody emission spectrum.
    """

    temperature: unyt_quantity
    cmb_heating: bool
    z: float

    @accepts(temperature=temperature_dim)
    def __init__(
        self,
        temperature: unyt_quantity,
        cmb_heating: bool = False,
        z: float = 0,
    ) -> None:
        """
        A function to generate a simple blackbody spectrum.

        Args:
            temperature (unyt_array)
                The temperature of the dust.

            cmb_heating (bool)
                Option for adding heating by CMB

            z (float)
                Redshift of the galaxy

        """

        EmissionBase.__init__(self, temperature)

        # emmissivity of true blackbody is 1
        emissivity = 1.0

        if cmb_heating:
            # calculate the factor by which the CMB boosts the
            # infrared luminosity
            self.apply_cmb_heating(emissivity=emissivity, z=z)

    def _lnu(self, nu: unyt_array) -> unyt_array:  # type: ignore[override]
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

        cmb_heating (bool)
            Option for adding heating by CMB

        z (float)
            Redshift of the galaxy

    """

    temperature: unyt_quantity
    emissivity: float
    cmb_heating: bool
    z: float

    @accepts(temperature=temperature_dim)
    def __init__(
        self,
        temperature: unyt_quantity,
        emissivity: float,
        cmb_heating: bool = False,
        z: float = 0,
    ) -> None:
        """
        Initialise the dust emission model.

        Args:
            temperature (unyt_array)
                The temperature of the dust.

            emissivity (float)
                The Emissivity (dimensionless).

            cmb_heating (bool)
                Option for adding heating by CMB

            z (float)
                Redshift of the galaxy

        """

        EmissionBase.__init__(self, temperature)

        if cmb_heating:
            # calculate the factor by which the CMB boosts the
            # infrared luminosity
            self.apply_cmb_heating(emissivity=emissivity, z=z)

        self.emissivity = emissivity

    def _lnu(self, nu: unyt_array) -> unyt_array:  # type: ignore[override]
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

        cmb_heating (bool)
                Option for adding heating by CMB

        z (float)
            Redshift of the galaxy

    """

    temperature: unyt_quantity
    emissivity: float
    alpha: float
    N_bb: float
    lam_0: unyt_quantity
    cmb_heating: bool
    z: float

    @accepts(temperature=temperature_dim)
    def __init__(
        self,
        temperature: unyt_quantity,
        emissivity: float,
        alpha: float,
        N_bb: float = 1.0,
        lam_0: unyt_quantity = 200.0 * um,
        cmb_heating: bool = False,
        z: float = 0,
    ) -> None:
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

            cmb_heating (bool)
                Option for adding heating by CMB

            z (float)
                Redshift of the galaxy

        """

        EmissionBase.__init__(self, temperature)

        if cmb_heating:
            # calculate the factor by which the CMB boosts the
            # infrared luminosity
            self.apply_cmb_heating(emissivity=emissivity, z=z)

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
            + (b3 + b4 * alpha) * self.temperature.to("K").value
        ) ** -1

        self.lam_c = (3.0 / 4.0) * lum * um

        # Calculate normalisation of the power-law term

        # See Casey+2012, Table 1 for the expression
        # Missing factors of lam_c and c in some places

        self.n_pl = (
            self.N_bb
            * (1 - np.exp(-((self.lam_0 / self.lam_c) ** emissivity)))
            * (c / self.lam_c) ** 3
            / (np.exp(h * c / (self.lam_c * kb * self.temperature)) - 1)
        )

    def _lnu(  # type: ignore[override]
        self, nu: unyt_array
    ) -> Union[NDArray[np.float64], unyt_array]:
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
        def _power_law(lam: unyt_array) -> float:
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

        def _blackbody(lam: unyt_array) -> unyt_array:
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
        grid (Grid object)
            The dust grid to use

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

    grid: Grid
    mdust: unyt_quantity
    ldust: Optional[float]
    template: str
    gamma: Optional[float]
    qpah: float
    umin: Optional[float]
    alpha: float
    p0: float
    verbose: bool

    @accepts(mdust=mass_dim)
    def __init__(
        self,
        grid: Grid,
        mdust: unyt_quantity,
        ldust: Optional[float] = None,
        template: str = "DL07",
        gamma: Optional[float] = None,
        qpah: float = 0.025,
        umin: Optional[float] = None,
        alpha: float = 2.0,
        p0: float = 125.0,
        verbose: bool = True,
    ) -> None:
        self.grid: Grid = grid
        self.mdust: unyt_quantity = mdust
        self.template: str = template
        self.ldust: Optional[float] = ldust
        self.gamma: Optional[float] = gamma
        self.qpah: float = qpah
        self.umin: Optional[float] = umin
        self.alpha: float = alpha
        self.p0: float = p0
        self.verbose: bool = verbose

    def dl07(self) -> None:
        """
        Draine and Li models
        For simplicity, only MW models are implemented
        (SMC model has only qpah=0.1%)
        These are the extended grids of DL07

        Attributes:
            grid: grid class
        """

        # Define the models parameters
        qpahs: NDArray[float] = self.grid.qpah
        umins: NDArray[float] = self.grid.umin
        alphas: NDArray[float] = self.grid.alpha

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

    def get_spectra(
        self,
        _lam: Union[NDArray[np.float64], unyt_array],
        dust_components: bool = False,
    ) -> Union[tuple[Sed, Sed], Sed]:
        """
        Returns the lnu for the provided wavelength grid

        Arguments:
            _lam (float/array-like, float)
                    An array of wavelengths (expected in AA, global unit)

            dust_components (boolean)
                    If True, returns the constituent dust components

        """

        if self.template == "DL07":
            print("Using the Draine & Li 2007 dust models")
            self.dl07()  # type: ignore
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


def apply_cmb_heating(temperature: float, emissivity: float, z: float):
    """
    Returns the dust temperature due to heating by the CMB
    (See implementation in da Cunha+2013)

    Args:
        temperature (float)
            The temperature of the dust
        emissivity (float)
            The emissivity index in the FIR (no unit)
        z (float)
            The redshift of the galaxy
    """

    # temperature of CMB at z=0
    _T_cmb_0 = 2.73
    _T_cmb_z = _T_cmb_0 * (1 + z)
    _exp_factor = 4.0 + emissivity

    _temperature = (
        temperature**_exp_factor
        + _T_cmb_z**_exp_factor
        - _T_cmb_0**_exp_factor
    ) ** (1 / _exp_factor)

    return _temperature


def u_mean_magdis12(mdust: float, ldust: float, p0: float):
    """
    P0 value obtained from stacking analysis in Magdis+12
    For alpha=2.0
    https://ui.adsabs.harvard.edu/abs/2012ApJ...760....6M/abstract
    """

    return ldust / (p0 * mdust)


def u_mean(umin: float, umax: float, gamma: float):
    """
    For fixed alpha=2.0
    """

    return (1.0 - gamma) * umin + gamma * np.log(umax / umin) / (
        umin ** (-1) - umax ** (-1)
    )


def solve_umin(umin: float, umax: float, u_avg: float, gamma: float):
    """
    For fixed alpha=2.0
    """

    return u_mean(umin, umax, gamma) - u_avg
