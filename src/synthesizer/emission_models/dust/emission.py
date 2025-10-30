"""A module defining emission generation classes.

This module provides classes for generating spectra from specific
parametrisations. Each of these will return a normalised spectrum.

Possible classes include:
    - Blackbody: A simple blackbody spectrum.
    - Greybody: A greybody spectrum.
    - Casey12: A dust emission spectrum using the Casey (2012) model.
    - IR_templates: A class to generates dust emission spectra using
        the Draine and Li (2007) model.

Example usage:

        blackbody = Blackbody(temperature=20 * K)
        greybody = Greybody(temperature=20 * K, emissivity=1.5)
        casey12 = Casey12(temperature=20 * K, emissivity=1.5, alpha=2.0)
        ir_templates = IR_templates(grid, mdust=1e6 * Msun)
        ir_templates.dl07()
        ir_templates.get_spectra(lam, intrinsic_sed=sed1, attenuated_sed=sed2)
"""

from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve
from unyt import (
    Hz,
    Lsun,
    Msun,
    angstrom,
    erg,
    s,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.synth_warnings import warn
from synthesizer.units import accepts


class IR_templates:
    """A class to generate a dust emission spectrum.

    Can use either:
    (i) Draine and Li model (2007) --
    DL07 - https://ui.adsabs.harvard.edu/abs/2007ApJ...657..810D/abstract
    Umax (Maximum radiation field heating the dust) is chosen as 1e7.
    Has less effect where the maximum is on the spectrum
    (ii) Astrodust + PAH model (2023) -- **Not implemented**
    Astrodust - https://ui.adsabs.harvard.edu/abs/2023ApJ...948...55H/abstract

    Attributes:
        grid (Grid object)
            The dust grid to use
        mdust (float):
            The mass of dust in the galaxy (Msun).
        dgr (float):
            The dust-to-gas ratio of the galaxy
        MH (float):
            The mass in hydrogen of the galaxy
        template (str):
            The IR template model to be used
            (Currently only Draine and Li 2007 model implemented)
        ldust (float):
            The dust luminosity of the galaxy (integrated from 0 to inf),
            obtained using energy balance here.
        gamma (float):
            Fraction of the dust mass that is associated with the
            power-law part of the starlight intensity distribution.
        qpah (float):
            Fraction of dust mass in the form of PAHs [good value=2.5%]
        umin (float):
            Radiation field heating majority of the dust.
        alpha (float):
            The power law normalisation [good value = 2.].
        p0 (float):
            Power absorbed per unit dust mass in a radiation field
            with U = 1
    """

    grid: Grid
    mdust: unyt_quantity
    dgr: float
    MH: Optional[unyt_quantity]
    ldust: Optional[unyt_quantity]
    template: str
    gamma: Optional[float]
    qpah: float
    umin: Optional[float]
    alpha: float
    p0: float
    verbose: bool

    @accepts(mdust=Msun.in_base("galactic"))
    def __init__(
        self,
        grid: Grid,
        mdust: unyt_quantity,
        dgr: float = 0.01,
        MH: Optional[unyt_quantity] = None,
        ldust: Optional[unyt_quantity] = None,
        template: str = "DL07",
        gamma: Optional[float] = None,
        qpah: float = 0.025,
        umin: Optional[float] = None,
        alpha: float = 2.0,
        p0: float = 125.0,
        verbose: bool = True,
    ) -> None:
        """Initialise the dust emission model.

        Args:
            grid (Grid):
                The dust grid to use
            mdust (unyt_quantity):
                The mass of dust in the galaxy (Msun).
            dgr (float):
                The dust-to-gas ratio of the galaxy
            MH (unyt_quantity):
                The mass in hydrogen
            template (str):
                The IR template model to be used
                (Currently only Draine and Li 2007 model implemented)
            ldust (unyt_quantity):
                The dust luminosity of the galaxy (integrated from 0 to inf),
                obtained using energy balance here.
            gamma (float):
                Fraction of the dust mass that is associated with the
                power-law part of the starlight intensity distribution.
            qpah (float):
                Fraction of dust mass in the form of PAHs [good value=2.5%]
            umin (float):
                Radiation field heating majority of the dust.
            alpha (float):
                The power law normalisation [good value = 2.].
            p0 (float):
                Power absorbed per unit dust mass in a radiation field
                with U = 1
            verbose (bool):
                Are we talking?
        """
        self.grid: Grid = grid
        self.mdust: unyt_quantity = mdust
        self.dgr: float = dgr
        self.template: str = template
        self.ldust: Optional[unyt_quantity] = ldust
        self.MH: Optional[unyt_quantity] = MH
        self.gamma: Optional[float] = gamma
        self.qpah: float = qpah
        self.umin: Optional[float] = umin
        self.alpha: float = alpha
        self.p0: float = p0
        self.verbose: bool = verbose

    def dl07(self) -> None:
        """Draine and Li models.

        For simplicity, only MW models are implemented
        (SMC model has only qpah=0.1%). These are the extended grids of DL07.
        """
        # Define the models parameters
        qpahs: NDArray[np.float32] = self.grid.qpah
        umins: NDArray[np.float32] = self.grid.umin
        alphas: NDArray[np.float32] = self.grid.alpha

        # Default Umax=1e7
        umax: float = 1e7

        if self.MH is None:
            warn(
                "No hydrogen gas mass provided, assuming a"
                f"dust-to-gas ratio of {self.dgr}"
            )
            # Calculate MH: Mass in hydrogen gas
            self.MH = 0.74 * self.mdust / self.dgr

        if (self.gamma is None) or (self.umin is None) or (self.alpha == 2.0):
            warn(
                "Gamma, Umin or alpha for DL07 model not provided, "
                "using default values"
            )
            warn(
                "Computing required values using Magdis+2012 stacking results"
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
            self.umin = fsolve(func, [1.0])

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

    @accepts(lam=angstrom)
    def get_spectra(
        self,
        lam,
        intrinsic_sed=None,
        attenuated_sed=None,
        dust_components=False,
        **kwargs,
    ):
        """Return the lnu for the provided wavelength grid.

        Arguments:
            lam (float/np.ndarray of float):
                    An array of wavelengths.
            intrinsic_sed (Sed):
                The intrinsic SED to scale with dust.
            attenuated_sed (Sed):
                The attenuated SED to scale with dust.
            dust_components (bool):
                    If True, returns the constituent dust components.
            **kwargs (dict):
                Additional keyword arguments to pass to the model.

        """
        # Calculate the dust luminosity
        if self.template == "DL07":
            if self.verbose:
                print("Using the Draine & Li 2007 dust models")
            if intrinsic_sed is not None and attenuated_sed is not None:
                if self.ldust is not None:
                    warn(
                        "Dust luminosity is already set by user"
                        "to {self.ldust}. Is this expected behaviour?"
                    )
                # Calculate the bolometric dust luminosity as the difference
                # between the intrinsic and attenuated
                ldust = (
                    intrinsic_sed.bolometric_luminosity
                    - attenuated_sed.bolometric_luminosity
                )
                self.ldust = ldust.to("Lsun")
            self.dl07()
        else:
            raise exceptions.UnimplementedFunctionality(
                f"{self.template} not a valid model!"
            )

        # Interpret the dust spectra for the given
        # wavelength range
        self.grid.interp_spectra(new_lam=lam)
        lnu_old = (
            (1.0 - self.gamma)
            * self.grid.spectra["diffuse"][self.qpah_id, self.umin_id][0]
            * (self.MH / Msun).value
        )

        lnu_young = (
            self.gamma
            * self.grid.spectra["pdr"][
                self.qpah_id, self.umin_id, self.alpha_id
            ][0]
            * (self.MH / Msun).value
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


def u_mean_magdis12(mdust: float, ldust: float, p0: float) -> float:
    """Calculate the mean radiation field heating the dust.

    P0 value obtained from stacking analysis in Magdis+12
    For alpha=2.0
    https://ui.adsabs.harvard.edu/abs/2012ApJ...760....6M/abstract
    """
    return ldust / (p0 * mdust)


def u_mean(umin: float, umax: float, gamma: float) -> float:
    """Calculate the mean radiation field heating the dust.

    For fixed alpha=2.0, get <U> for Draine and Li model
    """
    return (1.0 - gamma) * umin + gamma * np.log(umax / umin) / (
        umin ** (-1) - umax ** (-1)
    )


def solve_umin(umin: float, umax: float, u_avg: float, gamma: float) -> float:
    """Solve for Umin in the Draine and Li model.

    For fixed alpha=2.0, equation to solve to <U> in Draine and Li
    """
    return u_mean(umin, umax, gamma) - u_avg
