"""A submodule defining Draine & Li 2007 dust emission generators."""

from functools import partial
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fsolve
from unyt import (
    Lsun,
    Msun,
    angstrom,
    erg,
    s,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.generator import Generator
from synthesizer.emissions import LineCollection, Sed
from synthesizer.grid import Grid
from synthesizer.synth_warnings import warn
from synthesizer.units import accepts


def u_mean_magdis12(dust_mass: float, ldust: float, p0: float) -> float:
    """Calculate the mean radiation field heating the dust.

    P0 value obtained from stacking analysis in Magdis+12
    For alpha=2.0
    https://ui.adsabs.harvard.edu/abs/2012ApJ...760....6M/abstract

    Args:
        dust_mass (float):
            The dust mass.
        ldust (float):
            The dust luminosity.
        p0 (float):
            Power absorbed per unit dust mass in a radiation field with U = 1.

    Returns:
        float:
            The mean radiation field.
    """
    return ldust / (p0 * dust_mass)


def u_mean(umin: float, umax: float, gamma: float) -> float:
    """Calculate the mean radiation field heating the dust.

    For fixed alpha=2.0, get <U> for Draine and Li model

    Args:
        umin (float):
            Minimum radiation field.
        umax (float):
            Maximum radiation field.
        gamma (float):
            Fraction of dust mass in power-law component.

    Returns:
        float:
            The mean radiation field.
    """
    return (1.0 - gamma) * umin + gamma * np.log(umax / umin) / (
        umin ** (-1) - umax ** (-1)
    )


def solve_umin(umin: float, umax: float, u_avg: float, gamma: float) -> float:
    """Solve for Umin in the Draine and Li model.

    For fixed alpha=2.0, equation to solve to <U> in Draine and Li

    Args:
        umin (float):
            Minimum radiation field.
        umax (float):
            Maximum radiation field.
        u_avg (float):
            Average radiation field.
        gamma (float):
            Fraction of dust mass in power-law component.

    Returns:
        float:
            The difference between calculated and target mean field.
    """
    return u_mean(umin, umax, gamma) - u_avg


class DrainLi07(Generator):
    """A generator for DL07 emission spectrum based on energy balance.

    This class will generate a DL07 spectrum using templates from a dust grid.
    It will then scale the spectrum based on the energy absorbed by dust,
    calculated as the difference between the intrinsic and attenuated
    emissions. The "energy balance" method.

    This is based on the work of Draine & Li (2007):
    https://ui.adsabs.harvard.edu/abs/2007ApJ...657..810D/abstract

    Attributes:
        grid (Grid):
            The dust grid to use for template spectra.
        dust_mass (unyt_quantity):
            The mass of dust in the galaxy.
        dust_to_gas (float):
            The dust-to-gas ratio of the galaxy.
        hydrogen_mass (unyt_quantity):
            The mass in hydrogen of the galaxy.
        template (str):
            The IR template model to be used.
        gamma (float):
            Fraction of the dust mass that is associated with the
            power-law part of the starlight intensity distribution.
        pah_frac (float):
            Fraction of dust mass in the form of PAHs.
        umin (float):
            Radiation field heating majority of the dust.
        alpha (float):
            The power law normalisation.
        p0 (float):
            Power absorbed per unit dust mass in a radiation field with U = 1.
        verbose (bool):
            Whether to print verbose output.
        pah_frac_id (NDArray):
            Grid indices for pah_frac parameter.
        umin_id (NDArray):
            Grid indices for umin parameter.
        alpha_id (NDArray):
            Grid indices for alpha parameter.
    """

    grid: Grid
    dust_mass: Optional[unyt_quantity]
    dust_to_gas: float
    hydrogen_mass: Optional[unyt_quantity]
    template: str
    gamma: Optional[float]
    pah_frac: float
    umin: Optional[float]
    alpha: float
    p0: float
    verbose: bool
    pah_frac_id: Optional[NDArray]
    umin_id: Optional[NDArray]
    alpha_id: Optional[NDArray]

    @accepts(dust_mass=Msun.in_base("galactic"))
    def __init__(
        self,
        grid: Grid,
        dust_mass: Optional[unyt_quantity] = None,
        dust_to_gas: float = 0.01,
        hydrogen_mass: Optional[unyt_quantity] = None,
        template: str = "DL07",
        gamma: Optional[float] = None,
        pah_frac: float = 0.025,
        umin: Optional[float] = None,
        alpha: float = 2.0,
        p0: float = 125.0,
        verbose: bool = True,
        intrinsic: Union[str, EmissionModel] = "intrinsic",
        attenuated: Union[str, EmissionModel] = "attenuated",
    ) -> None:
        """Generate a DL07 spectrum using energy balance.

        Args:
            grid (Grid):
                The dust grid to use for template spectra.
            dust_mass (unyt_quantity, optional):
                The mass of dust in the galaxy. Will be extracted from
                the component or model if not provided.
            dust_to_gas (float):
                The dust-to-gas ratio of the galaxy.
            hydrogen_mass (unyt_quantity, optional):
                The mass in hydrogen. Will be calculated from dust_mass
                and dust_to_gas if not provided.
            template (str):
                The IR template model to be used.
            gamma (float, optional):
                Fraction of the dust mass that is associated with the
                power-law part of the starlight intensity distribution.
                Will be calculated if not provided.
            pah_frac (float):
                Fraction of dust mass in the form of PAHs.
            umin (float, optional):
                Radiation field heating majority of the dust.
                Will be calculated if not provided.
            alpha (float):
                The power law normalisation.
            p0 (float):
                Power absorbed per unit dust mass in a radiation field
                with U = 1.
            verbose (bool):
                Whether to print verbose output.
            intrinsic (str):
                The name of the intrinsic spectrum defining the unattenuated
                emission for energy balance.
            attenuated (str):
                The name of the attenuated spectrum defining the attenuated
                emission for energy balance.
        """
        # Call the parent init with required parameters and emissions
        Generator.__init__(
            self,
            required_params=("dust_mass", "dust_to_gas", "hydrogen_mass"),
            required_emissions=(intrinsic, attenuated),
        )

        self.grid = grid
        self.dust_mass = dust_mass
        self.dust_to_gas = dust_to_gas
        self.hydrogen_mass = hydrogen_mass
        self.template = template
        self.gamma = gamma
        self.pah_frac = pah_frac
        self.umin = umin
        self.alpha = alpha
        self.p0 = p0
        self.verbose = verbose

        # Initialize grid indices
        self.pah_frac_id = None
        self.umin_id = None
        self.alpha_id = None

    def _get_energy_balance_luminosity(self, emitter, model) -> unyt_quantity:
        """Calculate the energy absorbed by dust.

        Args:
            emitter (Stars/Gas/BlackHole):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.

        Returns:
            unyt_quantity:
                The bolometric luminosity absorbed by dust.
        """
        # Extract the emissions
        emissions = self._extract_spectra(
            emitter=emitter,
            per_particle=model.per_particle,
        )

        # For ease, unpack the intrinsic and attenuated emissions
        intrinsic = emissions[self.required_emissions[0]]
        attenuated = emissions[self.required_emissions[1]]

        # Calculate the bolometric luminosity absorbed by dust
        ldust = (
            intrinsic.bolometric_luminosity - attenuated.bolometric_luminosity
        )

        return ldust

    def _setup_dl07_parameters(
        self, dust_mass: unyt_quantity, ldust: unyt_quantity
    ) -> None:
        """Set up the DL07 model parameters and grid indices.

        Args:
            dust_mass (unyt_quantity):
                The dust mass.
            ldust (unyt_quantity):
                The dust luminosity for calculating mean radiation field.
        """
        if self.template != "DL07":
            raise exceptions.UnimplementedFunctionality(
                f"{self.template} not a valid model!"
            )

        if self.verbose:
            print("Using the Draine & Li 2007 dust models")

        # Define the model parameters from grid
        pah_fracs: NDArray[np.float32] = self.grid.qpah
        umins: NDArray[np.float32] = self.grid.umin
        alphas: NDArray[np.float32] = self.grid.alpha

        # Default Umax=1e7
        umax: float = 1e7

        # Calculate hydrogen mass if not provided
        hydrogen_mass = self.hydrogen_mass
        if hydrogen_mass is None:
            warn(
                "No hydrogen gas mass provided, assuming a "
                f"dust-to-gas ratio of {self.dust_to_gas}"
            )
            # Calculate hydrogen mass from dust mass and dust-to-gas ratio
            hydrogen_mass = 0.74 * dust_mass / self.dust_to_gas

        if (self.gamma is None) or (self.umin is None) or (self.alpha == 2.0):
            warn(
                "Gamma, Umin or alpha for DL07 model not provided, "
                "using default values"
            )
            warn(
                "Computing required values using Magdis+2012 stacking results"
            )

            u_avg = u_mean_magdis12(
                (dust_mass / Msun).value, (ldust / Lsun).value, self.p0
            )

            gamma = self.gamma
            if gamma is None:
                warn("Gamma not provided, choosing default gamma value as 5%")
                gamma = 0.05

            umin = self.umin
            if umin is None:
                func = partial(solve_umin, umax=umax, u_avg=u_avg, gamma=gamma)
                umin = fsolve(func, [1.0])[0]

        else:
            gamma = self.gamma
            umin = self.umin

        # Find nearest grid indices
        pah_frac_id = (
            pah_fracs
            == pah_fracs[np.argmin(np.abs(pah_fracs - self.pah_frac))]
        )
        umin_id = umins == umins[np.argmin(np.abs(umins - umin))]
        alpha_id = alphas == alphas[np.argmin(np.abs(alphas - self.alpha))]

        if np.sum(umin_id) == 0:
            raise exceptions.UnimplementedFunctionality(
                "No valid model templates found for the given values"
            )

        self.pah_frac_id = pah_frac_id
        self.umin_id = umin_id
        self.alpha_id = alpha_id

        # Store the calculated values
        self._calculated_gamma = gamma
        self._calculated_hydrogen_mass = hydrogen_mass

    def _generate_dl07_spectra(
        self, lams: unyt_array, dust_components: bool = False
    ) -> Union[Sed, tuple[Sed, Sed]]:
        """Generate the DL07 dust emission spectra.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            dust_components (bool):
                If True, returns the constituent dust components separately.

        Returns:
            Sed or tuple[Sed, Sed]:
                The generated dust emission SED(s).
        """
        # Ensure parameters are set up
        if self.pah_frac_id is None:
            raise exceptions.MissingAttribute(
                "DL07 parameters not set up. Call "
                "_setup_dl07_parameters first."
            )

        # Interpolate the dust spectra for the given wavelength range
        self.grid.interp_spectra(new_lam=lams)

        lnu_old = (
            (1.0 - self._calculated_gamma)
            * self.grid.spectra["diffuse"][self.pah_frac_id, self.umin_id][0]
            * (self._calculated_hydrogen_mass / Msun).value
        )

        lnu_young = (
            self._calculated_gamma
            * self.grid.spectra["pdr"][
                self.pah_frac_id, self.umin_id, self.alpha_id
            ][0]
            * (self._calculated_hydrogen_mass / Msun).value
        )

        sed_old = Sed(lam=lams, lnu=lnu_old * (erg / s))
        sed_young = Sed(lam=lams, lnu=lnu_young * (erg / s))

        # Replace NaNs with zero for wavelength regimes with no values given
        sed_old._lnu[np.isnan(sed_old._lnu)] = 0.0
        sed_young._lnu[np.isnan(sed_young._lnu)] = 0.0

        if dust_components:
            return sed_old, sed_young
        else:
            return sed_old + sed_young

    @accepts(lams=angstrom)
    def get_spectra(
        self,
        lams: unyt_array,
        dust_mass: Optional[unyt_quantity] = None,
        ldust: Optional[unyt_quantity] = None,
        dust_components: bool = False,
    ) -> Union[Sed, tuple[Sed, Sed]]:
        """Generate dust emission spectra for given wavelength grid.

        This will return the scaling free spectra for a given wavelength grid.
        It will not consider any emitter or model, so the dust parameters must
        have been provided directly to the class or passed as arguments.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            dust_mass (unyt_quantity, optional):
                The dust mass. If not provided, uses the value from the class.
            ldust (unyt_quantity, optional):
                The dust luminosity for parameter calculation. If not provided,
                a default value will be used for parameter setup.
            dust_components (bool):
                If True, returns the constituent dust components separately.

        Returns:
            Sed or tuple[Sed, Sed]:
                The generated dust emission SED(s).
        """
        # Use provided dust_mass or fall back to class attribute
        if dust_mass is None:
            if self.dust_mass is None:
                raise exceptions.MissingAttribute(
                    "dust_mass must be provided either to the class "
                    "constructor or to get_spectra method."
                )
            dust_mass = self.dust_mass

        # If ldust not provided, use a reasonable default for parameter setup
        if ldust is None:
            # Use a default dust luminosity based on dust mass
            # This is just for parameter setup, not for final scaling
            ldust = 1e10 * (dust_mass / (1e6 * Msun)) * Lsun

        # Set up DL07 parameters
        self._setup_dl07_parameters(dust_mass, ldust)

        # Generate the base spectra
        return self._generate_dl07_spectra(lams, dust_components)

    @accepts(lams=angstrom)
    def _generate_spectra(self, lams, emitter, model) -> Sed:
        """Generate the dust emission spectra.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.

        Returns:
            Sed:
                The generated dust emission SED.
        """
        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack the dust parameters
        dust_mass = params["dust_mass"]

        # Calculate the dust luminosity for energy balance
        ldust = self._get_energy_balance_luminosity(emitter, model)

        # Set up DL07 parameters using the dust luminosity
        self._setup_dl07_parameters(dust_mass, ldust)

        # Generate the base spectra
        sed = self._generate_dl07_spectra(lams)

        # Normalise the spectrum
        sed._lnu /= np.expand_dims(sed._bolometric_luminosity, axis=-1)

        # Apply the energy balance scaling
        sed._lnu *= ldust

        return sed

    @accepts(line_lams=angstrom)
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
    ) -> LineCollection:
        """Generate line emission spectra.

        DL07 emission does not produce line emission, so this
        method will simply return continuum emission.

        Args:
            line_ids (list):
                The IDs of the lines to generate.
            line_lams (unyt_array):
                The wavelength grid on which to generate the line spectra.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.

        Returns:
            LineCollection:
                The generated line collection.
        """
        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack the dust parameters
        dust_mass = params["dust_mass"]

        # Calculate the dust luminosity for energy balance
        ldust = self._get_energy_balance_luminosity(emitter, model)

        # Set up DL07 parameters using the dust luminosity
        self._setup_dl07_parameters(dust_mass, ldust)

        # Generate the base spectra
        sed = self._generate_dl07_spectra(line_lams)

        # Normalise the spectrum
        sed._lnu /= np.expand_dims(sed._bolometric_luminosity, axis=-1)

        # Apply the energy balance scaling
        sed._lnu *= ldust

        # Return as LineCollection with continuum only
        lines = LineCollection(
            line_ids,
            line_lams,
            lum=np.zeros(sed._lnu.shape) * erg / s,
            cont=sed.lnu,
        )

        return lines
