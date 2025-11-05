"""A submodule defining Draine & Li 2007 dust emission generators."""

from functools import partial
from typing import Optional, Union

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
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.components.component import Component
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust.dust_emission_base import (
    DustEmission,
)
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


class DraineLi07(DustEmission):
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
        dust_to_gas_ratio (float):
            The dust-to-gas ratio of the galaxy.
        hydrogen_mass (unyt_quantity):
            The mass in hydrogen of the galaxy.
        template (str):
            The IR template model to be used.
        gamma (float):
            Fraction of the dust mass that is associated with the
            power-law part of the starlight intensity distribution.
        pah_fraction (float):
            Fraction of dust mass in the form of PAHs (often called qpah).
        umin (float):
            Radiation field heating majority of the dust.
        alpha (float):
            The power law normalisation.
        power_per_unit_mass (float):
            Power absorbed per unit dust mass in a radiation field with U = 1.
        verbose (bool):
            Whether to print verbose output.
        pah_fraction_indices (NDArray):
            Grid indices for pah_fraction parameter.
        umin_indices (NDArray):
            Grid indices for umin parameter.
        alpha_indices (NDArray):
            Grid indices for alpha parameter.

        # Calculated values (available after parameter setup):
        radiation_field_average (float):
            Average radiation field intensity <U>.
        umin_calculated (float):
            Calculated minimum radiation field.
        gamma_calculated (float):
            Calculated gamma parameter.
        hydrogen_mass_calculated (unyt_quantity):
            Calculated hydrogen mass.
    """

    grid: Grid
    dust_mass: Optional[unyt_quantity]
    dust_to_gas_ratio: float
    hydrogen_mass: Optional[unyt_quantity]
    template: str
    gamma: Optional[float]
    pah_fraction: float
    umin: Optional[float]
    alpha: float
    power_per_unit_mass: float
    verbose: bool
    pah_fraction_indices: Optional[NDArray]
    umin_indices: Optional[NDArray]
    alpha_indices: Optional[NDArray]

    # Calculated values
    radiation_field_average: Optional[float]
    umin_calculated: Optional[float]
    gamma_calculated: Optional[float]
    hydrogen_mass_calculated: Optional[unyt_quantity]

    @accepts(dust_mass=Msun.in_base("galactic"))
    def __init__(
        self,
        grid: Grid,
        dust_mass: Optional[unyt_quantity] = None,
        dust_to_gas_ratio: float = 0.01,
        hydrogen_mass: Optional[unyt_quantity] = None,
        template: str = "DL07",
        gamma: Optional[float] = None,
        pah_fraction: float = 0.025,
        umin: Optional[float] = None,
        alpha: float = 2.0,
        power_per_unit_mass: float = 125.0,
        verbose: bool = True,
        do_cmb_heating: bool = False,
        intrinsic: Optional[EmissionModel] = None,
        attenuated: Optional[EmissionModel] = None,
        scaler: Optional[EmissionModel] = None,
    ) -> None:
        """Generate a DL07 spectrum.

        Args:
            grid (Grid):
                The dust grid to use for template spectra.
            dust_mass (unyt_quantity, optional):
                The mass of dust in the galaxy. Will be extracted from
                the component or model if not provided.
            dust_to_gas_ratio (float):
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
            pah_fraction (float):
                Fraction of dust mass in the form of PAHs.
            umin (float, optional):
                Radiation field heating majority of the dust.
                Will be calculated if not provided.
            alpha (float):
                The power law normalisation.
            power_per_unit_mass (float):
                Power absorbed per unit dust mass in a radiation field
                with U = 1.
            verbose (bool):
                Whether to print verbose output.
            do_cmb_heating (bool):
                Should we apply cmb heating?
            intrinsic (EmissionModel):
                The name of the intrinsic emission defining the unattenuated
                emission for energy balance.
            attenuated (EmissionModel):
                The name of the attenuated emission defining the attenuated
                emission for energy balance.
            scaler (EmissionModel):
                The name of the emission to use to scale the dust emission by.
        """
        # Call the parent init with required parameters and emissions
        DustEmission.__init__(
            self,
            intrinsic,
            attenuated,
            scaler,
            do_cmb_heating,
            required_params=(
                "dust_mass",
                "dust_to_gas_ratio",
                "hydrogen_mass",
                "pah_fraction",
            ),
        )

        self.grid = grid
        self.dust_mass = dust_mass
        self.dust_to_gas_ratio = dust_to_gas_ratio
        self.hydrogen_mass = hydrogen_mass
        self.template = template
        self.gamma = gamma
        self.pah_fraction = pah_fraction
        self.umin = umin
        self.alpha = alpha
        self.power_per_unit_mass = power_per_unit_mass
        self.verbose = verbose

        # Initialize grid indices
        self.pah_fraction_indices = None
        self.umin_indices = None
        self.alpha_indices = None

        # Initialize calculated values
        self.radiation_field_average = None
        self.umin_calculated = None
        self.gamma_calculated = None
        self.hydrogen_mass_calculated = None

    def _setup_dl07_parameters(
        self,
        dust_mass: unyt_quantity,
        ldust: unyt_quantity,
        dust_to_gas_ratio: float = None,
        hydrogen_mass: unyt_quantity = None,
        pah_fraction: float = None,
    ) -> None:
        """Set up the DL07 model parameters and grid indices.

        Args:
            dust_mass (unyt_quantity):
                The dust mass.
            ldust (unyt_quantity):
                The dust luminosity for calculating mean radiation field.
            dust_to_gas_ratio (float, optional):
                The dust-to-gas ratio to use. If None, uses instance default.
            hydrogen_mass (unyt_quantity, optional):
                The hydrogen mass to use. If None, uses instance default.
            pah_fraction (float, optional):
                The PAH fraction to use. If None, uses instance default.
        """
        if self.template != "DL07":
            raise exceptions.UnimplementedFunctionality(
                f"{self.template} not a valid model!"
            )

        # Define the model parameters from grid
        grid_pah_fractions: NDArray[np.float32] = self.grid.qpah
        grid_umin_values: NDArray[np.float32] = self.grid.umin
        grid_alpha_values: NDArray[np.float32] = self.grid.alpha

        # Use provided parameters or fall back to instance defaults
        effective_dust_to_gas_ratio = (
            dust_to_gas_ratio
            if dust_to_gas_ratio is not None
            else self.dust_to_gas_ratio
        )
        effective_hydrogen_mass = (
            hydrogen_mass if hydrogen_mass is not None else self.hydrogen_mass
        )
        effective_pah_fraction = (
            pah_fraction if pah_fraction is not None else self.pah_fraction
        )

        # Default maximum radiation field intensity
        umax: float = 1e7

        # Calculate hydrogen mass if not provided
        calculated_hydrogen_mass = effective_hydrogen_mass
        if calculated_hydrogen_mass is None:
            warn(
                "No hydrogen gas mass provided, assuming a "
                f"dust-to-gas ratio of {effective_dust_to_gas_ratio}"
            )
            # Calculate hydrogen mass from dust mass and dust-to-gas ratio
            calculated_hydrogen_mass = (
                0.74 * dust_mass / effective_dust_to_gas_ratio
            )

        if (self.gamma is None) or (self.umin is None) or (self.alpha == 2.0):
            warn(
                "Gamma, Umin or alpha for DL07 model not provided, "
                "using default values"
            )
            warn(
                "Computing required values using Magdis+2012 stacking results"
            )

            calculated_u_avg = u_mean_magdis12(
                (dust_mass / Msun).value,
                (ldust / Lsun).value,
                self.power_per_unit_mass,
            )

            calculated_gamma = self.gamma
            if calculated_gamma is None:
                warn("Gamma not provided, choosing default gamma value as 5%")
                calculated_gamma = 0.05

            calculated_umin = self.umin
            if calculated_umin is None:
                optimization_function = partial(
                    solve_umin,
                    umax=umax,
                    u_avg=calculated_u_avg,
                    gamma=calculated_gamma,
                )
                calculated_umin = fsolve(optimization_function, [1.0])[0]

        else:
            calculated_gamma = self.gamma
            calculated_umin = self.umin
            calculated_u_avg = u_mean(calculated_umin, umax, calculated_gamma)

        # Find nearest grid indices
        pah_fraction_indices = (
            grid_pah_fractions
            == grid_pah_fractions[
                np.argmin(np.abs(grid_pah_fractions - effective_pah_fraction))
            ]
        )
        umin_indices = (
            grid_umin_values
            == grid_umin_values[
                np.argmin(np.abs(grid_umin_values - calculated_umin))
            ]
        )
        alpha_indices = (
            grid_alpha_values
            == grid_alpha_values[
                np.argmin(np.abs(grid_alpha_values - self.alpha))
            ]
        )

        if np.sum(umin_indices) == 0:
            raise exceptions.UnimplementedFunctionality(
                "No valid model templates found for the given values"
            )

        self.pah_fraction_indices = pah_fraction_indices
        self.umin_indices = umin_indices
        self.alpha_indices = alpha_indices

        # Store the calculated values for later access
        self.gamma_calculated = calculated_gamma
        self.hydrogen_mass_calculated = calculated_hydrogen_mass
        self.radiation_field_average = calculated_u_avg
        self.umin_calculated = calculated_umin

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
        if self.pah_fraction_indices is None:
            raise exceptions.MissingAttribute(
                "DL07 parameters not set up. Call "
                "_setup_dl07_parameters first."
            )

        # Interpolate the dust spectra for the given wavelength range
        self.grid.interp_spectra(new_lam=lams)

        # Extract diffuse (old) component spectrum
        diffuse_luminosity = (
            (1.0 - self.gamma_calculated)
            * self.grid.spectra["diffuse"][
                np.where(self.pah_fraction_indices)[0][0],
                np.where(self.umin_indices)[0][0],
            ]
            * (self.hydrogen_mass_calculated / Msun).value
        )

        # Extract PDR (young) component spectrum
        pdr_luminosity = (
            self.gamma_calculated
            * self.grid.spectra["pdr"][
                np.where(self.pah_fraction_indices)[0][0],
                np.where(self.umin_indices)[0][0],
                np.where(self.alpha_indices)[0][0],
            ]
            * (self.hydrogen_mass_calculated / Msun).value
        )

        diffuse_sed = Sed(lam=lams, lnu=diffuse_luminosity * (erg / s / Hz))
        pdr_sed = Sed(lam=lams, lnu=pdr_luminosity * (erg / s / Hz))

        # Replace NaNs with zero for wavelength regimes with no values given
        diffuse_sed._lnu[np.isnan(diffuse_sed._lnu)] = 0.0
        pdr_sed._lnu[np.isnan(pdr_sed._lnu)] = 0.0

        if dust_components:
            return diffuse_sed, pdr_sed
        else:
            return diffuse_sed + pdr_sed

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

        # Set up DL07 parameters using instance defaults
        self._setup_dl07_parameters(dust_mass, ldust)

        # Generate the base spectra
        return self._generate_dl07_spectra(lams, dust_components)

    @accepts(lams=angstrom)
    def _generate_spectra(
        self,
        lams: unyt_array,
        emitter: Component,
        model: EmissionModel,
        emissions: dict,
        redshift: float = 0,
    ) -> Sed:
        """Generate the dust emission spectra.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.
            model (EmissionModel):
                The emission model generating the emission.
            emissions (dict):
                Dictionary containing all emissions generated so far.
            redshift (float):
                The redshift at which to calculate the CMB heating. (Ignored
                if not applying CMB heating).

        Returns:
            Sed:
                The generated dust emission SED.
        """
        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack all dust parameters
        dust_mass = params["dust_mass"]
        dust_to_gas_ratio = params["dust_to_gas_ratio"]
        hydrogen_mass = params["hydrogen_mass"]
        pah_fraction = params["pah_fraction"]

        # Calculate the dust luminosity for scaling
        ldust = self.get_scaling(emitter, model, emissions)

        # Set up DL07 parameters using the extracted parameters
        self._setup_dl07_parameters(
            dust_mass, ldust, dust_to_gas_ratio, hydrogen_mass, pah_fraction
        )

        # Generate the base spectra
        sed = self._generate_dl07_spectra(lams)

        # Get the bolometric luminosity with proper units
        bol_lum = sed._bolometric_luminosity

        # Normalise the DL07 spectrum
        sed._lnu /= bol_lum
        lnu = sed._lnu

        # Handle per particle scaling (we need to expand the scaling shape)
        if model is not None and model.per_particle:
            if not hasattr(ldust, "shape"):
                # ldust is a float, need to convert to array
                ldust = np.full(emitter.nparticles, ldust)
            ldust = ldust[:, np.newaxis]

        # Properly handle units: normalize then scale
        result = lnu * ldust
        sed._lnu = result.value if hasattr(result, "value") else result

        return sed

    @accepts(line_lams=angstrom)
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
        emissions,
        spectra,
        redshift=0,
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
            emissions (dict):
                Dictionary containing all emissions generated so far.
            spectra (dict):
                Dictionary containing all spectra generated so far
                (used for scaling).
            redshift (float):
                The redshift at which to calculate the CMB heating. (Ignored
                if not applying CMB heating).

        Returns:
            LineCollection:
                The generated line collection.
        """
        # If we are missing this spectra then we cannot generate lines
        if model.label not in spectra:
            raise exceptions.MissingSpectraType(
                f"Cannot generate lines for {model.label} as spectra are "
                "missing. Please generate spectra first or remove the "
                "generation of lines."
            )
        else:
            sed = spectra[model.label]

        # Get the required parameters
        params = self._extract_params(model, emitter)

        # Unpack all dust parameters
        dust_mass = params["dust_mass"]
        dust_to_gas_ratio = params["dust_to_gas_ratio"]
        hydrogen_mass = params["hydrogen_mass"]
        pah_fraction = params["pah_fraction"]

        # Calculate the dust luminosity for scaling
        ldust = self.get_scaling(emitter, model, spectra)

        # Set up DL07 parameters using the extracted parameters
        self._setup_dl07_parameters(
            dust_mass, ldust, dust_to_gas_ratio, hydrogen_mass, pah_fraction
        )

        # Generate the DL07 function
        lnu = self._generate_dl07_spectra(line_lams).lnu
        norm_lnu = self._generate_dl07_spectra(sed.lam).lnu

        # Create an SED object for convenience
        norm_sed = Sed(lam=sed.lam, lnu=norm_lnu)

        # Get the bolometric luminosity with proper units
        bol_lum = norm_sed._bolometric_luminosity

        # Normalise the DL07 spectrum
        lnu /= bol_lum

        # Handle per particle scaling (we need to expand the scaling shape)
        if model is not None and model.per_particle:
            if not hasattr(ldust, "shape"):
                # ldust is a float, need to convert to array
                ldust = np.full(emitter.nparticles, ldust)
            ldust = ldust[:, np.newaxis]

        # Properly handle units: normalize then scale
        result = lnu * ldust
        lnu = result.value if hasattr(result, "value") else result

        # Return as LineCollection with continuum only
        lines = LineCollection(
            line_ids,
            line_lams,
            lum=np.zeros(lnu.shape) * erg / s,
            cont=lnu * erg / s / Hz,
        )

        return lines

    @property
    def u_avg(self) -> Optional[float]:
        """Average radiation field intensity <U>.

        This property provides access to the calculated average radiation field
        intensity, which is commonly referenced in the literature.

        Returns:
            float or None: The average radiation field intensity, or None if
                parameters have not been calculated yet.
        """
        return self.radiation_field_average

    @property
    def calculated_parameters(self) -> dict:
        """Dictionary of all calculated DL07 parameters.

        Returns:
            dict: Dictionary containing all calculated parameters with
                descriptive names.
        """
        return {
            "average_radiation_field": self.radiation_field_average,
            "minimum_radiation_field": self.umin_calculated,
            "gamma_parameter": self.gamma_calculated,
            "hydrogen_mass": self.hydrogen_mass_calculated,
            "pah_fraction_used": self.pah_fraction,
            "alpha_parameter": self.alpha,
            "power_per_unit_mass": self.power_per_unit_mass,
        }
