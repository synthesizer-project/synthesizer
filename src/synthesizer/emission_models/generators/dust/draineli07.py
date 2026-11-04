"""A submodule defining Draine & Li 2007 dust emission generators."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from scipy.optimize import brentq
from spectres import spectres
from unyt import Hz, Lsun, Msun, angstrom, erg, s, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust.dust_emission_base import (
    DustEmission,
)
from synthesizer.emission_models.utils import get_param
from synthesizer.emissions import LineCollection, Sed
from synthesizer.extensions.grid_interpolation import interpolate_grid_array
from synthesizer.grid import Grid
from synthesizer.synth_warnings import warn
from synthesizer.units import accepts
from synthesizer.utils.operation_timers import timed
from synthesizer.utils.precision import verify_out_precision

if TYPE_CHECKING:
    from synthesizer.components.component import Component

# Parameters that can be set on the model, generator or emitter, but that
# aren't required (missing values are derived or fall back to a default)
_OPTIONAL_PARAMS = (
    "dust_to_gas_ratio",
    "hydrogen_mass",
    "qpah",
    "gamma",
    "umin",
    "alpha",
)

# Values used when a parameter isn't set anywhere
_DEFAULTS = {"dust_to_gas_ratio": 0.01, "qpah": 0.025, "alpha": 2.0}


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


def u_mean(
    umin: float,
    umax: float,
    gamma: float,
    alpha: float = 2.0,
) -> float:
    """Calculate the mean radiation field heating the dust.

    In the Draine & Li (2007) model a fraction (1 - gamma) of the dust mass
    is heated by umin, and the rest by a power law dM/dU ~ U^-alpha between
    umin and umax.

    Args:
        umin (float):
            Minimum radiation field.
        umax (float):
            Maximum radiation field.
        gamma (float):
            Fraction of dust mass in power-law component.
        alpha (float):
            The slope of the power-law component.

    Returns:
        float:
            The mean radiation field.
    """
    if np.isclose(alpha, 1.0):
        u_power_law = (umax - umin) / np.log(umax / umin)
    elif np.isclose(alpha, 2.0):
        u_power_law = np.log(umax / umin) / (umin ** (-1) - umax ** (-1))
    else:
        u_power_law = (
            (alpha - 1.0)
            / (alpha - 2.0)
            * (umin ** (2.0 - alpha) - umax ** (2.0 - alpha))
            / (umin ** (1.0 - alpha) - umax ** (1.0 - alpha))
        )
    return (1.0 - gamma) * umin + gamma * u_power_law


def solve_umin(
    umin: float,
    umax: float,
    u_avg: float,
    gamma: float,
    alpha: float = 2.0,
) -> float:
    """Solve for Umin in the Draine and Li model.

    Args:
        umin (float):
            Minimum radiation field.
        umax (float):
            Maximum radiation field.
        u_avg (float):
            Average radiation field.
        gamma (float):
            Fraction of dust mass in power-law component.
        alpha (float):
            The slope of the power-law component.

    Returns:
        float:
            The difference between calculated and target mean field.
    """
    return u_mean(umin, umax, gamma, alpha) - u_avg


class DraineLi07(DustEmission):
    """A generator for DL07 emission spectrum based on energy balance.

    This class will generate a DL07 spectrum using templates from a dust grid.
    It will then scale the spectrum based on the energy absorbed by dust,
    calculated as the difference between the intrinsic and attenuated
    emissions. The "energy balance" method.

    The templates are linearly interpolated in qpah, log10(umin) and alpha,
    so the spectrum varies smoothly with these parameters. Values outside the
    grid are clipped to its edges (with a warning).

    dust_to_gas_ratio, hydrogen_mass, qpah, gamma, umin and alpha can be set
    on the generator, where they override the emitter, or read from the
    emitter. If they are set on neither, hydrogen_mass is derived from the
    dust mass and dust_to_gas_ratio, gamma and umin are derived from the
    Magdis+12 stacking results, and dust_to_gas_ratio, qpah and alpha fall
    back to 0.01, 0.025 and 2.0.

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
        qpah (float):
            Fraction of dust mass in the form of PAHs.
        umin (float):
            Radiation field heating majority of the dust.
        alpha (float):
            The slope of the power-law part of the starlight intensity
            distribution.
        power_per_unit_mass (float):
            Power absorbed per unit dust mass in a radiation field with U = 1.
        verbose (bool):
            Whether to print verbose output.
    """

    grid: Grid
    dust_mass: Optional[unyt_quantity]
    dust_to_gas_ratio: Optional[float]
    hydrogen_mass: Optional[unyt_quantity]
    template: str
    gamma: Optional[float]
    qpah: Optional[float]
    umin: Optional[float]
    alpha: Optional[float]
    power_per_unit_mass: float
    verbose: bool

    @accepts(dust_mass=Msun.in_base("galactic"))
    def __init__(
        self,
        grid: Grid,
        dust_mass: Optional[unyt_quantity] = None,
        dust_to_gas_ratio: Optional[float] = None,
        hydrogen_mass: Optional[unyt_quantity] = None,
        template: str = "DL07",
        gamma: Optional[float] = None,
        qpah: Optional[float] = None,
        umin: Optional[float] = None,
        alpha: Optional[float] = None,
        power_per_unit_mass: float = 125.0,
        verbose: bool = True,
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
            dust_to_gas_ratio (float, optional):
                The dust-to-gas ratio of the galaxy. Read from the emitter
                if not provided, falling back to 0.01.
            hydrogen_mass (unyt_quantity, optional):
                The mass in hydrogen. Read from the emitter if not
                provided, otherwise calculated from dust_mass and
                dust_to_gas_ratio.
            template (str):
                The IR template model to be used.
            gamma (float, optional):
                Fraction of the dust mass that is associated with the
                power-law part of the starlight intensity distribution.
                Read from the emitter if not provided, falling back to 0.05.
            qpah (float, optional):
                Fraction of dust mass in the form of PAHs. Read from the
                emitter if not provided, falling back to 0.025.
            umin (float, optional):
                Radiation field heating majority of the dust. Read from the
                emitter if not provided, otherwise calculated from the
                Magdis+12 mean radiation field.
            alpha (float, optional):
                The slope of the power-law part of the starlight intensity
                distribution. Read from the emitter if not provided,
                falling back to 2.0.
            power_per_unit_mass (float):
                Power absorbed per unit dust mass in a radiation field
                with U = 1.
            verbose (bool):
                Whether to print verbose output.
            intrinsic (EmissionModel):
                The name of the intrinsic emission defining the unattenuated
                emission for energy balance.
            attenuated (EmissionModel):
                The name of the attenuated emission defining the attenuated
                emission for energy balance.
            scaler (EmissionModel):
                The name of the emission to use to scale the dust emission by.
        """
        # Call the parent init with required parameters and emissions. CMB
        # heating (da Cunha+13) heats a dust temperature, which the DL07
        # templates don't have, so it is never applied
        DustEmission.__init__(
            self,
            intrinsic,
            attenuated,
            scaler,
            do_cmb_heating=False,
            required_params=("dust_mass",),
        )

        self.grid = grid
        self.dust_mass = dust_mass
        self.dust_to_gas_ratio = dust_to_gas_ratio
        self.hydrogen_mass = hydrogen_mass
        self.template = template
        self.gamma = gamma
        self.qpah = qpah
        self.umin = umin
        self.alpha = alpha
        self.power_per_unit_mass = power_per_unit_mass
        self.verbose = verbose

    def __repr__(self):
        """Return a string representation of the DraineLi07 object."""
        # Start with base class repr components
        parts = []
        if self.is_energy_balance:
            parts.append(
                f"intrinsic={self._intrinsic.label}, "
                f"attenuated={self._attenuated.label}"
            )
        elif self.is_scaled:
            parts.append(f"scaler={self._scaler.label}")

        # Add DraineLi07 specific parameters
        parts.extend(
            [
                f"template={self.template}",
                f"qpah={self.qpah}",
                f"alpha={self.alpha}",
            ]
        )

        # Add optional parameters if set
        if self.dust_mass is not None:
            parts.append(f"dust_mass={self.dust_mass}")
        if self.gamma is not None:
            parts.append(f"gamma={self.gamma}")
        if self.umin is not None:
            parts.append(f"umin={self.umin}")

        return f"DraineLi07({', '.join(parts)})"

    def _extract_optional_params(self, model, emitter) -> dict:
        """Extract the parameters that aren't required.

        As in _extract_params, the priority is the model's fixed parameters,
        then the generator (if not None), then the emitter. Parameters that
        aren't found anywhere are None.

        Args:
            model (EmissionModel):
                The emission model generating the emission.
            emitter (Stars/Gas/BlackHole/Galaxy):
                The object emitting the emission.

        Returns:
            dict:
                The optional parameters.
        """
        model_fixed = model.fixed_parameters if model is not None else {}
        params = {}
        for param in _OPTIONAL_PARAMS:
            value = getattr(self, param) if param not in model_fixed else None
            if value is None:
                try:
                    value = get_param(
                        param, model, None, emitter, preserve_units=True
                    )
                except exceptions.MissingAttribute:
                    value = None
            params[param] = value
        return params

    def _setup_dl07_parameters(
        self,
        dust_mass: unyt_quantity,
        ldust: unyt_quantity,
        dust_to_gas_ratio: float = None,
        hydrogen_mass: unyt_quantity = None,
        qpah: float = None,
        gamma: float = None,
        umin: float = None,
        alpha: float = None,
    ) -> dict:
        """Set up the DL07 model parameters.

        Nothing is stored on the generator, since one generator can be shared
        by several models and emitters. Use the returned values to inspect the
        parameters of a particular call.

        Args:
            dust_mass (unyt_quantity):
                The dust mass.
            ldust (unyt_quantity):
                The dust luminosity for calculating mean radiation field.
            dust_to_gas_ratio (float, optional):
                The dust-to-gas ratio to use. If None, uses instance default.
            hydrogen_mass (unyt_quantity, optional):
                The hydrogen mass to use. If None, uses instance default.
            qpah (float, optional):
                The PAH fraction to use. If None, uses instance default.
            gamma (float, optional):
                The power-law dust mass fraction. If None, uses instance
                default.
            umin (float, optional):
                The minimum radiation field. If None, uses instance default.
            alpha (float, optional):
                The power-law slope. If None, uses instance default.

        Returns:
            dict:
                The parameters to interpolate the templates at (``qpah``,
                ``umin_calculated``, ``alpha``) and derived parameters
                (``gamma_calculated``, ``hydrogen_mass_calculated``,
                ``radiation_field_average``).
        """
        if self.template != "DL07":
            raise exceptions.UnimplementedFunctionality(
                f"{self.template} not a valid model!"
            )

        # Use the provided parameters, falling back to the instance values
        # and then the defaults
        def _resolve(name, value):
            if value is None:
                value = getattr(self, name)
            if value is None:
                value = _DEFAULTS.get(name)
            return value

        effective_dust_to_gas_ratio = _resolve(
            "dust_to_gas_ratio", dust_to_gas_ratio
        )
        effective_hydrogen_mass = _resolve("hydrogen_mass", hydrogen_mass)
        effective_qpah = _resolve("qpah", qpah)
        calculated_gamma = _resolve("gamma", gamma)
        calculated_umin = _resolve("umin", umin)
        effective_alpha = _resolve("alpha", alpha)

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

        if calculated_gamma is None or calculated_umin is None:
            warn(
                "Gamma or Umin for DL07 model not provided, computing the "
                "mean radiation field from the Magdis+2012 stacking results"
            )

            calculated_u_avg = u_mean_magdis12(
                (dust_mass / Msun).value,
                (ldust / Lsun).value,
                self.power_per_unit_mass,
            )

            if calculated_gamma is None:
                warn("Gamma not provided, choosing default gamma value as 5%")
                calculated_gamma = 0.05

            # The mean radiation field increases monotonically with umin,
            # so bracket the root rather than relying on a starting guess
            if calculated_umin is None:
                args = (
                    umax,
                    calculated_u_avg,
                    calculated_gamma,
                    effective_alpha,
                )
                if solve_umin(1e-10, *args) > 0:
                    raise exceptions.InconsistentArguments(
                        "No umin gives the Magdis+12 mean radiation field "
                        f"<U>={calculated_u_avg} with gamma="
                        f"{calculated_gamma} and alpha={effective_alpha}, "
                        "since the power-law component alone gives a "
                        "higher <U>. Provide umin, or use a smaller gamma "
                        "or larger alpha."
                    )
                calculated_umin = brentq(solve_umin, 1e-10, 1e6, args=args)

        else:
            calculated_u_avg = u_mean(
                calculated_umin, umax, calculated_gamma, effective_alpha
            )

        # Values outside the grid are clipped to its edges by the
        # interpolation, so say so
        for name, value in (
            ("qpah", effective_qpah),
            ("umin", calculated_umin),
            ("alpha", effective_alpha),
        ):
            axis = np.asarray(getattr(self.grid, name))
            if np.any(value < axis.min()) or np.any(value > axis.max()):
                warn(
                    f"DL07 {name}={value} is outside the grid range "
                    f"[{axis.min()}, {axis.max()}], using the grid edge."
                )

        return {
            "qpah": effective_qpah,
            "umin_calculated": calculated_umin,
            "alpha": effective_alpha,
            "gamma_calculated": calculated_gamma,
            "hydrogen_mass_calculated": calculated_hydrogen_mass,
            "radiation_field_average": calculated_u_avg,
        }

    def _interpolate_templates(self, setup: dict) -> tuple:
        """Interpolate the diffuse and PDR templates to the parameters.

        The templates are linearly interpolated in qpah, log10(umin) and
        (for the PDR template) alpha, on the grid's native wavelengths.
        Values outside the grid are clipped to its edges.

        Args:
            setup (dict):
                The parameters returned by ``_setup_dl07_parameters``.

        Returns:
            tuple:
                The diffuse and PDR templates on the native wavelengths.
        """
        dtype = self.grid._dtype

        def _as_axis(values):
            return np.ascontiguousarray(
                np.atleast_1d(np.asarray(values, dtype=float)), dtype=dtype
            )

        axes = (
            _as_axis(self.grid.qpah),
            _as_axis(np.log10(np.asarray(self.grid.umin, dtype=float))),
            _as_axis(self.grid.alpha),
        )
        coords = (
            _as_axis(setup["qpah"]),
            _as_axis(np.log10(setup["umin_calculated"])),
            _as_axis(setup["alpha"]),
        )
        nlam = len(self.grid._lam)

        # The diffuse template doesn't depend on alpha
        diffuse = interpolate_grid_array(
            np.ascontiguousarray(self.grid.spectra["diffuse"], dtype=dtype),
            axes[:2],
            coords[:2],
            2,
            1,
            nlam,
            "cic",
            1,
            dtype,
            ("qpah", "umin"),
        )[0]
        pdr = interpolate_grid_array(
            np.ascontiguousarray(self.grid.spectra["pdr"], dtype=dtype),
            axes,
            coords,
            3,
            1,
            nlam,
            "cic",
            1,
            dtype,
            ("qpah", "umin", "alpha"),
        )[0]

        return diffuse, pdr

    def _native_lnu(self, setup: dict) -> tuple:
        """Get the diffuse and PDR lnu on the grid's native wavelengths.

        Args:
            setup (dict):
                The parameters returned by ``_setup_dl07_parameters``.

        Returns:
            tuple:
                The diffuse and PDR lnu (erg/s/Hz, without units).
        """
        diffuse_template, pdr_template = self._interpolate_templates(setup)
        hydrogen_mass = (setup["hydrogen_mass_calculated"] / Msun).value

        # Extract diffuse (old) component spectrum
        diffuse = (
            (1.0 - setup["gamma_calculated"])
            * diffuse_template
            * hydrogen_mass
        )

        # Extract PDR (young) component spectrum
        pdr = setup["gamma_calculated"] * pdr_template * hydrogen_mass

        # Replace NaNs with zero for wavelength regimes with no values given
        diffuse[np.isnan(diffuse)] = 0.0
        pdr[np.isnan(pdr)] = 0.0

        return diffuse, pdr

    def _bolometric_luminosity(self, setup: dict) -> float:
        """Integrate the spectrum over the template's native wavelengths.

        This normalises the spectrum independently of the wavelengths it is
        evaluated on.

        Args:
            setup (dict):
                The parameters returned by ``_setup_dl07_parameters``.

        Returns:
            float:
                The bolometric luminosity in the luminosity unit of
                Sed.bolometric_luminosity.
        """
        diffuse, pdr = self._native_lnu(setup)
        return Sed(
            lam=self.grid._lam * angstrom,
            lnu=(diffuse + pdr) * (erg / s / Hz),
        )._bolometric_luminosity

    @accepts(lams=angstrom)
    def _generate_dl07_spectra(
        self,
        lams: unyt_array,
        setup: dict,
        dust_components: bool = False,
    ) -> Union[Sed, tuple[Sed, Sed]]:
        """Generate the DL07 dust emission spectra.

        Args:
            lams (unyt_array):
                The wavelength grid on which to generate the spectra.
            setup (dict):
                The parameters returned by ``_setup_dl07_parameters``.
            dust_components (bool):
                If True, returns the constituent dust components separately.

        Returns:
            Sed or tuple[Sed, Sed]:
                The generated dust emission SED(s).
        """
        # Resample the native spectra onto the requested wavelengths. This
        # reads the grid without modifying it, so the grid always holds its
        # native wavelengths and can be shared between concurrently executing
        # models.
        diffuse_sed, pdr_sed = (
            Sed(
                lam=lams,
                lnu=spectres(
                    lams.value, self.grid._lam, lnu, fill=0, verbose=False
                ).astype(self.grid._dtype, copy=False)
                * (erg / s / Hz),
            )
            for lnu in self._native_lnu(setup)
        )

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
        setup = self._setup_dl07_parameters(dust_mass, ldust)

        # Generate the base spectra
        return self._generate_dl07_spectra(lams, setup, dust_components)

    @accepts(lams=angstrom)
    @timed("DraineLi07._generate_spectra")
    @verify_out_precision()
    def _generate_spectra(
        self,
        lams: unyt_array,
        emitter: Component,
        model: EmissionModel,
        emissions: dict,
        redshift: float = 0,
        out_dtype=None,
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
                Unused, since DL07 templates don't support CMB heating.
            out_dtype (np.dtype):
                The output precision. Defaults to the global default output
                dtype.

        Returns:
            Sed:
                The generated dust emission SED.
        """
        # Get the required and optional parameters
        params = self._extract_params(model, emitter)
        params.update(self._extract_optional_params(model, emitter))
        dust_mass = params.pop("dust_mass")

        # Calculate the dust luminosity for scaling
        ldust = self.get_scaling(emitter, model, emissions)

        # Set up DL07 parameters using the extracted parameters
        setup = self._setup_dl07_parameters(dust_mass, ldust, **params)

        # Generate the base spectra
        sed = self._generate_dl07_spectra(lams, setup)

        # Get the bolometric luminosity over the native wavelengths
        bol_lum = self._bolometric_luminosity(setup)

        # Normalise the DL07 spectrum
        sed._lnu /= bol_lum
        lnu = sed._lnu

        # Scale the normalised emission, at the output precision
        sed._lnu = self.get_scaled_emission(
            lnu,
            ldust,
            emitter,
            model,
            out_dtype,
        )

        return sed

    @accepts(line_lams=angstrom)
    @verify_out_precision()
    def _generate_lines(
        self,
        line_ids,
        line_lams,
        emitter,
        model,
        emissions,
        spectra,
        redshift=0,
        out_dtype=None,
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
                Unused, since DL07 templates don't support CMB heating.
            out_dtype (np.dtype):
                The output precision. Defaults to the global default output
                dtype.

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

        # Get the required and optional parameters
        params = self._extract_params(model, emitter)
        params.update(self._extract_optional_params(model, emitter))
        dust_mass = params.pop("dust_mass")

        # Calculate the dust luminosity for scaling
        ldust = self.get_scaling(emitter, model, spectra)

        # Set up DL07 parameters using the extracted parameters
        setup = self._setup_dl07_parameters(dust_mass, ldust, **params)

        # Generate the DL07 function
        lnu = self._generate_dl07_spectra(line_lams, setup).lnu

        # Get the bolometric luminosity over the native wavelengths
        bol_lum = self._bolometric_luminosity(setup)

        # Normalise the DL07 spectrum
        lnu /= bol_lum

        # Scale the normalised emission, at the output precision
        lnu = self.get_scaled_emission(lnu, ldust, emitter, model, out_dtype)

        # Return as LineCollection with continuum only
        lines = LineCollection(
            line_ids,
            line_lams,
            lum=np.zeros(lnu.shape, dtype=lnu.dtype) * erg / s,
            cont=lnu * erg / s / Hz,
        )

        return lines
