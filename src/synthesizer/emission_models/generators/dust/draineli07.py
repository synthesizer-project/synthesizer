"""A submodule defining Draine & Li 2007 dust emission generators."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from scipy.optimize import brentq
from unyt import Hz, Lsun, Msun, angstrom, erg, s, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.base_model import EmissionModel
from synthesizer.emission_models.generators.dust.dust_emission_base import (
    DustEmission,
)
from synthesizer.emission_models.utils import get_param
from synthesizer.emissions import LineCollection, Sed
from synthesizer.extensions.particle_spectra import compute_particle_seds
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

    Any of umin, gamma and alpha can be arrays (e.g. one value per
    particle).

    Args:
        umin (float/np.ndarray):
            Minimum radiation field.
        umax (float):
            Maximum radiation field.
        gamma (float/np.ndarray):
            Fraction of dust mass in power-law component.
        alpha (float/np.ndarray):
            The slope of the power-law component.

    Returns:
        float/np.ndarray:
            The mean radiation field.
    """
    umin = np.asarray(umin, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    # The general expression is singular at alpha = 1 and 2, where the
    # limits are used instead
    with np.errstate(divide="ignore", invalid="ignore"):
        general = (
            (alpha - 1.0)
            / (alpha - 2.0)
            * (umin ** (2.0 - alpha) - umax ** (2.0 - alpha))
            / (umin ** (1.0 - alpha) - umax ** (1.0 - alpha))
        )
    u_power_law = np.where(
        np.isclose(alpha, 1.0),
        (umax - umin) / np.log(umax / umin),
        np.where(
            np.isclose(alpha, 2.0),
            np.log(umax / umin) / (umin ** (-1) - umax ** (-1)),
            general,
        ),
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

    As in DraineLiGrainCurves, the templates are linearly (CIC) interpolated
    in qpah, umin and alpha, each in the space of the grid axis (linear or
    log10), so the spectrum varies smoothly with these parameters. Values
    outside the grid are clamped to its edges.

    dust_to_gas_ratio, hydrogen_mass, qpah, gamma, umin and alpha can be set
    on the generator, where they override the emitter, or read from the
    emitter. Each can be a single value or an array with one value per
    particle (for per-particle models); particles with non-finite values or
    a non-positive umin produce no emission. If they are set on neither,
    hydrogen_mass is derived from the dust mass and dust_to_gas_ratio, gamma
    and umin are derived from the Magdis+12 stacking results, and
    dust_to_gas_ratio, qpah and alpha fall back to 0.01, 0.025 and 2.0.

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

        # Check the grid and prepare the arrays we interpolate over once
        self._validate_grid()

    @timed("DraineLi07._validate_grid")
    def _validate_grid(self):
        """Validate that the dust grid matches class expectations."""
        # Ensure the grid holds both DL07 templates
        missing = {"diffuse", "pdr"} - set(self.grid.spectra)
        if missing:
            raise exceptions.InconsistentArguments(
                f"DraineLi07 requires a grid with {sorted(missing)} spectra."
            )

        # Ensure the grid's extraction axes are the DL07 parameters, recording
        # whether each is linear or logarithmic
        self._axis_names = []
        self._axis_logged = []
        for name in ("qpah", "umin", "alpha"):
            if name in self.grid._extract_axes:
                axis_name, logged = name, False
            elif f"log10{name}" in self.grid._extract_axes:
                axis_name, logged = f"log10{name}", True
            else:
                raise exceptions.UnimplementedFunctionality(
                    f"DraineLi07 requires a grid with a {name} or "
                    f"log10{name} extraction axis."
                )
            self._axis_names.append(axis_name)
            self._axis_logged.append(logged)

            # Ensure the axis values are numerically well behaved before we
            # use them for interpolation
            axis = self.grid._extract_axes_values[axis_name]
            if not np.all(np.isfinite(axis)):
                raise exceptions.InconsistentArguments(
                    f"The DraineLi07 grid {axis_name} axis must contain only "
                    "finite values."
                )
            if np.any(np.diff(axis) <= 0.0):
                raise exceptions.InconsistentArguments(
                    f"The DraineLi07 grid {axis_name} axis must be strictly "
                    "increasing."
                )

        # Ensure the templates have one entry per grid point (the diffuse
        # template does not depend on alpha)
        dims = tuple(
            len(self.grid._extract_axes_values[name])
            for name in self._axis_names
        )
        for key, shape in (("diffuse", dims[:2]), ("pdr", dims)):
            if self.grid.spectra[key].shape[:-1] != shape:
                raise exceptions.InconsistentArguments(
                    f"The DraineLi07 {key} spectra have shape "
                    f"{self.grid.spectra[key].shape[:-1]} over the grid "
                    f"axes, expected {shape}."
                )

        # Cache resampled grids keyed by wavelength tuple for quick lookup.
        self._grid_cache = {}

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

        # Any parameter can be an array (e.g. one value per particle), as
        # long as they all have the same length (or a single value)
        sizes = {
            np.size(value)
            for value in (
                dust_mass,
                ldust,
                effective_hydrogen_mass,
                effective_qpah,
                calculated_gamma,
                calculated_umin,
                effective_alpha,
            )
            if value is not None
        }
        nparticles = max(sizes)
        if not sizes <= {1, nparticles}:
            raise exceptions.InconsistentArguments(
                f"DraineLi07 parameters have lengths {sorted(sizes)}, but "
                f"expected 1 or {nparticles}."
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
            # so bracket the root rather than relying on a starting guess.
            # Entries without a usable <U> are left as NaN and masked out.
            # ponytail: one brentq per value, vectorise if per-particle
            # solves get slow
            if calculated_umin is None:
                u_avgs, gammas, alphas = np.broadcast_arrays(
                    calculated_u_avg, calculated_gamma, effective_alpha
                )
                umins = np.full(u_avgs.shape, np.nan)
                for i, (u_avg, g, a) in enumerate(
                    zip(u_avgs.flat, gammas.flat, alphas.flat)
                ):
                    if not (np.isfinite(u_avg) and u_avg > 0):
                        continue
                    if solve_umin(1e-10, umax, u_avg, g, a) > 0:
                        raise exceptions.InconsistentArguments(
                            "No umin gives the Magdis+12 mean radiation "
                            f"field <U>={u_avg} with gamma={g} and "
                            f"alpha={a}, since the power-law component "
                            "alone gives a higher <U>. Provide umin, or use "
                            "a smaller gamma or larger alpha."
                        )
                    umins.flat[i] = brentq(
                        solve_umin, 1e-10, 1e6, args=(umax, u_avg, g, a)
                    )
                calculated_umin = umins

        else:
            calculated_u_avg = u_mean(
                calculated_umin, umax, calculated_gamma, effective_alpha
            )

        # Broadcast everything to one value per particle, masking entries we
        # can't evaluate (they produce no emission)
        hydrogen_mass = np.broadcast_to(
            np.atleast_1d((calculated_hydrogen_mass / Msun).value),
            (nparticles,),
        )
        qpah_arr, umin_arr, alpha_arr, gamma_arr = (
            np.broadcast_to(
                np.atleast_1d(np.asarray(v, dtype=float)), (nparticles,)
            )
            for v in (
                effective_qpah,
                calculated_umin,
                effective_alpha,
                calculated_gamma,
            )
        )
        valid = (
            np.isfinite(qpah_arr)
            & np.isfinite(umin_arr)
            & (umin_arr > 0.0)
            & np.isfinite(alpha_arr)
            & np.isfinite(gamma_arr)
            & np.isfinite(hydrogen_mass)
        )

        return {
            "qpah": qpah_arr,
            "umin_calculated": umin_arr,
            "alpha": alpha_arr,
            "gamma_calculated": gamma_arr,
            "hydrogen_mass_calculated": hydrogen_mass * Msun,
            "radiation_field_average": calculated_u_avg,
            "valid": valid,
        }

    @timed("DraineLi07._get_resampled_grid")
    def _get_resampled_grid(self, lam):
        """Get or create a resampled grid for given wavelengths.

        This method caches resampled grids for performance when the same
        wavelengths are requested multiple times.

        Args:
            lam (unyt_array):
                The target wavelength array.

        Returns:
            Grid: The resampled grid.
        """
        lam_arr = np.atleast_1d(lam.to(angstrom).ndview)
        cache_key = tuple(lam_arr)

        # Check if we already have this configuration cached.
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        # Resample the base grid to the requested wavelengths
        if lam_arr.size == 1:
            # For scalar wavelengths, make a copy to avoid mutating the base
            # grid.
            grid = copy.deepcopy(self.grid)
            spectra_at_lam = grid.get_spectra_at_lam(lam)
            for spectra_id, spectra in spectra_at_lam.items():
                grid.spectra[spectra_id] = spectra[..., np.newaxis]
            grid.lam = lam_arr
            grid._ensure_spectra_data_contiguous()
        else:
            # For arrays of wavelengths
            grid = self.grid.reduce_rest_frame_lam(lam)

        # Cache the resampled grid.
        self._grid_cache[cache_key] = grid

        return grid

    @timed("DraineLi07._interpolate_templates")
    def _interpolate_templates(self, grid, setup: dict) -> tuple:
        """Interpolate the diffuse and PDR templates to the parameters.

        The templates are linearly interpolated (CIC) in qpah, umin and (for
        the PDR template) alpha, each in the space of the grid axis (linear
        or log10). Values outside the grid are clamped to its edges and
        masked entries are zero.

        Args:
            grid (Grid):
                The grid to interpolate (the native or a resampled grid).
            setup (dict):
                The parameters returned by ``_setup_dl07_parameters``.

        Returns:
            tuple:
                The diffuse and PDR templates, each (nparticles, nlam).
        """
        valid = setup["valid"]
        nparticles = valid.size
        dtype = grid._dtype

        # Map the parameters onto the grid axes, replacing masked entries
        # with a valid value so the axis transform stays finite
        grid_axes = []
        coords = []
        for name, axis_name, logged in zip(
            ("qpah", "umin_calculated", "alpha"),
            self._axis_names,
            self._axis_logged,
        ):
            axis = grid._extract_axes_values[axis_name]
            values = np.where(valid, setup[name], 1.0)
            grid_axes.append(np.ascontiguousarray(axis, dtype=dtype))
            coords.append(
                np.ascontiguousarray(
                    np.log10(values) if logged else values, dtype=dtype
                )
            )

        weights = np.ones(nparticles)
        templates = []
        for key, ndim in (("diffuse", 2), ("pdr", 3)):
            templates.append(
                compute_particle_seds(
                    np.ascontiguousarray(grid.spectra[key], dtype=dtype),
                    tuple(grid_axes[:ndim]),
                    tuple(coords[:ndim]),
                    weights,
                    np.array(grid.spectra[key].shape[:-1], dtype=np.int32),
                    ndim,
                    nparticles,
                    grid.nlam,
                    "cic",
                    1,
                    valid,
                    None,
                    False,
                    dtype,
                    tuple(self._axis_names[:ndim]),
                )
            )

        return tuple(templates)

    def _component_lnu(self, grid, setup: dict) -> tuple:
        """Get the diffuse and PDR lnu on a grid's wavelengths.

        Args:
            grid (Grid):
                The grid to evaluate on (the native or a resampled grid).
            setup (dict):
                The parameters returned by ``_setup_dl07_parameters``.

        Returns:
            tuple:
                The diffuse and PDR lnu (erg/s/Hz, without units), each
                (nparticles, nlam), or (nlam,) for a single set of
                parameters.
        """
        diffuse_template, pdr_template = self._interpolate_templates(
            grid, setup
        )
        gamma = setup["gamma_calculated"][:, np.newaxis]
        hydrogen_mass = (setup["hydrogen_mass_calculated"] / Msun).value[
            :, np.newaxis
        ]

        # Extract diffuse (old) component spectrum
        diffuse = (1.0 - gamma) * diffuse_template * hydrogen_mass

        # Extract PDR (young) component spectrum
        pdr = gamma * pdr_template * hydrogen_mass

        # Replace NaNs with zero for wavelength regimes with no values given
        diffuse[np.isnan(diffuse)] = 0.0
        pdr[np.isnan(pdr)] = 0.0

        # Preserve the previous single spectrum shape for scalar inputs
        if diffuse.shape[0] == 1:
            return diffuse[0], pdr[0]
        return diffuse, pdr

    def _bolometric_luminosity(self, setup: dict):
        """Integrate the spectrum over the template's native wavelengths.

        This normalises the spectrum independently of the wavelengths it is
        evaluated on.

        Args:
            setup (dict):
                The parameters returned by ``_setup_dl07_parameters``.

        Returns:
            float/np.ndarray:
                The bolometric luminosity (one per particle) in the
                luminosity unit of Sed.bolometric_luminosity, with a trailing
                axis to broadcast against lnu.
        """
        diffuse, pdr = self._component_lnu(self.grid, setup)
        bol_lum = Sed(
            lam=self.grid.lam,
            lnu=(diffuse + pdr) * (erg / s / Hz),
        )._bolometric_luminosity
        # Masked particles have no emission, so avoid dividing zero by zero
        bol_lum = np.asarray(bol_lum)
        bol_lum = np.where(bol_lum > 0.0, bol_lum, 1.0)
        return bol_lum[..., np.newaxis]

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
        # Evaluate on the grid resampled onto the requested wavelengths. The
        # resampled grid is a copy, so the grid always holds its native
        # wavelengths and can be shared between concurrently executing
        # models.
        grid = self._get_resampled_grid(lams)
        diffuse_sed, pdr_sed = (
            Sed(lam=lams, lnu=lnu * (erg / s / Hz))
            for lnu in self._component_lnu(grid, setup)
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

        # One spectrum per particle only makes sense for a per-particle model
        if setup["valid"].size > 1 and not getattr(
            model, "per_particle", False
        ):
            raise exceptions.InconsistentArguments(
                "DraineLi07 parameters with one value per particle need a "
                "per-particle model."
            )

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

        # One spectrum per particle only makes sense for a per-particle model
        if setup["valid"].size > 1 and not getattr(
            model, "per_particle", False
        ):
            raise exceptions.InconsistentArguments(
                "DraineLi07 parameters with one value per particle need a "
                "per-particle model."
            )

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
