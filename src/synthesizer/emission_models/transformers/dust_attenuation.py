"""A module containing dust attenuation functionality.

This module contains classes for dust attenuation laws. These classes
provide a way to calculate the optical depth and transmission curves
for a given dust model.

Example usage::

    # Create a power law dust model with a slope of -1.0
    dust_model = PowerLaw(slope=-1.0)

    # Calculate the transmission curve
    transmission = dust_model.get_transmission(
        0.33, np.linspace(1000, 10000, 1000)
    )
"""

import copy
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from dust_extinction import grain_models
from scipy import interpolate
from unyt import (
    Msun,
    angstrom,
    cm,
    g,
    pc,
    um,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.emission_models.transformers.transformer import Transformer
from synthesizer.extensions.particle_spectra import compute_particle_seds
from synthesizer.extensions.timers import tic, toc
from synthesizer.grid import Grid
from synthesizer.synth_warnings import warn
from synthesizer.units import accepts

this_dir, this_filename = os.path.split(__file__)

__all__ = [
    "PowerLaw",
    "MWN18",
    "Calzetti2000",
    "GrainModels",
    "ParametricLi08",
    "DraineLiGrainCurves",
]

_RESET_SENTINEL = object()
_DRAINE_LI_MEAN_MOLECULAR_WEIGHT = 1.4
_HYDROGEN_MASS = 1.6738e-24 * g
_GAS_MASS_PER_H = (_DRAINE_LI_MEAN_MOLECULAR_WEIGHT * _HYDROGEN_MASS).to(Msun)


class AttenuationLaw(Transformer):
    """The base class for all attenuation laws.

    A child of this class should define its own get_tau method with any
    model specific behaviours. This will be used by get_transmission (which
    itself can be overloaded by the child if needed).

    Attributes:
        description (str):
            A description of the type of model. Defined on children classes.
        required_params (tuple):
            The name of any required parameters needed by the transformer
            when transforming an emission. These should either be
            available from an emitter or from the EmissionModel itself.
            If they are missing an exception will be raised.
    """

    def __init__(
        self, description, required_params=("tau_v",), require_tau_v=True
    ):
        """Initialise the parent and set common attributes.

        Args:
            description (str):
                A description of the type of model.
            required_params (tuple):
                List of required model attributes.
            require_tau_v (bool):
                Do we really need tau_v?
        """
        # Store the description of the model.
        self.description = description
        # Store user-supplied conversions between model arguments and parameter
        # names, e.g. allows the user to set e.g. slope = 'slope_young' on the
        # emitter or model and have that passed to the dust curve as slope
        self._name_transforms = {}
        # Stores overridden parameters temporarily
        self._temp_params = {}
        if ("tau_v" not in required_params) and (require_tau_v is True):
            raise exceptions.InconsistentArguments(
                "AttenuationLaw requires 'tau_v' as a parameter."
            )
        # Call the parent constructor
        Transformer.__init__(self, required_params=required_params)

    def __repr__(self):
        """Return a string representation of the AttenuationLaw object."""
        return f"{self.__class__.__name__}({self.description})"

    def get_tau(self, *args):
        """Compute the V-band normalised optical depth."""
        raise exceptions.UnimplementedFunctionality(
            "AttenuationLaw should not be instantiated directly!"
            " Instead use one to child models (" + ", ".join(__all__) + ")"
        )

    def get_tau_at_lam(self, *args):
        """Compute the optical depth at wavelength."""
        raise exceptions.UnimplementedFunctionality(
            "AttenuationLaw should not be instantiated directly!"
            " Instead use one to child models (" + ", ".join(__all__) + ")"
        )

    @accepts(lam=angstrom)
    def get_transmission(self, tau_v, lam, **dust_curve_kwargs):
        """Compute the transmission curve.

        Returns the transmitted flux/luminosity fraction based on an optical
        depth at a range of wavelengths.

        Args:
            tau_v (float/np.ndarray of float):
                Optical depth in the V-band. Can either be a single float or
                array.
            lam (np.ndarray of float):
                The wavelengths (with units) at which to calculate
                transmission.
            **dust_curve_kwargs (dict):
                Additional keyword arguments to be passed to the dust curve
                which have been defined on the emitter or model.

        Returns:
            np.ndarray of float:
                The transmission at each wavelength. Either (lam.size,) in
                shape for singular tau_v values or (tau_v.size, lam.size)
                tau_v is an array.
        """
        # Set any additional parameters on the dust curve
        self._set_params(**dust_curve_kwargs)

        try:
            # Get the optical depth at each wavelength
            tau_x_v = self.get_tau(lam)
        finally:
            # Always restore previous state
            self._reset_params()

        # Include the V band optical depth in the exponent
        # For a scalar we can just multiply but for an array we need to
        # broadcast
        if np.isscalar(tau_v):
            exponent = tau_v * tau_x_v
        else:
            if np.ndim(lam) == 0:
                exponent = tau_v[:] * tau_x_v
            else:
                exponent = tau_v[:, None] * tau_x_v

        return np.exp(-exponent)

    def _check_required_params(self):
        """Get the required parameters for the transformer.

        Given the input params, this method will return the required
        params by looking at required params which have been
        set as strings or None.

        """
        param_values = [
            getattr(self, param, None) for param in self._required_params
        ]

        required_params = []
        for param, value in zip(self._required_params, param_values):
            if value is None:
                required_params.append(param)
            elif isinstance(value, str):
                required_params.append(value)
                self._name_transforms[value] = param

        self._required_params = required_params

    def _transform(self, emission, emitter, model, mask, lam_mask):
        """Apply the dust attenuation to the emission.

        Args:
            emission (Line/Sed): The emission to transform.
            emitter (Stars/Gas/BlackHole/Galaxy): The object emitting the
                emission.
            model (EmissionModel): The emission model generating the emission.
            mask (np.ndarray): The mask to apply to the emission.
            lam_mask (np.ndarray): We must define this parameter in the
                transformer method, but it is not used in this case. If not
                None an error will be raised.

        Returns:
            Line/Sed: The transformed emission.
        """
        # Extract the required parameters
        params = self._extract_params(model, emission, emitter)

        # Ensure we aren't trying to use a wavelength mask
        if lam_mask is not None:
            raise exceptions.UnimplementedFunctionality(
                "Wavelength mask currently not supported in dust attenuation."
            )

        # Apply the transmission to the emission
        return emission.apply_attenuation(
            dust_curve=self,
            mask=mask,
            **params,
        )

    def _set_params(self, **params):
        """Set the parameters of the dust curve.

        This method will set any parameters defined in params as attributes
        of the dust curve. This allows for parameters to be set on the
        emitter or model and then used by the dust curve.

        Args:
            **params (dict):
                The parameters to set.
        """
        # Save existing state of only the attributes we will override,
        # then apply overrides mapped to the actual attribute names.
        self._temp_params = {}
        overrides = {}
        for key, value in params.items():
            attr = self._name_transforms.get(key, key)
            overrides[attr] = value
        for attr, value in overrides.items():
            prev = getattr(self, attr, _RESET_SENTINEL)
            self._temp_params[attr] = prev
            setattr(self, attr, value)

    def _reset_params(self):
        """Reset the parameters of the dust curve to their previous state."""
        for attr, prev in self._temp_params.items():
            if prev is _RESET_SENTINEL:
                # Attribute did not exist prior to override
                if hasattr(self, attr):
                    delattr(self, attr)
            else:
                setattr(self, attr, prev)
        self._temp_params = {}

    @accepts(lam=angstrom)
    def plot_attenuation(
        self,
        lam,
        fig=None,
        ax=None,
        label=None,
        figsize=(8, 6),
        show=True,
        **kwargs,
    ):
        """Plot the attenuation curve.

        Args:
            lam (np.ndarray of float):
                The wavelengths (with units) at which to calculate
                transmission.
            fig (matplotlib.figure.Figure):
                The figure to plot on. If None, a new figure will be created.
            ax (matplotlib.axes.Axes):
                The axis to plot on. If None, a new axis will be created.
            label (str):
                The label to use for the plot.
            figsize (tuple):
                The size of the figure to create if fig is None.
            show (bool):
                Whether to show the plot.
            **kwargs (dict):
                Keyword arguments to be provided to the `plot` call.

        Returns:
            fig, ax:
                The figure and axis objects.
        """
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        # Get the attenuation curve
        a_V = self.get_tau(lam)

        # Plot the transmission curve
        ax.plot(lam, a_V, label=label, **kwargs)

        # Add labels
        ax.set_xlabel(r"$\lambda/(\AA)$")
        ax.set_ylabel(r"A$_{\lambda}/$A$_{V}$")

        ax.set_yticks(np.arange(0, 10))
        ax.set_xlim(np.min(lam), np.max(lam))
        ax.set_ylim(0.0, 10)

        # Add a legend if the ax has labels to plot
        if any(ax.get_legend_handles_labels()[1]):
            ax.legend()

        # Show the plot
        if show:
            plt.show()

        return fig, ax

    @accepts(lam=angstrom)
    def plot_transmission(
        self,
        tau_v,
        lam,
        fig=None,
        ax=None,
        label=None,
        figsize=(8, 6),
        show=True,
    ):
        """Plot the transmission curve.

        Args:
            tau_v (float/np.ndarray of float):
                Optical depth in the V-band. Can either be a single float or
                array.
            lam (np.ndarray of float):
                The wavelengths (with units) at which to calculate
                transmission.
            fig (matplotlib.figure.Figure):
                The figure to plot on. If None, a new figure will be created.
            ax (matplotlib.axes.Axes):
                The axis to plot on. If None, a new axis will be created.
            label (str):
                The label to use for the plot.
            figsize (tuple):
                The size of the figure to create if fig is None.
            show (bool):
                Whether to show the plot.

        Returns:
            fig, ax:
                The figure and axis objects.
        """
        if fig is None:
            fig = plt.figure(figsize=figsize)
        if ax is None:
            ax = fig.add_subplot(111)

        # Get the transmission curve
        transmission = self.get_transmission(tau_v, lam)

        # Plot the transmission curve
        ax.plot(lam, transmission, label=label)

        # Add labels
        ax.set_xlabel("Wavelength (Angstrom)")
        ax.set_ylabel("Transmission")

        # Add a legend if the ax has labels to plot
        if any(ax.get_legend_handles_labels()[1]):
            ax.legend()

        # Show the plot
        if show:
            plt.show()

        return fig, ax


class PowerLaw(AttenuationLaw):
    """Custom power law dust curve.

    Attributes:
        slope (float):
            The slope of the power law.
    """

    def __init__(self, slope=-1.0):
        """Initialise the power law slope of the dust curve.

        Args:
            slope (float):
                The slope of the power law dust curve.
        """
        description = "simple power law dust curve"
        AttenuationLaw.__init__(
            self, description, required_params=("tau_v", "slope")
        )
        self.slope = slope

        self._check_required_params()

    def __repr__(self):
        """Return a string representation of the PowerLaw object."""
        return f"PowerLaw(slope={self.slope})"

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam):
        """Calculate optical depth at a wavelength.

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/np.ndarray of float: The optical depth.
        """
        return (lam / (5500.0 * angstrom)) ** self.slope

    @accepts(lam=angstrom)
    def get_tau(self, lam):
        """Calculate V-band normalised optical depth.

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths.

        Returns:
            float/np.ndarray of float: The optical depth.
        """
        return self.get_tau_at_lam(lam) / self.get_tau_at_lam(
            5500.0 * angstrom
        )


@accepts(lam=angstrom, cent_lam=angstrom, gamma=angstrom)
def N09Tau(lam, slope, cent_lam, ampl, gamma):
    """Generate the transmission curve for the Noll+2009 attenuation curve.

    Attenuation curve using a modified version of the Calzetti
    attenuation (Calzetti+2000) law allowing for a varying UV slope
    and the presence of a UV bump; from Noll+2009

    References:
        https://ui.adsabs.harvard.edu/abs/2009A%26A...499...69N

    Args:
        lam (np.ndarray of float):
            The input wavelength array (expected in AA units,
            global unit).
        slope (float):
            The slope of the attenuation curve.
        cent_lam (float):
            The central wavelength of the UV bump, expected in AA.
        ampl (float):
            The amplitude of the UV-bump.
        gamma (float):
            The width (FWHM) of the UV bump, in AA.

    Returns:
        np.ndarray of float: V-band normalised optical depth for
            given wavelength
    """
    # Performing some unit conversions to match the
    # Calzetti curve units which are in um
    _lam = np.linspace(0.01, 3.0, 10000, endpoint=True) * um
    _cent_lam = cent_lam.to("um")
    _gamma = gamma.to("um")
    lam_v = 0.55  # in um

    k_lam = np.zeros_like(_lam.value)

    # Masking for different regimes in the Calzetti curve
    ok1 = (_lam >= 0.12) * (_lam < 0.63)  # 0.12um<=lam<0.63um
    ok2 = (_lam >= 0.63) * (_lam < 3.1)  # 0.63um<=lam<=3.10um
    ok3 = _lam < 0.12  # lam<0.12um
    if np.sum(ok1) > 0:  # equation 1
        k_lam[ok1] = (
            -2.156
            + (1.509 / _lam.value[ok1])
            - (0.198 / _lam.value[ok1] ** 2)
            + (0.011 / _lam.value[ok1] ** 3)
        )
        func = interpolate.interp1d(
            _lam.value[ok1], k_lam[ok1], fill_value="extrapolate"
        )
    else:
        func = None
    if np.sum(ok2) > 0:  # equation 2
        k_lam[ok2] = -1.857 + (1.040 / _lam.value[ok2])
    if np.sum(ok3) > 0:
        # This will never be none
        if func is None:
            raise exceptions.InconsistentArguments("No data in the UV-optical")
        # Extrapolating the 0.12um<=lam<0.63um regime
        k_lam[ok3] = func(_lam.value[ok3])

    # Using the Calzetti attenuation curve normalised
    # to Av=4.05
    k_lam = 4.05 + 2.659 * k_lam
    k_v = 4.05 + 2.659 * (
        -2.156 + (1.509 / lam_v) - (0.198 / lam_v**2) + (0.011 / lam_v**3)
    )

    # UV bump feature expression from Noll+2009
    D_lam = (
        ampl
        * ((_lam * _gamma) ** 2)
        / ((_lam**2 - _cent_lam**2) ** 2 + (_lam * _gamma) ** 2)
    )

    # Normalising with the value at 0.55um, to obtain
    # normalised optical depth
    tau_x_v = (k_lam + D_lam) / k_v
    tau_x = tau_x_v * (_lam.value / lam_v) ** slope

    func = interpolate.interp1d(
        _lam, tau_x, bounds_error=False, fill_value=(tau_x[0], tau_x[-1])
    )

    return func(lam.to("um"))


class Calzetti2000(AttenuationLaw):
    """Calzetti attenuation curve.

    This includes options for the slope and UV-bump implemented in
    Noll et al. 2009.

    Attributes:
        slope (float):
            The slope of the attenuation curve.

        cent_lam (float):
            The central wavelength of the UV bump, expected in AA.

        ampl (float):
            The amplitude of the UV-bump.

        gamma (float):
            The width (FWHM) of the UV bump, in AA.

    """

    @accepts(cent_lam=angstrom, gamma=angstrom)
    def __init__(
        self,
        slope=0,
        cent_lam=2175 * angstrom,
        ampl=0,
        gamma=350 * angstrom,
    ):
        """Initialise the dust curve.

        Args:
            slope (float):
                The slope of the attenuation curve.

            cent_lam (float):
                The central wavelength of the UV bump, expected in AA.

            ampl (float):
                The amplitude of the UV-bump.

            gamma (float):
                The width (FWHM) of the UV bump, in AA.
        """
        description = (
            "Calzetti attenuation curve; with option"
            "for the slope and UV-bump implemented"
            "in Noll et al. 2009"
        )

        required_params = ("tau_v", "slope", "cent_lam", "ampl", "gamma")

        AttenuationLaw.__init__(self, description, required_params)

        # Define the parameters of the model.
        self.slope = slope
        self.cent_lam = cent_lam
        self.ampl = ampl
        self.gamma = gamma

        self._check_required_params()

    def __repr__(self):
        """Return a string representation of the Calzetti2000 object."""
        parts = [
            f"slope={self.slope}",
            f"cent_lam={self.cent_lam}",
            f"ampl={self.ampl}",
            f"gamma={self.gamma}",
        ]
        return f"Calzetti2000({', '.join(parts)})"

    @accepts(lam=angstrom)
    def get_tau(self, lam):
        """Calculate V-band normalised optical depth.

        (Uses the N09Tau function defined above.)

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float:
                The V-band noramlised optical depth.
        """
        return N09Tau(
            lam=lam,
            slope=self.slope,
            cent_lam=self.cent_lam,
            ampl=self.ampl,
            gamma=self.gamma,
        )

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam):
        """Calculate optical depth at a wavelength.

        (Uses the N09Tau function defined above.)

        Args:
            lam (float/array-like, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float
                The optical depth.
        """
        tau_x_v = N09Tau(
            lam=lam,
            slope=self.slope,
            cent_lam=self.cent_lam,
            ampl=self.ampl,
            gamma=self.gamma,
        )

        # V-band wavelength in micron
        # Since N09Tau uses micron for calculations
        lam_v = 0.55

        k_v = 4.05 + 2.659 * (
            -2.156 + (1.509 / lam_v) - (0.198 / lam_v**2) + (0.011 / lam_v**3)
        )

        return k_v * tau_x_v


class MWN18(AttenuationLaw):
    """Milky Way attenuation curve used in Narayanan+2018.

    Attributes:
        data (np.ndarray of float):
            The data describing the dust curve, loaded from MW_N18.npz.
        tau_lam_v (float):
            The V band optical depth.
    """

    def __init__(self):
        """Initialise the dust curve.

        This will load the data and get the V band optical depth by
        interpolation.
        """
        description = "MW extinction curve from Desika"
        AttenuationLaw.__init__(self, description)
        self.data = np.load(f"{this_dir}/../../data/MW_N18.npz")
        self.tau_lam_v = np.interp(
            5500.0, self.data.f.mw_df_lam[::-1], self.data.f.mw_df_chi[::-1]
        )

    def __repr__(self):
        """Return a string representation of the MWN18 object."""
        return "MWN18()"

    @accepts(lam=angstrom)
    def get_tau(self, lam, interp="cubic"):
        """Calculate V-band normalised optical depth.

        Args:
            lam (float/array, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array, float: The optical depth.
        """
        func = interpolate.interp1d(
            self.data.f.mw_df_lam[::-1],
            self.data.f.mw_df_chi[::-1],
            kind=interp,
            fill_value="extrapolate",
        )
        return func(lam) / self.tau_lam_v

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam, interp="cubic"):
        """Calculate the optical depth at a wavelength.

        Args:
            lam (float/array, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array, float
                The optical depth.
        """
        func = interpolate.interp1d(
            self.data.f.mw_df_lam[::-1],
            self.data.f.mw_df_chi[::-1],
            kind=interp,
            fill_value="extrapolate",
        )
        return func(lam)


class GrainModels(AttenuationLaw):
    """Grain model dust attenuation curves.

    These models are based on dust grain size, composition, and shape
    distributions constrained by observations of extinction, abundances,
    emission, and polarization. Some of these models can be used to
    estimate extinction at wavelengths inaccessible to observations
    (e.g., extreme UV below 912 Å). These models are taken from the
    astropy affiliated dust-extinction package
    (https://dust-extinction.readthedocs.io/en/latest/ for details).
    By default, we will use the Weingarter & Draine 2001 (WD01) models,
    and the submodel for the SMC bar.)

    Attributes:
        model (str):
            The dust grain model used.
            Available models are:
                DBP90: Desert, Boulanger, & Puget 1990, A&A, 237, 215
                WD01: Weingartner & Draine 2001, ApJ, 548, 296
                D03: Draine 2003, ARA&A, 41, 241; Draine 2003, ApJ, 598, 1017
                ZDA04: Zubko, Dwek, & Arendt 2004, ApJS, 152, 211
                C11: Compiegne et al. 2011, A&A, 525, 103
                J13: Jones et al. 2013, A&A, 558, 62
                HD23: Hensley & Draine 2023, ApJ, 948, 55
                Y24: Ysard et al. 2024, A&A, 684, 34
        submodel (str):
                The submodel to use within the main grain model.
                The submodels available for the different models
                listed below. All of them are self-explanatory with
                the RV defining the normalisation of the extinction
                curve, where RV = AV / E(B-V).
                DBP90: MWRV31
                WD01 MWRV31, MWRV40, MWRV55, LMCAvg, LMC2, SMCBar
                D03: MWRV31, MWRV40, MWRV55
                ZDA04: MWRV31
                C11: MWRV31
                J13: MWRV31
                HD23: MWRV31
                Y24: MWRV31
    """

    def __init__(self, model: str = "WD01", submodel: str = "SMCBar"):
        """Initialise the dust curve.

        Args:
            model (str):
                The dust grain model to use.
                Available models are:
                DBP90: Desert, Boulanger, & Puget 1990, A&A, 237, 215
                WD01: Weingartner & Draine 2001, ApJ, 548, 296
                D03: Draine 2003, ARA&A, 41, 241; Draine 2003, ApJ, 598, 1017
                ZDA04: Zubko, Dwek, & Arendt 2004, ApJS, 152, 211
                C11: Compiegne et al. 2011, A&A, 525, 103
                J13: Jones et al. 2013, A&A, 558, 62
                HD23: Hensley & Draine 2023, ApJ, 948, 55
                Y24: Ysard et al. 2024, A&A, 684, 34
            submodel (str):
                The submodel to use within the main grain model.
                The submodels available for the different models
                listed below. All of them are self-explanatory with
                the RV defining the normalisation of the extinction
                curve, where RV = AV / E(B-V).
                DBP90: MWRV31
                WD01 MWRV31, MWRV40, MWRV55, LMCAvg, LMC2, SMCBar
                D03: MWRV31, MWRV40, MWRV55
                ZDA04: MWRV31
                C11: MWRV31
                J13: MWRV31
                HD23: MWRV31
                Y24: MWRV31
        """
        AttenuationLaw.__init__(
            self,
            "Dust grain models from dust-extinction package",
        )
        available_models = {
            "DBP90": ["MWRV31"],
            "WD01": ["MWRV31", "MWRV40", "MWRV55", "LMCAvg", "LMC2", "SMCBar"],
            "D03": ["MWRV31", "MWRV40", "MWRV55"],
            "ZDA04": ["MWRV31"],
            "C11": ["MWRV31"],
            "J13": ["MWRV31"],
            "HD23": ["MWRV31"],
            "Y24": ["MWRV31"],
        }
        if model not in available_models:
            raise exceptions.InconsistentArguments(
                f"Model '{model}' not recognized. Available models are: "
                f"{', '.join(available_models.keys())}"
            )
        self.model = model
        # Get the correct model string if model is WD01
        if model == "WD01":
            alias_map = {
                "MW": "MWRV31",
                "LMC": "LMCAvg",
                "SMC": "SMCBar",
            }
            self.submodel = alias_map.get(submodel.upper(), submodel)
        else:
            self.submodel = submodel

        if self.submodel not in available_models[model]:
            raise exceptions.InconsistentArguments(
                f"Submodel '{submodel}' not recognized for model '{model}'. "
                f"Available submodels are: "
                f"{', '.join(available_models[model])}"
            )
        # Initialise the grain model and its submodel
        self.extmodel = getattr(grain_models, self.model)(self.submodel)

    def __repr__(self):
        """Return a string representation of the GrainModels object."""
        return f"GrainModels(model={self.model}, submodel={self.submodel})"

    @accepts(lam=angstrom)
    def get_tau(self, lam, interp="slinear"):
        """Calculate V-band normalised optical depth.

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/np.ndarray of float: The optical depth.
        """
        lam_v = 5500 * angstrom  # V-band wavelength
        out = self.get_tau_at_lam(lam, interp=interp) / self.get_tau_at_lam(
            lam_v, interp=interp
        )

        return out

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam, interp="slinear"):
        """Calculate optical depth at a wavelength.

        Args:
            lam (float/array-like, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            interp (str):
                The type of interpolation to use. Can be 'linear', 'nearest',
                'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic',
                'previous', or 'next'. 'zero', 'slinear', 'quadratic' and
                'cubic' refer to a spline interpolation of zeroth, first,
                second or third order. Uses scipy.interpolate.interp1d.

        Returns:
            float/array-like, float
                The optical depth.
        """
        # Inverse wavelength range in 1/um
        _inverse_lam_range = self.extmodel.data_x * 1 / um
        _lam_range = (1 / _inverse_lam_range).to("angstrom")
        # Change to increasing order
        _lam_range = np.unique(_lam_range[::-1])
        _lam = np.atleast_1d(lam.to("angstrom").value) * angstrom
        if np.any((_lam < np.min(_lam_range)) | (_lam > np.max(_lam_range))):
            warn(
                f"Wavelengths outside the range "
                f"{np.min(_lam_range):.1f} - {np.max(_lam_range):.1f} "
                f"Values are being extrapolated.",
                RuntimeWarning,
            )
            # Remove the first and last points to avoid edge effects
            lower = self.extmodel(_lam_range[0].to_astropy())
            upper = self.extmodel(_lam_range[-1].to_astropy())
            func = interpolate.interp1d(
                _lam_range.to("Angstrom").value,
                self.extmodel(_lam_range.to_astropy()),
                kind=interp,
                bounds_error=False,
                fill_value=(lower, upper),
            )
            out = func(_lam.to("Angstrom").value)
        else:
            out = self.extmodel(lam.to_astropy())

        return out


@accepts(lam=angstrom)
def Li08(lam, UV_slope, OPT_NIR_slope, FUV_slope, bump, model):
    """Drude-like parametric expression for the attenuation curve from Li+08.

    Args:
        lam (np.ndarray of float):
            The wavelengths (AA units) at which to calculate transmission.
        UV_slope (float):
            Dimensionless parameter describing the UV-FUV slope
        OPT_NIR_slope (float):
            Dimensionless parameter describing the optical/NIR slope
        FUV_slope (float):
            Dimensionless parameter describing the FUV slope
        bump (float):
            Dimensionless parameter describing the UV bump
            strength (0< bump <1)
        model (str):
            Via this parameter one can choose one of the templates for
            extinction/attenuation curves from: Calzetti, SMC, MW (R_V=3.1),
            and LMC

    Returns:
            np.ndarray of float/float: tau/tau_v at each input wavelength (lam)
    """
    # Empirical templates (Calzetti, SMC, MW RV=3.1, LMC)
    if model == "Calzetti":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 44.9, 7.56, 61.2, 0.0
    if model == "SMC":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 38.7, 3.83, 6.34, 0.0
    if model == "MW":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 14.4, 6.52, 2.04, 0.0519
    if model == "LMC":
        UV_slope, OPT_NIR_slope, FUV_slope, bump = 4.47, 2.39, -0.988, 0.0221

    # Converting lam from AA to um for ease
    _lam = lam.to("um")

    # Attenuation curve (normalized to Av)
    term1 = UV_slope / (
        (_lam.value / 0.08) ** OPT_NIR_slope
        + (_lam.value / 0.08) ** -OPT_NIR_slope
        + FUV_slope
    )
    term2 = (
        233.0
        * (
            1
            - UV_slope
            / (6.88**OPT_NIR_slope + 0.145**OPT_NIR_slope + FUV_slope)
            - bump / 4.6
        )
    ) / ((_lam.value / 0.046) ** 2.0 + (_lam.value / 0.046) ** -2.0 + 90.0)
    term3 = bump / (
        (_lam.value / 0.2175) ** 2.0 + (_lam.value / 0.2175) ** -2.0 - 1.95
    )

    AlamAV = term1 + term2 + term3

    return AlamAV


class ParametricLi08(AttenuationLaw):
    """Parametric, empirical attenuation curve.

    Implemented in Li+08, Evolution of the parameters up to high-z
    (z=12) studied in: Markov+23a,b

    Attributes:
        UV_slope (float):
            Dimensionless parameter describing the UV-FUV slope (0, 50)
        OPT_NIR_slope (float):
            Dimensionless parameter describing the optical/NIR slope (0, 10)
        FUV_slope (float):
            Dimensionless parameter describing the FUV slope (-1, 75)
        bump (float):
            Dimensionless parameter describing the UV bump
            strength (-0.005< bump <0.06)
        model (str):
            Fixing attenuation/extinction curve to one of the known
            templates: MW, SMC, LMC, Calzetti

    """

    def __init__(
        self,
        UV_slope=44.9,
        OPT_NIR_slope=7.56,
        FUV_slope=61.2,
        bump=0.0,
        model="Calzetti",
    ):
        """Initialise the dust curve.

        Args:
            UV_slope (float):
                Dimensionless parameter describing the UV-FUV slope (0, 50)
            OPT_NIR_slope (float):
                Dimensionless parameter describing the optical/NIR
                slope (0, 10)
            FUV_slope (float):
                Dimensionless parameter describing the FUV slope (-1, 75)
            bump (float):
                Dimensionless parameter describing the UV bump
                strength (-0.005< bump <0.06)
            model (str):
                Fixing attenuation/extinction curve to one of the known
                templates: MW, SMC, LMC, Calzetti
        """
        description = (
            "Parametric attenuation curve; with option"
            "for multiple slopes (UV_slope,OPT_NIR_slope,FUV_slope) and "
            "varying UV-bump strength (bump)."
            "Introduced in Li+08, see Markov+23,24 for application to "
            "a high-z dataset."
            "Empirical extinction/attenuation curves (MW, SMC, LMC, "
            "Calzetti) can be selected."
        )

        required_params = (
            "tau_v",
            "UV_slope",
            "OPT_NIR_slope",
            "FUV_slope",
            "bump",
        )

        AttenuationLaw.__init__(self, description, required_params)

        # Define the parameters of the model.
        self.UV_slope = UV_slope
        self.OPT_NIR_slope = OPT_NIR_slope
        self.FUV_slope = FUV_slope
        self.bump = bump

        # Get the correct model string
        if model == "MW":
            self.model = "MW"
        elif model == "LMC":
            self.model = "LMC"
        elif model == "SMC":
            self.model = "SMC"
        elif model == "Calzetti":
            self.model = "Calzetti"
        else:
            self.model = "Custom"

        self._check_required_params()

    def __repr__(self):
        """Return a string representation of the ParametricLi08 object."""
        return f"ParametricLi08(model={self.model})"

    @accepts(lam=angstrom)
    def get_tau(self, lam):
        """Calculate V-band normalised optical depth.

        (Uses the Li_08 function defined above.)

        Args:
            lam (float/np.ndarray of float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float: The V-band normalised optical depth.
        """
        return Li08(
            lam=lam,
            UV_slope=self.UV_slope,
            OPT_NIR_slope=self.OPT_NIR_slope,
            FUV_slope=self.FUV_slope,
            bump=self.bump,
            model=self.model,
        )

    @accepts(lam=angstrom)
    def get_tau_at_lam(self, lam):
        """Calculate optical depth at a wavelength.

        (Uses the Li_08 function defined above.)

        Args:
            lam (float/array-like, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).

        Returns:
            float/array-like, float
                The optical depth.
        """
        raise exceptions.UnimplementedFunctionality(
            "ParametricLi08 form is fit to the normalised Alam/Av values"
            "for the different models, so does not make sense to have this"
            "function. Use other attenuation curve models to get tau_v or Av"
        )


class DraineLiGrainCurves(AttenuationLaw):
    """Draine and Li extinction curves.

    Draine and Li extinction curves obtained from pre-processing
    the extinction efficiencies for the required grain size
    distribution. This is done in grid-generation repo under
    'grid-generation/src/synthesizer_grids/dust/
    create_dustextcurve_draine_li.py' for the required dust
    parameters. Currently only implemented for 2 grain sizes
    of graphites and silicates, and 1 size of PAHs.

    Attributes:
        grid_name (string):
            Name of the extinction curve grid (without hdf5
            extension)
        grid_dir (string):
            Location of the grid
        grain_dict (Dict):
            Dictionary containing the grain type ('graphite' or
            'silicate' or 'pahionised' or 'pahneutral') and their
            corresponding centre of the grain size distribution in
            microns.
            E.g. grain_dict = {'graphite': [0.01, 0.1]}
    """

    def __init__(
        self,
        grid_name: str,
        grid_dir: str,
        grain_dict: Dict = None,
        lam: unyt_array = None,
    ):
        """Initialise the Draine and Li extinction curves.

        Draine and Li extinction curves obtained from pre-processing the
        extinction efficiencies for the required grain size distribution. This
        is done in grid-generation repo under
        'grid-generation/src/synthesizer_grids/dust/
        create_dustextcurve_draine_li.py' for the required dust parameters.
        Currently only implemented for 2 grain sizes of graphites and
        silicates, and 1 size of PAHs.

        Attributes:
            grid_name (string):
                Name of the extinction curve grid (without hdf5
                extension)
            grid_dir (string):
                Location of the grid
            grain_dict (Dict):
                Dictionary containing the grain type ('graphite' or
                'silicate' or 'pahionised' or 'pahneutral') and their
                corresponding centre of the grain size distribution in
                microns.
                E.g. grain_dict = {'graphite': [0.01, 0.1]}
            lam (unyt_array, optional):
                The target wavelength array for the extinction curves.
                If provided, the grid will be resampled to these wavelengths
                once at init time for better performance.

        """
        # Attach information about the grid file
        self.grid_name = grid_name
        self.grid_dir = grid_dir

        # Attach the grain dict
        self.grain_dict = grain_dict

        # Store the target wavelengths if provided
        if lam is not None:
            self.lam = lam.to("Angstrom")

        # We always need to be passed a grain dict, we only have it as a
        # keyword argument so we can raise a clear error message if it is
        # not provided
        if self.grain_dict is None:
            raise exceptions.MissingArgument(
                "Provide `grain_dict` when initialising this dust law. "
                "For example, use grain_dict = {'graphite': [0.01, 0.1]}. "
                "The grain definition should correspond to the grid you are "
                "providing."
            )

        # Define the description for the attenuation law.
        description = (
            "DraineLiGrainCurves: Draine and Li dust grain model for "
            "extinction curves obtained from pre-processing the extinction "
            "efficiencies for the required grain size distribution. The "
            "different components and their relationship with the "
            "dust-to-gas ratio are extracted from a grid."
        )

        # Define the required parameters based on the grain types and sizes
        required_params = [
            f"sigmalos_{grain_type}_a{grain_size}um".replace(".", "p")
            for grain_type, grain_sizes in self.grain_dict.items()
            for grain_size in grain_sizes
        ]
        required_params.append("sigmalos_H")

        # Set up the parent class
        AttenuationLaw.__init__(
            self,
            description=description,
            required_params=required_params,
            require_tau_v=False,
        )

        # Create and attach the grid object containing the attenuation curves
        # for the different grain components
        self._base_grid = Grid(
            self.grid_name,
            self.grid_dir,
            ignore_lines=True,
        )
        # Use the base grid for initial validation
        self.grid = self._base_grid
        self._validate_grid()

        # Cache resampled grids keyed by wavelength tuple for quick lookup.
        self._grid_cache = {}

        # If lam was provided at init, pre-compute the resampled grid for that
        # wavelength.
        if hasattr(self, "lam"):
            self._get_resampled_grid(self.lam)

    def __repr__(self):
        """Return a string representation of the DraineLiGrainCurves object."""
        return f"DraineLiGrainCurves(grid_name={self.grid_name})"

    def _validate_grid(self):
        """Validate that the attenuation grid matches class expectations."""
        tic("DraineLiGrainCurves._validate_grid")

        try:
            # Ensure the supplied attenuation grid actually contains spectra.
            if not self.grid.available_spectra_emissions:
                raise exceptions.InconsistentArguments(
                    "DraineLiGrainCurves requires an attenuation grid with "
                    "spectra."
                )

            # Ensure the extraction machinery only has to deal with a single
            # dust to gas ratio axis.
            if len(self.grid._extract_axes) != 1:
                raise exceptions.UnimplementedFunctionality(
                    "DraineLiGrainCurves only supports attenuation grids with "
                    "a single dtg axis."
                )

            # Record whether the grid uses a linear or logarithmic dtg axis.
            self._dtg_axis_name = self.grid._extract_axes[0]
            if self._dtg_axis_name not in ("dtg", "log10dtg"):
                raise exceptions.UnimplementedFunctionality(
                    "DraineLiGrainCurves only supports attenuation grids with "
                    "a dtg or log10dtg extraction axis."
                )

            # Cache the set of available spectra and make sure every grain
            # component implied by grain_dict exists on the grid.
            self._available_grain_spectra = set(
                self.grid.available_spectra_emissions
            )
            expected_spectra = {
                param.split("sigmalos_", 1)[-1].replace("0p", "0.")
                for param in self._required_params
                if param != "sigmalos_H"
            }
            missing_spectra = expected_spectra - self._available_grain_spectra
            if missing_spectra:
                raise exceptions.InconsistentArguments(
                    "The provided attenuation grid is missing the following "
                    f"grain components: {sorted(missing_spectra)}"
                )

            # Ensure the dtg axis values are numerically well behaved before we
            # use them for particle validation and interpolation.
            grid_dtg = self.grid._extract_axes_values[self._dtg_axis_name]
            if not np.all(np.isfinite(grid_dtg)):
                raise exceptions.InconsistentArguments(
                    "The attenuation grid dtg axis must contain only finite "
                    "values."
                )
            if np.any(np.diff(grid_dtg) <= 0.0):
                raise exceptions.InconsistentArguments(
                    "The attenuation grid dtg axis must be strictly "
                    "increasing."
                )

            # Cache the dtg range once so get_tau_at_lam only needs the
            # per-call particle validation and extraction logic.
            self._grid_dtg_min = np.min(grid_dtg)
            self._grid_dtg_max = np.max(grid_dtg)
        finally:
            toc("DraineLiGrainCurves._validate_grid")

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
        # Create a hashable key from the wavelength array.
        lam_arr = np.atleast_1d(lam.to("Angstrom").ndview)
        cache_key = tuple(lam_arr)

        # Check if we already have this configuration cached.
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]

        # Resample the base grid to the requested wavelengths
        if lam_arr.size == 1:
            # For scalar wavelengths, make a copy to avoid mutating the base
            # grid
            grid = copy.deepcopy(self._base_grid)
            # Pass the original lam with units to get_spectra_at_lam
            spectra_at_lam = grid.get_spectra_at_lam(lam)
            for spectra_id, spectra in spectra_at_lam.items():
                grid.spectra[spectra_id] = spectra[..., np.newaxis]
            grid.lam = lam_arr
            grid._ensure_spectra_data_contiguous()
        else:
            # For arrays of wavelengths
            grid = self._base_grid.reduce_rest_frame_lam(lam)

        # Cache the resampled grid.
        self._grid_cache[cache_key] = grid

        return grid

    def _get_dtgs(self, sigmalos_H, sigmalos_dust):
        """Validate inputs and derive broadcast DTG quantities.

        Args:
            sigmalos_H (unyt_array):
                The line-of-sight hydrogen column density.
            sigmalos_dust (dict):
                The dust component column densities keyed by component name.

        Raises:
            exceptions.InconsistentArguments:
                Raised if the column densities do not have broadcast-compatible
                shapes, do not carry compatible units, or contain negative
                values.

        Returns:
            tuple:
                The common particle count, hydrogen column data, whether the
                hydrogen column is scalar, dust column data keyed by component,
                the shared column-density units, and the hydrogen validity
                information.
        """
        tic("DraineLiGrainCurves._get_dtgs")

        try:
            sigmalos_H_arr = sigmalos_H.ndview
            column_units = sigmalos_H.units
            hydrogen_is_scalar = np.ndim(sigmalos_H_arr) == 0
            if hydrogen_is_scalar:
                sigmalos_H_arr = sigmalos_H_arr.item()
                nparticles = 1
            else:
                nparticles = sigmalos_H_arr.size
            dust_arrays = {}

            for component_key, dust_col in sigmalos_dust.items():
                if not isinstance(dust_col, (unyt_quantity, unyt_array)):
                    raise exceptions.InconsistentArguments(
                        f"Provide units to the {component_key} quantity"
                    )

                try:
                    if dust_col.units == column_units:
                        dust_arr = dust_col.ndview
                    else:
                        dust_arr = dust_col.to(column_units).ndview
                except Exception as e:
                    raise exceptions.InconsistentArguments(
                        f"{component_key} must have units compatible with "
                        f"{column_units}"
                    ) from e

                dust_is_scalar = np.ndim(dust_arr) == 0
                if dust_is_scalar:
                    dust_arr = dust_arr.item()
                    dust_size = 1
                else:
                    dust_size = dust_arr.size

                dust_arrays[component_key] = (dust_arr, dust_is_scalar)
                if dust_size > nparticles:
                    nparticles = dust_size

            if hydrogen_is_scalar:
                valid_hydrogen = np.isfinite(sigmalos_H_arr) and (
                    sigmalos_H_arr > 0.0
                )
            else:
                if sigmalos_H_arr.size not in (1, nparticles):
                    raise exceptions.InconsistentArguments(
                        "sigmalos_H has length "
                        f"{sigmalos_H_arr.size}, but expected 1 or "
                        f"{nparticles}."
                    )
                valid_hydrogen = np.isfinite(sigmalos_H_arr) & (
                    sigmalos_H_arr > 0.0
                )

            for component_key, (
                dust_arr,
                dust_is_scalar,
            ) in dust_arrays.items():
                dust_size = 1 if dust_is_scalar else dust_arr.size
                if dust_size not in (1, nparticles):
                    raise exceptions.InconsistentArguments(
                        f"{component_key} has length {dust_size}, but "
                        f"expected 1 or {nparticles}."
                    )

            return (
                nparticles,
                sigmalos_H_arr,
                hydrogen_is_scalar,
                dust_arrays,
                column_units,
                valid_hydrogen,
            )
        finally:
            toc("DraineLiGrainCurves._get_dtgs")

    @accepts(lam=angstrom, sigmalos_H=Msun / pc**2)
    def get_tau_at_lam(
        self,
        lam: unyt_array,
        sigmalos_H: unyt_array,
        **sigmalos_dust: unyt_array,
    ):
        """Calculate optical depth at a wavelength.

        Args:
            lam (unyt_array, float):
                An array of wavelengths or a single wavelength at which to
                calculate optical depths (in AA, global unit).
            sigmalos_H (unyt array):
                Line-of-sight H density in units of Msun/pc^2
            sigmalos_dust (Dict: unyt_array):
                Dictionary containing the different
                line-of-sight dust density of the dust
                components. This should follow the format
                'sigmalos_{grain_type}_a{grain_bin_centre}um':
                unyt_array (units of Msun/pc^2). The function will
                unpack the contents.

        Returns:
            optical depth (unyt_array, float)
                The optical depth at the given wavlength ((N_dtg, N_lambda)
                if sigmalos input is array-like, otherwise shape (N_lambda,)).
                Dimensionless
        """
        tic("DraineLiGrainCurves.get_tau_at_lam")

        try:
            # Map the public keyword arguments onto the spectra names stored in
            # the attenuation grid.
            # TODO: modify the grid generation files to remove this remapping.
            component_datasets = {
                component_key: component_key.split("sigmalos_", 1)[-1].replace(
                    "0p", "0."
                )
                for component_key in sigmalos_dust
            }

            # Validate inputs and derive the broadcast DTG inputs before any
            # wavelength resampling or particle extraction is attempted.
            (
                nparticles,
                sigmalos_H,
                hydrogen_is_scalar,
                sigmalos_dust,
                column_units,
                valid_hydrogen,
            ) = self._get_dtgs(sigmalos_H, sigmalos_dust)

            # Get or create the resampled grid for this wavelength.
            grid = self._get_resampled_grid(lam)

            # Reuse the validated grid axis metadata cached during
            # construction.
            dtg_axis_name = self._dtg_axis_name

            # Verify all requested grain types are available
            for dataset_key in component_datasets.values():
                if dataset_key not in self._available_grain_spectra:
                    raise exceptions.InconsistentArguments(
                        "Grain type "
                        f"{dataset_key} not in the provided dust grid!"
                    )

            # Accumulate the contribution from each grain component into the
            # final optical-depth array.
            tau_all = np.zeros((nparticles, grid.nlam), dtype=np.float32)
            tau_scale = ((1.0 * cm**2) / _GAS_MASS_PER_H).to(
                1 / column_units
            ).value / 1.086
            grid_dtg_axis = grid._extract_axes_values[dtg_axis_name]
            grid_shape = np.array(grid.shape, dtype=np.int32)
            grid_weights = np.ones(nparticles)
            dtg = np.ones(nparticles, dtype=float)
            valid = np.empty(nparticles, dtype=bool)
            dust_scale = np.zeros(nparticles, dtype=float)

            # Loop over the dust components
            for component_key, (
                dust_col,
                dust_is_scalar,
            ) in sigmalos_dust.items():
                # Map the public keyword argument onto the spectra name stored
                # in the attenuation grid
                dataset_key = component_datasets[component_key]

                if dust_is_scalar:
                    if not (np.isfinite(dust_col) and dust_col > 0.0):
                        continue

                    if hydrogen_is_scalar:
                        if not valid_hydrogen:
                            continue
                        valid.fill(True)
                        dtg.fill(
                            dust_col
                            / (sigmalos_H * _DRAINE_LI_MEAN_MOLECULAR_WEIGHT)
                        )
                    else:
                        valid[:] = valid_hydrogen
                        if not np.any(valid):
                            continue
                        dtg.fill(1.0)
                        dtg[valid] = dust_col / (
                            sigmalos_H[valid]
                            * _DRAINE_LI_MEAN_MOLECULAR_WEIGHT
                        )
                else:
                    dust_finite = np.isfinite(dust_col)

                    if hydrogen_is_scalar:
                        if not valid_hydrogen:
                            continue
                        valid[:] = dust_finite & (dust_col > 0.0)
                        if not np.any(valid):
                            continue
                        dtg.fill(1.0)
                        dtg[valid] = dust_col[valid] / (
                            sigmalos_H * _DRAINE_LI_MEAN_MOLECULAR_WEIGHT
                        )
                    else:
                        valid[:] = (
                            valid_hydrogen & dust_finite & (dust_col > 0.0)
                        )
                        if not np.any(valid):
                            continue
                        dtg.fill(1.0)
                        dtg[valid] = dust_col[valid] / (
                            sigmalos_H[valid]
                            * _DRAINE_LI_MEAN_MOLECULAR_WEIGHT
                        )

                # Log the grid axis if we need to
                dtg_grid_values = (
                    np.log10(dtg) if dtg_axis_name == "log10dtg" else dtg
                )

                # Call the particle spectra extension directly for this single
                # extraction axis instead of going through generate_lnu.
                component_curves, _ = compute_particle_seds(
                    grid.spectra[dataset_key],
                    (grid_dtg_axis,),
                    (dtg_grid_values,),
                    grid_weights,
                    grid_shape,
                    1,
                    nparticles,
                    grid.nlam,
                    "cic",
                    1,
                    valid,
                    None,
                    (dtg_axis_name,),
                )

                # Convert from mag cm^2 / H nucleus into optical depth per dust
                # surface density, then multiply by the dust column itself.
                component_tau = component_curves * tau_scale

                if dust_is_scalar:
                    tau_all += component_tau * dust_col
                else:
                    dust_scale.fill(0.0)
                    dust_scale[valid] = dust_col[valid]
                    tau_all += component_tau * dust_scale[:, np.newaxis]

            # Preserve the previous scalar-like return shape for
            # single-particle inputs.
            if tau_all.shape[0] == 1:
                return tau_all[0]
            return tau_all
        finally:
            toc("DraineLiGrainCurves.get_tau_at_lam")

    @accepts(lam=angstrom, sigmalos_H=Msun / pc**2)
    def get_tau(
        self,
        lam: unyt_array,
        sigmalos_H: unyt_array = None,
        **sigmalos_dust: unyt_array,
    ):
        """Calculate optical depth normalised at V-band at a wavelength.

        Args:
            lam (float/array-like, float):
                An array of wavelengths or a single wavlength at which to
                calculate optical depths (in AA, global unit).
            sigmalos_H (unyt array):
                Line-of-sight H density in units of
                Msun/pc^2
            sigmalos_dust (Dict: unyt_array):
                Dictionary containing the different
                line-of-sight dust density of the dust
                components. This should follow the format
                'sigmalos_{grain_type}_a{grain_bin_centre}um':
                unyt_array (units of Msun/pc^2). The function will
                unpack the contents.

        Returns:
            optical depth/optical depth at V-band (unyt_array, float)
                The optical depth at the given wavlength ((N_dtg, N_lambda)
                if sigmalos input is array-like, otherwise shape (N_lambda,)).
                Dimensionless
        """
        if sigmalos_H is None and len(sigmalos_dust) == 0:
            sigmalos_H = getattr(self, "sigmalos_H", None)
            # Collect any attributes starting with 'sigmalos_'
            sigmalos_dust = {
                key: getattr(self, key)
                for key in vars(self)
                if key.startswith("sigmalos_") and key != "sigmalos_H"
            }
        tau_lam = self.get_tau_at_lam(lam, sigmalos_H, **sigmalos_dust)
        tau_V = self.get_tau_at_lam(
            5500 * angstrom, sigmalos_H, **sigmalos_dust
        )
        # tau_lam: (N, M) or (M,), tau_V: (N,) or scalar (M==1)
        if np.ndim(tau_V) <= 1:
            return tau_lam / tau_V
        return tau_lam / tau_V[:, np.newaxis]

    @accepts(lam=angstrom)
    def get_transmission(self, tau_v=None, lam=None, **dust_curve_kwargs):
        """Compute the transmission curve.

         Returns the transmitted flux/luminosity fraction based on an optical
        depth at a range of wavelengths.

        Args:
            tau_v (None):
                Optical depth at V-band, not required here.
            lam (np.ndarray of float):
                The wavelengths (with units) at which to calculate
                transmission.
            **dust_curve_kwargs (dict):
                Additional keyword arguments to be passed to the dust curve
                which have been defined on the emitter or model.

        Returns:
            np.ndarray of float:
                The transmission at each wavelength
        """
        if tau_v is not None:
            warn(
                "tau_v has been provided, but `DraineLiGrainCurves` does not "
                "use tau_v. Ignoring tau_v in the calculation."
            )

        # Set any additional parameters on the dust curve
        self._set_params(**dust_curve_kwargs)

        try:
            sigmalos_H = dust_curve_kwargs.get(
                "sigmalos_H", getattr(self, "sigmalos_H", None)
            )
            # Gather sigmalos_* dust components: start with attributes,
            # then override with kwargs
            sigmalos_dust = {}
            for key in vars(self):
                if key.startswith("sigmalos_") and key != "sigmalos_H":
                    sigmalos_dust[key] = getattr(self, key)
            for key, value in dust_curve_kwargs.items():
                if key.startswith("sigmalos_") and key != "sigmalos_H":
                    sigmalos_dust[key] = value
            # Compute tau_lam directly (tau_v is not used for this model)
            tau_lam = self.get_tau_at_lam(lam, sigmalos_H, **sigmalos_dust)
        finally:
            # Always restore previous state
            self._reset_params()

        return np.exp(-tau_lam)
