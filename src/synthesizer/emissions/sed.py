"""Functionality related to spectra storage and manipulation.

When a spectra is computed from a `Galaxy` or a Galaxy component the resulting
calculated spectra are stored in `Sed` objects. These provide helper functions
for quick manipulation of the spectra. Seds can contain a single spectra or
arbitrarily many, with all methods capable of acting on both consistently.

Example usage:

    sed = Sed(lams, lnu)
    sed.get_fnu(redshift)
    sed.apply_attenutation(tau_v=0.7)
    sed.get_photo_fnu(filters, nthreads=4)
"""

import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.interpolate import interp1d
from scipy.stats import linregress
from spectres import spectres
from unyt import (
    Hz,
    angstrom,
    c,
    cm,
    erg,
    eV,
    h,
    pc,
    s,
    unyt_array,
    unyt_quantity,
)

from synthesizer import exceptions
from synthesizer.conversions import lnu_to_llam
from synthesizer.extensions.timers import tic, toc
from synthesizer.photometry import PhotometryCollection
from synthesizer.synth_warnings import deprecated, warn
from synthesizer.units import Quantity, accepts
from synthesizer.utils import TableFormatter, rebin_1d, wavelength_to_rgba
from synthesizer.utils.integrate import integrate_last_axis


class Sed:
    """A class representing a spectral energy distribution (SED).

    Attributes:
        lam (Quantity, np.ndarray of float)
            The rest frame wavelength array.
        nu (Quantity, np.ndarray of float)
            The rest frame frequency array.
        lnu (Quantity, np.ndarray of float)
            The spectral luminosity density.
        bolometric_luminosity (Quantity, float)
            The bolometric luminosity.
        fnu (Quantity, np.ndarray of float)
            The spectral flux density.
        obslam (Quantity, np.ndarray of float)
            The observed wavelength array.
        obsnu (Quantity, np.ndarray of float)
            The observed frequency array.
        description (str):
            An optional descriptive string defining the Sed.
        redshift (float):
            The redshift of the Sed.
        photo_lnu (dict, float)
            The rest frame broadband photometry in arbitrary filters
            (filter_code: photometry).
        photo_fnu (dict, float)
            The observed broadband photometry in arbitrary filters
            (filter_code: photometry).
    """

    # Define Quantities, for details see units.py
    lam = Quantity("wavelength")
    nu = Quantity("frequency")
    lnu = Quantity("luminosity_density_frequency")
    fnu = Quantity("flux_density_frequency")
    obsnu = Quantity("frequency")
    obslam = Quantity("wavelength")

    @accepts(lam=angstrom, lnu=erg / s / Hz)
    def __init__(self, lam, lnu=None, description=None):
        """Initialise a new spectral energy distribution object.

        Args:
            lam (np.ndarray of float):
                The rest frame wavelength array. Default units are defined
                in `synthesizer.units`. If unmodified these will be Angstroms.
            lnu (np.ndarray of float):
                The spectral luminosity density. Default units are defined in
                `synthesizer.units`. If unmodified these will be erg/s/Hz
            description (str):
                An optional descriptive string defining the Sed.
        """
        start = tic()

        # Set the description
        self.description = description

        # Set the wavelength
        self.lam = lam

        # Calculate frequency
        self.nu = c / self.lam

        # If no lnu is provided create an empty array with the same shape as
        # lam.
        if lnu is None:
            self.lnu = np.zeros(self.lam.shape)
        else:
            self.lnu = lnu

        # Redshift of the SED
        self.redshift = 0

        # The wavelengths and frequencies in the observer frame
        self.obslam = None
        self.obsnu = None
        self.fnu = None

        # Broadband photometry
        self.photo_lnu = None
        self.photo_fnu = None

        toc("Creating Sed", start)

    def sum(self):
        """Sum the SED over all dimensions.

        For multidimensional `sed`'s, sum the luminosity to provide a 1D
        integrated SED.

        Returns:
            sed (object, Sed):
                Summed 1D SED.
        """
        start = tic()

        # Check that the lnu array is multidimensional
        if len(self._lnu.shape) > 1:
            # Define the axes to sum over to give only the final axis
            sum_over = tuple(range(0, len(self._lnu.shape) - 1))

            # Create a new sed object with the first Lnu dimension collapsed
            new_sed = Sed(
                self.lam, np.nansum(self._lnu, axis=sum_over) * self.lnu.units
            )

            # If fnu exists, sum that too
            if self.fnu is not None:
                new_sed.fnu = (
                    np.nansum(self._fnu, axis=sum_over) * self.fnu.units
                )
                new_sed.obsnu = self.obsnu
                new_sed.obslam = self.obslam
                new_sed.redshift = self.redshift

            toc("Summing Sed", start)

            return new_sed
        else:
            # If 1D, just return the original array
            return self

    def concat(self, *other_seds):
        """Concatenate the spectra arrays of multiple Sed objects.

        This will combine the arrays along the first axis. For example
        concatenating two Seds with Sed.lnu.shape = (10, 1000) and
        Sed.lnu.shape = (20, 1000) will result in a new Sed with
        Sed.lnu.shape = (30, 1000). The wavelength array of
        the resulting Sed will be the array on self.

        Incompatible spectra shapes will raise an error.

        Args:
            other_seds (object, Sed):
                Any number of Sed objects to concatenate with self. These must
                have the same wavelength array.

        Returns:
            Sed
                A new instance of Sed with the concatenated lnu arrays.

        Raises:
            InconsistentAddition
                If wavelength arrays are incompatible an error is raised.
        """
        # Define the new lnu to accumalate in
        new_lnu = self._lnu

        # Loop over the other seds
        for other_sed in other_seds:
            # Ensure the wavelength arrays are compatible
            # NOTE: this is probably overkill and too costly. We
            # could instead check the first and last entry and the shape.
            # In rare instances this could fail though.
            if not np.array_equal(self._lam, other_sed._lam):
                raise exceptions.InconsistentAddition(
                    "Wavelength grids must be identical"
                )

            # Get the other lnu array
            other_lnu = other_sed._lnu

            # If the the number of dimensions differ between the lnu arrays we
            # need to promote the smaller one
            if new_lnu.ndim < other_lnu.ndim:
                new_lnu = np.array((new_lnu,))
            elif new_lnu.ndim > other_lnu.ndim:
                other_lnu = np.array((other_lnu,))
            elif new_lnu.ndim == other_lnu.ndim == 1:
                new_lnu = np.array((new_lnu,))
                other_lnu = np.array((other_lnu,))

            # Concatenate this lnu array
            new_lnu = np.concatenate((new_lnu, other_lnu))

        return Sed(self.lam, new_lnu * self.lnu.units)

    def __add__(self, second_sed):
        """Add two Sed objects together.

        Overloads the addition operator to allow two Sed objects to be added
        together.

        Args:
            second_sed (object, Sed):
                The Sed object to combine with self.

        Returns:
            Sed
                A new instance of Sed with added lnu and fnu arrays.

        Raises:
            InconsistentAddition
                If wavelength arrays or lnu arrays are incompatible an error
                is raised.
        """
        # Ensure the wavelength arrays are compatible
        if not (
            self._lam[0] == second_sed._lam[0]
            and self._lam[-1] == second_sed._lam[-1]
        ):
            raise exceptions.InconsistentAddition(
                "Wavelength grids must be identical "
                f"({self.lam.min()} -> {self.lam.max()} "
                f"with shape {self._lam.shape} != "
                f"{second_sed.lam.min()} -> {second_sed.lam.max()} "
                f"with shape {second_sed._lam.shape})"
            )

        # Ensure the lnu arrays are compatible
        # This check is redudant for Sed.lnu.shape = (nlam, ) spectra but will
        # not erroneously error. Nor is it expensive.
        if self._lnu.shape[0] != second_sed._lnu.shape[0]:
            raise exceptions.InconsistentAddition(
                "SEDs must have same dimensions "
                f"({self._lnu.shape} != {second_sed._lnu.shape})"
            )

        # They're compatible, add them and make a new Sed
        new_sed = Sed(self.lam, lnu=self.lnu + second_sed.lnu)

        # If fnu exists on both then we need to add those too
        if (self.fnu is not None) and (second_sed.fnu is not None):
            new_sed.fnu = self.fnu + second_sed.fnu
            new_sed.obsnu = self.obsnu
            new_sed.obslam = self.obslam
            new_sed.redshift = self.redshift

        return new_sed

    def __radd__(self, second_sed):
        """Add two Sed objects together (reverse order).

        Overloads "reflected" addition to allow sed objects to be added
        together when in reverse order, i.e. second_sed + self.

        This may seem superfluous, but it is needed to enable the use of sum()
        on lists of Seds.

        Args:
            second_sed (object, Sed):
                The Sed object to combine with self.

        Returns:
            Sed:
                A new instance of Sed with added lnu arrays.

        Raises:
            InconsistentAddition
                If wavelength arrays or lnu arrays are incompatible an error
                is raised.
        """
        # Handle the int case explictly which is triggered by the use of sum
        if isinstance(second_sed, int) and second_sed == 0:
            return self
        return self.__add__(second_sed)

    def scale(self, scaling, inplace=False, mask=None, lam_mask=None):
        """Scale the lnu of the Sed object.

        Note: only acts on the rest frame spectra. To get the
        scaled fnu get_fnu must be called on the newly scaled
        Sed object.

        Args:
            scaling (float):
                The scaling to apply to lnu.
            inplace (bool):
                If True, the Sed object is modified in place. If False, a new
                Sed object is returned with the scaled lnu.
            mask (array-like, bool):
                A mask for the lnu array to apply the scaling to. This must
                be the same shape as the lnu array excluding the wavelength
                axis.
            lam_mask (array-like, bool):
                A mask for the wavelength array to apply the scaling to.

        Returns:
            Sed
                A new instance of Sed with scaled lnu.
        """
        # If we have units make sure they are ok and then strip them
        if isinstance(scaling, (unyt_array, unyt_quantity)):
            if self.lnu.units.dimensions != scaling.units.dimensions:
                raise exceptions.InconsistentMultiplication(
                    f"Incompatible units {self.lnu.units} and {scaling.units}"
                )
            else:
                scaling = scaling.to(self.lnu.units)
                scaling = scaling.value

        # Unpack the array's we'll need during scaling
        lnu = self._lnu.copy()
        units = self.lnu.units

        # If we have a wavelength mask apply it now
        if lam_mask is not None:
            lnu = lnu[..., lam_mask]

        # Handle a scalar scaling factor
        if np.isscalar(scaling):
            if mask is not None:
                lnu[mask] *= scaling
            else:
                lnu *= scaling

        # Handle a multi-element array scaling factor as long as it matches
        # the shape of the lnu array up to the dimensions of the scaling array
        elif isinstance(scaling, np.ndarray) and len(scaling.shape) < len(
            self.shape
        ):
            # We need to expand the scaling array to match the lnu array
            expand_axes = tuple(range(len(scaling.shape), len(self.shape)))
            new_scaling = np.ones(self.shape) * np.expand_dims(
                scaling, axis=expand_axes
            )

            # Apply the scaling
            if mask is not None:
                lnu[mask] *= new_scaling[mask]
            else:
                lnu *= new_scaling

        # If the scaling array is the same shape as the lnu array then we can
        # just multiply them together
        elif isinstance(scaling, np.ndarray) and scaling.shape == self.shape:
            if mask is not None:
                lnu[mask] *= scaling[mask]
            else:
                lnu *= scaling

        # Ok, if we have an array but the shapes don't match then we need to
        # create a new array where each element of the scaling array is
        # multipled by the whole lnu array producing a new lnu array of
        # shape (scaling.shape + lnu.shape)
        elif isinstance(scaling, np.ndarray):
            lnu = scaling[..., np.newaxis] * lnu

            # Masks are not supported in this case
            if mask is not None:
                raise exceptions.InconsistentMultiplication(
                    "Masking is not supported for scaling arrays"
                    " with different shapes"
                )

        # Otherwise, we've been handed a bad scaling factor
        else:
            out_str = f"Incompatible scaling factor with type {type(scaling)} "
            if hasattr(scaling, "shape"):
                out_str += f"and shape {scaling.shape} (expected {self.shape})"
            else:
                out_str += f"and value {scaling}"
            raise exceptions.InconsistentMultiplication(out_str)

        # Now complete the calculation if we need to
        if lam_mask is not None:
            new_lnu = self.lnu.copy()
            new_lnu[..., lam_mask] = lnu
        else:
            new_lnu = lnu * units

        # Return a new Sed object if we aren't scaling inplace
        if not inplace:
            return Sed(self.lam, lnu=new_lnu)

        self._lnu = new_lnu
        return self

    def __mul__(self, scaling):
        """Scale the lnu of the Sed object.

        Overide multiplication operator to allow lnu to be scaled.
        This only works scaling * x.

        Note: only acts on the rest frame spectra. To get the
        scaled fnu get_fnu must be called on the newly scaled
        Sed object.

        Args:
            scaling (float):
                The scaling to apply to lnu.

        Returns:
            Sed
                A new instance of Sed with scaled lnu.
        """
        return self.scale(scaling)

    def __rmul__(self, scaling):
        """Scale the lnu of the Sed object.

        As above but for x * scaling.

        Note: only acts on the rest frame spectra. To get the
        scaled fnu get_fnu must be called on the newly
        scaled Sed object.

        Args:
            scaling (float):
                The scaling to apply to lnu.

        Returns:
            Sed
                A new instance of Sed with scaled lnu.
        """
        return self.scale(scaling)

    def __str__(self):
        """Return a string representation of the SED object.

        Returns:
            table (str):
                A string representation of the SED object.
        """
        # Intialise the table formatter
        formatter = TableFormatter(self)

        return formatter.get_table("SED")

    @property
    def luminosity(self):
        """Get the spectra in terms of luminosity.

        Returns:
            luminosity (unyt_array):
                The luminosity array.
        """
        return self.lnu * self.nu

    @property
    def flux(self):
        """Get the spectra in terms fo flux.

        Returns:
            flux (unyt_array):
                The flux array.
        """
        return self.fnu * self.obsnu

    @property
    def llam(self):
        """Get the spectral luminosity density per Angstrom.

        Returns:
            luminosity (unyt_array):
                The spectral luminosity density per Angstrom array.
        """
        return self.nu * self.lnu / self.lam

    @property
    def luminosity_nu(self):
        """Alias to lnu.

        Returns:
            luminosity (unyt_array):
                The spectral luminosity density per Hz array.
        """
        return self.lnu

    @property
    def luminosity_lambda(self):
        """Alias to llam.

        Returns:
            luminosity (unyt_array):
                The spectral luminosity density per Angstrom array.
        """
        return self.llam

    @property
    def wavelength(self):
        """Alias to lam (wavelength array).

        Returns:
            wavelength (unyt_array):
                The wavelength array.
        """
        return self.lam

    @property
    def frequency(self):
        """Alias to nu (frequency array).

        Returns:
            frequency (unyt_array):
                The frequency array.
        """
        return self.nu

    @property
    def energy(self):
        """Get the wavelengths in terms of photon energies in eV.

        Returns:
            energy (unyt_array):
                The energy coordinate.
        """
        return (h * c / self.lam).to(eV)

    @property
    def ndim(self):
        """Get the dimensions of the spectra array.

        Returns:
            Tuple
                The shape of self.lnu
        """
        return np.ndim(self.lnu)

    @property
    def shape(self):
        """Get the shape of the spectra array.

        Returns:
            Tuple
                The shape of self.lnu
        """
        return self.lnu.shape

    @property
    def nlam(self):
        """Get the number of wavelength elements.

        Returns:
            int: The number of wavelength elements.
        """
        return self.lam.size

    @property
    def bolometric_luminosity(self):
        """Return the bolometric luminosity of the SED with units.

        This will integrate the SED using the trapezium method over the
        final axis (which is always the wavelength axis) for an arbitrary
        number of dimensions.

        Returns:
            bolometric_luminosity (unyt_array):
                The bolometric luminosity.
        """
        # Calculate the bolometric luminosity using the trapezium rule.
        # NOTE: the integration is done "backwards" when integrating over
        # frequency. It's faster to just multiply by -1 than to reverse the
        # array.
        integral = -integrate_last_axis(
            self._nu,
            self._lnu,
            method="trapz",
        )

        # Return the bolometric luminosity with units
        return integral * self.lnu.units * self.nu.units

    @property
    def _bolometric_luminosity(self):
        """Return the bolometric luminosity of the SED without units.

        This will integrate the SED using the trapezium method over the
        final axis (which is always the wavelength axis) for an arbitrary
        number of dimensions.

        Returns:
            bolometric_luminosity (float):
                The bolometric luminosity.
        """
        return self.bolometric_luminosity.value

    @accepts(nu=Hz)
    def get_lnu_at_nu(self, nu, kind=False):
        """Return lnu at a provided frequency using 1d interpolation.

        Args:
            nu (float/np.ndarray of float):
                The frequency(s) of interest.
            kind (str):
                Interpolation kind, see scipy.interp1d docs for more
                information. Possible values are 'linear', 'nearest',
                'zero', 'slinear', 'quadratic', 'cubic', 'previous', and
                'next'.

        Returns:
            unyt_array:
                The luminosity (lnu) at the provided wavelength.
        """
        return interp1d(self._nu, self._lnu, kind=kind)(nu) * self.lnu.units

    @accepts(lam=angstrom)
    def get_lnu_at_lam(self, lam, kind=False):
        """Return lnu at a provided wavelength.

        Args:
            lam (float/np.ndarray of float):
                The wavelength(s) of interest.
            kind (str):
                Interpolation kind, see scipy.interp1d docs for more
                information. Possible values are 'linear', 'nearest',
                'zero', 'slinear', 'quadratic', 'cubic', 'previous', and
                'next'.

        Returns:
            luminosity (unyt-array):
                The luminosity (lnu) at the provided wavelength.
        """
        return interp1d(self._lam, self._lnu, kind=kind)(lam) * self.lnu.units

    @deprecated(
        message=(
            "Deprecated in favour of bolometric_luminosity propery method"
        )
    )
    def measure_bolometric_luminosity(
        self, integration_method="trapz", nthreads=1
    ):
        """Calculate the bolometric luminosity of the SED.

        This will integrate the SED over the final axis (which is always the
        wavelength axis) for an arbitrary number of dimensions.

        Args:
            integration_method (str):
                The integration method used to calculate the bolometric
                luminosity. Options include 'trapz' and 'simps'.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.

        Returns:
            bolometric_luminosity (float):
                The bolometric luminosity.

        Raises:
            InconsistentArguments
                If `integration_method` is an incompatible option an error
                is raised.
        """
        start = tic()

        # Calculate the bolometric luminosity
        # NOTE: the integration is done "backwards" when integrating over
        # frequency. It's faster to just multiply by -1 than to reverse the
        # array.
        integral = -integrate_last_axis(
            self._nu,
            self._lnu,
            nthreads=nthreads,
            method=integration_method,
        )
        toc("Calculating bolometric luminosity", start)

        return integral * self.lnu.units * self.nu.units

    @accepts(window=angstrom)
    def measure_window_luminosity(
        self, window, integration_method="trapz", nthreads=1
    ):
        """Measure the luminosity in a spectral window.

        Args:
            window (tuple, float):
                The window in wavelength.
            integration_method (str):
                The integration method used to calculate the window
                luminosity. Options include 'trapz' and 'simps'.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.

        Returns:
            luminosity (float):
                The luminosity in the window.

        Raises:
            UnrecognisedOption
                If `integration_method` is an incompatible option
                an error is raised.
        """
        # Define the "transmission" for the window
        transmission = (self.lam > window[0]) & (self.lam < window[1])

        # Integrate the window
        # NOTE: the integration is done "backwards" when integrating over
        # frequency. It's faster to just multiply by -1 than to reverse the
        # array.
        luminosity = -(
            integrate_last_axis(
                self._nu,
                self._lnu * transmission,
                nthreads=nthreads,
                method=integration_method,
            )
            * self.lnu.units
            * Hz
        )

        return luminosity

    @accepts(window=angstrom)
    def measure_window_lnu(
        self, window, integration_method="trapz", nthreads=1
    ):
        """Measure lnu in a spectral window.

        Args:
            window (tuple, float):
                The window in wavelength.
            integration_method (str):
                The integration method to use on the window. Options include
                'average', or for integration 'trapz', and 'simps'.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.

        Returns:
            luminosity (float):
                The luminosity in the window.

        Raises:
            UnrecognisedOption
                If `integration_method` is an incompatible option an error
                is raised.
        """
        # Define a pseudo transmission function
        transmission = (self.lam > window[0]) & (self.lam < window[1])
        transmission = transmission.astype(float)

        # Apply the correct method
        if integration_method == "average":
            # Apply to the correct axis of the spectra
            if self.ndim >= 2:
                lnu = (
                    np.array(
                        [
                            np.sum(_lnu * transmission) / np.sum(transmission)
                            for _lnu in self._lnu.reshape(
                                -1, self._lnu.shape[-1]
                            )
                        ]
                    )
                    * self.lnu.units
                )

                lnu = lnu.reshape(self._lnu.shape[:-1])

            else:
                lnu = np.sum(self.lnu * transmission) / np.sum(transmission)

        else:
            # Luminosity integral
            lum = integrate_last_axis(
                self._nu,
                self._lnu * transmission / self.nu,
                nthreads=nthreads,
                method=integration_method,
            )

            # Transmission integral
            tran = integrate_last_axis(
                self._nu,
                transmission / self.nu,
                nthreads=nthreads,
                method=integration_method,
            )

            # Compute lnu
            lnu = lum / tran * self.lnu.units

        return lnu.to(self.lnu.units)

    @accepts(blue=angstrom, red=angstrom)
    def measure_break(self, blue, red, nthreads=1, integration_method="trapz"):
        """Measure a spectral break (e.g. the Balmer break) using two windows.

        Args:
            blue (tuple, float):
                The wavelength limits of the blue window.
            red (tuple, float):
                The wavelength limits of the red window.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.
            integration_method (str):
                The integration method used. Options include 'trapz'
                and 'simps'.

        Returns:
            float:
                The ratio of the luminosity in the two windows.

        Raises:
            UnrecognisedOption
                If `integration_method` is an incompatible option an error
                is raised.
        """
        return (
            self.measure_window_lnu(
                red,
                nthreads=nthreads,
                integration_method=integration_method,
            ).value
            / self.measure_window_lnu(
                blue,
                nthreads=nthreads,
                integration_method=integration_method,
            ).value
        )

    def measure_balmer_break(self, nthreads=1, integration_method="trapz"):
        """Measure the Balmer break.

        This will use two windows at (3400,3600) and (4150,4250).

        Args:
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.
            integration_method (str):
                The integration method used. Options include 'trapz'
                and 'simps'.

        Returns:
            float
                The Balmer break strength

        Raises:
            UnrecognisedOption
                If `integration_method` is an incompatible option an error
                is raised.
        """
        blue = (3400, 3600) * angstrom
        red = (4150, 4250) * angstrom

        return self.measure_break(
            blue, red, nthreads=nthreads, integration_method=integration_method
        )

    def measure_d4000(
        self, definition="Bruzual83", nthreads=1, integration_method="trapz"
    ):
        """Measure the D4000 index.

        This can optionally use either the Bruzual83 or Balogh definitions.

        Args:
            definition (str):
                The choice of definition: 'Bruzual83' or 'Balogh'.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.
            integration_method (str):
                The integration method used. Options include 'trapz'
                and 'simps'.

        Returns:
            float
                The Balmer break strength.

        Raises:
            UnrecognisedOption
                If `definition` or `integration_method` is an
                incompatible option an error is raised.
        """
        # Define the requested definition
        if definition == "Bruzual83":
            blue = (3750, 3950) * angstrom
            red = (4050, 4250) * angstrom

        elif definition == "Balogh":
            blue = (3850, 3950) * angstrom
            red = (4000, 4100) * angstrom
        else:
            raise exceptions.UnrecognisedOption(
                f"Unrecognised definition ({definition}). "
                "Options are 'Bruzual83' or 'Balogh'"
            )

        return self.measure_break(
            blue,
            red,
            nthreads=nthreads,
            integration_method=integration_method,
        )

    @accepts(window=angstrom)
    def measure_beta(
        self,
        window=(1250.0 * angstrom, 3000.0 * angstrom),
        nthreads=1,
        integration_method="trapz",
    ):
        """Measure the UV continuum slope (beta).

        If the provided window is len(2) a full fit to the spectra is performed
        otherwise the luminosity in two windows is calculated and used to
        determine the slope, similar to observations.

        Args:
            window (tuple, float):
                The window in which to measure in terms of wavelength.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.
            integration_method (str):
                The integration method used to calculate the window luminosity.
                Options include 'trapz' and 'simps'.

        Returns:
            float:
                The UV continuum slope (beta)

        Raises:
            UnrecognisedOption
                If `integration_method` is an incompatible option an error
                is raised.
        """
        # If a single window is provided
        if len(window) == 2:
            s = (self.lam > window[0]) & (self.lam < window[1])

            # Handle different spectra dimensions
            if self.ndim >= 2:
                beta = np.array(
                    [
                        linregress(
                            np.log10(self._lam[s]), np.log10(_lnu[..., s])
                        )[0]
                        - 2.0
                        for _lnu in self.lnu.reshape(-1, self.lnu.shape[-1])
                    ]
                )

                beta = beta.reshape(self.lnu.shape[:-1])

            else:
                beta = (
                    linregress(np.log10(self._lam[s]), np.log10(self._lnu[s]))[
                        0
                    ]
                    - 2.0
                )

        # If two windows are provided
        elif len(window) == 4:
            # Define the red and blue windows
            blue = window[:2]
            red = window[2:]

            # Measure the red and blue windows
            lnu_blue = self.measure_window_lnu(
                blue,
                nthreads=nthreads,
                integration_method=integration_method,
            )
            lnu_red = self.measure_window_lnu(
                red,
                nthreads=nthreads,
                integration_method=integration_method,
            )

            # Measure beta
            beta = (
                np.log10(lnu_blue / lnu_red)
                / np.log10(np.mean(blue) / np.mean(red))
                - 2.0
            )

        else:
            raise exceptions.InconsistentArguments(
                "A window of len 2 or 4 must be provided"
            )

        # Ensure the beta we return is a real number
        if np.any(np.isnan(beta)):
            if isinstance(beta, np.ndarray):
                beta[np.isnan(beta)] = 0.0
            elif isinstance(beta, (list, tuple)):
                beta = [0.0 if np.isnan(b) else b for b in beta]
            elif isinstance(beta, float):
                beta = 0.0

        return beta

    def get_fnu0(self):
        """Calculate the rest frame spectral flux density.

        Uses a standard distance of 10 pcs.

        This will also populate the observed wavelength and frequency arrays
        which in this case are the same as the emitted arrays.

        Returns:
            fnu (ndarray):
                Spectral flux density calcualted at d=10 pc.
        """
        # Get the observed wavelength and frequency arrays
        self.obslam = self._lam
        self.obsnu = self._nu

        # Compute the flux SED and apply unit conversions to get to nJy
        self.fnu = self.lnu / (4 * np.pi * (10 * pc) ** 2)

        return self.fnu

    def get_fnu(self, cosmo, z, igm=None):
        """Calculate the observed frame spectral energy distribution.

        This will also populate the observed wavelength and frequency arrays
        with the observer frame values.

        NOTE: if a redshift of 0 is passed the flux return will be calculated
        assuming a distance of 10 pc omitting IGM since at this distance
        IGM contribution makes no sense.

        Args:
            cosmo (astropy.cosmology):
                astropy cosmology instance.
            z (float):
                The redshift of the spectra.
            igm (igm):
                The IGM class. e.g. `synthesizer.igm.Inoue14`.
                Defaults to None.

        Returns:
            fnu (ndarray)
                Spectral flux density calcualted at d=10 pc

        """
        # Store the redshift for later use
        self.redshift = z

        # If we have a redshift of 0 then the below will break since the
        # distance will be 0. Instead call get_fnu0 to get the flux at 10 pc
        if self.redshift == 0:
            return self.get_fnu0()

        # Get the observed wavelength and frequency arrays
        self.obslam = self._lam * (1.0 + z)
        self.obsnu = self._nu / (1.0 + z)

        # Compute the luminosity distance
        luminosity_distance = cosmo.luminosity_distance(z).to("cm").value * cm

        # Finally, compute the flux SED and apply unit conversions to get
        # to nJy
        self.fnu = self.lnu * (1.0 + z) / (4 * np.pi * luminosity_distance**2)

        # If we are applying an IGM model apply it
        if igm is not None:
            # Support bot class references and instantiated objects
            if callable(igm):
                self._fnu *= igm().get_transmission(z, self.obslam)
            else:
                self._fnu *= igm.get_transmission(z, self.obslam)

        return self.fnu

    def get_photo_lnu(self, filters, verbose=True, nthreads=1):
        """Calculate broadband luminosities using a FilterCollection object.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.

        Returns:
            photo_lnu (dict):
                A dictionary of rest frame broadband luminosities.
        """
        # Intialise result dictionary
        photo_lnu = {}

        # Loop over filters
        for f in filters:
            # Apply the filter transmission curve and store the resulting
            # luminosity
            bb_lum = f.apply_filter(self._lnu, nu=self._nu, nthreads=nthreads)
            photo_lnu[f.filter_code] = bb_lum * self.lnu.units

        # Create the photometry collection and store it in the object
        self.photo_lnu = PhotometryCollection(filters, **photo_lnu)

        return self.photo_lnu

    def get_photo_fnu(self, filters, verbose=True, nthreads=1):
        """Calculate broadband fluxes using a FilterCollection object.

        Args:
            filters (FilterCollection):
                A FilterCollection object.
            verbose (bool):
                Are we talking?
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.

        Returns:
            (dict):
                A dictionary of fluxes in each filter in filters.
        """
        # Ensure fluxes actually exist
        if (self.obslam is None) | (self.fnu is None):
            return ValueError(
                (
                    "Fluxes not calculated, run `get_fnu` or "
                    "`get_fnu0` for observer frame or rest-frame "
                    "fluxes, respectively"
                )
            )

        # Set up flux dictionary
        photo_fnu = {}

        # Loop over filters in filter collection
        for f in filters:
            # Calculate and store the broadband flux in this filter
            bb_flux = f.apply_filter(
                self._fnu,
                nu=self._obsnu,
                nthreads=nthreads,
            )
            photo_fnu[f.filter_code] = bb_flux * self.fnu.units

        # Create the photometry collection and store it in the object
        self.photo_fnu = PhotometryCollection(filters, **photo_fnu)

        return self.photo_fnu

    def measure_colour(self, f1, f2):
        """Measure a broadband colour.

        Args:
            f1 (str):
                The blue filter code.
            f2 (str):
                The red filter code.

        Returns:
            (float):
                The broadband colour.
        """
        # Ensure fluxes exist
        if not bool(self.photo_fnu):
            raise ValueError(
                (
                    "Broadband fluxes not yet calculated, "
                    "run `get_photo_fnu` with a "
                    "FilterCollection"
                )
            )

        return 2.5 * np.log10(self.photo_fnu[f2] / self.photo_fnu[f1])

    @accepts(feature=angstrom, blue=angstrom, red=angstrom)
    def measure_index(self, feature, blue, red):
        """Measure an absorption feature index.

        Args:
            feature (tuple):
                Absorption feature window.
            blue (tuple):
                Blue continuum window for fitting.
            red (tuple):
                Red continuum window for fitting.

        Returns:
            index (float):
               Absorption feature index in units of wavelength
        """
        # self.lnu = np.array([self.lnu, self.lnu*2])

        # Measure the red and blue windows
        lnu_blue = self.measure_window_lnu(blue)
        lnu_red = self.measure_window_lnu(red)

        # Define the wavelength grid over the feature
        transmission = (self.lam > feature[0]) & (self.lam < feature[1])
        feature_lam = self.lam[transmission]

        # Extract mean values
        mean_blue = np.mean(blue)
        mean_red = np.mean(red)

        # Handle different spectra shapes
        if self.ndim >= 2:
            # Multiple spectra case

            # Perform polyfit for the continuum fit for all spectra
            continuum_fits = np.polyfit(
                [mean_blue, mean_red], [lnu_blue, lnu_red], 1
            )
            # Use the continuum fit to define the continuum for all spectra
            continuum = (
                np.column_stack(
                    continuum_fits[0]
                    * feature_lam.to(self.lam.units).value[:, np.newaxis]
                )
                + continuum_fits[1][:, np.newaxis]
            ) * self.lnu.units

            # Define the continuum subtracted spectrum for all SEDs
            feature_lum = self.lnu[:, transmission]
            feature_lum_continuum_subtracted = (
                -(feature_lum - continuum) / continuum
            )

            # Measure index for all SEDs
            index = np.trapz(
                feature_lum_continuum_subtracted, x=feature_lam, axis=1
            )

        else:
            # Single spectra case

            # Perform polyfit for the continuum fit
            continuum_fit = np.polyfit(
                [mean_blue, mean_red], [lnu_blue, lnu_red], 1
            )

            # Use the continuum fit to define the continuum
            continuum = (
                (continuum_fit[0] * feature_lam.to(self.lam.units).value)
                + continuum_fit[1]
            ) * self.lnu.units

            # Define the continuum subtracted spectrum
            feature_lum = self.lnu[transmission]
            feature_lum_continuum_subtracted = (
                -(feature_lum - continuum) / continuum
            )

            # Measure index
            index = np.trapz(feature_lum_continuum_subtracted, x=feature_lam)

        return index

    def get_resampled_sed(self, resample_factor=None, new_lam=None):
        """Resample the spectra onto a new set of wavelength points.

        This resampling can either be done by an integer number of wavelength
        elements per original wavelength element (i.e. up sampling),
        or by providing a new wavelength grid to resample on to.

        Args:
            resample_factor (int):
                The number of additional wavelength elements to
                resample to.
            new_lam (np.ndarray of float):
                The wavelength array to resample onto.

        Returns:
            Sed
                A new Sed with the rebinned rest frame spectra.

        Raises:
            InconsistentArgument
                Either resample factor or new_lam must be supplied. If neither
                or both are passed an error is raised.
        """
        start = tic()

        # Ensure we have what we need
        if resample_factor is None and new_lam is None:
            raise exceptions.InconsistentArguments(
                "Either resample_factor or new_lam must be specified"
            )

        # Both arguments are unecessary, tell the user what we will do
        if resample_factor is not None and new_lam is not None:
            warn("Got resample_factor and new_lam, ignoring resample_factor")

        # Resample the wavelength array
        if new_lam is None:
            new_lam = rebin_1d(self.lam, resample_factor, func=np.mean)

        # Evaluate the function at the desired wavelengths
        new_spectra = spectres(
            new_lam,
            self._lam,
            self._lnu,
            fill=0,
            verbose=False,
        )

        # Instantiate the new Sed
        sed = Sed(new_lam, new_spectra * self.lnu.units)

        # If self also has fnu we should resample those too and store the
        # shifted wavelengths and frequencies
        if self.fnu is not None:
            sed.obslam = sed.lam * (1.0 + self.redshift)
            sed.obsnu = sed.nu / (1.0 + self.redshift)
            sed.fnu = (
                spectres(
                    sed._obslam,
                    self._obslam,
                    self._fnu,
                    fill=0.0,
                )
                * self.fnu.units
            )
            sed.redshift = self.redshift

        # Clean up nans, we shouldn't get them but they do appear sometimes...
        sed._lnu = np.nan_to_num(sed._lnu)
        sed._fnu = np.nan_to_num(sed._fnu)
        sed._lam = np.nan_to_num(sed._lam)
        sed._nu = np.nan_to_num(sed._nu)
        sed._obslam = np.nan_to_num(sed._obslam)
        sed._obsnu = np.nan_to_num(sed._obsnu)

        toc("Resampling Sed", start)

        return sed

    def apply_instrument_lams(self, instrument, nthreads=1):
        """Apply an instrument to the spectra.

        This will resample a Sed object on to the instrument's wavelength
        array. It is actually just an intuitive wrapper around
        get_resampled_sed which will handle the actual resampling of
        both lnu and fnu (if applicable) using the instrument's
        wavelength array.

        Args:
            instrument (Instrument):
                An instance of the Instrument class.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.

        Returns:
            Sed
                A new Sed with the instrument applied.
        """
        # Ensure the instrument can actually do spectroscopy
        if not instrument.can_do_spectroscopy:
            raise exceptions.InconsistentArguments(
                f"{instrument.label} Instrument cannot do spectroscopy"
            )
        return self.get_resampled_sed(new_lam=instrument.lam)

    def apply_attenuation(
        self,
        tau_v,
        dust_curve,
        mask=None,
    ):
        """Apply attenuation to spectra.

        Args:
            tau_v (float/np.ndarray of float):
                The V-band optical depth for every star particle.
            dust_curve (synthesizer.emission_models.attenuation.*):
                An instance of one of the dust attenuation models. (defined in
                synthesizer/emission_models.attenuation.py)
            mask (array-like, bool):
                A mask array with an entry for each spectra. Masked out
                spectra will be ignored when applying the attenuation. Only
                applicable for Sed's holding an (N, Nlam) array.

        Returns:
            Sed
                A new Sed containing the rest frame spectra of self attenuated
                by the transmission defined from tau_v and the dust curve.
        """
        # Ensure the mask is compatible with the spectra
        if mask is not None:
            if self._lnu.ndim < 2:
                raise exceptions.InconsistentArguments(
                    "Masks are only applicable for Seds containing "
                    "multiple spectra"
                )
            if self._lnu.shape[: mask.ndim] != mask.shape:
                raise exceptions.InconsistentArguments(
                    "Mask and spectra are incompatible shapes "
                    f"({mask.shape}, {self._lnu.shape})"
                )

        # If tau_v is an array it needs to match the spectra shape
        if isinstance(tau_v, np.ndarray):
            if self._lnu.ndim < 2:
                raise exceptions.InconsistentArguments(
                    "Arrays of tau_v values are only applicable for Seds"
                    " containing multiple spectra"
                )
            if self._lnu.shape[0] != tau_v.size:
                raise exceptions.InconsistentArguments(
                    "tau_v and spectra are incompatible shapes "
                    f"({tau_v.shape}, {self._lnu.shape})"
                )

        # Compute the transmission
        transmission = dust_curve.get_transmission(tau_v, self.lam)

        # Get a copy of the rest frame spectra, we need to avoid
        # modifying the original
        spectra = np.copy(self._lnu)

        # Apply the transmission curve to the rest frame spectra with or
        # without applying a mask
        if mask is None:
            spectra *= transmission
        elif transmission.ndim > 1:
            spectra[mask] *= transmission[mask]
        else:
            spectra[mask] *= transmission

        return Sed(self.lam, lnu=spectra * self.lnu.units)

    @accepts(ionisation_energy=eV)
    def calculate_ionising_photon_production_rate(
        self, ionisation_energy=13.6 * eV, limit=100, nthreads=1
    ):
        """Calculate the ionising photon production rate.

        Args:
            ionisation_energy (unyt_array):
                The ionisation energy.
            limit (float/int):
                An upper bound on the number of subintervals
                used in the integration adaptive algorithm.
            nthreads (int):
                The number of threads to use for the integration. If -1 then
                all available threads are used.

        Returns:
            float:
                Ionising photon luminosity (s^-1).
        """
        # Convert lnu to llam
        llam = lnu_to_llam(self.lam, self.lnu)

        # Calculate ionisation wavelength
        ionisation_wavelength = h * c / ionisation_energy

        ionisation_mask = self.lam < ionisation_wavelength

        # Define integration arrays
        x = self._lam
        y = (llam * self.lam / h.to(erg / Hz) / c.to(angstrom / s)).value

        # Restrict arrays to ionisation regime
        x = x[ionisation_mask]
        if len(y.shape) == 1:
            y = y[ionisation_mask]
        else:
            y = y[..., ionisation_mask]

        # Add a final data point at the ionising energy to ensure full
        # coverage.
        x0 = ionisation_wavelength.to(angstrom).value
        if len(y.shape) == 1:
            y0 = np.interp(x0, x, y)
            y = np.append(y, y0)
        else:
            y0 = np.apply_along_axis(
                lambda y_: np.interp(x0, x, y_), axis=-1, arr=y
            )
            y0 = np.expand_dims(y0, -1)
            y = np.append(y, y0, axis=-1)

        x = np.append(x, x0)

        ion_photon_prod_rate = integrate_last_axis(x, y, nthreads=nthreads) / s

        return ion_photon_prod_rate

    def plot_spectra(self, **kwargs):
        """Plot the spectra.

        A wrapper for synthesizer.emissions.plot_spectra()
        """
        return plot_spectra(self, **kwargs)

    def plot_observed_spectra(self, **kwargs):
        """Plot the observed spectra.

        A wrapper for synthesizer.emissions.plot_observed_spectra()
        """
        return plot_observed_spectra(self, self.redshift, **kwargs)

    def plot_spectra_as_rainbow(self, **kwargs):
        """Plot the spectra as a rainbow.

        A wrapper for synthesizer.emissions.plot_spectra_as_rainbow()
        """
        return plot_spectra_as_rainbow(self, **kwargs)

    def plot_spectra_stack(
        self,
        nbin=100,
        fig=None,
        ax=None,
        show=False,
        order=None,
        quantity_to_plot="lnu",
        cmap="magma",
        vmin=None,
        vmax=None,
        xlimits=(),
    ):
        """Plot a stack of the top nbin spectra in a multi-spectra sed.

        This is only applicable for Sed object with ndim == 2. It will plot
        bin the top nbin spectra into nbin wavelength bins, populate a grid
        where x is the wavelength and each y row is one of the spectra, and
        then plot this grid as an image.

        This stack can optionally be ordered by their peak (from bluest
        peak to reddest peak), by their bolometric_luminosity (from dimmest to
        brightest), or by their luminosity at a specific wavelength (from
        dimmest to brightest). This is defined by the order keyword, "peak",
        "bolometric_luminosity", or a wavelength with units respectively.

        Args:
            nbin (int):
                The number of bins to use for the stacking.
            fig (matplotlib.pyplot.figure):
                The figure containing the axis. By default one is created in
                this function.
            ax (matplotlib.axes):
                The axis to plot the data on. By default one is created in
                this function.
            show (bool):
                Flag for whether to show the plot or just return the
                figure and axes.
            order (str/unyt_quantity):
                The order to stack the spectra by. Can be "peak",
                "bolometric_luminosity", a wavelength with units, or None. By
                default the spectra will be plotted with the same order they
                have in the Sed object.
            quantity_to_plot (str):
                The sed property to plot. Can be "lnu", "luminosity" or
                "llam" for rest frame spectra or "fnu", "flam" or "flux"
                for observed spectra. Defaults to "lnu".
            cmap (str):
                The matplotlib colormap to use for the plot. Defaults to
                "magma".
            vmin (float):
                The minimum value for the color scale. If None, then we will
                use 3 orders of magntitude below the maximum value.
            vmax (float):
                The maximum value for the color scale. If None, then we will
                use the maximum value of the spectra.
            xlimits (tuple of unyt_quantity):
                The limits for the x axis. If None, then we will use the
                minimum and maximum wavelength of the spectra.

        Returns:
            fig (matplotlib.pyplot.figure):
                The matplotlib figure object for the plot.
            ax (matplotlib.axes):
                The matplotlib axes object containing the plotted data.
        """
        # First ensure we have a valid Sed object to work with
        if self.ndim != 2:
            raise exceptions.InconsistentArguments(
                "This Sed object does not contain multiple spectra and thus "
                "cannot be stacked. Please use a Sed object with ndim > 1."
            )

        # Are we plotting in the rest_frame?
        rest_frame = quantity_to_plot in ("lnu", "llam", "luminosity")

        # If we aren't doing rest frame... check we actually have the
        # observed spectra
        if not rest_frame and self.fnu is None:
            raise exceptions.InconsistentArguments(
                "This Sed object does not contain observed spectra and thus "
                "cannot be stacked. Please use a Sed object with fnu."
            )

        # Define the wavelength bins based on nbin
        if len(xlimits) == 0:
            xlimits = (self.lam.min(), self.lam.max())
        elif len(xlimits) != 2:
            raise exceptions.InconsistentArguments(
                "xlimits must be a tuple of length 2!"
            )
        if not isinstance(xlimits[0], unyt_quantity):
            raise exceptions.InconsistentArguments(
                "xlimits must be a tuple of unyt_quantity!"
            )
        if not isinstance(xlimits[1], unyt_quantity):
            raise exceptions.InconsistentArguments(
                "xlimits must be a tuple of unyt_quantity!"
            )
        new_lams = (
            np.logspace(
                np.log10(xlimits[0]),
                np.log10(xlimits[1]),
                nbin,
            )
            * xlimits[0].units
        )

        # Resample the spectra to the new wavelength grid
        low_res_sed = self.get_resampled_sed(new_lam=new_lams)

        # Get the order of the spectra to stack
        if order == "peak":
            # Get the peak and it's index for each spectra
            peak_inds = np.argmax(low_res_sed._lnu, axis=-1)
            peak_lams = low_res_sed.lam[peak_inds].value

            # Sort the spectra by peak wavelength
            order = np.argsort(peak_lams)

            # Limit to the top nbin spectra
            order = order[:nbin]

        elif order == "bolometric_luminosity":
            # Get the bolometric luminosity for each spectra
            bolometric_luminosity = low_res_sed.bolometric_luminosity.value

            # Sort the spectra by bolometric luminosity
            order = np.argsort(bolometric_luminosity)[::-1]

            # Limit to the top nbin spectra
            order = order[:nbin]

        elif isinstance(order, unyt_quantity):
            # Ensure the wavelength is within the spectra range
            if order < low_res_sed.lam.min() or order > low_res_sed.lam.max():
                raise exceptions.InconsistentArguments(
                    f"Requested wavelength ({order}) is outside the "
                    f"range of the spectra ({low_res_sed.lam.min()}, "
                    f"{low_res_sed.lam.max()})"
                )

            # Get the luminosity at the requested wavelength for each spectra
            lum_at_lam = low_res_sed.get_lnu_at_lam(order).value

            # Sort the spectra by luminosity at the requested wavelength
            order = np.argsort(lum_at_lam)[::-1]

            # Limit to the top nbin spectra
            order = order[:nbin]

        elif order is None:
            # If no order is provided just use the first nbin spectra
            order = np.arange(nbin)

        else:
            raise exceptions.InconsistentArguments(
                f"Unrecognised order ({order}). Options are 'peak', "
                "'bolometric_luminosity', or a wavelength with units."
            )

        # Get the spectra to plot
        spectra = getattr(low_res_sed, quantity_to_plot)

        # Set up the array that will hold the stacked spectra
        stacked_spectra = (
            np.zeros((nbin, nbin), dtype=spectra.dtype) * spectra.units
        )

        # Loop over the resampled spectra and populate the grid from bottom
        # to top
        for row, i in enumerate(order):
            # Ensure we aren't trying to extract a spectra that doesn't exist
            if i >= spectra.shape[0]:
                continue

            # Populate the grid with the spectra
            stacked_spectra[row, :] = spectra[i, :]

        # Define a sensible normalisation for the color scale if not provided
        if vmin is None:
            vmin = stacked_spectra.max() / 1000.0
        if vmax is None:
            vmax = stacked_spectra.max()

        # If we don't already have a figure, make one
        if fig is None:
            # Set up the figure
            fig = plt.figure(figsize=(3.5, 3.5))

        # If we don't already have an axis, make one
        if ax is None:
            # Create the axes
            ax = fig.add_subplot(111)

        # Plot the stacked spectra as a heatmap
        im = ax.pcolormesh(
            new_lams,
            np.arange(nbin),
            stacked_spectra,
            shading="auto",
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax, clip=True),
        )

        # Set the x scale to log
        ax.set_xscale("log")

        # Parse the units for the labels and make them pretty
        x_units = self.lam.units.latex_repr
        y_units = spectra.units.latex_repr

        # Replace any \frac with a \ division
        pattern = r"\{(.*?)\}\{(.*?)\}"
        replacement = r"\1 \ / \ \2"
        x_units = re.sub(pattern, replacement, x_units).replace(r"\frac", "")
        y_units = re.sub(pattern, replacement, y_units).replace(r"\frac", "")

        # Label the x axis
        if rest_frame:
            ax.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")
        else:
            ax.set_xlabel(
                r"$\lambda_\mathrm{obs}/[\mathrm{" + x_units + r"}]$"
            )

        # Set the aspect ratio to be equal
        ax.set_aspect("auto")

        # Create a colorbar
        cbar = fig.colorbar(im, ax=ax)

        # Label the colorbar handling all possibilities
        if quantity_to_plot == "lnu":
            cbar.set_label(r"$L_{\nu}/[\mathrm{" + y_units + r"}]$")
        elif quantity_to_plot == "llam":
            cbar.set_label(r"$L_{\lambda}/[\mathrm{" + y_units + r"}]$")
        elif quantity_to_plot == "luminosity":
            cbar.set_label(r"$L/[\mathrm{" + y_units + r"}]$")
        elif quantity_to_plot == "fnu":
            cbar.set_label(r"$F_{\nu}/[\mathrm{" + y_units + r"}]$")
        elif quantity_to_plot == "flam":
            cbar.set_label(r"$F_{\lambda}/[\mathrm{" + y_units + r"}]$")
        else:
            cbar.set_label(r"$F/[\mathrm{" + y_units + r"}]$")

        # Are we showing the plot?
        if show:
            plt.show()

        return fig, ax


def plot_spectra(
    spectra,
    fig=None,
    ax=None,
    show=False,
    ylimits=(),
    xlimits=(),
    figsize=(3.5, 5),
    label=None,
    draw_legend=True,
    x_units=None,
    y_units=None,
    quantity_to_plot="lnu",
):
    """Plot a spectra or dictionary of spectra.

    Plots either a specific spectra or all spectra provided in a dictionary.
    The plotted "type" of spectra is defined by the quantity_to_plot keyword
    arrgument which defaults to "lnu".

    This is a generic plotting function to be used either directly or to be
    wrapped by helper methods through Synthesizer.

    Args:
        spectra (dict/Sed):
            The Sed objects from which to plot. This can either be a dictionary
            of Sed objects to plot multiple or a single Sed object to only plot
            one.
        fig (matplotlib.pyplot.figure):
            The figure containing the axis. By default one is created in this
            function.
        ax (matplotlib.axes):
            The axis to plot the data on. By default one is created in this
            function.
        show (bool):
            Flag for whether to show the plot or just return the
            figure and axes.
        ylimits (tuple):
            The limits to apply to the y axis. If not provided the limits
            will be calculated with the lower limit set to 1000 (100) times
            less than the peak of the spectrum for rest_frame (observed)
            spectra.
        xlimits (tuple):
            The limits to apply to the x axis. If not provided the optimal
            limits are found based on the ylimits.
        figsize (tuple):
            Tuple with size 2 defining the figure size.
        label (str):
            The label to give the spectra. Only applicable when Sed is a single
            spectra.
        draw_legend (bool):
            Whether to draw the legend.
        x_units (unyt.unit_object.Unit):
            The units of the x axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        y_units (unyt.unit_object.Unit):
            The units of the y axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        quantity_to_plot (str):
            The sed property to plot. Can be "lnu", "luminosity" or "llam"
            for rest frame spectra or "fnu", "flam" or "flux" for observed
            spectra. Defaults to "lnu".

    Returns:
        fig (matplotlib.pyplot.figure):
            The matplotlib figure object for the plot.
        ax (matplotlib.axes):
            The matplotlib axes object containing the plotted data.
    """
    # Check we have been given a valid quantity_to_plot
    if quantity_to_plot not in (
        "lnu",
        "llam",
        "luminosity",
        "fnu",
        "flam",
        "flux",
    ):
        raise exceptions.InconsistentArguments(
            f"{quantity_to_plot} is not a valid quantity_to_plot"
            "(can be 'fnu' or 'flam')"
        )

    # Are we plotting in the rest_frame?
    rest_frame = quantity_to_plot in ("lnu", "llam", "luminosity")

    # Make a singular Sed a dictionary for ease below
    if isinstance(spectra, Sed):
        spectra = {
            label if label is not None else "spectra": spectra,
        }

        # Don't draw a legend if not label given
        if label is None and draw_legend:
            warn("No label given, we will not draw a legend")
            draw_legend = False

    # If we don't already have a figure, make one
    if fig is None:
        # Set up the figure
        fig = plt.figure(figsize=figsize)

        # Define the axes geometry
        left = 0.15
        height = 0.6
        bottom = 0.1
        width = 0.8

        # Create the axes
        ax = fig.add_axes((left, bottom, width, height))

        # Set the scale to log log
        ax.loglog()

    # Loop over the dict we have been handed, we want to do this backwards
    # to ensure the most recent spectra are on top
    keys = list(spectra.keys())[::-1]
    seds = list(spectra.values())[::-1]
    for key, sed in zip(keys, seds):
        # Get the appropriate luminosity/flux and wavelengths
        if rest_frame:
            lam = sed.lam
        else:
            # Ensure we have fluxes
            if sed.fnu is None:
                raise exceptions.MissingSpectraType(
                    f"This Sed has no fluxes ({key})! Have you called "
                    "Sed.get_fnu()?"
                )

            # Ok everything is fine
            lam = sed.obslam

        plt_spectra = getattr(sed, quantity_to_plot)

        # Prettify the label if not latex
        if not any([c in key for c in ("$", "_")]):
            key = key.replace("_", " ").title()

        # Plot this spectra
        ax.plot(lam, plt_spectra, lw=1, alpha=0.8, label=key)

    # Do we not have y limtis?
    if len(ylimits) == 0:
        # Define initial xlimits
        ylimits = [np.inf, -np.inf]

        # Loop over spectra and get the total required limits
        for sed in spectra.values():
            # Get the maximum ignoring infinites
            okinds = np.logical_and(
                getattr(sed, quantity_to_plot) > 0,
                getattr(sed, quantity_to_plot) < np.inf,
            )
            if True not in okinds:
                continue
            max_val = np.nanmax(getattr(sed, quantity_to_plot)[okinds])

            # Derive the x limits
            y_up = 10 ** (np.log10(max_val) * 1.05)
            y_low = 10 ** (np.log10(max_val) - 5)

            # Update limits
            if y_low < ylimits[0]:
                ylimits[0] = y_low
            if y_up > ylimits[1]:
                ylimits[1] = y_up

    # Do we not have x limits?
    if len(xlimits) == 0:
        # Define initial xlimits
        xlimits = [np.inf, -np.inf]

        # Loop over spectra and get the total required limits
        for sed in spectra.values():
            # Derive the x limits from data above the ylimits
            plt_spectra = getattr(sed, quantity_to_plot)
            lam_mask = plt_spectra > ylimits[0]
            if rest_frame:
                lams_above = sed.lam[lam_mask]
            else:
                lams_above = sed.obslam[lam_mask]

            # Saftey skip if no values are above the limit
            if lams_above.size == 0:
                continue

            # Derive the x limits
            x_low = 10 ** (np.log10(np.min(lams_above)) * 0.9)
            x_up = 10 ** (np.log10(np.max(lams_above)) * 1.1)

            # Update limits
            if x_low < xlimits[0]:
                xlimits[0] = x_low
            if x_up > xlimits[1]:
                xlimits[1] = x_up

    # Set the limits
    if not np.isnan(xlimits[0]) and not np.isnan(xlimits[1]):
        ax.set_xlim(*xlimits)
    if not np.isnan(ylimits[0]) and not np.isnan(ylimits[1]):
        ax.set_ylim(*ylimits)

    # Make the legend
    if draw_legend and any(ax.get_legend_handles_labels()[1]):
        ax.legend(fontsize=8, labelspacing=0.0)

    # Parse the units for the labels and make them pretty
    if x_units is None:
        x_units = lam.units.latex_repr
    else:
        x_units = str(x_units)
    if y_units is None:
        y_units = plt_spectra.units.latex_repr
    else:
        y_units = str(y_units)

    # Replace any \frac with a \ division
    pattern = r"\{(.*?)\}\{(.*?)\}"
    replacement = r"\1 \ / \ \2"
    x_units = re.sub(pattern, replacement, x_units).replace(r"\frac", "")
    y_units = re.sub(pattern, replacement, y_units).replace(r"\frac", "")

    # Label the x axis
    if rest_frame:
        ax.set_xlabel(r"$\lambda/[\mathrm{" + x_units + r"}]$")
    else:
        ax.set_xlabel(r"$\lambda_\mathrm{obs}/[\mathrm{" + x_units + r"}]$")

    # Label the y axis handling all possibilities
    if quantity_to_plot == "lnu":
        ax.set_ylabel(r"$L_{\nu}/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "llam":
        ax.set_ylabel(r"$L_{\lambda}/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "luminosity":
        ax.set_ylabel(r"$L/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "fnu":
        ax.set_ylabel(r"$F_{\nu}/[\mathrm{" + y_units + r"}]$")
    elif quantity_to_plot == "flam":
        ax.set_ylabel(r"$F_{\lambda}/[\mathrm{" + y_units + r"}]$")
    else:
        ax.set_ylabel(r"$F/[\mathrm{" + y_units + r"}]$")

    # Are we showing?
    if show:
        plt.show()

    return fig, ax


def plot_observed_spectra(
    spectra,
    redshift,
    fig=None,
    ax=None,
    show=False,
    ylimits=(),
    xlimits=(),
    figsize=(3.5, 5),
    label=None,
    draw_legend=True,
    x_units=None,
    y_units=None,
    filters=None,
    quantity_to_plot="fnu",
):
    """Plot a spectra or dictionary of spectra.

    Plots either a specific observed spectra or all observed spectra
    provided in a dictionary.

    This function is a wrapper around plot_spectra.

    This is a generic plotting function to be used either directly or to be
    wrapped by helper methods through Synthesizer.

    Args:
        spectra (dict/Sed):
            The Sed objects from which to plot. This can either be a dictionary
            of Sed objects to plot multiple or a single Sed object to only plot
            one.
        redshift (float):
            The redshift of the observation.
        fig (matplotlib.pyplot.figure):
            The figure containing the axis. By default one is created in this
            function.
        ax (matplotlib.axes):
            The axis to plot the data on. By default one is created in this
            function.
        show (bool):
            Flag for whether to show the plot or just return the
            figure and axes.
        ylimits (tuple):
            The limits to apply to the y axis. If not provided the limits
            will be calculated with the lower limit set to 1000 (100) times
            less than the peak of the spectrum for rest_frame (observed)
            spectra.
        xlimits (tuple):
            The limits to apply to the x axis. If not provided the optimal
            limits are found based on the ylimits.
        figsize (tuple):
            Tuple with size 2 defining the figure size.
        label (str):
            The label to give the spectra. Only applicable when Sed is a single
            spectra.
        draw_legend (bool):
            Whether to draw the legend.
        x_units (unyt.unit_object.Unit):
            The units of the x axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        y_units (unyt.unit_object.Unit):
            The units of the y axis. This will be converted to a string
            and included in the axis label. By default the internal unit system
            is assumed unless this is passed.
        filters (FilterCollection):
            If given then the photometry is computed and both the photometry
            and filter curves are plotted
        quantity_to_plot (str):
            The sed property to plot. Can be "fnu", "flam", or "flux".
            Defaults to "fnu".

    Returns:
        fig (matplotlib.pyplot.figure)
            The matplotlib figure object for the plot.
        ax (matplotlib.axes)
            The matplotlib axes object containing the plotted data.
    """
    # Check we have been given a valid quantity_to_plot
    if quantity_to_plot not in ("fnu", "flam"):
        raise exceptions.InconsistentArguments(
            f"{quantity_to_plot} is not a valid quantity_to_plot"
            "(can be 'fnu' or 'flam')"
        )

    # Get the observed spectra plot
    fig, ax = plot_spectra(
        spectra,
        fig=fig,
        ax=ax,
        show=False,
        ylimits=ylimits,
        xlimits=xlimits,
        figsize=figsize,
        label=label,
        draw_legend=draw_legend,
        x_units=x_units,
        y_units=y_units,
        quantity_to_plot=quantity_to_plot,
    )

    # Are we including photometry and filters?
    if filters is not None:
        # Add a filter axis
        filter_ax = ax.twinx()
        filter_ax.set_ylim(0, None)

        # PLot each filter curve
        for f in filters:
            filter_ax.plot(f.lam * (1 + redshift), f.t)

        # Make a singular Sed a dictionary for ease below
        if isinstance(spectra, Sed):
            spectra = {
                label if label is not None else "spectra": spectra,
            }

        # Loop over spectra plotting photometry and filter curves
        for sed in spectra.values():
            # Get the photometry
            sed.get_photo_fnu(filters)

            # Plot the photometry for each filter
            for f in filters:
                piv_lam = f.pivwv()
                ax.scatter(
                    piv_lam * (1 + redshift),
                    sed.photo_fnu[f.filter_code],
                    zorder=4,
                )

    if show:
        plt.show()

    return fig, ax


def plot_spectra_as_rainbow(
    sed,
    figsize=(5, 0.5),
    lam_min=3000,
    lam_max=8000,
    include_xaxis=True,
    logged=False,
    min_log_lnu=-2.0,
    use_fnu=False,
):
    """Create a plot of the spectrum as a rainbow.

    Arguments:
        sed (synthesizer.emissions.Sed):
            A synthesizer.emissions object.
        figsize (tuple):
            Fig-size tuple (width, height).
        lam_min (float):
            The min wavelength to plot in Angstroms.
        lam_max (float):
            The max wavelength to plot in Angstroms.
        include_xaxis (bool):
            Flag whther to include x-axis ticks and label.
        logged (bool):
            Flag whether to use logged luminosity.
        min_log_lnu (float):
            Minium luminosity to plot relative to the maximum.
        use_fnu (bool):
            Whether to plot fluxes or luminosities. If True
            fluxes are plotted, otherwise luminosities.

    Returns:
        fig (matplotlib.pyplot.figure)
            The matplotlib figure object for the plot.
        ax (matplotlib.axes)
            The matplotlib axes object containing the plotted data.
    """
    # Take sum of Seds if two dimensional
    sed = sed.sum()

    if use_fnu:
        # Define filter for spectra
        wavelength_indices = np.logical_and(
            sed._obslam < lam_max, sed._obslam > lam_min
        )
        lam = sed.obslam[wavelength_indices].to("nm").value
        spectra = sed._fnu[wavelength_indices]
    else:
        # define filter for spectra
        wavelength_indices = np.logical_and(
            sed._lam < lam_max, sed._lam > lam_min
        )
        lam = sed.lam[wavelength_indices].to("nm").value
        spectra = sed._lnu[wavelength_indices]

    # Normalise spectrum
    max_val = np.max(spectra)
    if max_val > 0:
        spectra /= max_val

    # If logged rescale to between 0 and 1 using min_log_lnu
    if logged:
        spectra = (np.log10(spectra) - min_log_lnu) / (-min_log_lnu)
        spectra[spectra < min_log_lnu] = 0

    # initialise figure
    fig = plt.figure(figsize=figsize)

    # Initialise axes
    if include_xaxis:
        ax = fig.add_axes((0, 0.3, 1, 1))
        ax.set_xlabel(r"$\lambda/\AA$")
    else:
        ax = fig.add_axes((0, 0.0, 1, 1))
        ax.set_xticks([])

    # Set background
    ax.set_facecolor("black")

    # Always turn off y-ticks
    ax.set_yticks([])

    # Get an array of colours
    colours = np.array(
        [
            wavelength_to_rgba(lam_, alpha=spectra_)
            for lam_, spectra_ in zip(lam, spectra)
        ]
    )

    # Expand dimensions to get an image array
    im = np.expand_dims(colours, axis=0)

    # Show image
    ax.imshow(im, aspect="auto", extent=(lam_min, lam_max, 0, 1))

    return fig, ax
