"""Module containing dust emission functionality
"""
import numpy as np
from scipy import integrate
from unyt import h, c, kb, um, erg, s, Hz
from unyt import accepts
from unyt.dimensions import temperature as temperature_dim
from unyt import Angstrom, unyt_quantity, unyt_array

from synthesizer import exceptions
from synthesizer.utils import planck
from synthesizer.sed import Sed


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
