import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy import integrate
from unyt import c, h, nJy, erg, s, Hz, pc, angstrom, eV, unyt_array, Angstrom

from . import exceptions
from .utils import rebin_1d
from .units import Quantity, Units
from .igm import Inoue14


units = Units()


class Sed:

    """
    A class representing a spectral energy distribution (SED).

    Attributes:
        lam (ndarray)
            the wavelength grid in Angstroms
        nu (ndarray)
            frequency in Hz
        lnu (ndarray)
            the spectral luminosity density

    Methods:
    """

    # Define Quantities, for details see units.py
    lam = Quantity()
    nu = Quantity()
    lnu = Quantity()
    fnu = Quantity()
    obsnu = Quantity()
    obslam = Quantity()

    def __init__(self, lam, lnu=None, description=False):
        """
        Initialise a new spectral energy distribution object.

        Args:
            lam (ndarray)
                the wavelength grid in Angstroms
            lnu (ndarray)
                the spectral luminosity density

        """

        # Set the description
        self.description = description

        # Set the wavelength
        self.lam = lam  # \AA

        # If no lnu is provided create an empty array with the same shape as
        # lam.
        if lnu is None:
            self.lnu = np.zeros(self.lam.shape)
        else:
            self.lnu = lnu

        # Calculate frequency
        self.nu = (c / (self.lam)).to("Hz").value  # Hz

        # Redshift of the SED
        self.redshift = 0

        # The wavelengths and frequencies in the observer frame
        self.obslam = None
        self.obsnu = None
        self.fnu = None

        # Broadband photometry
        self.broadband_luminosities = None
        self.broadband_fluxes = None

    def concat(self, *other_seds):
        """
        Concatenate the spectra arrays of multiple Sed objects.

        This will combine the arrays along the first axis. For example
        concatenating two Seds with Sed.lnu.shape = (10, 1000) and
        Sed.lnu.shape = (20, 1000) will result in a new Sed with
        Sed.lnu.shape = (30, 1000). The wavelength array of
        the resulting Sed will be the array on self.

        Incompatible spectra shapes will raise an error.

        Args:
            other_seds (object, Sed)
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

            # Concatenate this lnu array
            new_lnu = np.concatenate(new_lnu, other_sed._lnu)

        return Sed(self._lam, new_lnu)

    def __add__(self, second_sed):
        """
        Overide addition operator to allow two Sed objects to be added
        together.

        Args:
            second_sed (object, Sed)
                The Sed object to combine with self.

        Returns:
            Sed
                A new instance of Sed with added lnu arrays.

        Raises:
            InconsistentAddition
                If wavelength arrays or lnu arrays are incompatible an error
                is raised.
        """

        # Ensure the wavelength arrays are compatible
        # # NOTE: this is probably overkill and too costly. We
        # could instead check the first and last entry and the shape.
        # In rare instances this could fail though.
        if not np.array_equal(self._lam, second_sed._lam):
            raise exceptions.InconsistentAddition("Wavelength grids must be identical")

        # Ensure the lnu arrays are compatible
        # This check is redudant for Sed.lnu.shape = (nlam, ) spectra but will
        # not erroneously error. Nor is it expensive.
        if self._lnu.shape[0] != second_sed._lnu.shape[0]:
            raise exceptions.InconsistentAddition("SEDs must have same dimensions")

        # They're compatible, add them
        return Sed(self._lam, lnu=self._lnu + second_sed._lnu)

    def __str__(self):
        """
        Overloads the __str__ operator. A summary can be achieved by
        print(sed) where sed is an instance of Sed.
        """

        # Set up string for printing
        pstr = ""

        # Add the content of the summary to the string to be printed
        pstr += "-" * 10 + "\n"
        pstr += "SUMMARY OF SED \n"
        pstr += f"Number of wavelength points: {len(self._lam)} \n"
        pstr += f"Wavelength range: [{np.min(self.lam):.2f}, \
            {np.max(self.lam):.2f}] \n"
        pstr += f"log10(Peak luminosity/{units.lnu}): \
            {np.log10(np.max(self.lnu)):.2f} \n"
        bolometric_luminosity = self.measure_bolometric_luminosity()
        pstr += f"log10(Bolometric luminosity/{bolometric_luminosity.units}): \
            {np.log10(bolometric_luminosity):.2f} \n"
        pstr += "-" * 10

        return pstr

    @property
    def _spec_dims(self):
        return np.ndim(self.lnu)

    def _get_lnu_at_nu(self, nu, kind=False):
        """
        A simple internal function for getting lnu at nu assuming the default
        unit system.

        Args:
            nu (array or float)
                frequency(s) of interest

        Returns:
            luminosity (array or float)
                luminosity (lnu) at the provided wavelength

        """

        return interp1d(self._nu, self._lnu, kind=kind)(nu)

    def get_lnu_at_nu(self, nu, kind=False):
        """
        Return lnu with units at a provided frequency using 1d interpolation.

        Args:
            wavelength (array or float)
                wavelength(s) of interest
            kind (str)
                interpolation kind

        Returns:
            luminosity (unyt_array)
                luminosity (lnu) at the provided wavelength

        """

        return self._get_lnu_at_nu(nu.to(units.nu).value, kind=kind) * units.lnu

    def _get_lnu_at_lam(self, lam, kind=False):
        """
        Return lnu without units at a provided wavelength using 1d
        interpolation.

        Args:
            lam (array or float)
                wavelength(s) of interest

        Returns:
            luminosity (array or float)
                luminosity (lnu) at the provided wavelength

        """

        return interp1d(self._lam, self._lnu, kind=kind)(lam)

    def get_lnu_at_lam(self, lam, kind=False):
        """
        Return lnu at a provided wavelength.

        Args:
            lam (array or float)
                wavelength(s) of interest

        Returns:
            luminosity (array or float)
                luminosity (lnu) at the provided wavelength

        """

        return self._get_lnu_at_lam(lam.to(units.lam).value, kind=kind) * units.lnu

    def measure_bolometric_luminosity(self, method="trapz"):
        """
        Calculate the bolometric luminosity of the SED by simply integrating
        the SED.

        Args:
            method (str)
                The method used to calculate the bolometric luminosity. Options
                include 'trapz' and 'quad'.

        Returns:
            bolometric_luminosity (float)
                The bolometric luminosity.

        """

        if method == "trapz":
            bolometric_luminosity = np.trapz(self.lnu[::-1], x=self.nu[::-1])
        if method == "quad":
            bolometric_luminosity = (
                integrate.quad(self._get_lnu_at_nu, 1e12, 1e16)[0] * units.luminosity
            )

        return bolometric_luminosity

    def measure_window_luminosity(self, window, method="trapz"):
        """
        Measure the luminosity in a spectral window.

        Args:
            window (tuple of floats)
                The window in wavelength

        Returns:
            luminosity (float)
                The luminosity in the window.
        """

        if method == "quad":
            """
            Convert wavelength limits to frequency limits and convert to
            base units.
            """
            lims = (c / np.array(window)).to(units.nu).value
            luminosity = (
                integrate.quad(self._get_lnu_at_nu, *lims)[0] * units.luminosity
            )

        if method == "trapz":
            # define a pseudo transmission function
            transmission = (self.lam > window[0]) & (self.lam < window[1])
            transmission = transmission.astype(float)
            luminosity = np.trapz(self.lnu[::-1] * transmission[::-1], x=self.nu[::-1])

        return luminosity.to(units.luminosity)

    def measure_window_lnu(self, window, method="trapz"):
        """
        Measure lnu in a spectral window.

        Args:
            window (tuple of floats)
                The window in wavelength
            method (str)
                The method to use for the integration

        Returns:
            luminosity (float)
                The luminosity in the window.
        """

        if method == "average":
            # define a pseudo transmission function
            transmission = (self.lam > window[0]) & (self.lam < window[1])
            transmission = transmission.astype(float)

            if self._spec_dims == 2:
                Lnu = (
                    np.array(
                        [
                            np.sum(_lnu * transmission) / np.sum(transmission)
                            for _lnu in self._lnu
                        ]
                    )
                    * units.lnu
                )

            else:
                Lnu = np.sum(self.lnu * transmission) / np.sum(transmission)

        if method == "trapz":
            # define a pseudo transmission function
            transmission = (self.lam > window[0]) & (self.lam < window[1])
            transmission = transmission.astype(float)

            nu = self.nu[::-1]

            if self._spec_dims == 2:
                Lnu = (
                    np.array(
                        [
                            np.trapz(_lnu[::-1] * transmission[::-1] / nu, x=nu)
                            / np.trapz(transmission[::-1] / nu, x=nu)
                            for _lnu in self._lnu
                        ]
                    )
                    * units.lnu
                )

            else:
                lnu = self.lnu[::-1]
                Lnu = np.trapz(lnu * transmission[::-1] / nu, x=nu) / np.trapz(
                    transmission[::-1] / nu, x=nu
                )

        # note: not yet adapted for multiple SEDs
        if method == "quad":
            # define limits in base units
            lims = (c / window).to(units.nu).value

            def func(x):
                return self._get_lnu_at_nu(x) / x

            def inv(x):
                return 1 / x

            Lnu = integrate.quad(func, *lims)[0] / integrate.quad(inv, *lims)[0]

            Lnu = Lnu * units.lnu

        return Lnu.to(units.lnu)

    def measure_break(self, blue, red):
        """
        Measure a spectral break (e.g. the Balmer break) or D4000 using two
        windows.

        Args:
            blue (tuple of floats)
                The blue window
            red (tuple of floats)
                The red window

        Returns:
            break
                The ratio of the luminosity in the two windows.
        """
        return self.measure_window_lnu(red) / self.measure_window_lnu(blue)

    def measure_balmer_break(self):
        """
        Measure the Balmer break using two windows at (3400,3600) and
        (4150,4250)

        Returns:
            float
                The Balmer break strength
        """

        blue = (3400, 3600) * Angstrom
        red = (4150, 4250) * Angstrom

        return self.measure_break(blue, red)

    def measure_d4000(self, definition="Bruzual83"):
        """
        Measure the D4000 index using either the Bruzual83 or Balogh
        definitions.

        Args:
            definition
                The choice of definition 'Bruzual83' or 'Balogh'

        Returns:
            float
                The Balmer break strength
        """
        if definition == "Bruzual83":
            blue = (3750, 3950) * Angstrom
            red = (4050, 4250) * Angstrom

        if definition == "Balogh":
            blue = (3850, 3950) * Angstrom
            red = (4000, 4100) * Angstrom

        return self.measure_break(blue, red)

    def measure_beta(self, window=(1250.0, 3000.0)):
        """
        Measure the UV continuum slope ($\beta$) measured using the provided
        window. If the window has len(2) a full fit to the spectra is performed
        else the luminosity in two windows is calculated and used to determine
        the slope, similar to observations.

        Returns:
            float
                The UV continuum slope ($\beta$)
        """

        # if a single window is provided
        if len(window) == 2:
            s = (self.lam > window[0]) & (self.lam < window[1])

            if self._spec_dims == 2:
                beta = np.array(
                    [
                        linregress(np.log10(self._lam[s]), np.log10(_lnu[..., s]))[0]
                        - 2.0
                        for _lnu in self.lnu
                    ]
                )

            else:
                beta = (
                    linregress(np.log10(self._lam[s]), np.log10(self._lnu[s]))[0] - 2.0
                )

        # if two windows are provided
        elif len(window) == 4:
            # define the red and blue windows
            blue = window[:2]
            red = window[2:]

            # measure the red and blue windows
            lnu_blue = self.measure_window_lnu(blue)
            lnu_red = self.measure_window_lnu(red)

            # measure beta
            beta = (
                np.log10(lnu_blue / lnu_red) / np.log10(np.mean(blue) / np.mean(red))
                - 2.0
            )

        else:
            # raise exception
            print("a window of len 2 or 4 must be provided")

        return beta

    def get_broadband_luminosities(self, filters):
        """
        Calculate broadband luminosities using a FilterCollection object

        Arguments:
            filters (filters.FilterCollection)
                A FilterCollection object.

        Returns:
            broadband_luminosities (dict)
                A dictionary of broadband luminosities.
        """

        self.broadband_luminosities = {}

        for f in filters:
            # Apply the filter transmission curve and store the resulting
            # luminosity
            bb_lum = f.apply_filter(self._lnu, nu=self._nu) * units.lnu
            self.broadband_luminosities[f.filter_code] = bb_lum

        return self.broadband_luminosities

    def get_fnu0(self):
        """
        Calculate a dummy observed frame spectral energy distribution.
        Useful when you want rest-frame quantities.

        Uses a standard distance of 10 pc

        Returns:
            fnu (ndarray)
                Spectral flux density calcualted at d=10 pc
        """

        # Get the observed wavelength and frequency arrays
        self.obslam = self._lam
        self.obsnu = self._nu

        # Compute the flux SED and apply unit conversions to get to nJy
        self.fnu = self._lnu / (4 * np.pi * (10 * pc).to("cm").value)
        self._fnu *= 1e23  # convert to Jy
        self._fnu *= 1e9  # convert to nJy

        return self.fnu

    def get_fnu(self, cosmo, z, igm=Inoue14):
        """
        Calculate the observed frame spectral energy distribution

        Args:
            cosmo (astropy.cosmology)
                astropy cosmology instance
            z (float)
                redshift
            igm (igm)
                IGM class

        Returns:
            fnu (ndarray)
                Spectral flux density calcualted at d=10 pc

        """

        # Store the redshift for later use
        self.redshift = z

        # Get the observed wavelength and frequency arrays
        self.obslam = self._lam * (1.0 + z)
        self.obsnu = self._nu / (1.0 + z)

        # Compute the luminosity distance
        luminosity_distance = cosmo.luminosity_distance(z).to("cm").value

        # Finally, compute the flux SED and apply unit conversions to get
        # to nJy
        self.fnu = self._lnu * (1.0 + z) / (4 * np.pi * luminosity_distance**2)
        self._fnu *= 1e23  # convert to Jy
        self._fnu *= 1e9  # convert to nJy

        # If we are applying an IGM model apply it
        if igm:
            self._fnu *= igm().T(z, self._obslam)

        return self.fnu

    def get_broadband_fluxes(self, fc, verbose=True):  # broad band flux/nJy
        """
        Calculate broadband luminosities using a FilterCollection object

        Args:
            fc (object)
                a FilterCollection object

        Returns:
            (dict)
                a dictionary of fluxes
        """

        if (self.obslam is None) | (self.fnu is None):
            return ValueError(
                (
                    "Fluxes not calculated, run `get_fnu` or "
                    "`get_fnu0` for observer frame or rest-frame "
                    "fluxes, respectively"
                )
            )

        self.broadband_fluxes = {}

        # loop over filters in filter collection
        for f in fc:
            # Check whether the filter transmission curve wavelength grid
            # and the spectral grid are the same array
            if not np.array_equal(f.lam, self.lam):
                if verbose:
                    print(
                        (
                            "WARNING: filter wavelength grid is not "
                            "the same as the SED wavelength grid."
                        )
                    )

            # Calculate and store the broadband flux in this filter
            bb_flux = f.apply_filter(self._fnu, nu=self._obsnu) * nJy
            self.broadband_fluxes[f.filter_code] = bb_flux

        return self.broadband_fluxes

    def measure_colour(self, f1, f2):
        """
        Measure a broadband colour.

        Args:
            f1 (str)
                blue filter code
            f2 (str)
                red filter code

        Returns:
            (float)
                the broadband colour.
        """

        if not bool(self.broadband_fluxes):
            raise ValueError(
                (
                    "Broadband fluxes not yet calculated, "
                    "run `get_broadband_fluxes` with a "
                    "FilterCollection"
                )
            )

        return 2.5 * np.log10(self.broadband_fluxes[f2] / self.broadband_fluxes[f1])

    def measure_index(self, feature, blue, red):
        """
        Measure an asorption feature index.

        Args:
            absorption (tuple)
                Absoprtion feature window.
            blue (tuple)
                Blue continuum window for fitting.
            red (tuple)
                Red continuum window for fitting.

        Returns:
            index (float)
                Absorption feature index in units of wavelength
        """

        # measure the red and blue windows
        lnu_blue = self.measure_window_lnu(blue)
        lnu_red = self.measure_window_lnu(red)

        # define the wavelength grid over the feature
        transmission = (self.lam > feature[0]) & (self.lam < feature[1])
        feature_lam = self.lam[transmission]

        # using the red and blue windows fit the continuum
        # note, this does not conserve units so we need to add them back in
        # later.

        if self._spec_dims == 2:
            # set up output array
            index = np.zeros(len(self.lnu)) * units.lam

            # Note: I'm sure this could be done better.
            for i, _lnu in enumerate(self.lnu):
                continuum_fit = np.polyfit(
                    [np.mean(blue), np.mean(red)], [lnu_blue[i], lnu_red[i]], 1
                )

                # use the continuum fit to define the continuum
                continuum = (
                    (continuum_fit[0] * feature_lam.to(units.lam).value)
                    + continuum_fit[1]
                ) * units.lnu

                # define the continuum subtracted spectrum
                feature_lum = _lnu[transmission]
                feature_lum_continuum_subtracted = (
                    -(feature_lum - continuum) / continuum
                )

                # measure index
                index[i] = np.trapz(feature_lum_continuum_subtracted, x=feature_lam)

        else:
            continuum_fit = np.polyfit(
                [np.mean(blue), np.mean(red)], [lnu_blue, lnu_red], 1
            )

            # use the continuum fit to define the continuum
            continuum = (
                (continuum_fit[0] * feature_lam.to(units.lam).value) + continuum_fit[1]
            ) * units.lnu

            # define the continuum subtracted spectrum
            feature_lum = self.lnu[transmission]

            feature_lum_continuum_subtracted = -(feature_lum - continuum) / continuum

            # measure index
            index = np.trapz(feature_lum_continuum_subtracted, x=feature_lam)

        return index

    def get_resampled_sed(self, n):
        """
        Resameple the spectra and create a new Sed object

        Args:
            n (int or float)

        Returns:
            (sed.Sed)
                A rebinned Sed
        """

        if isinstance(n, int):
            sed = Sed(
                rebin_1d(self.lam, n, func=np.mean), rebin_1d(self.lnu, n, func=np.mean)
            )

        elif isinstance(n, float):
            raise exceptions.UnimplementedFunctionality(
                "Non-integer resampling not yet implemented."
            )

        else:
            raise exceptions.InconsistentArguments(
                "Sampling factor must be integer or float."
            )

        return sed


def calculate_Q(lam, lnu, ionisation_energy=13.6 * eV, limit=100):
    """
    An improved function to calculate the ionising production rate.

    Args:
        lam (float array)
            wavelength grid
        lnu (float array)
            luminosity grid (erg/s/Hz)
        ionisation_energy (unyt_array)
            ionisation energy
        limit (float or int, optional)
            An upper bound on the number of subintervals
            used in the integration adaptive algorithm.

    Returns
        float
            ionising photon luminosity (s^-1)
    """

    if not isinstance(lam, unyt_array):
        lam = lam * angstrom

    if not isinstance(lnu, unyt_array):
        lnu = lnu * erg / s / Hz

    # convert lnu to llam
    llam = lnu * c / lam**2

    # convert llam to lum [THIS SEEMS REDUNDANT]
    lum = llam * lam

    # caculate ionisation wavelength
    ionisation_wavelength = h * c / ionisation_energy

    x = lam.to("Angstrom").value
    y = lum.to("erg/s").value / (h.to("erg/Hz").value * c.to("Angstrom/s").value)

    def f(x_):
        return np.interp(x_, x, y)

    return integrate.quad(
        f, 0, ionisation_wavelength.to("Angstrom").value, limit=limit
    )[0]


# def rebin(l, f, n):  # rebin SED [currently destroys original]
#     n_len = int(np.floor(len(l) / n))
#     _l = l[: n_len * n]
#     _f = f[: n_len * n]
#     nl = np.mean(_l.reshape(n_len, n), axis=1)
#     nf = np.sum(_f.reshape(n_len, n), axis=1) / n

#     return nl, nf
