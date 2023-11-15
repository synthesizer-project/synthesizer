""" Functionality related to spectra storage and manipulation.

When a spectra is computed from a `Galaxy` or a Galaxy component the resulting
calculated spectra are stored in `Sed` objects. These provide helper functions
for quick manipulation of the spectra. Seds can contain a single spectra or
arbitrarily many with all methods capable of acting on both consistently.

Example usage:

    sed = Sed(lams, lnu)
    sed.get_fnu(redshift)
    sed.apply_attenutation(tau_v=0.7)
    sed.get_broadband_fluxes(filters)
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress
from scipy import integrate
from unyt import c, h, nJy, erg, s, Hz, pc, angstrom, eV, unyt_array, cm

from synthesizer import exceptions
from synthesizer.dust.attenuation import PowerLaw
from synthesizer.utils import rebin_1d
from synthesizer.units import Quantity
from synthesizer.igm import Inoue14


class Sed:

    """
    A class representing a spectral energy distribution (SED).

    Attributes:
        lam (Quantity, array-like, float)
            The rest frame wavelength array.
        nu (Quantity, array-like, float)
            The rest frame frequency array.
        lnu (Quantity, array-like, float)
            The spectral luminosity density.
        fnu (Quantity, array-like, float)
            The spectral flux density.
        obslam (Quantity, array-like, float)
            The observed wavelength array.
        obsnu (Quantity, array-like, float)
            The observed frequency array.
        description (string)
            An optional descriptive string defining the Sed.
        redshift (float)
            The redshift of the Sed.
        broadband_luminosities (dict, float)
            The rest frame broadband photometry in arbitrary filters
            (filter_code: photometry).
        broadband_fluxes (dict, float)
            The observed broadband photometry in arbitrary filters
            (filter_code: photometry).
    """

    # Define Quantities, for details see units.py
    lam = Quantity()
    nu = Quantity()
    lnu = Quantity()
    fnu = Quantity()
    obsnu = Quantity()
    obslam = Quantity()
    luminosity = Quantity()
    llam = Quantity()

    def __init__(self, lam, lnu=None, description=None):
        """
        Initialise a new spectral energy distribution object.

        Args:
            lam (array-like, float)
                The rest frame wavelength array.
            lnu (array-like, float)
                The spectral luminosity density.
            description (string)
                An optional descriptive string defining the Sed.
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

            # Get the other lnu array
            other_lnu = other_sed._lnu

            # If the the number of dimensions differ between the lnu arrays we
            # need to promote the smaller one
            if new_lnu.ndim < other_lnu.ndim:
                new_lnu = np.array((new_lnu,))
            elif new_lnu.ndim > other_lnu.ndim:
                other_lnu = np.array((other_lnu,))

            # Concatenate this lnu array
            new_lnu = np.concatenate((new_lnu, other_lnu))

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

    def __mul__(self, scaling):
        """
        Overide multiplication operator to allow lnu to be scaled.
        This only works scaling * x.

        Note: only acts on the rest frame spectra. To get the scaled fnu get_fnu
        must be called on the newly scaled Sed object.

        Args:
            scaling (float)
                The scaling to apply to lnu.

        Returns:
            Sed
                A new instance of Sed with scaled lnu.
        """

        return Sed(self._lam, lnu=scaling * self._lnu)

    def __rmul__(self, scaling):
        """
        As above but for x * scaling.

        Note: only acts on the rest frame spectra. To get the scaled fnu get_fnu
        must be called on the newly scaled Sed object.

        Args:
            scaling (float)
                The scaling to apply to lnu.

        Returns:
            Sed
                A new instance of Sed with scaled lnu.
        """

        return Sed(self._lam, lnu=scaling * self._lnu)

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
        pstr += f"log10(Peak luminosity/{self.lnu.units}): \
            {np.log10(np.max(self.lnu)):.2f} \n"
        bolometric_luminosity = self.measure_bolometric_luminosity()
        pstr += f"log10(Bolometric luminosity/{bolometric_luminosity.units}): \
            {np.log10(bolometric_luminosity):.2f} \n"
        pstr += "-" * 10

        return pstr


    @property
    def luminosity(self):
        """
        Get the spectra in terms of luminosity.

        Returns
            luminosity (unyt_array)
                The luminosity array.
        """
        return self.lnu * self.nu

    @property
    def llam(self):
        """
        Get the spectral luminosity density per Angstrom.

        Returns
            luminosity (unyt_array)
                The spectral luminosity density per Angstrom array.
        """
        return self.nu * self.lnu / self.lam

    @property
    def luminosity_nu(self):
        """
        Alias to lnu.

        Returns
            luminosity (unyt_array)
                The spectral luminosity density per Hz array.
        """
        return self.lnu 
    
    @property
    def luminosity_lambda(self):
        """
        Alias to llam.

        Returns
            luminosity (unyt_array)
                The spectral luminosity density per Angstrom array.
        """
        return self.llam

    @property
    def _spec_dims(self):
        """
        Get the dimensions of the spectra array.

        Returns
            Tuple
                The shape of self.lnu
        """
        return np.ndim(self.lnu)

    def _get_lnu_at_nu(self, nu, kind=False):
        """
        A simple internal function for getting lnu at nu assuming the default
        unit system.

        Args:
            nu (float/array-like, float)
                The frequency(s) of interest.
            kind (str)
                Interpolation kind, see scipy.interp1d docs.

        Returns:
            luminosity (float/array-like, float)
                The luminosity (lnu) at the provided wavelength.
        """

        return interp1d(self._nu, self._lnu, kind=kind)(nu)

    def get_lnu_at_nu(self, nu, kind=False):
        """
        Return lnu with units at a provided frequency using 1d interpolation.

        Args:
            wavelength (float/array-like, float)
                The wavelength(s) of interest.
            kind (str)
                Interpolation kind, see scipy.interp1d docs.

        Returns:
            luminosity (unyt_array)
                The luminosity (lnu) at the provided wavelength.
        """

        return (
            self._get_lnu_at_nu(nu.to(self.nu.units).value, kind=kind) * self.lnu.units
        )

    def _get_lnu_at_lam(self, lam, kind=False):
        """
        Return lnu without units at a provided wavelength using 1d
        interpolation.

        Args:
            lam (float/array-like, float)
                The wavelength(s) of interest.
            kind (str)
                Interpolation kind, see scipy.interp1d docs.

        Returns:
            luminosity (float/array-like, float)
                The luminosity (lnu) at the provided wavelength.
        """

        return interp1d(self._lam, self._lnu, kind=kind)(lam)

    def get_lnu_at_lam(self, lam, kind=False):
        """
        Return lnu at a provided wavelength.

        Args:
            lam (float/array-like, float)
                The wavelength(s) of interest.
            kind (str)
                Interpolation kind, see scipy.interp1d docs.

        Returns:
            luminosity (unyt-array)
                The luminosity (lnu) at the provided wavelength.
        """

        return (
            self._get_lnu_at_lam(lam.to(self.lam.units).value, kind=kind)
            * self.lnu.units
        )

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

        Raises:
            UnrecognisedOption
                If method is an incompatible option an error is raised.
        """

        # Integrate using the requested method
        if method == "trapz":
            bolometric_luminosity = np.trapz(self.lnu[::-1], x=self.nu[::-1])
        elif method == "quad":
            bolometric_luminosity = (
                integrate.quad(self._get_lnu_at_nu, 1e12, 1e16)[0] * self.lnu.units * Hz
            )
        else:
            raise exceptions.UnrecognisedOption(
                f"Unrecognised integration method ({method}). "
                "Options are 'trapz' or 'quad'"
            )

        return bolometric_luminosity

    def measure_window_luminosity(self, window, method="trapz"):
        """
        Measure the luminosity in a spectral window.

        Args:
            window (tuple, float)
                The window in wavelength.
            method (str)
                The method used to calculate the bolometric luminosity. Options
                include 'trapz' and 'quad'.

        Returns:
            luminosity (float)
                The luminosity in the window.

        Raises:
            UnrecognisedOption
                If method is an incompatible option an error is raised.
        """

        # Integrate using the requested method
        if method == "quad":
            # Convert wavelength limits to frequency limits and convert to
            # base units.
            lims = (c / np.array(window)).to(self.nu.units).value
            luminosity = (
                integrate.quad(self._get_lnu_at_nu, *lims)[0] * self.lnu.units * Hz
            )

        elif method == "trapz":
            # Define a pseudo transmission function
            transmission = (self.lam > window[0]) & (self.lam < window[1])
            transmission = transmission.astype(float)
            luminosity = np.trapz(self.lnu[::-1] * transmission[::-1], x=self.nu[::-1])
        else:
            raise exceptions.UnrecognisedOption(
                f"Unrecognised integration method ({method}). "
                "Options are 'trapz' or 'quad'"
            )

        return luminosity.to(self.lnu.units * Hz)

    def measure_window_lnu(self, window, method="trapz"):
        """
        Measure lnu in a spectral window.

        Args:
            window (tuple, float)
                The window in wavelength.
            method (str)
                The method to use for the integration. Options include
                'average', 'trapz', and 'quad'.

        Returns:
            luminosity (float)
                The luminosity in the window.

         Raises:
            UnrecognisedOption
                If method is an incompatible option an error is raised.
        """

        # Apply the correct method
        if method == "average":
            # Define a pseudo transmission function
            transmission = (self.lam > window[0]) & (self.lam < window[1])
            transmission = transmission.astype(float)

            # Apply to the correct axis of the spectra
            if self._spec_dims == 2:
                Lnu = (
                    np.array(
                        [
                            np.sum(_lnu * transmission) / np.sum(transmission)
                            for _lnu in self._lnu
                        ]
                    )
                    * self.lnu.units
                )

            else:
                Lnu = np.sum(self.lnu * transmission) / np.sum(transmission)

        elif method == "trapz":
            # Define a pseudo transmission function
            transmission = (self.lam > window[0]) & (self.lam < window[1])
            transmission = transmission.astype(float)

            # Reverse the frequencies
            nu = self.nu[::-1]

            # Apply to the correct axis of the spectra
            if self._spec_dims == 2:
                Lnu = (
                    np.array(
                        [
                            np.trapz(_lnu[::-1] * transmission[::-1] / nu, x=nu)
                            / np.trapz(transmission[::-1] / nu, x=nu)
                            for _lnu in self._lnu
                        ]
                    )
                    * self.lnu.units
                )

            else:
                lnu = self.lnu[::-1]
                Lnu = np.trapz(lnu * transmission[::-1] / nu, x=nu) / np.trapz(
                    transmission[::-1] / nu, x=nu
                )

        # note: not yet adapted for multiple SEDs
        elif method == "quad":
            # define limits in base units
            lims = (c / window).to(self.nu.units).value

            def func(x):
                return self._get_lnu_at_nu(x) / x

            def inv(x):
                return 1 / x

            Lnu = integrate.quad(func, *lims)[0] / integrate.quad(inv, *lims)[0]

            Lnu = Lnu * self.lnu.units

        else:
            raise exceptions.UnrecognisedOption(
                f"Unrecognised integration method ({method}). "
                "Options are 'average', 'trapz' or 'quad'"
            )

        return Lnu.to(self.lnu.units)

    def measure_break(self, blue, red):
        """
        Measure a spectral break (e.g. the Balmer break) or D4000 using two
        windows.

        Args:
            blue (tuple, float)
                The wavelength limits of the blue window.
            red (tuple, float)
                The wavelength limits of the red window.

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

        blue = (3400, 3600) * angstrom
        red = (4150, 4250) * angstrom

        return self.measure_break(blue, red)

    def measure_d4000(self, definition="Bruzual83"):
        """
        Measure the D4000 index using either the Bruzual83 or Balogh
        definitions.

        Args:
            definition
                The choice of definition: 'Bruzual83' or 'Balogh'.

        Returns:
            float
                The Balmer break strength.

         Raises:
            UnrecognisedOption
                If definition is an incompatible option an error is raised.
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

        return self.measure_break(blue, red)

    def measure_beta(self, window=(1250.0, 3000.0)):
        """
        Measure the UV continuum slope ($\beta$) measured using the provided
        window. If the window has len(2) a full fit to the spectra is performed
        else the luminosity in two windows is calculated and used to determine
        the slope, similar to observations.

        Args:
            window (tuple, float)
                The window in which to measure in terms of wavelength.

        Returns:
            float
                The UV continuum slope ($\beta$)
        """

        # If a single window is provided
        if len(window) == 2:
            s = (self.lam > window[0]) & (self.lam < window[1])

            # Handle different spectra dimensions
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

        # If two windows are provided
        elif len(window) == 4:
            # Define the red and blue windows
            blue = window[:2]
            red = window[2:]

            # Measure the red and blue windows
            lnu_blue = self.measure_window_lnu(blue)
            lnu_red = self.measure_window_lnu(red)

            # Measure beta
            beta = (
                np.log10(lnu_blue / lnu_red) / np.log10(np.mean(blue) / np.mean(red))
                - 2.0
            )

        else:
            raise exceptions.InconsistentArguments(
                "A window of len 2 or 4 must be provided"
            )

        return beta

    def get_broadband_luminosities(self, filters):
        """
        Calculate broadband luminosities using a FilterCollection object

        Args:
            filters (filters.FilterCollection)
                A FilterCollection object.

        Returns:
            broadband_luminosities (dict)
                A dictionary of rest frame broadband luminosities.
        """

        # Intialise result dictionary
        self.broadband_luminosities = {}

        # Loop over filters
        for f in filters:
            # Apply the filter transmission curve and store the resulting
            # luminosity
            bb_lum = f.apply_filter(self._lnu, nu=self._nu) * self.lnu.units
            self.broadband_luminosities[f.filter_code] = bb_lum

        return self.broadband_luminosities

    def get_fnu0(self):
        """
        Calculate a dummy observed frame spectral energy distribution.
        Useful when you want rest-frame quantities.

        Uses a standard distance of 10 pcs.

        Returns:
            fnu (ndarray)
                Spectral flux density calcualted at d=10 pc.
        """

        # Get the observed wavelength and frequency arrays
        self.obslam = self._lam
        self.obsnu = self._nu

        # Compute the flux SED and apply unit conversions to get to nJy
        self.fnu = self.lnu / (4 * np.pi * (10 * pc) ** 2)

        return self.fnu

    def get_fnu(self, cosmo, z, igm=Inoue14):
        """
        Calculate the observed frame spectral energy distribution.

        Args:
            cosmo (astropy.cosmology)
                astropy cosmology instance.
            z (float)
                The redshift of the spectra.
            igm (igm)
                The IGM class.

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
        luminosity_distance = cosmo.luminosity_distance(z).to("cm").value * cm

        # Finally, compute the flux SED and apply unit conversions to get
        # to nJy
        self.fnu = self.lnu * (1.0 + z) / (4 * np.pi * luminosity_distance**2)

        # If we are applying an IGM model apply it
        if igm:
            self._fnu *= igm().T(z, self._obslam)

        return self.fnu

    def get_broadband_fluxes(self, fc, verbose=True):  # broad band flux/nJy
        """
        Calculate broadband luminosities using a FilterCollection object

        Args:
            fc (object)
                A FilterCollection object.

        Returns:
            (dict)
                A dictionary of fluxes in each filter in fc.
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
        self.broadband_fluxes = {}

        # Loop over filters in filter collection
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
                The blue filter code.
            f2 (str)
                The red filter code.

        Returns:
            (float)
                The broadband colour.
        """

        # Ensure fluxes exist
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

        # Measure the red and blue windows
        lnu_blue = self.measure_window_lnu(blue)
        lnu_red = self.measure_window_lnu(red)

        # Define the wavelength grid over the feature
        transmission = (self.lam > feature[0]) & (self.lam < feature[1])
        feature_lam = self.lam[transmission]

        # Using the red and blue windows fit the continuum
        # note, this does not conserve units so we need to add them back in
        # later.

        # Handle different spectra shapes
        if self._spec_dims == 2:
            # Multiple spectra case

            # Set up output array
            index = np.zeros(len(self.lnu)) * self.lam.units

            # Note: I'm sure this could be done better.
            for i, _lnu in enumerate(self.lnu):
                continuum_fit = np.polyfit(
                    [np.mean(blue), np.mean(red)], [lnu_blue[i], lnu_red[i]], 1
                )

                # Use the continuum fit to define the continuum
                continuum = (
                    (continuum_fit[0] * feature_lam.to(self.lam.units).value)
                    + continuum_fit[1]
                ) * self.lnu.units

                # Define the continuum subtracted spectrum
                feature_lum = _lnu[transmission]
                feature_lum_continuum_subtracted = (
                    -(feature_lum - continuum) / continuum
                )

                # Measure index
                index[i] = np.trapz(feature_lum_continuum_subtracted, x=feature_lam)

        else:
            # Single spectra case

            continuum_fit = np.polyfit(
                [np.mean(blue), np.mean(red)], [lnu_blue, lnu_red], 1
            )

            # Use the continuum fit to define the continuum
            continuum = (
                (continuum_fit[0] * feature_lam.to(self.lam.units).value)
                + continuum_fit[1]
            ) * self.lnu.units

            # Define the continuum subtracted spectrum
            feature_lum = self.lnu[transmission]

            feature_lum_continuum_subtracted = -(feature_lum - continuum) / continuum

            # Measure index
            index = np.trapz(feature_lum_continuum_subtracted, x=feature_lam)

        return index

    def get_resampled_sed(self, n):
        """
        Resameple the spectra and create a new Sed object.

        NOTE: This only resamples the rest frame spectra. For fluxes get_fnu
        must be called again after resampling.

        Args:
            n (int)
                The number of wavelength elements to resample to.

        Returns:
            (sed.Sed)
                A new Sed with the rebinned rest frame spectra.

        Raises:
            InconsistentArgument
                The sampling factor must be an integer.
        """

        # Resample the Sed
        if isinstance(n, int):
            sed = Sed(
                rebin_1d(self.lam, n, func=np.mean), rebin_1d(self.lnu, n, func=np.mean)
            )

        else:
            # Handle improper resampling factors
            raise exceptions.InconsistentArguments(
                "Sampling factor must be integer or float."
            )

        return sed

    def apply_attenuation(
        self,
        tau_v,
        dust_curve=PowerLaw(slope=-1.0),
        mask=None,
    ):
        """
        Apply attenuation to spectra.

        Args:
            tau_v (float/array-like, float)
                The V-band optical depth for every star particle.
            dust_curve (synthesizer.dust.attenuation.*)
                An instance of one of the dust attenuation models. (defined in
                synthesizer/dust/attenuation.py)
            mask (array-like, bool)
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
                    "Masks are only applicable for Seds containing multiple spectra"
                )
            if self._lnu.shape[0] != mask.size:
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
            if self._lnu.shape[0] != self.tau_v.size:
                raise exceptions.InconsistentArguments(
                    "tau_v and spectra are incompatible shapes "
                    f"({tau_v.shape}, {self._lnu.shape})"
                )

        # Compute the transmission
        transmission = dust_curve.get_transmission(tau_v, self._lam)

        # Get a copy of the rest frame spectra, we need to avoid
        # modifying the original
        spectra = np.copy(self._lnu)

        # Apply the transmission curve to the rest frame spectra with or
        # without applying a mask
        if mask is None:
            spectra *= transmission
        else:
            spectra[mask] *= transmission

        return Sed(self.lam, spectra)


def calculate_Q(lam, lnu, ionisation_energy=13.6 * eV, limit=100):
    """
    A function to calculate the ionising production rate directly from
    spectra.

    Args:
        lam (array-like, float)
            The wavelength array.
        lnu (array-like, float)
            The luminosity grid (erg/s/Hz).
        ionisation_energy (unyt_array)
            The ionisation energy.
        limit (float/int)
            An upper bound on the number of subintervals
            used in the integration adaptive algorithm.

    Returns
        float
            Ionising photon luminosity (s^-1).
    """

    # Apply units if not present
    if not isinstance(lam, unyt_array):
        lam = lam * angstrom
    if not isinstance(lnu, unyt_array):
        lnu = lnu * erg / s / Hz

    # Convert lnu to llam
    llam = lnu * c / lam**2

    # Caculate ionisation wavelength
    ionisation_wavelength = h * c / ionisation_energy

    # Defintion integration arrays
    x = lam.to(angstrom).value
    y = (llam * lam).to(erg / s).value / (
        h.to(erg / Hz).value * c.to(angstrom / s).value
    )

    return integrate.quad(
        lambda x_: np.interp(x_, x, y),
        0,
        ionisation_wavelength.to(angstrom).value,
        limit=limit,
    )[0]
