"""A module holding all photometric transmission filter functionality.

There are two main types of filter object in Synthesizer. Indivdual filters
described by a Filter object and Filters grouped into a FilterCollection.
These objects house all the functionality for working with filters with and
without a grid object.

Typical usage examples where trans is a transmission curve array, lams is a
wavelength array, fs is a list of SVO database filter codes, tophats is a
dictionary defining top hot filters and generics is a dictionary of
transmission curves:

    filt = Filter("generic/filter.1", transmission=trans, new_lam=lams)
    filt = Filter("top_hat/filter.1", lam_min=3000, lam_max=5500)
    filt = Filter("top_hat/filter.2", lam_eff=7000, lam_fwhm=2000)
    filt = Filter("JWST/NIRCam.F200W", new_lam=lams)
    filters = FilterCollection(
        filter_codes=fs,
        tophat_dict=tophats,
        generic_dict=generics,
        new_lam=lams
    )

"""

import urllib.request
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from urllib.error import URLError

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import integrate
from unyt import Angstrom, Hz, c, unyt_array, unyt_quantity

from synthesizer._version import __version__
from synthesizer.exceptions import (
    InconsistentAddition,
    InconsistentArguments,
    InconsistentWavelengths,
    SVOFilterNotFound,
    SVOInaccessible,
    WavelengthOutOfRange,
)
from synthesizer.units import Quantity

# To avoid circular imports while having the classes available for type
# checking we need to hide classes the reference each other within the
# TYPE_CHECKING flag which is set by mypy
if TYPE_CHECKING:
    from synthesizer.grid import Grid


class Filter:
    """
    A class holding a filter's transmission curve and wavelength array.

    A filter can either be retrieved from the
    `SVO database <http://svo2.cab.inta-csic.es/svo/theory/fps3/>`__,
    made from specific top hat filter properties, or made from a generic
    filter transmission curve and wavelength array.

    Also contains methods for calculating basic filter properties taken from
    `here <http://stsdas.stsci.edu/stsci_python_epydoc/SynphotManual.pdf>`__
    (page 42 (5.1))

    Attributes:
        filter_code: The full name defining this Filter.
        observatory: The name of the observatory
        instrument: The name of the instrument.
        filter_: The name of the filter.
        filter_type: A string describing the filter type: "SVO", "TopHat", or
                     "Generic".
        lam_min: If a top hat filter: The minimum wavelength where transmission
                 is nonzero.
        lam_max: If a top hat filter: The maximum wavelength where transmission
                 is nonzero.
        lam_eff: If a top hat filter: The effective wavelength of the filter
                 curve.
        lam_fwhm: If a top hat filter: The FWHM of the filter curve.
        svo_url: If an SVO filter: the url from which the data was extracted.
        t: The transmission curve.
        lam: The wavelength array for which the transmission is defined.
        nu: The frequency array for which the transmission is defined. Derived
            from self.lam.
        original_lam: The original wavelength extracted from SVO. In a non-SVO
                      filter self.original_lam == self.lam.
        original_nu: The original frequency derived from self.original_lam. In
                     a non-SVO filter self.original_nu == self.nu.
        original_t: The original transmission extracted from SVO. In a non-SVO
                    filter self.original_t == self.t.
    """

    filter_code: str
    observatory: Optional[str]
    instrument: Optional[str]
    filter_: Optional[str]
    filter_type: Optional[str]
    svo_url: Optional[str]
    t: NDArray[np.float64]
    _shifted_t: Optional[NDArray[np.float64]]
    original_t: NDArray[np.float64]

    # Define Quantitys
    _lam_min: np.float64
    lam_min: unyt_quantity
    lam_min = Quantity()
    _lam_max: np.float64
    lam_max: unyt_quantity
    lam_max = Quantity()
    _lam_eff: np.float64
    lam_eff: unyt_quantity
    lam_eff = Quantity()
    _lam_fwhm: np.float64
    lam_fwhm: unyt_quantity
    lam_fwhm = Quantity()
    _lam: NDArray[np.float64]
    lam: unyt_array
    lam = Quantity()
    _nu: NDArray[np.float64]
    nu: unyt_array
    nu = Quantity()
    _original_lam: NDArray[np.float64]
    original_lam: unyt_array
    original_lam = Quantity()
    _original_nu: NDArray[np.float64]
    original_nu: unyt_array
    original_nu = Quantity()

    def __init__(
        self,
        filter_code: str,
        transmission: Optional[NDArray[np.float64]] = None,
        lam_min: Optional[unyt_quantity] = None,
        lam_max: Optional[unyt_quantity] = None,
        lam_eff: Optional[unyt_quantity] = None,
        lam_fwhm: Optional[unyt_quantity] = None,
        new_lam: Optional[unyt_array] = None,
    ) -> None:
        """
        Initialise a filter.

        Args:
            filter_code: The full name defining this Filter.
            transmission: An array describing the filter's transmission curve.
                          Only used for generic filters.
            lam_min: If a top hat filter: The minimum wavelength where
                     transmission is nonzero.
            lam_max: If a top hat filter: The maximum wavelength where
                     transmission is nonzero.
            lam_eff: If a top hat filter: The effective wavelength of the
                     filter curve.
            lam_fwhm: If a top hat filter: The FWHM of the filter curve.
            new_lam: The wavelength array for which the transmission is
                     defined.
        """
        # Metadata of this filter
        self.filter_code = filter_code
        self.observatory = None
        self.instrument = None
        self.filter_ = None
        self.filter_type = None

        # Properties for a top hat filter
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.lam_eff = lam_eff
        self.lam_fwhm = lam_fwhm

        # Properties for a filter from SVO
        self.svo_url = None

        # Define transmission curve and wavelength of this filter (if these
        # are non, i.e we aren't working with a generic filter, we will
        # define these later in the private methods)
        self.t = transmission if transmission is not None else np.zeros(1)
        self.lam = new_lam
        self._shifted_t = None

        # Is this a generic filter? (Everything other than the label is defined
        # above.)
        if transmission is not None and new_lam is not None:
            self.filter_type = "Generic"

        # Is this a top hat filter?
        elif (lam_min is not None and lam_max is not None) or (
            lam_eff is not None and lam_fwhm is not None
        ):
            self._make_top_hat_filter()

        # Is this an SVO filter?
        elif "/" in filter_code and "." in filter_code:
            self._make_svo_filter()

        # Otherwise we haven't got a valid combination of inputs.
        else:
            raise InconsistentArguments(
                "Invalid combination of filter inputs. \n For a generic "
                "filter provide a transimision and wavelength array. \nFor "
                "a filter from the SVO database provide a filter code of the "
                "form Observatory/Instrument.Filter that matches the database."
                " \nFor a top hat provide either a minimum and maximum "
                "wavelength or an effective wavelength and FWHM."
            )

        # Define the original wavelength and transmission for property
        # calculation later. (Making sure we have copies)
        if self.lam is not None and self.t is not None:
            self.original_lam = self.lam.copy()
            self.original_t = self.t.copy()
        else:
            raise ValueError(
                "We shouldn't ever get to this point without having set"
                "self.lam and self.t"
            )

        # Calculate frequencies
        self.nu = (c / self.lam).to("Hz").value
        self.original_nu = (c / self.original_lam).to("Hz").value

        # Ensure transmission curves are in a valid range (we expect 0-1,
        # some SVO curves return strange values above this e.g. ~60-80)
        self.clip_transmission()

    @property
    def transmission(self) -> NDArray[np.float64]:
        """Alias for self.t."""
        return self.t

    def clip_transmission(self) -> None:
        """
        Clips transmission curve between 0 and 1.

        Some transmission curves from SVO can come with strange
        upper limits, the way we use them requires the maxiumum of a
        transmission curve is at most 1. So for one final check lets
        clip the transmission curve between 0 and 1
        """
        # Warn the user we are are doing this
        if self.t.max() > 1 or self.t.min() < 0:
            print(
                "Warning: Out of range transmission values found "
                f"(min={self.t.min()}, max={self.t.max()}). "
                "Transmission will be clipped to [0-1]"
            )
            self.t = np.clip(self.t, 0, 1)

    def _make_top_hat_filter(self) -> None:
        """Make a top hat filter from the Filter's attributes."""
        # Define the type of this filter
        self.filter_type = "TopHat"

        # If filter has been defined with an effective wavelength and FWHM
        # calculate the minimum and maximum wavelength.
        if self.lam_eff is not None and self.lam_fwhm is not None:
            self.lam_min = self.lam_eff - self.lam_fwhm / 2.0
            self.lam_max = self.lam_eff + self.lam_fwhm / 2.0

        # Otherwise, use the explict min and max

        # Define this top hat filters wavelength array (+/- 1000 Angstrom)
        # if it hasn't been provided
        if self.lam is None:
            self.lam = np.arange(
                np.max([0, self._lam_min - 1000]), self._lam_max + 1000, 1
            )

        # Define the transmission curve (1 inside, 0 outside)
        self.t = np.zeros(len(self.lam))
        s = (self.lam > self.lam_min) & (self.lam <= self.lam_max)
        self.t[s] = 1.0

    def _make_svo_filter(self) -> None:
        """
        Retrieve a filter's data from the SVO database.

        Raises:
            SVOFilterNotFound
                If a filter code cannot be matched to a database entry or a
                connection cannot be made to the database and error is thrown.
        """
        # Define the type of this filter
        self.filter_type = "SVO"

        # Get the information stored in the filter code
        self.observatory = self.filter_code.split("/")[0]
        self.instrument = self.filter_code.split("/")[1].split(".")[0]
        self.filter_ = self.filter_code.split(".")[-1]

        # Read directly from the SVO archive.
        self.svo_url = (
            f"http://svo2.cab.inta-csic.es/theory/"
            f"fps/getdata.php?format=ascii&id={self.observatory}"
            f"/{self.instrument}.{self.filter_}"
        )

        # Make a request for the data and handle a failure more informatively
        try:
            with urllib.request.urlopen(self.svo_url) as f:
                df: NDArray[np.float64] = np.loadtxt(f)
        except URLError:
            raise SVOInaccessible(
                (
                    f"The SVO Database at {self.svo_url} "
                    "is not responding. Is it down?"
                )
            )

        # Throw an error if we didn't find the filter.
        if df.size == 0:
            raise SVOFilterNotFound(
                f"Filter ({self.filter_code}) not in the database. "
                "Double check the database: http://svo2.cab.inta-csic.es/"
                "svo/theory/fps3/. This could also mean you have no"
                " connection."
            )

        # Extract the wavelength and transmission given by SVO
        self.original_lam = df[:, 0]
        self.original_t = df[:, 1]

        # If a new wavelength grid is provided, interpolate
        # the transmission curve on to that grid
        if isinstance(self._lam, np.ndarray):
            self._interpolate_wavelength()
        else:
            self.lam = self.original_lam
            self.t = self.original_t

    def _interpolate_wavelength(
        self,
        new_lam: Union[NDArray[np.float64], unyt_array] = None,
    ) -> None:
        """
        Interpolate the transmission curve onto the a new wavelength array.

        Args:
            new_lam: The wavelength array to interpolate onto. If None self.lam
                     is used.

        Returns:
            Transmission curve interpolated onto the new wavelength array.
        """
        # If we've been handed a wavelength array we must overwrite the
        # current one
        if new_lam is not None:
            # Warn the user if we're about to truncate the existing wavelength
            # array
            truncated: bool = False
            if new_lam.min() > self.lam[self.t > 0].min():
                truncated = True
            if new_lam.max() < self.lam[self.t > 0].max():
                truncated = True
            if truncated:
                print(
                    f"Warning: {self.filter_code} will be truncated where "
                    "transmission is non-zero "
                    "(old_lam_bounds = "
                    f"({self.lam[self.t > 0].min():.2e}, "
                    f"{self.lam[self.t > 0].max():.2e}), "
                    "new_lam_bounds = "
                    f"({new_lam.min():.2e}, {new_lam.max():.2e}))"
                )

            self.lam = new_lam

        # Perform interpolation
        self.t = np.interp(
            self._lam, self._original_lam, self.original_t, left=0.0, right=0.0
        )

        # Ensure we don't have 0 transmission
        if self.t.sum() == 0:
            raise InconsistentWavelengths(
                "Interpolated transmission curve has no non-zero values. "
                "Consider removing this filter or extending the wavelength "
                "range."
            )

        # And ensure transmission is in expected range
        self.clip_transmission()

    def apply_filter(
        self,
        arr: Union[unyt_array, NDArray[np.float64]],
        lam: Optional[Union[unyt_array, NDArray[np.float64]]] = None,
        nu: Optional[Union[unyt_array, NDArray[np.float64]]] = None,
        verbose: bool = True,
    ) -> np.float64:
        """
        Apply the transmission curve to an arbitrary dimensioned array.

        This will return the sum of the array convolved with the filter
        transmission curve along the wavelength axis (final axis).

        If no wavelength or frequency array is provided then the filters rest
        frame frequency is assumed.

        Args:
            arr: The array to convolve with the filter's transmission curve.
                 Can be any dimension but wavelength must be the final axis.
            lam: The wavelength array to integrate with respect to.
                 Defaults to the rest frame frequency if neither lams or nus
                 are provided.
            nu: The frequency array to integrate with respect to.
                Defaults to the rest frame frequency if neither lams or nus are
                provided.
            verbose: Are we talking?

        Returns:
            The array (arr) convolved with the transmission curve
            and summed along the wavelength axis.

        Raises:
            ValueError
                If the shape of the transmission and wavelength array differ
                the convolution cannot be done.
        """
        # Warn the user that frequencies take precedence over wavelengths
        # if both are provided
        if lam is not None and nu is not None:
            if verbose:
                print(
                    (
                        "WARNING: Both wavelengths and frequencies were "
                        "provided, frequencies take priority over wavelengths"
                        " for filter convolution."
                    )
                )

        # Get the correct x array to integrate w.r.t and work out if we need
        # to shift the transmission curve.
        xs: unyt_array
        need_shift: bool
        lam_arr: NDArray[np.float64]
        if nu is not None:
            # Ensure the passed frequencies have units
            if not isinstance(nu, unyt_array):
                nu *= Hz

            # Define the integration xs
            xs = nu

            # Do we need to shift?
            need_shift = not nu.value[0] == self._nu[0]

            # To shift the transmission we need the corresponding wavelength
            # with the units stripped off
            if need_shift:
                lam_arr = (c / nu).to(Angstrom).value

        elif lam is not None:
            # Ensure the passed wavelengths have no units
            if isinstance(lam, unyt_array):
                lam_arr = lam.value

            # Define the integration xs
            xs = lam_arr

            # Do we need to shift?
            need_shift = not lam[0] == self._lam[0]

        else:
            # Define the integration xs
            xs = self.nu

            # No shift needed
            need_shift = False

        # Do we need to shift?
        t: NDArray[np.float64]
        if need_shift:
            # Ok, shift the tranmission curve by interpolating onto the
            # provided wavelengths
            t = np.interp(
                lam_arr,
                self._original_lam,
                self.original_t,
                left=0.0,
                right=0.0,
            )

        else:
            # We can use the standard transmission array
            t = self.t

        # Check dimensions are ok
        if xs.size != arr.shape[-1]:
            raise ValueError(
                "Final dimension of array did not match "
                "x array shape (arr.shape[-1]=%d, "
                "xs.size=%d)" % (arr.shape[-1], xs.size)
            )

        # Store this observed frame transmission
        self._shifted_t = t

        # Get the mask that removes wavelengths we don't currently care about
        in_band: NDArray[np.bool_] = t > 0

        # Mask out wavelengths that don't contribute to this band
        arr_in_band: NDArray[np.float64] = arr.compress(in_band, axis=-1)
        xs_in_band: NDArray[np.float64] = xs[in_band]
        t_in_band: NDArray[np.float64] = t[in_band]

        # Multiply the IFU by the filter transmission curve
        transmission: NDArray[np.float64] = arr_in_band * t_in_band

        # Sum over the final axis to "collect" transmission in this filer
        sum_per_x: np.float64 = integrate.trapezoid(
            transmission / xs_in_band, xs_in_band, axis=-1
        )
        sum_den: np.float64 = integrate.trapezoid(
            t_in_band / xs_in_band, xs_in_band, axis=-1
        )
        sum_in_band: np.float64 = sum_per_x / sum_den

        return sum_in_band

    def pivwv(self) -> unyt_quantity:
        """
        Calculate the pivot wavelength.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            Pivot wavelength.
        """
        return (
            np.sqrt(
                np.trapz(
                    self._original_lam * self.original_t, x=self._original_lam
                )
                / np.trapz(
                    self.original_t / self._original_lam, x=self._original_lam
                )
            )
            * self.original_lam.units
        )

    def pivT(self) -> np.float64:
        """
        Calculate the transmission at the pivot wavelength.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            Transmission at pivot wavelength.
        """
        return np.interp(
            self.pivwv().value, self._original_lam, self.original_t
        )[0]

    def meanwv(self) -> unyt_quantity:
        """
        Calculate the mean wavelength.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            Mean wavelength.
        """
        return (
            np.exp(
                np.trapz(
                    np.log(self._original_lam)
                    * self.original_t
                    / self._original_lam,
                    x=self._original_lam,
                )
                / np.trapz(
                    self.original_t / self._original_lam, x=self._original_lam
                )
            )
            * self.original_lam.units
        )

    def bandw(self) -> unyt_quantity:
        """
        Calculate the bandwidth.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            The bandwidth.
        """
        # Calculate the left and right hand side.
        A = np.sqrt(
            np.trapz(
                (np.log(self._original_lam / self.meanwv().value) ** 2)
                * self.original_t
                / self._original_lam,
                x=self._original_lam,
            )
        )

        B = np.sqrt(
            np.trapz(
                self.original_t / self._original_lam, x=self._original_lam
            )
        )

        return self.meanwv() * (A / B)

    def fwhm(self) -> unyt_quantity:
        """
        Calculate the FWHM.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            The FWHM of the filter.
        """
        return np.sqrt(8.0 * np.log(2)) * self.bandw()

    def Tpeak(self) -> np.float64:
        """
        Calculate the peak transmission.

        For an SVO filter this uses the transmission from the database.

        Returns:
            The peak transmission.
        """
        return np.max(self.original_t)

    def rectw(self) -> unyt_quantity:
        """
        Calculate the rectangular width.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            The rectangular width.
        """
        return np.trapz(self.original_t, x=self._original_lam) / self.Tpeak()

    def max(self) -> unyt_quantity:
        """
        Calculate the longest wavelength where the transmission is >0.01.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            The maximum wavelength at which transmission is nonzero.
        """
        return self.original_lam[self.original_t > 1e-2][-1]

    def min(self) -> unyt_quantity:
        """
        Calculate the shortest wavelength where the transmission is >0.01.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            The minimum wavelength at which transmission is nonzero.
        """
        return self.original_lam[self.original_t > 1e-2][0]

    def mnmx(self) -> Tuple[unyt_quantity, unyt_quantity]:
        """
        Calculate the minimum and maximum wavelengths.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            The minimum wavelength.
            The maximum wavelength.
        """
        return (self.original_lam.min(), self.original_lam.max())


class FilterCollection:
    """
    A Collection of Filter objects.

    Holds a collection of filters (`Filter` objects) and enables various
    quality of life operations such as plotting, adding, looping, len,
    and comparisons as if the collection was a simple list.

    Filters can be derived from the
    `SVO database <http://svo2.cab.inta-csic.es/svo/theory/fps3/>`__
    , specific top hat filter
    properties or generic filter transmission curves and a wavelength array.

    All filters in the `FilterCollection` are defined in terms of the
    same wavelength array.

    In addition to creating `Filter`s from user defined arguments, a HDF5
    file of a `FilterCollection` can be created and later loaded at
    instantiation to load a saved `FilterCollection`.

    Attributes:
        filters: A list containing the individual `Filter` objects.
        filter_codes: A list of the names of each filter. For SVO filters
                      these have to have the form
                      "Observatory/Instrument.Filter" matching the database,
                      but for all other filter types this can be an arbitrary
                      label.
        lam: The wavelength array for which each filter's transmission curve is
            defined.
        nfilters: The number of filters in this collection.
        mean_lams: The mean wavelength of each Filter in the collection.
        pivot_lams: The mean wavelength of each Filter in the collection.
    """

    filters: Dict[str, Filter]
    filter_codes: List[str]
    nfilters: int

    # Define Quantitys
    _lam: NDArray[np.float64]
    lam: unyt_array
    lam = Quantity()
    _mean_lams: NDArray[np.float64]
    mean_lams: unyt_array
    mean_lams = Quantity()
    _pivot_lams: NDArray[np.float64]
    pivot_lams: unyt_array
    pivot_lams = Quantity()

    def __init__(
        self,
        filter_codes: Optional[List[str]] = None,
        tophat_dict: Optional[
            Dict[str, Dict[str, Union[float, unyt_quantity]]]
        ] = None,
        generic_dict: Optional[Dict[str, NDArray[np.float64]]] = None,
        filters: Optional[List[Filter]] = None,
        path: Optional[str] = None,
        new_lam: Optional[unyt_array] = None,
        fill_gaps: bool = True,
    ) -> None:
        """
        Intialise the FilterCollection.

        Args:
            filter_codes: A list of SVO filter codes, used to retrieve filter
                          data from the database or labels for each filter.
            tophat_dict: A dictionary containing the data to make a collection
                         of top hat filters from user defined properties.
                         The dictionary must have the form:

                    {<filter_code> : {"lam_eff": <effective_wavelength>,
                                      "lam_fwhm": <FWHM_of_filter>}, ...},

                or

                    {<filter_code> : {"lam_min": <minimum_nonzero_wavelength>,
                                      "lam_max": <maximum_nonzero_wavelength>},
                                      ...}.

            generic_dict: A dictionary containing the data to make a collection
                          of filters from user defined transmission curves. The
                          dictionary must have the form:

                    {<filter_code1> : {"transmission": <transmission_array>}}.

                          For generic filters new_lam must be provided.
            filters: A list of existing `Filter` objects to be added to the
                     collection.
            path: A filepath defining the HDF5 file from which to load the
                  FilterCollection.
            new_lam: The wavelength array to define the transmission curve on.
                     Can have units but Angstrom assumed.
            fill_gaps: Are we filling gaps in the wavelength array? Defaults to
                       True. This is only needed if new_lam has not been
                       passed. In that case the filters will be resampled onto
                       a universal wavelength grid and any gaps between filters
                       can be filled with the minimum average resolution of
                       all filters if fill_gaps is True.
                       NOTE: This will inflate the memory footprint of the
                       filter outside the region where transmission is
                       non-zero.
        """
        # Define lists to hold our filters and filter codes
        self.filters = {}
        self.filter_codes = []

        # Attribute for looping
        self._current_ind = 0

        # Ensure we haven't been passed both a path and parameters
        if path is not None:
            if filter_codes is not None:
                print(
                    "If a path is passed only the saved FilterCollection is "
                    "loaded! Create a separate FilterCollection with these "
                    "filter codes and add them."
                )
            if tophat_dict is not None:
                print(
                    "If a path is passed only the saved FilterCollection is "
                    "loaded! Create a separate FilterCollection with this "
                    "top hat dictionary and add them."
                )
            if generic_dict is not None:
                print(
                    "If a path is passed only the saved FilterCollection is "
                    "loaded! Create a separate FilterCollection with this "
                    "generic dictionary and add them."
                )

        # Are we loading an old filter collection?
        if path is not None:
            # Load the FilterCollection from the file
            self._load_filters(path)

        else:
            # Ok, we aren't loading one. Make the filters instead.

            # Do we have an wavelength array? If so we will resample the
            # transmissions.
            self.lam = new_lam

            # Let's make the filters
            if filter_codes is not None:
                self._include_svo_filters(filter_codes)
            if tophat_dict is not None:
                self._include_top_hat_filters(tophat_dict)
            if generic_dict is not None:
                self._include_generic_filters(generic_dict)
            if filters is not None:
                self._include_synthesizer_filters(filters)

            # How many filters are there?
            self.nfilters = len(self.filter_codes)

            # If we weren't passed a wavelength grid we need to resample the
            # filters onto a universal wavelength grid.
            if self.lam is None:
                self.resample_filters(fill_gaps=fill_gaps)

        # If we were passed a wavelength array we need to resample on to
        # it. NOTE: this can also be done for a loaded FilterCollection
        # so we just do it here outside the logic
        if new_lam is not None:
            self.resample_filters(new_lam)

        # Calculate mean and pivot wavelengths for each filter
        self.mean_lams = self.calc_mean_lams()
        self.pivot_lams = self.calc_pivot_lams()

    def _load_filters(self, path: str) -> None:
        """
        Load a `FilterCollection` from a HDF5 file.

        Args:
            path: The file path from which to load the `FilterCollection`.
        """
        # Open the HDF5 file
        hdf: h5py.File = h5py.File(path, "r")

        # Warn if the synthesizer versions don't match
        if hdf["Header"].attrs["synthesizer_version"] != __version__:
            print(
                "WARNING: Synthesizer versions differ between the code and "
                "FilterCollection file! This is probably fine but there "
                "is no gaurantee it won't cause errors."
            )

        # Get the wavelength units
        lam_units: str = hdf["Header"].attrs["Wavelength_units"]

        # Get the FilterCollection level attributes and datasets,
        # We apply the units to ensure conversions are done correctly
        # within the Quantity instantiation
        self.nfilters = hdf["Header"].attrs["nfilters"]
        self.lam = unyt_array(hdf["Header"]["Wavelengths"][:], lam_units)
        self.filter_codes = hdf["Header"].attrs["filter_codes"]

        # Loop over the groups and make the filters
        for filter_code in self.filter_codes:
            # Open the filter group
            f_grp: h5py.Group = hdf[filter_code.replace("/", ".")]

            # Get the filter type
            filter_type: str = f_grp.attrs["filter_type"]

            # For SVO filters we don't want to send a request to the
            # database so instead instatiate it as a generic filter and
            # overwrite some attributes after the fact
            filt: Filter
            if filter_type == "SVO":
                filt = Filter(
                    filter_code,
                    transmission=f_grp["Transmission"][:],
                    new_lam=self.lam,
                )

                # Set the SVO specific attributes
                filt.filter_type = filter_type
                filt.svo_url = f_grp.attrs["svo_url"]
                filt.observatory = f_grp.attrs["observatory"]
                filt.instrument = f_grp.attrs["instrument"]
                filt.filter_ = f_grp.attrs["filter_"]
                filt.original_lam = unyt_array(
                    f_grp["Original_Wavelength"][:], lam_units
                )
                filt.original_t = f_grp["Original_Transmission"][:]

            elif filter_type == "TopHat":
                # For a top hat filter we can pass the related parameters
                # and build the filter as normal

                # Set up key word params, we have to do this to handle to
                # two methods for creating top hat filters
                tophat_dict: Dict[str, Optional[unyt_quantity]] = {
                    key: None
                    for key in [
                        "lam_min",
                        "lam_max",
                        "lam_eff",
                        "lam_fwhm",
                    ]
                }

                # Loop over f_grp keys and set those that exist
                for key in f_grp.attrs.keys():
                    if "lam" in key:
                        tophat_dict[key] = unyt_quantity(
                            f_grp.attrs[key],
                            lam_units,
                        )

                # Create the filter
                filt = Filter(filter_code, **tophat_dict)

            else:
                # For a generic filter just set the transmission and
                # wavelengths
                filt = Filter(
                    filter_code,
                    transmission=f_grp["Transmission"][:],
                    new_lam=self.lam,
                )

            # Store the created filter
            self.filters[filter_code] = filt

        hdf.close()

    def _include_svo_filters(self, filter_codes: List[str]) -> None:
        """
        Populate the `FilterCollection` with filters from SVO.

        Args:
            filter_codes: A list of SVO filter codes, used to retrieve filter
                          data from the database.
        """
        # Loop over the given filter codes
        for f in filter_codes:
            # Get filter from SVO
            _filter: Filter = Filter(f, new_lam=self.lam)

            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def _include_top_hat_filters(
        self,
        tophat_dict: Dict[str, Dict[str, Union[float, unyt_quantity]]],
    ) -> None:
        """
        Populate the `FilterCollection` with user defined top-hat filters.

        Args:
            tophat_dict: A dictionary containing the data to make a collection
                         of top hat filters from user defined properties.
                         The dictionary must have the form:

                    {<filter_code> : {"lam_eff": <effective_wavelength>,
                                      "lam_fwhm": <FWHM_of_filter>}, ...},

                or

                    {<filter_code> : {"lam_min": <minimum_nonzero_wavelength>,
                                      "lam_max": <maximum_nonzero_wavelength>},
                                      ...}.
        """
        # Loop over the keys of the dictionary
        for key in tophat_dict:
            # Get this filter's properties
            if "lam_min" in tophat_dict[key]:
                lam_min = tophat_dict[key]["lam_min"]
            else:
                lam_min = None
            if "lam_max" in tophat_dict[key]:
                lam_max = tophat_dict[key]["lam_max"]
            else:
                lam_max = None
            if "lam_eff" in tophat_dict[key]:
                lam_eff = tophat_dict[key]["lam_eff"]
            else:
                lam_eff = None
            if "lam_fwhm" in tophat_dict[key]:
                lam_fwhm = tophat_dict[key]["lam_fwhm"]
            else:
                lam_fwhm = None

            # Instantiate the filter
            _filter: Filter = Filter(
                key,
                lam_min=lam_min,
                lam_max=lam_max,
                lam_eff=lam_eff,
                lam_fwhm=lam_fwhm,
                new_lam=self.lam,
            )

            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def _include_generic_filters(
        self,
        generic_dict: Dict[str, NDArray[np.float64]],
    ) -> None:
        """
        Populate the `FilterCollection` with user defined filters.

        Args:
            generic_dict: A dictionary containing the data to make a collection
                          of filters from user defined transmission curves. The
                          dictionary must have the form:

                    {<filter_code1> : {"transmission": <transmission_array>}}.
        """
        # Loop over the keys of the dictionary
        for key in generic_dict:
            # Get this filter's properties
            t: NDArray[np.float64] = generic_dict[key]

            # Instantiate the filter
            _filter: Filter = Filter(key, transmission=t, new_lam=self.lam)

            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def _include_synthesizer_filters(self, filters: List[Filter]) -> None:
        """
        Populate the `FilterCollection` with a list of `Filter` objects.

        Args:
            filters: A list of existing `Filter` objects to be added to the
                     FilterCollection.
        """
        # Loop over the given filter codes
        for _filter in filters:
            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def __add__(self, other_filters: "FilterCollection") -> "FilterCollection":
        """
        Addition operator.

        Enable the addition of FilterCollections and Filters with
        filtercollection1 + filtercollection2 or filtercollection + filter
        syntax.

        Returns:
            This filter collection containing the filter/filters from
            other_filters.
        """
        # Are we adding a collection or a single filter?
        if isinstance(other_filters, FilterCollection):
            # Loop over the filters in other_filters
            for key in other_filters.filters:
                # Store the filter and its code
                self.filters[key] = other_filters.filters[key]
                self.filter_codes.append(
                    other_filters.filters[key].filter_code
                )

        elif isinstance(other_filters, Filter):
            # Store the filter and its code
            self.filters[other_filters.filter_code] = other_filters
            self.filter_codes.append(other_filters.filter_code)

        else:
            raise InconsistentAddition(
                "Cannot add non-filter objects together!"
            )

        # Update the number of filters we have
        self.nfilters = len(self.filter_codes)

        # Now resample the filters onto the filter collection's wavelength
        # array,
        # NOTE: If the new filter extends beyond the filter collection's
        # wavlength array a warning is given and that filter curve will
        # truncated at the limits. This is because we can't have the
        # filter collection's wavelength array modified, if that were
        # to happen it could become inconsistent with Sed wavelength arrays
        # and photometry would be impossible.
        self.resample_filters(new_lam=self.lam)

        return self

    def __len__(self) -> int:
        """Overload the len operator to return how many filters there are."""
        return len(self.filters)

    def __iter__(self) -> "FilterCollection":
        """
        Allow iteration.

        Overload iteration to allow simple looping over filter objects,
        combined with __next__ this enables for f in FilterCollection syntax
        """
        return self

    def __next__(self) -> Filter:
        """
        Next in the iteration.

        Overload iteration to allow simple looping over filter objects,
        combined with __iter__ this enables for f in FilterCollection syntax
        """
        # Check we haven't finished
        if self._current_ind >= self.nfilters:
            self._current_ind = 0
            raise StopIteration
        else:
            # Increment index
            self._current_ind += 1

            # Return the filter
            return self.filters[self.filter_codes[self._current_ind - 1]]

    def __ne__(self, other_filters: object) -> bool:
        """
        Not equal comparison.

        Enables the != comparison of two filter collections. If the filter
        collections contain the same filter codes they are guaranteed to
        be identical.

        Args:
            other_filters: The other FilterCollection to be compared to self.

        Returns:
            True if the FilterCollections are not the same.
        """
        # Ensure we have been passed a FilterCollection
        if not isinstance(other_filters, FilterCollection):
            raise ValueError(
                "Cannot compare a FilterCollection to a non-FilterCollection!"
            )

        # Do they have the same number of filters?
        if self.nfilters != other_filters.nfilters:
            return True

        # Ok they do, so do they have the same filter codes? (elementwise test)
        not_equal: bool = False
        for n in range(self.nfilters):
            if self.filter_codes[n] != other_filters.filter_codes[n]:
                not_equal = True
                break

        return not_equal

    def __eq__(self, other_filters: object) -> bool:
        """
        Equal comparison.

        Enables the == comparison of two filter collections. If the filter
        collections contain the same filter codes they are guaranteed to
        be identical.

        Args:
            other_filters: The other FilterCollection to be compared to self.

        Returns:
            True/False Are the FilterCollections the same?
        """
        # Ensure we have been passed a FilterCollection
        if not isinstance(other_filters, FilterCollection):
            raise ValueError(
                "Cannot compare a FilterCollection to a non-FilterCollection!"
            )

        # Do they have the same number of filters?
        if self.nfilters != other_filters.nfilters:
            return False

        # Ok they do, so do they have the same filter codes? (elementwise test)
        equal: bool = True
        for n in range(self.nfilters):
            if self.filter_codes[n] != other_filters.filter_codes[n]:
                equal = False
                break

        return equal

    def __getitem__(self, key: str) -> Filter:
        """
        Enable dictionary key syntax.

        Enables the extraction of filter objects from the FilterCollection by
        getitem syntax (FilterCollection[key] rather than
        FilterCollection.filters[key]).

        Args:
            key: The filter code of the desired filter.

        Returns:
            The Filter object stored at self.filters[key].

        Raises:
            KeyError
                When the filter does not exist in self.filters an error is
                raised.
        """
        return self.filters[key]

    def get_non_zero_lam_lims(self) -> Tuple[unyt_quantity, unyt_quantity]:
        """
        Find the minimum and maximum wavelengths with non-zero transmission.

        Returns:
            Minimum wavelength with non-zero transmission.
            Maximum wavelength with non-zero transmission.
        """
        # Get the minimum and maximum wavelength at which transmission is
        # non-zero
        min_lam: float = np.inf
        max_lam: float = 0.0
        for f in self.filters:
            this_min: float = np.min(
                self.filters[f]._lam[self.filters[f].t > 0]
            )
            this_max: float = np.max(
                self.filters[f]._lam[self.filters[f].t > 0]
            )
            if this_min < min_lam:
                min_lam = this_min
            if this_max > max_lam:
                max_lam = this_max

        # It's possible to be here without having set self.lam, in that
        # case we use the last filter in the iteration.
        if self.lam is not None:
            return min_lam * self.lam.units, max_lam * self.lam.units
        return (
            min_lam * self.filters[f].lam.units,
            max_lam * self.filters[f].lam.units,
        )

    def _merge_filter_lams(self, fill_gaps: bool) -> NDArray[np.float64]:
        """
        Merge the wavelength arrays of multiple filters.

        Overlapping transmission adopt the values of one of the arrays.

        If a gap is found between filters it can be populated with the minimum
        average wavelength resolution of all filters if fill_gaps is True.

        Args:
            fill_gaps: Are we filling gaps in the wavelength array?

        Returns:
            The combined wavelength array with gaps filled and overlaps
            removed
        """
        # Get the indices sorted by pivot wavelength
        piv_lams: List[unyt_quantity] = [f.pivwv() for f in self]
        sinds: NDArray[np.int32] = np.argsort(piv_lams)

        # Get filter arrays in pivot wavelength order
        arrays: List[NDArray[np.float64]] = [
            self.filters[fc]._lam[self.filters[fc].t > 0]
            for fc in np.array(self.filter_codes)[sinds]
        ]

        # Include 10 zero transmission points either side of the wavelength
        # arrays
        for i, lam in enumerate(arrays):
            for _ in range(10):
                lam = np.insert(lam, 0, lam[0] - (lam[1] - lam[0]))
                lam = np.append(lam, lam[-1] + (lam[-1] - lam[-2]))
            arrays[i] = lam

        # Combine everything together in order
        new_lam: NDArray[np.float64] = np.concatenate(arrays)

        # Remove any duplicate values
        new_lam = np.unique(new_lam)

        # New remove any overlaps by iteratively removing negative differences
        # between adjacent elements
        diffs: NDArray[np.float64] = np.diff(new_lam)
        while np.min(diffs) < 0:
            end_val: np.float64 = new_lam[-1]
            new_lam = new_lam[:-1][diffs > 0]
            new_lam = np.append(new_lam, end_val)
            diffs = np.diff(new_lam)

        # Are we filling gaps?
        if fill_gaps:
            # Get the minimum resolution (largest gap between bins) of
            # each filter for gap filling
            min_res: np.float64 = np.max(
                [np.max(np.diff(arr)) for arr in arrays]
            )

            # Get the minimum resolution of the new array
            min_res_new: np.float64 = np.max(np.diff(new_lam))

            # Fill any gaps until the minimum resolution is reached
            while min_res_new > min_res:
                # Get the indices of the gaps
                gaps: NDArray[np.int32] = np.where(diffs > min_res)[0]

                # Loop over the gaps and fill them
                for g in gaps:
                    new_lam = np.insert(
                        new_lam, g + 1, (new_lam[g] + new_lam[g + 1]) / 2
                    )

                # Get the new minimum resolution
                diffs = np.diff(new_lam)
                min_res_new = np.max(np.diff(new_lam))

        return new_lam

    def resample_filters(
        self,
        new_lam: Optional[unyt_array] = None,
        lam_size: Optional[int] = None,
        fill_gaps: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Resample all filters onto a single wavelength array.

        If no wavelength grid is provided then the wavelength array of each
        individual Filter will be combined to cover the full range of the
        FilterCollection. Any overlapping ranges will take the values from one
        of the overlapping filters, any gaps between filters can be filled
        with the minimum average resolution of all filters to ensure a
        continuous array without needlessly inflating the memory footprint
        of any lam sized arrays.

        Alternatively, if new_lam is not passed, lam_size can be passed
        in which case a wavelength array from the minimum Filter wavelength
        to the maximum Filter wavelength will be generated with lam_size
        wavelength bins.

        Warning:
            If working with a Grid without passing the Grid wavelength
            array to a FilterCollection the wavelengths arrays will not
            agree producing at best array errors and at worst incorrect
            results from broadband photometry calculations.

        Args:
            new_lam: Wavelength array on which to sample filters. Wavelengths
                     should be in Angstrom. Defaults to None and an array is
                     derived.
            lam_size: The desired number of wavelength bins in the new
                      wavelength array, if no explicit array has been passed.
            fill_gaps: Are we filling gaps in the wavelength array? Defaults to
                       False.
            verbose: Are we talking?
        """
        # Do we need to find a wavelength array from the filters?
        if new_lam is None:
            # Get the wavelength limits
            min_lam: unyt_quantity
            max_lam: unyt_quantity
            min_lam, max_lam = self.get_non_zero_lam_lims()

            # Are we making an array with a fixed size?
            if lam_size is not None:
                # Create wavelength array
                new_lam = np.linspace(min_lam, max_lam, lam_size)

            else:
                # Ok, we are trying to be clever, merge the filter wavelength
                # arrays into a single array.
                new_lam = self._merge_filter_lams(fill_gaps=fill_gaps)

            if verbose:
                print(
                    "Calculated wavelength array: \n"
                    + "min = %.2e Angstrom\n" % new_lam.min()
                    + "max = %.2e Angstrom\n" % new_lam.max()
                    + "FilterCollection.lam.size = %d" % new_lam.size
                )

        # Set the wavelength array
        self.lam = new_lam

        # Loop over filters unifying them onto this wavelength array
        # NOTE: Filters already on self.lam will be uneffected but doing a
        # np.all condition to check for matches and skip them is more expensive
        # than just doing the interpolation for all filters
        for fcode in self.filters:
            f: Filter = self.filters[fcode]
            f._interpolate_wavelength(self.lam)

    def unify_with_grid(
        self,
        grid: "Grid",
        loop_spectra: bool = False,
    ) -> None:
        """
        Unify a grid with this FilterCollection.

        This will interpolate the grid onto the wavelength grid of this
        FilterCollection.

        Args:
            grid: The grid to be unified with this FilterCollection.
            loop_spectra: Flag for whether to do the interpolation over the
                          whole grid, or loop over the first axes. The latter
                          is less memory intensive, but slower. Defaults to
                          False.
        """
        # Interpolate the grid onto this wavelength grid
        grid.interp_spectra(self.lam, loop_spectra)

    def _transmission_curve_ax(
        self,
        ax: Axes,
        **kwargs: Dict[str, Any],
    ) -> None:
        """
        Add filter transmission curves to a given axes.

        Args:
            ax: The axis to plot the transmission curves in.
        """
        # Loop over the filters plotting their curves.
        for key in self.filters:
            f: Filter = self.filters[key]
            # Mypy has a bug that means kwargs expansion fails. We therefore
            # have to ignore the types here specifically.
            ax.plot(f._lam, f.t, label=f.filter_code, **kwargs)  # type: ignore

        # Label the axes
        ax.set_xlabel(r"$\rm \lambda/\AA$")
        ax.set_ylabel(r"$\rm T_{\lambda}$")

    def plot_transmission_curves(
        self,
        show: bool = False,
        fig: Optional[Figure] = None,
        ax: Optional[Axes] = None,
        **kwargs: Dict[str, Any],
    ) -> Tuple[Figure, Axes]:
        """
        Plot all Filter transmission curves.

        Args:
            show: Are we showing the output?

        Returns:
            The matplotlib figure object containing the plot.
            The matplotlib axis object containg the plot.
        """
        # Set up figure
        if fig is None:
            fig = plt.figure(figsize=(5.0, 3.5))

        if ax is None:
            left: float = 0.1
            height: float = 0.8
            bottom: float = 0.15
            width: float = 0.85

            # Add an axis to hold plot
            ax = fig.add_axes((left, bottom, width, height))

        # Make plot
        self._transmission_curve_ax(ax, **kwargs)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.2),
            fancybox=True,
            shadow=True,
            ncol=3,
        )

        # Are we showing?
        if show:
            plt.show()

        return fig, ax

    def calc_pivot_lams(self) -> unyt_array:
        """
        Calculate the rest frame pivot wavelengths of all filters.

        Returns:
            An array containing the rest frame pivot wavelengths of each filter
            in the same order as self.filter_codes.
        """
        # Calculate each filters pivot wavelength
        pivot_lams: NDArray[np.float64] = np.zeros(len(self))
        for ind, f in enumerate(self):
            pivot_lams[ind] = f.pivwv().value

        return pivot_lams * self.lam.units

    def calc_mean_lams(self) -> unyt_array:
        """
        Calculate the rest frame mean wavelengths of all filters.

        Returns:
            An array containing the rest frame mean wavelengths of each
            filter in the same order as self.filter_codes.
        """
        # Calculate each filters pivot wavelength
        mean_lams: NDArray[np.float64] = np.zeros(len(self))
        for ind, f in enumerate(self):
            mean_lams[ind] = f.meanwv().value

        return mean_lams * self.lam.units

    def find_filter(
        self,
        rest_frame_lam: Union[np.float64, unyt_quantity],
        redshift: Optional[float] = None,
        method: str = "pivot",
    ) -> Filter:
        """
        Find the Filter that probes a rest frame wavelength.

        If a redshift is provided then the wavelength is shifted into the
        observer frame and the filter that probes that wavelength in the
        observed frame is returned.

        Three Methods are provided to decide which filter to return:
            "pivot" (default)  - The filter with the closest pivot wavelength
                                 is returned.
            "mean"             - The filter with the closest mean wavelength is
                                 returned.
            "transmission"     - The filter with the peak transmission at the
                                 wavelength is returned.

        Args:
            rest_frame_lam: The wavelength to find the nearest filter to.
            redshift: The redshift of the observation. None for rest_frame,
                      defaults to None.
            method: The method to decide which filter to return. Either "pivot"
                    (default), "mean", or "transmission".

        Returns:
            The closest Filter in this FilterCollection. The filter-code
            of this filter is also printed.

        Raises:
            WavelengthOutOfRange:
                If the passed wavelength is out of range of any of the filters
                then an error is thrown.
        """
        # Remove units
        rest_frame_lam_unitless = (
            rest_frame_lam.value
            if isinstance(rest_frame_lam, unyt_quantity)
            else rest_frame_lam
        )

        # Are we working in a shifted frame or not?
        lam: np.float64
        if redshift is not None:
            # Get the shifted wavelength
            lam = rest_frame_lam_unitless * (1 + redshift)

        else:
            # Get the rest frame wavelength
            lam = rest_frame_lam_unitless

        # Which method are we using?
        ind: np.int32
        if method == "pivot":
            # Calculate each filters pivot wavelength
            pivot_lams: NDArray[np.float64] = self._pivot_lams

            # Find the index of the closest pivot wavelength to lam
            ind = np.argmin(np.abs(pivot_lams - lam))

        elif method == "mean":
            # Calculate each filters mean wavelength
            mean_lams: NDArray[np.float64] = self._mean_lams

            # Find the index of the closest mean wavelength to lam
            ind = np.argmin(np.abs(mean_lams - lam))

        elif method == "transmission":
            # Compute the transmission in each filter at lam
            transmissions: NDArray[np.float64] = np.zeros(len(self))
            for i, f in enumerate(self):
                transmissions[i] = f.t[np.argmin(np.abs(self._lam - lam))]

            # Find the index of the filter with the peak transmission
            ind = np.argmax(transmissions)

        else:
            raise InconsistentArguments(
                "Method not recognised! Can be either 'pivot', "
                "'mean'' or 'transmission'"
            )

        # Get the filter code and object for the found filter
        fcode: str = self.filter_codes[ind]
        out_f: Filter = self.filters[fcode]

        # Get the transmission
        transmission: np.float64 = out_f.t[np.argmin(np.abs(self._lam - lam))]

        # Ensure the transmission is non-zero at the desired wavelength
        if transmission == 0:
            if method == "pivot" or method == "mean":
                if redshift is None:
                    raise WavelengthOutOfRange(
                        "The wavelength "
                        f"(rest_frame_lam={rest_frame_lam:.2e} "
                        "Angstrom) has 0 transmission in the closest "
                        f"Filter ({fcode}). Try method='transmission'."
                    )
                else:
                    raise WavelengthOutOfRange(
                        f"The wavelength (rest_frame_lam={rest_frame_lam:.2e} "
                        f"Angstrom, observed_lam={lam:.2e} Angstrom)"
                        " has 0 transmission in the closest "
                        f"Filter ({fcode}). Try method='transmission'."
                    )
            else:
                if redshift is None:
                    raise WavelengthOutOfRange(
                        f"The wavelength (rest_frame_lam={rest_frame_lam:.2e} "
                        "Angstrom) does not fall in any Filters."
                    )
                else:
                    raise WavelengthOutOfRange(
                        f"The wavelength (rest_frame_lam={rest_frame_lam:.2e} "
                        f"Angstrom, observed_lam={lam:.2e} Angstrom)"
                        " does not fall in any Filters."
                    )

        if redshift is None:
            print(
                "Filter containing rest_frame_lam=%.2e Angstrom: %s"
                % (lam, fcode)
            )
        else:
            print(
                "Filter containing rest_frame_lam=%.2e Angstrom "
                "(with observed wavelength=%.2e Angstrom): %s"
                % (rest_frame_lam, lam, fcode)
            )

        return out_f

    def write_filters(self, path: str) -> None:
        """
        Write the current state of the FilterCollection to a HDF5 file.

        Args:
            path: The file path at which to save the FilterCollection.
        """
        # Open the HDF5 file  (will overwrite existing file at path)
        hdf: h5py.File = h5py.File(path, "w")

        # Create header group
        head: h5py.Group = hdf.create_group("Header")

        # Include the Synthesizer version
        head.attrs["synthesizer_version"] = __version__

        # Wrtie the FilterCollection attributes
        head.attrs["nfilters"] = self.nfilters

        # Write the wavelengths
        head.create_dataset("Wavelengths", data=self._lam)

        # Store the wavelength units
        head.attrs["Wavelength_units"] = str(self.lam.units)

        # Write the filter codes
        head.attrs["filter_codes"] = self.filter_codes

        # For each filter...
        for fcode, filt in self.filters.items():
            # Create the filter group
            f_grp: h5py.Group = hdf.create_group(fcode.replace("/", "."))

            # Write out the filter type
            f_grp.attrs["filter_type"] = filt.filter_type

            # Write out the filter code
            f_grp.attrs["filter_code"] = filt.filter_code

            # Write out the type specific attributes
            if filt.filter_type == "SVO":
                f_grp.attrs["svo_url"] = filt.svo_url
                f_grp.attrs["observatory"] = filt.observatory
                f_grp.attrs["instrument"] = filt.instrument
                f_grp.attrs["filter_"] = filt.filter_
            elif filt.filter_type == "TopHat":
                if filt._lam_min is not None:
                    f_grp.attrs["lam_min"] = filt._lam_min
                    f_grp.attrs["lam_max"] = filt._lam_max
                else:
                    f_grp.attrs["lam_eff"] = filt._lam_eff
                    f_grp.attrs["lam_fwhm"] = filt._lam_fwhm

            # Create transmission dataset
            f_grp.create_dataset("Transmission", data=filt.t)

            # For an SVO filter we need the original wavelength and
            # transmission curves
            if filt.filter_type == "SVO":
                f_grp.create_dataset(
                    "Original_Wavelength", data=filt._original_lam
                )
                f_grp.create_dataset(
                    "Original_Transmission", data=filt.original_t
                )

        hdf.close()


def UVJ(new_lam: Optional[NDArray[np.float64]] = None) -> FilterCollection:
    """
    Return a FilterCollection containing UVJ tophat filters.

    Args:
        new_lam: The wavelength array for which each filter's transmission
                 curve is defined.

    Returns:
        A FilterCollection containing top hat UVJ filters.
    """
    # Define the UVJ filters dictionary.
    tophat_dict: Dict[str, Dict[str, float]] = {
        "U": {"lam_eff": 3650, "lam_fwhm": 660},
        "V": {"lam_eff": 5510, "lam_fwhm": 880},
        "J": {"lam_eff": 12200, "lam_fwhm": 2130},
    }

    return FilterCollection(tophat_dict=tophat_dict, new_lam=new_lam)
