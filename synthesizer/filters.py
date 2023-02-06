import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from scipy import integrate

import synthesizer.exceptions as exceptions


def UVJ(new_lam=None):
    """
    Helper function to produce a FilterCollection containing UVJ tophat filters.

    Parameters
    ----------
    new_lam : array-like (float)
        The wavelength array for which each filter's transmission curve is
        defined.

    Returns
    -------
    FilterCollection
        A filter collection containing top hat UVJ filters.
    """

    # Define the UVJ filters dictionary.
    tophat_dict = {'U': {'lam_eff': 3650, 'lam_fwhm': 660},
                   'V': {'lam_eff': 5510, 'lam_fwhm': 880},
                   'J': {'lam_eff': 12200, 'lam_fwhm': 2130}}

    return FilterCollection(tophat_dict=tophat_dict, new_lam=new_lam)


class FilterCollection:
    """
    Holds a collection of filters and enables various quality of life
    operations such as plotting, adding, looping and len as if the collection
    was a simple list.

    Filters can be derived from the SVO database
    (http://svo2.cab.inta-csic.es/svo/theory/fps3/), specific top hat filter
    properties or generic filter transmission curves and a wavelength array.

    All filters are defined in terms of the same wavelength array.

    Attributes
    ----------
    filters : list (Filter)
        A list containing the individual Filter objects.
    filter_codes : list (string)
        A list of the names of each filter. For SVO filters these have to have
        the form "Observatory/Instrument.Filter" matching the database, but for
        all other filter types this can be an arbitrary label.
    lam : array-like (float)
        The wavelength array for which each filter's transmission curve is
        defined.
    nfilters : inta
        The number of filters in this collection.

    Methods
    -------
    plot_transmission_curves
        A helper function to quickly plot all the transmission curves.

    """

    def __init__(self, filter_codes=None, tophat_dict=None, generic_dict=None,
                 new_lam=None):
        """
        Intialise the FilterCollection.

        Parameters
        ----------
        filter_codes : list (string)
            A list of SVO filter codes, used to retrieve filter data from the
            database.
        tophat_dict : dict
            A dictionary containing the data to make a collection of top hat
            filters from user defined properties. The dictionary must have
            the form:
            {<filter_code> : {"lam_eff": <effective_wavelength>,
                              "lam_fwhm": <FWHM_of_filter>}, ...},
            or:
            {<filter_code> : {"lam_min": <minimum_nonzero_wavelength>,
                              "lam_max": <maximum_nonzero_wavelength>}, ...}.
        generic_dict : dict
            A dictionary containing the data to make a collection of filters
            from user defined transmission curves. The dictionary must have
            the form:
            {<filter_code1> : {"transmission": <transmission_array>}}.
            For generic filters new_lam must be provided.
        new_lam : array-like (float)
            The wavelength array to define the tranmission curve on.
        """

        # Define lists to hold our filters and filter codes
        self.filters = {}
        self.filter_codes = []

        # Do we have an wavelength array? If so we will resample the
        # transmission.
        self.lam = new_lam

        # Let's make the filters
        if filter_codes is not None:
            self._make_svo_collection(filter_codes)
        if tophat_dict is not None:
            self._make_top_hat_collection(tophat_dict)
        if generic_dict is not None:
            self._make_generic_colleciton(generic_dict)

        # Atrributes to enable looping
        self._current_ind = 0
        self.nfilters = len(self.filter_codes)

    def _make_svo_collection(self, filter_codes):
        """
        Populate the FilterCollection with filters from SVO.

        Parameters
        ----------
        filter_codes : list (string)
            A list of SVO filter codes, used to retrieve filter data from the
            database.
        """

        # Loop over the given filter codes
        for f in filter_codes:

            # Get filter from SVO
            _filter = Filter(f, new_lam=self.lam)

            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def _make_top_hat_collection(self, tophat_dict):
        """
        Populate the FilterCollection with user defined top hat filters.

        Parameters
        ----------
        tophat_dict : dict
            A dictionary containing the data to make a collection of top hat
            filters from user defined properties. The dictionary must have
            the form:
            {<filter_code> : {"lam_eff": <effective_wavelength>,
                              "lam_fwhm": <FWHM_of_filter>}, ...},
            or:
            {<filter_code> : {"lam_min": <minimum_nonzero_wavelength>,
                              "lam_max": <maximum_nonzero_wavelength>}, ...}.
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
            _filter = Filter(key, lam_min=lam_min, lam_max=lam_max,
                             lam_eff=lam_eff, lam_fwhm=lam_fwhm,
                             new_lam=self.lam)

            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def _make_generic_colleciton(self, generic_dict):
        """
        Populate the FilterCollection with user defined filters.

        Parameters
        ----------
        generic_dict : dict
            A dictionary containing the data to make a collection of filters
            from user defined transmission curves. The dictionary must have
            the form:
            {<filter_code1>: <transmission_array>}.
            For generic filters new_lam must be provided.
        """

        # Loop over the keys of the dictionary
        for key in generic_dict:

            # Get this filter's properties
            t = generic_dict[key]

            # Instantiate the filter
            _filter = Filter(key, transmission=t, new_lam=self.lam)

            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def __add__(self, other_filters):
        """
        Enable the addition of FilterCollections and Filters with
        filtercollection1 + filtercollection2 or filtercollection + filter
        syntax.

        Returns
        -------
        FilterCollection
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
                    other_filters.filters[key].filter_code)

        elif isinstance(other_filters, Filter):

            # Store the filter and its code
            self.filters[other_filters.filter_code] = other_filters
            self.filter_codes.append(other_filters.filter_code)

        else:
            raise exceptions.InconsistentAddition(
                "Cannot add non-filter objects together!"
            )

        # Update the number of filters we have
        self.nfilters = len(self.filter_codes)

        return self

    def __len__(self):
        """
        Overload the len operator to return how many filters there are.
        """
        return len(self.filters)

    def __iter__(self):
        """
        Overload iteration to allow simple looping over filter objects,
        combined with __next__ this enables for f in FilterCollection syntax
        """
        return self

    def __next__(self):
        """
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

    def __ne__(self, other_filters):
        """
        Enables the != comparison of two filter collections. If the filter
        collections contain the same filter codes they are guaranteed to
        be identical.

        Parameters
        ----------
        other_filters : obj (FilterCollection)
            The other FilterCollection to be compared to self.

        Returns
        -------
        True/False : bool
             Are the FilterCollections the same?
        """

        # Do they have the same number of filters?
        if self.nfilters != other_filters.nfilters:
            return True

        # Ok they do, so do they have the same filter codes? (elementwise test)
        not_equal = False
        for n in range(self.nfilters):
            if self.filter_codes[n] != other_filters.filter_codes[n]:
                not_equal = True
                break

        return not_equal

    def __eq__(self, other_filters):
        """
        Enables the == comparison of two filter collections. If the filter
        collections contain the same filter codes they are guaranteed to
        be identical.

        Parameters
        ----------
        other_filters : obj (FilterCollection)
            The other FilterCollection to be compared to self.

        Returns
        -------
        True/False : bool
             Are the FilterCollections the same?
        """

        # Do they have the same number of filters?
        if self.nfilters != other_filters.nfilters:
            return False

        # Ok they do, so do they have the same filter codes? (elementwise test)
        equal = True
        for n in range(self.nfilters):
            if self.filter_codes[n] != other_filters.filter_codes[n]:
                equal = False
                break

        return equal

    def _transmission_curve_ax(self, ax, add_filter_label=True):
        """
        Add filter transmission curves to a give axes

        Parameters
        ----------
        ax : obj (matplotlib.axis)
            The axis to plot the transmission curves in.
        add_filter_label : bool
            Are we labelling the filter? (NotYetImplemented)

        """

        # TODO: Add colours

        # Loop over the filters plotting their curves.
        for key in self.filters:
            f = self.filters[key]
            ax.plot(f.lam, f.t, label=f.filter_code)

            # TODO: Add label with automatic placement

        # Label the axes
        ax.set_xlabel(r'$\rm \lambda/\AA$')
        ax.set_ylabel(r'$\rm T_{\lambda}$')

    def plot_transmission_curves(self, show=False):
        """
        Create a filter transmission curve plot

        Parameters
        ----------
        show : bool
            Are we showing the output?

        Returns
        -------
        fig obj (matplotlib.Figure)
            The matplotlib figure object containing the plot.
        ax obj (matplotlib.axis)
            The matplotlib axis object containg the plot.
        """

        # Set up figure
        fig = plt.figure(figsize=(5., 3.5))
        left = 0.1
        height = 0.8
        bottom = 0.15
        width = 0.85

        # Add an axis to hold plot
        ax = fig.add_axes((left, bottom, width, height))

        # Make plot
        self._transmission_curve_ax(ax, add_filter_label=True)

        # Are we showing?
        if show:
            plt.show()

        return fig, ax


class Filter:
    """
    A class holding a filter's transmission curve and wavelength array.

    A filter can either be retrieved from the SVO database
    (http://svo2.cab.inta-csic.es/svo/theory/fps3/), made from specific top hat
    filter properties or made from a generic filter transmission curve and
    wavelength array.

    Also contains methods for calculating basic filter properties taken
    from (page 42 (5.1)):
        http://stsdas.stsci.edu/stsci_python_epydoc/SynphotManual.pdf

    Attributes
    ----------
    filter_code : string
        The full name defining this Filter.
    observatory : string
        The name of the observatory
    instrument : string
        The name of the instrument.
    filter_ : string
        The name of the filter.
    filter_type : string
        A string describing the filter type: "SVO", "TopHat", or "Generic".
    lam_min : float
        If a top hat filter: The minimum wavelength where transmission is
        nonzero.
    lam_max : float
        If a top hat filter: The maximum wavelength where transmission is
        nonzero.
    lam_eff : float
        If a top hat filter: The effective wavelength of the filter curve.
    lam_fwhm : float
        If a top hat filter: The FWHM of the filter curve.
    svo_url : string
        If an SVO filter: the url from which the data was extracted.
    t : array-like (float)
        The transmission curve.
    lam : array-like (float)
        The wavelength array for which the transmission is defined.
    original_lam : array-like (float)
        The original wavelength extracted from SVO.
    original_t : array-like (float)
        The original transmission extracted from SVO.

    Methods
    -------
    pivwv:
        Calculate pivot wavelength
    """

    def __init__(self, filter_code, transmission=None, lam_min=None,
                 lam_max=None, lam_eff=None, lam_fwhm=None, new_lam=None):
        """
        Initialise a filter.

        Parameters
        ----------
        filter_code : string
            The full name defining this Filter.
        transmission : array-like (float)
            An array describing the filter's transmission curve. Only used for
            generic filters.
        lam_min : float
            If a top hat filter: The minimum wavelength where transmission is
            nonzero.
        lam_max : float
            If a top hat filter: The maximum wavelength where transmission is
            nonzero.
        lam_eff : float
            If a top hat filter: The effective wavelength of the filter curve.
        lam_fwhm : float
            If a top hat filter: The FWHM of the filter curve.
        new_lam : array-like (float)
            The wavelength array for which the transmission is defined.

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

        # Define transmission curve and wavelength (if provided) of
        # this filter
        self.t = transmission
        self.lam = new_lam
        self.original_lam = new_lam
        self.original_t = transmission

        # Is this a generic filter? (Everything other the label is defined
        # above.)
        if transmission is not None and new_lam is not None:
            self.filter_type = "Generic"

        # Is this a top hat filter?
        elif ((lam_min is not None and lam_max is not None) or
              (lam_eff is not None and lam_fwhm is not None)):
            self._make_top_hat_filter()

        # Is this an SVO filter?
        elif "/" in filter_code and "." in filter_code:
            self._make_svo_filter()

        # Otherwise we haven't got a valid combination of inputs.
        else:
            raise exceptions.InconsistentArguments(
                "Invalid combination of filter inputs. \n For a generic "
                "filter provide a transimision and wavelength array. \nFor "
                "a filter from the SVO database provide a filter code of the "
                "form Observatory/Instrument.Filter that matches the database."
                " \nFor a top hat provide either a minimum and maximum "
                "wavelength or an effective wavelength and FWHM."
            )

        # Define the original wavelength and transmission for property
        # calculation later.
        if self.original_lam is None:
            self.original_lam = self.lam
            self.original_t = self.t

    def _make_top_hat_filter(self):
        """
        Make a top hat filter from the input.
        """

        # Define the type of this filter
        self.filter_type = "TopHat"

        # If filter has been defined with an effective wavelength and FWHM
        # calculate the minimum and maximum wavelength.
        if self.lam_eff is not None and self.lam_fwhm is not None:
            self.lam_min = self.lam_eff - self.lam_fwhm/2.
            self.lam_max = self.lam_eff + self.lam_fwhm/2.

        # Otherwise, use the explict min and max

        # Define this top hat filters wavelength array (+/- 1000 Angstrom)
        # if it hasn't been provided
        if self.lam is None:
            self.lam = np.arange(np.max([0, self.lam_min - 1000]),
                                 self.lam_max + 1000, 1)

        # Define the transmission curve (1 inside, 0 outside)
        self.t = np.zeros(len(self.lam))
        s = (self.lam > self.lam_min) & (self.lam <= self.lam_max)
        self.t[s] = 1.0

    def _make_svo_filter(self):
        """
        Retrieve a filter's data from the SVO database.

        Raises
        ------
        SVOFilterNotFound
            If a filter code cannot be matched to a database entry or a
            connection cannot be made to the database and error is thrown.
        """

        # Define the type of this filter
        self.filter_type = "SVO"

        # Get the information stored in the filter code
        self.observatory = self.filter_code.split('/')[0]
        self.instrument = self.filter_code.split('/')[1].split('.')[0]
        self.filter_ = self.filter_code.split('.')[-1]

        # Read directly from the SVO archive.
        self.svo_url = (f'http://svo2.cab.inta-csic.es/theory/'
                        f'fps/getdata.php?format=ascii&id={self.observatory}'
                        f'/{self.instrument}.{self.filter_}')
        with urllib.request.urlopen(self.svo_url) as f:
            df = np.loadtxt(f)

        # Throw an error if we didn't find the filter.
        if df.size == 0:
            raise exceptions.SVOFilterNotFound(
                "Filter (" + self.filter_code + ") not in the database. "
                "Double check the database: http://svo2.cab.inta-csic.es/"
                "svo/theory/fps3/. This could also mean you have no connection."
            )

        # Extract the wavelength and transmission given by SVO
        self.original_lam = df[:, 0]
        self.original_t = df[:, 1]

        # If a new wavelength grid is provided, interpolate
        # the transmission curve on to that grid
        if isinstance(self.lam, np.ndarray):
            self.t = self._interpolate_wavelength()
        else:
            self.lam = self.original_lam
            self.t = self.original_t

    def _interpolate_wavelength(self):
        """
        Interpolates a filter transmission curve onto the Filter's wavelength
        array.

        Returns
        -------
        array-like (float)
            Transmission curve interpolated onto the new wavelength array.
        """

        return np.interp(self.lam, self.original_lam, self.original_t,
                         left=0.0, right=0.0)

    def apply_filter(self, arr, xs=None):
        """
        Apply this filter's transmission curve to an arbitrary dimensioned
        array returning the sum of the array convolved with the filter
        transmission curve along the wavelength axis.

        Parameters
        ----------
        arr :  array-like (float)
            The array to convolve with the filter's transmission curve. Can
            be any dimension but wavelength must be the final axis.

        Returns
        -------
        sum_in_band : array-like (float)
            The array (arr) convolved with the transmission curve and summed
            along the wavelength axis.

        Raises
        ------
        ValueError
            If the shape of the transmission and wavelength array differ the
            convolution cannot be done.
        """

        # Check dimensions are ok
        if x is None:
            if self.lam.size != arr.shape[-1]:
                raise ValueError("Final dimension of array did not match "
                                 "wavelength array size (arr.shape[-1]=%d, "
                                 "transmission.size=%d)" % (arr.shape[-1],
                                                            self.lam.size))
            else:
                if xs.size != arr.shape[-1]:
                raise ValueError("Final dimension of array did not match "
                                 "wavelength array size (arr.shape[-1]=%d, "
                                 "xs.size=%d)" % (arr.shape[-1],
                                                  xs.size))

        # Get the mask that removes wavelengths we don't currently care about
        in_band = self.t > 0

        # Multiply the IFU by the filter transmission curve
        arr_in_band = arr.compress(in_band, axis=-1) * self.t[in_band]

        # Get the xs in the band, this could be lam or nu.
        if xs is not None:
            xs_in_band = xs[in_band]
        else:
            xs_in_band = self.lam[in_band]

        # Sum over the final axis to "collect" transmission in this filer
        sum_per_x = integrate.trapezoid(arr_in_band * self.t[in_band]
                                        / xs_in_band, xs_in_band, axis=-1)
        sum_den = integrate.trapezoid(self.t[in_band] / xs_in_band, xs_in_band,
                                      axis=-1)
        sum_in_band = sum_per_x / sum_den

        return sum_in_band

    def pivwv(self):
        """
        Calculate the pivot wavelength.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            Pivot wavelength.
        """

        return np.sqrt(np.trapz(self.original_lam * self.original_t,
                                x=self.original_lam) /
                       np.trapz(self.original_t / self.original_lam,
                                x=self.original_lam))

    def pivT(self):
        """
        Calculate the transmission at the pivot wavelength.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            Transmission at pivot wavelength.
        """

        return np.interp(self.pivwv(), self.original_lam, self.original_t)

    def meanwv(self):
        """
        Calculate the mean wavelength.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            Mean wavelength.        
        """

        return np.exp(np.trapz(np.log(self.original_lam) * self.original_t /
                               self.original_lam, x=self.original_lam) /
                      np.trapz(self.original_t / self.original_lam,
                               x=self.original_lam))

    def bandw(self):
        """
        Calculate the bandwidth.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            The bandwidth.
        """

        # Calculate the left and right hand side.
        A = np.sqrt(np.trapz((np.log(self.original_lam /
                                     self.meanwv())**2) * self.original_t /
                             self.original_lam, x=self.original_lam))

        B = np.sqrt(np.trapz(self.original_t / self.original_lam,
                             x=self.original_lam))

        return self.meanwv() * (A / B)

    def fwhm(self):
        """
        Calculate the FWHM.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            The FWHM of the filter.
        """

        return np.sqrt(8. * np.log(2)) * self.bandw()

    def Tpeak(self):
        """
        Calculate the peak transmission

        For an SVO filter this uses the transmission from the database.

        Returns
        -------
        float
            The peak transmission.
        """

        return np.max(self.original_t)

    def rectw(self):
        """
        Calculate the rectangular width.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            The rectangular width.
        """

        return np.trapz(self.original_t, x=self.original_lam) / self.Tpeak()

    def max(self):
        """
        Calculate the longest wavelength where the transmission is still >0.01

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            The maximum wavelength at which transmission is nonzero.
        """

        return self.original_lam[self.original_t > 1E-2][-1]

    def min(self):
        """
        Calculate the shortest wavelength where the transmission is still >0.01

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            The minimum wavelength at which transmission is nonzero.
        """

        return self.original_lam[self.original_t > 1E-2][0]

    def mnmx(self):
        """
        Calculate the minimum and maximum wavelengths.

        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns
        -------
        float
            The minimum wavelength.
        float
            The maximum wavelength.
        """

        return (self.original_lam.min(), self.original_lam.max())
