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
import h5py
import numpy as np
import urllib.request
import matplotlib.pyplot as plt
from scipy import integrate
from unyt import Angstrom, c, Hz, unyt_array, unyt_quantity
from urllib.error import URLError

import synthesizer.exceptions as exceptions
from synthesizer.units import Quantity
from synthesizer._version import __version__


def UVJ(new_lam=None):
    """
    Helper function to produce a FilterCollection containing
    UVJ tophat filters.

    Args:
        new_lam (array-like, float)
            The wavelength array for which each filter's transmission curve is
            defined.

    Returns:
        FilterCollection
            A FilterCollection containing top hat UVJ filters.
    """

    # Define the UVJ filters dictionary.
    tophat_dict = {
        "U": {"lam_eff": 3650, "lam_fwhm": 660},
        "V": {"lam_eff": 5510, "lam_fwhm": 880},
        "J": {"lam_eff": 12200, "lam_fwhm": 2130},
    }

    return FilterCollection(tophat_dict=tophat_dict, new_lam=new_lam)


class FilterCollection:
    """
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
        filters (dict, Filter)
            A list containing the individual `Filter` objects.
        filter_codes (list, string)
            A list of the names of each filter. For SVO filters these have to
            have the form "Observatory/Instrument.Filter" matching the
            database, but for all other filter types this can be an arbitrary
            label.
        lam (Quantity, array-like, float)
            The wavelength array for which each filter's transmission curve is
            defined.
        nfilters (int)
            The number of filters in this collection.
        mean_lams (Quantity, array-like, float)
            The mean wavelength of each Filter in the collection.
        pivot_lams (Quantity, array-like, float)
            The mean wavelength of each Filter in the collection.
    """

    # Define Quantitys
    lam = Quantity()
    mean_lams = Quantity()
    pivot_lams = Quantity()

    def __init__(
        self,
        filter_codes=None,
        tophat_dict=None,
        generic_dict=None,
        filters=None,
        path=None,
        new_lam=None,
    ):
        """
        Intialise the FilterCollection.

        Args:
            filter_codes  (list, string)
                A list of SVO filter codes, used to retrieve filter data from
                the database.
            tophat_dict (dict)
                A dictionary containing the data to make a collection of top
                hat filters from user defined properties. The dictionary must
                have the form:
                    {<filter_code> : {"lam_eff": <effective_wavelength>,
                                      "lam_fwhm": <FWHM_of_filter>}, ...},
                or:
                    {<filter_code> : {"lam_min": <minimum_nonzero_wavelength>,
                                      "lam_max": <maximum_nonzero_wavelength>},
                                      ...}.
            generic_dict (dict, float)
                A dictionary containing the data to make a collection of
                filters from user defined transmission curves. The dictionary
                must have the form:
                    {<filter_code1> : {"transmission": <transmission_array>}}.
                For generic filters new_lam must be provided.
            filters (list, Filter)
                A list of existing `Filter` objects to be added to the
                collection.
            path (string)
                A filepath defining the HDF5 file from which to load the
                FilterCollection.
            new_lam (array-like, float)
                The wavelength array to define the transmission curve on. Can
                have units but Angstrom assumed.
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
                self.resample_filters()

        # Calculate mean and pivot wavelengths for each filter
        self.mean_lams = self.calc_mean_lams()
        self.pivot_lams = self.calc_pivot_lams()

    def _load_filters(self, path):
        """
        Loads a `FilterCollection` from a HDF5 file.

        Args:
            path (str)
                The file path from which to load the `FilterCollection`.
        """

        # Open the HDF5 file
        hdf = h5py.File(path, "r")

        # Warn if the synthesizer versions don't match
        if hdf["Header"].attrs["synthesizer_version"] != __version__:
            print(
                "WARNING: Synthesizer versions differ between the code and "
                "FilterCollection file! This is probably fine but there "
                "is no gaurantee it won't cause errors."
            )

        # Get the wavelength units
        lam_units = hdf["Header"].attrs["Wavelength_units"]

        # Get the FilterCollection level attributes and datasets,
        # We apply the units to ensure conversions are done correctly
        # within the Quantity instantiation
        self.nfilters = hdf["Header"].attrs["nfilters"]
        self.lam = unyt_array(hdf["Header"]["Wavelengths"][:], lam_units)
        self.filter_codes = hdf["Header"].attrs["filter_codes"]

        # Loop over the groups and make the filters
        for filter_code in self.filter_codes:
            # Open the filter group
            f_grp = hdf[filter_code.replace("/", ".")]

            # Get the filter type
            filter_type = f_grp.attrs["filter_type"]

            # For SVO filters we don't want to send a request to the
            # database so instead instatiate it as a generic filter and
            # overwrite some attributes after the fact
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
                tophat_dict = {
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

    def _include_svo_filters(self, filter_codes):
        """
        Populate the `FilterCollection` with filters from SVO.

        Args:
            filter_codes (list, string)
                A list of SVO filter codes, used to retrieve filter data from
                the database.
        """

        # Loop over the given filter codes
        for f in filter_codes:
            # Get filter from SVO
            _filter = Filter(f, new_lam=self.lam)

            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def _include_top_hat_filters(self, tophat_dict):
        """
        Populate the `FilterCollection` with user defined top-hat filters.

        Args:
            tophat_dict (dict)
                A dictionary containing the data to make a collection of top
                hat filters from user defined properties. The dictionary must
                have the form:
                    {<filter_code> : {"lam_eff": <effective_wavelength>,
                                      "lam_fwhm": <FWHM_of_filter>}, ...},
                or:
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
            _filter = Filter(
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

    def _include_generic_filters(self, generic_dict):
        """
        Populate the `FilterCollection` with user defined filters.

        Args:
            generic_dict (dict)
                A dictionary containing the data to make a collection of
                filters from user defined transmission curves. The dictionary
                must have the form:
                    {<filter_code1> : {"transmission": <transmission_array>}}.
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

    def _include_synthesizer_filters(self, filters):
        """
        Populate the `FilterCollection` with a list of individual
        `Filter` objects.

        Args:
            filter_codes (list, string)
                A list of SVO filter codes, used to retrieve filter data from
                the database.
        """

        # Loop over the given filter codes
        for _filter in filters:
            # Store the filter and its code
            self.filters[_filter.filter_code] = _filter
            self.filter_codes.append(_filter.filter_code)

    def __add__(self, other_filters):
        """
        Enable the addition of FilterCollections and Filters with
        filtercollection1 + filtercollection2 or filtercollection + filter
        syntax.

        Returns:
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
                    other_filters.filters[key].filter_code
                )

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

        Args:
            other_filters  (obj, FilterCollection)
                The other FilterCollection to be compared to self.

        Returns:
            True/False (bool)
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

        Args:
            other_filters (obj, FilterCollection)
                The other FilterCollection to be compared to self.

        Returns:
            True/False (bool)
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

    def __getitem__(self, key):
        """
        Enables the extraction of filter objects from the FilterCollection by
        getitem syntax (FilterCollection[key] rather than
        FilterCollection.filters[key]).

        Args:
            key (string)
                The filter code of the desired filter.

        Returns:
            Filter
                The Filter object stored at self.filters[key].

        Raises:
            KeyError
                When the filter does not exist in self.filters an error is
                raised.
        """

        return self.filters[key]

    def resample_filters(self, new_lam=None, lam_resolution=1, verbose=True):
        """
        Resamples all filters onto a single wavelength array. If no wavelength
        grid is provided an array encompassing all filter transmission curves
        is derived with resolution stated by lam_resolution.

        Warning:
            If working with a Grid without passing the Grid wavelength
            array to a FilterCollection the wavelengths arrays will not
            agree producing at best array errors and at worst incorrect
            results from broadband photometry calculations.

        Args:
            new_lam (array-like, float)
                Wavelength array on which to sample filters. Wavelengths
                should be in Angstrom. Defaults to None and an array is
                derived.
            lam_resolution (float)
                The desired resolution of the derived wavelength array. Only
                used when new_lam is not provided.
            verbose (bool)
                Are we talking?
        """

        # Do we need to find a wavelength array from the filters?
        if new_lam is None:
            # Set up values for looping
            min_lam = np.inf
            max_lam = 0

            # Loop over filters getting the minimum and maximum wavelengths,
            # and highest resolution from the individual filters.
            for f in self.filters:
                this_min = np.min(self.filters[f]._lam)
                this_max = np.max(self.filters[f]._lam)
                if this_min < min_lam:
                    min_lam = this_min
                if this_max > max_lam:
                    max_lam = this_max

            # Create wavelength array
            new_lam = np.arange(
                min_lam, max_lam + lam_resolution, lam_resolution
            )

            if verbose:
                print(
                    "Calculated wavelength array: \n"
                    + "min = %.2e Angstrom\n" % min_lam
                    + "max = %.2e Angstrom\n" % max_lam
                    + "FilterCollection.lam.size = %d" % new_lam.size
                )

        # Set the wavelength array
        self.lam = new_lam

        # Loop over filters unifying them onto this wavelength array
        # NOTE: Filters already on self.lam will be uneffected but doing a
        # np.all condition to check for matches and skip them is more expensive
        # than just doing the interpolation for all filters
        for fcode in self.filters:
            f = self.filters[fcode]
            f.t = f._interpolate_wavelength(self.lam)

    def _transmission_curve_ax(self, ax):
        """
        Add filter transmission curves to a given axes.

        Args:
            ax  (matplotlib.axis)
                The axis to plot the transmission curves in.
            add_filter_label : bool
                Are we labelling the filter? (NotYetImplemented)
        """

        # TODO: Add colours

        # Loop over the filters plotting their curves.
        for key in self.filters:
            f = self.filters[key]
            ax.plot(f._lam, f.t, label=f.filter_code)

        # Label the axes
        ax.set_xlabel(r"$\rm \lambda/\AA$")
        ax.set_ylabel(r"$\rm T_{\lambda}$")

    def plot_transmission_curves(self, show=False):
        """
        Create a filter transmission curve plot of all Filters in the
        FilterCollection.

        Args:
            show (bool)
                Are we showing the output?

        Returns:
            fig (matplotlib.Figure)
                The matplotlib figure object containing the plot.
            ax obj (matplotlib.axis)
                The matplotlib axis object containg the plot.
        """

        # Set up figure
        fig = plt.figure(figsize=(5.0, 3.5))
        left = 0.1
        height = 0.8
        bottom = 0.15
        width = 0.85

        # Add an axis to hold plot
        ax = fig.add_axes((left, bottom, width, height))

        # Make plot
        self._transmission_curve_ax(ax)

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

    def calc_pivot_lams(self):
        """
        Calculates the rest frame pivot wavelengths of all filters in this
        FilterCollection.

        Returns:
            pivot_lams (ndarray, float)
                An array containing the rest frame pivot wavelengths of each
                filter in the same order as self.filter_codes.
        """

        # Calculate each filters pivot wavelength
        pivot_lams = np.zeros(len(self))
        for ind, f in enumerate(self):
            pivot_lams[ind] = f.pivwv()

        return pivot_lams

    def calc_mean_lams(self):
        """
        Calculates the rest frame mean wavelengths of all filters in this
        FilterCollection.

        Returns:
            mean_lams (ndarray, float)
                An array containing the rest frame mean wavelengths of each
                filter in the same order as self.filter_codes.
        """

        # Calculate each filters pivot wavelength
        mean_lams = np.zeros(len(self))
        for ind, f in enumerate(self):
            mean_lams[ind] = f.meanwv()

        return mean_lams

    def find_filter(self, rest_frame_lam, redshift=None, method="pivot"):
        """
        Takes a rest frame target wavelength and returns the filter that probes
        that wavelength.

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
            rest_frame_lam (float):
                The wavelength to find the nearest filter to.
            redshift (float):
                The redshift of the observation. None for rest_frame, defaults
                to None.
            method (str):
                The method to decide which filter to return. Either "pivot"
                (default), "mean", or "transmission".

        Returns:
            synthesizer.Filter
                The closest Filter in this FilterCollection. The filter-code
                of this filter is also printed.

        Raises:
            WavelengthOutOfRange:
                If the passed wavelength is out of range of any of the filters
                then an error is thrown.
        """

        # Are we working in a shifted frame or not?
        if redshift is not None:
            # Get the shifted wavelength
            lam = rest_frame_lam * (1 + redshift)

        else:
            # Get the rest frame wavelength
            lam = rest_frame_lam

        # Which method are we using?
        if method == "pivot":
            # Calculate each filters pivot wavelength
            pivot_lams = self._pivot_lams

            # Find the index of the closest pivot wavelength to lam
            ind = np.argmin(np.abs(pivot_lams - lam))

        elif method == "mean":
            # Calculate each filters mean wavelength
            mean_lams = self._mean_lams

            # Find the index of the closest mean wavelength to lam
            ind = np.argmin(np.abs(mean_lams - lam))

        elif method == "transmission":
            # Compute the transmission in each filter at lam
            transmissions = np.zeros(len(self))
            for ind, f in enumerate(self):
                transmissions[ind] = f.t[np.argmin(np.abs(self._lam - lam))]

            # Find the index of the filter with the peak transmission
            ind = np.argmax(transmissions)

        else:
            raise exceptions.InconsistentArguments(
                "Method not recognised! Can be either 'pivot', "
                "'mean'' or 'transmission'"
            )

        # Get the filter code and object for the found filter
        fcode = self.filter_codes[ind]
        f = self.filters[fcode]

        # Get the transmission
        transmission = f.t[np.argmin(np.abs(self._lam - lam))]

        # Ensure the transmission is non-zero at the desired wavelength
        if transmission == 0:
            if method == "pivot" or method == "mean":
                if redshift is None:
                    raise exceptions.WavelengthOutOfRange(
                        "The wavelength "
                        f"(rest_frame_lam={rest_frame_lam:.2e} "
                        "Angstrom) has 0 transmission in the closest "
                        f"Filter ({fcode}). Try method='transmission'."
                    )
                else:
                    raise exceptions.WavelengthOutOfRange(
                        f"The wavelength (rest_frame_lam={rest_frame_lam:.2e} "
                        f"Angstrom, observed_lam={lam:.2e} Angstrom)"
                        " has 0 transmission in the closest "
                        f"Filter ({fcode}). Try method='transmission'."
                    )
            else:
                if redshift is None:
                    raise exceptions.WavelengthOutOfRange(
                        f"The wavelength (rest_frame_lam={rest_frame_lam:.2e} "
                        "Angstrom) does not fall in any Filters."
                    )
                else:
                    raise exceptions.WavelengthOutOfRange(
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

        return f

    def write_filters(self, path):
        """
        Writes the current state of the FilterCollection to a HDF5 file.

        Args:
            path (str)
                The file path at which to save the FilterCollection.
        """

        # Open the HDF5 file  (will overwrite existing file at path)
        hdf = h5py.File(path, "w")

        # Create header group
        head = hdf.create_group("Header")

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
            f_grp = hdf.create_group(fcode.replace("/", "."))

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
        filter_code (string)
            The full name defining this Filter.
        observatory (string)
            The name of the observatory
        instrument (string)
            The name of the instrument.
        filter_ (string)
            The name of the filter.
        filter_type (string)
            A string describing the filter type: "SVO", "TopHat", or "Generic".
        lam_min (Quantity)
            If a top hat filter: The minimum wavelength where transmission is
            nonzero.
        lam_max (Quantity)
            If a top hat filter: The maximum wavelength where transmission is
            nonzero.
        lam_eff (Quantity)
            If a top hat filter: The effective wavelength of the filter curve.
        lam_fwhm (Quantity)
            If a top hat filter: The FWHM of the filter curve.
        svo_url (string)
            If an SVO filter: the url from which the data was extracted.
        t (array-like, float)
            The transmission curve.
        lam (Quantity, array-like, float)
            The wavelength array for which the transmission is defined.
        nu (Quantity, array-like, float)
            The frequency array for which the transmission is defined. Derived
            from self.lam.
        original_lam (Quantity, array-like, float)
            The original wavelength extracted from SVO. In a non-SVO filter
            self.original_lam == self.lam.
        original_nu (Quantity, array-like, float)
            The original frequency derived from self.original_lam. In a non-SVO
            filter self.original_nu == self.nu.
        original_t (array-like, float)
            The original transmission extracted from SVO. In a non-SVO filter
            self.original_t == self.t.
    """

    # Define Quantitys
    lam_min = Quantity()
    lam_max = Quantity()
    lam_eff = Quantity()
    lam_fwhm = Quantity()
    lam = Quantity()
    nu = Quantity()
    original_lam = Quantity()
    original_nu = Quantity()

    def __init__(
        self,
        filter_code,
        transmission=None,
        lam_min=None,
        lam_max=None,
        lam_eff=None,
        lam_fwhm=None,
        new_lam=None,
    ):
        """
        Initialise a filter.

        Args:
            filter_code (string)
                The full name defining this Filter.
            transmission : array-like (float)
                An array describing the filter's transmission curve. Only used
                for generic filters.
            lam_min (float)
                If a top hat filter: The minimum wavelength where transmission
                is nonzero.
            lam_max (float)
                If a top hat filter: The maximum wavelength where transmission
                is nonzero.
            lam_eff (float)
                If a top hat filter: The effective wavelength of the filter
                curve.
            lam_fwhm (float)
                If a top hat filter: The FWHM of the filter curve.
            new_lam (array-like, float)
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
        if self.original_t is None:
            self.original_t = self.t

        # Calculate frequencies
        self.nu = (c / self.lam).to("Hz").value
        self.original_nu = (c / self.original_lam).to("Hz").value

    def _make_top_hat_filter(self):
        """
        Make a top hat filter from the Filter's attributes.
        """

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

    def _make_svo_filter(self):
        """
        Retrieve a filter's data from the SVO database based on the Filter's
        attributes.

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
                df = np.loadtxt(f)
        except URLError:
            raise exceptions.SVOInaccessible(
                (
                    f"The SVO Database at {self.svo_url} "
                    "is not responding. Is it down?"
                )
            )

        # Throw an error if we didn't find the filter.
        if df.size == 0:
            raise exceptions.SVOFilterNotFound(
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
            self.t = self._interpolate_wavelength()
        else:
            self.lam = self.original_lam
            self.t = self.original_t

    def _interpolate_wavelength(self, new_lam=None):
        """
        Interpolates a filter transmission curve onto the Filter's wavelength
        array.

        Args:
            new_lam (array-like, float)
                The wavelength array to interpolate onto. If None self.lam
                is used.

        Returns:
            array-like (float)
                Transmission curve interpolated onto the new wavelength array.
        """

        # If we've been handed a wavelength array we must overwrite the
        # current one
        if new_lam is not None:
            # Warn the user if we're about to truncate the existing wavelength
            # array
            truncated = False
            if new_lam.min() > self.lam.min():
                truncated = True
            if new_lam.max() < self.lam.max():
                truncated = True
            if truncated:
                print(
                    f"Warning: {self.filter_code} is being truncated "
                    "(old_lam_bounds = "
                    f"({self.lam.min():.2e}, {self.lam.max():.2e}), "
                    "new_lam_bounds = "
                    f"({new_lam.min():.2e}, {new_lam.max():.2e}))"
                )

            self.lam = new_lam

        # Perform interpolation
        return np.interp(
            self._lam, self._original_lam, self.original_t, left=0.0, right=0.0
        )

    def apply_filter(self, arr, lam=None, nu=None, verbose=True):
        """
        Apply this filter's transmission curve to an arbitrary dimensioned
        array returning the sum of the array convolved with the filter
        transmission curve along the wavelength axis (final axis).

        If no wavelength or frequency array is provided then the filters rest
        frame frequency is assumed.

        Args:
            arr (array-like, float)
                The array to convolve with the filter's transmission curve. Can
                be any dimension but wavelength must be the final axis.
            lam (unyt_array/array-like, float)
                The wavelength array to integrate with respect to.
                Defaults to the rest frame frequency if neither lams or nus are
                provided.
            nu :  (unyt_array/array-like, float)
                The frequency array to integrate with respect to.
                Defaults to the rest frame frequency if neither lams or nus are
                provided.
            verbose (bool)
                Are we talking?

        Returns:
            float
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
                lam = (c / nu).to(Angstrom).value

        elif lam is not None:
            # Ensure the passed wavelengths have no units
            if isinstance(lam, unyt_array):
                lam = lam.value

            # Define the integration xs
            xs = lam

            # Do we need to shift?
            need_shift = not lam[0] == self._lam[0]

        else:
            # Define the integration xs
            xs = self._nu

            # No shift needed
            need_shift = False

        # Do we need to shift?
        if need_shift:
            # Ok, shift the tranmission curve by interpolating onto the
            # provided wavelengths
            t = np.interp(
                lam, self._original_lam, self.original_t, left=0.0, right=0.0
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
        in_band = t > 0

        # Mask out wavelengths that don't contribute to this band
        arr_in_band = arr.compress(in_band, axis=-1)
        xs_in_band = xs[in_band]
        t_in_band = t[in_band]

        # Multiply the IFU by the filter transmission curve
        transmission = arr_in_band * t_in_band

        # Sum over the final axis to "collect" transmission in this filer
        sum_per_x = integrate.trapezoid(
            transmission / xs_in_band, xs_in_band, axis=-1
        )
        sum_den = integrate.trapezoid(
            t_in_band / xs_in_band, xs_in_band, axis=-1
        )
        sum_in_band = sum_per_x / sum_den

        return sum_in_band

    def pivwv(self):
        """
        Calculate the pivot wavelength.
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
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

    def pivT(self):
        """
        Calculate the transmission at the pivot wavelength.
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
                Transmission at pivot wavelength.
        """

        return np.interp(
            self.pivwv().value, self._original_lam, self.original_t
        )

    def meanwv(self):
        """
        Calculate the mean wavelength.
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
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

    def bandw(self):
        """
        Calculate the bandwidth.
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
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

    def fwhm(self):
        """
        Calculate the FWHM.
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
                The FWHM of the filter.
        """

        return np.sqrt(8.0 * np.log(2)) * self.bandw()

    def Tpeak(self):
        """
        Calculate the peak transmission
        For an SVO filter this uses the transmission from the database.

        Returns:
            float
                The peak transmission.
        """

        return np.max(self.original_t)

    def rectw(self):
        """
        Calculate the rectangular width.
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
                The rectangular width.
        """

        return np.trapz(self.original_t, x=self._original_lam) / self.Tpeak()

    def max(self):
        """
        Calculate the longest wavelength where the transmission is still >0.01
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
                The maximum wavelength at which transmission is nonzero.
        """

        return self.original_lam[self.original_t > 1e-2][-1]

    def min(self):
        """
        Calculate the shortest wavelength where the transmission is still >0.01
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
                The minimum wavelength at which transmission is nonzero.
        """

        return self.original_lam[self.original_t > 1e-2][0]

    def mnmx(self):
        """
        Calculate the minimum and maximum wavelengths.
        For an SVO filter this uses the wavelength and transmission from
        the database.

        Returns:
            float
                The minimum wavelength.
            float
                The maximum wavelength.
        """

        return (self.original_lam.min(), self.original_lam.max())
