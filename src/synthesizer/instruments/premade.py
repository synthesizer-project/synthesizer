"""A submodule defining some premade instruments.

This module contains some commonly used instruments which can be imported,
instantiated, and used directly. Each premade instrument is a child of the
Instrument class with the appropriate properties fixed.

The premade instruments are:
    - JWST/NIRCam.Wide
    - JWST/NIRCam.Medium
    - JWST/NIRCam.Narrow
    - JWST/NIRCam (InstrumentCollection combining all NIRCam modes)
    - JWST/NIRSpec
    - JWST/MIRI
    - HST/WFC3
    - HST/ACS_WFC
    - Euclid/NISP
    - Euclid/NISP0
    - Euclid/VIS

"""

import os

import h5py
from unyt import arcsecond

from synthesizer import exceptions
from synthesizer.instruments import Instrument
from synthesizer.instruments.filters import FilterCollection

__all__ = [
    "JWSTNIRCamWide",
    "JWSTNIRCamMedium",
    "JWSTNIRCamNarrow",
]

# We need the relative path to the instruments_cache directory
# to store and retrieve the instrument files
INSTRUMENT_CACHE_DIR = os.path.join(
    os.path.dirname(__file__),
    "instrument_cache",
)


class PremadeInstrument(Instrument):
    """A base class for premade instruments.

    This class is a child of the Instrument class and is used to define
    premade instruments. It is not intended to be used directly, but
    rather as a base class for other premade instruments. Essentially,
    we include this class to reduce common methods and boilerplate code.

    By default this will just include the predefinable properties of the
    Instrument class. The user can optionally specify their own depth,
    depth aperture radius, and SNRs, or explicit noise maps to be used.
    Similarly the user can pass their own PSFs to be used.

    Alternatively, calling the load method will attempt to load the instrument
    from a file.

        ```python
        instrument = <premade instrument>.load(**kwargs)
        ```

    The upside of this is any "heavy attributes" (e.g. PSFs or
    full noise maps) will automatically be included from the file. This file
    needs to be downloaded using the download tool:

        ```bash
        synthesizer-download --instrument <instrument_name>
        ```
    Where the instrument name is the name of the class (e.g. JWSTNIRCamWide).

    This will store the downloaded instrument HDF5 file in the
    instruments_cache directory from which this class can load the instrument.
    The user is then still free to pass their own depth, depth aperture radius,
    SNRs, or explicit noise maps to be used.

    This class introduces no extra attributes or methods to the base
    Instrument class. It is simply a wrapper around the base class to
    provide a premade instrument with the properties of JWST's NIRCam
    instrument. For a list of attributes and methods see the Instrument class.
    """

    @classmethod
    def load(cls, filter_subset=(), filter_lams=None, **kwargs):
        """Load the instrument from a file.

        This method will attempt to load the instrument from a file. If the
        file does not exist, it will raise an error. The user can pass their
        own depth, depth aperture radius, SNRs, or explicit noise maps to be
        used.

        Args:
            filter_subset (list/tuple):
                A list of filters to include in the instrument. This should
                be a list of strings with the filter names. If this is not
                provided, all filters will be included.
            filter_lams (unyt_array, optional):
                The wavelength array of the filters (with units). This
                should be a unyt array with the same length as the number
                of filters. If this is not provided, the default wavelength
                array will be used.
            **kwargs: Keyword arguments to pass to the Instrument class. These
                include any argument to the Instrument class:
                    filters (FilterCollection, optional):
                        The filters of the Instrument. Default is None.
                    resolution (unyt_quantity, optional):
                        The resolution of the Instrument (with units). Default
                        is None.
                    lam (unyt_array, optional):
                        The wavelength array of the Instrument (with units).
                        Default is None.
                    depth (dict/float, optional):
                        The depth of the Instrument in apparent mags. If
                        filters are passed depth must be a dictionaries of
                        depths with and entry per filter. Default is None.
                    depth_app_radius (unyt_quantity, optional):
                        The depth aperture radius of the Instrument (with
                        units). If this is omitted but SNRs and depths are
                        provided, it is assumed that the depth is a point
                        source depth. Default is None.
                    snrs (unyt_array, optional):
                        The signal-to-noise ratios of the Instrument.
                        Default is None.
                    psfs (unyt_array, optional):
                        The PSFs of the Instrument. If doing imaging this
                        should be a dictionary of PSFs with an entry for each
                        filter. If doing resolved spectroscopy this should be
                        an array. Default is None.
                    noise_maps (unyt_array, optional):
                        The noise maps of the Instrument. If doing imaging
                        this should be a dictionary of noise maps with an
                        entry for each filter. If doing resolved spectroscopy
                        this should be an array with noise as a function of
                        wavelength. Default is None.

        """
        # Check if the instrument cache file exists and if not, raise an error
        # and suggest the user download it
        if not os.path.exists(cls._instrument_cache_file):
            raise exceptions.MissingInstrumentFile(
                f"Instrument cache file {cls._instrument_cache_file} "
                " does not exist. Please download the instrument using "
                " synthesizer-download --instrument <instrument_name>"
            )

        # Load the instrument from the file
        with h5py.File(cls._instrument_cache_file, "r") as hdf:
            # Get a new instance of the class
            inst = cls._from_hdf5(hdf, **kwargs)

        # Get the subset of filters if provided
        if len(filter_subset) > 0:
            inst.filters = inst.filters.select(*filter_subset)

        # Do we need to resample the filters?
        if filter_lams is not None:
            inst.filters.resample_filters(
                new_lam=filter_lams,
                verbose=False,
            )

        return inst

    def __init__(
        self,
        label="",
        filters=None,
        resolution=None,
        lam=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
    ):
        """Initialize the PremadeInstrument class.

        Args:
            label (str):
                The label of the instrument. Default is "".
            filters (FilterCollection):
                The filters of the instrument. Default is None.
            resolution (unyt_quantity):
                The resolution of the instrument (with units). Default is None.
            lam (unyt_array):
                The wavelength array of the instrument (with units). Default
                is None.
            depth (dict/float):
                The depth of the instrument in apparent mags. If filters are
                passed depth must be a dictionaries of depths with and entry
                per filter. Default is None.
            depth_app_radius (unyt_quantity):
                The depth aperture radius of the instrument (with units). If
                this is omitted but SNRs and depths are provided, it is assumed
                that the depth is a point source depth. Default is None.
            snrs (unyt_array):
                The signal-to-noise ratios of the instrument. Default is None.
            psfs (unyt_array):
                The PSFs of the instrument. If doing imaging this should be a
                dictionary of PSFs with an entry for each filter. If doing
                resolved spectroscopy this should be an array. Default is None.
            noise_maps (unyt_array):
                The noise maps of the instrument. If doing imaging this should
                be a dictionary of noise maps with an entry for each filter.
                If doing resolved spectroscopy this should be an array with
                noise as a function of wavelength. Default is None.

        """
        # Call the parent constructor
        Instrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=resolution,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
        )


class JWSTNIRCamWide(PremadeInstrument):
    """A class containing the properties of JWST's NIRCam instrument (Wide).

    Default label: "JWST.NIRCam.Wide"

    Resolution: 0.031"

    Available filters:
        - F070W
        - F090W
        - F115W
        - F150W
        - F150W2
        - F200W
        - F277W
        - F356W
        - F444W

    Note that we give every filter the short wavelength channel resolution
    as if the long wavelength channel has been resampled to match
    the higher resolution of the short wavelength channel.

    If this class is loaded from a file then it will also include PSFs for each
    filter, derived using WebbPSF (using the defaults arguments).

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/JWST_NIRCam_Wide.hdf5"

    def __init__(
        self,
        label="JWST.NIRCam.Wide",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
    ):
        """Initialize the JWST NIRCam Wide instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.NIRCam.Wide".
            filter_lams (unyt_array):
                An optional wavelength array to resample the filter
                transmission curves to. This should be a unyt array. Default
                is None.
            depth (dict/float):
                The depth of the instrument in apparent mags. If filters are
                passed depth must be a dictionary of depths with an entry
                per filter. Default is None.
            depth_app_radius (unyt_quantity):
                The depth aperture radius of the instrument (with units). If
                this is omitted but SNRs and depths are provided, it is assumed
                that the depth is a point source depth. Default is None.
            snrs (unyt_array):
                The signal-to-noise ratios of the instrument. Default is None.
            psfs (unyt_array):
                The PSFs of the instrument. If doing imaging this should be a
                dictionary of PSFs with an entry for each filter. If doing
                resolved spectroscopy this should be an array. Default is None.
            noise_maps (unyt_array):
                The noise maps of the instrument. If doing imaging this should
                be a dictionary of noise maps with an entry for each filter.
                If doing resolved spectroscopy this should be an array with
                noise as a function of wavelength. Default is None.
            filter_subset (list/tuple):
                A list of filters defining a subset of the filters to include
                in the instrument. This should be a list of strings with the
                filter names. If this is not provided, all filters defined in
                this class will be included.
        """
        # Create the list of SVO filters to get
        if len(filter_subset) > 0:
            filter_codes = filter_subset
        else:
            filter_codes = [
                "JWST/NIRCam.F070W",
                "JWST/NIRCam.F090W",
                "JWST/NIRCam.F115W",
                "JWST/NIRCam.F150W",
                "JWST/NIRCam.F150W2",
                "JWST/NIRCam.F200W",
                "JWST/NIRCam.F277W",
                "JWST/NIRCam.F356W",
                "JWST/NIRCam.F444W",
            ]

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        # NIRCam
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.031 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
        )


class JWSTNIRCamMedium(PremadeInstrument):
    """A class containing the properties of JWST's NIRCam instrument (Medium).

    Default label: "JWST.NIRCam.Medium"

    Resolution: 0.031"

    Available filters:
        - F140M
        - F162M
        - F182M
        - F210M
        - F250M
        - F300M
        - F335M
        - F360M
        - F410M
        - F430M
        - F460M
        - F480M

    Note that we give every filter the short wavelength channel resolution
    as if the long wavelength channel has been resampled to match
    the higher resolution of the short wavelength channel.

    If this class is loaded from a file then it will also include PSFs for each
    filter, derived using WebbPSF (using the defaults arguments).

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/JWST_NIRCam_Medium.hdf5"

    def __init__(
        self,
        label="JWST.NIRCam.Medium",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
    ):
        """Initialize the JWST NIRCam Medium instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.NIRCam.Medium".
            filter_lams (unyt_array):
                An optional wavelength array to resample the filter
                transmission curves to. This should be a unyt array. Default
                is None.
            depth (dict/float):
                The depth of the instrument in apparent mags. If filters are
                passed depth must be a dictionary of depths with an entry
                per filter. Default is None.
            depth_app_radius (unyt_quantity):
                The depth aperture radius of the instrument (with units). If
                this is omitted but SNRs and depths are provided, it is assumed
                that the depth is a point source depth. Default is None.
            snrs (unyt_array):
                The signal-to-noise ratios of the instrument. Default is None.
            psfs (unyt_array):
                The PSFs of the instrument. If doing imaging this should be a
                dictionary of PSFs with an entry for each filter. If doing
                resolved spectroscopy this should be an array. Default is None.
            noise_maps (unyt_array):
                The noise maps of the instrument. If doing imaging this should
                be a dictionary of noise maps with an entry for each filter.
                If doing resolved spectroscopy this should be an array with
                noise as a function of wavelength. Default is None.
            filter_subset (list/tuple):
                A list of filters defining a subset of the filters to include
                in the instrument. This should be a list of strings with the
                filter names. If this is not provided, all filters defined in
                this class will be included.
        """
        # Create the list of SVO filters to get
        if len(filter_subset) > 0:
            filter_codes = filter_subset
        else:
            filter_codes = [
                "JWST/NIRCam.F140M",
                "JWST/NIRCam.F162M",
                "JWST/NIRCam.F182M",
                "JWST/NIRCam.F210M",
                "JWST/NIRCam.F250M",
                "JWST/NIRCam.F300M",
                "JWST/NIRCam.F335M",
                "JWST/NIRCam.F360M",
                "JWST/NIRCam.F410M",
                "JWST/NIRCam.F430M",
                "JWST/NIRCam.F460M",
                "JWST/NIRCam.F480M",
            ]

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        # NIRCam
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.031 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
        )


class JWSTNIRCamNarrow(PremadeInstrument):
    """A class containing the properties of JWST's NIRCam instrument (Narrow).

    Default label: "JWST.NIRCam.Narrow"

    Resolution: 0.031"

    Available filters:
        - F164N
        - F187N
        - F212N
        - F323N
        - F405N
        - F466N
        - F470N

    Note that we give every filter the short wavelength channel resolution
    as if the long wavelength channel has been resampled to match
    the higher resolution of the short wavelength channel.

    If this class is loaded from a file then it will also include PSFs for each
    filter, derived using WebbPSF (using the defaults arguments).

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/JWST_NIRCam_Narrow.hdf5"

    def __init__(
        self,
        label="JWST.NIRCam.Narrow",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
    ):
        """Initialize the JWST NIRCam Narrow instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.NIRCam.Narrow".
            filter_lams (unyt_array):
                An optional wavelength array to resample the filter
                transmission curves to. This should be a unyt array. Default
                is None.
            depth (dict/float):
                The depth of the instrument in apparent mags. If filters are
                passed depth must be a dictionary of depths with an entry
                per filter. Default is None.
            depth_app_radius (unyt_quantity):
                The depth aperture radius of the instrument (with units). If
                this is omitted but SNRs and depths are provided, it is assumed
                that the depth is a point source depth. Default is None.
            snrs (unyt_array):
                The signal-to-noise ratios of the instrument. Default is None.
            psfs (unyt_array):
                The PSFs of the instrument. If doing imaging this should be a
                dictionary of PSFs with an entry for each filter. If doing
                resolved spectroscopy this should be an array. Default is None.
            noise_maps (unyt_array):
                The noise maps of the instrument. If doing imaging this should
                be a dictionary of noise maps with an entry for each filter.
                If doing resolved spectroscopy this should be an array with
                noise as a function of wavelength. Default is None.
            filter_subset (list/tuple):
                A list of filters defining a subset of the filters to include
                in the instrument. This should be a list of strings with the
                filter names. If this is not provided, all filters defined in
                this class will be included.
        """
        # Create the list of SVO filters to get
        if len(filter_subset) > 0:
            filter_codes = filter_subset
        else:
            filter_codes = [
                "JWST/NIRCam.F164N",
                "JWST/NIRCam.F187N",
                "JWST/NIRCam.F212N",
                "JWST/NIRCam.F323N",
                "JWST/NIRCam.F405N",
                "JWST/NIRCam.F466N",
                "JWST/NIRCam.F470N",
            ]

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        # NIRCam
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.031 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
        )
