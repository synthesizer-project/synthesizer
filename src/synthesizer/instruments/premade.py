"""A submodule defining some premade instruments.

This module contains some commonly used instruments which can be imported,
instantiated, and used directly. Each premade instrument is a child of the
Instrument class with the appropriate properties fixed.

The premade instruments are:
    - JWST/NIRCam.Wide
    - JWST/NIRCam.Medium
    - JWST/NIRCam.Narrow
    - JWST/NIRCam
    - JWST/MIRI
    - HST/WFC3.UVIS.Wide
    - HST/WFC3.UVIS.Medium
    - HST/WFC3.UVIS.Narrow
    - HST/WFC3.UVI
    - HST/WFC3.IR.Wide
    - HST/WFC3.IR.Medium
    - HST/WFC3.IR.Narrow
    - HST/WFC3.IR
    - HST/ACS_WFC.Wide
    - HST/ACS_WFC.Medium
    - HST/ACS_WFC.Narrow
    - HST/ACS_WFC
    - Euclid/NISP
    - Euclid/VIS

Example usage:

    # Get NIRCam Wide instrument
    jwst_nircam = JWSTNIRCam()

    # Get MIRI with a custom wavelength array
    miri = JWSTMIRI(filter_lams=lams)

    # Get HST WFC3 UVIS with a custom depth
    wfc3_uvis = HSTWFC3UVISWide(depth=30.0, snrs=10.0)

    # Get HST WFC3 IR with a subset of filters
    wfc3_ir = HSTWFC3IRWide(filter_subset=
        [
            "HST/WFC3.IR.F110W",
            "HST/WFC3.IR.F160W",
        ]
    )

    # Load a premade instrument from a file
    jwst_nircam = JWSTNIRCam.load()

    # Load a premade instrument from a file with a custom wavelength array
    jwst_nircam = JWSTNIRCam.load(filter_lams=lams)

    # Load a premade instrument from a file with a custom noise array
    jwst_nircam = JWSTNIRCam.load(noise_maps=noise_arrays)

"""

import os

import h5py
import numpy as np
from unyt import angstrom, arcsecond

from synthesizer import INSTRUMENT_CACHE_DIR, exceptions
from synthesizer.instruments import Instrument, InstrumentCollection
from synthesizer.instruments.filters import Filter, FilterCollection

__all__ = [
    "JWSTNIRCamWide",
    "JWSTNIRCamMedium",
    "JWSTNIRCamNarrow",
    "JWSTNIRCam",
    "JWSTMIRI",
    "HSTWFC3UVISWide",
    "HSTWFC3UVISMedium",
    "HSTWFC3UVISNarrow",
    "HSTWFC3UVIS",
    "HSTWFC3IRWide",
    "HSTWFC3IRMedium",
    "HSTWFC3IRNarrow",
    "HSTWFC3IR",
    "HSTACSWFCWide",
    "HSTACSWFCMedium",
    "HSTACSWFCNarrow",
    "HSTACSWFC",
    "EuclidNISP",
    "EuclidVIS",
    "GALEX",
]


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
        **kwargs,
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Call the parent constructor
        Instrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=resolution,
            lam=lam,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class PremadeInstrumentCollectionFactory:
    """Base class for factory classes that create instrument collections.

    This class provides a common pattern for creating
    InstrumentCollections that contain multiple related instruments
    (e.g., GALEX FUV and NUV).

    Child classes should define an `instruments` class attribute as a
    dictionary mapping instrument names to instrument classes. Kwargs
    for individual instruments can be passed as `<name>_kwargs`.

    Example:
        ```python
        class GALEX(PremadeInstrumentCollectionFactory):
            instruments = {
                "fuv": GALEXFUV,
                "nuv": GALEXNUV,
            }

        # Create a GALEX collection with custom kwargs for each instrument
        galex = GALEX(fuv_kwargs={"depth": 25.0}, nuv_kwargs={"depth": 26.0})
        ```
    """

    _is_premade_factory = True
    instruments = {}  # Override in child classes

    def __new__(cls, **kwargs):
        """Create an InstrumentCollection with all instruments.

        Args:
            **kwargs: Keyword arguments in the form `<name>_kwargs` where
                <name> matches a key in the `instruments` class attribute.
                Each `<name>_kwargs` should be a dictionary of keyword
                arguments to pass to that instrument's constructor.

        Returns:
            InstrumentCollection: Collection containing all instruments
                defined in the `instruments` class attribute.
        """
        collection = InstrumentCollection()

        instruments_list = []
        for name, inst_cls in cls.instruments.items():
            inst_kwargs = kwargs.get(f"{name}_kwargs", {})
            if inst_kwargs is None:
                inst_kwargs = {}
            instruments_list.append(inst_cls(**inst_kwargs))

        collection.add_instruments(*instruments_list)

        return collection


class JWSTNIRCamWide(PremadeInstrument):
    """A class containing JWST's NIRCam instrument (Wide).

    Default label: "JWST.NIRCam.Wide"

    Resolution: 0.031"

    Available filters:
        - F070W
        - F090W
        - F115W
        - F150W
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

    # Define the available filters
    available_filters = [
        "JWST/NIRCam.F070W",
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F444W",
    ]

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
        **kwargs,
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

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
            **kwargs,
        )


class JWSTNIRCamMedium(PremadeInstrument):
    """A class containing JWST's NIRCam instrument (Medium).

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

    # Define the available filters
    available_filters = [
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
        **kwargs,
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

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
            **kwargs,
        )


class JWSTNIRCamNarrow(PremadeInstrument):
    """A class containing JWST's NIRCam instrument (Narrow).

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

    # Define the available filters
    available_filters = [
        "JWST/NIRCam.F164N",
        "JWST/NIRCam.F187N",
        "JWST/NIRCam.F212N",
        "JWST/NIRCam.F323N",
        "JWST/NIRCam.F405N",
        "JWST/NIRCam.F466N",
        "JWST/NIRCam.F470N",
    ]

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
        **kwargs,
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

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
            **kwargs,
        )


class JWSTNIRCam(PremadeInstrument):
    """A class containing JWST's NIRCam instrument.

    Default label: "JWST.NIRCam"

    Resolution: 0.031"

    Available filters:
        - F070W
        - F090W
        - F115W
        - F140M
        - F150W
        - F162M
        - F164N
        - F182M
        - F187N
        - F200W
        - F210M
        - F212N
        - F250M
        - F277W
        - F300M
        - F323N
        - F335M
        - F356W
        - F360M
        - F405N
        - F410M
        - F430M
        - F444W
        - F460M
        - F466N
        - F470N
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
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/JWST_NIRCam.hdf5"

    # Define the available filters
    available_filters = [
        "JWST/NIRCam.F070W",
        "JWST/NIRCam.F090W",
        "JWST/NIRCam.F115W",
        "JWST/NIRCam.F140M",
        "JWST/NIRCam.F150W",
        "JWST/NIRCam.F162M",
        "JWST/NIRCam.F164N",
        "JWST/NIRCam.F182M",
        "JWST/NIRCam.F187N",
        "JWST/NIRCam.F200W",
        "JWST/NIRCam.F210M",
        "JWST/NIRCam.F212N",
        "JWST/NIRCam.F250M",
        "JWST/NIRCam.F277W",
        "JWST/NIRCam.F300M",
        "JWST/NIRCam.F323N",
        "JWST/NIRCam.F335M",
        "JWST/NIRCam.F356W",
        "JWST/NIRCam.F360M",
        "JWST/NIRCam.F405N",
        "JWST/NIRCam.F410M",
        "JWST/NIRCam.F430M",
        "JWST/NIRCam.F444W",
        "JWST/NIRCam.F460M",
        "JWST/NIRCam.F466N",
        "JWST/NIRCam.F470N",
        "JWST/NIRCam.F480M",
    ]

    def __init__(
        self,
        label="JWST.NIRCam",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the JWST NIRCam instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.NIRCam".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
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
            **kwargs,
        )


class JWSTNIRSpec(PremadeInstrument):
    """A placeholder for JWST's NIRSpec instrument.

    This class will be implemented in a future update.
    """

    # TODO: Implement NIRSpec class with appropriate filters, resolution, etc.
    pass


class JWSTMIRI(PremadeInstrument):
    """A class containing JWST's MIRI instrument.

    Default label: "JWST.MIRI"

    Resolution: 0.11"

    Available filters:
        - F560W
        - F770W
        - F1000W
        - F1065C
        - F1130W
        - F1140C
        - F1280W
        - F1500W
        - F1550C
        - F1800W
        - F2100W
        - F2300C
        - F2550W


    If this class is loaded from a file then it will also include PSFs for each
    filter, derived using WebbPSF (using the defaults arguments).

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/JWST_MIRI.hdf5"

    # Define the available filters
    available_filters = [
        "JWST/MIRI.F560W",
        "JWST/MIRI.F770W",
        "JWST/MIRI.F1000W",
        "JWST/MIRI.F1065C",
        "JWST/MIRI.F1130W",
        "JWST/MIRI.F1140C",
        "JWST/MIRI.F1280W",
        "JWST/MIRI.F1500W",
        "JWST/MIRI.F1550C",
        "JWST/MIRI.F1800W",
        "JWST/MIRI.F2100W",
        "JWST/MIRI.F2300C",
        "JWST/MIRI.F2550W",
    ]

    def __init__(
        self,
        label="JWST.MIRI",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the JWST MIRI instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.11 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3UVISWide(PremadeInstrument):
    """A class containing HST's WFC3/UVIS instrument (Wide).

    Default label: "HST.WFC3.UVIS.Wide"

    Resolution: 0.04"

    Available filters:
        - F218W
        - F225W
        - F275W
        - F336W
        - F350LP
        - F390W
        - F438W
        - F475W
        - F555W
        - F606W
        - F625W
        - F775W
        - F814W
        - F850LP

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_UVIS_Wide.hdf5"

    # Define the available filters
    available_filters = [
        "HST/WFC3_UVIS1.F218W",
        "HST/WFC3_UVIS1.F225W",
        "HST/WFC3_UVIS1.F275W",
        "HST/WFC3_UVIS1.F336W",
        "HST/WFC3_UVIS1.F350LP",
        "HST/WFC3_UVIS1.F390W",
        "HST/WFC3_UVIS1.F438W",
        "HST/WFC3_UVIS1.F475W",
        "HST/WFC3_UVIS1.F555W",
        "HST/WFC3_UVIS1.F606W",
        "HST/WFC3_UVIS1.F625W",
        "HST/WFC3_UVIS1.F775W",
        "HST/WFC3_UVIS1.F814W",
        "HST/WFC3_UVIS1.F850LP",
    ]

    def __init__(
        self,
        label="HST.WFC3.UVIS.Wide",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/UVIS Wide instrument.

        Args:
            label (str):
                The label of the instrument. Default is "HST.WFC3.UVIS.Wide".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.04 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3UVISMedium(PremadeInstrument):
    """A class containing HST's WFC3/UVIS instrument (Medium).

    Default label: "HST.WFC3.UVIS.Medium"

    Resolution: 0.04"

    Available filters:
        - F410M
        - F467M
        - F547M
        - F621M
        - F689M
        - F763M
        - F845M

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = (
        f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_UVIS_Medium.hdf5"
    )

    # Define the available filters
    available_filters = [
        "HST/WFC3_UVIS1.F410M",
        "HST/WFC3_UVIS1.F467M",
        "HST/WFC3_UVIS1.F547M",
        "HST/WFC3_UVIS1.F621M",
        "HST/WFC3_UVIS1.F689M",
        "HST/WFC3_UVIS1.F763M",
        "HST/WFC3_UVIS1.F845M",
    ]

    def __init__(
        self,
        label="HST.WFC3.UVIS.Medium",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/UVIS Medium instrument.

        Args:
            label (str):
                The label of the instrument. Default is "HST.WFC3.UVIS.Medium".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.04 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3UVISNarrow(PremadeInstrument):
    """A class containing HST's WFC3/UVIS instrument (Narrow).

    Default label: "HST.WFC3.UVIS.Narrow"

    Resolution: 0.04"

    Available filters:
        - F280N
        - F343N
        - F373N
        - F395N
        - F469N
        - F487N
        - F502N
        - F631N
        - F645N
        - F656N
        - F657N
        - F658N
        - F665N
        - F673N
        - F680N
        - F953N
        - FQ232N
        - FQ243N
        - FQ378N
        - FQ387N
        - FQ422M
        - FQ436N
        - FQ437N
        - FQ492N
        - FQ508N
        - FQ575N
        - FQ619N
        - FQ634N
        - FQ672N
        - FQ674N
        - FQ727N
        - FQ750N
        - FQ889N
        - FQ906N
        - FQ924N
        - FQ937N

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = (
        f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_UVIS_Narrow.hdf5"
    )

    # Define the available filters
    available_filters = [
        "HST/WFC3_UVIS1.F280N",
        "HST/WFC3_UVIS1.F343N",
        "HST/WFC3_UVIS1.F373N",
        "HST/WFC3_UVIS1.F395N",
        "HST/WFC3_UVIS1.F469N",
        "HST/WFC3_UVIS1.F487N",
        "HST/WFC3_UVIS1.F502N",
        "HST/WFC3_UVIS1.F631N",
        "HST/WFC3_UVIS1.F645N",
        "HST/WFC3_UVIS1.F656N",
        "HST/WFC3_UVIS1.F657N",
        "HST/WFC3_UVIS1.F658N",
        "HST/WFC3_UVIS1.F665N",
        "HST/WFC3_UVIS1.F673N",
        "HST/WFC3_UVIS1.F680N",
        "HST/WFC3_UVIS1.F953N",
        "HST/WFC3_UVIS1.FQ232N",
        "HST/WFC3_UVIS1.FQ243N",
        "HST/WFC3_UVIS1.FQ378N",
        "HST/WFC3_UVIS1.FQ387N",
        "HST/WFC3_UVIS1.FQ422M",
        "HST/WFC3_UVIS1.FQ436N",
        "HST/WFC3_UVIS1.FQ437N",
        "HST/WFC3_UVIS1.FQ492N",
        "HST/WFC3_UVIS1.FQ508N",
        "HST/WFC3_UVIS1.FQ575N",
        "HST/WFC3_UVIS1.FQ619N",
        "HST/WFC3_UVIS1.FQ634N",
        "HST/WFC3_UVIS1.FQ672N",
        "HST/WFC3_UVIS1.FQ674N",
        "HST/WFC3_UVIS1.FQ727N",
        "HST/WFC3_UVIS1.FQ750N",
        "HST/WFC3_UVIS1.FQ889N",
        "HST/WFC3_UVIS1.FQ906N",
        "HST/WFC3_UVIS1.FQ924N",
        "HST/WFC3_UVIS1.FQ937N",
    ]

    def __init__(
        self,
        label="HST.WFC3.UVIS.Narrow",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/UVIS Narrow instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.04 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3UVIS(PremadeInstrument):
    """A class containing HST's WFC3/UVIS instrument.

    Default label: "HST.WFC3.UVIS"

    Resolution: 0.04"

    Available filters:
        - F218W
        - F225W
        - F275W
        - F336W
        - F350LP
        - F390W
        - F438W
        - F475W
        - F555W
        - F606W
        - F625W
        - F775W
        - F814W
        - F850LP
        - F410M
        - F467M
        - F547M
        - F621M
        - F689M
        - F763M
        - F845M
        - F280N
        - F343N
        - F373N
        - F395N
        - F469N
        - F487N
        - F502N
        - F631N
        - F645N
        - F656N
        - F657N
        - F658N
        - F665N
        - F673N
        - F680N
        - F953N
        - FQ232N
        - FQ243N
        - FQ378N
        - FQ387N
        - FQ422M
        - FQ436N
        - FQ437N
        - FQ492N
        - FQ508N
        - FQ575N
        - FQ619N
        - FQ634N
        - FQ672N
        - FQ674N
        - FQ727N
        - FQ750N
        - FQ889N
        - FQ906N
        - FQ924N
        - FQ937N


    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_UVIS.hdf5"

    # Define the available filters
    available_filters = [
        # Wide
        "HST/WFC3_UVIS1.F218W",
        "HST/WFC3_UVIS1.F225W",
        "HST/WFC3_UVIS1.F275W",
        "HST/WFC3_UVIS1.F336W",
        "HST/WFC3_UVIS1.F350LP",
        "HST/WFC3_UVIS1.F390W",
        "HST/WFC3_UVIS1.F438W",
        "HST/WFC3_UVIS1.F475W",
        "HST/WFC3_UVIS1.F555W",
        "HST/WFC3_UVIS1.F606W",
        "HST/WFC3_UVIS1.F625W",
        "HST/WFC3_UVIS1.F775W",
        "HST/WFC3_UVIS1.F814W",
        "HST/WFC3_UVIS1.F850LP",
        # Medium
        "HST/WFC3_UVIS1.F410M",
        "HST/WFC3_UVIS1.F467M",
        "HST/WFC3_UVIS1.F547M",
        "HST/WFC3_UVIS1.F621M",
        "HST/WFC3_UVIS1.F689M",
        "HST/WFC3_UVIS1.F763M",
        "HST/WFC3_UVIS1.F845M",
        # Narrow
        "HST/WFC3_UVIS1.F280N",
        "HST/WFC3_UVIS1.F343N",
        "HST/WFC3_UVIS1.F373N",
        "HST/WFC3_UVIS1.F395N",
        "HST/WFC3_UVIS1.F469N",
        "HST/WFC3_UVIS1.F487N",
        "HST/WFC3_UVIS1.F502N",
        "HST/WFC3_UVIS1.F631N",
        "HST/WFC3_UVIS1.F645N",
        "HST/WFC3_UVIS1.F656N",
        "HST/WFC3_UVIS1.F657N",
        "HST/WFC3_UVIS1.F658N",
        "HST/WFC3_UVIS1.F665N",
        "HST/WFC3_UVIS1.F673N",
        "HST/WFC3_UVIS1.F680N",
        "HST/WFC3_UVIS1.F953N",
        "HST/WFC3_UVIS1.FQ232N",
        "HST/WFC3_UVIS1.FQ243N",
        "HST/WFC3_UVIS1.FQ378N",
        "HST/WFC3_UVIS1.FQ387N",
        "HST/WFC3_UVIS1.FQ422M",
        "HST/WFC3_UVIS1.FQ436N",
        "HST/WFC3_UVIS1.FQ437N",
        "HST/WFC3_UVIS1.FQ492N",
        "HST/WFC3_UVIS1.FQ508N",
        "HST/WFC3_UVIS1.FQ575N",
        "HST/WFC3_UVIS1.FQ619N",
        "HST/WFC3_UVIS1.FQ634N",
        "HST/WFC3_UVIS1.FQ672N",
        "HST/WFC3_UVIS1.FQ674N",
        "HST/WFC3_UVIS1.FQ727N",
        "HST/WFC3_UVIS1.FQ750N",
        "HST/WFC3_UVIS1.FQ889N",
        "HST/WFC3_UVIS1.FQ906N",
        "HST/WFC3_UVIS1.FQ924N",
        "HST/WFC3_UVIS1.FQ937N",
    ]

    def __init__(
        self,
        label="HST.WFC3.UVIS",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/UVIS instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.04 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3IRWide(PremadeInstrument):
    """A class containing the properties of HST's WFC3/IR instrument (Wide).

    Default label: "HST.WFC3.IR.Wide"

    Resolution: 0.13"

    Available filters:
        - F105W
        - F110W
        - F125W
        - F140W
        - F160W

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_IR_Wide.hdf5"

    # Define the available filters
    available_filters = [
        "HST/WFC3_IR.F105W",
        "HST/WFC3_IR.F110W",
        "HST/WFC3_IR.F125W",
        "HST/WFC3_IR.F140W",
        "HST/WFC3_IR.F160W",
    ]

    def __init__(
        self,
        label="HST.WFC3.IR.Wide",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/IR Wide instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.13 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3IRMedium(PremadeInstrument):
    """A class containing the properties of HST's WFC3/IR instrument (Medium).

    Default label: "HST.WFC3.IR.Medium"

    Resolution: 0.13"

    Available filters:
        - F098M
        - F127M
        - F139M
        - F153M

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_IR_Medium.hdf5"

    # Define the available filters
    available_filters = [
        "HST/WFC3_IR.F098M",
        "HST/WFC3_IR.F127M",
        "HST/WFC3_IR.F139M",
        "HST/WFC3_IR.F153M",
    ]

    def __init__(
        self,
        label="HST.WFC3.IR.Medium",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/IR Medium instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.13 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3IRNarrow(PremadeInstrument):
    """A class containing the properties of HST's WFC3/IR instrument (Narrow).

    Default label: "HST.WFC3.IR.Narrow"

    Resolution: 0.13"

    Available filters:
        - F126N
        - F128N
        - F130N
        - F132N
        - F164N
        - F167N

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_IR_Narrow.hdf5"

    # Define the available filters
    available_filters = [
        "HST/WFC3_IR.F126N",
        "HST/WFC3_IR.F128N",
        "HST/WFC3_IR.F130N",
        "HST/WFC3_IR.F132N",
        "HST/WFC3_IR.F164N",
        "HST/WFC3_IR.F167N",
    ]

    def __init__(
        self,
        label="HST.WFC3.IR.Narrow",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/IR Narrow instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.13 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTWFC3IR(PremadeInstrument):
    """A class containing the properties of HST's WFC3/IR instrument.

    Default label: "HST.WFC3.IR"

    Resolution: 0.13"

    Available filters:
        - F098M
        - F105W
        - F110W
        - F125W
        - F126N
        - F127M
        - F128N
        - F130N
        - F132N
        - F139M
        - F140W
        - F153M
        - F160W
        - F164N
        - F167N

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_WFC3_IR.hdf5"

    # Define the available filters
    available_filters = [
        "HST/WFC3_IR.F098M",
        "HST/WFC3_IR.F105W",
        "HST/WFC3_IR.F110W",
        "HST/WFC3_IR.F125W",
        "HST/WFC3_IR.F126N",
        "HST/WFC3_IR.F127M",
        "HST/WFC3_IR.F128N",
        "HST/WFC3_IR.F130N",
        "HST/WFC3_IR.F132N",
        "HST/WFC3_IR.F139M",
        "HST/WFC3_IR.F140W",
        "HST/WFC3_IR.F153M",
        "HST/WFC3_IR.F160W",
        "HST/WFC3_IR.F164N",
        "HST/WFC3_IR.F167N",
    ]

    def __init__(
        self,
        label="HST.WFC3.IR",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST WFC3/IR instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.13 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTACSWFCWide(PremadeInstrument):
    """A class containing the properties of HST's ACS/WFC instrument (Wide).

    Default label: "HST.ACS.WFC.Wide"

    Resolution: 0.05"

    Available filters:
        - F435W
        - F475W
        - F555W
        - F606W
        - F625W
        - F775W
        - F814W
        - F850LP

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_ACS_WFC_Wide.hdf5"

    # Define the available filters
    available_filters = [
        "HST/ACS_WFC.F435W",
        "HST/ACS_WFC.F475W",
        "HST/ACS_WFC.F555W",
        "HST/ACS_WFC.F606W",
        "HST/ACS_WFC.F625W",
        "HST/ACS_WFC.F775W",
        "HST/ACS_WFC.F814W",
        "HST/ACS_WFC.F850LP",
    ]

    def __init__(
        self,
        label="HST.ACS.WFC.Wide",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST ACS/WFC Wide instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.05 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTACSWFCMedium(PremadeInstrument):
    """A class containing the properties of HST's ACS/WFC instrument (Medium).

    Default label: "HST.ACS.WFC.Medium"

    Resolution: 0.05"

    Available filters:
        - F550M

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_ACS_WFC_Medium.hdf5"

    # Define the available filters
    available_filters = [
        "HST/ACS_WFC.F550M",
    ]

    def __init__(
        self,
        label="HST.ACS.WFC.Medium",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST ACS/WFC Medium instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.05 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTACSWFCNarrow(PremadeInstrument):
    """A class containing the properties of HST's ACS/WFC instrument (Narrow).

    Default label: "HST.ACS.WFC.Narrow"

    Resolution: 0.05"

    Available filters:
        - F502N
        - F658N
        - F660N

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_ACS_WFC_Narrow.hdf5"

    # Define the available filters
    available_filters = [
        "HST/ACS_WFC.F502N",
        "HST/ACS_WFC.F658N",
        "HST/ACS_WFC.F660N",
    ]

    def __init__(
        self,
        label="HST.ACS.WFC.Narrow",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST ACS/WFC Narrow instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.05 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class HSTACSWFC(PremadeInstrument):
    """A class containing the properties of HST's ACS/WFC instrument.

    Default label: "HST.ACS.WFC"

    Resolution: 0.05"

    Available filters:
        - F435W
        - F475W
        - F502N
        - F550M
        - F555W
        - F606W
        - F625W
        - F658N
        - F660N
        - F775W
        - F814W
        - F850LP

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/HST_ACS_WFC.hdf5"

    # Define the available filters
    available_filters = [
        "HST/ACS_WFC.F435W",
        "HST/ACS_WFC.F475W",
        "HST/ACS_WFC.F502N",
        "HST/ACS_WFC.F550M",
        "HST/ACS_WFC.F555W",
        "HST/ACS_WFC.F606W",
        "HST/ACS_WFC.F625W",
        "HST/ACS_WFC.F658N",
        "HST/ACS_WFC.F660N",
        "HST/ACS_WFC.F775W",
        "HST/ACS_WFC.F814W",
        "HST/ACS_WFC.F850LP",
    ]

    def __init__(
        self,
        label="HST.ACS.WFC",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the HST ACS/WFC instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.05 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class EuclidNISP(PremadeInstrument):
    """A class containing the properties of Euclid's NISP instrument.

    Default label: "Euclid.NISP"

    Resolution: 0.30"

    Available filters:
        - Y
        - J
        - H

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/Euclid_NISP.hdf5"

    # Define the available filters
    available_filters = [
        "Euclid/NISP.Y",
        "Euclid/NISP.J",
        "Euclid/NISP.H",
    ]

    def __init__(
        self,
        label="Euclid.NISP",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the Euclid NISP instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.30 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class EuclidVIS(PremadeInstrument):
    """A class containing the properties of Euclid's VIS instrument.

    Default label: "Euclid.VIS"

    Resolution: 0.10"

    Available filters:
        - VIS

    For further details see the PremadeInstrument class.
    """

    # Define the filepath to the instrument cache file (it will be here
    # after downloading)
    _instrument_cache_file = f"{INSTRUMENT_CACHE_DIR}/Euclid_VIS.hdf5"

    # Define the available filters
    available_filters = [
        "Euclid/VIS.vis",
    ]

    def __init__(
        self,
        label="Euclid.VIS",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        filter_subset=(),
        **kwargs,
    ):
        """Initialize the Euclid VIS instrument.

        Args:
            label (str):
                The label of the instrument. Default is "JWST.MIRI".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Create the list of SVO filters to get
        filter_codes = filter_subset or self.available_filters

        # Get the filters from SVO
        filters = FilterCollection(
            filter_codes=filter_codes,
            new_lam=filter_lams,
        )

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=0.10 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class GALEXFUV(PremadeInstrument):
    """A class containing the properties of the GALEX instrument.

    Default label: "GALEXFUV"

    Resolution: 6"

    Available filters:
        - FUV

    For further details see the PremadeInstrument class.
    """

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError(
            "GALEXFUV does not support loading from an instrument cache file. "
            "Please instantiate it directly using GALEXFUV(...)."
        )

    def __init__(
        self,
        label="GALEXFUV",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        **kwargs,
    ):
        """Initialize the GALEX FUV instrument.

        Args:
            label (str):
                The label of the instrument. Default is "GALEXFUV".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Since SVO returns effective area rather than transmission curves
        # for GALEX, here we define the transmission curves manually rather
        # than using SVO
        fuv_lam = (
            np.array(
                [
                    1340.6205,
                    1350.4851,
                    1370.2143,
                    1399.808,
                    1449.131,
                    1477.0806,
                    1500.098,
                    1519.8272,
                    1549.4209,
                    1608.6084,
                    1648.0668,
                    1705.6102,
                    1750.0008,
                    1810.8324,
                ]
            )
            * angstrom
        )

        fuv_trans = np.array(
            [
                9.07e-07,
                0.11537571,
                0.17650714,
                0.12312477,
                0.33665425,
                0.36851147,
                0.35043035,
                0.34698632,
                0.26174673,
                0.25485868,
                0.16014802,
                0.11881973,
                0.10504364,
                0.000860099,
            ]
        )

        fuv_filter = Filter(
            "GALEX/GALEX.FUV", transmission=fuv_trans, new_lam=fuv_lam
        )

        filters = FilterCollection(filters=[fuv_filter])

        # Resample if requested
        if filter_lams is not None:
            filters.resample_filters(new_lam=filter_lams, verbose=False)

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=6 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class GALEXNUV(PremadeInstrument):
    """A class containing the properties of the GALEX instrument.

    Default label: "GALEXNUV"

    Resolution: 8"

    Available filters:
        - NUV

    For further details see the PremadeInstrument class.
    """

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError(
            "GALEXNUV does not support loading from an instrument cache file. "
            "Please instantiate it directly using GALEXNUV(...)."
        )

    def __init__(
        self,
        label="GALEXNUV",
        filter_lams=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        **kwargs,
    ):
        """Initialize the GALEX NUV instrument.

        Args:
            label (str):
                The label of the instrument. Default is "GALEXNUV".
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
            **kwargs: Keyword arguments to pass to the Instrument class.
        """
        # Since SVO returns effective area rather than transmission curves for
        # GALEX, here we define the transmission curves manually rather than
        # using SVO
        nuv_lam = (
            np.array(
                [
                    1687.5251,
                    1699.0338,
                    1748.3567,
                    1837.138,
                    1899.6137,
                    1948.9366,
                    1998.2595,
                    2050.8707,
                    2151.1606,
                    2200.4835,
                    2253.0946,
                    2300.7735,
                    2348.4523,
                    2445.4541,
                    2553.9645,
                    2595.0669,
                    2646.0339,
                    2697.001,
                    2797.2909,
                    2849.902,
                    2897.5809,
                    2999.5149,
                    3007.7354,
                ]
            )
            * angstrom
        )

        nuv_trans = np.array(
            [
                9.07e-07,
                0.024109075,
                0.031858129,
                0.16100903,
                0.26519076,
                0.3289052,
                0.44858503,
                0.4718322,
                0.59495605,
                0.6164812,
                0.56223782,
                0.5303806,
                0.47785924,
                0.53468563,
                0.51488249,
                0.50024539,
                0.46236113,
                0.38056556,
                0.11020967,
                0.033580141,
                0.012054991,
                0.016360021,
                9.07e-07,
            ]
        )

        nuv_filter = Filter(
            "GALEX/GALEX.NUV", transmission=nuv_trans, new_lam=nuv_lam
        )

        filters = FilterCollection(filters=[nuv_filter])

        # Resample if requested
        if filter_lams is not None:
            filters.resample_filters(new_lam=filter_lams, verbose=False)

        # Call the parent constructor with the appropriate parameters
        PremadeInstrument.__init__(
            self,
            label=label,
            filters=filters,
            resolution=8 * arcsecond,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
            **kwargs,
        )


class GALEX(PremadeInstrumentCollectionFactory):
    """Factory class returning a GALEX InstrumentCollection.

    Contains FUV and NUV instruments.

    This allows the following:

    from synthesizer.instruments import GALEX
    galex = GALEX()
    for inst in galex:
        print(inst.label, inst.resolution)

    which would return:

    GALEXFUV 6 arcsec
    GALEXNUV 8 arcsec

    For further details see the PremadeInstrumentCollectionFactory class.
    """

    instruments = {
        "fuv": GALEXFUV,
        "nuv": GALEXNUV,
    }
