"""A module defining a class for observational instruments.

This module contains the `Instrument` class, which is used to define
observational instruments for use in the `synthesizer` package. The
`Instrument` class contains everything needed to define a telescope or
spectrograph, including the filters, resolution, wavelength array, depth,
depth aperture radius, signal-to-noise ratios, PSFs and noise_maps.

`Instrument` objects can define instruments to be used for synthetic:
    - Photometry (with or without noise)
    - Imaging (with or without PSFs and noise)
    - Spectroscopy (with or without noise)
    - Resolved Spectroscopy (with or without PSFs and noise)

Example usage:
    # Create an Instrument object
    instrument = Instrument(
        label="HST",
        filters=FilterCollection(...),
        resolution=0.1 * kpc,
        noise_maps={...},
        psfs={...},
        )
    print(instrument)
"""

import h5py
from unyt import angstrom, arcsecond, kpc, unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.instruments.filters import FilterCollection
from synthesizer.instruments.instrument_collection import InstrumentCollection
from synthesizer.units import Quantity, accepts
from synthesizer.utils.ascii_table import TableFormatter


class Instrument:
    """A class containing the properties defining an observational instrument.

    This class contains everything needed to define a telescope or
    spectrograph, including the filters, resolution, wavelength array,
    depth, depth aperture radius, signal-to-noise ratios, PSFs and noise_maps.

    `Instrument` objects can define instruments to be used for synthetic:
        - Photometry (with or without noise)
        - Imaging (with or without PSFs and noise)
        - Spectroscopy (with or without noise)
        - Resolved Spectroscopy (with or without PSFs and noise)

    Attributes:
        label (str):
            The label of the Instrument.
        filters (FilterCollection):
            The filters of the Instrument.
        resolution (unyt_array):
            The resolution of the Instrument.
        lam (unyt_array):
            The wavelength array of the Instrument.
        depth (unyt_array):
            The depth of the Instrument.
        depth_app_radius (unyt_array):
            The depth aperture radius of the Instrument.
        snrs (unyt_array):
            The signal-to-noise ratios of the Instrument.
        psfs (unyt_array):
            The PSFs of the Instrument.
        noise_maps (unyt_array):
            The noise maps of the Instrument.
    """

    # Define quantities
    lam = Quantity("wavelength")

    @accepts(resolution=(kpc, arcsecond), lam=angstrom)
    def __init__(
        self,
        label,
        filters=None,
        resolution=None,
        lam=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
    ):
        """Initialize an Instrument object.

        Args:
            label (str):
                The label for the Instrument, e.g. HST-WFC3, JWST-NIRCam, etc.
            filters (FilterCollection, optional):
                The filters of the Instrument. Default is None.
            resolution (unyt_quantity, optional):
                The resolution of the Instrument (with units). Default is None.
            lam (unyt_array, optional):
                The wavelength array of the Instrument (with units). Default
                is None.
            depth (dict/float, optional):
                The depth of the Instrument in apparent mags. If filters are
                passed depth must be a dictionaries of depths with and entry
                per filter. Default is None.
            depth_app_radius (unyt_quantity, optional):
                The depth aperture radius of the Instrument (with units).
                If this is omitted but SNRs and depths are provided, it is
                assumed that the depth is a point source depth.
                Default is None.
            snrs (unyt_array, optional):
                The signal-to-noise ratios of the Instrument.
                Default is None.
            psfs (unyt_array, optional):
                The PSFs of the Instrument. If doing imaging this should be
                a dictionary of PSFs with an entry for each filter. If doing
                resolved spectroscopy this should be an array.
                Default is None.
            noise_maps (unyt_array, optional):
                The noise maps of the Instrument. If doing imaging this should
                be a dictionary of noise maps with an entry for each filter.
                If doing resolved spectroscopy this should be an array with
                noise as a function of wavelength.
                Default is None.
        """
        # Set the label of the Instrument
        self.label = label

        # Set the filters of the Instrument (applicable for photometry and
        # imaging)
        self.filters = filters

        # Set the resolution of the Instrument (applicable for imaging)
        if (
            isinstance(resolution, (unyt_array, unyt_quantity))
            or resolution is None
        ):
            self.resolution = resolution
        else:
            raise exceptions.InconsistentArguments(
                "Resolution must have units."
            )

        # Set the wavelength array for the Instrument (applicable for
        # spectroscopy)
        self.lam = lam

        # Set the depth of the Instrument (applicable for imaging)
        self.depth = depth

        # Set the depth aperture radius of the Instrument (applicable for
        # imaging)
        self.depth_app_radius = depth_app_radius

        # Set the signal-to-noise ratio of the Instrument (applicable for
        # spectroscopy and imaging)
        self.snrs = snrs

        # Set the PSFs of the Instrument (applicable for imaging and resolved
        # spectroscopy)
        self.psfs = psfs

        # Set the noise maps of the Instrument (applicable for imaging and
        # resolved spectroscopy)
        self.noise_maps = noise_maps

    @property
    def _resolution(self):
        """Return the resolution of the Instrument without the units.

        This emulates the behaviour of a Quantity object but here we can
        have either angular or Cartesian resolutions. There's no need for
        any complex unit handling here so we instead just introduced
        this special case property.

        Returns:
            float:
                The resolution of the Instrument.
        """
        return None if self.resolution is None else self.resolution.value

    @property
    def can_do_photometry(self):
        """Return whether the Instrument do photometry.

        Returns:
            bool:
                Whether the Instrument do photometry.
        """
        return self.filters is not None

    @property
    def can_do_imaging(self):
        """Return whether the Instrument do simple imaging.

        This flags whether the Instrument do imaging basic
        imaging without PSFs and noise.

        Returns:
            bool:
                Whether the Instrument do simple imaging.
        """
        return self.can_do_photometry and self.resolution is not None

    @property
    def can_do_psf_imaging(self):
        """Return whether the Instrument do imaging with PSFs.

        Returns:
            bool:
                Whether the Instrument do imaging with PSFs.
        """
        return self.can_do_imaging and self.psfs is not None

    @property
    def can_do_noisy_imaging(self):
        """Return whether the Instrument do imaging with noise.

        This is a bit more complex than the other flags as it can be true
        for various different noise definitions.

        We ignore the depth aperature radius here since a depth and SNR
        without it is assumed to be a point source depth.

        Returns:
            bool:
                Whether the Instrument do imaging with noise.
        """
        # Check we have a compatible noise definition
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None

        return self.can_do_imaging and have_noise

    @property
    def can_do_spectroscopy(self):
        """Return whether the Instrument do spectroscopy.

        Returns:
            bool:
                Whether the Instrument do spectroscopy.
        """
        return self.lam is not None

    @property
    def can_do_noisy_spectroscopy(self):
        """Return whether the Instrument can do spectroscopy with noise.

        This is a bit more complex than the other flags as it can be true
        for various different noise definitions.

        We ignore the depth aperature radius here since a depth and SNR
        without it is assumed to be a point source depth.

        Returns:
            bool:
                Whether the Instrument do spectroscopy
                with noise.
        """
        # Check we have a compatible noise definition
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None

        return self.can_do_spectroscopy and have_noise

    @property
    def can_do_resolved_spectroscopy(self):
        """Return whether the Instrument do resolved spectroscopy.

        Returns:
            bool:
                Whether the Instrument do simple resolved
                spectroscopy.
        """
        return self.can_do_spectroscopy and self.resolution is not None

    @property
    def can_do_psf_spectroscopy(self):
        """Return whether the Instrument can do smoothed resolved spectroscopy.

        Returns:
            bool:
                Whether the Instrument do smoothed resolved
                spectroscopy.
        """
        return self.can_do_resolved_spectroscopy and self.psfs is not None

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """Return whether the Instrument can do noisy resolved spectroscopy.

        This is a bit more complex than the other flags as it can be true
        for various different noise definitions.

        We ignore the depth aperature radius here since a depth and SNR
        without it is assumed to be a point source depth.

        Returns:
            bool:
                Whether the Instrument do noisy resolved
                spectroscopy.
        """
        # Check we have a compatible noise definition
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None

        return self.can_do_resolved_spectroscopy and have_noise

    @classmethod
    def _from_hdf5(cls, group):
        """Create an Instrument from an HDF5 group.

        Args:
            group (h5py.Group):
                The group containing the Instrument attributes.

        Returns:
            Instrument:
                The Instrument created from the HDF5 group.
        """
        # Unpack the filters if they are present
        if "Filters" in group:
            filters = FilterCollection._from_hdf5(group["Filters"])
        else:
            filters = None

        #  Unpack the resolution
        if "Resolution" in group:
            resolution = unyt_array(
                group["Resolution"][...], group["Resolution"].attrs["units"]
            )
        else:
            resolution = None

        # Unpack the wavelenths
        if "Wavelength" in group:
            lam = unyt_array(
                group["Wavelength"][...], group["Wavelength"].attrs["units"]
            )
        else:
            lam = None

        # Unpack the depths, these can be either a group of datasets, a
        # single dataset or not present
        if "Depth" in group and isinstance(group["Depth"], h5py.Group):
            depth = {
                key: unyt_array(value[...], value.attrs["units"])
                for key, value in group["Depth"].items()
            }
        elif "Depth" in group:
            depth = unyt_array(
                group["Depth"][...], group["Depth"].attrs["units"]
            )
        else:
            depth = None

        # Unpack the depth aperture radius
        if "DepthApertureRadius" in group:
            depth_app_radius = unyt_array(
                group["DepthApertureRadius"][...],
                group["DepthApertureRadius"].attrs["units"],
            )
        else:
            depth_app_radius = None

        # Unpack the SNRs, these can be either a group of datasets, a
        # single dataset or not present
        if "SNRs" in group and isinstance(group["SNRs"], h5py.Group):
            snrs = {
                key: unyt_array(value[...], value.attrs["units"])
                for key, value in group["SNRs"].items()
            }
        elif "SNRs" in group:
            snrs = unyt_array(group["SNRs"][...], group["SNRs"].attrs["units"])
        else:
            snrs = None

        # Unpack the PSFs, these can be either a group of datasets, a
        # single dataset or not present
        if "PSFs" in group and isinstance(group["PSFs"], h5py.Group):
            psfs = {}
            for key in group["PSFs"]:
                # We can also have a group of datasets or a single dataset here
                if isinstance(group["PSFs"][key], h5py.Group):
                    for subkey in group["PSFs"][key]:
                        psfs[f"{key}/{subkey}"] = unyt_array(
                            group["PSFs"][key][subkey][...],
                            group["PSFs"][key][subkey].attrs["units"],
                        )
                else:
                    psfs[key] = unyt_array(
                        group["PSFs"][key][...],
                        group["PSFs"][key].attrs["units"],
                    )
        elif "PSFs" in group:
            psfs = unyt_array(group["PSFs"][...], group["PSFs"].attrs["units"])
        else:
            psfs = None

        # Unpack the noise maps, these can be either a group of datasets, a
        # single dataset or not present
        if "NoiseMaps" in group and isinstance(group["NoiseMaps"], h5py.Group):
            noise_maps = {
                key: unyt_array(value[...], value.attrs["units"])
                for key, value in group["NoiseMaps"].items()
            }
        elif "NoiseMaps" in group:
            noise_maps = unyt_array(
                group["NoiseMaps"][...], group["NoiseMaps"].attrs["units"]
            )
        else:
            noise_maps = None

        # Create the Instrument
        return cls(
            label=group.attrs["label"],
            filters=filters,
            resolution=resolution,
            lam=lam,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            psfs=psfs,
            noise_maps=noise_maps,
        )

    def to_hdf5(self, group):
        """Save the Instrument to an HDF5 group.

        Args:
            group (h5py.Group):
                The group in which to save the Instrument.

        """
        # Write out the label, technically pointless because the group is
        # named with the label but makes loading easier
        group.attrs["label"] = self.label

        # Write out the filters into a Filters group
        if self.filters is not None:
            filters_group = group.create_group("Filters")
            self.filters._write_filters_to_group(filters_group)

        # Write out the simple datasets
        if self.resolution is not None:
            ds = group.create_dataset(
                "Resolution", data=self.resolution.value, dtype=float
            )
            ds.attrs["units"] = str(self.resolution.units)
        if self.lam is not None:
            ds = group.create_dataset(
                "Wavelength", data=self.lam.value, dtype=float
            )
            ds.attrs["units"] = str(self.lam.units)
        if self.depth_app_radius is not None:
            ds = group.create_dataset(
                "DepthApertureRadius",
                data=self.depth_app_radius.value,
                dtype=float,
            )
            ds.attrs["units"] = str(self.depth_app_radius.units)

        # Write out the depth, SNRs, PSFs and noise which may be a group of
        # datasets or a single datasets or not present
        if self.depth is not None:
            if isinstance(self.depth, dict):
                depth_group = group.create_group("Depth")
                for key, value in self.depth.items():
                    ds = depth_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = "dimensionless"
            else:
                ds = group.create_dataset(
                    "Depth", data=self.depth.value, dtype=float
                )
                ds.attrs["units"] = "dimensionless"

        if self.snrs is not None:
            if isinstance(self.snrs, dict):
                snrs_group = group.create_group("SNRs")
                for key, value in self.snrs.items():
                    ds = snrs_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = "dimensionless"
            else:
                ds = group.create_dataset(
                    "SNRs", data=self.snrs.value, dtype=float
                )
                ds.attrs["units"] = "dimensionless"

        if self.psfs is not None:
            if isinstance(self.psfs, dict):
                psfs_group = group.create_group("PSFs")
                for key, value in self.psfs.items():
                    ds = psfs_group.create_dataset(
                        key, data=value, dtype=float
                    )
                    ds.attrs["units"] = "dimensionless"
            else:
                ds = group.create_dataset("PSFs", data=self.psfs, dtype=float)
                ds.attrs["units"] = "dimensionless"

        if self.noise_maps is not None:
            if isinstance(self.noise_maps, dict):
                noise_group = group.create_group("NoiseMaps")
                for key, value in self.noise_maps.items():
                    ds = noise_group.create_dataset(
                        key, data=value.value, dtype=float
                    )
                    ds.attrs["units"] = str(value.units)
            else:
                ds = group.create_dataset(
                    "NoiseMaps", data=self.noise_maps.value, dtype=float
                )
                ds.attrs["units"] = str(self.noise_maps.units)

    def __str__(self):
        """Return a string representation of the Instrument.

        Returns:
               str:
                   The string representation of the Instrument.
        """
        formatter = TableFormatter(self)

        return formatter.get_table("Instrument")

    def __add__(self, other):
        """Combine two Instruments into an Instrument Collection.

        Note, the combined instruments must have different labels, combining
        two instruments together into a new single instrument would be
        ill-defined since the only thing that could differ would be the
        filters, in which case the new filters should be added to the
        filter collection itself.

        Args:
            other (Instrument):
                The Instrument to combine with this one.

        Returns:
            InstrumentCollection:
                The Instrument Collection containing the two Instruments.
        """
        # Ensure other is an Instrument or InstrumentCollection
        if not isinstance(other, (Instrument, InstrumentCollection)):
            raise exceptions.InconsistentAddition(
                f"Cannot combine Instrument with {type(other)}."
            )

        # If we have an instrument collection just use the instrument
        # collection add method
        if isinstance(other, InstrumentCollection):
            return other + self

        # Check if the labels are the same
        if self.label == other.label:
            raise exceptions.InconsistentAddition(
                "Adding two instruments with the same label is ill-defined. "
                "If you want to add extra filters to an instrument, use the "
                "add_filters method."
            )

        # Create a new instrument Collection
        collection = InstrumentCollection()

        # Add the two instruments to the Collection
        collection.add_instruments(self, other)

        return collection

    def add_filters(self, filters, psfs=None, noise_maps=None):
        """Add filters to the Instrument.

        If PSFs or noise maps are provided, an entry for each new filter
        must be provided in a dict passed to the psfs or noise_maps
        arguments.

        Args:
            filters (FilterCollection):
                The filters to add to the Instrument.
            psfs (dict, optional):
                The PSFs for the new filters. Default is None.
            noise_maps (dict, optional):
                The noise maps for the new filters. Default is None.
        """
        # Combine the filters together
        self.filters += filters

        # Ensure we have an entry for each filter code in the psfs and
        # noise_maps
        if psfs is not None and set(psfs.keys()) != set(filters.filter_codes):
            raise exceptions.InconsistentAddition(
                "PSFs missing for filters: "
                f"{set(filters.filter_codes) - set(psfs.keys())}"
            )
        if noise_maps is not None and set(noise_maps.keys()) != set(
            filters.filter_codes
        ):
            raise exceptions.InconsistentAddition(
                "Noise maps missing for filters: "
                f"{set(filters.filter_codes) - set(noise_maps.keys())}"
            )

        # If PSFs are provided, add them to the psfs
        if psfs is not None:
            self.psfs.update(psfs)

        # If noise maps are provided, add them to the noise noise_maps
        if noise_maps is not None:
            self.noise_maps.update(noise_maps)
