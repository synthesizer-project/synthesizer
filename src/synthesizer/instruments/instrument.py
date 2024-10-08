""""""

from unyt import angstrom, kpc

from synthesizer import exceptions
from synthesizer.instruments.instrument_collection import InstrumentCollection
from synthesizer.units import Quantity, accepts
from synthesizer.utils.ascii_table import TableFormatter


class Instrument:
    """"""

    # Define quantities
    resoluton = Quantity()
    lam = Quantity()

    @accepts(resolution=kpc, lam=angstrom)
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
        """"""
        # Set the label of the Instrument
        self.label = label

        # Set the filters of the Instrument (applicable for photometry and
        # imaging)
        self.filters = filters

        # Set the resolution of the Instrument (applicable for imaging)
        self.resolution = resolution

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
    def can_do_photometry(self):
        """
        Return whether the Instrument can be used for photometry.

        Returns:
            bool:
                Whether the Instrument can be used for photometry.
        """
        return self.filters is not None

    @property
    def can_do_imaging(self):
        """
        Return whether the Instrument can be used for simple imaging.

        This flags whether the Instrument can be used for imaging basic
        imaging without PSFs and noise.

        Returns:
            bool:
                Whether the Instrument can be used for simple imaging.
        """
        return self.can_do_photometry and self.resolution is not None

    @property
    def can_do_psf_imaging(self):
        """
        Return whether the Instrument can be used for imaging with PSFs.

        Returns:
            bool:
                Whether the Instrument can be used for imaging with PSFs.
        """
        return self.can_do_imaging and self.psfs is not None

    @property
    def can_do_noisy_imaging(self):
        """
        Return whether the Instrument can be used for imaging with noise.

        This is a bit more complex than the other flags as it can be true
        for various different noise definitions.

        We ignore the depth aperature radius here since a depth and SNR
        without it is assumed to be a point source depth.

        Returns:
            bool:
                Whether the Instrument can be used for imaging with noise.
        """
        # Check we have a compatible noise definition
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None

        return self.can_do_imaging and have_noise

    @property
    def can_do_spectroscopy(self):
        """
        Return whether the Instrument can be used for spectroscopy.

        Returns:
            bool:
                Whether the Instrument can be used for spectroscopy.
        """
        return self.lam is not None

    @property
    def can_do_noisy_spectroscopy(self):
        """
        Return whether the Instrument can be used for spectroscopy with noise.

        This is a bit more complex than the other flags as it can be true
        for various different noise definitions.

        We ignore the depth aperature radius here since a depth and SNR
        without it is assumed to be a point source depth.

        Returns:
            bool:
                Whether the Instrument can be used for spectroscopy
                with noise.
        """
        # Check we have a compatible noise definition
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None

        return self.can_do_spectroscopy and have_noise

    @property
    def can_do_resolved_spectroscopy(self):
        """
        Return whether the Instrument can be used for resolved spectroscopy.

        Returns:
            bool:
                Whether the Instrument can be used for simple resolved
                spectroscopy.
        """
        return self.can_do_spectroscopy and self.resolution is not None

    @property
    def can_do_psf_spectroscopy(self):
        """
        Return whether the Instrument can do smoothed resolved spectroscopy.

        Returns:
            bool:
                Whether the Instrument can be used for smoothed resolved
                spectroscopy.
        """
        return self.can_do_resolved_spectroscopy and self.psfs is not None

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """
        Return whether the Instrument can do noisy resolved spectroscopy.

        This is a bit more complex than the other flags as it can be true
        for various different noise definitions.

        We ignore the depth aperature radius here since a depth and SNR
        without it is assumed to be a point source depth.

        Returns:
            bool:
                Whether the Instrument can be used for noisy resolved
                spectroscopy.
        """
        # Check we have a compatible noise definition
        have_noise = self.noise_maps is not None
        have_noise |= self.snrs is not None and self.depth is not None

        return self.can_do_resolved_spectroscopy and have_noise

    @classmethod
    def from_hdf5(cls, group):
        """
        Create an Instrument from an HDF5 group.

        Args:
            group (h5py.Group):
                The group containing the Instrument attributes.

        Returns:
            Instrument:
                The Instrument created from the HDF5 group.
        """
        raise exceptions.NotImplementedError(
            "Loading an Instrument from an HDF5 file is not yet implemented."
        )

    def to_hdf5(self, group):
        """
        Save the Instrument to an HDF5 group.

        Args:
            group (h5py.Group):
                The group in which to save the Instrument.

        """
        raise exceptions.NotImplementedError(
            "Saving an Instrument from an HDF5 file is not yet implemented."
        )

    def __str__(self):
        """
        Return a string representation of the Instrument.

           Returns:
               str:
                   The string representation of the Instrument.
        """
        formatter = TableFormatter(self)

        return formatter.get_table("Instrument")

    def __add__(self, other):
        """
        Combine two Instruments into an Instrument Collection.

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
        """
        Add filters to the Instrument.

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
