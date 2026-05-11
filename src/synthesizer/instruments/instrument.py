"""Generic Instrument factory entry point.

`Instrument` is a generic factory which will instantiate the correct
instrument type based on the supplied arguments. Configurations that do not
map cleanly onto one of these concrete classes are rejected explicitly.

Use of the `Instrument` factory is optional, and users can construct the
specific instrument type directly if they know which configuration they need.

The `Instrument` is maintained for backwards compatibility with the original
implementation of instruments and may one day be deprecated and removed.
"""

from synthesizer import exceptions
from synthesizer.instruments.integrated_field_unit import IntegratedFieldUnit
from synthesizer.instruments.photometric_imager import PhotometricImager
from synthesizer.instruments.photometric_instrument import (
    PhotometricInstrument,
)
from synthesizer.instruments.spectroscopic_instrument import (
    SpectroscopicInstrument,
)


class Instrument:
    """Factory for constructing specialised instrument classes.

    This class is a convenience API. `Instrument(...)` returns one of the
    specialised instrument subclasses based on the supplied arguments.
    Malformed argument combinations are rejected explicitly.
    """

    def __new__(cls, *args, **kwargs):
        """Return the correct specialised instrument.

        Args:
            *args: Positional arguments are not supported by the factory
                and will be rejected explicitly. All arguments must be
                passed as keyword arguments.
            **kwargs: Keyword arguments which are used to determine the correct
                specialised instrument type to construct. Supported arguments
                are:
                - label (str): A label for the instrument.
                - filters (list of str): A list of filter names for photometric
                    instruments.
                - resolution (float): The spectral resolution for spectroscopic
                    instruments, or the spatial resolution for imaging
                    instruments.
                - lam (float): The central wavelength for spectroscopic
                    instruments, or the central wavelength for photometric
                    instruments without filters.
                - depth (float): The depth of the instrument in magnitudes.
                - depth_app_radius (float): The aperture radius for the depth
                    measurement in arcseconds.
                - snrs (dict): A mapping from filter names to SNR values for
                    photometric instruments.
                - psfs (dict): A mapping from filter names to PSF models for
                    photometric instruments.
                - noise_maps (dict): A mapping from filter names to noise maps
                    for photometric instruments.
                - noise_source_maps (dict): A mapping from filter names to
                    noise source maps for photometric instruments with
                    correlated noise.

        Returns:
            InstrumentBase: An instance of a specialised instrument subclass
                inheriting from `InstrumentBase`, constructed according to
                the supplied arguments.
        """
        # Kwargs are safer for the factory, let the user know if they supplied
        # any positional arguments by mistake
        if len(args) > 0:
            raise exceptions.InconsistentArguments(
                "Instrument(...) only accepts keyword arguments. "
                "Pass values explicitly, for example "
                "Instrument(label='my_instrument', ...)."
            )

        # Unpack all the supported arguments
        label = kwargs.get("label", None)
        filters = kwargs.get("filters", None)
        resolution = kwargs.get("resolution", None)
        lam = kwargs.get("lam", None)
        depth = kwargs.get("depth", None)
        depth_app_radius = kwargs.get("depth_app_radius", None)
        snrs = kwargs.get("snrs", None)
        psfs = kwargs.get("psfs", None)
        noise_maps = kwargs.get("noise_maps", None)
        noise_source_maps = kwargs.get("noise_source_maps", None)

        # Get a list of what we were provided for reporting in error messages
        # if needs be (no point doing )
        present = sorted(
            key
            for key, value in {
                "label": label,
                "filters": filters,
                "resolution": resolution,
                "lam": lam,
                "depth": depth,
                "depth_app_radius": depth_app_radius,
                "snrs": snrs,
                "psfs": psfs,
                "noise_maps": noise_maps,
                "noise_source_maps": noise_source_maps,
            }.items()
            if value is not None
        )

        # Resolve the correct instrument type based on the supplied arguments
        target_cls = None

        # Photometric imaging case
        if filters is not None and lam is None and resolution is not None:
            target_cls = PhotometricImager

        # Photometric filters case
        elif filters is not None and lam is None and resolution is None:
            target_cls = PhotometricInstrument

        # Spectroscopic Integrated Field Unit case
        elif filters is None and lam is not None and resolution is not None:
            target_cls = IntegratedFieldUnit

        # Spectroscopic case
        elif filters is None and lam is not None and resolution is None:
            target_cls = SpectroscopicInstrument

        else:
            raise exceptions.InconsistentArguments(
                "Instrument(...) could not map the supplied arguments to a "
                "supported concrete instrument type. Construct a specialised "
                "class directly if you need a configuration outside the "
                "supported photometric, imaging, spectroscopic, or IFU "
                f"cases. Received arguments: {present}"
            )

        return target_cls(**kwargs)

    @classmethod
    def _from_hdf5(cls, group, **kwargs):
        """Dispatch HDF5 loading to the appropriate specialised class."""
        instrument_type = group.attrs.get("instrument_type")

        if instrument_type == "photometric":
            return PhotometricInstrument._from_hdf5(group, **kwargs)
        if instrument_type == "photometric_imager":
            return PhotometricImager._from_hdf5(group, **kwargs)
        if instrument_type == "spectroscopic":
            return SpectroscopicInstrument._from_hdf5(group, **kwargs)
        if instrument_type == "ifu":
            return IntegratedFieldUnit._from_hdf5(group, **kwargs)

        raise exceptions.InconsistentArguments(
            "Unsupported instrument_type "
            f"'{instrument_type}' in serialized instrument."
        )
