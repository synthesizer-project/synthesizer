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


class Instrument:
    """Factory for constructing specialised instrument classes.

    This class is a convenience API. `Instrument(...)` returns one of the
    specialised instrument subclasses based on the supplied arguments.
    Malformed argument combinations are rejected explicitly.
    """

    def __new__(cls, *args, **kwargs):
        """Return the correct specialised instrument."""
        if cls is not Instrument:
            return super().__new__(cls)

        if args:
            raise exceptions.InconsistentArguments(
                "Instrument(...) only accepts keyword arguments. "
                "Pass values explicitly, for example "
                "Instrument(label='my_instrument', ...)."
            )

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

        if filters is not None and lam is None:
            if resolution is not None:
                from synthesizer.instruments.photometric_imager import (
                    PhotometricImager,
                )

                target_cls = PhotometricImager
            else:
                from synthesizer.instruments.photometric_instrument import (
                    PhotometricInstrument,
                )

                target_cls = PhotometricInstrument
        elif lam is not None and filters is None:
            if noise_source_maps is not None:
                raise exceptions.InconsistentArguments(
                    "Instrument(...) could not map the supplied arguments to "
                    "a supported concrete instrument type. Construct a "
                    "specialised class directly if you need a configuration "
                    "outside the supported photometric, imaging, "
                    "spectroscopic, or IFU cases. Received arguments: "
                    f"{present}"
                )

            if resolution is not None:
                from synthesizer.instruments.integrated_field_unit import (
                    IntegratedFieldUnit,
                )

                target_cls = IntegratedFieldUnit
            else:
                from synthesizer.instruments.spectroscopic_instrument import (
                    SpectroscopicInstrument,
                )

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
        from synthesizer.instruments.integrated_field_unit import (
            IntegratedFieldUnit,
        )
        from synthesizer.instruments.photometric_imager import (
            PhotometricImager,
        )
        from synthesizer.instruments.photometric_instrument import (
            PhotometricInstrument,
        )
        from synthesizer.instruments.spectroscopic_instrument import (
            SpectroscopicInstrument,
        )

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
