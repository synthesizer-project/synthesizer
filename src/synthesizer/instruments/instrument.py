"""Compatibility instrument entry point and dispatcher."""

from inspect import signature

from synthesizer.instruments.generic_instrument import (
    GenericInstrument,
    unpack_instrument_payload,
)


def _bind_constructor_args(args, kwargs):
    bound = signature(GenericInstrument.__init__).bind_partial(
        None, *args, **kwargs
    )
    return bound.arguments


def _resolve_instrument_class(*args, instrument_type=None, **kwargs):
    if instrument_type == "photometric":
        from synthesizer.instruments.photometric_instrument import (
            PhotometricInstrument,
        )

        return PhotometricInstrument

    if instrument_type == "photometric_imager":
        from synthesizer.instruments.photometric_imager import (
            PhotometricImager,
        )

        return PhotometricImager

    if instrument_type == "spectroscopic":
        from synthesizer.instruments.spectroscopic_instrument import (
            SpectroscopicInstrument,
        )

        return SpectroscopicInstrument

    if instrument_type == "ifu":
        from synthesizer.instruments.integrated_field_unit import (
            IntegratedFieldUnit,
        )

        return IntegratedFieldUnit

    arguments = _bind_constructor_args(args, kwargs)
    have_filters = arguments.get("filters") is not None
    have_lam = arguments.get("lam") is not None
    have_resolution = arguments.get("resolution") is not None

    if have_filters and not have_lam:
        if have_resolution:
            from synthesizer.instruments.photometric_imager import (
                PhotometricImager,
            )

            return PhotometricImager

        from synthesizer.instruments.photometric_instrument import (
            PhotometricInstrument,
        )

        return PhotometricInstrument

    if have_lam and not have_filters:
        if arguments.get("noise_source_maps") is not None:
            return GenericInstrument

        if have_resolution:
            from synthesizer.instruments.integrated_field_unit import (
                IntegratedFieldUnit,
            )

            return IntegratedFieldUnit

        from synthesizer.instruments.spectroscopic_instrument import (
            SpectroscopicInstrument,
        )

        return SpectroscopicInstrument

    return GenericInstrument


class Instrument(GenericInstrument):
    """Backwards-compatible instrument factory."""

    def __new__(cls, *args, **kwargs):
        """Dispatch construction to the appropriate concrete instrument."""
        if cls is not Instrument:
            return super().__new__(cls)

        target_cls = _resolve_instrument_class(*args, **kwargs)
        return target_cls(*args, **kwargs)

    @classmethod
    def _from_hdf5(cls, group, **kwargs):
        target_cls = _resolve_instrument_class(
            instrument_type=group.attrs.get("instrument_type")
        )
        if target_cls is GenericInstrument:
            payload = unpack_instrument_payload(group, **kwargs)
            target_cls = _resolve_instrument_class(**payload)

        return target_cls._from_hdf5(group, **kwargs)
