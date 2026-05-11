"""Compatibility instrument entry point and dispatcher.

`Instrument` is retained as the generic user-facing constructor for backwards
compatibility. Construction is dispatched to the most appropriate concrete
instrument type based on the supplied arguments:

- `PhotometricInstrument`
- `PhotometricImager`
- `SpectroscopicInstrument`
- `IntegratedFieldUnit`

Configurations that do not map cleanly onto one of these concrete classes are
rejected explicitly.

Users who want a specific concrete type can construct the specialised classes
directly instead of going through this dispatcher.
"""

from inspect import signature

import h5py
from unyt import unyt_array

from synthesizer import exceptions
from synthesizer.instruments.filters import FilterCollection


def _bind_constructor_args(args, kwargs):
    def _signature_probe(
        label,
        filters=None,
        resolution=None,
        lam=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
        noise_source_maps=None,
    ):
        return None

    bound = signature(_signature_probe).bind_partial(*args, **kwargs)
    return bound.arguments


def _unsupported_configuration(arguments):
    """Raise a consistent error for unsupported instrument payloads."""
    present = sorted(
        key
        for key, value in arguments.items()
        if key != "self" and value is not None
    )
    raise exceptions.InconsistentArguments(
        "Instrument(...) could not map the supplied arguments to a supported "
        "concrete instrument type. Construct a specialised class directly if "
        "you need a configuration outside the supported photometric, imaging, "
        "spectroscopic, or IFU cases. Received arguments: "
        f"{present}"
    )


def unpack_instrument_payload(group, **kwargs):
    """Read common instrument fields from an HDF5 group."""
    if "Filters" in group:
        filters = FilterCollection._from_hdf5(group["Filters"])
    else:
        filters = None

    if "Resolution" in group:
        resolution = unyt_array(
            group["Resolution"][...], group["Resolution"].attrs["units"]
        )
    else:
        resolution = None

    if "Wavelength" in group:
        lam = unyt_array(
            group["Wavelength"][...], group["Wavelength"].attrs["units"]
        )
    else:
        lam = None

    if "Depth" in group and isinstance(group["Depth"], h5py.Group):
        depth = {
            key: unyt_array(value[...], value.attrs["units"])
            for key, value in group["Depth"].items()
        }
    elif "Depth" in group:
        depth = unyt_array(group["Depth"][...], group["Depth"].attrs["units"])
    else:
        depth = None

    if "DepthApertureRadius" in group:
        depth_app_radius = unyt_array(
            group["DepthApertureRadius"][...],
            group["DepthApertureRadius"].attrs["units"],
        )
    else:
        depth_app_radius = None

    if "SNRs" in group and isinstance(group["SNRs"], h5py.Group):
        snrs = {
            key: unyt_array(value[...], value.attrs["units"])
            for key, value in group["SNRs"].items()
        }
    elif "SNRs" in group:
        snrs = unyt_array(group["SNRs"][...], group["SNRs"].attrs["units"])
    else:
        snrs = None

    if "PSFs" in group and isinstance(group["PSFs"], h5py.Group):
        psfs = {}
        for key in group["PSFs"]:
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

    if "NoiseSourceMaps" in group and isinstance(
        group["NoiseSourceMaps"], h5py.Group
    ):
        noise_source_maps = {
            key: unyt_array(value[...], value.attrs["units"])
            for key, value in group["NoiseSourceMaps"].items()
        }
    else:
        noise_source_maps = None

    payload = {
        "label": group.attrs["label"],
        "filters": filters,
        "resolution": resolution,
        "lam": lam,
        "depth": depth,
        "depth_app_radius": depth_app_radius,
        "snrs": snrs,
        "psfs": psfs,
        "noise_maps": noise_maps,
        "noise_source_maps": noise_source_maps,
    }
    payload.update(kwargs)
    return payload


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

    if instrument_type not in (None, ""):
        raise exceptions.InconsistentArguments(
            "Unsupported instrument_type "
            f"'{instrument_type}' in serialized instrument."
        )

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
            _unsupported_configuration(arguments)

        if have_resolution:
            from synthesizer.instruments.integrated_field_unit import (
                IntegratedFieldUnit,
            )

            return IntegratedFieldUnit

        from synthesizer.instruments.spectroscopic_instrument import (
            SpectroscopicInstrument,
        )

        return SpectroscopicInstrument

    _unsupported_configuration(arguments)


class Instrument:
    """Backwards-compatible instrument factory.

    This class is a convenience API rather than part of the concrete
    instrument hierarchy. In normal use `Instrument(...)` returns one of the
    concrete specialised instrument classes.
    """

    def __new__(cls, *args, **kwargs):
        """Dispatch construction to the appropriate concrete instrument."""
        target_cls = _resolve_instrument_class(*args, **kwargs)
        return target_cls(*args, **kwargs)

    @classmethod
    def _from_hdf5(cls, group, **kwargs):
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

        target_cls = _resolve_instrument_class(
            instrument_type=group.attrs.get("instrument_type")
        )

        if target_cls is PhotometricInstrument:
            return PhotometricInstrument._from_hdf5(group, **kwargs)
        if target_cls is PhotometricImager:
            return PhotometricImager._from_hdf5(group, **kwargs)
        if target_cls is SpectroscopicInstrument:
            return SpectroscopicInstrument._from_hdf5(group, **kwargs)
        if target_cls is IntegratedFieldUnit:
            return IntegratedFieldUnit._from_hdf5(group, **kwargs)

        raise exceptions.InconsistentArguments(
            "Instrument._from_hdf5 could not resolve a supported concrete "
            "instrument class from the serialized metadata."
        )
