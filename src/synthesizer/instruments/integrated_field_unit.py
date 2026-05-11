"""Specialised integrated field unit instrument."""

import h5py
from unyt import arcsecond, kpc, unyt_array

from synthesizer import exceptions
from synthesizer.instruments.instrument_base import _hashable_state
from synthesizer.instruments.spectroscopic_instrument import (
    SpectroscopicInstrument,
)
from synthesizer.units import accepts


class IntegratedFieldUnit(SpectroscopicInstrument):
    """Concrete instrument for resolved spectroscopic configurations.

    This specialisation extends `SpectroscopicInstrument` with spatial
    resolution and optional PSFs for integral-field or other resolved
    spectroscopic use cases.
    """

    @accepts(resolution=(kpc, arcsecond))
    def __init__(
        self,
        label,
        lam,
        resolution,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        psfs=None,
        noise_maps=None,
    ):
        """Initialise an integrated field unit instrument."""
        super().__init__(
            label=label,
            lam=lam,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            noise_maps=noise_maps,
        )
        self.resolution = resolution
        self.psfs = psfs
        self._validate()

    @property
    def instrument_type(self):
        """Return the serialised type tag for this instrument."""
        return "ifu"

    @property
    def can_do_resolved_spectroscopy(self):
        """Return whether this instrument supports resolved spectroscopy."""
        return True

    @property
    def can_do_psf_spectroscopy(self):
        """Return whether this instrument supports PSF spectroscopy."""
        return self.psfs is not None

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """Return whether this instrument supports noisy IFU work."""
        return self.can_do_noisy_spectroscopy

    def _validate(self):
        super()._validate()
        if self.resolution is None:
            raise exceptions.MissingArgument(
                "IntegratedFieldUnit requires a resolution."
            )

    def _comparison_state(self):
        return super()._comparison_state() + (
            _hashable_state(self.resolution),
            _hashable_state(self.psfs),
        )

    def to_hdf5(self, group):
        """Write the integrated field unit to an HDF5 group."""
        super().to_hdf5(group)

        ds = group.create_dataset(
            "Resolution", data=self.resolution.value, dtype=float
        )
        ds.attrs["units"] = str(self.resolution.units)

        if self.psfs is not None:
            ds = group.create_dataset("PSFs", data=self.psfs, dtype=float)
            ds.attrs["units"] = "dimensionless"

    @classmethod
    def load(cls, filepath, **kwargs):
        """Load an integrated field unit from an HDF5 file."""
        with h5py.File(filepath, "r") as hdf:
            return cls._from_hdf5(hdf, **kwargs)

    @classmethod
    def _from_hdf5(cls, group, **kwargs):
        lam = unyt_array(
            group["Wavelength"][...], group["Wavelength"].attrs["units"]
        )
        resolution = unyt_array(
            group["Resolution"][...], group["Resolution"].attrs["units"]
        )

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

        if "PSFs" in group:
            psfs = unyt_array(group["PSFs"][...], group["PSFs"].attrs["units"])
        else:
            psfs = None

        if "NoiseMaps" in group:
            noise_maps = unyt_array(
                group["NoiseMaps"][...], group["NoiseMaps"].attrs["units"]
            )
        else:
            noise_maps = None

        payload = {
            "label": group.attrs["label"],
            "lam": lam,
            "resolution": resolution,
            "depth": depth,
            "depth_app_radius": depth_app_radius,
            "snrs": snrs,
            "psfs": psfs,
            "noise_maps": noise_maps,
        }
        payload.update(kwargs)

        return cls(
            label=payload["label"],
            lam=payload["lam"],
            resolution=payload["resolution"],
            depth=payload["depth"],
            depth_app_radius=payload["depth_app_radius"],
            snrs=payload["snrs"],
            psfs=payload["psfs"],
            noise_maps=payload["noise_maps"],
        )
