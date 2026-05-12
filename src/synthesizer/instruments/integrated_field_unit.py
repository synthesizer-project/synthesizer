"""Specialised Integrated Field Unit instrument.

This instrument is designed to hold the attributes required by resolved
spectroscopy. It extends :class:`SpectroscopicInstrument` with spatial
resolution, optional PSFs, and noise definitions.
"""

import h5py
from unyt import angstrom, arcsecond, kpc, unyt_array

from synthesizer import exceptions
from synthesizer.instruments.instrument_base import _hashable_state
from synthesizer.instruments.spectroscopic_instrument import (
    SpectroscopicInstrument,
)
from synthesizer.units import accepts
from synthesizer.utils.operation_timers import timed


class IntegratedFieldUnit(SpectroscopicInstrument):
    """Integrated Field Unit instrument class.

    A class containing the attributes and methods required to produce resolved
    spectroscopy. It extends :class:`SpectroscopicInstrument` with spatial
    resolution, optional PSFs, and noise definitions.

    Attributes:
        resolution (unyt_array): The spatial resolution of the instrument, in
            kpc or arcseconds.
        psfs (array): An optional array with spatial point spread functions as
            a function of wavelength, in dimensionless units. If a 2D array is
            supplied, every wavelength is assumed to have the same PSF. If
            a 3D array is supplied, the last axis must be the wavelength axis.
    """

    @accepts(
        resolution=(kpc, arcsecond),
        lam=angstrom,
        depth_app_radius=(kpc, arcsecond),
    )
    @timed("IntegratedFieldUnit.__init__")
    def __init__(
        self,
        label,
        lam,
        resolution,
        psfs=None,
        depth=None,
        depth_app_radius=None,
        snrs=None,
        noise_maps=None,
    ):
        """Initialise an integrated field unit instrument.

        Args:
            label (str): A label for the instrument.
            lam (unyt_array): The wavelength array defining the spectral
                coverage of the instrument.
            resolution (unyt_array): The spatial resolution of the instrument,
                in kpc or arcseconds.
            psfs (array, optional): An optional array with spatial point spread
                functions as a function of wavelength, in dimensionless units.
                If a 2D array is supplied, every wavelength is assumed to have
                the same PSF. If a 3D array is supplied, the last axis must be
                the wavelength axis.
            depth (unyt_quantity, optional): The depth of the instrument, in
                the same units as the image surface brightness.
            depth_app_radius (unyt_quantity, optional): The aperture radius for
                the depth measurement, in resolution units.
            snrs (unyt_quantity, optional): The signal-to-noise ratio of the
                instrument, in dimensionless units.
            noise_maps (array, optional): An optional array with noise map as
                a function of wavelength, in the same units as the image noise.
                If a 2D array is supplied, every wavelength is assumed to have
                the same noise map. If a 3D array is supplied, the last axis
                must be the wavelength axis.
        """
        # Initialise the shared spectroscopic instrument first
        super().__init__(
            label=label,
            lam=lam,
            depth=depth,
            depth_app_radius=depth_app_radius,
            snrs=snrs,
            noise_maps=noise_maps,
        )

        # Attach the IFU specific attributes
        self.resolution = resolution
        self.psfs = psfs

        # Ensure we have been handed the correct information
        self._validate()

    @timed("IntegratedFieldUnit._validate")
    def _validate(self):
        """Validate the instrument attributes.

        Raises:
            MissingArgument: If any required attributes are missing.
        """
        # Perform the shared validation first
        super()._validate()

        # Ensure we actually have the resolution... otherwise we are not
        # really an IFU!
        if self.resolution is None:
            raise exceptions.MissingArgument(
                "IntegratedFieldUnit requires a resolution."
            )

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

    @timed("IntegratedFieldUnit._comparison_state")
    def _comparison_state(self):
        """Return a tuple describing the IFU comparison state.

        Returns:
            tuple: Hashable representation of the instrument state.
        """
        return super()._comparison_state() + (
            _hashable_state(self.resolution),
            _hashable_state(self.psfs),
        )

    @timed("IntegratedFieldUnit.to_hdf5")
    def to_hdf5(self, group):
        """Write the integrated field unit to an HDF5 group.

        Args:
            group (h5py.Group): Group into which the instrument should be
                serialised.
        """
        super().to_hdf5(group)

        ds = group.create_dataset(
            "Resolution", data=self.resolution.value, dtype=float
        )
        ds.attrs["units"] = str(self.resolution.units)

        if self.psfs is not None:
            ds = group.create_dataset("PSFs", data=self.psfs, dtype=float)
            ds.attrs["units"] = "dimensionless"

    @classmethod
    @timed("IntegratedFieldUnit.load")
    def load(cls, filepath, **kwargs):
        """Load an integrated field unit from an HDF5 file.

        Args:
            filepath (str or PathLike): Path to the HDF5 file.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            IntegratedFieldUnit: The loaded instrument.
        """
        with h5py.File(filepath, "r") as hdf:
            return cls._from_hdf5(hdf, **kwargs)

    @classmethod
    @timed("IntegratedFieldUnit._from_hdf5")
    def _from_hdf5(cls, group, **kwargs):
        """Load an integrated field unit from an HDF5 group.

        Args:
            group (h5py.Group): Group containing the serialised instrument.
            **kwargs: Attribute overrides applied after deserialisation.

        Returns:
            IntegratedFieldUnit: The loaded instrument.
        """
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
