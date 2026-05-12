"""Tests for instrument and instrument collection behaviour."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from unyt import angstrom, arcsecond

from synthesizer import exceptions
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer.components.component import Component
from synthesizer.instruments import (
    GALEX,
    FilterCollection,
    Instrument,
    InstrumentCollection,
    IntegratedFieldUnit,
    PhotometricImager,
    PhotometricInstrument,
    SpectroscopicInstrument,
)
from synthesizer.instruments.premade import GALEXFUV, GALEXNUV


class DummyImageCollection:
    """Minimal image collection used to inspect forwarded noise arguments."""

    def __init__(self):
        """Initialise the argument-capturing image collection."""
        self.received_aperture_radius = None

    def apply_noise_from_snrs(self, snrs, depths, aperture_radius):
        """Capture the aperture radius routed by the caller."""
        # Record the forwarded aperture radius so the test can verify that the
        # routing path uses the instrument attribute with the correct name.
        self.received_aperture_radius = aperture_radius
        return aperture_radius


class DummyNoiseInstrument:
    """Minimal instrument used to inspect delegated noise application calls."""

    def __init__(self, expected_result):
        """Initialise the noise-call recording instrument."""
        self.label = "inst"
        self.depth_app_radius = object()
        self.expected_result = expected_result
        self.calls = []

    def apply_noises(self, image_collection, aperture_radius=None):
        """Record the delegated noise application call and return a marker."""
        self.calls.append(
            {
                "image_collection": image_collection,
                "aperture_radius": aperture_radius,
            }
        )
        return self.expected_result


class DummySpectroscopicInstrument:
    """Minimal instrument used to inspect delegated spectroscopy calls."""

    def __init__(self):
        """Initialise the spectroscopy-call recording instrument."""
        self.label = "inst"
        self.calls = []

    def apply_lam_array(self, sed):
        """Record the delegated wavelength-application call."""
        marker = object()
        self.calls.append({"sed": sed, "result": marker})
        return marker


class DummyPsfInstrument:
    """Minimal instrument used to inspect delegated PSF application calls."""

    def __init__(self):
        """Initialise the PSF-call recording instrument."""
        self.label = "inst"
        self.psfs = {"F090W": np.ones((3, 3), dtype=float)}
        self.calls = []

    def apply_psfs(self, image_collection, psf_resample_factor=1):
        """Record the delegated PSF application call and return a marker."""
        marker = object()
        self.calls.append(
            {
                "image_collection": image_collection,
                "psf_resample_factor": psf_resample_factor,
                "result": marker,
            }
        )
        return marker


def test_galex_filters_are_not_duplicated_in_collection():
    """Ensure the premade GALEX collection keeps one filter per instrument."""
    fuv = GALEXFUV()
    nuv = GALEXNUV()
    galex = GALEX()

    assert fuv.filters.filter_codes == ["GALEX/GALEX.FUV"]
    assert nuv.filters.filter_codes == ["GALEX/GALEX.NUV"]
    assert galex["GALEXFUV"].filters.filter_codes == ["GALEX/GALEX.FUV"]
    assert galex["GALEXNUV"].filters.filter_codes == ["GALEX/GALEX.NUV"]
    assert galex.all_filters.filter_codes.count("GALEX/GALEX.NUV") == 1
    assert galex.all_filters.filter_codes == [
        "GALEX/GALEX.FUV",
        "GALEX/GALEX.NUV",
    ]


def test_instrument_collection_does_not_mutate_existing_filters():
    """Ensure collection filter bookkeeping does not alter members."""
    lam = np.linspace(1000, 3000, 32) * angstrom
    first_filters = FilterCollection(
        generic_dict={"filter_a": np.ones(lam.size)},
        new_lam=lam,
    )
    second_filters = FilterCollection(
        generic_dict={"filter_b": np.ones(lam.size)},
        new_lam=lam,
    )

    first = Instrument(
        label="first",
        filters=first_filters,
        resolution=1 * arcsecond,
    )
    second = Instrument(
        label="second",
        filters=second_filters,
        resolution=1 * arcsecond,
    )

    collection = InstrumentCollection()
    collection.add_instruments(first, second)

    assert first.filters.filter_codes == ["filter_a"]
    assert second.filters.filter_codes == ["filter_b"]
    assert collection.all_filters.filter_codes == ["filter_a", "filter_b"]


def test_unsupported_mixed_noise_configuration_raises():
    """Unsupported mixed-mode configurations should fail explicitly."""
    with pytest.raises(exceptions.InconsistentArguments):
        Instrument(
            label="spec",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolution=1 * arcsecond,
            noise_source_maps={"filter_a": np.ones((8, 8))},
        )


def test_add_filters_does_not_mutate_on_invalid_noise_payload():
    """Invalid add_filters input should not partially mutate the instrument."""
    lam = np.linspace(1000, 3000, 32) * angstrom
    base_filters = FilterCollection(
        generic_dict={"filter_a": np.ones(lam.size)},
        new_lam=lam,
    )
    new_filters = FilterCollection(
        generic_dict={"filter_b": np.ones(lam.size)},
        new_lam=lam,
    )

    inst = Instrument(
        label="test",
        filters=base_filters,
        resolution=1 * arcsecond,
        noise_source_maps={"filter_a": np.ones((8, 8))},
    )

    with pytest.raises(exceptions.MissingArgument):
        inst.add_filters(
            new_filters,
            noise_maps={"filter_b": np.ones((8, 8))},
        )

    assert inst.filters.filter_codes == ["filter_a"]
    assert set(inst.noise_source_maps.keys()) == {"filter_a"}
    assert inst.noise_maps is None


def test_photometric_imager_inherits_add_filters_unchanged():
    """Imagers should reuse the photometric add_filters implementation."""
    assert PhotometricImager.add_filters is PhotometricInstrument.add_filters


@pytest.mark.parametrize(
    ("method", "images_attr", "noise_attr"),
    [
        (
            BaseGalaxy.apply_noise_to_images_lnu,
            "images_lnu",
            "images_noise_lnu",
        ),
        (
            BaseGalaxy.apply_noise_to_images_fnu,
            "images_fnu",
            "images_noise_fnu",
        ),
    ],
)
def test_base_galaxy_noise_routing_uses_depth_app_radius(
    method, images_attr, noise_attr
):
    """Base-galaxy noise routing should delegate to the instrument."""
    image = object()
    expected_result = object()
    galaxy = SimpleNamespace(
        images_lnu={},
        images_fnu={},
        images_psf_lnu={},
        images_psf_fnu={},
        images_noise_lnu={},
        images_noise_fnu={},
        stars=None,
        black_holes=None,
    )
    getattr(galaxy, images_attr)["inst"] = {"stellar": image}
    instrument = DummyNoiseInstrument(expected_result)

    returned = method(galaxy, instrument, apply_to_psf=False)

    assert instrument.calls[0]["image_collection"] is image
    assert (
        instrument.calls[0]["aperture_radius"] is instrument.depth_app_radius
    )
    assert returned["stellar"] is expected_result
    assert getattr(galaxy, noise_attr)["inst"]["stellar"] is expected_result


@pytest.mark.parametrize(
    ("method", "images_attr", "noise_attr"),
    [
        (
            Component.apply_noise_to_images_lnu,
            "images_lnu",
            "images_noise_lnu",
        ),
        (
            Component.apply_noise_to_images_fnu,
            "images_fnu",
            "images_noise_fnu",
        ),
    ],
)
def test_component_noise_routing_uses_depth_app_radius(
    method, images_attr, noise_attr
):
    """Component noise routing should delegate to the instrument."""
    image = object()
    expected_result = object()
    component = SimpleNamespace(
        images_lnu={},
        images_fnu={},
        images_psf_lnu={},
        images_psf_fnu={},
        images_noise_lnu={},
        images_noise_fnu={},
    )
    getattr(component, images_attr)["inst"] = {"stellar": image}
    instrument = DummyNoiseInstrument(expected_result)

    returned = method(component, instrument, apply_to_psf=False)

    assert instrument.calls[0]["image_collection"] is image
    assert (
        instrument.calls[0]["aperture_radius"] is instrument.depth_app_radius
    )
    assert returned["stellar"] is expected_result
    assert getattr(component, noise_attr)["inst"]["stellar"] is expected_result


@pytest.mark.parametrize(
    ("method", "images_attr", "psf_attr"),
    [
        (
            BaseGalaxy.apply_psf_to_images_lnu,
            "images_lnu",
            "images_psf_lnu",
        ),
        (
            BaseGalaxy.apply_psf_to_images_fnu,
            "images_fnu",
            "images_psf_fnu",
        ),
    ],
)
def test_base_galaxy_psf_routing_delegates_to_instrument(
    method, images_attr, psf_attr
):
    """Base-galaxy PSF routing should delegate to the instrument."""
    image_collection = object()
    galaxy = SimpleNamespace(
        images_lnu={},
        images_fnu={},
        images_psf_lnu={},
        images_psf_fnu={},
        stars=None,
        black_holes=None,
    )
    getattr(galaxy, images_attr)["inst"] = {"stellar": image_collection}
    instrument = DummyPsfInstrument()

    returned = method(galaxy, instrument, psf_resample_factor=3)

    assert instrument.calls[0]["image_collection"] is image_collection
    assert instrument.calls[0]["psf_resample_factor"] == 3
    assert returned["stellar"] is instrument.calls[0]["result"]
    assert (
        getattr(galaxy, psf_attr)["inst"]["stellar"]
        is instrument.calls[0]["result"]
    )


@pytest.mark.parametrize(
    ("method", "images_attr", "psf_attr"),
    [
        (
            Component.apply_psf_to_images_lnu,
            "images_lnu",
            "images_psf_lnu",
        ),
        (
            Component.apply_psf_to_images_fnu,
            "images_fnu",
            "images_psf_fnu",
        ),
    ],
)
def test_component_psf_routing_delegates_to_instrument(
    method, images_attr, psf_attr
):
    """Component PSF routing should delegate to the instrument."""
    image_collection = object()
    component = SimpleNamespace(
        images_lnu={},
        images_fnu={},
        images_psf_lnu={},
        images_psf_fnu={},
    )
    getattr(component, images_attr)["inst"] = {"stellar": image_collection}
    instrument = DummyPsfInstrument()

    returned = method(component, instrument, psf_resample_factor=3)

    assert instrument.calls[0]["image_collection"] is image_collection
    assert instrument.calls[0]["psf_resample_factor"] == 3
    assert returned["stellar"] is instrument.calls[0]["result"]
    assert (
        getattr(component, psf_attr)["inst"]["stellar"]
        is instrument.calls[0]["result"]
    )


def test_spectroscopic_instrument_apply_lam_array_delegates_to_resampling():
    """apply_lam_array should use the existing SED resampling primitive."""
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.linspace(1000, 3000, 32) * angstrom,
    )
    sed = SimpleNamespace()
    expected = object()
    captured = {}

    def fake_apply_instrument_lams(passed_instrument, nthreads=1):
        captured["instrument"] = passed_instrument
        captured["nthreads"] = nthreads
        return expected

    sed.apply_instrument_lams = fake_apply_instrument_lams

    result = instrument.apply_lam_array(sed, nthreads=4)

    assert result is expected
    assert captured["instrument"] is instrument
    assert captured["nthreads"] == 4


def test_base_galaxy_spectroscopy_routing_delegates_to_instrument():
    """Base-galaxy spectroscopy should delegate to the instrument."""
    sed = object()
    galaxy = SimpleNamespace(
        spectra={"stellar": sed},
        spectroscopy={},
        stars=None,
        black_holes=None,
    )
    instrument = DummySpectroscopicInstrument()

    returned = BaseGalaxy.get_spectroscopy(galaxy, instrument)

    assert instrument.calls[0]["sed"] is sed
    assert returned["stellar"] is instrument.calls[0]["result"]
    assert (
        galaxy.spectroscopy["inst"]["stellar"] is instrument.calls[0]["result"]
    )


def test_component_spectroscopy_routing_delegates_to_instrument():
    """Component spectroscopy should delegate to the instrument."""
    sed = object()
    particle_sed = object()
    component = SimpleNamespace(
        spectra={"stellar": sed},
        spectroscopy={},
        particle_spectra={"stellar": particle_sed},
        particle_spectroscopy={},
    )
    instrument = DummySpectroscopicInstrument()

    returned = Component.get_spectroscopy(component, instrument)

    assert instrument.calls[0]["sed"] is sed
    assert instrument.calls[1]["sed"] is particle_sed
    assert returned["stellar"] is instrument.calls[0]["result"]
    assert (
        component.spectroscopy["inst"]["stellar"]
        is instrument.calls[0]["result"]
    )
    assert (
        component.particle_spectroscopy["inst"]["stellar"]
        is (instrument.calls[1]["result"])
    )


@pytest.mark.parametrize(
    ("instrument", "expected_type"),
    [
        (
            PhotometricInstrument(
                label="phot",
                filters=FilterCollection(
                    generic_dict={"filter_a": np.ones(32)},
                    new_lam=np.linspace(1000, 3000, 32) * angstrom,
                ),
            ),
            PhotometricInstrument,
        ),
        (
            PhotometricImager(
                label="img",
                filters=FilterCollection(
                    generic_dict={"filter_b": np.ones(32)},
                    new_lam=np.linspace(1000, 3000, 32) * angstrom,
                ),
                resolution=1 * arcsecond,
            ),
            PhotometricImager,
        ),
        (
            SpectroscopicInstrument(
                label="spec",
                lam=np.linspace(1000, 3000, 32) * angstrom,
            ),
            SpectroscopicInstrument,
        ),
        (
            IntegratedFieldUnit(
                label="ifu",
                lam=np.linspace(1000, 3000, 32) * angstrom,
                resolution=1 * arcsecond,
            ),
            IntegratedFieldUnit,
        ),
    ],
)
def test_instrument_hdf5_roundtrip_preserves_specialised_type(
    tmp_path, instrument, expected_type
):
    """Serialisation should round-trip specialised instrument classes."""
    path = tmp_path / f"{instrument.label}.hdf5"

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf.create_group("Instrument"))

    with h5py.File(path, "r") as hdf:
        loaded = Instrument._from_hdf5(hdf["Instrument"])

    assert isinstance(loaded, expected_type)
    assert loaded == instrument
