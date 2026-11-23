"""Tests for instrument and instrument collection behaviour."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from astropy.cosmology import Planck18
from unyt import Hz, angstrom, arcsecond, erg, kpc, nJy, s, unyt_quantity

from synthesizer import exceptions
from synthesizer.base_galaxy import BaseGalaxy
from synthesizer.components.component import Component
from synthesizer.emissions import Sed
from synthesizer.imaging import SpectralCube
from synthesizer.imaging.postprocess import _postprocess_existing_data_cubes
from synthesizer.instruments import (
    GALEX,
    FilterCollection,
    Instrument,
    InstrumentCollection,
    IntegratedFieldUnit,
    PhotometricImager,
    PhotometricInstrument,
    SpectroscopicInstrument,
    get_lams_from_resolving_power,
    get_lsfs_from_resolving_power,
)
from synthesizer.instruments.premade import GALEXFUV, GALEXNUV, JWSTNIRSpec
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy


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

    @property
    def can_do_psf_imaging(self):
        """Return that this dummy instrument cannot apply PSFs."""
        return False

    @property
    def can_do_noisy_imaging(self):
        """Return that this dummy instrument can apply imaging noise."""
        return True

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

    def __init__(self, noisy=False):
        """Initialise the spectroscopy-call recording instrument."""
        self.label = "inst"
        self.calls = []
        self.noisy = noisy

    @property
    def can_do_noisy_spectroscopy(self):
        """Return whether this dummy instrument should add noise."""
        return self.noisy

    @property
    def can_do_lsf_spectroscopy(self):
        """Return whether this dummy instrument should apply an LSF."""
        return False

    def apply_lam_array_lnu(self, sed):
        """Record the delegated rest frame wavelength-application call."""
        marker = object()
        self.calls.append({"sed": sed, "result": marker, "quantity": "lnu"})
        return marker

    def apply_lam_array_fnu(self, sed):
        """Record the delegated observer frame wavelength-application call."""
        marker = object()
        self.calls.append({"sed": sed, "result": marker, "quantity": "fnu"})
        return marker

    def apply_noise(self, spectrum):
        """Record the delegated spectroscopy-noise call."""
        marker = object()
        self.calls.append({"spectrum": spectrum, "noise_result": marker})
        return marker


class DummyIfuInstrument:
    """Minimal IFU used to inspect delegated cube-generation calls."""

    def __init__(self, do_psf=False, do_noise=False):
        """Initialise the IFU call recorder."""
        self.label = "ifu_inst"
        self.calls = []
        self.do_psf = do_psf
        self.do_noise = do_noise

    @property
    def can_do_psf_spectroscopy(self):
        """Return whether this dummy IFU should apply a PSF."""
        return self.do_psf

    @property
    def can_do_noisy_resolved_spectroscopy(self):
        """Return whether this dummy IFU should apply noise."""
        return self.do_noise

    def generate_data_cube(self, **kwargs):
        """Record the delegated cube-generation call and return a marker."""
        marker = object()
        self.calls.append({"kwargs": kwargs, "result": marker})
        return marker

    def apply_psf(self, observable):
        """Record the delegated IFU-PSF call and return a marker."""
        marker = object()
        self.calls.append({"psf_input": observable, "psf_result": marker})
        return marker

    def apply_noise(self, observable):
        """Record the delegated IFU-noise call and return a marker."""
        marker = object()
        self.calls.append({"noise_input": observable, "noise_result": marker})
        return marker


class DummyPsfInstrument:
    """Minimal instrument used to inspect delegated PSF application calls."""

    def __init__(self):
        """Initialise the PSF-call recording instrument."""
        self.label = "inst"
        self.psfs = {"F090W": np.ones((3, 3), dtype=float)}
        self.psf_resample_factor = 1
        self.calls = []

    @property
    def can_do_psf_imaging(self):
        """Return that this dummy instrument can apply PSFs."""
        return True

    @property
    def can_do_noisy_imaging(self):
        """Return that this dummy instrument cannot apply noise."""
        return False

    def apply_psfs(self, image_collection):
        """Record the delegated PSF application call and return a marker."""
        marker = object()
        self.calls.append(
            {
                "image_collection": image_collection,
                "psf_resample_factor": self.psf_resample_factor,
                "result": marker,
            }
        )
        return marker


class DummyImageGenerationInstrument:
    """Minimal instrument used to inspect delegated image generation calls."""

    def __init__(self, expected_result):
        """Initialise the image-generation call recorder."""
        self.label = "inst"
        self.resolution = 1 * kpc
        self.calls = []
        self.expected_result = expected_result

    @property
    def can_do_psf_imaging(self):
        """Return that this dummy instrument cannot apply PSFs."""
        return False

    @property
    def can_do_noisy_imaging(self):
        """Return that this dummy instrument cannot apply noise."""
        return False

    def generate_images(self, **kwargs):
        """Record the delegated image-generation call and return a marker."""
        self.calls.append(kwargs)
        return self.expected_result


class DummyComponent(Component):
    """Minimal concrete Component used for unit tests."""

    def __init__(self, **kwargs):
        """Initialise a basic component with test-controlled attributes."""
        super().__init__(component_type="Stars", fesc=0.0)
        for key, value in kwargs.items():
            setattr(self, key, value)

    def get_mask(
        self,
        attr,
        thresh,
        op,
        mask=None,
        attr_override_obj=None,
    ):
        """Return the provided mask unchanged for tests."""
        return mask

    def get_weighted_attr(self, attr, weights, **kwargs):
        """Return a weighted attribute placeholder for tests."""
        return None


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
        "first",
        filters=first_filters,
        resolution=1 * arcsecond,
    )
    second = Instrument(
        "second",
        filters=second_filters,
        resolution=1 * arcsecond,
    )

    collection = InstrumentCollection()
    collection.add_instruments(first, second)

    assert first.filters.filter_codes == ["filter_a"]
    assert second.filters.filter_codes == ["filter_b"]
    assert collection.all_filters.filter_codes == ["filter_a", "filter_b"]


def test_ifu_accepts_noise_source_maps_configuration():
    """IFUs should store source-noise templates for future use."""
    instrument = Instrument(
        "spec",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        resolution=1 * arcsecond,
        noise_source_maps={"filter_a": np.ones((8, 8))},
    )

    assert instrument.noise_source_maps is not None


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
        "test",
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


def test_instrument_factory_accepts_single_positional_label():
    """Factory should accept one positional argument as the label."""
    lam = np.linspace(1000, 3000, 32) * angstrom
    filters = FilterCollection(
        generic_dict={"filter_a": np.ones(lam.size)},
        new_lam=lam,
    )

    inst = Instrument("test", filters=filters, resolution=1 * arcsecond)

    assert inst.label == "test"


def test_instrument_factory_rejects_multiple_positional_arguments():
    """Factory should reject more than one positional argument."""
    with pytest.raises(exceptions.InconsistentArguments, match="at most one"):
        Instrument("a", "b")


def test_instrument_factory_rejects_duplicate_positional_and_keyword_label():
    """Factory should reject positional and keyword labels together."""
    with pytest.raises(
        exceptions.InconsistentArguments,
        match="both a positional label and a label keyword",
    ):
        Instrument("a", label="b")


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
    instrument.psf_resample_factor = 3

    returned = method(galaxy, instrument)

    assert instrument.calls[0]["image_collection"] is image_collection
    assert instrument.calls[0]["psf_resample_factor"] == 3
    assert returned["stellar"] is instrument.calls[0]["result"]
    assert (
        getattr(galaxy, psf_attr)["inst"]["stellar"]
        is instrument.calls[0]["result"]
    )


@pytest.mark.parametrize(
    ("method", "images_attr"),
    [
        (BaseGalaxy.apply_psf_to_images_lnu, "images_lnu"),
        (BaseGalaxy.apply_psf_to_images_fnu, "images_fnu"),
    ],
)
def test_base_galaxy_psf_routing_includes_gas_components(method, images_attr):
    """Deprecated galaxy PSF wrappers should still route gas images."""
    gas = SimpleNamespace(calls=[], images_lnu={}, images_fnu={})
    getattr(gas, images_attr)["inst"] = {"gas": object()}

    def apply_psf_to_images(instrument, limit_to=None):
        gas.calls.append((instrument, limit_to))

    if images_attr == "images_lnu":
        gas.apply_psf_to_images_lnu = apply_psf_to_images
    else:
        gas.apply_psf_to_images_fnu = apply_psf_to_images

    galaxy = SimpleNamespace(
        images_lnu={"inst": {}},
        images_fnu={"inst": {}},
        images_psf_lnu={},
        images_psf_fnu={},
        stars=None,
        gas=gas,
        black_holes=None,
    )
    instrument = DummyPsfInstrument()

    method(galaxy, instrument)

    assert gas.calls == [(instrument, None)]


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
    instrument.psf_resample_factor = 3

    returned = method(component, instrument)

    assert instrument.calls[0]["image_collection"] is image_collection
    assert instrument.calls[0]["psf_resample_factor"] == 3
    assert returned["stellar"] is instrument.calls[0]["result"]
    assert (
        getattr(component, psf_attr)["inst"]["stellar"]
        is instrument.calls[0]["result"]
    )


def _observed_top_hat_sed(z):
    """Build a top-hat Sed observed at redshift z.

    The emission is only between 5000 and 6000 A in the rest frame, so
    between 5000 * (1 + z) and 6000 * (1 + z) A in the observer frame.

    Args:
        z (float):
            The redshift to observe the Sed at.

    Returns:
        Sed:
            The Sed with its fluxes computed at z.
    """
    # Build the top-hat luminosity and compute its fluxes at z
    lam = np.arange(1000.0, 20001.0, 10.0) * angstrom
    inside = (lam >= 5000 * angstrom) & (lam <= 6000 * angstrom)
    sed = Sed(lam, np.where(inside, 1e28, 0.0) * erg / s / Hz)
    sed.get_fnu(Planck18, z)
    return sed


def test_apply_lam_array_fnu_uses_observer_frame_wavelengths():
    """apply_lam_array_fnu should treat the instrument grid as observer frame.

    A Sed with emission only between 5000 and 6000 A in the rest frame should
    be seen by an instrument covering 9000-13000 A at z=1, with the observed
    wavelengths of the result matching the instrument's.
    """
    # Observe the top-hat at z=1
    z = 1.0
    sed = _observed_top_hat_sed(z)

    # Define an instrument whose observer frame window contains the emission
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
    )

    # Resample onto the instrument grid
    spec = instrument.apply_lam_array_fnu(sed)

    # The observed grid should be the instrument grid and the rest frame
    # grid should be that shifted back by (1 + z)
    np.testing.assert_allclose(spec.obslam, instrument.lam)
    np.testing.assert_allclose(spec.lam, instrument.lam / (1.0 + z))

    # The flux should land in the observer frame emission window
    seen = spec.fnu.value > 0
    assert seen.any()
    assert spec.obslam[seen].min() >= 9990.0 * angstrom
    assert spec.obslam[seen].max() <= 12010.0 * angstrom


def test_apply_lam_array_lnu_uses_rest_frame_wavelengths():
    """apply_lam_array_lnu should treat the instrument grid as rest frame."""
    # Observe the top-hat at z=1
    z = 1.0
    sed = _observed_top_hat_sed(z)

    # Define an instrument whose rest frame window contains the emission
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(4000.0, 7001.0, 10.0) * angstrom,
    )

    # Resample onto the instrument grid
    spec = instrument.apply_lam_array_lnu(sed)

    # The rest frame grid should be the instrument grid and the observed grid
    # should be that shifted forward by (1 + z)
    np.testing.assert_allclose(spec.lam, instrument.lam)
    np.testing.assert_allclose(spec.obslam, instrument.lam * (1.0 + z))

    # The luminosity should land in the rest frame emission window
    seen = spec.lnu.value > 0
    assert seen.any()
    assert spec.lam[seen].min() >= 4990.0 * angstrom
    assert spec.lam[seen].max() <= 6010.0 * angstrom


def test_apply_lam_array_is_deprecated_alias_of_fnu():
    """apply_lam_array should warn and match apply_lam_array_fnu."""
    sed = _observed_top_hat_sed(1.0)
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
    )

    # The deprecated method warns about its removal
    with pytest.warns(FutureWarning, match="1.4.0"):
        spec = instrument.apply_lam_array(sed)

    # It gives the same result as the observer frame method
    expected = instrument.apply_lam_array_fnu(sed)
    np.testing.assert_allclose(spec.obslam, expected.obslam)
    np.testing.assert_allclose(spec.fnu, expected.fnu)


def test_apply_instrument_lams_is_deprecated_alias_of_fnu():
    """Sed.apply_instrument_lams should warn and match apply_lam_array_fnu."""
    sed = _observed_top_hat_sed(1.0)
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
    )

    # The deprecated method warns about its removal
    with pytest.warns(FutureWarning, match="1.4.0"):
        spec = sed.apply_instrument_lams(instrument)

    # It gives the same result as the observer frame method
    expected = instrument.apply_lam_array_fnu(sed)
    np.testing.assert_allclose(spec.obslam, expected.obslam)
    np.testing.assert_allclose(spec.fnu, expected.fnu)


@pytest.mark.parametrize("quantity", ["lnu", "fnu"])
def test_base_galaxy_spectroscopy_routing_delegates_to_instrument(quantity):
    """Base-galaxy spectroscopy should delegate to the instrument."""
    sed = object()
    galaxy = SimpleNamespace(
        spectra={"stellar": sed},
        spectroscopy_lnu={},
        spectroscopy_fnu={},
        stars=None,
        black_holes=None,
    )
    instrument = DummySpectroscopicInstrument()

    returned = BaseGalaxy._get_spectroscopy(galaxy, instrument, quantity)

    # The matching instrument method was used and the result stored in the
    # matching dictionary only
    assert instrument.calls[0]["sed"] is sed
    assert instrument.calls[0]["quantity"] == quantity
    assert returned["stellar"] is instrument.calls[0]["result"]
    spectroscopy = getattr(galaxy, f"spectroscopy_{quantity}")
    assert spectroscopy["inst"]["stellar"] is instrument.calls[0]["result"]
    other = "fnu" if quantity == "lnu" else "lnu"
    assert getattr(galaxy, f"spectroscopy_{other}") == {}


@pytest.mark.parametrize("quantity", ["lnu", "fnu"])
def test_component_spectroscopy_routing_delegates_to_instrument(quantity):
    """Component spectroscopy should delegate to the instrument."""
    sed = object()
    particle_sed = object()
    component = SimpleNamespace(
        spectra={"stellar": sed},
        spectroscopy_lnu={},
        spectroscopy_fnu={},
        particle_spectra={"stellar": particle_sed},
        particle_spectroscopy_lnu={},
        particle_spectroscopy_fnu={},
    )
    instrument = DummySpectroscopicInstrument()

    returned = Component._get_spectroscopy(component, instrument, quantity)

    # The matching instrument method was used for both the integrated and
    # particle spectra, with the results stored in the matching dictionaries
    assert instrument.calls[0]["sed"] is sed
    assert instrument.calls[1]["sed"] is particle_sed
    assert {call["quantity"] for call in instrument.calls} == {quantity}
    assert returned["stellar"] is instrument.calls[0]["result"]
    spectroscopy = getattr(component, f"spectroscopy_{quantity}")
    particle_spectroscopy = getattr(
        component, f"particle_spectroscopy_{quantity}"
    )
    assert spectroscopy["inst"]["stellar"] is instrument.calls[0]["result"]
    assert (
        particle_spectroscopy["inst"]["stellar"]
        is instrument.calls[1]["result"]
    )


def test_component_spectroscopy_applies_noise_when_configured():
    """Component spectroscopy should apply instrument noise when configured."""
    sed = object()
    component = SimpleNamespace(
        spectra={"stellar": sed},
        spectroscopy_lnu={},
        spectroscopy_fnu={},
        particle_spectra={},
        particle_spectroscopy_lnu={},
        particle_spectroscopy_fnu={},
    )
    instrument = DummySpectroscopicInstrument(noisy=True)

    returned = Component._get_spectroscopy(component, instrument, "fnu")

    assert instrument.calls[0]["sed"] is sed
    assert instrument.calls[1]["spectrum"] is instrument.calls[0]["result"]
    assert returned["stellar"] is instrument.calls[1]["noise_result"]
    assert (
        component.spectroscopy_fnu["inst"]["stellar"]
        is instrument.calls[1]["noise_result"]
    )


def _top_hat_galaxy(z):
    """Build a particle galaxy holding an observed top-hat spectrum.

    Args:
        z (float):
            The redshift of the galaxy and its spectrum.

    Returns:
        ParticleGalaxy:
            The galaxy with the spectrum stored under "toy".
    """
    galaxy = ParticleGalaxy(redshift=z)
    galaxy.spectra["toy"] = _observed_top_hat_sed(z)
    return galaxy


def test_galaxy_get_spectroscopy_fnu_uses_observer_frame_wavelengths():
    """get_spectroscopy_fnu should see the emission at observed wavelengths."""
    # Observe the top-hat at z=1 with an instrument covering its observer
    # frame window
    galaxy = _top_hat_galaxy(1.0)
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
    )

    spec = galaxy.get_spectroscopy_fnu(instrument)["toy"]

    # The result is stored in the fnu dictionary, on the instrument grid,
    # with the flux in the observer frame emission window
    assert galaxy.spectroscopy_fnu["spec"]["toy"] is spec
    assert galaxy.spectroscopy_lnu == {}
    np.testing.assert_allclose(spec.obslam, instrument.lam)
    seen = spec.fnu.value > 0
    assert seen.any()
    assert spec.obslam[seen].min() >= 9990.0 * angstrom
    assert spec.obslam[seen].max() <= 12010.0 * angstrom


@pytest.mark.parametrize(
    "quantity,frame_lam",
    [("lnu", "lam"), ("fnu", "obslam")],
)
def test_component_plot_spectroscopy_uses_matching_frame(quantity, frame_lam):
    """Component spectroscopy plots should read and plot the matching frame.

    Args:
        quantity (str):
            The frame to plot, "lnu" or "fnu".
        frame_lam (str):
            The Sed wavelength attribute the plot should use as its x axis.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Store the observed top-hat in only the requested frame's dictionary,
    # so plotting the wrong frame raises a KeyError
    sed = _observed_top_hat_sed(1.0)
    component = SimpleNamespace(
        redshift=1.0,
        spectroscopy_lnu={},
        spectroscopy_fnu={},
        spectroscopy_lsf_lnu={},
        spectroscopy_lsf_fnu={},
    )
    getattr(component, f"spectroscopy_{quantity}")["spec"] = {"toy": sed}

    fig, ax = Component._plot_spectroscopy(component, "spec", quantity)

    # The plotted wavelengths are those of the requested frame
    np.testing.assert_allclose(
        ax.lines[0].get_xdata(),
        getattr(sed, frame_lam).value,
    )
    plt.close(fig)


def test_galaxy_get_spectroscopy_is_deprecated_alias_of_fnu():
    """get_spectroscopy and spectroscopy should warn and alias the fnu API."""
    galaxy = _top_hat_galaxy(1.0)
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
    )

    # The deprecated method warns and stores into the fnu dictionary
    with pytest.warns(FutureWarning, match="1.4.0"):
        returned = galaxy.get_spectroscopy(instrument)
    assert returned is galaxy.spectroscopy_fnu["spec"]

    # The deprecated attribute warns and points at the fnu dictionary
    with pytest.warns(FutureWarning, match="1.4.0"):
        assert galaxy.spectroscopy is galaxy.spectroscopy_fnu

    # The component attributes alias the fnu dictionaries in the same way
    component = SimpleNamespace(
        spectroscopy_fnu={}, particle_spectroscopy_fnu={}
    )
    with pytest.warns(FutureWarning, match="1.4.0"):
        assert (
            Component.spectroscopy.fget(component)
            is component.spectroscopy_fnu
        )
    with pytest.warns(FutureWarning, match="1.4.0"):
        assert (
            Component.particle_spectroscopy.fget(component)
            is component.particle_spectroscopy_fnu
        )


def test_component_image_generation_delegates_to_instrument():
    """Component image generation should delegate to the instrument."""
    photometry = object()
    expected_result = object()
    component = SimpleNamespace(
        morphology=SimpleNamespace(),
        component_type="stellar",
        redshift=None,
        model_param_cache={"stellar": {}},
        photo_lnu={"stellar": photometry},
        photo_fnu={},
        particle_photo_lnu={},
        particle_photo_fnu={},
        images_lnu={},
        images_fnu={},
    )
    instrument = DummyImageGenerationInstrument(expected_result)

    returned = Component._generate_images(
        component,
        "stellar",
        fov=5 * kpc,
        instrument=instrument,
        img_type="smoothed",
        kernel="kernel",
        kernel_threshold=2,
        nthreads=3,
        cosmo="cosmo",
        phot_type="lnu",
    )

    assert instrument.calls[0]["photometry"] is photometry
    assert instrument.calls[0]["fov"] == 5 * kpc
    assert instrument.calls[0]["img_type"] == "smoothed"
    assert instrument.calls[0]["kernel"] == "kernel"
    assert instrument.calls[0]["kernel_threshold"] == 2
    assert instrument.calls[0]["nthreads"] == 3
    assert instrument.calls[0]["emitter"] is component
    assert instrument.calls[0]["cosmo"] == "cosmo"
    assert returned is expected_result
    assert component.images_lnu["inst"]["stellar"] is expected_result


def test_integrated_field_unit_generate_data_cube_uses_instrument_logic():
    """IFU cube generation should be owned by the instrument."""
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        resolution=1 * arcsecond,
    )
    expected = object()
    component = SimpleNamespace(
        morphology=SimpleNamespace(),
        spectra={"stellar": object()},
        particle_spectra={},
    )

    def fake_generate_parametric_component_cube(**kwargs):
        assert kwargs["component"] is component
        assert kwargs["sed"] is component.spectra["stellar"]
        assert kwargs["fov"] == 5 * arcsecond
        assert np.array_equal(kwargs["lam"], instrument.lam)
        assert kwargs["quantity"] == "lnu"
        return expected

    instrument._generate_parametric_component_cube = (
        fake_generate_parametric_component_cube
    )

    result = instrument.generate_data_cube(
        component=component,
        fov=5 * arcsecond,
        sed="stellar",
        cube_type="smoothed",
        kernel="kernel",
        kernel_threshold=2,
        quantity="lnu",
        nthreads=3,
        cosmo="cosmo",
    )

    assert result is expected


def test_integrated_field_unit_generate_data_cube_rejects_unknown_label():
    """String SED labels should fail fast when the component lacks them."""
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        resolution=1 * arcsecond,
    )
    component = SimpleNamespace(
        component_type="Stars",
        spectra={},
        particle_spectra={},
    )

    with pytest.raises(ValueError, match="missing"):
        instrument.generate_data_cube(
            component=component,
            fov=5 * arcsecond,
            sed="missing",
            cube_type="hist",
        )


def test_data_cube_postprocess_keeps_raw_store_unchanged():
    """Cube post-processing should not overwrite the raw cube cache."""
    raw_cube = object()
    owner = SimpleNamespace(data_cubes_lnu={"ifu": {"stellar": raw_cube}})

    class DummyIfu:
        label = "ifu"
        can_do_psf_spectroscopy = True
        can_do_noisy_resolved_spectroscopy = True

        def apply_psf(self, cube):
            assert cube is raw_cube
            return "psf"

        def apply_noise(self, cube):
            assert cube == "psf"
            return "noise"

    result = _postprocess_existing_data_cubes(
        owner,
        instrument=DummyIfu(),
        quantity="lnu",
    )

    assert owner.data_cubes_lnu["ifu"]["stellar"] is raw_cube
    assert owner.processed_data_cubes_lnu["ifu"]["stellar"] == "noise"
    assert result["stellar"] == "noise"


@pytest.fixture
def spectral_cube_with_central_source():
    """Return a small cube with a unit source centred in every slice."""
    lam = np.linspace(1000, 3000, 3) * angstrom
    cube = SpectralCube(
        resolution=1 * arcsecond,
        fov=9 * arcsecond,
        lam=lam,
    )
    cube.arr = np.zeros((9, 9, len(lam)), dtype=float)
    cube.arr[4, 4, :] = 1.0
    cube.units = nJy
    cube.quantity = "fnu"
    return cube


class TestIntegratedFieldUnitPsfApplication:
    """Tests for instrument-owned PSF application on IFU cubes."""

    @staticmethod
    def _instrument(psfs, psf_pixel_scale=1 * arcsecond, psf_lam=None):
        """Construct a minimal IFU for PSF-application tests."""
        return IntegratedFieldUnit(
            label="ifu",
            lam=np.linspace(1000, 3000, 3) * angstrom,
            resolution=1 * arcsecond,
            psfs=psfs,
            psf_pixel_scale=psf_pixel_scale,
            psf_lam=psf_lam,
        )

    def test_apply_psf_to_cube_with_2d_psf(
        self, spectral_cube_with_central_source
    ):
        """A normalised 2D PSF should broaden every cube slice."""
        psf = np.ones((3, 3), dtype=float)
        psf /= psf.sum()
        cube = spectral_cube_with_central_source
        original_flux = cube.arr.sum(axis=(0, 1)).copy()

        result = self._instrument(psf).apply_psf_to_cube(cube)

        np.testing.assert_allclose(result.arr.sum(axis=(0, 1)), original_flux)
        assert np.all(result.arr[4, 4, :] < cube.arr[4, 4, :])

    def test_apply_psf_to_cube_uses_wavelength_specific_psfs(
        self, spectral_cube_with_central_source
    ):
        """A 3D PSF should use the kernel matching each wavelength slice."""
        psfs = np.zeros((3, 3, 3), dtype=float)
        psfs[1, 1, 0] = 1.0
        psfs[1, 0, 1] = 1.0
        psfs[0, 1, 2] = 1.0

        result = self._instrument(psfs).apply_psf_to_cube(
            spectral_cube_with_central_source
        )

        peak_locations = [
            np.unravel_index(
                np.argmax(result.arr[:, :, index]), result.arr.shape[:2]
            )
            for index in range(result.arr.shape[2])
        ]
        assert peak_locations == [(4, 4), (4, 3), (3, 4)]

    def test_apply_psf_to_cube_returns_new_cube_by_default(
        self, spectral_cube_with_central_source
    ):
        """PSF application should preserve metadata in a fresh cube."""
        psf = np.zeros((3, 3), dtype=float)
        psf[1, 1] = 1.0
        cube = spectral_cube_with_central_source
        result = self._instrument(psf).apply_psf_to_cube(cube)

        assert result is not cube
        assert result.resolution == cube.resolution
        np.testing.assert_allclose(result.fov, cube.fov)
        np.testing.assert_array_equal(result.npix, cube.npix)
        np.testing.assert_allclose(result.lam, cube.lam)
        assert result.units == cube.units
        assert result.quantity == cube.quantity

    def test_apply_psf_to_cube_can_update_cube_inplace(
        self, spectral_cube_with_central_source
    ):
        """PSF application should support explicit in-place updates."""
        psf = np.ones((3, 3), dtype=float) / 9
        result = self._instrument(psf).apply_psf_to_cube(
            spectral_cube_with_central_source, inplace=True
        )

        assert result is spectral_cube_with_central_source
        assert np.all(result.arr[4, 4, :] < 1.0)

    def test_apply_psf_to_cube_requires_psfs(
        self, spectral_cube_with_central_source
    ):
        """PSF application should fail explicitly without configured PSFs."""
        instrument = IntegratedFieldUnit(
            label="ifu",
            lam=np.linspace(1000, 3000, 3) * angstrom,
            resolution=1 * arcsecond,
        )

        with pytest.raises(exceptions.MissingArgument):
            instrument.apply_psf_to_cube(spectral_cube_with_central_source)

    def test_apply_psf_to_cube_requires_psf_pixel_scale(self):
        """IFUs with PSFs should require their spatial pixel scale."""
        psf = np.ones((3, 3), dtype=float) / 9

        with pytest.raises(exceptions.MissingArgument):
            IntegratedFieldUnit(
                label="ifu",
                lam=np.linspace(1000, 3000, 3) * angstrom,
                resolution=1 * arcsecond,
                psfs=psf,
            )

    @pytest.mark.parametrize(
        "psf_pixel_scale",
        [
            0 * arcsecond,
            -1 * arcsecond,
            np.inf * arcsecond,
            np.nan * arcsecond,
        ],
        ids=("zero", "negative", "infinite", "nan"),
    )
    def test_psf_pixel_scale_must_be_positive_and_finite(
        self, psf_pixel_scale
    ):
        """IFU PSF pixel scales should be positive finite quantities."""
        psf = np.ones((3, 3), dtype=float) / 9

        with pytest.raises(exceptions.InconsistentArguments):
            self._instrument(psf, psf_pixel_scale=psf_pixel_scale)

    def test_apply_psf_to_cube_rejects_coarser_psf_pixel_scale(
        self, spectral_cube_with_central_source
    ):
        """PSF sampling cannot be coarser than the cube sampling."""
        psf = np.ones((3, 3), dtype=float) / 9
        instrument = self._instrument(psf, psf_pixel_scale=2 * arcsecond)

        with pytest.raises(exceptions.InconsistentArguments):
            instrument.apply_psf_to_cube(spectral_cube_with_central_source)

    def test_apply_psf_to_cube_interpolates_psf_wavelength_planes(
        self, spectral_cube_with_central_source
    ):
        """A 3D PSF should interpolate planes from a distinct grid."""
        psfs = np.zeros((3, 3, 2), dtype=float)
        psfs[1, 1, 0] = 1.0
        psfs[1, 0, 1] = 1.0
        instrument = self._instrument(
            psfs,
            psf_lam=np.array([900.0, 2100.0]) * angstrom,
        )

        result = instrument.apply_psf_to_cube(
            spectral_cube_with_central_source
        )

        np.testing.assert_allclose(
            result.arr[4, 4, :],
            [11 / 12, 1 / 12, 0],
        )
        np.testing.assert_allclose(
            result.arr[4, 3, :],
            [1 / 12, 11 / 12, 1],
        )

    def test_psf_plane_count_must_match_psf_lam(self):
        """A 3D PSF must have one plane per PSF wavelength."""
        psfs = np.zeros((3, 3, 2), dtype=float)
        psfs[1, 1, :] = 1.0

        with pytest.raises(exceptions.InconsistentArguments):
            self._instrument(psfs)

    @pytest.mark.parametrize(
        "psf_lam",
        [
            np.array([[1000.0, 2000.0, 3000.0]]) * angstrom,
            np.array([]) * angstrom,
            np.array([1000.0, np.inf, 3000.0]) * angstrom,
            np.array([1000.0, 0.0, 3000.0]) * angstrom,
            np.array([1000.0, 1000.0, 3000.0]) * angstrom,
        ],
        ids=(
            "two_dimensional",
            "empty",
            "infinite",
            "non_positive",
            "unordered",
        ),
    )
    def test_psf_lam_must_be_a_valid_wavelength_grid(self, psf_lam):
        """PSF wavelength grids must be one-dimensional and increasing."""
        psfs = np.zeros((3, 3, 3), dtype=float)
        psfs[1, 1, :] = 1.0

        with pytest.raises(exceptions.InconsistentArguments):
            self._instrument(psfs, psf_lam=psf_lam)

    def test_psf_lam_is_normalized_to_the_instrument_grid(self):
        """IFUs compare using their PSFs resampled to the instrument grid."""
        psfs = np.zeros((3, 3, 3), dtype=float)
        psfs[1, 1, :] = 1.0
        psf_lam = np.array([1000.0, 2000.0, 3000.0]) * angstrom
        instrument = self._instrument(psfs, psf_lam=psf_lam)
        identical = self._instrument(psfs.copy(), psf_lam=psf_lam.copy())
        different = self._instrument(
            psfs.copy(), psf_lam=psf_lam + 1 * angstrom
        )

        assert instrument == identical
        assert instrument == different
        np.testing.assert_allclose(instrument.psf_lam, instrument.lam)
        np.testing.assert_allclose(different.psf_lam, different.lam)

    def test_apply_psf_to_cube_with_resampling_does_not_mutate_input(
        self, spectral_cube_with_central_source
    ):
        """Default PSF supersampling should not mutate the input cube."""
        cube = spectral_cube_with_central_source
        original_arr = cube.arr.copy()
        original_resolution = cube.resolution
        original_npix = cube.npix.copy()
        psf = np.ones((3, 3), dtype=float) / 9

        result = self._instrument(
            psf, psf_pixel_scale=0.5 * arcsecond
        ).apply_psf_to_cube(cube)

        assert result is not cube
        assert cube.resolution == original_resolution
        assert np.array_equal(cube.npix, original_npix)
        assert np.array_equal(cube.arr, original_arr)

    def test_nearly_integer_psf_pixel_scale_rounds_to_integer_factor(
        self, spectral_cube_with_central_source
    ):
        """Near-integer PSF scale ratios should use the integer path."""
        psf = np.ones((3, 3), dtype=float) / 9
        cube = spectral_cube_with_central_source
        exact = self._instrument(
            psf, psf_pixel_scale=0.5 * arcsecond
        ).apply_psf_to_cube(cube)
        nearly_integer_scale = cube.resolution / (2.0 + 1.0e-10)
        rounded = self._instrument(
            psf, psf_pixel_scale=nearly_integer_scale
        ).apply_psf_to_cube(cube)

        np.testing.assert_allclose(rounded.arr, exact.arr)

    def test_apply_psf_to_cube_supports_noninteger_resampling(
        self, spectral_cube_with_central_source
    ):
        """Non-integer PSF scale ratios should use spatial interpolation."""
        cube = spectral_cube_with_central_source
        psf = np.ones((3, 3), dtype=float) / 9
        result = self._instrument(
            psf, psf_pixel_scale=cube.resolution / 1.5
        ).apply_psf_to_cube(cube)

        assert result.arr.shape == cube.arr.shape
        assert result.resolution == cube.resolution

    def test_apply_psf_to_cube_requires_generated_cube(self):
        """PSF application should fail when the cube array is absent."""
        empty_cube = SpectralCube(
            resolution=1 * arcsecond,
            fov=9 * arcsecond,
            lam=np.linspace(1000, 3000, 3) * angstrom,
        )
        psf = np.ones((3, 3), dtype=float) / 9

        with pytest.raises(exceptions.MissingIFU):
            self._instrument(psf).apply_psf_to_cube(empty_cube)

    def test_postprocess_with_ifu_psf_preserves_raw_cube(
        self, spectral_cube_with_central_source
    ):
        """IFU PSF post-processing should retain the raw cube cache."""
        psf = np.ones((3, 3), dtype=float) / 9
        instrument = self._instrument(psf)
        raw_cube = spectral_cube_with_central_source
        owner = SimpleNamespace(
            data_cubes_lnu={instrument.label: {"stellar": raw_cube}}
        )

        result = _postprocess_existing_data_cubes(
            owner, instrument=instrument, quantity="lnu"
        )

        processed_cube = owner.processed_data_cubes_lnu[instrument.label][
            "stellar"
        ]
        assert owner.data_cubes_lnu[instrument.label]["stellar"] is raw_cube
        assert result["stellar"] is processed_cube
        assert processed_cube is not raw_cube
        assert not np.allclose(processed_cube.arr, raw_cube.arr)

    @pytest.mark.parametrize(
        ("psfs", "psf_pixel_scale", "psf_lam"),
        [
            (None, None, None),
            (np.ones((3, 3), dtype=float) / 9, 0.5 * arcsecond, None),
            (
                np.repeat(
                    (np.ones((3, 3), dtype=float) / 9)[:, :, None],
                    2,
                    axis=2,
                ),
                0.5 * arcsecond,
                np.array([900.0, 2100.0]) * angstrom,
            ),
        ],
        ids=("without_psfs", "two_dimensional_psfs", "custom_psf_lam"),
    )
    def test_hdf5_roundtrip_preserves_psf_metadata(
        self, tmp_path, psfs, psf_pixel_scale, psf_lam
    ):
        """IFU HDF5 serialisation should preserve PSF metadata."""
        instrument = IntegratedFieldUnit(
            label="ifu",
            lam=np.linspace(1000, 3000, 3) * angstrom,
            resolution=1 * arcsecond,
            psfs=psfs,
            psf_pixel_scale=psf_pixel_scale,
            psf_lam=psf_lam,
        )
        expected_psf_lam = instrument.psf_lam
        expected_psfs = instrument.psfs
        filepath = tmp_path / "ifu.hdf5"

        with h5py.File(filepath, "w") as hdf:
            instrument.to_hdf5(hdf)
        with h5py.File(filepath, "r") as hdf:
            assert "PSFWavelength" in hdf
            assert "units" in hdf["PSFWavelength"].attrs
            np.testing.assert_allclose(
                hdf["PSFWavelength"][...],
                expected_psf_lam.to("angstrom").value,
            )
            restored = IntegratedFieldUnit._from_hdf5(hdf)

        np.testing.assert_allclose(
            restored.psf_lam.to("angstrom").value,
            expected_psf_lam.to("angstrom").value,
        )
        if psf_pixel_scale is None:
            assert restored.psf_pixel_scale is None
            assert restored.psfs is None
        else:
            assert restored.psf_pixel_scale.units == psf_pixel_scale.units
            np.testing.assert_allclose(
                restored.psf_pixel_scale.value, psf_pixel_scale.value
            )
            np.testing.assert_allclose(restored.psfs, expected_psfs)

    def test_hdf5_load_without_psf_wavelength_defaults_to_lam(self, tmp_path):
        """Legacy IFU files should use the instrument grid for PSF planes."""
        psfs = np.repeat(
            (np.ones((3, 3), dtype=float) / 9)[:, :, None], 3, axis=2
        )
        instrument = self._instrument(psfs)
        filepath = tmp_path / "legacy_ifu.hdf5"

        with h5py.File(filepath, "w") as hdf:
            instrument.to_hdf5(hdf)
        with h5py.File(filepath, "a") as hdf:
            del hdf["PSFWavelength"]
        with h5py.File(filepath, "r") as hdf:
            restored = IntegratedFieldUnit._from_hdf5(hdf)

        np.testing.assert_allclose(restored.psf_lam, instrument.lam)


def test_integrated_field_unit_noise_needs_a_noise_model(
    spectral_cube_with_central_source,
):
    """IFU noise should fail explicitly without a noise model to apply."""
    cube = spectral_cube_with_central_source

    # Without a depth and SNR or noise maps there is no noise to apply
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=cube.lam,
        resolution=1 * arcsecond,
    )
    with pytest.raises(exceptions.MissingArgument):
        instrument.apply_noise_to_cube(cube)

    # Only spectral cubes can be given IFU noise
    with pytest.raises(exceptions.InconsistentArguments):
        instrument.apply_noise(object())

    # Correlated noise from noise source maps is not implemented yet
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=cube.lam,
        resolution=1 * arcsecond,
        noise_source_maps={"filter_a": np.ones((9, 9))},
    )
    with pytest.raises(exceptions.UnimplementedFunctionality):
        instrument.apply_noise_to_cube(cube)


def test_component_style_cube_routing_can_delegate_to_ifu():
    """Component cube routing should be expressible via the IFU method."""
    component = object()
    instrument = DummyIfuInstrument()

    result = instrument.generate_data_cube(
        component=component,
        fov="fov",
        sed="stellar",
        cube_type="hist",
        kernel=None,
        kernel_threshold=1.0,
        quantity="fnu",
        cosmo="cosmo",
        nthreads=5,
    )

    assert result is instrument.calls[0]["result"]
    assert instrument.calls[0]["kwargs"]["component"] is component
    assert instrument.calls[0]["kwargs"]["fov"] == "fov"
    assert instrument.calls[0]["kwargs"]["sed"] == "stellar"
    assert instrument.calls[0]["kwargs"]["cube_type"] == "hist"
    assert instrument.calls[0]["kwargs"]["kernel"] is None
    assert instrument.calls[0]["kwargs"]["kernel_threshold"] == 1.0
    assert instrument.calls[0]["kwargs"]["quantity"] == "fnu"
    assert instrument.calls[0]["kwargs"]["cosmo"] == "cosmo"
    assert instrument.calls[0]["kwargs"]["nthreads"] == 5


def test_galaxy_style_cube_routing_delegates_to_components():
    """Galaxy cube orchestration should delegate to component get methods."""
    stellar_cube = object()
    blackhole_cube = object()

    class DummyCube:
        def __init__(self, marker):
            self.marker = marker

        def __add__(self, other):
            return (self.marker, other.marker)

    stars = SimpleNamespace(calls=[])
    black_holes = SimpleNamespace(calls=[])
    instrument = SimpleNamespace(label="inst", resolution=1 * kpc)

    def stellar_get_data_cube(*args, **kwargs):
        stars.calls.append({"args": args, "kwargs": kwargs})
        return DummyCube(stellar_cube)

    def blackhole_get_data_cube(*args, **kwargs):
        black_holes.calls.append({"args": args, "kwargs": kwargs})
        return DummyCube(blackhole_cube)

    stars.get_data_cube = stellar_get_data_cube
    black_holes.get_data_cube = blackhole_get_data_cube

    galaxy = SimpleNamespace(
        galaxy_type="Particle",
        redshift=None,
        model_param_cache={},
        stars=stars,
        gas=None,
        black_holes=black_holes,
        data_cubes_lnu={},
        data_cubes_fnu={},
    )

    result = ParticleGalaxy.get_data_cube(
        galaxy,
        fov="fov",
        instrument=instrument,
        stellar_spectra="stellar",
        blackhole_spectra="agn",
        cube_type="hist",
        kernel=None,
        kernel_threshold=1.0,
        quantity="lnu",
        nthreads=2,
        cosmo="cosmo",
    )

    assert result == (stellar_cube, blackhole_cube)
    assert stars.calls[0]["args"] == ("stellar",)
    assert stars.calls[0]["kwargs"]["instrument"] is instrument
    assert black_holes.calls[0]["args"] == ("agn",)
    assert black_holes.calls[0]["kwargs"]["instrument"] is instrument


def test_parametric_galaxy_cube_routing_delegates_to_components():
    """Parametric galaxy cube orchestration should delegate to components."""
    stellar_cube = object()

    class DummyCube:
        def __init__(self, marker):
            self.marker = marker

    stars = SimpleNamespace(calls=[])
    instrument = SimpleNamespace(label="inst", resolution=1 * kpc)

    def stellar_get_data_cube(*args, **kwargs):
        stars.calls.append({"args": args, "kwargs": kwargs})
        return DummyCube(stellar_cube)

    stars.get_data_cube = stellar_get_data_cube

    galaxy = SimpleNamespace(
        galaxy_type="Parametric",
        redshift=None,
        model_param_cache={},
        stars=stars,
        gas=None,
        black_holes=None,
        data_cubes_lnu={},
        data_cubes_fnu={},
    )

    result = ParametricGalaxy.get_data_cube(
        galaxy,
        fov="fov",
        instrument=instrument,
        stellar_spectra="stellar",
        quantity="lnu",
    )

    assert result.marker is stellar_cube
    assert stars.calls[0]["args"] == ("stellar",)
    assert stars.calls[0]["kwargs"]["instrument"] is instrument


def test_particle_galaxy_cube_routing_rejects_missing_component():
    """Legacy component-specific cube requests should fail clearly."""
    galaxy = SimpleNamespace(
        galaxy_type="Particle",
        redshift=None,
        model_param_cache={},
        stars=None,
        gas=None,
        black_holes=None,
        data_cubes_lnu={},
        data_cubes_fnu={},
    )
    instrument = SimpleNamespace(label="inst", resolution=1 * kpc)

    with pytest.raises(ValueError, match="no stellar component"):
        ParticleGalaxy.get_data_cube(
            galaxy,
            fov="fov",
            instrument=instrument,
            stellar_spectra="stellar",
        )


def test_photometric_imager_validate_rejects_partial_psf_dict():
    """Imaging payload dictionaries must cover every configured filter."""
    filters = FilterCollection(
        generic_dict={"filter_a": np.ones(32), "filter_b": np.ones(32)},
        new_lam=np.linspace(1000, 3000, 32) * angstrom,
    )

    with pytest.raises(exceptions.MissingArgument, match="psfs"):
        PhotometricImager(
            label="img",
            filters=filters,
            resolution=1 * arcsecond,
            psfs={"filter_a": np.ones((3, 3))},
        )


def test_add_filters_requires_imaging_payloads_for_existing_maps():
    """Adding filters should require matching payloads once maps exist."""
    base_filters = FilterCollection(
        generic_dict={"filter_a": np.ones(32)},
        new_lam=np.linspace(1000, 3000, 32) * angstrom,
    )
    extra_filters = FilterCollection(
        generic_dict={"filter_b": np.ones(32)},
        new_lam=np.linspace(1000, 3000, 32) * angstrom,
    )
    instrument = PhotometricImager(
        label="img",
        filters=base_filters,
        resolution=1 * arcsecond,
        psfs={"filter_a": np.ones((3, 3))},
    )

    with pytest.raises(exceptions.InconsistentAddition, match="psfs"):
        instrument.add_filters(extra_filters)


def test_photometric_instrument_comparison_uses_filter_shapes():
    """Equality should distinguish filters that share codes but not curves."""
    lam = np.linspace(1000, 3000, 32) * angstrom
    instrument_a = PhotometricInstrument(
        label="phot",
        filters=FilterCollection(
            generic_dict={"filter_a": np.ones(32)},
            new_lam=lam,
        ),
    )
    instrument_b = PhotometricInstrument(
        label="phot",
        filters=FilterCollection(
            generic_dict={"filter_a": np.linspace(0, 1, 32)},
            new_lam=lam,
        ),
    )

    assert instrument_a != instrument_b


def test_jwst_nirspec_placeholder_raises_clear_error():
    """Placeholder premade instruments should raise NotImplementedError."""
    with pytest.raises(NotImplementedError, match="placeholder"):
        JWSTNIRSpec()


def test_component_level_combined_cube_uses_model_cache():
    """Component cube generation should combine labels already requested."""

    class DummyCube:
        def __init__(self, marker):
            self.marker = marker

        def __deepcopy__(self, memo):
            return DummyCube(self.marker)

        def __iadd__(self, other):
            self.marker = (self.marker, other.marker)
            return self

    component = SimpleNamespace(
        component_type="Stars",
        spectra={"line": object(), "continuum": object()},
        particle_spectra={},
        model_param_cache={
            "line": {},
            "continuum": {},
            "nebular": {"combine": ["line", "continuum"]},
        },
        data_cubes_lnu={},
        data_cubes_fnu={},
    )
    instrument = DummyIfuInstrument()

    original_generate = instrument.generate_data_cube

    def generate_data_cube(**kwargs):
        label = kwargs["sed"]
        marker = DummyCube(label)
        instrument.calls.append({"kwargs": kwargs, "result": marker})
        return marker

    instrument.generate_data_cube = generate_data_cube

    result = Component._generate_data_cubes(
        component,
        "line",
        "continuum",
        "nebular",
        fov="fov",
        instrument=instrument,
        quantity="lnu",
    )

    assert result["nebular"].marker == ("line", "continuum")
    assert set(component.data_cubes_lnu[instrument.label]) == {
        "line",
        "continuum",
        "nebular",
    }
    instrument.generate_data_cube = original_generate


def test_component_data_cube_applies_ifu_postprocessing_when_configured():
    """Component cubes should apply IFU PSF/noise when configured."""
    component = DummyComponent(
        spectra={"stellar": object()},
        particle_spectra={},
        model_param_cache={},
    )
    instrument = DummyIfuInstrument(do_psf=True, do_noise=True)

    result = component.get_data_cube(
        "stellar",
        fov="fov",
        instrument=instrument,
        quantity="lnu",
    )

    assert instrument.calls[1]["psf_input"] is instrument.calls[0]["result"]
    assert (
        instrument.calls[2]["noise_input"] is instrument.calls[1]["psf_result"]
    )
    assert result is instrument.calls[2]["noise_result"]
    assert (
        component.data_cubes_lnu[instrument.label]["stellar"]
        is (instrument.calls[0]["result"])
    )
    assert (
        component.processed_data_cubes_lnu[instrument.label]["stellar"]
        is (instrument.calls[2]["noise_result"])
    )


def test_component_data_cube_only_caches_supported_quantities():
    """Component cube caching should only attach lnu and fnu families."""
    component = DummyComponent(
        spectra={"stellar": object()},
        particle_spectra={},
    )
    instrument = DummyIfuInstrument()

    result = component.get_data_cube(
        "stellar",
        fov="fov",
        instrument=instrument,
        quantity="llam",
    )

    assert result is instrument.calls[0]["result"]
    assert component.data_cubes_lnu[instrument.label]["stellar"] is result
    assert component.data_cubes_fnu == {}


def test_parametric_ifu_hist_cube_raises():
    """Parametric components should reject histogram cube generation."""
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        resolution=1 * arcsecond,
    )
    component = SimpleNamespace(
        component_type="Stars",
        morphology=SimpleNamespace(),
        spectra={"stellar": object()},
        particle_spectra={},
    )

    with pytest.raises(exceptions.InconsistentArguments):
        instrument.generate_data_cube(
            component=component,
            fov=5 * arcsecond,
            sed="stellar",
            cube_type="hist",
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


@pytest.mark.parametrize(
    "instrument",
    [
        SpectroscopicInstrument(
            label="spec-r",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolving_power=1000.0,
        ),
        IntegratedFieldUnit(
            label="ifu-r",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolution=1 * arcsecond,
            resolving_power=500.0,
        ),
        # NumPy scalars are constants too, and only float64 happens to be a
        # subclass of float, so the rest need covering explicitly.
        SpectroscopicInstrument(
            label="spec-r-float32",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolving_power=np.float32(1000.0),
        ),
        SpectroscopicInstrument(
            label="spec-r-int64",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolving_power=np.int64(1000),
        ),
        IntegratedFieldUnit(
            label="ifu-r-float32",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolution=1 * arcsecond,
            resolving_power=np.float32(500.0),
        ),
    ],
)
def test_resolving_power_roundtrips(tmp_path, instrument):
    """A constant resolving_power should be saved and restored."""
    path = tmp_path / f"{instrument.label}.hdf5"

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf.create_group("Instrument"))

    with h5py.File(path, "r") as hdf:
        assert hdf["Instrument"].attrs["resolving_power"] == pytest.approx(
            instrument.resolving_power
        )
        loaded = Instrument._from_hdf5(hdf["Instrument"])

    assert loaded.resolving_power == pytest.approx(instrument.resolving_power)
    assert loaded == instrument


def test_callable_resolving_power_is_not_serialised(tmp_path):
    """A callable resolving_power cannot be written to HDF5 and is dropped."""
    instrument = SpectroscopicInstrument(
        label="spec-callable-r",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        resolving_power=lambda lam: 1000.0,
    )
    path = tmp_path / "spec-callable-r.hdf5"

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf.create_group("Instrument"))

    with h5py.File(path, "r") as hdf:
        assert "resolving_power" not in hdf["Instrument"].attrs
        loaded = Instrument._from_hdf5(hdf["Instrument"])

    assert loaded.resolving_power is None


def test_spectroscopic_scalar_noise_roundtrip(tmp_path):
    """Scalar depths keep their units and scalar SNRs stay floats."""
    instrument = SpectroscopicInstrument(
        label="spec-scalar-noise",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        depth=100 * nJy,
        snrs=5.0,
        depth_lam=2000 * angstrom,
    )
    path = tmp_path / "spec-scalar-noise.hdf5"

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf.create_group("Instrument"))

    with h5py.File(path, "r") as hdf:
        loaded = Instrument._from_hdf5(hdf["Instrument"])

    assert isinstance(loaded.depth, unyt_quantity)
    assert loaded.depth == 100 * nJy
    assert isinstance(loaded.snrs, float)
    assert loaded.snrs == 5.0
    assert loaded.depth_lam == 2000 * angstrom


def test_ifu_noise_settings_roundtrip(tmp_path):
    """The IFU aperture depth and noise template should be restored."""
    instrument = IntegratedFieldUnit(
        label="ifu-noise",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        resolution=1 * arcsecond,
        depth=100 * nJy,
        snrs=5.0,
        depth_app_radius=0.1 * kpc,
        depth_window=np.array([1500.0, 2500.0]) * angstrom,
        noise_shape_lam=np.linspace(1000, 3000, 8) * angstrom,
        noise_shape_std=np.ones(8) * nJy,
        template_on_detector=False,
    )
    path = tmp_path / "ifu-noise.hdf5"

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf.create_group("Instrument"))

    with h5py.File(path, "r") as hdf:
        loaded = Instrument._from_hdf5(hdf["Instrument"])

    assert loaded.depth_app_radius == 0.1 * kpc
    np.testing.assert_allclose(loaded.depth_window, instrument.depth_window)
    np.testing.assert_allclose(loaded.noise_shape_std, np.ones(8) * nJy)
    assert not loaded.template_on_detector
    assert loaded == instrument


@pytest.mark.parametrize(
    "kwargs,error",
    [
        # A spectrum has no labels to key depths by
        (
            {"depth": {"a": 100 * nJy}, "snrs": 5.0},
            exceptions.InconsistentArguments,
        ),
        # The noise is added in physical units
        ({"depth": 100.0, "snrs": 5.0}, exceptions.InconsistentArguments),
        # A noise template needs a wavelength or window to scale it at
        (
            {
                "depth": 100 * nJy,
                "snrs": 5.0,
                "noise_shape_lam": np.linspace(1000, 3000, 8) * angstrom,
                "noise_shape_std": np.ones(8) * nJy,
            },
            exceptions.MissingArgument,
        ),
        # But not both
        (
            {
                "depth": 100 * nJy,
                "snrs": 5.0,
                "depth_lam": 2000 * angstrom,
                "depth_window": np.array([1500.0, 2500.0]) * angstrom,
            },
            exceptions.InconsistentArguments,
        ),
        # And neither without a depth
        ({"depth_lam": 2000 * angstrom}, exceptions.MissingArgument),
    ],
)
def test_spectroscopic_noise_settings_are_validated(kwargs, error):
    """Inconsistent noise settings are refused on initialisation."""
    with pytest.raises(error):
        SpectroscopicInstrument(
            label="invalid-noise",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            **kwargs,
        )


def _observed_flat_spectrum(instrument):
    """Build a flat spectrum observed at z=1 on an instrument's wavelengths.

    Args:
        instrument (SpectroscopicInstrument):
            The instrument whose (observer frame) wavelengths to use.

    Returns:
        Sed:
            The flat spectrum, with its luminosities and fluxes.
    """
    lam = np.arange(1000.0, 20001.0, 10.0) * angstrom
    sed = Sed(lam, np.full(lam.size, 1e28) * erg / s / Hz)
    sed.get_fnu(Planck18, 1.0)
    return instrument.apply_lam_array_fnu(sed)


@pytest.mark.parametrize(
    "depth,quantity",
    [(100 * nJy, "fnu"), (1e27 * erg / s / Hz, "lnu")],
)
def test_apply_noise_uses_the_quantity_of_the_depth(depth, quantity):
    """Flat noise of depth / snrs should go on the spectrum the depth is in."""
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
        depth=depth,
        snrs=5.0,
    )
    spec = _observed_flat_spectrum(instrument)

    noisy = instrument.apply_noise(spec, rng_seed=42)

    # The noise is added to the spectrum the depth is in, and the other
    # spectrum is carried over without noise
    other = "lnu" if quantity == "fnu" else "fnu"
    np.testing.assert_allclose(
        getattr(noisy, quantity) - getattr(spec, quantity), noisy.noise_arr
    )
    np.testing.assert_array_equal(getattr(noisy, other), getattr(spec, other))

    # Every element has a 1-sigma noise of depth / snrs
    sigma = (depth / 5.0).to(getattr(spec, quantity).units).value
    np.testing.assert_allclose(noisy.weight_map, 1 / sigma**2)
    np.testing.assert_allclose(np.std(noisy.noise_arr.value), sigma, rtol=0.15)


def test_apply_noise_scales_a_noise_template_to_the_depth():
    """A template's shape should be kept, and scaled to depth / snrs."""
    # A template rising linearly with wavelength, on its own wavelengths
    template_lam = np.linspace(8000.0, 14000.0, 7) * angstrom
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
        depth=100 * nJy,
        snrs=5.0,
        depth_lam=11000 * angstrom,
        noise_shape_lam=template_lam,
        noise_shape_std=template_lam.value * nJy,
        template_on_detector=False,
    )
    spec = _observed_flat_spectrum(instrument)

    noisy = instrument.apply_noise(spec, rng_seed=42)

    # The noise is 20 nJy at 11000 A and rises in proportion to wavelength
    sigma = (20 * nJy).to(spec.fnu.units).value
    np.testing.assert_allclose(
        1 / np.sqrt(noisy.weight_map),
        sigma * instrument.lam.value / 11000.0,
    )


def test_apply_noise_adds_noise_maps_as_they_are():
    """Fixed noise maps should be added to the spectrum unchanged."""
    lam = np.arange(9000.0, 13001.0, 10.0) * angstrom
    noise_maps = np.linspace(-1.0, 1.0, lam.size) * nJy
    instrument = SpectroscopicInstrument(
        label="spec", lam=lam, noise_maps=noise_maps
    )
    spec = _observed_flat_spectrum(instrument)

    noisy = instrument.apply_noise(spec)

    np.testing.assert_allclose(noisy.noise_arr.to_value(nJy), noise_maps.value)
    np.testing.assert_allclose(
        (noisy.fnu - spec.fnu).to_value(nJy), noise_maps.value, atol=1e-8
    )

    # Noise maps that don't match the wavelengths are refused
    instrument = SpectroscopicInstrument(
        label="spec", lam=lam, noise_maps=noise_maps[:-1]
    )
    with pytest.raises(exceptions.InconsistentArguments):
        instrument.apply_noise(spec)


def test_simspin_noise_has_the_requested_snr():
    """SimSpin noise needs no depth and gives one spectrum S/N = snr."""
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=np.arange(9000.0, 13001.0, 10.0) * angstrom,
    )
    spec = _observed_flat_spectrum(instrument)

    noisy = instrument.apply_noise(spec, rng_seed=42, simspin_snr=10.0)

    # A single spectrum is its own median, so sigma = |F| / snr
    np.testing.assert_allclose(
        1 / np.sqrt(noisy.weight_map), spec.fnu.value / 10.0
    )

    # The S/N must be positive
    with pytest.raises(exceptions.InconsistentArguments):
        instrument.apply_noise(spec, simspin_snr=0.0)


@pytest.mark.parametrize("template_on_detector", [True, False])
def test_ifu_noise_correlates_elements_sharing_detector_pixels(
    template_on_detector,
):
    """Noise drawn on the detector should correlate neighbouring elements."""
    # A cube sampling each 40 A detector pixel 4 times
    lam = np.arange(1000.0, 3001.0, 10.0) * angstrom
    cube = SpectralCube(resolution=1 * arcsecond, fov=20 * arcsecond, lam=lam)
    cube.arr = np.zeros((20, 20, lam.size))
    cube.units = nJy
    cube.quantity = "fnu"

    # A flat noise template on the detector pixels (or not)
    template_lam = np.arange(1000.0, 3001.0, 40.0) * angstrom
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=lam,
        resolution=1 * arcsecond,
        depth=100 * nJy,
        snrs=5.0,
        depth_lam=2000 * angstrom,
        noise_shape_lam=template_lam,
        noise_shape_std=np.ones(template_lam.size) * nJy,
        template_on_detector=template_on_detector,
    )

    noisy = instrument.apply_noise(cube, rng_seed=42)

    # Either way every voxel has a 1-sigma noise of depth / snrs
    np.testing.assert_allclose(noisy.weight_map, 1 / 20.0**2)

    # Two elements between the same pair of detector pixels are correlated
    # only if the noise is drawn on the detector
    corr = np.corrcoef(
        noisy.noise_arr.value[:, :, 1].ravel(),
        noisy.noise_arr.value[:, :, 2].ravel(),
    )[0, 1]
    if template_on_detector:
        assert corr > 0.5
    else:
        assert abs(corr) < 0.25


def test_ifu_aperture_depth_is_converted_to_per_pixel(
    spectral_cube_with_central_source,
):
    """An aperture depth should be spread over the pixels it covers."""
    cube = spectral_cube_with_central_source
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=cube.lam,
        resolution=1 * arcsecond,
        depth=100 * nJy,
        snrs=5.0,
        depth_app_radius=2 * arcsecond,
    )

    noisy = instrument.apply_noise_to_cube(cube, rng_seed=42)

    # The 20 nJy aperture noise is shared between pi (2 / 1)^2 pixels
    sigma = 20.0 / np.sqrt(np.pi * 2.0**2)
    np.testing.assert_allclose(noisy.weight_map, 1 / sigma**2)

    # The input cube is left as it was
    assert cube.arr.sum() == 3.0


@pytest.mark.parametrize(
    "resolving_power",
    [True, np.bool_(True), 0, -1.0, np.nan, np.inf, 1 + 2j],
)
def test_resolving_power_must_be_positive_number(resolving_power):
    """Booleans and non-positive constants are not resolving powers."""
    with pytest.raises(exceptions.InconsistentArguments, match="positive"):
        SpectroscopicInstrument(
            label="invalid-r",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolving_power=resolving_power,
        )


def test_lams_from_constant_resolving_power():
    """A constant resolving power gives distinct, evenly log-spaced lams."""
    lams = get_lams_from_resolving_power(
        1000.0 * angstrom, 2000.0 * angstrom, 100.0
    )

    # Every wavelength is distinct, rising and within the requested range
    assert np.all(np.diff(lams.value) > 0)
    assert lams[0] == 1000 * angstrom
    assert lams[-1] <= 2000 * angstrom

    # Each step is lambda / R
    np.testing.assert_allclose(lams.value[1:] / lams.value[:-1], 1.01)


def test_lams_from_varying_resolving_power():
    """A wavelength-dependent resolving power sets each step."""

    def resolving_power(lam):
        return lam.to_value("angstrom") / 10

    lams = get_lams_from_resolving_power(
        1000.0 * angstrom, 2005.0 * angstrom, resolving_power
    )

    # R = lambda / 10 A means a constant 10 A step, so 101 wavelengths
    # from 1000 A to 2000 A
    np.testing.assert_allclose(np.diff(lams.value), 10.0)
    assert len(lams) == 101
    np.testing.assert_allclose(lams[-1].value, 2000.0)


def _observed_line_sed(z, line_lam=5500.0, nspec=None):
    """Build an Sed with a flat continuum and one narrow line observed at z.

    Args:
        z (float):
            The redshift to observe the Sed at.
        line_lam (float):
            The rest frame wavelength of the line in angstrom.
        nspec (int, optional):
            If given the Sed holds this many copies of the spectrum, as
            particle spectra do.

    Returns:
        Sed:
            The Sed with its fluxes computed at z.
    """
    # Build the continuum and line, and compute the fluxes at z
    lam = np.arange(1000.0, 20001.0, 1.0) * angstrom
    lnu = np.full(lam.size, 1e28)
    lnu[np.argmin(np.abs(lam.value - line_lam))] += 1e31
    if nspec is not None:
        lnu = np.tile(lnu, (nspec, 1))
    sed = Sed(lam, lnu * erg / s / Hz)
    sed.get_fnu(Planck18, z)
    return sed


def _lsf_instrument(kernel, lsf_resample_factor=1):
    """Build an instrument covering 10000-12000 A with one LSF kernel.

    Args:
        kernel (np.ndarray):
            The LSF kernel to use at every wavelength (None for no LSFs).
        lsf_resample_factor (int):
            The LSF supersampling factor.

    Returns:
        SpectroscopicInstrument:
            The instrument, with a wavelength element every 10 A.
    """
    lam = np.arange(10000.0, 12001.0, 10.0) * angstrom
    return SpectroscopicInstrument(
        label="spec",
        lam=lam,
        lsfs=kernel,
        lsf_resample_factor=lsf_resample_factor,
    )


def test_apply_lsf_uses_each_elements_own_kernel():
    """Each wavelength element should collect light with its own kernel."""
    sed = _observed_line_sed(1.0)

    # Every wavelength element keeps its own light, apart from the element
    # after the one holding the line, whose kernel collects the light of the
    # element before it
    lam = np.arange(10000.0, 12001.0, 10.0) * angstrom
    lsfs = np.tile([0.0, 1.0, 0.0], (lam.size, 1))
    lsfs[101] = [0.0, 0.0, 1.0]
    instrument = SpectroscopicInstrument(label="spec", lam=lam, lsfs=lsfs)

    spec = instrument.apply_lsf(sed, "observer")

    # The line is therefore seen in its own element and in the next one
    # (whose flux density is slightly higher, as it is narrower in frequency)
    line = (spec.fnu - spec.fnu[0]).value
    assert line[100] > 0
    np.testing.assert_allclose(line[101], line[100], rtol=1e-2)
    np.testing.assert_allclose(
        np.delete(line, [100, 101]), 0, atol=1e-6 * line[100]
    )


@pytest.mark.parametrize("lsf_resample_factor", [1, 4])
def test_apply_lsf_keeps_a_flat_spectrum_flat(lsf_resample_factor):
    """A flat spectrum should stay flat, including at the ends."""
    # A flat spectrum that extends beyond the instrument wavelengths
    lam = np.arange(1000.0, 20001.0, 1.0) * angstrom
    sed = Sed(lam, np.full(lam.size, 1e28) * erg / s / Hz)
    sed.get_fnu(Planck18, 1.0)

    # Unevenly spaced instrument wavelengths and a wide kernel, so the
    # elements at the ends need the light from beyond them
    lam = np.geomspace(10000.0, 12000.0, 201) * angstrom
    kernel = np.exp(-0.5 * (np.arange(-8, 9) / 3.0) ** 2)
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=lam,
        lsfs=np.tile(kernel / kernel.sum(), (lam.size, 1)),
        lsf_resample_factor=lsf_resample_factor,
    )

    spec = instrument.apply_lsf(sed, "observer")

    # It stays flat to 0.2% (past the ends, the spacing of the wavelengths
    # is taken to carry on as it is at the ends)
    np.testing.assert_allclose(
        spec.fnu, instrument.apply_lam_array_fnu(sed).fnu, rtol=2e-3
    )


def test_apply_lsf_with_wavelengths_shared_with_the_spectrum():
    """A flat spectrum should stay flat on a subset of its own wavelengths."""
    # Cut the instrument wavelengths out of the spectrum's own, so that
    # the edges of the two sets of wavelength bins coincide
    lam = np.arange(1000.0, 9000.0, 0.1) * angstrom
    sed = Sed(lam, np.ones(lam.size) * erg / s / Hz)
    instrument = SpectroscopicInstrument(
        label="spec",
        lam=lam[45725:45837],
        lsfs=np.tile([0.25, 0.5, 0.25], (112, 1)),
    )

    spec = instrument.apply_lsf(sed, "rest")

    np.testing.assert_allclose(spec.lnu.value, 1.0)


def test_lsf_resample_factor_keeps_a_lines_position():
    """Finer LSF bins should keep a line's position within an element."""
    # A line observed at 11002 A, a fifth of a wavelength element from the
    # middle of the element at 11000 A
    sed = _observed_line_sed(1.0, line_lam=5501.0)

    centres = {}
    for factor in (1, 5):
        # A Gaussian LSF with a sigma of one wavelength element
        offsets = np.arange(-4 * factor, 4 * factor + 1) / factor
        kernel = np.exp(-0.5 * offsets**2)
        instrument = _lsf_instrument(kernel / kernel.sum(), factor)
        spec = instrument.apply_lsf(sed, "observer")

        # Find the centre of the line
        line = (spec.fnu - spec.fnu[0]).value
        centres[factor] = np.sum(line * instrument.lam.value) / np.sum(line)

    # Without finer bins the line is seen at the middle of its element,
    # with bins a fifth of an element wide it stays where it is
    np.testing.assert_allclose(centres[1], 11000.0, atol=0.1)
    np.testing.assert_allclose(centres[5], 11002.0, atol=0.1)


@pytest.mark.parametrize(
    "frame,quantity,scale,frame_lam",
    [
        ("rest", "lnu", 0.5, "lam"),
        ("observer", "fnu", 1.0, "obslam"),
        ("observed", "fnu", 1.0, "obslam"),
    ],
)
def test_apply_lsf_uses_matching_frame(frame, quantity, scale, frame_lam):
    """apply_lsf should apply the LSF in the frame it is given."""
    # Put the instrument wavelengths around the line in the requested frame
    # (the line is at 5500 A in the rest frame and 11000 A observed at z=1,
    # in the middle of the instrument's 101st wavelength element)
    sed = _observed_line_sed(1.0)
    kernel = np.array([0.2, 0.5, 0.3])
    lam = scale * np.arange(10000.0, 12001.0, 10.0) * angstrom
    instrument = SpectroscopicInstrument(
        label="spec", lam=lam, lsfs=np.tile(kernel, (lam.size, 1))
    )

    spec = instrument.apply_lsf(sed, frame)
    plain = getattr(instrument, f"apply_lam_array_{quantity}")(sed)

    # The result is on the instrument wavelengths in that frame, with the
    # other frame's wavelengths following from the redshift
    np.testing.assert_allclose(getattr(spec, frame_lam), instrument.lam)
    np.testing.assert_allclose(spec.obslam, spec.lam * 2.0)

    # The line (above the flat continuum) has the shape of the kernel, in
    # the same order, in both the luminosities and the fluxes
    for arr in (spec.lnu, spec.fnu):
        line = (arr - arr[0]).value[99:102]
        np.testing.assert_allclose(line / line.sum(), kernel, atol=1e-3)

    # The light in the line is unchanged
    line = (getattr(spec, quantity) - getattr(spec, quantity)[0]).value
    plain_line = (getattr(plain, quantity) - getattr(plain, quantity)[0]).value
    np.testing.assert_allclose(line.sum(), plain_line[100], rtol=1e-3)


def test_apply_lsf_reuses_its_matrix_for_the_same_wavelengths():
    """The matrix applying the LSF is reused for the same wavelengths only."""
    instrument = _lsf_instrument(np.array([0.25, 0.5, 0.25]))
    sed = _observed_line_sed(1.0)

    # Spectra on the same wavelengths share one matrix
    matrix = instrument._get_lsf_matrix(sed._obslam)
    assert instrument._get_lsf_matrix(sed._obslam.copy()) is matrix

    # Spectra on other wavelengths get their own, and so the same
    # spectroscopy as from an instrument that has been used for nothing else
    other = _observed_line_sed(1.01)
    unused = _lsf_instrument(np.array([0.25, 0.5, 0.25]))
    np.testing.assert_array_equal(
        instrument.apply_lsf(other, "observer").fnu,
        unused.apply_lsf(other, "observer").fnu,
    )

    # No more than eight matrices are kept
    for z in np.arange(1.02, 1.11, 0.01):
        instrument._get_lsf_matrix(_observed_line_sed(z)._obslam)
    assert len(instrument._lsf_matrices) == 8


def test_apply_lsf_errors():
    """apply_lsf should raise without LSFs, a known frame or fluxes."""
    sed = _observed_line_sed(1.0)
    instrument = _lsf_instrument(np.array([0.25, 0.5, 0.25]))

    # An instrument without LSFs has nothing to apply
    with pytest.raises(exceptions.MissingArgument):
        _lsf_instrument(None).apply_lsf(sed, "observer")

    # The frame of the instrument wavelengths must be "rest" or "observer"
    with pytest.raises(exceptions.InconsistentArguments):
        instrument.apply_lsf(sed, "fnu")

    # The observer frame needs fluxes
    with pytest.raises(exceptions.MissingSpectraType):
        instrument.apply_lsf(Sed(sed.lam, sed.lnu), "observer")


@pytest.mark.parametrize("lsf_resample_factor", [0, 2.5])
def test_lsf_resample_factor_must_be_positive_integer(lsf_resample_factor):
    """LSF resample factors that are not positive integers are refused."""
    with pytest.raises(exceptions.InconsistentArguments):
        _lsf_instrument(None, lsf_resample_factor)


def test_lsf_resample_factor_roundtrips(tmp_path):
    """The LSF resample factor should be saved and restored."""
    instrument = _lsf_instrument(np.array([0.25, 0.5, 0.25]), 4)
    path = tmp_path / "spec.hdf5"

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf)

    loaded = SpectroscopicInstrument.load(path)
    assert loaded.lsf_resample_factor == 4
    assert loaded == instrument

    # The factor is part of what makes two instruments equal
    assert loaded != _lsf_instrument(np.array([0.25, 0.5, 0.25]), 1)

    # Files written before the factor existed load with a factor of 1
    with h5py.File(path, "a") as hdf:
        del hdf["LSFResampleFactor"]
    assert SpectroscopicInstrument.load(path).lsf_resample_factor == 1


@pytest.mark.parametrize("quantity", ["lnu", "fnu"])
def test_galaxy_spectroscopy_applies_lsf(quantity):
    """Galaxy spectroscopy should be stored with and without the LSF."""
    sed = _observed_line_sed(1.0)
    galaxy = ParticleGalaxy(redshift=1.0)
    galaxy.spectra["toy"] = sed
    instrument = _lsf_instrument(np.array([0.25, 0.5, 0.25]))
    get_spectroscopy = getattr(galaxy, f"get_spectroscopy_{quantity}")
    frame = "rest" if quantity == "lnu" else "observer"

    spec = get_spectroscopy(instrument)["toy"]

    # The spectroscopy with the LSF is returned and stored in its own
    # dictionary, next to the spectroscopy without the LSF
    spectroscopy = getattr(galaxy, f"spectroscopy_{quantity}")
    spectroscopy_lsf = getattr(galaxy, f"spectroscopy_lsf_{quantity}")
    assert spectroscopy_lsf["spec"]["toy"] is spec
    np.testing.assert_allclose(
        getattr(spec, quantity),
        getattr(instrument.apply_lsf(sed, frame), quantity),
    )
    apply_lam_array = getattr(instrument, f"apply_lam_array_{quantity}")
    np.testing.assert_allclose(
        getattr(spectroscopy["spec"]["toy"], quantity),
        getattr(apply_lam_array(sed), quantity),
    )

    # Once the instrument has no LSF, the spectroscopy stored with it is
    # discarded
    instrument.set_lsfs(None)
    get_spectroscopy(instrument)
    assert "spec" not in spectroscopy_lsf


def test_component_spectroscopy_applies_lsf_to_particle_spectra():
    """Component spectroscopy should apply the LSF to particle spectra too."""
    # A component with an integrated spectrum and three single precision
    # particle spectra, each the same as the integrated one
    particle_sed = _observed_line_sed(1.0, nspec=3)
    particle_sed.cast(np.float32)
    component = SimpleNamespace(
        redshift=1.0,
        spectra={"toy": _observed_line_sed(1.0)},
        particle_spectra={"toy": particle_sed},
        spectroscopy_fnu={},
        spectroscopy_lsf_fnu={},
        particle_spectroscopy_fnu={},
        particle_spectroscopy_lsf_fnu={},
    )
    instrument = _lsf_instrument(np.array([0.25, 0.5, 0.25]))

    returned = Component._get_spectroscopy(component, instrument, "fnu")

    # The spectroscopy with the LSF is returned, and each particle's
    # matches it, in the precision of the particle spectra
    spec = component.spectroscopy_lsf_fnu["spec"]["toy"]
    particle_spec = component.particle_spectroscopy_lsf_fnu["spec"]["toy"]
    assert returned["toy"] is spec
    assert particle_spec.fnu.shape == (3, instrument.lam.size)
    assert particle_spec.fnu.dtype == np.float32
    np.testing.assert_allclose(particle_spec.fnu[0], spec.fnu, rtol=1e-5)

    # The particle spectroscopy without the LSF is kept as well
    plain = component.particle_spectroscopy_fnu["spec"]["toy"]
    assert not np.allclose(plain.fnu, particle_spec.fnu)

    # Once the instrument has no LSF, the spectroscopy stored with it is
    # discarded
    instrument.set_lsfs(None)
    Component._get_spectroscopy(component, instrument, "fnu")
    assert "spec" not in component.spectroscopy_lsf_fnu
    assert "spec" not in component.particle_spectroscopy_lsf_fnu


@pytest.mark.parametrize(
    "resolving_power,resolving_power_lam",
    [
        (500.0, None),
        (lambda lam: 500.0, None),
        (np.array([500.0, 500.0]), np.array([9000.0, 11000.0]) * angstrom),
    ],
)
def test_lsfs_from_resolving_power(resolving_power, resolving_power_lam):
    """A resolving power should give Gaussian kernels of the right width."""
    lam = np.arange(9950.0, 10051.0, 10.0) * angstrom
    lsfs = get_lsfs_from_resolving_power(
        lam,
        resolving_power,
        lsf_resample_factor=4,
        resolving_power_lam=resolving_power_lam,
    )

    # Each wavelength has a normalised kernel with an odd number of bins
    assert lsfs.shape[0] == lam.size
    assert lsfs.shape[1] % 2 == 1
    np.testing.assert_allclose(lsfs.sum(axis=1), 1.0)

    # R = 500 at 10000 A is a full width at half maximum of 20 A, which is
    # two 10 A wavelength elements, so 8 bins with 4 bins to an element
    offsets = np.arange(lsfs.shape[1]) - lsfs.shape[1] // 2
    sigma = np.sqrt(np.sum(lsfs[5] * offsets**2))
    np.testing.assert_allclose(
        sigma, 8 / (2 * np.sqrt(2 * np.log(2))), rtol=1e-2
    )


def test_resolving_power_table_must_cover_wavelengths():
    """A tabulated resolving power should cover the wavelengths."""
    lam = np.arange(10000.0, 12001.0, 10.0) * angstrom
    resolving_power = np.array([500.0, 1000.0])

    # A table that stops within a wavelength element of each end is fine
    get_lsfs_from_resolving_power(
        lam,
        resolving_power,
        resolving_power_lam=np.array([10005.0, 11995.0]) * angstrom,
    )

    # One that stops further short is refused
    with pytest.raises(exceptions.InconsistentArguments):
        get_lsfs_from_resolving_power(
            lam,
            resolving_power,
            resolving_power_lam=np.array([10500.0, 12000.0]) * angstrom,
        )


def test_set_lsfs_on_an_existing_instrument():
    """LSFs set on an existing instrument should be applied."""
    lam = np.arange(10000.0, 12001.0, 10.0) * angstrom
    sed = _observed_line_sed(1.0)

    # A resolving power alone does not give an instrument an LSF
    instrument = SpectroscopicInstrument(
        label="spec", lam=lam, resolving_power=500.0
    )
    assert not instrument.can_do_lsf_spectroscopy

    # Setting the LSFs made from it gives the same spectrum as passing
    # them when the instrument is created
    lsfs = get_lsfs_from_resolving_power(lam, 500.0, lsf_resample_factor=4)
    instrument.set_lsfs(lsfs, lsf_resample_factor=4)
    expected = SpectroscopicInstrument(
        label="spec", lam=lam, lsfs=lsfs, lsf_resample_factor=4
    )
    assert instrument.lsf_resample_factor == 4
    np.testing.assert_array_equal(
        instrument.apply_lsf(sed, "observer").fnu,
        expected.apply_lsf(sed, "observer").fnu,
    )

    # Setting other LSFs replaces them
    kernel = np.array([0.25, 0.5, 0.25])
    instrument.set_lsfs(kernel, lsf_resample_factor=1)
    np.testing.assert_array_equal(
        instrument.apply_lsf(sed, "observer").fnu,
        _lsf_instrument(kernel).apply_lsf(sed, "observer").fnu,
    )

    # And None removes them
    instrument.set_lsfs(None)
    assert not instrument.can_do_lsf_spectroscopy
