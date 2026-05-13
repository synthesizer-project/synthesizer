"""Tests for instrument and instrument collection behaviour."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from unyt import angstrom, arcsecond, kpc

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


class DummyIfuInstrument:
    """Minimal IFU used to inspect delegated cube-generation calls."""

    def __init__(self):
        """Initialise the IFU call recorder."""
        self.label = "ifu_inst"
        self.calls = []

    def generate_data_cube(self, **kwargs):
        """Record the delegated cube-generation call and return a marker."""
        marker = object()
        self.calls.append({"kwargs": kwargs, "result": marker})
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


class DummyImageGenerationInstrument:
    """Minimal instrument used to inspect delegated image generation calls."""

    def __init__(self, expected_result):
        """Initialise the image-generation call recorder."""
        self.label = "inst"
        self.resolution = 1 * kpc
        self.calls = []
        self.expected_result = expected_result

    def generate_images(self, **kwargs):
        """Record the delegated image-generation call and return a marker."""
        self.calls.append(kwargs)
        return self.expected_result


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


def test_integrated_field_unit_cube_placeholders_raise():
    """IFU cube post-processing placeholders should fail explicitly."""
    instrument = IntegratedFieldUnit(
        label="ifu",
        lam=np.linspace(1000, 3000, 32) * angstrom,
        resolution=1 * arcsecond,
    )

    with pytest.raises(exceptions.UnimplementedFunctionality):
        instrument.apply_psf_to_cube(object())

    with pytest.raises(exceptions.UnimplementedFunctionality):
        instrument.apply_psf(object())

    with pytest.raises(exceptions.UnimplementedFunctionality):
        instrument.apply_noise_to_cube(object())

    with pytest.raises(exceptions.UnimplementedFunctionality):
        instrument.apply_noise(object())


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


def test_component_data_cube_only_caches_supported_quantities():
    """Component cube caching should only attach lnu and fnu families."""
    component = SimpleNamespace(
        component_type="Stars",
        spectra={"stellar": object()},
        particle_spectra={},
        data_cubes_lnu={},
        data_cubes_fnu={},
    )
    instrument = DummyIfuInstrument()

    result = Component.get_data_cube(
        component,
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
