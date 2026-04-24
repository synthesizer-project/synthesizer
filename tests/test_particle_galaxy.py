"""Tests for particle galaxy chunking helpers."""

import numpy as np
from unyt import Hz, Mpc, Msun, Myr, angstrom, erg, km, kpc, s, unyt_array, yr

from synthesizer.emissions.line import LineCollection
from synthesizer.emissions.sed import Sed
from synthesizer.imaging.image import Image
from synthesizer.imaging.image_collection import ImageCollection
from synthesizer.imaging.spectral_cube import SpectralCube
from synthesizer.instruments import FilterCollection
from synthesizer.particle.blackholes import BlackHoles
from synthesizer.particle.galaxy import Galaxy
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars
from synthesizer.photometry import PhotometryCollection
from synthesizer.pipeline.pipeline_utils import (
    accumulate_pipeline_results_from_child,
    clear_pipeline_outputs,
)


def _make_particle_galaxy():
    """Construct a simple particle galaxy for split/combine tests."""
    stars = Stars(
        initial_masses=np.arange(1, 6) * Msun,
        ages=np.arange(1, 6) * Myr,
        metallicities=np.linspace(0.01, 0.05, 5),
        current_masses=np.arange(6, 11) * Msun,
        coordinates=np.arange(15).reshape(5, 3) * Mpc,
        velocities=np.arange(15, 30).reshape(5, 3) * km / s,
        tau_v=np.linspace(0.1, 0.5, 5),
        smoothing_lengths=np.linspace(0.01, 0.05, 5) * Mpc,
        redshift=0.1,
        centre=np.zeros(3) * Mpc,
    )
    gas = Gas(
        masses=np.arange(1, 5) * Msun,
        metallicities=np.linspace(0.02, 0.08, 4),
        star_forming=np.array([True, False, True, False]),
        coordinates=np.arange(12).reshape(4, 3) * Mpc,
        velocities=np.arange(12, 24).reshape(4, 3) * km / s,
        smoothing_lengths=np.linspace(0.02, 0.05, 4) * Mpc,
        dust_to_metal_ratio=np.linspace(0.1, 0.4, 4),
        tau_v=np.linspace(0.2, 0.5, 4),
        redshift=0.1,
        centre=np.zeros(3) * Mpc,
    )
    black_holes = BlackHoles(
        masses=np.array([10.0, 20.0]) * Msun,
        accretion_rates=np.array([0.1, 0.2]) * Msun / yr,
        metallicities=np.array([0.01, 0.02]),
        coordinates=np.arange(6).reshape(2, 3) * Mpc,
        velocities=np.arange(6, 12).reshape(2, 3) * km / s,
        smoothing_lengths=np.array([0.03, 0.04]) * Mpc,
        tau_v=np.array([0.3, 0.6]),
        redshift=0.1,
        centre=np.zeros(3) * Mpc,
    )
    return Galaxy(
        name="chunked",
        stars=stars,
        gas=gas,
        black_holes=black_holes,
        redshift=0.1,
        centre=np.zeros(3) * Mpc,
    )


def _make_filters():
    """Create a small filter collection for photometry tests."""
    return FilterCollection(
        generic_dict={
            "filter_a": np.ones(4),
            "filter_b": np.full(4, 0.5),
        },
        new_lam=np.linspace(1000, 4000, 4) * angstrom,
    )


def _make_image_collection(scale):
    """Create a simple image collection."""
    img = Image(
        resolution=1 * kpc,
        fov=2 * kpc,
        img=unyt_array(np.full((2, 2), scale), erg / s / Hz),
    )
    return ImageCollection(
        resolution=1 * kpc,
        fov=2 * kpc,
        imgs={"filter_a": img},
    )


def _make_cube(scale):
    """Create a simple spectral cube."""
    lam = np.array([1000.0, 2000.0]) * angstrom
    cube = SpectralCube(resolution=1 * kpc, fov=2 * kpc, lam=lam)
    cube.arr = np.full((2, 2, 2), scale)
    cube.units = erg / s / Hz
    cube.sed = Sed(lam=lam, lnu=unyt_array([scale, 2 * scale], erg / s / Hz))
    cube.quantity = "lnu"
    return cube


def test_particle_galaxy_split_preserves_particle_data():
    """Splitting should preserve all particle data and star chunk sizes."""
    galaxy = _make_particle_galaxy()

    galaxy.spectra["preexisting"] = Sed(
        lam=np.array([1000.0, 2000.0]) * angstrom,
        lnu=unyt_array([1.0, 2.0], erg / s / Hz),
    )

    children = galaxy.split(max_npart=4)

    assert len(children) == 2
    assert all(child.stars.nparticles <= 4 for child in children)
    assert children[0].gas is galaxy.gas
    assert children[0].black_holes is galaxy.black_holes
    assert children[1].gas is None
    assert children[1].black_holes is None

    assert (
        sum(
            child.stars.nstars for child in children if child.stars is not None
        )
        == 5
    )
    assert (
        sum(
            child.gas.nparticles for child in children if child.gas is not None
        )
        == 4
    )
    assert (
        sum(
            child.black_holes.nbh
            for child in children
            if child.black_holes is not None
        )
        == 2
    )

    np.testing.assert_allclose(
        np.concatenate(
            [
                child.stars.initial_masses.value
                for child in children
                if child.stars is not None
            ]
        ),
        galaxy.stars.initial_masses.value,
    )
    np.testing.assert_allclose(
        np.concatenate(
            [
                child.gas.masses.value
                for child in children
                if child.gas is not None
            ]
        ),
        galaxy.gas.masses.value,
    )
    np.testing.assert_allclose(
        np.concatenate(
            [
                child.black_holes.masses.value
                for child in children
                if child.black_holes is not None
            ]
        ),
        galaxy.black_holes.masses.value,
    )
    np.testing.assert_allclose(
        np.concatenate(
            [
                child.stars.tau_v
                for child in children
                if child.stars is not None
            ]
        ),
        galaxy.stars.tau_v,
    )

    assert all(child.spectra == {} for child in children)
    assert all(child.stars.spectra == {} for child in children if child.stars)


def test_particle_galaxy_combines_additive_child_outputs():
    """Additive outputs from child galaxies should sum onto the parent."""
    galaxy = _make_particle_galaxy()
    children = galaxy.split(max_npart=4)
    filters = _make_filters()
    lam = np.array([1000.0, 2000.0]) * angstrom
    stellar_scale = 0.0

    clear_pipeline_outputs(galaxy)

    for index, child in enumerate(children, start=1):
        scale = float(index)

        child.spectra["galaxy"] = Sed(
            lam=lam,
            lnu=unyt_array([scale, 2 * scale], erg / s / Hz),
        )
        child.lines["line"] = LineCollection(
            line_ids=["H 1 1215.67A"],
            lam=np.array([1215.67]) * angstrom,
            lum=unyt_array([scale], erg / s),
            cont=unyt_array([0.1 * scale], erg / s / Hz),
        )
        child.lines["line"].get_flux0()
        child.photo_lnu["galaxy"] = PhotometryCollection(
            filters,
            unyt_array([scale, 2 * scale], erg / s / Hz),
        )
        child.spectroscopy["inst"] = {
            "galaxy": Sed(
                lam=lam,
                lnu=unyt_array([2 * scale, 3 * scale], erg / s / Hz),
            )
        }
        child.images_lnu["inst"] = _make_image_collection(scale)
        child.data_cubes_lnu = {"cube": _make_cube(scale)}

        if child.stars is not None:
            stellar_scale += scale
            child.stars.spectra["stellar"] = Sed(
                lam=lam,
                lnu=unyt_array([3 * scale, 4 * scale], erg / s / Hz),
            )
            child.stars.photo_lnu["stellar"] = PhotometryCollection(
                filters,
                unyt_array([4 * scale, 5 * scale], erg / s / Hz),
            )
            child.stars.sfh = np.array([scale, 2 * scale])
            child.stars.sfzh = np.array([[scale, 0.0], [0.0, scale]])

    accumulate_pipeline_results_from_child(galaxy, *children)

    total_scale = sum(range(1, len(children) + 1))

    np.testing.assert_allclose(
        galaxy.spectra["galaxy"].lnu.value,
        np.array([total_scale, 2 * total_scale]),
    )
    np.testing.assert_allclose(
        galaxy.lines["line"].luminosity.value,
        np.array([total_scale]),
    )
    np.testing.assert_allclose(
        galaxy.lines["line"].flux.value,
        children[0].lines["line"].flux.value * total_scale,
    )
    np.testing.assert_allclose(
        galaxy.photo_lnu["galaxy"].photometry.value,
        np.array([total_scale, 2 * total_scale]),
    )
    np.testing.assert_allclose(
        galaxy.spectroscopy["inst"]["galaxy"].lnu.value,
        np.array([2 * total_scale, 3 * total_scale]),
    )
    assert np.sum(galaxy.images_lnu["inst"]["filter_a"].arr) == 4 * total_scale
    assert np.sum(galaxy.data_cubes_lnu["cube"].arr) == 8 * total_scale

    np.testing.assert_allclose(
        galaxy.stars.spectra["stellar"].lnu.value,
        np.array([3 * stellar_scale, 4 * stellar_scale]),
    )
    np.testing.assert_allclose(
        galaxy.stars.photo_lnu["stellar"].photometry.value,
        np.array([4 * stellar_scale, 5 * stellar_scale]),
    )
    np.testing.assert_allclose(
        galaxy.stars.sfh,
        np.array([stellar_scale, 2 * stellar_scale]),
    )
    np.testing.assert_allclose(
        galaxy.stars.sfzh,
        np.array([[stellar_scale, 0.0], [0.0, stellar_scale]]),
    )
