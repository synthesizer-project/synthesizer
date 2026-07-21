"""Test suite for emission line mapping.

This module contains unit tests for the LineMapper instrument and the
get_line_maps_luminosity/get_line_maps_flux interfaces on both
Components and Galaxies.
"""

import numpy as np
import pytest
from astropy.cosmology import Planck18 as cosmo
from unyt import Mpc, Msun, Myr, kpc

from synthesizer import exceptions
from synthesizer.imaging.image_collection import ImageCollection
from synthesizer.instruments import LineMapper
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.parametric.galaxy import Galaxy as ParametricGalaxy
from synthesizer.parametric.morphology import Gaussian2D
from synthesizer.particle.galaxy import Galaxy as ParticleGalaxy
from synthesizer.particle.stars import Stars


@pytest.fixture
def line_ids(test_grid):
    """Return a small subset of line ids available on the test grid."""
    return list(test_grid.available_lines[:3])


@pytest.fixture
def per_particle_nebular_model(nebular_emission_model):
    """Return a per-particle nebular emission model."""
    nebular_emission_model.set_per_particle(True)
    return nebular_emission_model


@pytest.fixture
def particle_galaxy_with_lines(per_particle_nebular_model, line_ids):
    """Return a particle galaxy with per-particle lines computed."""
    nstars = 12
    rng = np.random.default_rng(42)
    coordinates = rng.normal(0.0, 0.01, (nstars, 3)) * Mpc
    stars = Stars(
        initial_masses=rng.uniform(0.1, 10, nstars) * 1e6 * Msun,
        ages=rng.uniform(4, 7, nstars) * Myr,
        metallicities=rng.uniform(0.01, 0.1, nstars),
        redshift=1.0,
        tau_v=rng.uniform(0.1, 0.9, nstars),
        coordinates=coordinates,
        smoothing_lengths=rng.uniform(0.005, 0.01, nstars) * Mpc,
    )
    galaxy = ParticleGalaxy(
        stars=stars,
        redshift=1.0,
        centre=coordinates.mean(axis=0),
    )
    galaxy.get_lines(line_ids, per_particle_nebular_model)
    galaxy.get_observed_lines(cosmo=cosmo)
    return galaxy


@pytest.fixture
def parametric_galaxy_with_lines(nebular_emission_model, line_ids, test_grid):
    """Return a parametric galaxy with lines computed."""
    stars = ParametricStars(
        test_grid.log10ages,
        test_grid.metallicities,
        sf_hist=1e7 * Myr,
        metal_dist=0.01,
        initial_mass=1e8 * Msun,
    )
    stars.morphology = Gaussian2D(
        x_mean=0 * Mpc,
        y_mean=0 * Mpc,
        stddev_x=0.01 * Mpc,
        stddev_y=0.01 * Mpc,
    )
    galaxy = ParametricGalaxy(stars=stars, redshift=1.0)
    galaxy.get_lines(line_ids, nebular_emission_model)
    galaxy.get_observed_lines(cosmo=cosmo)
    return galaxy


class TestLineMapper:
    """Tests for the LineMapper instrument class."""

    def test_construction(self, line_ids):
        """A LineMapper should construct with just line_ids and resolution."""
        inst = LineMapper("test", line_ids=line_ids, resolution=1 * kpc)
        assert inst.line_ids == line_ids
        assert inst.can_do_line_mapping
        assert not inst.can_do_psf_line_mapping
        assert not inst.can_do_noisy_line_mapping
        assert inst.instrument_type == "line_mapper"

    def test_requires_line_ids(self):
        """LineMapper should reject an empty line_ids list."""
        with pytest.raises(exceptions.MissingArgument):
            LineMapper("test", line_ids=[], resolution=1 * kpc)

    def test_requires_resolution(self, line_ids):
        """LineMapper should reject a missing resolution."""
        with pytest.raises(TypeError):
            LineMapper("test", line_ids=line_ids)

    def test_psf_capability_flag(self, line_ids):
        """can_do_psf_line_mapping should reflect configured PSFs."""
        psf = np.ones((3, 3))
        inst = LineMapper(
            "test",
            line_ids=line_ids,
            resolution=1 * kpc,
            psfs={lid: psf for lid in line_ids},
        )
        assert inst.can_do_psf_line_mapping

    def test_partial_psf_dict_rejected(self, line_ids):
        """A PSF dict missing entries for some lines should be rejected."""
        psf = np.ones((3, 3))
        with pytest.raises(exceptions.MissingArgument):
            LineMapper(
                "test",
                line_ids=line_ids,
                resolution=1 * kpc,
                psfs={line_ids[0]: psf},
            )

    def test_noise_capability_flag(self, line_ids):
        """can_do_noisy_line_mapping should reflect SNR/depth config."""
        from unyt import erg, s

        inst = LineMapper(
            "test",
            line_ids=line_ids,
            resolution=1 * kpc,
            snrs=5.0,
            depth=1e30 * erg / s,
        )
        assert inst.can_do_noisy_line_mapping

    def test_hdf5_roundtrip(self, line_ids, tmp_path):
        """A LineMapper should round-trip through HDF5 serialisation."""
        import h5py

        inst = LineMapper("test", line_ids=line_ids, resolution=1 * kpc)
        path = tmp_path / "line_mapper.hdf5"
        with h5py.File(path, "w") as hdf:
            inst.to_hdf5(hdf)

        with h5py.File(path, "r") as hdf:
            loaded = LineMapper._from_hdf5(hdf)

        assert loaded.line_ids == inst.line_ids
        assert loaded.resolution == inst.resolution


class TestParticleGalaxyLineMaps:
    """Tests for line maps on particle galaxies and components."""

    def test_smoothed_luminosity_maps(
        self, particle_galaxy_with_lines, line_ids, kernel
    ):
        """Smoothed luminosity line maps produce one Image per line."""
        instrument = LineMapper("inst", line_ids=line_ids, resolution=1 * Mpc)
        imgs = particle_galaxy_with_lines.get_line_maps_luminosity(
            "nebular",
            line_ids=line_ids,
            fov=0.1 * Mpc,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )
        assert isinstance(imgs, ImageCollection)
        assert set(imgs.keys()) == set(line_ids)
        for lid in line_ids:
            assert imgs[lid].arr.shape == tuple(imgs.npix)

    def test_hist_luminosity_maps(self, particle_galaxy_with_lines, line_ids):
        """Histogram luminosity line maps should also work."""
        instrument = LineMapper("inst", line_ids=line_ids, resolution=1 * Mpc)
        imgs = particle_galaxy_with_lines.get_line_maps_luminosity(
            "nebular",
            line_ids=line_ids,
            fov=0.1 * Mpc,
            instrument=instrument,
            img_type="hist",
        )
        assert set(imgs.keys()) == set(line_ids)

    def test_flux_maps(self, particle_galaxy_with_lines, line_ids, kernel):
        """Flux line maps should be generated from observed lines."""
        instrument = LineMapper("inst", line_ids=line_ids, resolution=1 * Mpc)
        imgs = particle_galaxy_with_lines.get_line_maps_flux(
            "nebular",
            line_ids=line_ids,
            fov=0.1 * Mpc,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )
        assert set(imgs.keys()) == set(line_ids)

    def test_component_level_maps(
        self, particle_galaxy_with_lines, line_ids, kernel
    ):
        """Component-level get_line_maps_luminosity should work directly."""
        instrument = LineMapper("inst", line_ids=line_ids, resolution=1 * Mpc)
        stars = particle_galaxy_with_lines.stars
        comp_imgs = stars.get_line_maps_luminosity(
            "nebular",
            line_ids=line_ids,
            fov=0.1 * Mpc,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )
        assert set(comp_imgs.keys()) == set(line_ids)

    def test_psf_and_noise_applied(
        self, particle_galaxy_with_lines, line_ids, kernel
    ):
        """PSF and noise should be applied and cached when configured."""
        from unyt import erg, s

        psf = np.outer(np.hanning(9), np.hanning(9))
        instrument = LineMapper(
            "inst-psf",
            line_ids=line_ids,
            resolution=1 * Mpc,
            psfs={lid: psf for lid in line_ids},
            snrs=5.0,
            depth=1e30 * erg / s,
        )
        particle_galaxy_with_lines.get_line_maps_luminosity(
            "nebular",
            line_ids=line_ids,
            fov=0.1 * Mpc,
            instrument=instrument,
            img_type="smoothed",
            kernel=kernel,
        )
        stars = particle_galaxy_with_lines.stars
        assert "inst-psf" in stars.line_maps_psf_lnu
        assert "inst-psf" in stars.line_maps_noise_lnu
        psf_lines = stars.line_maps_psf_lnu["inst-psf"]["nebular"]
        assert set(psf_lines.keys()) == set(line_ids)

    def test_missing_line_raises(self, particle_galaxy_with_lines, kernel):
        """Requesting an ungenerated line should raise MissingLines."""
        instrument = LineMapper(
            "inst", line_ids=["made up line"], resolution=1 * Mpc
        )
        with pytest.raises(exceptions.MissingLines):
            particle_galaxy_with_lines.get_line_maps_luminosity(
                "nebular",
                line_ids=["made up line"],
                fov=0.1 * Mpc,
                instrument=instrument,
                img_type="smoothed",
                kernel=kernel,
            )

    def test_missing_flux_raises(self, per_particle_nebular_model, line_ids):
        """Requesting flux maps before get_observed_lines should raise."""
        nstars = 5
        rng = np.random.default_rng(1)
        coordinates = rng.normal(0.0, 0.01, (nstars, 3)) * Mpc
        stars = Stars(
            initial_masses=rng.uniform(0.1, 10, nstars) * 1e6 * Msun,
            ages=rng.uniform(4, 7, nstars) * Myr,
            metallicities=rng.uniform(0.01, 0.1, nstars),
            redshift=1.0,
            tau_v=rng.uniform(0.1, 0.9, nstars),
            coordinates=coordinates,
            smoothing_lengths=rng.uniform(0.005, 0.01, nstars) * Mpc,
        )
        galaxy = ParticleGalaxy(
            stars=stars, redshift=1.0, centre=coordinates.mean(axis=0)
        )
        galaxy.get_lines(line_ids, per_particle_nebular_model)

        instrument = LineMapper("inst", line_ids=line_ids, resolution=1 * Mpc)
        with pytest.raises(exceptions.MissingAttribute):
            galaxy.get_line_maps_flux(
                "nebular",
                line_ids=line_ids,
                fov=0.1 * Mpc,
                instrument=instrument,
                img_type="hist",
            )


class TestParametricGalaxyLineMaps:
    """Tests for line maps on parametric galaxies."""

    def test_smoothed_luminosity_maps(
        self, parametric_galaxy_with_lines, line_ids
    ):
        """Parametric galaxies produce smoothed maps via density grid."""
        instrument = LineMapper("inst", line_ids=line_ids, resolution=1 * Mpc)
        imgs = parametric_galaxy_with_lines.get_line_maps_luminosity(
            "nebular",
            line_ids=line_ids,
            fov=0.1 * Mpc,
            instrument=instrument,
            img_type="smoothed",
        )
        assert set(imgs.keys()) == set(line_ids)
        for lid in line_ids:
            assert imgs[lid].arr.sum() >= 0

    def test_hist_rejected(self, parametric_galaxy_with_lines, line_ids):
        """Parametric galaxies cannot produce histogram line maps."""
        instrument = LineMapper("inst", line_ids=line_ids, resolution=1 * Mpc)
        with pytest.raises(exceptions.InconsistentArguments):
            parametric_galaxy_with_lines.get_line_maps_luminosity(
                "nebular",
                line_ids=line_ids,
                fov=0.1 * Mpc,
                instrument=instrument,
                img_type="hist",
            )
