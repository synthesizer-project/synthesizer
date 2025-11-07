"""Tests for the Particles base class and its methods."""

import numpy as np
import pytest
from unyt import Mpc, Msun, km, rad, s, unyt_array

from synthesizer import exceptions
from synthesizer.particle.particles import CoordinateGenerator, Particles


class TestParticlesInitialization:
    """Tests for initializing the Particles base class."""

    def test_init_assigns_all_attributes(self):
        """Particles assign all provided attributes."""
        coords = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]) * Mpc
        vels = np.array([[0.0, 10.0, 0.0], [20.0, 0.0, 0.0]]) * (km / s)
        masses = np.array([1.0e10, 2.0e10]) * Msun
        redshift = 0.5
        soft = 0.01 * Mpc
        centre = np.zeros(3) * Mpc

        sp = Particles(
            coordinates=coords,
            velocities=vels,
            masses=masses,
            redshift=redshift,
            softening_lengths=soft,
            nparticles=2,
            centre=centre,
        )

        assert sp.nparticles == 2
        np.testing.assert_allclose(sp.coordinates, coords)
        np.testing.assert_allclose(sp.velocities, vels)
        np.testing.assert_allclose(sp.masses, masses)
        assert sp.redshift == redshift
        np.testing.assert_allclose(sp.softening_lengths, soft)
        np.testing.assert_allclose(sp.centre, centre)
        assert sp.particle_spectra == {}
        assert sp.particle_lines == {}
        assert sp.particle_photo_lnu == {}
        assert sp.particle_photo_fnu == {}
        assert sp.radii is None


class TestParticlesCentreAndRadii:
    """Tests for centre-of-mass, radii computations, and aperture mask."""

    @pytest.fixture
    def simple_parts(self):
        """Create a simple Particles instance with two particles."""
        coords = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]) * Mpc
        vels = np.zeros((2, 3)) * (km / s)
        masses = np.array([1.0e10, 2.0e10]) * Msun
        centre = np.zeros(3) * Mpc
        return Particles(
            coordinates=coords,
            velocities=vels,
            masses=masses,
            redshift=0.1,
            softening_lengths=0.01 * Mpc,
            nparticles=2,
            centre=centre,
        )

    def test_calculate_centre_of_mass(self, simple_parts):
        """Compute correct mass-weighted centre."""
        sp = simple_parts
        sp.calculate_centre_of_mass()
        # calculate_centre_of_mass sets attribute 'center'
        expected = np.array([1 / 3, 4 / 3, 0.0])
        np.testing.assert_allclose(sp.center, expected)

    def test_get_radii_success(self, simple_parts):
        """get_radii should return distance norms and store to sp.radii."""
        sp = simple_parts
        radii = sp.get_radii()
        np.testing.assert_allclose(radii.to_value(Mpc), [1.0, 2.0])
        np.testing.assert_allclose(sp.radii.to_value(Mpc), [1.0, 2.0])

    def test_get_radii_without_centre_raises(self, simple_parts):
        """get_radii should raise if centre is None."""
        sp = simple_parts
        sp.centre = None
        with pytest.raises(exceptions.InconsistentArguments):
            sp.get_radii()

    def test__aperture_mask_behavior(self, simple_parts):
        """_aperture_mask selects correct particles within given radius."""
        sp = simple_parts
        mask = sp._aperture_mask(1.5 * Mpc)
        assert mask.tolist() == [True, False]

    def test__aperture_mask_without_centre(self, simple_parts):
        """_aperture_mask should error if centre is not set."""
        sp = simple_parts
        sp.centre = None
        with pytest.raises(ValueError):
            sp._aperture_mask(1.0 * Mpc)


class TestParticlesRadiusCalculations:
    """Tests for internal _get_radius and radius convenience methods."""

    @pytest.fixture
    def parts(self):
        """Create a Particles instance with three particles and radii."""
        sp = Particles(
            coordinates=np.zeros((3, 3)) * Mpc,
            velocities=np.zeros((3, 3)) * (km / s),
            masses=np.ones(3) * Msun,
            redshift=0.1,
            softening_lengths=0.01 * Mpc,
            nparticles=3,
            centre=np.zeros(3) * Mpc,
        )
        sp.radii = np.array([1.0, 2.0, 4.0]) * Mpc
        return sp

    @pytest.mark.parametrize(
        "nparticles,weights,frac,expected",
        [
            (0, np.array([]), 0.5, 0.0),
            (1, np.array([5.0]), 0.3, 0.3),
            (2, np.array([1.0, 1.0]), 0.0, 0.0),
            (2, np.array([1.0, 1.0]), 1.0, 2.0),
            (2, np.array([0.0, 0.0]), 0.5, 0.0),
        ],
    )
    def test__get_radius_special_cases(
        self, nparticles, weights, frac, expected
    ):
        """Handle edge cases for zero particles, zero weights, and frac=0/1."""
        sp = Particles(
            coordinates=np.zeros((nparticles, 3)) * Mpc,
            velocities=np.zeros((nparticles, 3)) * (km / s),
            masses=np.ones(nparticles) * Msun,
            redshift=0.1,
            softening_lengths=0.01 * Mpc,
            nparticles=nparticles,
            centre=np.zeros(3) * Mpc,
        )
        sp.radii = (
            np.array([1.0, 2.0] + [0.0] * max(0, nparticles - 2))[:nparticles]
            * Mpc
        )
        r = sp._get_radius(weights, frac)
        assert pytest.approx(expected) == r.to_value(Mpc)

    def test__get_radius_interpolation(self, parts):
        """Interpolate correctly for intermediate frac values."""
        sp = parts
        weights = np.array([1.0, 2.0, 1.0])
        r = sp._get_radius(weights, 0.5)
        # interpolation yields 1.5
        assert pytest.approx(1.5) == r.to_value(Mpc)

    def test_get_half_mass_radius(self, parts):
        """get_half_mass_radius should return correct half-mass radius."""
        r = parts.get_half_mass_radius()
        assert pytest.approx(1.5) == r.to_value(Mpc)


class TestParticlesPhotometry:
    """Tests for photometry luminosity and flux getters."""

    @pytest.fixture
    def phot_parts(self):
        """Create a Particles instance with dummy spectra."""
        sp = Particles(
            coordinates=np.zeros((1, 3)) * Mpc,
            velocities=np.zeros((1, 3)) * (km / s),
            masses=np.ones(1) * Msun,
            redshift=0.1,
            softening_lengths=0.01 * Mpc,
            nparticles=1,
            centre=np.zeros(3) * Mpc,
        )

        class DummySpec:
            def __init__(self):
                self.called_lnu = False
                self.called_fnu = False

            def get_photo_lnu(self, filters, verbose, nthreads):
                self.called_lnu = True
                return {"F": unyt_array([1.0], "erg/s/Hz")}

            def get_photo_fnu(self, filters, verbose, nthreads):
                self.called_fnu = True
                return {"F": unyt_array([2.0], "erg/s/cm**2/Hz")}

        sp.particle_spectra["D"] = DummySpec()
        return sp

    def test_get_particle_photo_lnu_and_property(self, phot_parts):
        """Deprecated property should match."""
        sp = phot_parts
        out = sp.get_particle_photo_lnu(
            filters=None, verbose=False, nthreads=1
        )
        assert "D" in out
        assert "D" in sp.particle_photo_lnu
        assert str(out["D"]["F"].units) == "erg/(Hz*s)"
        assert sp.particle_photo_luminosities is sp.particle_photo_lnu

    def test_get_particle_photo_fnu_and_property(self, phot_parts):
        """Deprecated property should match."""
        sp = phot_parts
        out = sp.get_particle_photo_fnu(
            filters=None, verbose=False, nthreads=1
        )
        assert "D" in out
        assert "D" in sp.particle_photo_fnu
        assert str(out["D"]["F"].units) == "erg/(Hz*cm**2*s)"
        assert sp.particle_photo_fluxes is sp.particle_photo_fnu


class TestParticlesProjection:
    """Tests for projected angular coordinates and smoothing lengths."""

    @pytest.fixture
    def simple_parts(self):
        """Create a simple Particles instance with two particles."""
        coords = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]) * Mpc
        sp = Particles(
            coordinates=coords,
            velocities=np.zeros((2, 3)) * (km / s),
            masses=np.ones(2) * Msun,
            redshift=0.0,
            softening_lengths=0.01 * Mpc,
            nparticles=2,
            centre=np.zeros(3) * Mpc,
        )
        # set smoothing lengths for test
        sp._smoothing_lengths = np.array([0.5, 1.0])
        sp.smoothing_lengths = sp._smoothing_lengths * Mpc

        # mock luminosity distance
        def fake_ld(cosmo):
            return np.array([10.0, 10.0]) * Mpc

        sp.get_luminosity_distance = fake_ld
        sp.get_angular_diameter_distance = fake_ld
        return sp

    def test_get_projected_angular_coordinates_requires_input(
        self, simple_parts
    ):
        """Require cosmo or los_dists."""
        sp = simple_parts
        with pytest.raises(exceptions.InconsistentArguments):
            sp.get_projected_angular_coordinates()

    def test_get_projected_angular_coordinates_los(self, simple_parts):
        """get_projected_angular_coordinates returns correct arctan values."""
        sp = simple_parts
        los = np.array([10.0, 10.0]) * Mpc
        ang = sp.get_projected_angular_coordinates(los_dists=los)
        assert ang.shape == (2, 3)
        assert pytest.approx(np.arctan2(1.0, 10.0)) == ang[0, 0].to_value(rad)

    def test_get_projected_angular_smoothing_lengths(self, simple_parts):
        """Returns correct shape and units."""
        sp = simple_parts
        los = np.array([10.0, 20.0]) * Mpc
        sml = sp.get_projected_angular_smoothing_lengths(los_dists=los)
        assert sml.shape == (2,)
        assert sml.units == rad

    def test_get_projected_angular_imaging_props(self, simple_parts):
        """Combine coords and smoothing lengths."""
        sp = simple_parts
        coords, smls = sp.get_projected_angular_imaging_props(cosmo=None)
        assert coords.shape[0] == smls.shape[0]
        assert coords.units == rad and smls.units == rad


class TestParticlesRotation:
    """Tests for rotation methods including edge-on and face-on."""

    @pytest.fixture
    def simple_parts(self):
        """Create a simple Particles instance with one particle."""
        coords = np.array([[1.0, 0.0, 0.0]]) * Mpc
        vels = np.array([[0.0, 1.0, 0.0]]) * (km / s)
        return Particles(
            coordinates=coords,
            velocities=vels,
            masses=np.ones(1) * Msun,
            redshift=0.1,
            softening_lengths=0.01 * Mpc,
            nparticles=1,
            centre=np.zeros(3) * Mpc,
        )

    def test_rotate_particles_inplace_and_copy(self, simple_parts):
        """rotate_particles should modify in-place."""
        sp = simple_parts
        before = sp.coordinates.copy()
        sp.rotate_particles(phi=0 * rad, theta=0 * rad, inplace=True)
        np.testing.assert_allclose(sp.coordinates, before)
        new = sp.rotate_particles(phi=0 * rad, theta=0 * rad, inplace=False)
        assert new is not sp

    def test_rotate_edge_on_and_face_on(self, simple_parts):
        """rotate_edge_on and rotate_face_on should return Particles."""
        sp = simple_parts
        edge = sp.rotate_edge_on(inplace=False)
        face = sp.rotate_face_on(inplace=False)
        assert isinstance(edge, Particles)
        assert isinstance(face, Particles)


class TestParticlesAngularMomentum:
    """Tests for angular momentum property and its error conditions."""

    @pytest.fixture
    def simple_parts(self):
        """Create a simple Particles instance with one particle."""
        coords = np.array([[1.0, 0.0, 0.0]]) * Mpc
        vels = np.array([[0.0, 1.0, 0.0]]) * (km / s)
        return Particles(
            coordinates=coords,
            velocities=vels,
            masses=np.ones(1) * Msun,
            redshift=0.1,
            softening_lengths=0.01 * Mpc,
            nparticles=1,
            centre=np.zeros(3) * Mpc,
        )

    def test_angular_momentum_correctness(self, simple_parts):
        """angular_momentum should compute r x v * m correctly."""
        sp = simple_parts
        manual = (
            np.cross(
                sp.coordinates.to_value(Mpc), sp.velocities.to_value(km / s)
            )
            * sp.masses.to_value(Msun)[:, None]
        )
        expected = np.sum(manual, axis=0)
        out = sp.angular_momentum.to_value("Mpc*Msun*km/s")
        assert np.allclose(out, expected)

    @pytest.mark.parametrize("attr", ["coordinates", "velocities", "masses"])
    def test_angular_momentum_missing_attr_raises(self, simple_parts, attr):
        """angular_momentum should raise if a required attribute is None."""
        sp = simple_parts
        setattr(sp, attr, None)
        with pytest.raises(exceptions.InconsistentArguments):
            _ = sp.angular_momentum


class TestParticlesMasking:
    """Tests for the get_mask method with various operations and errors."""

    @pytest.fixture
    def simple_parts(self):
        """Create a simple Particles instance with three particles."""
        coords = np.zeros((3, 3)) * Mpc
        vels = np.zeros((3, 3)) * (km / s)
        masses = unyt_array([1, 2, 3], "Msun")
        sp = Particles(
            coordinates=coords,
            velocities=vels,
            masses=masses,
            redshift=0.1,
            softening_lengths=0.01 * Mpc,
            nparticles=3,
            centre=np.zeros(3) * Mpc,
        )
        sp.dummy = np.array([5.0, 10.0, 15.0])
        return sp

    def test_get_mask_basic_operations(self, simple_parts):
        """get_mask should return correct boolean masks for operations."""
        sp = simple_parts
        m1 = sp.get_mask("dummy", 10.0, ">=")
        assert m1.tolist() == [False, True, True]
        m2 = sp.get_mask("dummy", 10.0, ">", mask=m1)
        assert m2.tolist() == [False, False, True]

    def test_get_mask_none_attribute_raises(self, simple_parts):
        """get_mask should raise if attribute is None."""
        sp = simple_parts
        sp.dummy = None
        with pytest.raises(exceptions.MissingMaskAttribute):
            sp.get_mask("dummy", 1.0, "<")

    def test_get_mask_inconsistent_units_raises(self, simple_parts):
        """get_mask should error if only one of attr/thresh has units."""
        sp = simple_parts
        # attr has units, thresh none
        with pytest.raises(exceptions.InconsistentArguments):
            sp.get_mask("masses", 1.0, "==")
        # thresh has units, attr none
        with pytest.raises(exceptions.InconsistentArguments):
            sp.get_mask("dummy", 1.0 * Mpc, "==")

    def test_get_mask_invalid_op_raises(self, simple_parts):
        """get_mask should raise for invalid comparison operator."""
        sp = simple_parts
        with pytest.raises(exceptions.InconsistentArguments):
            sp.get_mask("dummy", 5.0, "invalid")


class TestWeightedAttributes:
    """Tests for weighted attribute and luminosity/flux weighting."""

    def test_get_weighted_attr_array_and_string(self, unit_mass_stars):
        """Check both array and attribute-string weights."""
        w_arr = unit_mass_stars.current_masses.to_value(Msun)
        out1 = unit_mass_stars.get_weighted_attr("coordinates", w_arr, axis=0)
        out2 = unit_mass_stars.get_weighted_attr(
            "coordinates", "current_masses", axis=0
        )
        assert out1 == pytest.approx(out2)

    def test_lum_and_flux_weighted_attr(self, unit_emission_stars):
        """Check get_lum_weighted_attr and get_flux_weighted_attr equal."""
        stars = unit_emission_stars
        z = stars.coordinates.to_value("Mpc")[:, 2]
        expected = np.average(z, weights=np.ones_like(z))
        assert stars.get_lum_weighted_attr(
            "coordinates", "FAKE", "fake", axis=0
        ) == pytest.approx(expected)
        assert stars.get_flux_weighted_attr(
            "coordinates", "FAKE", "fake", axis=0
        ) == pytest.approx(expected)


class TestCoordinateGenerator:
    """Tests for the CoordinateGenerator utility methods."""

    def test_generate_3D_gaussian_shape(self):
        """generate_3D_gaussian should produce (n,3) array."""
        pts = CoordinateGenerator.generate_3D_gaussian(5)
        assert pts.shape == (5, 3)

    def test_generate_2D_Sersic_unimplemented(self):
        """generate_2D_Sersic should raise UnimplementedFunctionality."""
        with pytest.raises(exceptions.UnimplementedFunctionality):
            CoordinateGenerator.generate_2D_Sersic(10)

    def test_generate_3D_spline_unimplemented(self):
        """generate_3D_spline should raise UnimplementedFunctionality."""
        with pytest.raises(exceptions.UnimplementedFunctionality):
            CoordinateGenerator.generate_3D_spline(5, kernel_func=lambda x: x)
