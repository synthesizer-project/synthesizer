"""Test suite for particle based black holes."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from unyt import K, Msun, g, kg, s, unyt_array, yr

from synthesizer import exceptions
from synthesizer.particle.blackholes import BlackHoles
from synthesizer.utils import scalar_to_array


def test_get_spectra_defaults_to_global_out_dtype(particle_black_hole):
    """get_spectra should defer to the global default output dtype."""
    from synthesizer.utils.precision import resolve_out_dtype

    emission_model = MagicMock()
    emission_model.per_particle = False
    emission_model.label = "mock"
    emission_model._get_spectra.return_value = ({"mock": MagicMock()}, {})

    particle_black_hole.get_spectra(emission_model)

    # The out_dtype kwarg passes through unresolved (None) and resolves to
    # the global default (float64 unless changed) at the point of use.
    passed = emission_model._get_spectra.call_args.kwargs["out_dtype"]
    assert passed is None
    assert resolve_out_dtype(passed) == np.float64


class TestBlackHolesInit:
    """Test suite for initialising BlackHoles instances."""

    def test_scalar_to_array(self):
        """Test that scalar_to_array works in various situations."""
        # Scalar with no units
        arr = scalar_to_array(1)
        assert isinstance(arr, np.ndarray), (
            f"Scalar with no units failed: 1->{arr}"
        )

        # Scalar with units
        arr = scalar_to_array(1 * s)
        assert isinstance(arr, unyt_array), (
            f"Scalar with units failed: 1 * s->{arr}"
        )
        assert arr.units == s, f"Scalar with units failed: 1 * s->{arr}"
        assert arr.shape == (1,), (
            f"Scalar with units shape is wrong: {arr.shape} "
            f"(value: {arr}, type: {type(arr)})"
        )

        # Check that an array without units is returned as is
        arr = scalar_to_array(np.arange(10))
        assert isinstance(arr, np.ndarray), (
            f"Array without units failed: {np.arange(10)}->{arr}"
        )

        # Check that an array with units is returned as is
        arr = scalar_to_array(np.arange(10) * s)
        assert isinstance(arr, unyt_array), (
            f"Array with units failed: {np.arange(10) * s}->{arr}"
        )
        assert arr.units == s, (
            f"Array with units failed: {np.arange(10) * s}->{arr}"
        )

        # Check that a ndim = 2 array without units is returned as is
        arr = scalar_to_array(np.arange(10).reshape(2, 5))
        assert isinstance(arr, np.ndarray), (
            "2D array without units failed: "
            f"{np.arange(10).reshape(2, 5)}->{arr}"
        )

        # Check that a ndim = 2 array with units is returned as is
        arr = scalar_to_array(np.arange(10).reshape(2, 5) * s)
        assert isinstance(arr, unyt_array), (
            "2D array with units failed:"
            f" {np.arange(10).reshape(2, 5) * s}->{arr}"
        )

        # Check that a 1 element aray without units is returned as is
        arr = scalar_to_array(np.array([1]))
        assert isinstance(arr, np.ndarray), (
            f"1 element array without units failed: {np.array([1])}->{arr}"
        )
        assert arr.shape == (1,), (
            f"1 element array without units failed: {np.array([1])}->{arr}"
        )
        assert arr.ndim == 1, (
            f"1 element array without units failed: {np.array([1])}->{arr}"
        )

    def test_transmission_fractions_store_covering_fractions(self):
        """Test deterministic transmission fractions mirror inputs."""
        bhs = BlackHoles(
            masses=np.array([1e8, 2e8]) * Msun,
            covering_fraction_blr=np.array([0.2, 0.3]),
            covering_fraction_nlr=np.array([0.1, 0.4]),
        )

        np.testing.assert_allclose(bhs.transmission_fraction_blr, [0.2, 0.3])
        np.testing.assert_allclose(bhs.transmission_fraction_nlr, [0.1, 0.4])
        np.testing.assert_allclose(
            bhs.transmission_fraction_escape,
            [0.7, 0.3],
        )
        np.testing.assert_allclose(bhs.covering_fraction, [0.3, 0.7])
        np.testing.assert_allclose(bhs.escape_fraction, [0.7, 0.3])

    def test_random_transmission_fractions_are_one_hot(self):
        """Test random transmission fractions are consistent one-hot draws."""
        np.random.seed(0)
        bhs = BlackHoles(
            masses=np.array([1e8, 2e8, 3e8]) * Msun,
            covering_fraction_blr=np.array([0.2, 0.3, 0.0]),
            covering_fraction_nlr=np.array([0.1, 0.4, 0.5]),
        )

        random_total = (
            bhs.random_transmission_fraction_escape
            + bhs.random_transmission_fraction_nlr
            + bhs.random_transmission_fraction_blr
        )
        np.testing.assert_allclose(random_total, np.ones(3))

        assert np.all(
            np.isin(
                bhs.random_transmission_fraction_escape,
                [0.0, 1.0],
            )
        )
        assert np.all(
            np.isin(
                bhs.random_transmission_fraction_nlr,
                [0.0, 1.0],
            )
        )
        assert np.all(
            np.isin(
                bhs.random_transmission_fraction_blr,
                [0.0, 1.0],
            )
        )

    @pytest.mark.parametrize(
        ("blr", "nlr", "message"),
        [
            (np.array([-0.1]), np.array([0.2]), "covering_fraction_blr"),
            (np.array([0.2]), np.array([-0.1]), "covering_fraction_nlr"),
            (np.array([1.1]), np.array([0.0]), "covering_fraction_blr"),
            (np.array([0.0]), np.array([1.1]), "covering_fraction_nlr"),
        ],
    )
    def test_invalid_covering_fractions_raise(self, blr, nlr, message):
        """Test invalid covering fractions are rejected individually."""
        with pytest.raises(exceptions.InconsistentArguments, match=message):
            BlackHoles(
                masses=np.array([1e8]) * Msun,
                covering_fraction_blr=blr,
                covering_fraction_nlr=nlr,
            )


class TestBlueBumpTemperature:
    """Test suite for the blue bump temperature."""

    def test_calculated_from_mass_and_accretion_rate(self):
        """Test the thin disc temperature at 10 Schwarzschild radii.

        T scales as Mdot^(1/4) M^(-1/2), with 7.456e4 K at M = 1e8 Msun and
        Mdot = 1 Msun/yr.
        """
        bhs = BlackHoles(
            masses=unyt_array([1e7, 1e8, 1e9], Msun),
            accretion_rates=unyt_array([1.0, 1.0, 16.0], Msun / yr),
        )

        for name in ("blue_bump_temperature", "blue_bump_temperatures"):
            np.testing.assert_allclose(
                getattr(bhs, name).to("K").value,
                7.4564e4 * np.array([10**0.5, 1.0, 10**-0.5 * 2]),
                rtol=1e-4,
            )

    def test_independent_of_input_units(self):
        """Test masses and accretion rates in other units give the same T."""
        bhs_msun = BlackHoles(
            masses=unyt_array([1e8], Msun),
            accretion_rates=unyt_array([1.0], Msun / yr),
        )
        bhs_cgs = BlackHoles(
            masses=unyt_array([1e8], Msun).to(kg),
            accretion_rates=unyt_array([1.0], Msun / yr).to(g / s),
        )

        np.testing.assert_allclose(
            bhs_cgs.blue_bump_temperature.to("K").value,
            bhs_msun.blue_bump_temperature.to("K").value,
        )

    @pytest.mark.parametrize(
        "kwarg", ["blue_bump_temperature", "blue_bump_temperatures"]
    )
    def test_user_value_is_kept(self, kwarg):
        """Test a user set temperature is not replaced by the calculation.

        The value can be passed with the singular or plural name, and is
        then available under both.
        """
        bhs = BlackHoles(
            masses=unyt_array([1e8, 1e9], Msun),
            accretion_rates=unyt_array([1.0, 1.0], Msun / yr),
            **{kwarg: unyt_array([1e5, 2e5], K)},
        )

        for name in ("blue_bump_temperature", "blue_bump_temperatures"):
            np.testing.assert_allclose(
                getattr(bhs, name).to("K").value, [1e5, 2e5]
            )

    @pytest.mark.parametrize(
        "kwarg", ["blue_bump_temperature", "blue_bump_temperatures"]
    )
    def test_none_is_calculated(self, kwarg):
        """Test a temperature passed as None is replaced by the calculation."""
        bhs = BlackHoles(
            masses=unyt_array([1e8], Msun),
            accretion_rates=unyt_array([1.0], Msun / yr),
            **{kwarg: None},
        )

        for name in ("blue_bump_temperature", "blue_bump_temperatures"):
            np.testing.assert_allclose(
                getattr(bhs, name).to("K").value, [7.4564e4], rtol=1e-4
            )

    def test_not_calculated_without_accretion_rate(self):
        """Test no temperature is calculated without an accretion rate."""
        bhs = BlackHoles(masses=unyt_array([1e8], Msun))

        assert not hasattr(bhs, "blue_bump_temperature")
        assert not hasattr(bhs, "blue_bump_temperatures")
