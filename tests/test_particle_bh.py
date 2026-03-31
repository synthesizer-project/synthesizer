"""Test suite for particle based black holes."""

import numpy as np
from unyt import Msun, s, unyt_array

from synthesizer.particle.blackholes import BlackHoles
from synthesizer.utils import scalar_to_array


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
