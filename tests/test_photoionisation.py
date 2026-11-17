"""Tests for the photoionisation utilities."""

import numpy as np
import pytest
from unyt import cm, m, s

from synthesizer import exceptions
from synthesizer.photoionisation import calculate_Q_from_U, calculate_U_from_Q


def test_q_u_round_trip():
    """calculate_U_from_Q inverts calculate_Q_from_U."""
    q = calculate_Q_from_U(1e-2, 100.0 * cm**-3)
    assert q.units == s**-1
    u = calculate_U_from_Q(q, 100.0 * cm**-3)
    assert u.units.is_dimensionless
    assert np.isclose(u, 1e-2)


def test_density_units_are_converted():
    """A density in other units gives the same U."""
    q = 1e50 / s
    np.testing.assert_allclose(
        calculate_U_from_Q(q, 1e8 * m**-3),
        calculate_U_from_Q(q, 100.0 * cm**-3),
    )


def test_arrays():
    """Arrays of Q and n_h give an array of U (e.g. per particle)."""
    q = np.array([1e49, 1e50, 1e51]) / s
    n_h = np.array([10.0, 100.0, 1000.0]) * cm**-3
    u = calculate_U_from_Q(q, n_h)
    assert u.shape == (3,)
    np.testing.assert_allclose(calculate_Q_from_U(u.value, n_h), q)


def test_missing_units_raise():
    """A density without units is rejected."""
    with pytest.raises(exceptions.MissingUnits):
        calculate_Q_from_U(1e-2, 100.0)
