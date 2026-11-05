"""Tests for the photoionisation utilities."""

import numpy as np

from synthesizer.photoionisation import calculate_Q_from_U, calculate_U_from_Q


def test_q_u_round_trip():
    """calculate_U_from_Q inverts calculate_Q_from_U."""
    q = calculate_Q_from_U(1e-2, 100.0)
    assert np.isclose(calculate_U_from_Q(q, 100.0), 1e-2)
