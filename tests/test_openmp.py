"""Tests for OpenMP integration."""

from synthesizer.extensions.openmp_check import check_openmp


def test_openmp():
    """Test the openmp_check extension."""
    assert check_openmp()
