"""Tests for the ``accepts`` decorator."""

import numpy as np
import pytest
from unyt import Msun, angstrom, cm, g, km, m, pc, s

from synthesizer import exceptions
from synthesizer.units import accepts


def test_accepts_converts_positional_arguments():
    """Positional arguments should be converted to the requested units."""

    @accepts(length=m)
    def func(length):
        return length

    out = func(1.0 * km)

    assert np.isclose(out.to(m).value, 1000.0)


def test_accepts_converts_keyword_arguments():
    """Keyword arguments should also be converted to the requested units."""

    @accepts(length=m)
    def func(*, length):
        return length

    out = func(length=2.0 * km)

    assert np.isclose(out.to(m).value, 2000.0)


def test_accepts_allows_none_for_optional_arguments():
    """None should pass through unchanged for optional unit-checked args."""

    @accepts(length=m)
    def func(length=None):
        return length

    assert func(None) is None


def test_accepts_raises_for_incompatible_units():
    """Incompatible units should raise the synthesizer unit error."""

    @accepts(length=m)
    def func(length):
        return length

    with pytest.raises(exceptions.IncorrectUnits):
        func(1.0 * s)


def test_accepts_supports_multiple_allowed_units():
    """A tuple of allowed units should accept any compatible option."""

    @accepts(value=(m, s))
    def func(value):
        return value

    out = func(3.0 * km)

    assert np.isclose(out.to(m).value, 3000.0)


def test_accepts_converts_all_values_in_var_positional_parameter():
    """Naming a ``*args`` parameter should validate every stored value."""

    @accepts(lengths=m)
    def func(*lengths):
        return lengths

    out = func(1.0 * km, 20.0 * cm)

    assert len(out) == 2
    assert np.isclose(out[0].to(m).value, 1000.0)
    assert np.isclose(out[1].to(m).value, 0.2)


def test_accepts_var_positional_parameter_raises_for_bad_units():
    """Any incompatible value in a ``*args`` tuple should raise."""

    @accepts(lengths=m)
    def func(*lengths):
        return lengths

    with pytest.raises(exceptions.IncorrectUnits):
        func(1.0 * km, 2.0 * s)


def test_accepts_leaves_unregistered_var_positional_values_untouched():
    """Unnamed ``*args`` rules should preserve the incoming values."""

    @accepts(length=m)
    def func(length, *extras):
        return length, extras

    length, extras = func(1.0 * km, 2.0, "hello")

    assert np.isclose(length.to(m).value, 1000.0)
    assert extras == (2.0, "hello")


def test_accepts_converts_all_values_in_var_keyword_parameter():
    """Naming a ``**kwargs`` parameter should validate every stored value."""

    @accepts(lam=angstrom, sigmalos_dust=Msun / pc**2)
    def func(lam, **sigmalos_dust):
        return lam, sigmalos_dust

    lam, sigmalos_dust = func(
        1500 * angstrom,
        graphite=1.0 * Msun / pc**2,
        silicate=2.0 * g / pc**2,
    )

    assert np.isclose(lam.to(angstrom).value, 1500.0)
    assert np.isclose(sigmalos_dust["graphite"].to(Msun / pc**2).value, 1.0)
    assert np.isclose(
        sigmalos_dust["silicate"].to(Msun / pc**2).value,
        (2.0 * g / pc**2).to(Msun / pc**2).value,
    )


def test_accepts_var_keyword_parameter_raises_for_bad_units():
    """Any incompatible value in a ``**kwargs`` dict should raise."""

    @accepts(sigmalos_dust=Msun / pc**2)
    def func(**sigmalos_dust):
        return sigmalos_dust

    with pytest.raises(exceptions.IncorrectUnits):
        func(graphite=1.0 * angstrom)


def test_accepts_preserves_legacy_matching_inside_var_keyword_dict():
    """Explicit unit rules for individual kwarg names should still work."""

    @accepts(graphite=Msun / pc**2)
    def func(**sigmalos_dust):
        return sigmalos_dust

    sigmalos_dust = func(
        graphite=1.0 * g / pc**2,
        silicate=np.array([1.0, 2.0]),
    )

    assert np.isclose(
        sigmalos_dust["graphite"].to(Msun / pc**2).value,
        (1.0 * g / pc**2).to(Msun / pc**2).value,
    )
    assert np.all(sigmalos_dust["silicate"] == np.array([1.0, 2.0]))


def test_accepts_leaves_unregistered_var_keyword_values_untouched():
    """Values in ``**kwargs`` remain unchanged if no rule applies to them."""

    @accepts(lam=angstrom)
    def func(lam, **sigmalos_dust):
        return lam, sigmalos_dust

    lam, sigmalos_dust = func(1500 * angstrom, graphite=np.array([1.0, 2.0]))

    assert np.isclose(lam.to(angstrom).value, 1500.0)
    assert np.all(sigmalos_dust["graphite"] == np.array([1.0, 2.0]))


def test_accepts_var_keyword_parameter_handles_empty_dict():
    """An empty ``**kwargs`` dict should be accepted cleanly."""

    @accepts(sigmalos_dust=Msun / pc**2)
    def func(**sigmalos_dust):
        return sigmalos_dust

    assert func() == {}


def test_accepts_handles_positional_and_var_keyword_together():
    """Standard and ``**kwargs`` validation should compose correctly."""

    @accepts(lam=angstrom, sigmalos_dust=Msun / pc**2)
    def func(lam, **sigmalos_dust):
        return lam, sigmalos_dust

    lam, sigmalos_dust = func(5000 * angstrom, graphite=3.0 * g / pc**2)

    assert np.isclose(lam.to(angstrom).value, 5000.0)
    assert np.isclose(
        sigmalos_dust["graphite"].to(Msun / pc**2).value,
        (3.0 * g / pc**2).to(Msun / pc**2).value,
    )


def test_accepts_var_keyword_preserves_arrays():
    """Array values inside a ``**kwargs`` dict should remain array-shaped."""

    @accepts(sigmalos_dust=Msun / pc**2)
    def func(**sigmalos_dust):
        return sigmalos_dust

    out = func(graphite=np.array([1.0, 2.0]) * g / cm**2)

    assert out["graphite"].shape == (2,)
