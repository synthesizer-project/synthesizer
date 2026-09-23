"""Tests for the optional NUMA interleave policy."""

import pytest

from synthesizer._numa import maybe_interleave_memory


@pytest.mark.parametrize(
    "value",
    ["", "0", "false", "no", "FALSE", "off", "OFF", " 0 ", "maybe"],
)
def test_interleave_off_by_default(monkeypatch, value):
    """The policy is only applied when the environment asks for it.

    Anything that is not an explicit yes leaves the policy alone. "off" is the
    one that matters: the docs say the feature is off by default, so it is
    what a user reaches for, and a deny list would have read it as a yes.
    """
    monkeypatch.setenv("SYNTHESIZER_NUMA_INTERLEAVE", value)
    assert maybe_interleave_memory() is False


def test_interleave_unset_is_off(monkeypatch):
    """An absent variable behaves like an explicit off."""
    monkeypatch.delenv("SYNTHESIZER_NUMA_INTERLEAVE", raising=False)
    assert maybe_interleave_memory() is False


def test_interleave_on_never_raises(monkeypatch):
    """Requesting the policy returns a bool on every platform.

    libnuma is absent on macOS and on Linux images without numactl, so the
    helper has to degrade to False rather than raise on import of the package.
    """
    monkeypatch.setenv("SYNTHESIZER_NUMA_INTERLEAVE", "1")
    assert isinstance(maybe_interleave_memory(), bool)
