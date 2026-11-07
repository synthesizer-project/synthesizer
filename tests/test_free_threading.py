"""Tests for the free-threading utilities and GIL-safe extension modules."""

import importlib.util
import os
import subprocess
import sys
import sysconfig

import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.utils import free_threading
from synthesizer.utils.free_threading import resolve_threads

FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.fixture
def no_gil(monkeypatch):
    """Pretend the GIL is disabled so threads are used."""
    monkeypatch.setattr(free_threading, "gil_enabled", lambda: False)


@pytest.fixture
def four_cores(monkeypatch):
    """Pretend there are 4 cores available."""
    monkeypatch.setattr(
        free_threading.os, "process_cpu_count", lambda: 4, raising=False
    )
    monkeypatch.setattr(free_threading.os, "cpu_count", lambda: 4)


class TestResolveThreads:
    """Test resolving a requested number of threads."""

    def test_serial_is_unchanged(self):
        """Test a single thread is always allowed."""
        assert resolve_threads(1, "nr_threads") == 1

    @pytest.mark.parametrize("value", [0, -2, 1.5, "4", True, None])
    def test_invalid_values_raise(self, value):
        """Test anything other than a positive integer is rejected."""
        with pytest.raises(exceptions.InconsistentArguments, match="nr_x"):
            resolve_threads(value, "nr_x")

    def test_numpy_integers_accepted(self, no_gil):
        """Test numpy integers are accepted and converted to int."""
        nthreads = resolve_threads(np.int64(2), "nr_threads")
        assert nthreads == 2
        assert type(nthreads) is int

    def test_gil_falls_back_to_serial(self, monkeypatch):
        """Test threads are disabled with a warning under the GIL."""
        monkeypatch.setattr(free_threading, "gil_enabled", lambda: True)
        with pytest.warns(RuntimeWarning, match="nr_x=4 .* GIL is enabled"):
            assert resolve_threads(4, "nr_x") == 1

    def test_no_gil_uses_threads(self, no_gil):
        """Test the requested threads are used without the GIL."""
        assert resolve_threads(2, "nr_threads") == 2

    def test_oversubscription_warns(self, no_gil, four_cores):
        """Test a warning when threads x OpenMP threads exceed the cores."""
        with pytest.warns(RuntimeWarning, match="only 4 cores"):
            assert resolve_threads(4, "nr_threads", nthreads=2) == 4

    def test_all_cores_nthreads_warns(self, no_gil, four_cores):
        """Test nthreads=-1 counts as one OpenMP thread per core."""
        with pytest.warns(RuntimeWarning, match="only 4 cores"):
            assert resolve_threads(2, "nr_threads", nthreads=-1) == 2


# Load one extension module straight from its file, so the synthesizer
# package (and the third party modules it imports) is not imported first, and
# report whether the GIL ended up enabled.
_LOAD_EXTENSION = """
import importlib.util, sys
spec = importlib.util.spec_from_file_location(sys.argv[1], sys.argv[2])
spec.loader.exec_module(importlib.util.module_from_spec(spec))
print(sys._is_gil_enabled())
"""

EXTENSION_MODULES = [
    "synthesizer.extensions.atomic_timing_check",
    "synthesizer.extensions.column_density",
    "synthesizer.extensions.doppler_particle_spectra",
    "synthesizer.extensions.grid_interpolation",
    "synthesizer.extensions.integrated_spectra",
    "synthesizer.extensions.integration",
    "synthesizer.extensions.kernel",
    "synthesizer.extensions.observed_spectra",
    "synthesizer.extensions.openmp_check",
    "synthesizer.extensions.particle_spectra",
    "synthesizer.extensions.photometry",
    "synthesizer.extensions.reductions",
    "synthesizer.extensions.sfzh",
    "synthesizer.extensions.spectra_operations",
    "synthesizer.extensions.timers",
    "synthesizer.imaging.extensions.circular_aperture",
    "synthesizer.imaging.extensions.image",
]


@pytest.mark.skipif(
    not FREE_THREADED_BUILD,
    reason="Requires a free-threaded build of Python",
)
@pytest.mark.parametrize("module", EXTENSION_MODULES)
def test_extension_does_not_enable_gil(module):
    """Test an extension module doesn't re-enable the GIL when imported.

    Each module is loaded in a fresh interpreter with PYTHON_GIL unset, since
    setting it (as CI does to run the threaded tests) stops Python ever
    re-enabling the GIL, which would make this test pass regardless.
    """
    path = importlib.util.find_spec(module).origin
    env = {k: v for k, v in os.environ.items() if k != "PYTHON_GIL"}
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always::RuntimeWarning",
            "-c",
            _LOAD_EXTENSION,
            module,
            path,
        ],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )

    # Python names the module which caused the GIL to be enabled
    enabled_by = [line for line in result.stderr.splitlines() if "GIL" in line]
    assert not any("'synthesizer." in line for line in enabled_by), enabled_by

    # NOTE: with ATOMIC_TIMING an extension imports the synthesizer package
    # (for the timer capsules) during its own initialisation, so a third
    # party module can enable the GIL first. Python then wouldn't report a
    # missing declaration on this module, so we can't tell either way.
    if enabled_by:
        pytest.skip(
            f"Inconclusive, the GIL was enabled first by: {enabled_by}"
        )

    assert result.stdout.strip() == "False"
