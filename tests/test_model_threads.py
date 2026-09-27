"""Tests for executing emission models concurrently (nr_model_threads).

Model threading is only enabled on free-threaded builds of Python with the
GIL disabled. Most of these tests patch the GIL check so the threaded
execution path is exercised on every build: the results must not depend on
whether the threads actually run in parallel.
"""

import subprocess
import sys
import sysconfig
import threading
import time

import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.emission_models import EmissionModel, PacmanEmission
from synthesizer.emission_models import model_queue as model_queue_module
from synthesizer.emission_models.model_queue import (
    ModelQueue,
    resolve_model_threads,
)
from synthesizer.emission_models.parameters import ParameterList
from synthesizer.pipeline import Pipeline

FREE_THREADED_BUILD = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


@pytest.fixture
def no_gil(monkeypatch):
    """Pretend the GIL is disabled so model threads are used."""
    monkeypatch.setattr(model_queue_module, "gil_enabled", lambda: False)


@pytest.fixture
def varied_model(test_grid):
    """Return an expanded PacmanEmission with 3 x 2 variations."""
    return PacmanEmission(
        test_grid,
        tau_v=ParameterList([0.1, 0.5, 1.0], label_modifier="tauv%.1f"),
        fesc=ParameterList([0.0, 0.5], label_modifier="fesc%.1f"),
    ).expand_models()


def _get_spectra(stars, model, nr_model_threads, per_particle):
    """Generate spectra and return copies of every saved lnu array."""
    stars.clear_all_emissions()
    model.set_per_particle(per_particle)
    stars.get_spectra(model, nr_model_threads=nr_model_threads)
    spectra = stars.particle_spectra if per_particle else stars.spectra
    return {label: sed.lnu.value.copy() for label, sed in spectra.items()}


def _get_lines(stars, model, line_ids, nr_model_threads):
    """Generate lines and return copies of every saved luminosity array."""
    stars.clear_all_emissions()
    stars.get_lines(line_ids, model, nr_model_threads=nr_model_threads)
    return {
        label: (
            lines.luminosity.value.copy(),
            lines.continuum.value.copy(),
        )
        for label, lines in stars.lines.items()
    }


def _assert_same(serial, threaded):
    """Assert two dictionaries of emission arrays are identical."""
    assert serial.keys() == threaded.keys()
    for label in serial:
        np.testing.assert_array_equal(
            serial[label],
            threaded[label],
            err_msg=f"Threaded output differs for {label}",
        )


class TestResolveModelThreads:
    """Test resolving the requested number of model threads."""

    def test_serial_is_unchanged(self):
        """Test a single thread is always allowed."""
        assert resolve_model_threads(1) == 1

    @pytest.mark.parametrize("value", [0, -2, 1.5, "4", True, None])
    def test_invalid_values_raise(self, value):
        """Test anything other than a positive integer is rejected."""
        with pytest.raises(exceptions.InconsistentArguments):
            resolve_model_threads(value)

    def test_numpy_integers_accepted(self, no_gil):
        """Test numpy integers are accepted and converted to int."""
        nthreads = resolve_model_threads(np.int64(2))
        assert nthreads == 2
        assert type(nthreads) is int

    def test_gil_falls_back_to_serial(self, monkeypatch):
        """Test model threads are disabled with a warning under the GIL."""
        monkeypatch.setattr(model_queue_module, "gil_enabled", lambda: True)
        with pytest.warns(RuntimeWarning, match="GIL is enabled"):
            assert resolve_model_threads(4) == 1

    def test_no_gil_uses_threads(self, no_gil):
        """Test the requested threads are used without the GIL."""
        assert resolve_model_threads(2) == 2

    def test_oversubscription_warns(self, no_gil, monkeypatch):
        """Test a warning when model x OpenMP threads exceed the cores."""
        monkeypatch.setattr(
            model_queue_module.os,
            "process_cpu_count",
            lambda: 4,
            raising=False,
        )
        monkeypatch.setattr(model_queue_module.os, "cpu_count", lambda: 4)
        with pytest.warns(RuntimeWarning, match="only 4 cores"):
            assert resolve_model_threads(4, nthreads=2) == 4


class TestExecute:
    """Test ModelQueue.execute directly."""

    def test_dependencies_run_first(self, varied_model):
        """Test every model is processed after all of its dependencies."""
        queue = ModelQueue(varied_model)
        finished = set()
        lock = threading.Lock()
        violations = []

        def process(model):
            with lock:
                missing = set(queue.dependencies[model.label]) - finished
            if missing:
                violations.append((model.label, missing))

            # Give other workers a chance to run out of order if they could.
            time.sleep(0.001)
            with lock:
                finished.add(model.label)

        queue.execute(process, {}, {}, nr_threads=4)

        assert violations == []
        assert finished == set(queue.models)

    def test_all_models_processed_once(self, varied_model):
        """Test the threaded executor processes each model exactly once."""
        queue = ModelQueue(varied_model)
        processed = []
        lock = threading.Lock()

        def process(model):
            with lock:
                processed.append(model.label)

        queue.execute(process, {}, {}, nr_threads=4)

        assert sorted(processed) == sorted(queue.models)

    @pytest.mark.parametrize("nr_threads", [1, 4])
    def test_exception_propagates(self, varied_model, nr_threads):
        """Test an exception raised for one model reaches the caller."""
        queue = ModelQueue(varied_model)
        bad_label = next(
            label
            for label, deps in queue.dependencies.items()
            if len(deps) > 0
        )

        def process(model):
            if model.label == bad_label:
                raise ValueError(f"failed on {model.label}")

        with pytest.raises(ValueError, match=bad_label):
            queue.execute(process, {}, {}, nr_threads=nr_threads)


class TestThreadedParity:
    """Test threaded execution gives identical results to serial."""

    @pytest.mark.parametrize("per_particle", [False, True])
    def test_spectra(
        self, no_gil, varied_model, random_part_stars, per_particle
    ):
        """Test spectra from a varied model match serial execution."""
        serial = _get_spectra(random_part_stars, varied_model, 1, per_particle)
        threaded = _get_spectra(
            random_part_stars, varied_model, 4, per_particle
        )
        _assert_same(serial, threaded)

    def test_spectra_repeated(self, no_gil, varied_model, random_part_stars):
        """Test repeated threaded runs are stable."""
        serial = _get_spectra(random_part_stars, varied_model, 1, False)
        for _ in range(5):
            threaded = _get_spectra(random_part_stars, varied_model, 8, False)
            _assert_same(serial, threaded)

    def test_lines(self, no_gil, test_grid, varied_model, random_part_stars):
        """Test lines from a varied model match serial execution."""
        line_ids = test_grid.available_lines
        serial = _get_lines(random_part_stars, varied_model, line_ids, 1)
        threaded = _get_lines(random_part_stars, varied_model, line_ids, 4)
        _assert_same(serial, threaded)

    def test_existing_spectra_reused(
        self, no_gil, varied_model, random_part_stars
    ):
        """Test supplied spectra are reused rather than regenerated."""
        varied_model.set_per_particle(False)
        spectra, _ = varied_model._get_spectra({"stellar": random_part_stars})
        incident = spectra["incident"]

        spectra, _ = varied_model._get_spectra(
            {"stellar": random_part_stars},
            spectra={"incident": incident},
            nr_model_threads=4,
        )

        assert spectra["incident"] is incident

    def test_error_carries_model_label(
        self, no_gil, monkeypatch, varied_model, random_part_stars
    ):
        """Test errors raised on a worker still name the failing model."""

        def fail(*args, **kwargs):
            raise ValueError("combination failed")

        monkeypatch.setattr(EmissionModel, "_combine_spectra", fail)

        with pytest.raises(ValueError, match="combination failed") as excinfo:
            random_part_stars.get_spectra(varied_model, nr_model_threads=4)

        if sys.version_info >= (3, 11):
            notes = "".join(getattr(excinfo.value, "__notes__", []))
        else:
            notes = str(excinfo.value)
        assert "EmissionModel.label" in notes


class TestPipeline:
    """Test the Pipeline resolves model threads once at construction."""

    def test_pipeline_stores_resolved_threads(self, no_gil, varied_model):
        """Test the resolved thread count is stored on the Pipeline."""
        pipeline = Pipeline(varied_model, nr_model_threads=2, verbose=0)
        assert pipeline.nr_model_threads == 2

    def test_pipeline_falls_back_under_gil(self, monkeypatch, varied_model):
        """Test the Pipeline falls back to serial execution under the GIL."""
        monkeypatch.setattr(model_queue_module, "gil_enabled", lambda: True)
        with pytest.warns(RuntimeWarning, match="GIL is enabled"):
            pipeline = Pipeline(varied_model, nr_model_threads=2, verbose=0)
        assert pipeline.nr_model_threads == 1


@pytest.mark.skipif(
    not FREE_THREADED_BUILD,
    reason="Requires a free-threaded build of Python",
)
def test_extensions_do_not_enable_gil():
    """Test none of synthesizer's extension modules re-enable the GIL.

    Third party modules (e.g. astropy.table) may still re-enable it, so this
    checks the warning Python emits for each module that does, in a fresh
    interpreter where no extension has been imported yet.
    """
    modules = [
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
    result = subprocess.run(
        [
            sys.executable,
            "-W",
            "always::RuntimeWarning",
            "-c",
            "; ".join(f"import {module}" for module in modules),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    offenders = [
        line
        for line in result.stderr.splitlines()
        if "GIL" in line and "'synthesizer." in line
    ]
    assert offenders == []
