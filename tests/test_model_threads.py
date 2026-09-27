"""Tests for executing emission models concurrently (nr_model_threads).

Model threading is only enabled on free-threaded builds of Python with the
GIL disabled. Most of these tests patch the GIL check so the threaded
execution path is exercised on every build: the results must not depend on
whether the threads actually run in parallel.
"""

import importlib.util
import os
import subprocess
import sys
import sysconfig
import threading
import time

import numpy as np
import pytest
from unyt import K, Msun

from synthesizer import exceptions
from synthesizer.emission_models import (
    DustEmission,
    EmissionModel,
    PacmanEmission,
)
from synthesizer.emission_models import model_queue as model_queue_module
from synthesizer.emission_models.generators.dust import Casey12, DraineLi07
from synthesizer.emission_models.model_queue import (
    ModelQueue,
    resolve_model_threads,
)
from synthesizer.emission_models.parameters import ParameterList
from synthesizer.grid import Grid
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

    def test_all_cores_nthreads_warns(self, no_gil, monkeypatch):
        """Test nthreads=-1 counts as one OpenMP thread per core."""
        monkeypatch.setattr(
            model_queue_module.os,
            "process_cpu_count",
            lambda: 4,
            raising=False,
        )
        monkeypatch.setattr(model_queue_module.os, "cpu_count", lambda: 4)
        with pytest.warns(RuntimeWarning, match="only 4 cores"):
            assert resolve_model_threads(2, nthreads=-1) == 2


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


@pytest.fixture
def casey12_variants(test_grid):
    """Return temperature variants which all share one Casey12 generator."""
    return DustEmission(
        Casey12(temperature=20 * K),
        emitter="stellar",
        grid=test_grid,
        temperature=ParameterList(
            [10 * K, 20 * K, 30 * K, 40 * K, 50 * K, 60 * K],
            label_modifier="T%s",
        ),
    ).expand_models()


@pytest.fixture
def dl07_grid():
    """Return the Draine & Li dust emission grid, if it is available."""
    try:
        return Grid("draine_li_dust_emission_grid_MW_3p1.hdf5")
    except Exception:
        pytest.skip("Draine & Li dust emission grid not available")


@pytest.fixture
def dl07_variants(test_grid, dl07_grid):
    """Return qpah variants which all share one DraineLi07 generator."""
    return DustEmission(
        DraineLi07(
            dl07_grid,
            dust_mass=1e7 * Msun,
            hydrogen_mass=1e9 * Msun,
            gamma=0.05,
            umin=1.0,
            alpha=2.5,
        ),
        emitter="stellar",
        grid=test_grid,
        qpah=ParameterList(list(dl07_grid.qpah[:4]), label_modifier="q%.4f"),
    ).expand_models()


class TestDustGenerators:
    """Test dust generators shared between concurrently executing models.

    Variants of a model whose generator doesn't depend on other models all
    share one generator instance, so generation must not store per-call state
    on the generator (or its grid).
    """

    def test_variants_share_generator(self, casey12_variants):
        """Test the variants really do share a generator instance."""
        models = casey12_variants._models.values()
        generators = {id(model.generator) for model in models}
        assert len(generators) == 1

    def test_casey12_parity(self, no_gil, casey12_variants, random_part_stars):
        """Test Casey12 variants match serial execution."""
        serial = _get_spectra(random_part_stars, casey12_variants, 1, False)
        for _ in range(5):
            threaded = _get_spectra(
                random_part_stars, casey12_variants, 8, False
            )
            _assert_same(serial, threaded)

    def test_dl07_parity(self, no_gil, dl07_variants, random_part_stars):
        """Test DraineLi07 variants match serial execution."""
        serial = _get_spectra(random_part_stars, dl07_variants, 1, False)
        for _ in range(5):
            threaded = _get_spectra(random_part_stars, dl07_variants, 8, False)
            _assert_same(serial, threaded)

    def test_dl07_leaves_grid_unchanged(
        self, dl07_variants, dl07_grid, random_part_stars
    ):
        """Test generating emission doesn't resample the shared grid."""
        lam = dl07_grid.lam.copy()
        diffuse = dl07_grid.spectra["diffuse"].copy()

        random_part_stars.get_spectra(dl07_variants)

        np.testing.assert_array_equal(dl07_grid.lam, lam)
        np.testing.assert_array_equal(dl07_grid.spectra["diffuse"], diffuse)


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
    # party module can enable the GIL first. That is only inconclusive for
    # this module, so it is only a failure if nothing else enabled it.
    if not enabled_by:
        assert result.stdout.strip() == "False"
