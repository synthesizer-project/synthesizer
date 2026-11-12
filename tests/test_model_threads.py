"""Tests for executing emission models concurrently (nr_model_threads).

Model threading is only enabled on free-threaded builds of Python with the
GIL disabled. Most of these tests patch the GIL check so the threaded
execution path is exercised on every build: the results must not depend on
whether the threads actually run in parallel.
"""

import os
import sys
import threading
import time

import numpy as np
import pytest
from unyt import K, Msun

from synthesizer import GRID_DIR
from synthesizer.emission_models import (
    DustEmission,
    EmissionModel,
    IncidentEmission,
    PacmanEmission,
)
from synthesizer.emission_models.generators.dust import (
    Blackbody,
    Casey12,
    DraineLi07,
)
from synthesizer.emission_models.model_queue import ModelQueue
from synthesizer.emission_models.parameters import ParameterList
from synthesizer.pipeline import Pipeline
from synthesizer.utils import free_threading


@pytest.fixture
def no_gil(monkeypatch):
    """Pretend the GIL is disabled so model threads are used."""
    monkeypatch.setattr(free_threading, "gil_enabled", lambda: False)


@pytest.fixture
def varied_model(test_grid):
    """Return an expanded PacmanEmission with 3 x 2 variations."""
    return PacmanEmission(
        test_grid,
        tau_v=ParameterList([0.1, 0.5, 1.0], label_modifier="tauv%.1f"),
        fesc=ParameterList([0.0, 0.5], label_modifier="fesc%.1f"),
    ).expand_models()


def _get_spectra(stars, model, nr_model_threads, per_particle):
    """Generate spectra and return copies of every saved lnu array.

    Args:
        stars (Stars): The stars to generate the spectra for.
        model (EmissionModel): The emission model to use.
        nr_model_threads (int): The number of model threads to use.
        per_particle (bool): Whether to generate per-particle spectra.

    Returns:
        dict: A copy of each saved spectrum's lnu, keyed by model label.
    """
    # Start from a clean slate so nothing is reused from an earlier run
    stars.clear_all_emissions()
    model.set_per_particle(per_particle)
    stars.get_spectra(model, nr_model_threads=nr_model_threads)

    # Copy the arrays so later runs can't change what we compare
    spectra = stars.particle_spectra if per_particle else stars.spectra
    return {label: sed.lnu.value.copy() for label, sed in spectra.items()}


def _get_lines(stars, model, line_ids, nr_model_threads):
    """Generate lines and return copies of every saved luminosity array.

    Args:
        stars (Stars): The stars to generate the lines for.
        model (EmissionModel): The emission model to use.
        line_ids (list): The IDs of the lines to generate.
        nr_model_threads (int): The number of model threads to use.

    Returns:
        dict: A copy of each saved line collection's luminosity and
            continuum, keyed by model label.
    """
    # Start from a clean slate so nothing is reused from an earlier run
    stars.clear_all_emissions()
    stars.get_lines(line_ids, model, nr_model_threads=nr_model_threads)

    # Copy the arrays so later runs can't change what we compare
    return {
        label: (
            lines.luminosity.value.copy(),
            lines.continuum.value.copy(),
        )
        for label, lines in stars.lines.items()
    }


def _assert_same(serial, threaded):
    """Assert two dictionaries of emission arrays are identical.

    Args:
        serial (dict): The arrays from a serial run, keyed by model label.
        threaded (dict): The arrays from a threaded run, keyed by model
            label.
    """
    # The same models must have been saved, with identical emission
    assert serial.keys() == threaded.keys()
    for label in serial:
        np.testing.assert_array_equal(
            serial[label],
            threaded[label],
            err_msg=f"Threaded output differs for {label}",
        )


class TestExecute:
    """Test ModelQueue.execute directly."""

    def test_dependencies_run_first(self, no_gil, varied_model):
        """Test every model is processed after all of its dependencies."""
        # Record the models which have finished, and any model processed
        # before all of its dependencies had
        queue = ModelQueue(varied_model)
        finished = set()
        lock = threading.Lock()
        violations = []

        def process(model):
            """Check the model's dependencies have finished, then finish it.

            Args:
                model (EmissionModel): The model being processed.
            """
            with lock:
                missing = set(queue.dependencies[model.label]) - finished
            if missing:
                violations.append((model.label, missing))

            # Give other workers a chance to run out of order if they could.
            time.sleep(0.001)
            with lock:
                finished.add(model.label)

        # Every model should run, and only once its dependencies have
        queue.execute(process, {}, {}, nr_model_threads=4)
        assert violations == []
        assert finished == set(queue.models)

    def test_all_models_processed_once(self, no_gil, varied_model):
        """Test the threaded executor processes each model exactly once."""
        # Record every model processed
        queue = ModelQueue(varied_model)
        processed = []
        lock = threading.Lock()

        def process(model):
            """Record the model as processed.

            Args:
                model (EmissionModel): The model being processed.
            """
            with lock:
                processed.append(model.label)

        # Each model should appear exactly once
        queue.execute(process, {}, {}, nr_model_threads=4)
        assert sorted(processed) == sorted(queue.models)

    @pytest.mark.parametrize("nr_threads", [1, 4])
    def test_exception_propagates(self, no_gil, varied_model, nr_threads):
        """Test an exception raised for one model reaches the caller."""
        # Fail on a model with dependencies, so others have already run
        queue = ModelQueue(varied_model)
        bad_label = next(
            label
            for label, deps in queue.dependencies.items()
            if len(deps) > 0
        )

        def process(model):
            """Fail on the chosen model.

            Args:
                model (EmissionModel): The model being processed.

            Raises:
                ValueError: If the model is the one chosen to fail.
            """
            if model.label == bad_label:
                raise ValueError(f"failed on {model.label}")

        # The error should reach us unchanged
        with pytest.raises(ValueError, match=bad_label):
            queue.execute(process, {}, {}, nr_model_threads=nr_threads)

    def test_done_error_on_worker_does_not_hang(self, no_gil, varied_model):
        """Test an error marking a model done on a worker thread is raised."""
        # Keep the real done to call from the calling worker, and somewhere
        # to put the error execute raises
        queue = ModelQueue(varied_model)
        done = queue.done
        result = {}

        def run():
            """Execute the queue, recording the error it raises."""
            try:
                queue.execute(process, {}, {}, nr_model_threads=4)
            except RuntimeError as error:
                result["error"] = error

        # Execute on a thread of our own so a hang fails this test rather
        # than blocking the suite. That thread is the calling worker.
        caller = threading.Thread(target=run, daemon=True)

        def failing_done(model, emissions, particle_emissions):
            """Fail to mark a model done on any thread but the caller.

            Args:
                model (EmissionModel): The model to mark done.
                emissions (dict): The integrated emissions.
                particle_emissions (dict): The particle emissions.

            Raises:
                RuntimeError: If called on a worker thread.
            """
            if threading.current_thread() is not caller:
                raise RuntimeError("done failed")
            done(model, emissions, particle_emissions)

        def process(model):
            """Slow the calling worker so the other workers take models too.

            Args:
                model (EmissionModel): The model being processed.
            """
            if threading.current_thread() is caller:
                time.sleep(0.01)

        # Run with the failing done, giving up if execute hangs
        queue.done = failing_done
        caller.start()
        caller.join(timeout=30)

        # Execute should have finished, raising the worker's error
        assert not caller.is_alive(), "execute hung"
        assert "done failed" in str(result.get("error"))

    def test_worker_start_failure_stops_pool(
        self, no_gil, monkeypatch, varied_model
    ):
        """Test a worker failing to start stops the workers already running."""
        # Let the first worker start, then fail to start any more
        queue = ModelQueue(varied_model)
        start = threading.Thread.start
        nstarted = []

        def failing_start(thread):
            """Start the first worker only.

            Args:
                thread (threading.Thread): The thread to start.

            Raises:
                RuntimeError: For every thread after the first.
            """
            if len(nstarted) > 0:
                raise RuntimeError("can't start new thread")
            nstarted.append(thread)
            start(thread)

        monkeypatch.setattr(threading.Thread, "start", failing_start)

        def process(model):
            """Take long enough that the started worker is still busy.

            Args:
                model (EmissionModel): The model being processed.
            """
            time.sleep(0.05)

        # The start error should reach us, with the started worker finished
        with pytest.raises(RuntimeError, match="can't start new thread"):
            queue.execute(process, {}, {}, nr_model_threads=4)
        assert not nstarted[0].is_alive()


class TestThreadedParity:
    """Test threaded execution gives identical results to serial."""

    @pytest.mark.parametrize("per_particle", [False, True])
    def test_spectra(
        self, no_gil, varied_model, random_part_stars, per_particle
    ):
        """Test spectra from a varied model match serial execution."""
        # Generate the same spectra serially and on threads, which must match
        serial = _get_spectra(random_part_stars, varied_model, 1, per_particle)
        threaded = _get_spectra(
            random_part_stars, varied_model, 4, per_particle
        )
        _assert_same(serial, threaded)

    def test_spectra_repeated(self, no_gil, varied_model, random_part_stars):
        """Test repeated threaded runs are stable."""
        # Any race should show up as a mismatch in at least one of the runs
        serial = _get_spectra(random_part_stars, varied_model, 1, False)
        for _ in range(5):
            threaded = _get_spectra(random_part_stars, varied_model, 8, False)
            _assert_same(serial, threaded)

    def test_lines(self, no_gil, test_grid, varied_model, random_part_stars):
        """Test lines from a varied model match serial execution."""
        # Generate the same lines serially and on threads, which must match
        line_ids = test_grid.available_lines
        serial = _get_lines(random_part_stars, varied_model, line_ids, 1)
        threaded = _get_lines(random_part_stars, varied_model, line_ids, 4)
        _assert_same(serial, threaded)

    @pytest.mark.parametrize("nr_model_threads", [1, 4])
    def test_lines_generator_without_dependencies(
        self, no_gil, test_grid, random_part_stars, nr_model_threads
    ):
        """Test a generator with no dependencies gets the line wavelengths.

        The generator is combined first, so it can run before any extraction
        has produced lines.
        """
        # Combine a blackbody (a generator with no dependencies) with an
        # extraction
        dust = DustEmission(
            Blackbody(temperature=1e4 * K),
            emitter="stellar",
            label="blackbody",
        )
        incident = IncidentEmission(test_grid, label="incident")
        model = EmissionModel(
            "total",
            combine=(dust, incident),
            emitter="stellar",
        )
        # Generate the spectra the generator needs, then the lines
        line_ids = test_grid.available_lines[:3]
        random_part_stars.clear_all_emissions()
        random_part_stars.get_spectra(model)
        random_part_stars.get_lines(
            line_ids,
            model,
            nr_model_threads=nr_model_threads,
        )

        # The generator's lines should have the extraction's wavelengths
        lines = random_part_stars.lines
        np.testing.assert_array_equal(
            lines["blackbody"].lam,
            lines["incident"].lam,
        )

    def test_vel_shifted_flags(self, no_gil, varied_model, random_part_stars):
        """Test velocity shift flags propagate as they do in serial."""
        # Record the velocity shift flags from a serial and a threaded run
        varied_model.set_per_particle(False)
        varied_model.set_vel_shift(True, set_all=True)
        flags = []
        for nr_model_threads in (1, 4):
            random_part_stars.clear_all_emissions()
            random_part_stars.get_spectra(
                varied_model,
                nr_model_threads=nr_model_threads,
            )
            flags.append(
                {
                    label: sed.vel_shifted
                    for label, sed in random_part_stars.spectra.items()
                }
            )
        # They should match, with the root flagged as velocity shifted
        assert flags[0] == flags[1]
        assert flags[1][varied_model.label]

    def test_existing_spectra_reused(
        self, no_gil, varied_model, random_part_stars
    ):
        """Test supplied spectra are reused rather than regenerated."""
        # Generate the spectra once to get an incident spectrum to supply
        varied_model.set_per_particle(False)
        spectra, _ = varied_model._get_spectra({"stellar": random_part_stars})
        incident = spectra["incident"]

        # Generating again with it supplied should keep the same object
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
            """Fail every combination.

            Args:
                *args: Ignored.
                **kwargs: Ignored.

            Raises:
                ValueError: Always.
            """
            raise ValueError("combination failed")

        # Make every combination fail, then generate on threads
        monkeypatch.setattr(EmissionModel, "_combine_spectra", fail)
        with pytest.raises(ValueError, match="combination failed") as excinfo:
            random_part_stars.get_spectra(varied_model, nr_model_threads=4)

        # The error should name the failing model (as a note on 3.11+, in
        # the message before that)
        if sys.version_info >= (3, 11):
            notes = "".join(getattr(excinfo.value, "__notes__", []))
        else:
            notes = str(excinfo.value)
        assert "EmissionModel.label" in notes


@pytest.fixture
def casey12_variants(test_grid):
    """Return temperature variants which all share one Casey12 generator."""
    return DustEmission(
        Casey12(temperature=20 * K, emissivity=2.0, alpha=2.0),
        emitter="stellar",
        grid=test_grid,
        temperature=ParameterList(
            [10 * K, 20 * K, 30 * K, 40 * K, 50 * K, 60 * K],
            label_modifier="T%s",
        ),
    ).expand_models()


@pytest.fixture
def dl07_generator():
    """Return a DraineLi07 generator, if its grid is available."""
    # Only skip if the grid isn't downloaded, any other error should fail
    grid_name = "draine_li_dust_emission_grid_MW_3p1"
    if not os.path.exists(os.path.join(GRID_DIR, f"{grid_name}.hdf5")):
        pytest.skip("Draine & Li dust emission grid not available")
    return DraineLi07(
        grid_name,
        dust_mass=1e7 * Msun,
        hydrogen_mass=1e9 * Msun,
        gamma=0.05,
        umin=1.0,
        alpha=2.5,
    )


@pytest.fixture
def dl07_variants(test_grid, dl07_generator):
    """Return qpah variants which all share one DraineLi07 generator."""
    return DustEmission(
        dl07_generator,
        emitter="stellar",
        grid=test_grid,
        qpah=ParameterList(
            list(dl07_generator.grid.qpah[:4]), label_modifier="q%.4f"
        ),
    ).expand_models()


class TestDustGenerators:
    """Test dust generators shared between concurrently executing models.

    Variants of a model whose generator doesn't depend on other models all
    share one generator instance, so generation must not store per-call state
    on the generator (or its grid).
    """

    def test_variants_share_generator(self, casey12_variants):
        """Test the variants really do share a generator instance."""
        # Otherwise the parity tests below wouldn't test sharing at all
        models = casey12_variants._models.values()
        generators = {id(model.generator) for model in models}
        assert len(generators) == 1

    def test_casey12_parity(self, no_gil, casey12_variants, random_part_stars):
        """Test Casey12 variants match serial execution."""
        # Any per-call state stored on the shared generator should show up
        # as a mismatch in at least one of the runs
        serial = _get_spectra(random_part_stars, casey12_variants, 1, False)
        for _ in range(5):
            threaded = _get_spectra(
                random_part_stars, casey12_variants, 8, False
            )
            _assert_same(serial, threaded)

    def test_dl07_parity(self, no_gil, dl07_variants, random_part_stars):
        """Test DraineLi07 variants match serial execution."""
        # Any per-call state stored on the shared generator (or its grid)
        # should show up as a mismatch in at least one of the runs
        serial = _get_spectra(random_part_stars, dl07_variants, 1, False)
        for _ in range(5):
            threaded = _get_spectra(random_part_stars, dl07_variants, 8, False)
            _assert_same(serial, threaded)


class TestPipeline:
    """Test the Pipeline resolves model threads once at construction."""

    def test_pipeline_stores_resolved_threads(self, no_gil, varied_model):
        """Test the resolved thread count is stored on the Pipeline."""
        pipeline = Pipeline(varied_model, nr_model_threads=2, verbose=0)
        assert pipeline.nr_model_threads == 2

    def test_pipeline_falls_back_under_gil(self, monkeypatch, varied_model):
        """Test the Pipeline falls back to serial execution under the GIL."""
        # Pretend the GIL is enabled, whatever build we are on, then
        # requesting threads should warn and fall back to serial
        monkeypatch.setattr(free_threading, "gil_enabled", lambda: True)
        with pytest.warns(RuntimeWarning, match="GIL is enabled"):
            pipeline = Pipeline(varied_model, nr_model_threads=2, verbose=0)
        assert pipeline.nr_model_threads == 1
