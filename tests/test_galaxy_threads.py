"""Tests for processing galaxies concurrently in the Pipeline.

Galaxy threads are only used on free-threaded builds of Python with the GIL
disabled. These tests patch the GIL check so the threaded path is exercised
on every build: the outputs must not depend on whether the threads actually
run in parallel.
"""

import numpy as np
import pytest
from astropy.cosmology import Planck18
from unyt import Msun, Myr

from synthesizer.emission_models.attenuation import Inoue14
from synthesizer.parametric import SFH, Stars, ZDist
from synthesizer.parametric.galaxy import Galaxy
from synthesizer.pipeline import Pipeline
from synthesizer.utils import free_threading


@pytest.fixture
def no_gil(monkeypatch):
    """Pretend the GIL is disabled so galaxy threads are used."""
    monkeypatch.setattr(free_threading, "gil_enabled", lambda: False)


def _make_galaxies(grid, ngal=7):
    """Return distinct parametric galaxies at a range of redshifts.

    Every galaxy differs (SFH, metallicity, mass and redshift), so any
    galaxy's outputs ending up in the wrong place would be caught.
    """
    galaxies = []
    for i in range(ngal):
        # Give each galaxy its own star formation history, metallicity and
        # mass
        stars = Stars(
            grid.log10ages,
            grid.metallicities,
            sf_hist=SFH.Constant(max_age=(50 + 20 * i) * Myr),
            metal_dist=ZDist.DeltaConstant(metallicity=0.005 + 0.002 * i),
            initial_mass=(1 + i) * 1e9 * Msun,
        )

        # And its own redshift (so each has a different IGM transmission)
        galaxies.append(Galaxy(stars=stars, redshift=1.0 + 0.5 * i))

    return galaxies


def _stellar_mass(gal):
    """Return a per-galaxy quantity for the analysis function test."""
    return gal.stars.initial_mass


def _run(grid, model, nr_galaxy_threads):
    """Run a pipeline over fresh galaxies and return it."""
    # Set up the pipeline with an analysis function and fresh galaxies (the
    # pipeline consumes its galaxies, so each run needs its own)
    pipeline = Pipeline(model, verbose=0, nr_galaxy_threads=nr_galaxy_threads)
    pipeline.add_analysis_func(_stellar_mass, "StellarMass")
    pipeline.add_galaxies(_make_galaxies(grid))

    # Signal operations covering per-galaxy outputs, IGM transmission, lines
    # and the cosmic SED (which is accumulated across galaxies)
    pipeline.get_spectra()
    pipeline.get_observed_spectra(Planck18, igm=Inoue14())
    pipeline.get_lines(line_ids=["H 1 6562.80A"])
    pipeline.get_cosmic_sed()

    # Run it
    pipeline.run()

    return pipeline


def _assert_same(serial, threaded, path="outputs"):
    """Assert two nested output containers are identical."""
    # Recurse into dictionaries, checking they have the same keys
    if isinstance(serial, dict):
        assert serial.keys() == threaded.keys(), path
        for key in serial:
            _assert_same(serial[key], threaded[key], f"{path}/{key}")

    # Recurse into lists and tuples, checking they have the same length
    elif isinstance(serial, (list, tuple)):
        assert len(serial) == len(threaded), path
        for i, (s, t) in enumerate(zip(serial, threaded)):
            _assert_same(s, t, f"{path}[{i}]")

    # Anything else is an array (or scalar) which must match exactly
    else:
        np.testing.assert_array_equal(
            np.asarray(serial), np.asarray(threaded), err_msg=path
        )


def test_gil_falls_back_to_serial(monkeypatch, nebular_emission_model):
    """Test galaxy threads are disabled with a warning under the GIL."""
    # Pretend the GIL is enabled, whatever build we are on
    monkeypatch.setattr(free_threading, "gil_enabled", lambda: True)

    # Requesting threads should warn and fall back to serial
    with pytest.warns(RuntimeWarning, match="nr_galaxy_threads=4"):
        pipeline = Pipeline(
            nebular_emission_model, verbose=0, nr_galaxy_threads=4
        )
    assert pipeline.nr_galaxy_threads == 1


@pytest.mark.parametrize("nr_galaxy_threads", [2, 4])
def test_threaded_matches_serial(
    no_gil, test_grid, nebular_emission_model, nr_galaxy_threads
):
    """Test threaded outputs, counts and ordering match a serial run."""
    # Run the same galaxies serially and on threads
    serial = _run(test_grid, nebular_emission_model, 1)
    threaded = _run(test_grid, nebular_emission_model, nr_galaxy_threads)

    # Every output must be identical, including its galaxy ordering
    for attr in (
        "lnu_spectra",
        "fnu_spectra",
        "line_lums",
        "line_cont_lums",
        "cosmic_lnus",
        "_analysis_results",
    ):
        _assert_same(getattr(serial, attr), getattr(threaded, attr), attr)

    # The worker threads' counts must sum to the serial counts
    assert serial._op_counts == threaded._op_counts


def test_worker_error_propagates(no_gil, test_grid, nebular_emission_model):
    """Test an error on a worker thread is raised by run."""

    # An analysis function which always fails (analysis functions run on
    # the worker threads)
    def fail(gal):
        raise ValueError("boom")

    # Set up a threaded pipeline using it
    pipeline = Pipeline(nebular_emission_model, verbose=0, nr_galaxy_threads=2)
    pipeline.add_analysis_func(fail, "Fail")
    pipeline.add_galaxies(_make_galaxies(test_grid))
    pipeline.get_spectra()

    # The worker's error should be raised by run on the main thread
    with pytest.raises(ValueError, match="boom"):
        pipeline.run()
