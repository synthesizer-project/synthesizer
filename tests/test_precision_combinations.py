"""Tests for sensible combinations of input and output precision.

Particle data, grids and outputs can each be float32 or float64 independently.
These tests run the main workflows over representative combinations, using
physically realistic magnitudes, and check that each one:

- runs without a mixed-precision error,
- returns outputs at the requested precision,
- contains no overflowed (inf) or NaN values, and
- agrees with the all-float64 result to float32 accuracy.
"""

import itertools

import numpy as np
import pytest
from astropy.cosmology import Planck18
from unyt import (
    K,
    Mpc,
    Msun,
    Myr,
    angstrom,
    deg,
    km,
    kpc,
    s,
    unyt_array,
    unyt_quantity,
    yr,
)

from synthesizer import set_default_out_dtype
from synthesizer.emission_models import (
    IncidentEmission,
    PacmanEmission,
    UnifiedAGN,
)
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.emission_models.transformers import PowerLaw
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle import BlackHoles, Stars

F32, F64 = np.float32, np.float64
DTYPES = (F32, F64)


def _bits(dtype):
    """Return the width of a float dtype in bits (for test ids)."""
    return np.dtype(dtype).itemsize * 8


# (particle, grid, output) precision combinations covering the uniform cases
# and mixed inputs with reduced precision outputs
COMBINATIONS = [
    (F32, F32, F32),
    (F64, F64, F64),
    (F32, F64, F32),
    (F64, F32, F32),
]
COMBINATION_IDS = [
    f"part{_bits(p)}-grid{_bits(g)}-out{_bits(o)}" for p, g, o in COMBINATIONS
]
PAIRS = list(itertools.product(DTYPES, DTYPES))

# float32 carries ~7 significant figures, but sums over particles and grid
# interpolation accumulate some rounding, so compare relative to the peak
RTOL = 1e-3

# The all-float64 results, computed once per workflow
_REFERENCES = {}


def _reference(func, *args):
    """Return the all-float64 result of ``func``, computing it once."""
    key = (func.__qualname__, *args)
    if key not in _REFERENCES:
        _REFERENCES[key] = func(*args)
    return _REFERENCES[key]


@pytest.fixture(scope="module")
def grids():
    """Return the test grid at both precisions."""
    return {
        F64: Grid("test_grid.hdf5"),
        F32: Grid("test_grid.hdf5", use_precision=F32),
    }


@pytest.fixture(scope="module")
def agn_grids():
    """Return the test AGN line region grids at both precisions."""
    return {
        dtype: (
            Grid("test_grid_agn-nlr.hdf5", use_precision=dtype),
            Grid("test_grid_agn-blr.hdf5", use_precision=dtype),
        )
        for dtype in DTYPES
    }


@pytest.fixture(autouse=True)
def reset_default_out_dtype():
    """Make sure no test leaks a changed default output dtype."""
    yield
    set_default_out_dtype(F64)


def _stars(dtype, n=50, seed=0):
    """Build particle stars with realistic masses, ages and velocities."""
    rng = np.random.default_rng(seed)

    def arr(values):
        return np.asarray(values, dtype=dtype)

    return Stars(
        initial_masses=unyt_array(arr(10 ** rng.uniform(4, 8, n)), Msun),
        ages=unyt_array(arr(10 ** rng.uniform(0, 3, n)), Myr),
        metallicities=arr(rng.uniform(0.001, 0.02, n)),
        redshift=1.0,
        tau_v=arr(rng.uniform(0.1, 1.0, n)),
        coordinates=unyt_array(arr(rng.normal(0, 1, (n, 3))), kpc),
        velocities=unyt_array(arr(rng.normal(0, 200, (n, 3))), km / s),
    )


def _check(result, reference, dtype):
    """Check a result's precision, finiteness and agreement with float64."""
    result = np.asarray(result)
    reference = np.asarray(reference)
    assert result.dtype == dtype
    assert np.all(np.isfinite(result)), "result overflowed or has NaNs"
    np.testing.assert_allclose(
        result,
        reference,
        rtol=RTOL,
        atol=RTOL * np.max(np.abs(reference)),
    )


def _attenuated_with_dust(grid, per_particle=True):
    return PacmanEmission(
        grid=grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        dust_emission=Greybody(temperature=30 * K, emissivity=1.5),
        per_particle=per_particle,
    )


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
@pytest.mark.parametrize("vel_shift", [False, True], ids=["", "vel_shift"])
def test_incident_spectra(grids, part, grid, out, vel_shift):
    """Incident particle spectra (optionally Doppler shifted) work."""

    def spectra(part, grid, out, vel_shift):
        stars = _stars(part)
        model = IncidentEmission(grid=grids[grid], per_particle=True)
        stars.get_spectra(model, out_dtype=out, vel_shift=vel_shift)
        return (
            stars.spectra[model.label]._lnu,
            stars.particle_spectra[model.label]._lnu,
        )

    reference = _reference(spectra, F64, F64, F64, vel_shift)
    for result, ref in zip(spectra(part, grid, out, vel_shift), reference):
        _check(result, ref, out)


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
def test_stellar_emission(grids, part, grid, out):
    """Spectra, fluxes, photometry and lines work at every combination.

    The particles here carry up to 1e8 Msun, so energy balance dust emission
    and line luminosities are well beyond the float32 range in erg/s.
    """
    filters = FilterCollection(
        tophat_dict={
            "U": {"lam_eff": 3500 * angstrom, "lam_fwhm": 500 * angstrom},
            "V": {"lam_eff": 5500 * angstrom, "lam_fwhm": 800 * angstrom},
        },
        new_lam=grids[F64].lam,
    )

    def emission(part, grid, out):
        set_default_out_dtype(out)
        stars = _stars(part)
        model = _attenuated_with_dust(grids[grid])
        stars.get_spectra(model)
        sed = stars.spectra[model.label]
        sed.get_fnu(Planck18, stars.redshift)
        photometry = sed.get_photo_fnu(filters)
        stars.get_lines(grids[grid].available_lines[:10], model)
        lines = stars.lines[model.label]
        return (
            sed._lnu,
            stars.particle_spectra[model.label]._lnu,
            sed._fnu,
            lines._luminosity,
            lines._continuum,
            stars.particle_lines[model.label]._luminosity,
        ), np.array([photometry[f].value for f in filters.filter_codes])

    results, photometry = emission(part, grid, out)
    references, ref_photometry = _reference(emission, F64, F64, F64)
    for result, ref in zip(results, references):
        _check(result, ref, out)
    assert np.all(np.isfinite(photometry))
    np.testing.assert_allclose(photometry, ref_photometry, rtol=RTOL)


@pytest.mark.parametrize(
    "grid, out", PAIRS, ids=[f"grid{_bits(g)}-out{_bits(o)}" for g, o in PAIRS]
)
def test_parametric_spectra(grids, grid, out):
    """Parametric stars work with either grid and output precision."""

    def spectra(grid, out):
        g = grids[grid]
        stars = ParametricStars(
            g.log10ages,
            g.metallicities,
            sf_hist=SFH.Constant(max_age=100 * Myr),
            metal_dist=ZDist.DeltaConstant(metallicity=0.01),
            initial_mass=unyt_quantity(1e10, Msun),
        )
        model = _attenuated_with_dust(g, per_particle=False)
        stars.tau_v = 0.5
        stars.get_spectra(model, out_dtype=out)
        return stars.spectra[model.label]._lnu

    _check(spectra(grid, out), _reference(spectra, F64, F64), out)


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_black_hole_derived_properties_do_not_overflow(dtype):
    """Derived black hole properties keep the input precision."""
    bh = BlackHoles(
        masses=unyt_array(np.full(3, 1e8, dtype), Msun),
        accretion_rates=unyt_array(np.ones(3, dtype), Msun / yr),
        inclinations=np.zeros(3, dtype) * deg,
    )

    for prop in (
        bh.bolometric_luminosities,
        bh.accretion_rate_eddington,
        bh.eddington_ratio,
    ):
        assert prop.dtype == dtype
        assert np.all(np.isfinite(prop))


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
def test_agn_spectra_and_lines(agn_grids, part, grid, out):
    """AGN spectra and lines work at every combination.

    Black hole bolometric luminosities (the grid weights) are ~1e45 erg/s,
    far beyond the float32 range in erg/s.
    """

    def emission(part, grid, out):
        set_default_out_dtype(out)
        bh = BlackHoles(
            masses=unyt_array(np.array([1e6, 1e7, 1e8, 1e9], part), Msun),
            accretion_rates=unyt_array(
                np.array([0.01, 0.1, 1.0, 1.0], part), Msun / yr
            ),
            inclinations=np.array([10, 30, 50, 70], part) * deg,
            coordinates=np.zeros((4, 3), part) * Mpc,
            metallicities=np.full(4, 0.01, part),
        )
        nlr, blr = agn_grids[grid]
        model = UnifiedAGN(
            nlr_grid=nlr,
            blr_grid=blr,
            torus_emission_model=Greybody(temperature=1000 * K, emissivity=2),
            per_particle=True,
        )
        bh.get_spectra(model)
        bh.get_lines(blr.available_lines[:10], model)
        return (
            bh.particle_spectra[model.label]._lnu,
            bh.spectra[model.label]._lnu,
            bh.lines[model.label]._luminosity,
        )

    reference = _reference(emission, F64, F64, F64)
    for result, ref in zip(emission(part, grid, out), reference):
        _check(result, ref, out)
