"""Tests for sensible combinations of input and output precision.

Particle data, grids and outputs can each be float32 or float64 independently.
These tests run the main workflows over every combination, using physically
realistic magnitudes, and check that each one:

- runs without a mixed-precision error,
- returns outputs at the requested precision,
- contains no overflowed (inf) or NaN values, and
- agrees with the all-float64 result to float32 accuracy.
"""

import itertools

import numpy as np
import pytest
from astropy.cosmology import Planck18
from synthesizer.extensions.particle_spectra import compute_particle_seds
from unyt import (
    K,
    Lsun,
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
from synthesizer.units import Units

F32, F64 = np.float32, np.float64
DTYPES = (F32, F64)

# Every (particle, grid, output) precision combination
COMBINATIONS = list(itertools.product(DTYPES, DTYPES, DTYPES))


def _bits(dtype):
    """Return the width of a float dtype in bits (for test ids)."""
    return np.dtype(dtype).itemsize * 8


COMBINATION_IDS = [
    f"part{_bits(p)}-grid{_bits(g)}-out{_bits(o)}" for p, g, o in COMBINATIONS
]
PAIRS = list(itertools.product(DTYPES, DTYPES))

# float32 carries ~7 significant figures, but sums over particles and grid
# interpolation accumulate some rounding, so compare relative to the peak
RTOL = 1e-3


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


# Luminosities only fit in float32 when stored in Lsun (the default). With a
# units file set to erg/s instead, the tests below check that Synthesizer
# handles the values that can't fit correctly (warnings, or errors with
# advice that works) rather than skipping.
ERG_LUMINOSITIES = Units().luminosity != Lsun


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


def _run_particle_model(make_model, grids, part, grid, out, **kwargs):
    """Get spectra for a model at one precision combination."""
    stars = _stars(part)
    model = make_model(grids[grid])
    stars.get_spectra(model, out_dtype=out, **kwargs)
    return (
        stars.spectra[model.label]._lnu,
        stars.particle_spectra[model.label]._lnu,
    )


def _incident(grid):
    return IncidentEmission(grid=grid, per_particle=True)


def _attenuated_with_dust(grid, per_particle=True):
    return PacmanEmission(
        grid=grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        dust_emission=Greybody(temperature=30 * K, emissivity=1.5),
        per_particle=per_particle,
    )


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
@pytest.mark.parametrize(
    "make_model",
    [_incident, _attenuated_with_dust],
    ids=["incident", "attenuated+dust"],
)
def test_particle_spectra(make_model, grids, part, grid, out):
    """Particle spectra work at every precision combination."""
    spectra, particle_spectra = _run_particle_model(
        make_model, grids, part, grid, out
    )
    ref_spectra, ref_particle_spectra = _run_particle_model(
        make_model, grids, F64, F64, F64
    )

    _check(spectra, ref_spectra, out)
    _check(particle_spectra, ref_particle_spectra, out)


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
def test_velocity_shifted_spectra(grids, part, grid, out):
    """Doppler-shifted particle spectra work at every combination."""
    spectra, particle_spectra = _run_particle_model(
        _incident, grids, part, grid, out, vel_shift=True
    )
    ref_spectra, ref_particle_spectra = _run_particle_model(
        _incident, grids, F64, F64, F64, vel_shift=True
    )

    _check(spectra, ref_spectra, out)
    _check(particle_spectra, ref_particle_spectra, out)


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

    _check(spectra(grid, out), spectra(F64, F64), out)


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
def test_fluxes_and_photometry(grids, part, grid, out):
    """Observed fluxes and photometry work at every combination."""
    filters = FilterCollection(
        tophat_dict={
            "U": {"lam_eff": 3500 * angstrom, "lam_fwhm": 500 * angstrom},
            "V": {"lam_eff": 5500 * angstrom, "lam_fwhm": 800 * angstrom},
        },
        new_lam=grids[F64].lam,
    )

    def observe(part, grid, out):
        stars = _stars(part)
        model = _attenuated_with_dust(grids[grid])
        stars.get_spectra(model, out_dtype=out)
        sed = stars.spectra[model.label]
        sed.get_fnu(Planck18, stars.redshift)
        photometry = sed.get_photo_fnu(filters)
        return sed._fnu, np.array(
            [photometry[f].value for f in filters.filter_codes]
        )

    fnu, photometry = observe(part, grid, out)
    ref_fnu, ref_photometry = observe(F64, F64, F64)

    _check(fnu, ref_fnu, out)
    assert np.all(np.isfinite(photometry))
    np.testing.assert_allclose(photometry, ref_photometry, rtol=RTOL)


@pytest.mark.parametrize(
    "part, grid",
    PAIRS,
    ids=[f"part{_bits(p)}-grid{_bits(g)}" for p, g in PAIRS],
)
def test_sfzh(grids, part, grid):
    """The SFZH works with either particle and grid precision."""

    def sfzh(part, grid):
        g = grids[grid]
        return _stars(part).get_sfzh(g.log10ages, g.metallicities)

    result, reference = sfzh(part, grid), sfzh(F64, F64)
    values = np.asarray(getattr(result, "sfzh", result))
    ref_values = np.asarray(getattr(reference, "sfzh", reference))
    assert np.all(np.isfinite(values))
    np.testing.assert_allclose(
        values, ref_values, rtol=RTOL, atol=RTOL * ref_values.max()
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=["f32", "f64"])
def test_black_hole_derived_properties_do_not_overflow(dtype):
    """Derived black hole properties stay finite and physically sensible.

    Bolometric luminosities (~1e45 erg/s here) only fit in float32 in Lsun
    (the default units), otherwise they must stay float64. Dimensionless
    ratios always keep the input precision.
    """
    bh = BlackHoles(
        masses=unyt_array(np.full(3, 1e8, dtype), Msun),
        accretion_rates=unyt_array(np.ones(3, dtype), Msun / yr),
        inclinations=np.zeros(3, dtype) * deg,
    )

    assert np.all(np.isfinite(bh.bolometric_luminosities))
    expected = dtype if Units().luminosity == Lsun else F64
    assert bh.bolometric_luminosities.dtype == expected
    for ratio in (bh.accretion_rate_eddington, bh.eddington_ratio):
        assert ratio.dtype == dtype
        assert np.all(np.isfinite(ratio))


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
def test_lines(grids, part, grid, out):
    """Line luminosities work at every combination without overflowing.

    The particles here carry up to 1e8 Msun, giving line luminosities well
    beyond the float32 range in erg/s. In Lsun they fit; with erg/s
    luminosities float32 outputs must warn that they overflowed.
    """

    def lines(part, grid, out):
        set_default_out_dtype(out)
        stars = _stars(part)
        model = _attenuated_with_dust(grids[grid])
        stars.get_spectra(model)
        stars.get_lines(grids[grid].available_lines[:10], model)
        integrated = stars.lines[model.label]
        per_particle = stars.particle_lines[model.label]
        return (
            integrated._luminosity,
            integrated._continuum,
            per_particle._luminosity,
        )

    if ERG_LUMINOSITIES and out == F32:
        with pytest.warns(RuntimeWarning, match=r"overflowed\s+to\s+inf"):
            lines(part, grid, out)
        return

    for result, reference in zip(lines(part, grid, out), lines(F64, F64, F64)):
        _check(result, reference, out)


@pytest.mark.parametrize("part, grid, out", COMBINATIONS, ids=COMBINATION_IDS)
def test_agn_spectra_and_lines(agn_grids, part, grid, out):
    """AGN spectra and lines work at every combination.

    Black hole bolometric luminosities (the grid weights) are ~1e45 erg/s,
    far beyond the float32 range in erg/s. In Lsun everything fits. With
    erg/s luminosities:

    - float32 black holes can't have float32 luminosities, so extraction
      must raise a mixed precision error advising conversion to float64;
    - float32 outputs from float64 black holes must still give correct
      spectra (the weights are applied at float64, with a warning), while
      the line luminosities themselves overflow (with a warning).
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

    if ERG_LUMINOSITIES and part == F32:
        with pytest.raises(TypeError, match="to float64"):
            emission(part, grid, out)
        return

    reference = emission(F64, F64, F64)

    if ERG_LUMINOSITIES and out == F32:
        with pytest.warns(RuntimeWarning) as record:
            spectra, integrated, _ = emission(part, grid, out)
        # (warning text may be wrapped over several lines)
        messages = " ".join(" ".join(str(w.message).split()) for w in record)
        assert "weights are too large" in messages
        assert "overflowed to inf" in messages
        _check(spectra, reference[0], out)
        _check(integrated, reference[1], out)
        return

    for result, ref in zip(emission(part, grid, out), reference):
        _check(result, ref, out)


def _huge_weight_inputs(grid_value):
    """Build extension inputs with weights far beyond the float32 range."""
    axis = np.array([0.0, 1.0, 2.0])
    grid_spectra = np.full((3, 5), grid_value)
    part_props = (np.array([0.5, 1.5]),)
    weights = np.array([1e45, 3e45])
    return grid_spectra, (axis,), part_props, weights


@pytest.mark.parametrize("method", ["cic", "ngp"])
def test_weights_beyond_output_precision_are_applied_at_float64(method):
    """Weights too large for float32 outputs still give correct results.

    Each output value (1e45 * 1e-16) fits in float32, but the weight alone
    does not, so it must be applied at float64 and a warning raised.
    """
    grid_spectra, axes, part_props, weights = _huge_weight_inputs(1e-16)
    args = (
        grid_spectra,
        axes,
        part_props,
        weights,
        np.array([3], np.int32),
        1,
        weights.size,
        5,
        method,
        1,
        None,
        None,
        False,
    )

    reference = compute_particle_seds(*args, F64, ("x", "w"))
    with pytest.warns(RuntimeWarning, match="weights are too large"):
        result = compute_particle_seds(*args, F32, ("x", "w"))

    assert result.dtype == F32
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result, reference, rtol=1e-6)


def test_output_overflow_warns():
    """Output values too large for the output precision raise a warning."""
    grid_spectra, axes, part_props, weights = _huge_weight_inputs(1.0)

    with pytest.warns(RuntimeWarning, match="overflowed to inf"):
        compute_particle_seds(
            grid_spectra,
            axes,
            part_props,
            weights,
            np.array([3], np.int32),
            1,
            weights.size,
            5,
            "cic",
            1,
            None,
            None,
            False,
            F32,
            ("x", "w"),
        )
