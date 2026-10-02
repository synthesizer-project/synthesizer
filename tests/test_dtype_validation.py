"""Tests that unsupported dtypes are rejected, not silently reinterpreted.

The C++ extensions dispatch on dtype via a `dispatch_float` helper that, by
design, treats anything that isn't float64 as float32 (see
python_to_cpp.h). Every call site is expected to validate its input arrays'
dtypes *before* calling into that dispatch, so an unsupported dtype (e.g.
float16, int32) raises a clear TypeError rather than being silently
reinterpreted as float32 -- which would corrupt every downstream
computation. These tests lock in that validation for the entry points that
matter most for scientific correctness: particle spectra weighting, star
formation history weighting, line-of-sight column density, and numerical
integration.

The tests below those check that float32 and float64 inputs and outputs
combine correctly, and that Synthesizer never creates mismatched precisions
itself.
"""

import itertools

import numpy as np
import pytest
from astropy.cosmology import Planck18
from unyt import (
    Hz,
    K,
    Lsun,
    Mpc,
    Msun,
    Myr,
    angstrom,
    deg,
    erg,
    km,
    kpc,
    s,
    unyt_array,
    unyt_quantity,
    yr,
)

from synthesizer import exceptions, set_default_out_dtype
from synthesizer.emission_models import (
    IncidentEmission,
    PacmanEmission,
    UnifiedAGN,
)
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.emission_models.transformers import (
    GrainModels,
    ParametricLi08,
    PowerLaw,
)
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.instruments import FilterCollection
from synthesizer.load_data.utils import cast_component_dtype
from synthesizer.parametric import SFH, ZDist
from synthesizer.parametric import Stars as ParametricStars
from synthesizer.particle import BlackHoles, Stars
from synthesizer.synth_warnings import InternalPrecisionWarning
from synthesizer.units import Units
from synthesizer.utils.precision import (
    convert_array_dtype,
    max_abs,
    verify_out_precision,
    weighted_sum_fits,
)


def test_cast_component_dtype_handles_generic_component_data():
    """Floating arrays from any component source should cast with units."""
    ages = np.array([1.0, 2.0], dtype=np.float64) * Myr
    metallicities = np.array([0.01, 0.02], dtype=np.float16)
    particle_ids = np.array([1, 2], dtype=np.int64)
    component = {
        "ages": ages,
        "metallicities": metallicities,
        "particle_ids": particle_ids,
        "star_forming": np.array([True, False]),
        "optional": None,
    }

    cast = cast_component_dtype(component, np.float32)

    assert cast["ages"].dtype == np.float32
    assert cast["ages"].units == Myr
    assert cast["metallicities"].dtype == np.float32
    assert cast["particle_ids"] is particle_ids
    assert cast["star_forming"] is component["star_forming"]
    assert cast["optional"] is None
    assert component["ages"].dtype == np.float64


class TestUnsupportedDtypesAreRejected:
    """Unsupported floating-point dtypes must raise, not misdispatch."""

    def test_integration_rejects_float16(self):
        """trapz/simps extensions should reject float16 inputs."""
        from synthesizer.extensions.integration import trapz_last_axis

        xs = np.linspace(0, 1, 8, dtype=np.float16)
        ys = np.ones((2, 8), dtype=np.float16)

        with pytest.raises(TypeError):
            trapz_last_axis(xs, ys, 1, None)

    def test_compute_particle_seds_rejects_float16_particle_property(self):
        """compute_particle_seds should reject a float16 particle property."""
        from synthesizer.extensions.particle_spectra import (
            compute_particle_seds,
        )

        axis = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        grid_spectra = np.arange(15, dtype=np.float64).reshape(3, 5) + 1.0
        grid_dims = np.array([3], dtype=np.int32)
        part_props = (np.array([0.25, 1.5, -1.0, 3.0], dtype=np.float16),)
        weights = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)

        with pytest.raises(TypeError):
            compute_particle_seds(
                grid_spectra,
                (axis,),
                part_props,
                weights,
                grid_dims,
                1,
                weights.size,
                5,
                "ngp",
                1,
                None,
                None,
                False,
                np.float64,
                ("x",),
            )

    def test_compute_sfzh_rejects_float16_grid_axis(self):
        """compute_sfzh should reject a float16 grid axis."""
        from synthesizer.extensions.sfzh import compute_sfzh

        axis16 = np.array([0.0, 1.0, 2.0], dtype=np.float16)
        part_props = (np.array([0.25, 1.5, -1.0, 3.0], dtype=np.float64),)
        weights = np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64)
        grid_dims = np.array([3], dtype=np.int32)

        with pytest.raises(TypeError):
            compute_sfzh(
                (axis16,),
                part_props,
                weights,
                grid_dims,
                1,
                weights.size,
                "ngp",
                1,
                None,
                ("x",),
                np.float64,
            )

    def test_compute_column_density_rejects_int32_positions(self):
        """compute_column_density should reject non-floating positions."""
        from synthesizer.extensions.column_density import (
            compute_column_density,
        )

        kernel = np.ones(8, dtype=np.float64)
        truncated_kernel = np.ones((4, 4), dtype=np.float64)
        pos_i = np.zeros((1, 3), dtype=np.int32)
        pos_j = np.zeros((1, 3), dtype=np.float64)
        smls = np.ones(1, dtype=np.float64)
        surf_den_vals = np.ones(1, dtype=np.float64)

        with pytest.raises(TypeError):
            compute_column_density(
                kernel,
                truncated_kernel,
                pos_i,
                pos_j,
                smls,
                (surf_den_vals,),
                1,
                1,
                8,
                4,
                4,
                1.0,
                1,
                8,
                1,
            )


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


def _realistic_stars(dtype, n=50, seed=0):
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


class TestPrecisionCombinations:
    """Tests for sensible combinations of input and output precision.

    Particle data, grids and outputs can each be float32 or float64
    independently. Run the main workflows over representative combinations,
    using physically realistic magnitudes, and check that each one:

    - runs without a mixed-precision error,
    - returns outputs at the requested precision,
    - contains no overflowed (inf) or NaN values, and
    - agrees with the all-float64 result to float32 accuracy.
    """

    @pytest.fixture(autouse=True)
    def reset_default_out_dtype(self):
        """Make sure no test leaks a changed default output dtype."""
        yield
        set_default_out_dtype(F64)

    @pytest.mark.parametrize(
        "part, grid, out", COMBINATIONS, ids=COMBINATION_IDS
    )
    @pytest.mark.parametrize("vel_shift", [False, True], ids=["", "vel_shift"])
    def test_incident_spectra(self, grids, part, grid, out, vel_shift):
        """Incident particle spectra (optionally Doppler shifted) work."""

        def spectra(part, grid, out, vel_shift):
            stars = _realistic_stars(part)
            model = IncidentEmission(grid=grids[grid], per_particle=True)
            stars.get_spectra(model, out_dtype=out, vel_shift=vel_shift)
            return (
                stars.spectra[model.label]._lnu,
                stars.particle_spectra[model.label]._lnu,
            )

        reference = _reference(spectra, F64, F64, F64, vel_shift)
        for result, ref in zip(spectra(part, grid, out, vel_shift), reference):
            _check(result, ref, out)

    @pytest.mark.parametrize(
        "part, grid, out", COMBINATIONS, ids=COMBINATION_IDS
    )
    def test_stellar_emission(self, grids, part, grid, out):
        """Spectra, fluxes, photometry and lines work at every combination.

        The particles here carry up to 1e8 Msun, so energy balance dust
        emission and line luminosities are well beyond the float32 range in
        erg/s.
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
            stars = _realistic_stars(part)
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
        "grid, out",
        PAIRS,
        ids=[f"grid{_bits(g)}-out{_bits(o)}" for g, o in PAIRS],
    )
    def test_parametric_spectra(self, grids, grid, out):
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
    def test_black_hole_derived_properties_do_not_overflow(self, dtype):
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

    @pytest.mark.parametrize(
        "part, grid, out", COMBINATIONS, ids=COMBINATION_IDS
    )
    def test_agn_spectra_and_lines(self, agn_grids, part, grid, out):
        """AGN spectra and lines work at every combination.

        Black hole bolometric luminosities (the grid weights) are ~1e45 erg/s,
        far beyond the float32 range in erg/s.
        """

        def emission(part, grid, out):
            set_default_out_dtype(out)
            bh = BlackHoles(
                masses=unyt_array(np.array([1e6, 1e7, 1e8, 1e9], part), Msun),
                accretion_rates=unyt_array(
                    np.array([0.01, 0.1, 1.0, 1.0], part),
                    Msun / yr,
                ),
                inclinations=np.array([10, 30, 50, 70], part) * deg,
                coordinates=np.zeros((4, 3), part) * Mpc,
                metallicities=np.full(4, 0.01, part),
            )
            nlr, blr = agn_grids[grid]
            model = UnifiedAGN(
                nlr_grid=nlr,
                blr_grid=blr,
                torus_emission_model=Greybody(
                    temperature=1000 * K, emissivity=2
                ),
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


def _sed32(nspec=None):
    """Build a float32 Sed, optionally with one spectrum per particle."""
    lam = unyt_array(np.geomspace(1e-4, 1e11, 50), angstrom)
    shape = (lam.size,) if nspec is None else (nspec, lam.size)
    return Sed(lam, unyt_array(np.ones(shape, np.float32), erg / s / Hz))


class TestInternalPrecisionConsistency:
    """Synthesizer should never creates mismatched precisions itself.

    Each test covers a case where the user's inputs were consistent but a
    Synthesizer-created array used to have a different precision, either
    raising a mixed-precision error or silently promoting the output.
    """

    @pytest.mark.parametrize(
        "curve",
        [PowerLaw(), GrainModels(), ParametricLi08()],
        ids=type,
    )
    def test_attenuation_keeps_sed_precision(self, curve):
        """A Python float tau_v or float64 curve shouldn't promote the Sed."""
        sed = _sed32(nspec=3)

        attenuated = sed.apply_attenuation(0.3, curve)

        assert attenuated._lnu.dtype == np.float32
        assert np.all(np.isfinite(attenuated._lnu))

    def test_per_particle_scaling_takes_sed_precision(self):
        """Per-particle float64 scalings are converted to the Sed precision."""
        sed = _sed32(nspec=3)

        scaled = sed.scale(np.array([1.0, 2.0, 3.0]))

        assert scaled._lnu.dtype == np.float32
        np.testing.assert_allclose(scaled._lnu[:, 0], [1.0, 2.0, 3.0])

    def test_interp_spectra_keeps_grid_precision(self):
        """Interpolating a float32 grid onto new wavelengths keeps float32."""
        grid = Grid("test_grid.hdf5", use_precision=np.float32)

        grid.interp_spectra(np.linspace(1000.0, 20000.0, 100) * angstrom)

        assert grid.lam.dtype == np.float32
        for spectra in grid.spectra.values():
            assert spectra.dtype == np.float32

    def test_integrated_luminosities_keep_sed_precision(self):
        """Bolometric and window luminosities keep the precision of lnu."""
        lam = unyt_array(np.linspace(1000.0, 20000.0, 200), angstrom)
        lnu = unyt_array(np.full((3, 200), 1e20, np.float32), erg / s / Hz)
        sed = Sed(lam, lnu)
        sed64 = Sed(lam, lnu.astype(np.float64))

        results = (
            (sed.bolometric_luminosity, sed64.bolometric_luminosity),
            (
                sed.measure_bolometric_luminosity(),
                sed64.measure_bolometric_luminosity(),
            ),
            (
                sed.measure_window_luminosity((2000, 5000) * angstrom),
                sed64.measure_window_luminosity((2000, 5000) * angstrom),
            ),
        )
        for result, reference in results:
            assert result.units == Units().luminosity
            assert result.dtype == np.float32
            np.testing.assert_allclose(
                result.value, reference.value, rtol=1e-5
            )

    def test_sed_luminosity_keeps_sed_precision(self):
        """Sed.luminosity and Sed.llam keep the precision of lnu."""
        lam = unyt_array(np.linspace(1000.0, 20000.0, 50), angstrom)
        lnu = unyt_array(np.full(50, 1e20, np.float32), erg / s / Hz)
        sed = Sed(lam, lnu)
        lnu64 = lnu.astype(np.float64)

        results = (
            (sed.luminosity, lnu64 * sed.nu, Units().luminosity),
            (
                sed.llam,
                lnu64 * sed.nu / sed.lam,
                Units().luminosity_density_wavelength,
            ),
        )
        for result, expected, units in results:
            assert result.dtype == np.float32
            assert result.units == units
            np.testing.assert_allclose(
                result.value,
                expected.to(units).value,
                rtol=1e-6,
            )

    def test_broadening_keeps_sed_precision(self):
        """Broadening and resampling keep the precision of the Sed."""
        sed = _sed32(nspec=3)

        assert sed.doppler_broaden(100 * km / s)._lnu.dtype == np.float32
        resampled = sed.get_resampled_sed(new_lam=sed.lam[::2])
        assert resampled._lnu.dtype == np.float32

    @pytest.mark.parametrize("img_type", ["hist", "smoothed"])
    def test_normalised_images_keep_signal_precision(self, img_type):
        """Normalised float32 images keep the precision of the signal."""
        from synthesizer.imaging import Image
        from synthesizer.kernel_functions import Kernel

        rng = np.random.default_rng(0)
        n = 50
        coords = unyt_array(
            rng.uniform(-0.4, 0.4, (n, 3)).astype(np.float32), kpc
        )
        signal = unyt_array(np.full(n, 1e8, np.float32), Lsun)
        norm = unyt_array(np.full(n, 1e10, np.float32), Msun)

        img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
        if img_type == "hist":
            img.generate_img_hist(signal, coords, normalisation=norm)
        else:
            img.generate_img_smoothed(
                signal,
                coordinates=coords,
                smoothing_lengths=unyt_array(
                    np.full(n, 0.05, np.float32), kpc
                ),
                kernel=Kernel(),
                normalisation=norm,
            )

        arr = np.asarray(img.arr)
        assert arr.dtype == np.float32
        assert not np.any(np.isinf(arr))
        np.testing.assert_allclose(arr[np.isfinite(arr)], 1e8, rtol=1e-5)


class TestConvertArrayDtype:
    """Tests for the general precision conversion utility."""

    def test_converts_keeping_units(self):
        """Arrays are converted with their units preserved."""
        arr = unyt_array(np.array([1.0, 2.0]), erg / s)
        converted = convert_array_dtype(arr, np.float32)
        assert converted.dtype == np.float32
        assert converted.units == arr.units
        np.testing.assert_allclose(converted.value, [1.0, 2.0])

    def test_returns_matching_arrays_untouched(self):
        """No copy is made when the array already has the dtype."""
        arr = np.ones(3, np.float32)
        assert convert_array_dtype(arr, np.float32) is arr

    def test_overflow_raises(self):
        """Values too large for the target precision raise an error."""
        with pytest.raises(exceptions.PrecisionOverflow, match="lums"):
            convert_array_dtype(np.array([1e45]), np.float32, name="lums")

    def test_scalars(self):
        """Scalars are converted to scalars of the target dtype."""
        converted = convert_array_dtype(2.0, np.float32)
        assert isinstance(converted, np.float32)


class TestOverflowBound:
    """Tests for the tools bounding weighted sums against overflow."""

    def test_max_abs(self):
        """The largest absolute value is found, with or without units."""
        assert max_abs(np.array([-3.0, 2.0])) == 3.0
        assert max_abs(unyt_array([1.0, -5.0], "erg/s")) == 5.0
        assert max_abs(np.array([])) == 0.0

    def test_weighted_sum_fits(self):
        """The bound fits only when it is safely below the dtype maximum."""
        big = float(np.finfo(np.float32).max)
        assert weighted_sum_fits(np.float32, 1.0, 1e10)
        assert not weighted_sum_fits(np.float32, big, 1.0)
        assert weighted_sum_fits(np.float64, big, 1e-10)
        assert not weighted_sum_fits(np.float32, np.inf, 1.0)
        assert not weighted_sum_fits(np.float32, np.nan, 1.0)


class TestVerifyOutPrecision:
    """Tests for the decorator verifying outputs respect out_dtype."""

    def test_matching_output_passes(self):
        """Outputs at the requested precision are returned as they are."""

        @verify_out_precision()
        def func(out_dtype=None):
            return np.ones(3, dtype=out_dtype)

        assert func(out_dtype=np.float32).dtype == np.float32

    def test_mismatched_output_warns_and_converts(self):
        """Outputs at the wrong precision are converted with a warning."""

        @verify_out_precision()
        def func(out_dtype=None):
            return np.ones(3)

        with pytest.warns(InternalPrecisionWarning, match="report"):
            result = func(out_dtype=np.float32)
        assert result.dtype == np.float32

    def test_mismatched_output_that_overflows_raises(self):
        """Converting an output that doesn't fit raises an error."""

        @verify_out_precision()
        def func(out_dtype=None):
            return np.full(3, 1e45)

        with pytest.warns(InternalPrecisionWarning):
            with pytest.raises(exceptions.PrecisionOverflow):
                func(out_dtype=np.float32)

    def test_overflowed_output_raises(self):
        """Reduced precision outputs holding inf raise an error."""

        @verify_out_precision()
        def func(out_dtype=None):
            return np.full(3, np.inf, dtype=np.float32)

        with pytest.raises(exceptions.PrecisionOverflow, match="inf"):
            func(out_dtype=np.float32)

    @pytest.mark.parametrize("strided", [False, True])
    def test_overflowed_output_raises_threaded(self, strided):
        """A single -inf is caught threaded and in strided views."""

        @verify_out_precision()
        def func(nthreads=1, out_dtype=None):
            arr = np.ones((200, 100), dtype=np.float32)
            arr[-1, -2] = -np.inf
            return arr[:, ::2] if strided else arr

        with pytest.raises(exceptions.PrecisionOverflow, match="inf"):
            func(nthreads=4, out_dtype=np.float32)

    @pytest.mark.parametrize("finite", [False, True])
    def test_cannot_overflow_skips_scan(self, finite):
        """The inf scan is skipped only when outputs are provably finite."""

        @verify_out_precision(
            cannot_overflow=lambda dtype, scale, **kwargs: finite
        )
        def func(scale, out_dtype=None):
            return np.full(3, np.inf, dtype=np.float32)

        if finite:
            assert np.isinf(func(1.0, out_dtype=np.float32)).all()
        else:
            with pytest.raises(exceptions.PrecisionOverflow, match="inf"):
                func(1.0, out_dtype=np.float32)

    def test_checks_output_objects(self):
        """The arrays inside Synthesizer output objects are checked."""

        @verify_out_precision()
        def func(out_dtype=None):
            return Sed(_sed32().lam, unyt_array(np.ones(50), erg / s / Hz))

        with pytest.warns(InternalPrecisionWarning):
            sed = func(out_dtype=np.float32)
        assert sed._lnu.dtype == np.float32

    def test_requires_out_dtype_argument(self):
        """Only functions taking out_dtype can be decorated."""
        with pytest.raises(TypeError, match="out_dtype"):

            @verify_out_precision()
            def func():
                return None
