"""Tests that Synthesizer never creates mismatched precisions itself.

Each test here covers a case where the user's inputs were consistent but a
Synthesizer-created array used to have a different precision, either raising
a mixed-precision error or silently promoting the output.
"""

import numpy as np
import pytest
from unyt import (
    Hz,
    K,
    Msun,
    Myr,
    angstrom,
    erg,
    km,
    kpc,
    s,
    unyt_array,
)

from synthesizer import exceptions
from synthesizer.emission_models import IncidentEmission, PacmanEmission
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.emission_models.transformers import (
    GrainModels,
    ParametricLi08,
    PowerLaw,
)
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.particle import Stars
from synthesizer.utils.precision import (
    InternalPrecisionWarning,
    convert_array_dtype,
    verify_out_precision,
)


def _stars(dtype, n=4):
    """Build a small particle Stars object at a single precision."""
    return Stars(
        initial_masses=unyt_array(np.full(n, 1e6, dtype=dtype), Msun),
        ages=unyt_array(np.linspace(1.0, 50.0, n, dtype=dtype), Myr),
        metallicities=np.full(n, 0.01, dtype=dtype),
        redshift=1.0,
        tau_v=np.full(n, 0.3, dtype=dtype),
        coordinates=unyt_array(np.zeros((n, 3), dtype=dtype), kpc),
    )


def _sed32(nspec=None):
    """Build a float32 Sed, optionally with one spectrum per particle."""
    lam = unyt_array(np.geomspace(1e-4, 1e11, 50), angstrom)
    shape = (lam.size,) if nspec is None else (nspec, lam.size)
    return Sed(lam, unyt_array(np.ones(shape, np.float32), erg / s / Hz))


def test_generated_dust_emission_follows_out_dtype(test_grid):
    """Dust emission generators should honour out_dtype (per particle)."""
    stars = _stars(np.float32)
    model = PacmanEmission(
        grid=test_grid,
        tau_v="tau_v",
        dust_curve=PowerLaw(),
        dust_emission=Greybody(temperature=30 * K, emissivity=1.5),
        per_particle=True,
    )

    stars.get_spectra(model, out_dtype=np.float32)

    assert stars.particle_spectra[model.label]._lnu.dtype == np.float32
    assert stars.spectra[model.label]._lnu.dtype == np.float32


@pytest.mark.parametrize(
    "curve", [PowerLaw(), GrainModels(), ParametricLi08()], ids=type
)
def test_attenuation_keeps_sed_precision(curve):
    """A Python float tau_v or a float64 curve shouldn't promote the Sed."""
    sed = _sed32(nspec=3)

    attenuated = sed.apply_attenuation(0.3, curve)

    assert attenuated._lnu.dtype == np.float32
    assert np.all(np.isfinite(attenuated._lnu))


def test_per_particle_scaling_takes_sed_precision():
    """Per-particle float64 scalings are converted to the Sed precision."""
    sed = _sed32(nspec=3)

    scaled = sed.scale(np.array([1.0, 2.0, 3.0]))

    assert scaled._lnu.dtype == np.float32
    np.testing.assert_allclose(scaled._lnu[:, 0], [1.0, 2.0, 3.0])


def test_interp_spectra_keeps_grid_precision():
    """Interpolating a float32 grid onto new wavelengths keeps float32."""
    grid = Grid("test_grid.hdf5", use_precision=np.float32)

    grid.interp_spectra(np.linspace(1000.0, 20000.0, 100) * angstrom)

    assert grid.lam.dtype == np.float32
    for spectra in grid.spectra.values():
        assert spectra.dtype == np.float32


def test_sed_luminosity_keeps_sed_precision():
    """Sed.luminosity keeps the precision of lnu."""
    lam = unyt_array(np.linspace(1000.0, 20000.0, 50), angstrom)
    lnu = unyt_array(np.full(50, 1e20, np.float32), erg / s / Hz)
    sed = Sed(lam, lnu)

    lum = sed.luminosity

    assert lum.dtype == np.float32
    expected = (lnu.astype(np.float64) * sed.nu).to(lum.units)
    np.testing.assert_allclose(lum.value, expected.value, rtol=1e-6)


def test_line_subset_is_contiguous(test_grid):
    """Selecting a subset of per-particle lines gives contiguous arrays."""
    stars = _stars(np.float32)
    model = IncidentEmission(grid=test_grid, per_particle=True)
    line_ids = test_grid.available_lines[:3]

    stars.get_lines(line_ids, model)
    lines = stars.particle_lines[model.label]

    assert lines._luminosity.flags.c_contiguous
    assert lines._continuum.flags.c_contiguous
    lines.scale(np.ones(stars.nparticles))


def test_broadening_keeps_sed_precision():
    """Broadening and resampling keep the precision of the Sed."""
    sed = _sed32(nspec=3)

    assert sed.doppler_broaden(100 * km / s)._lnu.dtype == np.float32
    resampled = sed.get_resampled_sed(new_lam=sed.lam[::2])
    assert resampled._lnu.dtype == np.float32


@pytest.mark.parametrize("img_type", ["hist", "smoothed"])
def test_image_normalisation_does_not_overflow(img_type):
    """Normalised float32 images don't overflow before normalising.

    signal * normalisation (1e30 * 1e10) exceeds float32, while the
    normalised image (1e30) does not.
    """
    from synthesizer.imaging import Image
    from synthesizer.kernel_functions import Kernel

    rng = np.random.default_rng(0)
    n = 50
    coords = unyt_array(rng.uniform(-0.4, 0.4, (n, 3)).astype(np.float32), kpc)
    signal = unyt_array(np.full(n, 1e30, np.float32), erg / s)
    norm = unyt_array(np.full(n, 1e10, np.float32), Msun)

    img = Image(resolution=0.1 * kpc, fov=1.0 * kpc)
    if img_type == "hist":
        img.generate_img_hist(signal, coords, normalisation=norm)
    else:
        img.generate_img_smoothed(
            signal,
            coordinates=coords,
            smoothing_lengths=unyt_array(np.full(n, 0.05, np.float32), kpc),
            kernel=Kernel(),
            normalisation=norm,
        )

    arr = np.asarray(img.arr)
    assert arr.dtype == np.float32
    assert not np.any(np.isinf(arr))
    np.testing.assert_allclose(arr[np.isfinite(arr)], 1e30, rtol=1e-5)


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

    def test_overflow_keep_returns_input(self):
        """With overflow="keep" values that don't fit are left alone."""
        arr = np.array([1e45])
        assert convert_array_dtype(arr, np.float32, overflow="keep") is arr

    def test_scalars(self):
        """Scalars are converted to scalars of the target dtype."""
        converted = convert_array_dtype(2.0, np.float32)
        assert isinstance(converted, np.float32)


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

    def test_checks_select_returned_values(self):
        """Only the flagged values of a returned tuple are checked."""

        @verify_out_precision(True, False)
        def func(out_dtype=None):
            return np.ones(3, np.float32), np.ones(3)

        first, second = func(out_dtype=np.float32)
        assert first.dtype == np.float32
        assert second.dtype == np.float64

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
