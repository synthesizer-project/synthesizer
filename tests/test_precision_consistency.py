"""Tests that Synthesizer never creates mismatched precisions itself.

Each test here covers a case where the user's inputs were consistent but a
Synthesizer-created array used to have a different precision, either raising
a mixed-precision error or silently promoting the output.
"""

import numpy as np
import pytest
from unyt import Hz, K, Msun, Myr, angstrom, erg, kpc, s, unyt_array

from synthesizer import exceptions
from synthesizer.emission_models import IncidentEmission, PacmanEmission
from synthesizer.emission_models.generators.dust.greybody import Greybody
from synthesizer.emission_models.transformers import (
    CoveringFraction,
    GrainModels,
    ParametricLi08,
    PowerLaw,
)
from synthesizer.emissions import Sed
from synthesizer.grid import Grid
from synthesizer.particle import Stars


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


def test_per_particle_fraction_on_integrated_emission_raises(test_grid):
    """Per-particle fractions can't be applied to integrated emission."""
    sed = _sed32()
    model = PacmanEmission(grid=test_grid)

    emitter = _stars(np.float32)
    emitter.fcov = np.full(emitter.nparticles, 0.2)

    with pytest.raises(exceptions.InconsistentArguments, match="per_particle"):
        CoveringFraction(covering_attrs=("fcov",))._transform(
            sed, emitter, model, None, None
        )


def test_interp_spectra_keeps_grid_precision():
    """Interpolating a float32 grid onto new wavelengths keeps float32."""
    grid = Grid("test_grid.hdf5", use_precision=np.float32)

    grid.interp_spectra(np.linspace(1000.0, 20000.0, 100) * angstrom)

    assert grid.lam.dtype == np.float32
    for spectra in grid.spectra.values():
        assert spectra.dtype == np.float32


@pytest.mark.parametrize(
    "scaling",
    [1e45, np.full(3, 1e45)],
    ids=["python-float", "per-particle-array"],
)
def test_large_scalings_do_not_overflow(scaling):
    """Scalings too large for float32 still give representable results.

    e.g. an AGN template scaled by a bolometric luminosity of ~1e45. The
    multiply must happen before rounding to float32.
    """
    sed = Sed(
        unyt_array(np.linspace(1000.0, 20000.0, 50), angstrom),
        unyt_array(np.full((3, 50), 1e-16, np.float32), erg / s / Hz),
    )

    scaled = sed.scale(scaling)

    assert scaled._lnu.dtype == np.float32
    np.testing.assert_allclose(scaled._lnu, 1e29, rtol=1e-6)


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
