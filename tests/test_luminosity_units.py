"""Tests for luminosities stored in Synthesizer's internal units.

Luminosities are stored in the configured ``luminosity`` unit (Lsun by
default, which keeps realistic values within the float32 range) and grids are
converted to these internal units when loaded. These tests check the results
are the same whatever that unit is, so they hold for any units file.
"""

import h5py
import numpy as np
from unyt import Msun, Myr, deg, erg, kpc, s, unyt_array, unyt_quantity, yr

from synthesizer import set_default_out_dtype
from synthesizer.emission_models import NebularLineEmission
from synthesizer.grid import Grid
from synthesizer.particle import BlackHoles, Stars
from synthesizer.units import Units


def _raw(grid, *keys):
    """Read a dataset and its units straight from a grid file."""
    with h5py.File(grid.grid_filename, "r") as hf:
        dset = hf["/".join(keys)]
        return dset[...].astype(np.float64), dset.attrs.get("Units")


def test_grid_line_luminosities_are_in_internal_units(test_grid):
    """Grid line luminosities are stored in the internal luminosity unit."""
    raw, units = _raw(test_grid, "lines", "luminosity")
    lums = test_grid.line_lums["nebular"]

    assert lums.units == Units().luminosity
    np.testing.assert_allclose(
        lums.to(units).value, raw, rtol=1e-6, atol=1e-6 * raw.max()
    )


def test_agn_grid_is_normalised_per_internal_weight_unit():
    """AGN grid spectra are per unit bolometric luminosity in internal units.

    The grid file is normalised per erg/s of bolometric luminosity; after
    loading it must be per unit of the internal luminosity unit.
    """
    grid = Grid("test_grid_agn-blr.hdf5")
    raw, _ = _raw(grid, "spectra", "incident")
    per_internal = unyt_quantity(1.0, Units().luminosity).to(erg / s).value

    np.testing.assert_allclose(
        grid.spectra["incident"], raw * per_internal, rtol=1e-6
    )


def test_extracted_line_luminosities_match_the_grid(test_grid):
    """A star on a grid point gets its mass times the grid's luminosity."""
    raw, units = _raw(test_grid, "lines", "luminosity")
    iage, imet = 10, 5
    mass = 1e6
    stars = Stars(
        initial_masses=unyt_array([mass], Msun),
        ages=unyt_array([10 ** test_grid.log10ages[iage]], "yr"),
        metallicities=np.array([test_grid.metallicities[imet]]),
        redshift=0.0,
        coordinates=unyt_array(np.zeros((1, 3)), kpc),
    )
    model = NebularLineEmission(grid=test_grid, per_particle=True)

    stars.get_lines(
        test_grid.available_lines, model, grid_assignment_method="ngp"
    )
    lines = stars.particle_lines[model.label]

    np.testing.assert_allclose(
        lines.luminosity.to(units).value[0],
        mass * raw[iage, imet],
        rtol=1e-5,
    )


def test_float32_line_luminosities_fit_in_internal_units(test_grid):
    """float32 line luminosities are finite whenever their values fit.

    Young 1e8 Msun populations have line luminosities of ~1e43 erg/s, beyond
    the float32 range in erg/s but well within it in Lsun (the default).
    """
    n = 10
    stars = Stars(
        initial_masses=unyt_array(np.full(n, 1e8, np.float32), Msun),
        ages=unyt_array(np.full(n, 3.0, np.float32), Myr),
        metallicities=np.full(n, 0.01, np.float32),
        redshift=0.0,
        coordinates=unyt_array(np.zeros((n, 3), np.float32), kpc),
    )
    model = NebularLineEmission(grid=test_grid, per_particle=True)

    set_default_out_dtype(np.float32)
    try:
        stars.get_lines(test_grid.available_lines, model)
    finally:
        set_default_out_dtype(np.float64)
    lums = stars.particle_lines[model.label]._luminosity

    set_default_out_dtype(np.float64)
    stars.get_lines(test_grid.available_lines, model)
    true_max = np.max(stars.particle_lines[model.label]._luminosity)

    assert lums.dtype == np.float32
    assert np.all(np.isfinite(lums)) == (true_max < np.finfo(np.float32).max)


def test_eddington_accretion_rate_is_unit_independent():
    """The Eddington-scaled accretion rate matches the Eddington ratio.

    Bolometric and Eddington luminosities are stored in different units, so
    they must be converted to matching units before dividing.
    """
    bh = BlackHoles(
        masses=unyt_array(np.full(3, 1e8), Msun),
        accretion_rates=unyt_array(np.ones(3), Msun / yr),
        inclinations=np.zeros(3) * deg,
    )

    np.testing.assert_allclose(
        bh.accretion_rate_eddington, bh.eddington_ratio, rtol=1e-6
    )
