"""Tests for sharpening ionisation edges in tabulated SEDs for cloudy."""

import numpy as np
import pytest
from unyt import Angstrom, Hz, angstrom, c, erg, eV, h, s

from synthesizer.emissions import Sed
from synthesizer.photoionisation import (
    Ions,
    add_ionisation_edges,
    cloudy17,
    cloudy23,
)

LYMAN = float((h * c / Ions.energy["HI"]).to("angstrom").value)


def step_sed(lam, jump=100.0):
    """A flat SED that drops by ``jump`` blueward of the Lyman edge."""
    return np.where(lam < LYMAN, 1.0 / jump, 1.0) * 1e20


def loglog_onto(new_lam, lam, lnu):
    """Interpolate an SED in log-log space, as cloudy does for a table."""
    return 10 ** np.interp(np.log10(new_lam), np.log10(lam), np.log10(lnu))


def q_rate(lam, lnu):
    """Ionising photon rate of an SED using synthesizer."""
    sed = Sed(lam=lam * Angstrom, lnu=lnu * erg / s / Hz)
    return float(
        sed.calculate_ionising_photon_production_rate(
            ionisation_energy=13.6 * eV
        )
    )


class TestAddIonisationEdges:
    """Tests for add_ionisation_edges."""

    def test_inserts_points_around_coarse_edges(self):
        """A pair of points brackets each edge, with neighbour values."""
        lam = np.arange(100.0, 2000.0, 10.0)
        lnu = step_sed(lam)
        new_lam, new_lnu = add_ionisation_edges(lam, lnu)

        # Three edges (H I, He I, He II) inside the range, two points each
        assert len(new_lam) == len(lam) + 6
        assert np.all(np.diff(new_lam) > 0)

        # The H I pair sits either side of the edge with the values of the
        # nearest original points on each side
        blue = np.where(new_lam < LYMAN)[0][-1]
        red = np.where(new_lam > LYMAN)[0][0]
        assert new_lam[blue] == pytest.approx(LYMAN * (1 - 1e-4))
        assert new_lam[red] == pytest.approx(LYMAN * (1 + 1e-4))
        assert new_lnu[blue] == lnu[lam < LYMAN][-1]
        assert new_lnu[red] == lnu[lam > LYMAN][0]

    def test_original_points_unchanged(self):
        """Every original point is kept with its original value."""
        lam = np.arange(100.0, 2000.0, 10.0)
        lnu = step_sed(lam)
        new_lam, new_lnu = add_ionisation_edges(lam, lnu)
        idx = np.searchsorted(new_lam, lam)
        np.testing.assert_array_equal(new_lam[idx], lam)
        np.testing.assert_array_equal(new_lnu[idx], lnu)

    def test_inputs_not_modified(self):
        """The input arrays are left untouched."""
        lam = np.arange(2000.0, 100.0, -10.0)  # decreasing order
        lnu = step_sed(lam)
        lam0, lnu0 = lam.copy(), lnu.copy()
        add_ionisation_edges(lam, lnu)
        np.testing.assert_array_equal(lam, lam0)
        np.testing.assert_array_equal(lnu, lnu0)

    def test_finely_sampled_edge_left_alone(self):
        """No points are added where the grid already resolves the edge."""
        off = 1e-4 * LYMAN
        lam = np.array([800.0, LYMAN - 0.5 * off, LYMAN + 0.5 * off, 1500.0])
        new_lam, _ = add_ionisation_edges(lam, step_sed(lam), ions=("HI",))
        np.testing.assert_array_equal(new_lam, lam)

    def test_edges_outside_range_skipped(self):
        """Edges outside the wavelength range are ignored."""
        lam = np.arange(1000.0, 5000.0, 10.0)
        new_lam, _ = add_ionisation_edges(lam, step_sed(lam))
        assert len(new_lam) == len(lam)

    def test_units_preserved(self):
        """Inputs with unyt units come back with the same units."""
        lam = np.arange(100.0, 2000.0, 10.0) * angstrom
        lnu = step_sed(lam.value) * erg / s / Hz
        new_lam, new_lnu = add_ionisation_edges(lam.to("nm"), lnu)
        assert new_lam.units == lam.to("nm").units
        assert new_lnu.units == lnu.units
        assert len(new_lam) == len(lam) + 6

    @pytest.mark.parametrize("jump", [10.0, 100.0, 1e4])
    def test_fixes_q_after_cloudy_style_interpolation(self, jump):
        """Q is preserved when a coarse SED is re-gridded like cloudy does.

        On a 10 A grid with points at 905 and 915 A, log-log interpolation
        onto a mesh with a point at 911.55 A smears the Lyman edge and
        inflates Q. With the edge points inserted, Q is preserved.
        """
        lam = np.arange(500.0, 2000.0, 10.0) + 5.0  # ..., 905, 915, ...
        lnu = step_sed(lam, jump=jump)
        # Cloudy-like mesh over the same range, with points either side of
        # the edge at the positions of cloudy's own mesh (911.55, 913.27 A)
        mesh = np.sort(
            np.concatenate(
                [np.geomspace(lam[0], lam[-1], 2000), [911.55, 913.27]]
            )
        )
        q_true = q_rate(lam, lnu)

        q_smeared = q_rate(mesh, loglog_onto(mesh, lam, lnu))
        # The bug: Q is inflated (from ~2% for a 10x drop at the edge to
        # ~2x for a 10^4x drop in this flat test SED)
        assert q_smeared / q_true > 1.01

        fl, fn = add_ionisation_edges(lam, lnu)
        q_fixed = q_rate(mesh, loglog_onto(mesh, fl, fn))
        assert q_fixed / q_true == pytest.approx(1.0, rel=0.01)


@pytest.mark.parametrize("module", [cloudy17, cloudy23])
class TestTableSed:
    """Tests for ShapeCommands.table_sed in each cloudy version."""

    def test_writes_sharpened_sed(self, module, tmp_path):
        """The written table has the edge points and increasing energy."""
        lam = np.arange(100.0, 2000.0, 10.0)
        lnu = step_sed(lam)
        cmds = module.ShapeCommands.table_sed(
            "model", lam, lnu, output_dir=str(tmp_path)
        )
        assert cmds == ['table SED "model.sed" \n']
        table = np.loadtxt(tmp_path / "model.sed")
        assert table.shape == (len(lam) + 6, 2)
        assert np.all(np.diff(table[:, 0]) > 0)

    def test_can_disable_sharpening(self, module, tmp_path):
        """sharpen_ionisation_edges=False writes the original points."""
        lam = np.arange(100.0, 2000.0, 10.0)
        module.ShapeCommands.table_sed(
            "model",
            lam,
            step_sed(lam),
            output_dir=str(tmp_path),
            sharpen_ionisation_edges=False,
        )
        assert np.loadtxt(tmp_path / "model.sed").shape == (len(lam), 2)

    def test_inputs_not_modified(self, module, tmp_path):
        """Neither lam nor lnu (with zeros) is modified in place."""
        lam = np.arange(100.0, 2000.0, 10.0)
        lnu = step_sed(lam)
        lnu[:5] = 0.0
        lam0, lnu0 = lam.copy(), lnu.copy()
        module.ShapeCommands.table_sed(
            "model", lam, lnu, output_dir=str(tmp_path)
        )
        np.testing.assert_array_equal(lam, lam0)
        np.testing.assert_array_equal(lnu, lnu0)
