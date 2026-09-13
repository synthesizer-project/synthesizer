"""Tests for dust attenuation transformers."""

from pathlib import Path

import h5py
import numpy as np
import pytest
from unyt import Msun, angstrom, cm, g, pc

from synthesizer import exceptions
from synthesizer.emission_models.transformers.dust_attenuation import (
    DraineLiGrainCurves,
)


def _write_draine_li_grid(path: Path, log_on_read=False):
    """Write a minimal attenuation grid in the standard Grid format."""
    with h5py.File(path, "w") as hdf:
        hdf.attrs["axes"] = np.array(["dtg"], dtype=object)
        hdf.attrs["WeightVariable"] = "None"

        axes = hdf.create_group("axes")
        dtg = axes.create_dataset("dtg", data=np.array([0.1, 0.2]))
        dtg.attrs["Units"] = "dimensionless"
        dtg.attrs["log_on_read"] = log_on_read

        spectra = hdf.create_group("spectra")
        wavelength = spectra.create_dataset(
            "wavelength", data=np.array([1000.0, 2000.0, 3000.0])
        )
        wavelength.attrs["Units"] = "Å"
        spectra.create_dataset(
            "graphite_a0.01um",
            data=np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        )
        spectra.create_dataset(
            "silicate_a0.1um",
            data=np.array([[0.5, 1.0, 1.5], [1.0, 2.0, 3.0]]),
        )


@pytest.fixture
def draine_li_grid(tmp_path):
    """Create a temporary Draine-Li attenuation grid."""
    grid_path = tmp_path / "draine_li_test_grid.hdf5"
    _write_draine_li_grid(grid_path)
    return grid_path


@pytest.fixture
def draine_li_log_grid(tmp_path):
    """Create a temporary Draine-Li attenuation grid with log dtg."""
    grid_path = tmp_path / "draine_li_log_test_grid.hdf5"
    _write_draine_li_grid(grid_path, log_on_read=True)
    return grid_path


def _curve_to_tau(curve, hydrogen_col):
    """Convert a curve in mag cm^2 / H to optical depth."""
    mu = 1.4
    m_h = 1.6738e-24 * g
    gas_mass_per_h = (mu * m_h).to(Msun)
    return (
        (((curve / 1.086) * cm**2).to(pc**2) / gas_mass_per_h) * hydrogen_col
    ).value


def test_draine_li_uses_grid_extraction(draine_li_grid):
    """Particle extraction reproduces the expected attenuation curves."""
    dust_curve = DraineLiGrainCurves(
        lam=np.array([1500.0, 2500.0]) * angstrom,
        grid_name=draine_li_grid.name,
        grid_dir=draine_li_grid.parent,
        grain_dict={"graphite": [0.01], "silicate": [0.1]},
    )

    sigmalos_h = np.array([1.0, 1.0]) * Msun / pc**2
    graphite = np.array([0.14, 0.28]) * Msun / pc**2
    silicate = np.array([0.14, 0.14]) * Msun / pc**2

    tau = dust_curve.get_tau_at_lam(
        np.array([1500.0, 2500.0]) * angstrom,
        sigmalos_H=sigmalos_h,
        sigmalos_graphite_a0p01um=graphite,
        sigmalos_silicate_a0p1um=silicate,
    )

    expected = np.vstack(
        [
            _curve_to_tau(np.array([1.5, 2.5]), 1.0 * Msun / pc**2),
            _curve_to_tau(np.array([3.0, 5.0]), 1.0 * Msun / pc**2),
        ]
    )
    expected += np.vstack(
        [
            _curve_to_tau(np.array([0.75, 1.25]), 1.0 * Msun / pc**2),
            _curve_to_tau(np.array([0.75, 1.25]), 1.0 * Msun / pc**2),
        ]
    )

    np.testing.assert_allclose(tau, expected)


def test_draine_li_masks_zero_and_nan_columns(draine_li_grid):
    """Zero and NaN line-of-sight columns should contribute zero tau."""
    dust_curve = DraineLiGrainCurves(
        lam=np.array([1500.0, 2500.0]) * angstrom,
        grid_name=draine_li_grid.name,
        grid_dir=draine_li_grid.parent,
        grain_dict={"graphite": [0.01], "silicate": [0.1]},
    )

    sigmalos_h = np.array([1.0, 0.0, np.nan, 1.0]) * Msun / pc**2
    graphite = np.array([0.14, 0.14, 0.14, np.nan]) * Msun / pc**2
    silicate = np.array([0.14, 0.14, 0.14, 0.14]) * Msun / pc**2

    tau = dust_curve.get_tau_at_lam(
        np.array([1500.0, 2500.0]) * angstrom,
        sigmalos_H=sigmalos_h,
        sigmalos_graphite_a0p01um=graphite,
        sigmalos_silicate_a0p1um=silicate,
    )

    expected = np.zeros((4, 2))
    expected[0] = _curve_to_tau(
        np.array([1.5, 2.5]), 1.0 * Msun / pc**2
    ) + _curve_to_tau(np.array([0.75, 1.25]), 1.0 * Msun / pc**2)
    expected[3] = _curve_to_tau(np.array([0.75, 1.25]), 1.0 * Msun / pc**2)

    np.testing.assert_allclose(tau, expected)
    assert np.all(np.isfinite(tau))


def test_draine_li_resamples_non_native_wavelengths(draine_li_grid):
    """Non-native wavelength requests should use grid resampling."""
    dust_curve = DraineLiGrainCurves(
        lam=np.array([1500.0, 2500.0]) * angstrom,
        grid_name=draine_li_grid.name,
        grid_dir=draine_li_grid.parent,
        grain_dict={"graphite": [0.01], "silicate": [0.1]},
    )

    tau = dust_curve.get_tau_at_lam(
        np.array([1500.0, 2500.0]) * angstrom,
        sigmalos_H=np.array([1.0, 1.0]) * Msun / pc**2,
        sigmalos_graphite_a0p01um=np.array([0.14, 0.14]) * Msun / pc**2,
        sigmalos_silicate_a0p1um=np.array([0.14, 0.14]) * Msun / pc**2,
    )

    expected = _curve_to_tau(
        np.array([1.5, 2.5]), 1.0 * Msun / pc**2
    ) + _curve_to_tau(np.array([0.75, 1.25]), 1.0 * Msun / pc**2)

    np.testing.assert_allclose(tau, np.vstack([expected, expected]))


def test_draine_li_resamples_scalar_wavelengths(draine_li_grid):
    """Scalar wavelength requests should also use grid resampling."""
    dust_curve = DraineLiGrainCurves(
        lam=np.array([1500.0]) * angstrom,
        grid_name=draine_li_grid.name,
        grid_dir=draine_li_grid.parent,
        grain_dict={"graphite": [0.01], "silicate": [0.1]},
    )

    tau = dust_curve.get_tau_at_lam(
        np.array([1500.0]) * angstrom,
        sigmalos_H=np.array([1.0, 1.0]) * Msun / pc**2,
        sigmalos_graphite_a0p01um=np.array([0.14, 0.14]) * Msun / pc**2,
        sigmalos_silicate_a0p1um=np.array([0.14, 0.14]) * Msun / pc**2,
    )

    expected = _curve_to_tau(
        np.array([1.5]), 1.0 * Msun / pc**2
    ) + _curve_to_tau(np.array([0.75]), 1.0 * Msun / pc**2)

    np.testing.assert_allclose(tau, np.vstack([expected, expected]))


def test_draine_li_supports_log10_dtg_grids(draine_li_log_grid):
    """Logarithmic dtg extraction grids should work correctly."""
    dust_curve = DraineLiGrainCurves(
        lam=np.array([1500.0, 2500.0]) * angstrom,
        grid_name=draine_li_log_grid.name,
        grid_dir=draine_li_log_grid.parent,
        grain_dict={"graphite": [0.01]},
    )

    tau = dust_curve.get_tau_at_lam(
        np.array([1500.0, 2500.0]) * angstrom,
        sigmalos_H=np.array([1.0, 1.0]) * Msun / pc**2,
        sigmalos_graphite_a0p01um=np.array([0.14, 0.28]) * Msun / pc**2,
    )

    expected = np.vstack(
        [
            _curve_to_tau(np.array([1.5, 2.5]), 1.0 * Msun / pc**2),
            _curve_to_tau(np.array([3.0, 5.0]), 1.0 * Msun / pc**2),
        ]
    )

    np.testing.assert_allclose(tau, expected)


def test_draine_li_ignores_negative_columns(draine_li_grid):
    """Negative column densities are treated as non-contributing inputs."""
    dust_curve = DraineLiGrainCurves(
        lam=np.array([1000.0]) * angstrom,
        grid_name=draine_li_grid.name,
        grid_dir=draine_li_grid.parent,
        grain_dict={"graphite": [0.01]},
    )

    tau = dust_curve.get_tau_at_lam(
        np.array([1000.0]) * angstrom,
        sigmalos_H=np.array([1.0, 1.0]) * Msun / pc**2,
        sigmalos_graphite_a0p01um=np.array([-0.14, -0.14]) * Msun / pc**2,
    )

    np.testing.assert_allclose(tau, np.zeros((2, 1)))


def test_draine_li_rejects_invalid_grid_axis(tmp_path):
    """The attenuation grid must expose exactly one dtg-like axis."""
    grid_path = tmp_path / "bad_draine_li_grid.hdf5"
    with h5py.File(grid_path, "w") as hdf:
        hdf.attrs["axes"] = np.array(["size"], dtype=object)
        hdf.attrs["WeightVariable"] = "None"

        axes = hdf.create_group("axes")
        size = axes.create_dataset("size", data=np.array([0.1, 0.2]))
        size.attrs["Units"] = "dimensionless"
        size.attrs["log_on_read"] = False

        spectra = hdf.create_group("spectra")
        wavelength = spectra.create_dataset(
            "wavelength", data=np.array([1000.0, 2000.0, 3000.0])
        )
        wavelength.attrs["Units"] = "Å"
        spectra.create_dataset(
            "graphite_a0.01um",
            data=np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        )

    with pytest.raises(exceptions.UnimplementedFunctionality):
        DraineLiGrainCurves(
            lam=np.array([1000.0]) * angstrom,
            grid_name=grid_path.name,
            grid_dir=grid_path.parent,
            grain_dict={"graphite": [0.01]},
        )


def test_draine_li_rejects_missing_component_grid(tmp_path):
    """The attenuation grid must contain every requested grain component."""
    grid_path = tmp_path / "missing_component_draine_li_grid.hdf5"
    with h5py.File(grid_path, "w") as hdf:
        hdf.attrs["axes"] = np.array(["dtg"], dtype=object)
        hdf.attrs["WeightVariable"] = "None"

        axes = hdf.create_group("axes")
        dtg = axes.create_dataset("dtg", data=np.array([0.1, 0.2]))
        dtg.attrs["Units"] = "dimensionless"
        dtg.attrs["log_on_read"] = False

        spectra = hdf.create_group("spectra")
        wavelength = spectra.create_dataset(
            "wavelength", data=np.array([1000.0, 2000.0, 3000.0])
        )
        wavelength.attrs["Units"] = "Å"
        spectra.create_dataset(
            "graphite_a0.01um",
            data=np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]),
        )

    with pytest.raises(exceptions.InconsistentArguments):
        DraineLiGrainCurves(
            lam=np.array([1000.0]) * angstrom,
            grid_name=grid_path.name,
            grid_dir=grid_path.parent,
            grain_dict={"graphite": [0.01], "silicate": [0.1]},
        )


def test_draine_li_tau_independent_of_hydrogen(draine_li_grid):
    """Tau should depend only on dust column density, not hydrogen column."""
    dust_curve = DraineLiGrainCurves(
        lam=np.array([1500.0, 2500.0]) * angstrom,
        grid_name=draine_li_grid.name,
        grid_dir=draine_li_grid.parent,
        grain_dict={"graphite": [0.01], "silicate": [0.1]},
    )

    lam = np.array([1500.0, 2500.0]) * angstrom
    graphite = np.array([0.14, 0.14]) * Msun / pc**2
    silicate = np.array([0.14, 0.14]) * Msun / pc**2

    # Case 1: Base hydrogen column density
    sigmalos_h_1 = np.array([1.0, 1.0]) * Msun / pc**2
    tau_1 = dust_curve.get_tau_at_lam(
        lam,
        sigmalos_H=sigmalos_h_1,
        sigmalos_graphite_a0p01um=graphite,
        sigmalos_silicate_a0p1um=silicate,
    )

    # Case 2: Half hydrogen column density (double the dust-to-gas ratio)
    sigmalos_h_2 = np.array([0.5, 0.5]) * Msun / pc**2
    tau_2 = dust_curve.get_tau_at_lam(
        lam,
        sigmalos_H=sigmalos_h_2,
        sigmalos_graphite_a0p01um=graphite,
        sigmalos_silicate_a0p1um=silicate,
    )

    np.testing.assert_allclose(tau_1, tau_2)


class TestParameterApplicationIsIsolated:
    """Test parameters found on a model or emitter don't mutate the curve.

    A dust curve can be shared by many models, and the parameters it needs
    can come from each of those models rather than from the curve itself. They
    are applied to a copy so that one model's parameters cannot be seen by
    another, whether the two are transformed one after the other or at the
    same time.
    """

    def test_the_curve_is_left_alone(self):
        """Test a parameter passed for one call does not stick."""
        from synthesizer.emission_models.attenuation import PowerLaw

        curve = PowerLaw(slope=-1.0)
        lam = np.linspace(1000, 10000, 100) * angstrom

        curve.get_transmission(0.5, lam, slope=-0.5)

        assert curve.slope == -1.0

    def test_the_parameter_reaches_the_calculation(self):
        """Test the applied parameter is the one used."""
        from synthesizer.emission_models.attenuation import PowerLaw

        lam = np.linspace(1000, 10000, 100) * angstrom

        overridden = PowerLaw(slope=-1.0).get_transmission(
            0.5, lam, slope=-0.5
        )
        directly = PowerLaw(slope=-0.5).get_transmission(0.5, lam)

        assert np.allclose(overridden, directly)

    def test_concurrent_calls_do_not_interfere(self):
        """Test two threads sharing a curve each get their own answer.

        The threads are held inside the calculation by a barrier, so every
        one of them is between having its parameters applied and having read
        them. Applying them to the shared curve rather than to a copy would
        leave them all reading whichever thread wrote last.
        """
        import threading
        from concurrent.futures import ThreadPoolExecutor

        from synthesizer.emission_models.attenuation import PowerLaw

        slopes = [-2.0, -1.5, -1.0, -0.5, -0.1]
        barrier = threading.Barrier(len(slopes))

        class BarrierPowerLaw(PowerLaw):
            """A power law which waits for its peers before reading slope."""

            def get_tau(self, lam):
                """Return the optical depth, once every thread is here."""
                barrier.wait(timeout=10)
                return super().get_tau(lam)

        shared = BarrierPowerLaw(slope=-1.0)
        lam = np.linspace(1000, 10000, 1000) * angstrom

        expected = {
            slope: PowerLaw(slope=slope).get_transmission(0.5, lam)
            for slope in slopes
        }

        with ThreadPoolExecutor(max_workers=len(slopes)) as pool:
            results = list(
                pool.map(
                    lambda slope: (
                        slope,
                        shared.get_transmission(0.5, lam, slope=slope),
                    ),
                    slopes,
                )
            )

        for slope, transmission in results:
            assert np.allclose(transmission, expected[slope])

        assert shared.slope == -1.0
