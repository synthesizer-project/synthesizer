"""Tests for PhotometryCollection array-based construction and lookup."""

import numpy as np
import pytest
from unyt import Hz, angstrom, c, erg, s

from synthesizer import exceptions
from synthesizer.instruments import FilterCollection
from synthesizer.photometry import PhotometryCollection


def _make_filters(lam):
    """Create a simple FilterCollection for tests."""
    tophat_dict = {
        "f1": {"lam_eff": 2000 * angstrom, "lam_fwhm": 400 * angstrom},
        "f2": {"lam_eff": 3000 * angstrom, "lam_fwhm": 500 * angstrom},
        "f3": {"lam_eff": 4200 * angstrom, "lam_fwhm": 600 * angstrom},
    }
    return FilterCollection(tophat_dict=tophat_dict, new_lam=lam)


def test_photometry_collection_array_constructor_filter_first():
    """Array constructor should use filter-first layout."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)

    # Shape: (nfilters, nobj)
    arr = np.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]]) * erg / s / Hz
    pc = PhotometryCollection(
        filters,
        photometry=arr,
    )

    assert pc.shape == (3, 2)
    np.testing.assert_allclose(pc["f1"].value, [1.0, 0.5])
    np.testing.assert_allclose(pc["f3"].value, [3.0, 2.5])
    assert pc.photometry.shape == (3, 2)


def test_photometry_collection_requires_units_on_input_array():
    """Constructor should reject plain numpy arrays without units."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)

    arr = np.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]])
    with pytest.raises(exceptions.MissingUnits):
        PhotometryCollection(
            filters,
            photometry=arr,
        )


def test_photometry_collection_select_preserves_lookup():
    """Selecting filters should keep correct array-backed lookup."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)
    nu = (c / lam).to("Hz").value

    # Build realistic broadband values via batched filter application.
    spectrum = np.vstack(
        [
            np.ones(len(lam)),
            np.linspace(0.1, 1.0, len(lam)),
        ]
    )
    bb = filters.apply_filters(spectrum, nu=nu, integration_method="trapz")
    pc = PhotometryCollection(
        filters,
        photometry=bb * erg / s / Hz,
    )

    sub = pc.select("f2", "f3")
    assert sub.filter_codes == ["f2", "f3"]
    np.testing.assert_allclose(sub["f2"].value, pc["f2"].value)
    np.testing.assert_allclose(sub["f3"].value, pc["f3"].value)


class TestPhotometryThreading:
    """Test photometry generation with and without threading."""

    def test_threaded_particle_photometry_ngp(
        self,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test per-particle photometry with threading (NGP grid assignment).

        Serial and parallel execution should give identical results.
        """
        nebular_emission_model.set_per_particle(True)

        # Get spectra first
        random_part_stars.get_spectra(
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Create a simple filter collection (must match the spectra wavelength)
        lam = np.linspace(1000, 5000, 500) * angstrom
        filters = _make_filters(lam)

        # Compute photometry with 1 thread
        serial_photo = random_part_stars.get_particle_photo_lnu(
            filters,
            verbose=False,
            nthreads=1,
        )

        # Clear and recompute with multiple threads
        random_part_stars.clear_all_photometry()
        threaded_photo = random_part_stars.get_particle_photo_lnu(
            filters,
            verbose=False,
            nthreads=4,
        )

        # Verify results are identical (compare the nebular emission)
        assert np.allclose(
            serial_photo["nebular"].photometry,
            threaded_photo["nebular"].photometry,
            rtol=1e-10,
        ), "Serial and threaded particle photometry differ"

    def test_threaded_integrated_photometry_ngp(
        self,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test integrated photometry with threading (NGP grid assignment).

        Serial and parallel execution should give identical results.
        """
        nebular_emission_model.set_per_particle(False)

        # Get spectra first
        random_part_stars.get_spectra(
            nebular_emission_model,
            grid_assignment_method="ngp",
        )

        # Create a simple filter collection (must match the spectra wavelength)
        lam = np.linspace(1000, 5000, 500) * angstrom
        filters = _make_filters(lam)

        # Compute photometry with 1 thread
        serial_photo = random_part_stars.get_photo_lnu(
            filters,
            verbose=False,
            nthreads=1,
        )

        # Clear and recompute with multiple threads
        random_part_stars.clear_all_photometry()
        threaded_photo = random_part_stars.get_photo_lnu(
            filters,
            verbose=False,
            nthreads=4,
        )

        # Verify results are identical (compare the nebular emission)
        assert np.allclose(
            serial_photo["nebular"].photometry,
            threaded_photo["nebular"].photometry,
            rtol=1e-10,
        ), "Serial and threaded integrated photometry differ"

    def test_threaded_particle_photometry_cic(
        self,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test per-particle photometry with threading (CIC grid assignment).

        Serial and parallel execution should give identical results.
        """
        nebular_emission_model.set_per_particle(True)

        # Get spectra first
        random_part_stars.get_spectra(
            nebular_emission_model,
            grid_assignment_method="cic",
        )

        # Create a simple filter collection (must match the spectra wavelength)
        lam = np.linspace(1000, 5000, 500) * angstrom
        filters = _make_filters(lam)

        # Compute photometry with 1 thread
        serial_photo = random_part_stars.get_particle_photo_lnu(
            filters,
            verbose=False,
            nthreads=1,
        )

        # Clear and recompute with multiple threads
        random_part_stars.clear_all_photometry()
        threaded_photo = random_part_stars.get_particle_photo_lnu(
            filters,
            verbose=False,
            nthreads=4,
        )

        # Verify results are identical (compare the nebular emission)
        assert np.allclose(
            serial_photo["nebular"].photometry,
            threaded_photo["nebular"].photometry,
            rtol=1e-10,
        ), "Serial and threaded particle photometry differ (CIC)"

    def test_threaded_integrated_photometry_cic(
        self,
        nebular_emission_model,
        random_part_stars,
    ):
        """Test integrated photometry with threading (CIC grid assignment).

        Serial and parallel execution should give identical results.
        """
        nebular_emission_model.set_per_particle(False)

        # Get spectra first
        random_part_stars.get_spectra(
            nebular_emission_model,
            grid_assignment_method="cic",
        )

        # Create a simple filter collection (must match the spectra wavelength)
        lam = np.linspace(1000, 5000, 500) * angstrom
        filters = _make_filters(lam)

        # Compute photometry with 1 thread
        serial_photo = random_part_stars.get_photo_lnu(
            filters,
            verbose=False,
            nthreads=1,
        )

        # Clear and recompute with multiple threads
        random_part_stars.clear_all_photometry()
        threaded_photo = random_part_stars.get_photo_lnu(
            filters,
            verbose=False,
            nthreads=4,
        )

        # Verify results are identical (compare the nebular emission)
        assert np.allclose(
            serial_photo["nebular"].photometry,
            threaded_photo["nebular"].photometry,
            rtol=1e-10,
        ), "Serial and threaded integrated photometry differ (CIC)"
