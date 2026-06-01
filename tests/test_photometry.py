"""Tests for PhotometryCollection array-based construction and lookup."""

import numpy as np
import pytest
from unyt import Hz, angstrom, c, erg, nJy, s

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


def test_photometry_collection_preserves_units_on_input_array():
    """Constructor should preserve units on input array."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)

    arr = np.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]]) * erg / s / Hz
    pc = PhotometryCollection(
        filters,
        photometry=arr,
    )

    assert pc.photometry is not None
    assert pc.photo_lnu is not None
    assert pc.photo_fnu is None

    assert pc.photometry.units.same_dimensions_as(erg / s / Hz)
    assert pc.photo_lnu.units.same_dimensions_as(erg / s / Hz)

    assert np.allclose(
        pc.photometry.to(erg / s / Hz).value,
        arr.to(erg / s / Hz).value,
    )

    arr = np.array([[1.0, 0.5], [2.0, 1.5], [3.0, 2.5]]) * nJy
    pc = PhotometryCollection(
        filters,
        photometry=arr,
    )

    assert pc.photometry is not None
    assert pc.photo_lnu is None
    assert pc.photo_fnu is not None
    assert pc.photometry.units.same_dimensions_as(nJy)
    assert pc.photo_fnu.units.same_dimensions_as(nJy)

    assert np.allclose(
        pc.photometry.to(nJy).value,
        arr.to(nJy).value,
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


def test_apply_filters_supports_float32_inputs_and_outputs():
    """Batched photometry should preserve a float32 precision family."""
    lam = np.linspace(1000, 5000, 500, dtype=np.float32) * angstrom
    filters = _make_filters(lam)
    nu = np.ascontiguousarray((c / lam).to("Hz").value, dtype=np.float32)
    spectra = np.ascontiguousarray(
        np.vstack(
            [
                np.ones(len(lam), dtype=np.float32),
                np.linspace(0.1, 1.0, len(lam), dtype=np.float32),
            ]
        ),
        dtype=np.float32,
    )

    photometry = filters.apply_filters(
        spectra,
        nu=nu,
        integration_method="trapz",
        out_dtype=np.float32,
    )

    assert photometry.dtype == np.float32


def test_apply_filters_supports_float64_output_from_float32_inputs():
    """Users should be able to request float64 output independently."""
    lam = np.linspace(1000, 5000, 500, dtype=np.float32) * angstrom
    filters = _make_filters(lam)
    nu = np.ascontiguousarray((c / lam).to("Hz").value, dtype=np.float32)
    spectra = np.ascontiguousarray(
        np.vstack(
            [
                np.ones(len(lam), dtype=np.float32),
                np.linspace(0.1, 1.0, len(lam), dtype=np.float32),
            ]
        ),
        dtype=np.float32,
    )

    photometry = filters.apply_filters(
        spectra,
        nu=nu,
        integration_method="trapz",
        out_dtype=np.float64,
    )

    assert photometry.dtype == np.float64


def test_apply_filters_precision_combinations_agree_with_float64_reference():
    """Photometry precision combinations should agree within tolerance."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)

    spectra64 = np.ascontiguousarray(
        np.vstack(
            [
                np.linspace(0.2, 1.4, len(lam), dtype=np.float64),
                0.5 + 0.3 * np.sin(np.linspace(0.0, 8.0, len(lam))),
                np.exp(-0.5 * np.linspace(-3.0, 3.0, len(lam)) ** 2),
            ]
        ),
        dtype=np.float64,
    )
    nu64 = np.ascontiguousarray((c / lam).to("Hz").value, dtype=np.float64)

    reference = filters.apply_filters(
        spectra64,
        nu=nu64,
        integration_method="trapz",
        out_dtype=np.float64,
    )

    spectra32 = np.ascontiguousarray(spectra64, dtype=np.float32)
    nu32 = np.ascontiguousarray(nu64, dtype=np.float32)

    float32_to_float32 = filters.apply_filters(
        spectra32,
        nu=nu32,
        integration_method="trapz",
        out_dtype=np.float32,
    )
    float32_to_float64 = filters.apply_filters(
        spectra32,
        nu=nu32,
        integration_method="trapz",
        out_dtype=np.float64,
    )
    float64_to_float32 = filters.apply_filters(
        spectra64,
        nu=nu64,
        integration_method="trapz",
        out_dtype=np.float32,
    )

    assert reference.dtype == np.float64
    assert float32_to_float32.dtype == np.float32
    assert float32_to_float64.dtype == np.float64
    assert float64_to_float32.dtype == np.float32

    np.testing.assert_allclose(
        float32_to_float32,
        reference,
        rtol=5e-5,
        atol=5e-7,
    )
    np.testing.assert_allclose(
        float32_to_float64,
        reference,
        rtol=5e-5,
        atol=5e-7,
    )
    np.testing.assert_allclose(
        float64_to_float32,
        reference,
        rtol=5e-6,
        atol=5e-8,
    )


def test_apply_filters_rejects_mismatched_precision_families():
    """The extension should reject mixed input precision families clearly."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)
    nu = np.ascontiguousarray((c / lam).to("Hz").value, dtype=np.float64)
    spectra = np.ascontiguousarray(
        np.vstack(
            [
                np.ones(len(lam), dtype=np.float32),
                np.linspace(0.1, 1.0, len(lam), dtype=np.float32),
            ]
        ),
        dtype=np.float32,
    )

    with pytest.raises(TypeError, match="same floating-point dtype"):
        filters.apply_filters(
            spectra,
            nu=nu,
            integration_method="trapz",
            out_dtype=np.float32,
        )


def test_photometry_collection_addition():
    """Photometry collections with matching filters should add."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)

    first = PhotometryCollection(
        filters,
        photometry=np.array([1.0, 2.0, 3.0]) * erg / s / Hz,
    )
    second = PhotometryCollection(
        filters,
        photometry=np.array([0.5, 1.5, 2.5]) * erg / s / Hz,
    )

    total = first + second

    assert total.filter_codes == ["f1", "f2", "f3"]
    np.testing.assert_allclose(total.photometry.value, [1.5, 3.5, 5.5])


def test_photometry_collection_addition_requires_matching_filters():
    """Photometry collections with different filters should not add."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)
    other_filters = FilterCollection(
        tophat_dict={
            "f1": {"lam_eff": 2000 * angstrom, "lam_fwhm": 400 * angstrom},
            "f4": {"lam_eff": 3200 * angstrom, "lam_fwhm": 500 * angstrom},
            "f5": {"lam_eff": 4200 * angstrom, "lam_fwhm": 600 * angstrom},
        },
        new_lam=lam,
    )

    first = PhotometryCollection(
        filters,
        photometry=np.array([1.0, 2.0, 3.0]) * erg / s / Hz,
    )
    second = PhotometryCollection(
        other_filters,
        photometry=np.array([0.5, 1.5, 2.5]) * erg / s / Hz,
    )

    with pytest.raises(exceptions.InconsistentAddition):
        _ = first + second


def test_photometry_collection_sum():
    """Photometry collections should support Python sum()."""
    lam = np.linspace(1000, 5000, 500) * angstrom
    filters = _make_filters(lam)

    first = PhotometryCollection(
        filters,
        photometry=np.array([1.0, 2.0, 3.0]) * erg / s / Hz,
    )
    second = PhotometryCollection(
        filters,
        photometry=np.array([0.5, 1.5, 2.5]) * erg / s / Hz,
    )

    total = sum([first, second])

    np.testing.assert_allclose(total.photometry.value, [1.5, 3.5, 5.5])


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
