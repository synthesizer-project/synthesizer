"""Regression tests for loading cached instruments and filters."""

import h5py
import numpy as np
from unyt import arcsecond

from synthesizer.instruments import FilterCollection
from synthesizer.instruments.premade import GALEXFUV, GALEXNUV, JWSTNIRCam


def test_filter_collection_from_hdf5_reharmonises_svo_filters(tmp_path):
    """Loading should reinterpolate SVO filters onto collection wavelengths."""
    path = tmp_path / "bad_filters.hdf5"

    collection_lam = np.linspace(1000, 2000, 5)
    stored_lam = np.linspace(1000, 2000, 7)
    original_lam = np.linspace(900, 2100, 9)
    original_t = np.array([0.0, 0.0, 0.2, 0.7, 1.0, 0.7, 0.2, 0.0, 0.0])
    stored_t = np.interp(
        stored_lam, original_lam, original_t, left=0.0, right=0.0
    )

    with h5py.File(path, "w") as hdf:
        header = hdf.create_group("Header")
        header.attrs["synthesizer_version"] = "test"
        header.attrs["nfilters"] = 1
        header.attrs["Wavelength_units"] = "angstrom"
        header.attrs["filter_codes"] = ["Test/Inst.Filter"]
        header.create_dataset("Wavelengths", data=collection_lam)

        filter_group = hdf.create_group("Test.Inst.Filter")
        filter_group.attrs["filter_type"] = "SVO"
        filter_group.attrs["filter_code"] = "Test/Inst.Filter"
        filter_group.attrs["svo_url"] = "https://example.com/filter"
        filter_group.attrs["observatory"] = "Test"
        filter_group.attrs["instrument"] = "Inst"
        filter_group.attrs["filter_"] = "Filter"
        filter_group.create_dataset("Transmission", data=stored_t)
        filter_group.create_dataset("Original_Wavelength", data=original_lam)
        filter_group.create_dataset("Original_Transmission", data=original_t)

    with h5py.File(path, "r") as hdf:
        filters = FilterCollection._from_hdf5(hdf)

    assert filters.filter_codes == ["Test/Inst.Filter"]
    assert len(filters.lam) == len(collection_lam)
    assert len(filters["Test/Inst.Filter"].lam) == len(collection_lam)
    assert len(filters["Test/Inst.Filter"].t) == len(collection_lam)
    assert np.allclose(filters["Test/Inst.Filter"]._lam, collection_lam)
    assert np.allclose(
        filters["Test/Inst.Filter"].t,
        np.interp(
            collection_lam, original_lam, original_t, left=0.0, right=0.0
        ),
    )


def test_filter_collection_from_hdf5_reharmonises_generic_filters(tmp_path):
    """Loading should also reharmonise cached generic/manual filters."""
    path = tmp_path / "bad_generic_filters.hdf5"

    collection_lam = np.linspace(1000, 2000, 5)
    stored_lam = np.linspace(1000, 2000, 7)
    original_lam = np.linspace(900, 2100, 9)
    original_t = np.array([0.0, 0.0, 0.2, 0.7, 1.0, 0.7, 0.2, 0.0, 0.0])
    stored_t = np.interp(
        stored_lam, original_lam, original_t, left=0.0, right=0.0
    )

    with h5py.File(path, "w") as hdf:
        header = hdf.create_group("Header")
        header.attrs["synthesizer_version"] = "test"
        header.attrs["nfilters"] = 1
        header.attrs["Wavelength_units"] = "angstrom"
        header.attrs["filter_codes"] = ["GALEX/GALEX.NUV"]
        header.create_dataset("Wavelengths", data=collection_lam)

        filter_group = hdf.create_group("GALEX.GALEX.NUV")
        filter_group.attrs["filter_type"] = "Generic"
        filter_group.attrs["filter_code"] = "GALEX/GALEX.NUV"
        filter_group.create_dataset("Transmission", data=stored_t)
        filter_group.create_dataset("Original_Wavelength", data=original_lam)
        filter_group.create_dataset("Original_Transmission", data=original_t)

    with h5py.File(path, "r") as hdf:
        filters = FilterCollection._from_hdf5(hdf)

    assert filters.filter_codes == ["GALEX/GALEX.NUV"]
    assert len(filters.lam) == len(collection_lam)
    assert len(filters["GALEX/GALEX.NUV"].lam) == len(collection_lam)
    assert len(filters["GALEX/GALEX.NUV"].t) == len(collection_lam)
    assert np.allclose(filters["GALEX/GALEX.NUV"]._lam, collection_lam)
    assert np.allclose(
        filters["GALEX/GALEX.NUV"].t,
        np.interp(
            collection_lam, original_lam, original_t, left=0.0, right=0.0
        ),
    )


def test_galex_manual_filters_round_trip_with_original_grids(tmp_path):
    """Saved GALEX manual filters keep original grids for later reloads."""
    path = tmp_path / "galex_filters.hdf5"

    filters = GALEXFUV().filters + GALEXNUV().filters
    filters.write_filters(path)

    with h5py.File(path, "r") as hdf:
        fuv_group = hdf["GALEX.GALEX.FUV"]
        nuv_group = hdf["GALEX.GALEX.NUV"]
        assert "Original_Wavelength" in fuv_group
        assert "Original_Transmission" in fuv_group
        assert "Original_Wavelength" in nuv_group
        assert "Original_Transmission" in nuv_group

        loaded = FilterCollection._from_hdf5(hdf)

    assert loaded.filter_codes == ["GALEX/GALEX.FUV", "GALEX/GALEX.NUV"]
    for code in loaded.filter_codes:
        assert len(loaded[code].lam) == len(loaded.lam)
        assert len(loaded[code].t) == len(loaded.lam)


def test_cached_premade_load_preserves_filter_subset(tmp_path):
    """Premade cached loads should still support selecting filter subsets."""
    path = tmp_path / "nircam.hdf5"

    instrument = JWSTNIRCam(
        psfs={
            "JWST/NIRCam.F070W": np.ones((2, 2)),
            "JWST/NIRCam.F090W": np.full((2, 2), 2.0),
        },
        filter_subset=("JWST/NIRCam.F070W", "JWST/NIRCam.F090W"),
    )

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf)

    loaded = JWSTNIRCam.load(
        filepath=path,
        filter_subset=("JWST/NIRCam.F090W",),
    )

    assert loaded.filters.filter_codes == ["JWST/NIRCam.F090W"]
    assert set(loaded.psfs) == {"JWST/NIRCam.F090W"}
    assert loaded.resolution == 0.031 * arcsecond
