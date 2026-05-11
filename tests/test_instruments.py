"""Tests for instrument and instrument collection behaviour."""

import h5py
import numpy as np
import pytest
from unyt import angstrom, arcsecond

from synthesizer import exceptions
from synthesizer.instruments import (
    GALEX,
    FilterCollection,
    Instrument,
    InstrumentCollection,
    IntegratedFieldUnit,
    PhotometricImager,
    PhotometricInstrument,
    SpectroscopicInstrument,
)
from synthesizer.instruments.premade import GALEXFUV, GALEXNUV


def test_galex_filters_are_not_duplicated_in_collection():
    """Ensure the premade GALEX collection keeps one filter per instrument."""
    fuv = GALEXFUV()
    nuv = GALEXNUV()
    galex = GALEX()

    assert fuv.filters.filter_codes == ["GALEX/GALEX.FUV"]
    assert nuv.filters.filter_codes == ["GALEX/GALEX.NUV"]
    assert galex["GALEXFUV"].filters.filter_codes == ["GALEX/GALEX.FUV"]
    assert galex["GALEXNUV"].filters.filter_codes == ["GALEX/GALEX.NUV"]
    assert galex.all_filters.filter_codes.count("GALEX/GALEX.NUV") == 1
    assert galex.all_filters.filter_codes == [
        "GALEX/GALEX.FUV",
        "GALEX/GALEX.NUV",
    ]


def test_instrument_collection_does_not_mutate_existing_filters():
    """Ensure collection filter bookkeeping does not alter members."""
    lam = np.linspace(1000, 3000, 32) * angstrom
    first_filters = FilterCollection(
        generic_dict={"filter_a": np.ones(lam.size)},
        new_lam=lam,
    )
    second_filters = FilterCollection(
        generic_dict={"filter_b": np.ones(lam.size)},
        new_lam=lam,
    )

    first = Instrument(
        label="first",
        filters=first_filters,
        resolution=1 * arcsecond,
    )
    second = Instrument(
        label="second",
        filters=second_filters,
        resolution=1 * arcsecond,
    )

    collection = InstrumentCollection()
    collection.add_instruments(first, second)

    assert first.filters.filter_codes == ["filter_a"]
    assert second.filters.filter_codes == ["filter_b"]
    assert collection.all_filters.filter_codes == ["filter_a", "filter_b"]


def test_unsupported_mixed_noise_configuration_raises():
    """Unsupported mixed-mode configurations should fail explicitly."""
    with pytest.raises(exceptions.InconsistentArguments):
        Instrument(
            label="spec",
            lam=np.linspace(1000, 3000, 32) * angstrom,
            resolution=1 * arcsecond,
            noise_source_maps={"filter_a": np.ones((8, 8))},
        )


def test_add_filters_does_not_mutate_on_invalid_noise_payload():
    """Invalid add_filters input should not partially mutate the instrument."""
    lam = np.linspace(1000, 3000, 32) * angstrom
    base_filters = FilterCollection(
        generic_dict={"filter_a": np.ones(lam.size)},
        new_lam=lam,
    )
    new_filters = FilterCollection(
        generic_dict={"filter_b": np.ones(lam.size)},
        new_lam=lam,
    )

    inst = Instrument(
        label="test",
        filters=base_filters,
        resolution=1 * arcsecond,
        noise_source_maps={"filter_a": np.ones((8, 8))},
    )

    with pytest.raises(exceptions.MissingArgument):
        inst.add_filters(
            new_filters,
            noise_maps={"filter_b": np.ones((8, 8))},
        )

    assert inst.filters.filter_codes == ["filter_a"]
    assert set(inst.noise_source_maps.keys()) == {"filter_a"}
    assert inst.noise_maps is None


def test_photometric_imager_inherits_add_filters_unchanged():
    """Imagers should reuse the photometric add_filters implementation."""
    assert PhotometricImager.add_filters is PhotometricInstrument.add_filters


@pytest.mark.parametrize(
    ("instrument", "expected_type"),
    [
        (
            PhotometricInstrument(
                label="phot",
                filters=FilterCollection(
                    generic_dict={"filter_a": np.ones(32)},
                    new_lam=np.linspace(1000, 3000, 32) * angstrom,
                ),
            ),
            PhotometricInstrument,
        ),
        (
            PhotometricImager(
                label="img",
                filters=FilterCollection(
                    generic_dict={"filter_b": np.ones(32)},
                    new_lam=np.linspace(1000, 3000, 32) * angstrom,
                ),
                resolution=1 * arcsecond,
            ),
            PhotometricImager,
        ),
        (
            SpectroscopicInstrument(
                label="spec",
                lam=np.linspace(1000, 3000, 32) * angstrom,
            ),
            SpectroscopicInstrument,
        ),
        (
            IntegratedFieldUnit(
                label="ifu",
                lam=np.linspace(1000, 3000, 32) * angstrom,
                resolution=1 * arcsecond,
            ),
            IntegratedFieldUnit,
        ),
    ],
)
def test_instrument_hdf5_roundtrip_preserves_specialised_type(
    tmp_path, instrument, expected_type
):
    """Serialisation should round-trip specialised instrument classes."""
    path = tmp_path / f"{instrument.label}.hdf5"

    with h5py.File(path, "w") as hdf:
        instrument.to_hdf5(hdf.create_group("Instrument"))

    with h5py.File(path, "r") as hdf:
        loaded = Instrument._from_hdf5(hdf["Instrument"])

    assert isinstance(loaded, expected_type)
    assert loaded == instrument
