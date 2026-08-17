"""Tests for the parameter variation machinery.

This covers the ParameterList/ParameterDistribution markers, the guards which
stop an unexpanded marker being used as a value, and (in later steps) the
expansion of a model tree into its variant models.
"""

import numpy as np
import pytest

from synthesizer import exceptions
from synthesizer.emission_models.parameters import (
    VARIATION_TYPES,
    ParameterDistribution,
    ParameterList,
    ParameterLogNormalDist,
    ParameterLogUniformDist,
    ParameterNormalDist,
    ParameterUniformDist,
    find_variations,
)
from synthesizer.emission_models.utils import get_param


class MockModel:
    """A minimal stand in for an EmissionModel with fixed parameters."""

    def __init__(self, label="test_model", **fixed_parameters):
        """Store the label and fixed parameters."""
        self.label = label
        self.fixed_parameters = fixed_parameters


class TestParameterList:
    """Test the ParameterList marker."""

    def test_values_and_length(self):
        """Test the values are stored and the length is the value count."""
        plist = ParameterList([0.0, 0.1, 0.5], label_modifier="fesc_%.2f")

        assert plist.values == [0.0, 0.1, 0.5]
        assert len(plist) == 3
        assert list(plist) == [0.0, 0.1, 0.5]

    def test_accepts_arrays(self):
        """Test an array of values is normalised to a list."""
        plist = ParameterList(np.array([0.1, 0.2]), label_modifier="fesc_%.2f")

        assert len(plist) == 2

    def test_suffix_formatting(self):
        """Test the suffix is the formatted value with a leading underscore."""
        plist = ParameterList([0.1], label_modifier="fesc_%.2f")

        assert plist.suffix(0.1) == "_fesc_0.10"

    def test_suffix_strips_units(self):
        """Test a value with units is formatted without the unit string."""
        from unyt import Myr

        plist = ParameterList([10 * Myr], label_modifier="age_%.1f")

        assert plist.suffix(10 * Myr) == "_age_10.0"

    def test_empty_values_raises(self):
        """Test an empty list of values is rejected."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterList([], label_modifier="fesc_%.2f")

    def test_bad_label_modifier_raises(self):
        """Test a label_modifier without a format spec is rejected."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterList([0.1, 0.2], label_modifier="fesc")

        with pytest.raises(exceptions.InconsistentArguments):
            ParameterList([0.1, 0.2], label_modifier=None)

    def test_colliding_labels_raise(self):
        """Test values which format identically are rejected, not deduped."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterList([0.101, 0.102], label_modifier="fesc_%.2f")

    def test_repr(self):
        """Test the repr includes the values and the label modifier."""
        plist = ParameterList([0.1], label_modifier="fesc_%.2f")

        assert "0.1" in repr(plist)
        assert "fesc_%.2f" in repr(plist)


class TestParameterDistribution:
    """Test the ParameterDistribution flavours."""

    def test_base_class_cannot_sample(self):
        """Test the base class refuses to sample."""
        dist = ParameterDistribution(n=3, seed=42, label_modifier="x_%.2f")

        with pytest.raises(exceptions.UnimplementedFunctionality):
            dist.realise()

    def test_bad_n_raises(self):
        """Test a non-positive or non-integer n is rejected."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterUniformDist(0.0, 1.0, n=0, label_modifier="x_%.2f")

        with pytest.raises(exceptions.InconsistentArguments):
            ParameterUniformDist(0.0, 1.0, n=1.5, label_modifier="x_%.2f")

    def test_missing_label_modifier_raises(self):
        """Test the label_modifier is required."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterUniformDist(0.0, 1.0, n=3)

    def test_realise_returns_parameter_list(self):
        """Test realise produces a ParameterList of the right length."""
        dist = ParameterUniformDist(
            0.0, 1.0, n=4, seed=42, label_modifier="fesc_%.3f"
        )
        plist = dist.realise()

        assert isinstance(plist, ParameterList)
        assert len(plist) == 4
        assert plist.label_modifier == "fesc_%.3f"

    def test_realise_is_reproducible(self):
        """Test a fixed seed gives the same values every time."""
        kwargs = {
            "n": 5,
            "seed": 42,
            "label_modifier": "fesc_%.4f",
        }
        first = ParameterUniformDist(0.0, 1.0, **kwargs).realise()
        second = ParameterUniformDist(0.0, 1.0, **kwargs).realise()

        assert first.values == second.values

    def test_different_seeds_differ(self):
        """Test different seeds give different values."""
        first = ParameterUniformDist(
            0.0, 1.0, n=5, seed=1, label_modifier="fesc_%.4f"
        ).realise()
        second = ParameterUniformDist(
            0.0, 1.0, n=5, seed=2, label_modifier="fesc_%.4f"
        ).realise()

        assert first.values != second.values

    def test_uniform_within_bounds(self):
        """Test uniform samples fall within the requested bounds."""
        plist = ParameterUniformDist(
            2.0, 3.0, n=50, seed=42, label_modifier="x_%.4f"
        ).realise()

        assert all(2.0 <= value < 3.0 for value in plist.values)

    def test_uniform_bad_bounds_raise(self):
        """Test low must be below high."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterUniformDist(1.0, 1.0, n=3, label_modifier="x_%.2f")

    def test_log_uniform_within_bounds(self):
        """Test log uniform samples fall within the linear bounds."""
        plist = ParameterLogUniformDist(
            1e-3, 1e-1, n=50, seed=42, label_modifier="x_%.5f"
        ).realise()

        assert all(1e-3 <= value <= 1e-1 for value in plist.values)

    def test_log_uniform_rejects_non_positive_low(self):
        """Test a low of zero is rejected when sampling in log space."""
        dist = ParameterLogUniformDist(
            0.0, 1.0, n=3, seed=42, label_modifier="x_%.2f"
        )

        with pytest.raises(exceptions.InconsistentArguments):
            dist.realise()

    def test_normal_mean(self):
        """Test normal samples scatter around the mean.

        This samples directly rather than through realise so the statistics
        can be checked with a large number of samples without running into
        the unique label requirement.
        """
        dist = ParameterNormalDist(
            5.0, 0.1, n=200, seed=42, label_modifier="x_%.4f"
        )
        values = dist._sample(np.random.default_rng(42), 200)

        assert np.isclose(np.mean(values), 5.0, atol=0.05)

    def test_normal_bad_sigma_raises(self):
        """Test sigma must be positive."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterNormalDist(1.0, 0.0, n=3, label_modifier="x_%.2f")

    def test_log_normal_is_positive(self):
        """Test log normal samples are positive and centred in log space."""
        dist = ParameterLogNormalDist(
            -2.0, 0.1, n=200, seed=42, label_modifier="x_%.5f"
        )
        values = dist._sample(np.random.default_rng(42), 200)

        assert all(value > 0 for value in values)
        assert np.isclose(np.mean(np.log10(values)), -2.0, atol=0.05)

    def test_too_many_samples_for_precision_raises(self):
        """Test realising more samples than the label can distinguish raises.

        Labels are the keys emissions are stored under so they must be
        unique. A coarse label_modifier with many samples cannot satisfy that,
        and the error should say so without dumping every value.
        """
        dist = ParameterUniformDist(
            0.0, 1.0, n=200, seed=42, label_modifier="fesc_%.2f"
        )

        with pytest.raises(
            exceptions.InconsistentArguments, match="more precision"
        ) as excinfo:
            dist.realise()

        # The message must stay readable, not list all 200 values
        assert len(str(excinfo.value)) < 500

    def test_repr(self):
        """Test the reprs include the distribution parameters."""
        dist = ParameterUniformDist(
            0.0, 1.0, n=3, seed=42, label_modifier="x_%.2f"
        )

        assert "low=0.0" in repr(dist)
        assert "n=3" in repr(dist)
        assert "seed=42" in repr(dist)


class TestFindVariations:
    """Test the variation marker discovery helper."""

    def test_finds_markers_only(self):
        """Test only variation markers are returned."""
        model = MockModel(
            fesc=ParameterList([0.1, 0.2], label_modifier="fesc_%.2f"),
            tau_v=ParameterUniformDist(
                0.0, 1.0, n=3, seed=42, label_modifier="tau_v_%.2f"
            ),
            plain=0.5,
            pointer="some_attr",
        )

        variations = find_variations(model)

        assert set(variations) == {"fesc", "tau_v"}
        assert all(
            isinstance(value, VARIATION_TYPES) for value in variations.values()
        )

    def test_no_markers_is_empty(self):
        """Test a model with no variations returns an empty dict."""
        assert find_variations(MockModel(fesc=0.5)) == {}


class TestUnexpandedGuards:
    """Test that unexpanded markers cannot be used as values."""

    @pytest.mark.parametrize(
        "marker",
        [
            ParameterList([0.1, 0.2], label_modifier="fesc_%.2f"),
            ParameterUniformDist(
                0.0, 1.0, n=3, seed=42, label_modifier="fesc_%.2f"
            ),
        ],
    )
    def test_get_param_raises(self, marker):
        """Test get_param refuses to return a marker as a value."""
        model = MockModel(fesc=marker)

        with pytest.raises(
            exceptions.InconsistentArguments, match="expand_models"
        ):
            get_param("fesc", model, None, None)

    def test_to_hdf5_raises(self, test_grid, tmp_path):
        """Test writing a model with a marker to HDF5 is refused."""
        import h5py

        from synthesizer.emission_models import StellarEmissionModel

        model = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
            fesc=ParameterList([0.1, 0.2], label_modifier="fesc_%.2f"),
        )

        with h5py.File(tmp_path / "model.hdf5", "w") as hdf:
            with pytest.raises(
                exceptions.InconsistentArguments, match="expand_models"
            ):
                model.to_hdf5(hdf.create_group("Model"))
