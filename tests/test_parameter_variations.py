"""Tests for the parameter variation machinery.

This covers the ParameterList/ParameterDistribution markers, the guards which
stop an unexpanded marker being used as a value, and (in later steps) the
expansion of a model tree into its variant models.
"""

import numpy as np
import pytest
import unyt
from unyt import dimensionless

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
from synthesizer.emission_models.transformers import EscapingFraction
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

    def test_labels_must_match_sample_count(self):
        """Test explicit labels name every sample."""
        with pytest.raises(exceptions.InconsistentArguments, match="n=3"):
            ParameterUniformDist(0.0, 1.0, n=3, labels=["low", "high"])

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


@pytest.fixture
def suffix_test_model(test_grid):
    r"""Return a small tree with a diamond and string references.

    The tree looks like (arrows point from a model to its dependents)::

        incident --> transmitted --> combined --> scaled
                 \-------------------^

    "combined" depends on "incident" by two routes (directly and through
    "transmitted"), making it a diamond. "scaled" scales by the label of
    "incident", i.e. by a string reference rather than a model object.
    """
    from synthesizer.emission_models import StellarEmissionModel

    incident = StellarEmissionModel(
        label="incident",
        grid=test_grid,
        extract="incident",
    )
    transmitted = StellarEmissionModel(
        label="transmitted",
        apply_to=incident,
        transformer=EscapingFraction(("fesc",)),
        fesc=0.1,
    )
    combined = StellarEmissionModel(
        label="combined",
        combine=(incident, transmitted),
    )
    return StellarEmissionModel(
        label="scaled",
        combine=(combined,),
        scale_by="incident",
    )


class TestDownstreamLabels:
    """Test the dependent walk used to find the models a variation affects."""

    def test_leaf_gives_whole_tree(self, suffix_test_model):
        """Test walking from a leaf reaches every dependent."""
        assert suffix_test_model._get_downstream_labels("incident") == {
            "incident",
            "transmitted",
            "combined",
            "scaled",
        }

    def test_midtree_excludes_dependencies(self, suffix_test_model):
        """Test walking from mid tree does not reach what it depends on."""
        assert suffix_test_model._get_downstream_labels("transmitted") == {
            "transmitted",
            "combined",
            "scaled",
        }

    def test_root_is_only_itself(self, suffix_test_model):
        """Test the root has no dependents."""
        assert suffix_test_model._get_downstream_labels("scaled") == {"scaled"}

    def test_unknown_label_raises(self, suffix_test_model):
        """Test walking from a label that isn't in the tree raises."""
        with pytest.raises(exceptions.InconsistentArguments):
            suffix_test_model._get_downstream_labels("not_a_model")


class TestAddLabelSuffix:
    """Test suffixing every label in a tree."""

    def test_all_labels_suffixed(self, suffix_test_model):
        """Test every model in the tree gains the suffix."""
        suffix_test_model.add_label_suffix("fesc_0.10")

        assert set(suffix_test_model._models) == {
            "incident_fesc_0.10",
            "transmitted_fesc_0.10",
            "combined_fesc_0.10",
            "scaled_fesc_0.10",
        }
        assert suffix_test_model.label == "scaled_fesc_0.10"

    def test_string_scale_by_rewritten(self, suffix_test_model):
        """Test a string scale_by reference follows the relabelling."""
        suffix_test_model.add_label_suffix("fesc_0.10")

        assert suffix_test_model.scale_by == ("incident_fesc_0.10",)

    def test_model_references_preserved(self, suffix_test_model):
        """Test object references still point at the right models."""
        suffix_test_model.add_label_suffix("fesc_0.10")

        combined = suffix_test_model["combined_fesc_0.10"]
        assert {child.label for child in combined.combine} == {
            "incident_fesc_0.10",
            "transmitted_fesc_0.10",
        }
        transmitted = suffix_test_model["transmitted_fesc_0.10"]
        assert transmitted.apply_to.label == "incident_fesc_0.10"

    def test_suffixes_compose(self, suffix_test_model):
        """Test suffixing twice appends both suffixes."""
        suffix_test_model.add_label_suffix("a")
        suffix_test_model.add_label_suffix("b")

        assert suffix_test_model.label == "scaled_a_b"


class TestAddLabelSuffixDownstream:
    """Test suffixing a model and its dependents only."""

    def test_dependencies_keep_labels(self, suffix_test_model):
        """Test only the model and its dependents are relabelled."""
        suffix_test_model.add_label_suffix_downstream(
            "transmitted", "fesc_0.10"
        )

        assert set(suffix_test_model._models) == {
            "incident",
            "transmitted_fesc_0.10",
            "combined_fesc_0.10",
            "scaled_fesc_0.10",
        }

    def test_diamond_relabelled_once(self, suffix_test_model):
        """Test a dependent reached by two routes is relabelled once."""
        suffix_test_model.add_label_suffix_downstream("incident", "fesc_0.10")

        assert set(suffix_test_model._models) == {
            "incident_fesc_0.10",
            "transmitted_fesc_0.10",
            "combined_fesc_0.10",
            "scaled_fesc_0.10",
        }

    def test_string_scale_by_rewritten(self, suffix_test_model):
        """Test a string reference to a relabelled model is rewritten."""
        suffix_test_model.add_label_suffix_downstream("incident", "fesc_0.10")

        assert suffix_test_model.scale_by == ("incident_fesc_0.10",)

    def test_string_scale_by_left_alone(self, suffix_test_model):
        """Test a string reference to an unaffected model is left alone."""
        suffix_test_model.add_label_suffix_downstream(
            "transmitted", "fesc_0.10"
        )

        assert suffix_test_model.scale_by == ("incident",)

    def test_string_apply_to_rewritten(self, test_grid):
        """Test a string apply_to reference is rewritten."""
        from synthesizer.emission_models import StellarEmissionModel

        # A string apply_to means "reuse this emission from the emitter", so
        # build a tree where that string names a model in the tree
        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = StellarEmissionModel(
            label="attenuated",
            combine=(incident,),
        )
        transformed = StellarEmissionModel(
            label="transformed",
            apply_to="incident",
            transformer=EscapingFraction(("fesc",)),
            fesc=0.1,
            related_models=model,
        )

        transformed.add_label_suffix_downstream("transformed", "x")

        assert transformed.apply_to == "incident"

        transformed.add_label_suffix("y")

        assert transformed.apply_to == "incident_y"

    def test_fixed_parameter_strings_left_alone(self, test_grid):
        """Test emitter attribute aliases are not treated as labels."""
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = StellarEmissionModel(
            label="transmitted",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            # A string here points at an attribute on the emitter, and happens
            # to share a name with a model in the tree
            fesc="incident",
        )

        model.add_label_suffix("x")

        assert model.fixed_parameters["fesc"] == "incident"

    def test_collision_raises(self, suffix_test_model):
        """Test relabelling onto an existing label raises."""
        with pytest.raises(exceptions.InconsistentArguments):
            suffix_test_model._relabel_models({"incident": "transmitted"})

    def test_duplicate_new_labels_raise(self, suffix_test_model):
        """Test relabelling two models to the same label raises."""
        with pytest.raises(exceptions.InconsistentArguments):
            suffix_test_model._relabel_models(
                {"incident": "same", "transmitted": "same"}
            )

    def test_unknown_label_raises(self, suffix_test_model):
        """Test relabelling a model that isn't in the tree raises."""
        with pytest.raises(exceptions.InconsistentArguments):
            suffix_test_model._relabel_models({"not_a_model": "x"})


class TestAddLabelPrefix:
    """Test prefixing every label in a tree.

    add_label_prefix predates the suffix methods and used to relabel by
    looping over relabel, which does not rewrite string references. It now
    shares the bulk relabelling path so those references follow the rename.
    """

    def test_all_labels_prefixed(self, suffix_test_model):
        """Test every model in the tree gains the prefix."""
        suffix_test_model.add_label_prefix("stellar")

        assert set(suffix_test_model._models) == {
            "stellar_incident",
            "stellar_transmitted",
            "stellar_combined",
            "stellar_scaled",
        }
        assert suffix_test_model.label == "stellar_scaled"

    def test_string_scale_by_rewritten(self, suffix_test_model):
        """Test a string scale_by reference follows the prefixing.

        This is a regression test. Prefixing used to leave the string
        pointing at the old label, silently detaching it from the tree.
        """
        suffix_test_model.add_label_prefix("stellar")

        assert suffix_test_model.scale_by == ("stellar_incident",)

    def test_model_references_preserved(self, suffix_test_model):
        """Test object references still point at the right models."""
        suffix_test_model.add_label_prefix("stellar")

        combined = suffix_test_model["stellar_combined"]
        assert {child.label for child in combined.combine} == {
            "stellar_incident",
            "stellar_transmitted",
        }
        assert (
            suffix_test_model["stellar_transmitted"].apply_to.label
            == "stellar_incident"
        )

    def test_prefix_and_suffix_compose(self, suffix_test_model):
        """Test a prefix and a suffix can both be applied."""
        suffix_test_model.add_label_prefix("stellar")
        suffix_test_model.add_label_suffix("fesc_0.10")

        assert suffix_test_model.label == "stellar_scaled_fesc_0.10"


class TestCopyNode:
    """Test copying a single model out of a tree."""

    def test_copy_is_detached(self, suffix_test_model):
        """Test a copy has no tree wiring of its own."""
        original = suffix_test_model["combined"]
        copied = original._copy_node()

        assert copied is not original
        assert len(copied._children) == 0
        assert len(copied._parents) == 0
        assert copied._models == {}

    def test_label_can_be_changed(self, suffix_test_model):
        """Test the copy can be relabelled without touching the original."""
        original = suffix_test_model["combined"]
        copied = original._copy_node(label="combined_variant")

        assert copied.label == "combined_variant"
        assert original.label == "combined"

    def test_fixed_parameters_are_independent(self, suffix_test_model):
        """Test changing a copy's fixed parameters leaves the original be."""
        original = suffix_test_model["transmitted"]
        copied = original._copy_node()

        copied.fixed_parameters["fesc"] = 0.9

        assert original.fixed_parameters["fesc"] == 0.1

    def test_masks_are_independent(self, suffix_test_model):
        """Test changing a copy's masks leaves the original alone."""
        original = suffix_test_model["transmitted"]
        original.add_mask("ages", "<", 10 * unyt.Myr)
        copied = original._copy_node()

        copied.masks[0]["thresh"] = 100 * unyt.Myr
        copied.masks.append(dict(copied.masks[0]))

        assert len(original.masks) == 1
        assert original.masks[0]["thresh"] == 10 * unyt.Myr

    def test_combine_is_independent(self, suffix_test_model):
        """Test changing a copy's combine list leaves the original alone."""
        original = suffix_test_model["combined"]
        copied = original._copy_node()

        copied._combine.pop()

        assert len(original.combine) == 2

    def test_related_models_are_independent(self, suffix_test_model):
        """Test changing a copy's related models leaves the original alone."""
        original = suffix_test_model["combined"]
        copied = original._copy_node()

        copied.related_models.add(suffix_test_model["incident"])

        assert len(original.related_models) == 0

    def test_shares_expensive_state(self, suffix_test_model):
        """Test the grid and transformer are shared, not duplicated."""
        original = suffix_test_model["transmitted"]
        copied = original._copy_node()

        assert copied.transformer is original.transformer

        extraction = suffix_test_model["incident"]
        assert extraction._copy_node().grid is extraction.grid


class TestGeneratorRewiring:
    """Test the generator dependency handling shared by copy and replace."""

    def test_generator_model_attrs_found(self, energy_balance_model):
        """Test the generator's model pointers are discovered."""
        assert set(energy_balance_model._generator_model_attrs) == {
            "_intrinsic",
            "_attenuated",
        }

    def test_no_attrs_without_generator(self, suffix_test_model):
        """Test a non generating model reports no generator pointers."""
        assert suffix_test_model._generator_model_attrs == ()

    def test_generator_is_copied(self, energy_balance_model):
        """Test a generator holding model refs is copied, not shared."""
        copied = energy_balance_model._copy_node(label="dust_variant")

        assert copied.generator is not energy_balance_model.generator

    def test_rewiring_copy_leaves_original(
        self, energy_balance_model, test_grid
    ):
        """Test rewiring a copy's generator does not rewire the original's.

        This is the failure mode the generator copy exists to prevent: without
        it both models share one generator and each variant would silently
        overwrite the previous one's dependencies.
        """
        from synthesizer.emission_models import StellarEmissionModel

        original_intrinsic = energy_balance_model.generator._intrinsic
        replacement = StellarEmissionModel(
            label="other_intrinsic",
            grid=test_grid,
            extract="incident",
        )

        copied = energy_balance_model._copy_node(label="dust_variant")
        copied._replace_generator_dependency(original_intrinsic, replacement)

        assert copied.generator._intrinsic is replacement
        assert energy_balance_model.generator._intrinsic is original_intrinsic

    def test_rewiring_preserves_the_other_half(
        self, energy_balance_model, test_grid
    ):
        """Test replacing the intrinsic model keeps the attenuated model."""
        from synthesizer.emission_models import StellarEmissionModel

        attenuated = energy_balance_model.generator._attenuated
        replacement = StellarEmissionModel(
            label="other_intrinsic",
            grid=test_grid,
            extract="incident",
        )

        copied = energy_balance_model._copy_node(label="dust_variant")
        copied._replace_generator_dependency(
            copied.generator._intrinsic, replacement
        )

        assert copied.generator._attenuated is attenuated
        assert copied.generator.is_energy_balance

    def test_replace_model_still_rewires(
        self, energy_balance_model, test_grid
    ):
        """Test replace_model still rewires generator dependencies.

        replace_model now shares the rewiring helper, so this guards against
        the refactor changing its behaviour.
        """
        from synthesizer.emission_models import StellarEmissionModel

        replacement = StellarEmissionModel(
            label="intrinsic",
            grid=test_grid,
            extract="transmitted",
        )

        energy_balance_model.replace_model("intrinsic", replacement)

        assert energy_balance_model.generator._intrinsic is replacement


@pytest.fixture
def varied_model_factory(test_grid):
    """Return a factory building trees with variations attached.

    The tree built is::

        incident --> transmitted --> combined

    with "combined" depending on "incident" both directly and through
    "transmitted", so it is a diamond.
    """
    from synthesizer.emission_models import StellarEmissionModel

    def _build(incident_kwargs=None, transmitted_kwargs=None):
        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
            **(incident_kwargs or {}),
        )
        transmitted = StellarEmissionModel(
            label="transmitted",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            **{"fesc": 0.1, **(transmitted_kwargs or {})},
        )
        return StellarEmissionModel(
            label="combined",
            combine=(incident, transmitted),
        )

    return _build


class TestExpandNoVariations:
    """Test expanding a tree with nothing to expand."""

    def test_returns_equivalent_copy(self, varied_model_factory):
        """Test a tree with no variations expands to a copy of itself."""
        model = varied_model_factory()
        expanded = model.expand_models()

        assert expanded is not model
        assert set(expanded._models) == set(model._models)

    def test_copy_is_independent(self, varied_model_factory):
        """Test the expanded copy shares no mutable state with the original."""
        model = varied_model_factory()
        expanded = model.expand_models()

        expanded["transmitted"].fixed_parameters["fesc"] = 0.9

        assert model["transmitted"].fixed_parameters["fesc"] == 0.1
        assert expanded["incident"] is not model["incident"]


class TestExpandSingleVariation:
    """Test expanding one variation."""

    def test_one_model_per_value(self, varied_model_factory):
        """Test the varied model and its dependents are duplicated."""
        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList(
                    [0.0, 0.1, 0.5], label_modifier="fesc_%.2f"
                )
            }
        )
        expanded = model.expand_models()

        assert set(expanded._models) == {
            # The dependency is shared, not duplicated
            "incident",
            "transmitted_fesc_0.00",
            "transmitted_fesc_0.10",
            "transmitted_fesc_0.50",
            "combined_fesc_0.00",
            "combined_fesc_0.10",
            "combined_fesc_0.50",
        }

    def test_values_are_set(self, varied_model_factory):
        """Test each variant carries its own value, not the declaration."""
        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList([0.0, 0.5], label_modifier="fesc_%.2f")
            }
        )
        expanded = model.expand_models()

        assert (
            expanded["transmitted_fesc_0.00"].fixed_parameters["fesc"] == 0.0
        )
        assert (
            expanded["transmitted_fesc_0.50"].fixed_parameters["fesc"] == 0.5
        )

    def test_variant_params_recorded(self, varied_model_factory):
        """Test each variant records what distinguishes it."""
        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList([0.0, 0.5], label_modifier="fesc_%.2f")
            }
        )
        expanded = model.expand_models()

        assert expanded["combined_fesc_0.50"].variant_params == {"fesc": 0.5}
        # The shared dependency belongs to no variant
        assert expanded["incident"].variant_params == {}

    def test_root_variants_are_related_models(self, varied_model_factory):
        """Test the other root variants are reachable as related models."""
        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList([0.0, 0.5], label_modifier="fesc_%.2f")
            }
        )
        expanded = model.expand_models()

        assert {related.label for related in expanded.related_models} == {
            "combined_fesc_0.50"
        }

    def test_wiring_is_per_variant(self, varied_model_factory):
        """Test each variant's models point at that variant's models."""
        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList([0.0, 0.5], label_modifier="fesc_%.2f")
            }
        )
        expanded = model.expand_models()

        combined = expanded["combined_fesc_0.50"]
        assert {child.label for child in combined.combine} == {
            "incident",
            "transmitted_fesc_0.50",
        }

    def test_diamond_duplicated_once(self, varied_model_factory):
        """Test a dependent reached by two routes is duplicated once.

        "combined" depends on "incident" directly and through "transmitted".
        Recursing each dependency path independently would give one copy of
        "combined" per path per value, i.e. nine for three values instead of
        three.
        """
        model = varied_model_factory(
            incident_kwargs={
                "fesc": ParameterList(
                    [0.0, 0.1, 0.5], label_modifier="fesc_%.2f"
                )
            }
        )
        expanded = model.expand_models()

        combined_labels = [
            label for label in expanded._models if label.startswith("combined")
        ]
        assert len(combined_labels) == 3


class TestExpandMultipleVariations:
    """Test variations multiplying out."""

    def test_independent_variations_multiply(self, varied_model_factory):
        """Test two variations on the same model give every combination."""
        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList(
                    [0.0, 0.1, 0.5], label_modifier="fesc_%.2f"
                ),
                "tau_v": ParameterList(
                    [0.1, 0.2, 0.3, 0.4], label_modifier="tau_v_%.1f"
                ),
            }
        )
        expanded = model.expand_models()

        transmitted = [
            label
            for label in expanded._models
            if label.startswith("transmitted")
        ]
        assert len(transmitted) == 12

    def test_stacked_variations_multiply(self, varied_model_factory):
        """Test a varied model depending on a varied model gives the product.

        "incident" is varied over four values and "transmitted", which depends
        on it, over three. Expanding "incident" first leaves four copies of
        "transmitted", each of which is then expanded into three.
        """
        model = varied_model_factory(
            incident_kwargs={
                "some_param": ParameterList(
                    [1.0, 2.0, 3.0, 4.0], label_modifier="p_%.1f"
                )
            },
            transmitted_kwargs={
                "fesc": ParameterList(
                    [0.0, 0.1, 0.5], label_modifier="fesc_%.2f"
                )
            },
        )
        expanded = model.expand_models()

        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("incident")
                ]
            )
            == 4
        )
        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("transmitted")
                ]
            )
            == 12
        )
        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("combined")
                ]
            )
            == 12
        )

    def test_suffixes_carried_through(self, varied_model_factory):
        """Test a downstream variant's label carries both suffixes."""
        model = varied_model_factory(
            incident_kwargs={
                "some_param": ParameterList([1.0], label_modifier="p_%.1f")
            },
            transmitted_kwargs={
                "fesc": ParameterList([0.5], label_modifier="fesc_%.2f")
            },
        )
        expanded = model.expand_models()

        assert "transmitted_p_1.0_fesc_0.50" in expanded._models

    def test_variant_params_accumulate(self, varied_model_factory):
        """Test a variant records every value which distinguishes it."""
        model = varied_model_factory(
            incident_kwargs={
                "some_param": ParameterList([1.0], label_modifier="p_%.1f")
            },
            transmitted_kwargs={
                "fesc": ParameterList([0.5], label_modifier="fesc_%.2f")
            },
        )
        expanded = model.expand_models()

        assert expanded["transmitted_p_1.0_fesc_0.50"].variant_params == {
            "some_param": 1.0,
            "fesc": 0.5,
        }

    def test_expansion_is_reproducible(self, varied_model_factory):
        """Test expanding the same tree twice gives the same labels.

        The tree is walked through sets of model objects, so this guards
        against label order depending on memory addresses.
        """

        def _labels():
            model = varied_model_factory(
                incident_kwargs={
                    "some_param": ParameterList(
                        [1.0, 2.0], label_modifier="p_%.1f"
                    )
                },
                transmitted_kwargs={
                    "fesc": ParameterList(
                        [0.0, 0.5], label_modifier="fesc_%.2f"
                    ),
                    "tau_v": ParameterList(
                        [0.1, 0.2], label_modifier="tau_v_%.1f"
                    ),
                },
            )
            return sorted(model.expand_models()._models)

        assert _labels() == _labels()


class TestExpandDistributions:
    """Test expanding a sampled variation."""

    def test_sampled_into_models(self, varied_model_factory):
        """Test a distribution produces one model per sample."""
        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterUniformDist(
                    0.0, 1.0, n=4, seed=42, label_modifier="fesc_%.3f"
                )
            }
        )
        expanded = model.expand_models()

        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("transmitted")
                ]
            )
            == 4
        )

    def test_seeded_expansion_is_reproducible(self, varied_model_factory):
        """Test a seeded distribution gives the same models every time."""

        def _labels():
            model = varied_model_factory(
                transmitted_kwargs={
                    "fesc": ParameterUniformDist(
                        0.0, 1.0, n=4, seed=42, label_modifier="fesc_%.3f"
                    )
                }
            )
            return sorted(model.expand_models()._models)

        assert _labels() == _labels()

    def test_samples_shared_across_branches(self, varied_model_factory):
        """Test a distribution is sampled once, not once per branch.

        "incident" is varied over two values, so "transmitted" is duplicated
        before its own distribution is expanded. Both copies must use the same
        samples, otherwise the two branches would not be comparable.
        """
        model = varied_model_factory(
            incident_kwargs={
                "some_param": ParameterList(
                    [1.0, 2.0], label_modifier="p_%.1f"
                )
            },
            transmitted_kwargs={
                "fesc": ParameterUniformDist(
                    0.0, 1.0, n=3, seed=42, label_modifier="fesc_%.3f"
                )
            },
        )
        expanded = model.expand_models()

        first = sorted(
            model_.variant_params["fesc"]
            for label, model_ in expanded.items()
            if label.startswith("transmitted_p_1.0")
        )
        second = sorted(
            model_.variant_params["fesc"]
            for label, model_ in expanded.items()
            if label.startswith("transmitted_p_2.0")
        )

        assert first == second


class TestExpandGeneratorModels:
    """Test expanding a variation feeding a generator dependency."""

    def test_generator_deps_are_per_variant(
        self, test_grid, energy_balance_model
    ):
        """Test each variant's generator points at that variant's models."""
        energy_balance_model["intrinsic"].fix_parameters(
            some_param=ParameterList([1.0, 2.0], label_modifier="p_%.1f")
        )
        expanded = energy_balance_model.expand_models()

        first = expanded["dust_emission_p_1.0"]
        second = expanded["dust_emission_p_2.0"]

        assert first.generator is not second.generator
        assert first.generator._intrinsic.label == "intrinsic_p_1.0"
        assert second.generator._intrinsic.label == "intrinsic_p_2.0"
        assert first.generator._attenuated.label == "attenuated_p_1.0"


class TestExpandRelatedModels:
    """Test expanding a variation in a related model subtree."""

    def test_related_subtree_is_expanded(self, test_grid):
        """Test a varied related model is expanded and stays reachable."""
        from synthesizer.emission_models import StellarEmissionModel

        unrelated = StellarEmissionModel(
            label="unrelated",
            grid=test_grid,
            extract="transmitted",
            fesc=ParameterList([0.1, 0.2], label_modifier="fesc_%.2f"),
        )
        root = StellarEmissionModel(
            label="root",
            grid=test_grid,
            extract="incident",
            related_models=unrelated,
        )

        expanded = root.expand_models()

        # The root the user built is still the root
        assert expanded.label == "root"

        # And both variants of the related model are reachable
        assert set(expanded._models) == {
            "root",
            "unrelated_fesc_0.10",
            "unrelated_fesc_0.20",
        }


class TestExpandRemovesRedundantWork:
    """Test that unvaried models are only computed once.

    This is the reason the machinery exists: building variations by copying a
    whole model tree recomputes the models a variation cannot affect.
    """

    def test_shared_dependency_queued_once(self, varied_model_factory):
        """Test an unvaried dependency appears once in the execution graph."""
        from synthesizer.emission_models.model_queue import ModelQueue

        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList(
                    [0.0, 0.1, 0.5], label_modifier="fesc_%.2f"
                )
            }
        )
        expanded = model.expand_models()

        queue = ModelQueue(expanded)

        # The extraction is shared by all three variants, so it is queued once
        assert (
            len(
                [
                    label
                    for label in queue.models
                    if label.startswith("incident")
                ]
            )
            == 1
        )

        # While everything downstream of the variation is queued per variant
        assert (
            len(
                [
                    label
                    for label in queue.models
                    if label.startswith("combined")
                ]
            )
            == 3
        )

    def test_all_variants_are_queued(self, varied_model_factory):
        """Test every variant is reached through the related models."""
        from synthesizer.emission_models.model_queue import ModelQueue

        model = varied_model_factory(
            transmitted_kwargs={
                "fesc": ParameterList(
                    [0.0, 0.1, 0.5], label_modifier="fesc_%.2f"
                )
            }
        )
        expanded = model.expand_models()

        assert set(ModelQueue(expanded).models) == set(expanded._models)


class TestExpandEndToEnd:
    """Test generating emissions from an expanded model."""

    def test_all_variants_generated(self, test_grid, particle_stars_A):
        """Test one get_spectra call produces every variant's spectra."""
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = StellarEmissionModel(
            label="escaped",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            fesc=ParameterList([0.1, 0.5], label_modifier="fesc_%.2f"),
        )

        expanded = model.expand_models()
        particle_stars_A.get_spectra(expanded)

        # Every variant is attached to the emitter under its own label
        assert "escaped_fesc_0.10" in particle_stars_A.spectra
        assert "escaped_fesc_0.50" in particle_stars_A.spectra

        # The shared dependency was generated once, under its own label
        assert "incident" in particle_stars_A.spectra

    def test_variants_used_their_own_values(self, test_grid, particle_stars_A):
        """Test each variant's emission reflects its own parameter value.

        EscapingFraction scales the emission it is applied to by
        (1 - covering_fraction), so the ratio of each variant's luminosity to
        the shared incident luminosity pins down exactly which value it used.
        """
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = StellarEmissionModel(
            label="escaped",
            apply_to=incident,
            transformer=EscapingFraction(("covering_fraction",)),
            covering_fraction=ParameterList(
                [0.1, 0.5], label_modifier="fcov_%.2f"
            ),
        )

        expanded = model.expand_models()
        particle_stars_A.get_spectra(expanded)

        incident_lum = particle_stars_A.spectra[
            "incident"
        ].bolometric_luminosity
        low_cov = particle_stars_A.spectra[
            "escaped_fcov_0.10"
        ].bolometric_luminosity
        high_cov = particle_stars_A.spectra[
            "escaped_fcov_0.50"
        ].bolometric_luminosity

        assert np.isclose(low_cov / incident_lum, 0.9)
        assert np.isclose(high_cov / incident_lum, 0.5)

    def test_expanded_model_writes_to_hdf5(self, test_grid, tmp_path):
        """Test an expanded model can be written to HDF5.

        The unexpanded model cannot, since a variation is not a value, so this
        checks the expansion really does remove every declaration.
        """
        import h5py

        from synthesizer.emission_models import StellarEmissionModel

        model = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
            fesc=ParameterList([0.1, 0.5], label_modifier="fesc_%.2f"),
        )
        expanded = model.expand_models()

        with h5py.File(tmp_path / "model.hdf5", "w") as hdf:
            for label, variant in expanded.items():
                variant.to_hdf5(hdf.create_group(label))

            assert set(hdf.keys()) == {
                "incident_fesc_0.10",
                "incident_fesc_0.50",
            }


@pytest.fixture
def expanded_model(varied_model_factory):
    """Return an expanded tree with three variants of two models."""
    model = varied_model_factory(
        transmitted_kwargs={
            "fesc": ParameterList([0.0, 0.1, 0.5], label_modifier="fesc_%.2f")
        }
    )
    return model.expand_models()


class TestSelect:
    """Test selecting models by glob pattern."""

    def test_selects_a_variant_family(self, expanded_model):
        """Test a pattern selects every model matching it."""
        selected = expanded_model.select("*_fesc_0.10")

        assert [model.label for model in selected] == [
            "combined_fesc_0.10",
            "transmitted_fesc_0.10",
        ]

    def test_selects_one_model_family(self, expanded_model):
        """Test a prefix pattern selects every variant of one model."""
        selected = expanded_model.select("transmitted_*")

        assert [model.label for model in selected] == [
            "transmitted_fesc_0.00",
            "transmitted_fesc_0.10",
            "transmitted_fesc_0.50",
        ]

    def test_plain_label_selects_itself(self, expanded_model):
        """Test a pattern with no wildcard selects only that model."""
        selected = expanded_model.select("incident")

        assert [model.label for model in selected] == ["incident"]

    def test_multiple_patterns_are_deduplicated(self, expanded_model):
        """Test overlapping patterns don't return a model twice."""
        selected = expanded_model.select("transmitted_*", "*_fesc_0.10")

        labels = [model.label for model in selected]
        assert len(labels) == len(set(labels))
        assert "transmitted_fesc_0.10" in labels

    def test_ordering_is_reproducible(self, expanded_model):
        """Test the selection is ordered by label, not by tree traversal."""
        selected = expanded_model.select("*")

        labels = [model.label for model in selected]
        assert labels == sorted(labels)

    def test_no_match_is_empty(self, expanded_model):
        """Test a query which matches nothing returns nothing, not an error."""
        assert expanded_model.select("*_fesc_0.99") == []

    def test_returns_the_models_themselves(self, expanded_model):
        """Test the selection returns live models, not copies."""
        for model in expanded_model.select("transmitted_*"):
            model.set_save(False)

        assert expanded_model["transmitted_fesc_0.10"].save is False
        assert expanded_model["combined_fesc_0.10"].save is True


class TestSettingsAreInherited:
    """Test a variant inherits the settings of the model it came from."""

    def test_save_flags_are_inherited(self, test_grid):
        """Test switching off saving before expanding survives expansion."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            dust_curve=PowerLaw(slope=-1),
            tau_v=ParameterList([0.1, 0.5], label_modifier="tauv%.1f"),
        )
        total = StellarEmissionModel(
            label="total",
            combine=(attenuated, incident),
        )

        # Only the total is wanted, said before the expansion
        total.save_spectra("total")

        expanded = total.expand_models()

        # Each variant carries the flag of the model it was copied from
        assert sorted(expanded.saved_labels) == [
            "total_tauv0.1",
            "total_tauv0.5",
        ]

    def test_per_particle_is_inherited(self, test_grid):
        """Test another setting also carries into every variant."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            dust_curve=PowerLaw(slope=-1),
            tau_v=ParameterList([0.1, 0.5], label_modifier="tauv%.1f"),
        )
        attenuated.set_per_particle(True)

        expanded = attenuated.expand_models()

        for model in expanded.select("attenuated*"):
            assert model.per_particle is True


class TestSaveEmissionGlobs:
    """Test the save flag helpers accepting glob patterns."""

    def test_pattern_saves_a_variant_family(self, expanded_model):
        """Test a pattern saves every model matching it and nothing else."""
        expanded_model.save_spectra("*_fesc_0.10")

        assert sorted(expanded_model.saved_labels) == [
            "combined_fesc_0.10",
            "transmitted_fesc_0.10",
        ]

    def test_plain_labels_still_work(self, expanded_model):
        """Test passing exact labels behaves as it always did."""
        expanded_model.save_spectra("incident", "combined_fesc_0.50")

        assert sorted(expanded_model.saved_labels) == [
            "combined_fesc_0.50",
            "incident",
        ]

    def test_unmatched_pattern_raises(self, expanded_model):
        """Test an unmatched pattern raises rather than saving nothing.

        Without this the save flags would all be switched off and the caller
        would be left with an emission model that produces nothing.
        """
        with pytest.raises(exceptions.InconsistentArguments):
            expanded_model.save_spectra("*_fesc_0.99")

    def test_save_flags_untouched_when_raising(self, expanded_model):
        """Test a failed call leaves the save flags as they were."""
        before = sorted(expanded_model.saved_labels)

        with pytest.raises(exceptions.InconsistentArguments):
            expanded_model.save_spectra("*_fesc_0.99")

        assert sorted(expanded_model.saved_labels) == before

    def test_save_lines_accepts_patterns(self, expanded_model):
        """Test the lines alias also accepts patterns."""
        expanded_model.save_lines("transmitted_*")

        assert sorted(expanded_model.saved_labels) == [
            "transmitted_fesc_0.00",
            "transmitted_fesc_0.10",
            "transmitted_fesc_0.50",
        ]


class TestSharedDeclarations:
    """Test one declaration handed to several models is a single variation.

    A premade model passes a parameter like the escape fraction down to every
    one of its models which needs it. Those are one variation, not one per
    model, so they have to vary in step rather than multiplying out into a
    combination for every model that happened to receive it.
    """

    def test_a_shared_list_varies_in_step(self, test_grid):
        """Test models given the same list are varied together."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import EscapingFraction

        shared = ParameterList([0.1, 0.5], label_modifier="fesc_%.2f")

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        first = StellarEmissionModel(
            label="first",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            fesc=shared,
        )
        second = StellarEmissionModel(
            label="second",
            apply_to=first,
            transformer=EscapingFraction(("fesc",)),
            fesc=shared,
        )

        expanded = second.expand_models()

        # Two variants, not the four a pair of independent lists would give
        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("second")
                ]
            )
            == 2
        )

    def test_separate_lists_still_multiply(self, test_grid):
        """Test equal but separate declarations remain independent.

        The grouping is by declaration, not by value, so two lists which happen
        to hold the same values are still two variations.
        """
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import EscapingFraction

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        first = StellarEmissionModel(
            label="first",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            fesc=ParameterList([0.1, 0.5], label_modifier="a_%.2f"),
        )
        second = StellarEmissionModel(
            label="second",
            apply_to=first,
            transformer=EscapingFraction(("fesc",)),
            fesc=ParameterList([0.1, 0.5], label_modifier="b_%.2f"),
        )

        expanded = second.expand_models()

        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("second")
                ]
            )
            == 4
        )

    def test_a_shared_distribution_is_sampled_once(self, test_grid):
        """Test models given one distribution get the same sampled values."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import EscapingFraction

        shared = ParameterUniformDist(
            0.0, 1.0, n=3, seed=42, label_modifier="fesc_%.3f"
        )

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        first = StellarEmissionModel(
            label="first",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            fesc=shared,
        )
        second = StellarEmissionModel(
            label="second",
            apply_to=first,
            transformer=EscapingFraction(("fesc",)),
            fesc=shared,
        )

        expanded = second.expand_models()

        # Sampling per model would give six variants with mismatched values
        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("second")
                ]
            )
            == 3
        )

        # And each variant used one value for both models
        for label, model in expanded.items():
            if not label.startswith("second"):
                continue
            partner = expanded[label.replace("second", "first")]
            assert (
                model.fixed_parameters["fesc"]
                == partner.fixed_parameters["fesc"]
            )


class TestPremadeModelVariations:
    """Test the machinery on a premade model, as a user would meet it."""

    def test_a_premade_model_expands(self, test_grid):
        """Test varying one parameter of a premade model."""
        from synthesizer.emission_models import PacmanEmission

        model = PacmanEmission(
            test_grid,
            fesc=ParameterList([0.0, 0.3], label_modifier="fesc_%.1f"),
        )
        expanded = model.expand_models()

        assert (
            len(
                [
                    label
                    for label in expanded._models
                    if label.startswith("emergent")
                ]
            )
            == 2
        )

    def test_two_parameters_give_every_combination(self, test_grid):
        """Test two parameters of a premade model multiply out.

        This is the case which caught a stale dependency left behind by an
        earlier expansion: the models a replaced model depended on still point
        back at it, and following those pointers collected models which were no
        longer in the tree at all.
        """
        from synthesizer.emission_models import PacmanEmission

        model = PacmanEmission(
            test_grid,
            tau_v=ParameterList([0.1, 0.5, 1.0], label_modifier="tauv_%.1f"),
            fesc=ParameterList([0.0, 0.3], label_modifier="fesc_%.1f"),
        )
        expanded = model.expand_models()

        emergent = [
            label for label in expanded._models if label.startswith("emergent")
        ]
        assert len(emergent) == 6

        # Every combination appears exactly once
        assert len(set(emergent)) == 6

    def test_the_shared_models_are_not_duplicated(self, test_grid):
        """Test the parts a variation cannot reach are computed once.

        This is the whole point: the grid extractions do not depend on the
        escape fraction or the optical depth, so there is no reason to build
        them again for every combination.
        """
        from synthesizer.emission_models import PacmanEmission
        from synthesizer.emission_models.model_queue import ModelQueue

        model = PacmanEmission(
            test_grid,
            tau_v=ParameterList([0.1, 0.5, 1.0], label_modifier="tauv_%.1f"),
            fesc=ParameterList([0.0, 0.3], label_modifier="fesc_%.1f"),
        )
        expanded = model.expand_models()
        queue = ModelQueue(expanded)

        assert (
            len(
                [
                    label
                    for label in queue.models
                    if label.startswith("incident")
                ]
            )
            == 1
        )

        # And far fewer models than duplicating the whole tree per combination
        assert len(expanded._models) < 6 * len(model._models)


class TestExpansionOrderDoesNotMatter:
    """Test the result is the same however the expansion order falls out.

    Copying a cone leaves the declarations further up it shared between the
    copies, so the grouping sees them as one variation. That is fine, and is
    what makes the expansion robust to the order the variations happen to be
    expanded in: those copies sit in branches which cannot reach each other, so
    varying them in step still produces every combination. The invariant is
    load bearing and worth stating, because the bookkeeping which decides the
    order cannot always tell which declaration a model is waiting on.
    """

    def test_a_model_with_two_declarations(self, test_grid):
        """Test a model holding two declarations expands to the product."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import EscapingFraction

        # The shared declaration sits on both models, the other on one of them,
        # which makes the order the two are expanded in ambiguous
        shared = ParameterList([0.1, 0.2], label_modifier="s%.1f")
        other = ParameterList([1.0, 2.0, 3.0], label_modifier="o%.1f")

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        first = StellarEmissionModel(
            label="first",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            fesc=shared,
            other=other,
        )
        second = StellarEmissionModel(
            label="second",
            apply_to=first,
            transformer=EscapingFraction(("fesc",)),
            fesc=shared,
        )

        expanded = second.expand_models()

        # Every combination of the two declarations, once each
        variants = expanded.select("second*")
        assert len(variants) == 6

        seen = set()
        for model in variants:
            # The shared declaration still varies the two models in step
            assert (
                model.fixed_parameters["fesc"]
                == model.apply_to.fixed_parameters["fesc"]
            )
            seen.add(
                (
                    model.fixed_parameters["fesc"],
                    model.apply_to.fixed_parameters["other"],
                )
            )

        assert seen == {
            (fesc, other_value)
            for fesc in (0.1, 0.2)
            for other_value in (1.0, 2.0, 3.0)
        }

    def test_the_root_keeps_its_place(self, test_grid):
        """Test the model the user called is still the root of the result.

        The root is replaced by a copy of itself whenever it falls inside a
        cone, which happens on nearly every expansion, so it is worth pinning.
        """
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import EscapingFraction

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        root = StellarEmissionModel(
            label="root",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            fesc=ParameterList([0.1, 0.2], label_modifier="f%.1f"),
            tau=ParameterList([1.0, 2.0], label_modifier="t%.1f"),
        )

        expanded = root.expand_models()

        # The root carries one of the variant labels, and the others are all
        # reachable from it
        assert expanded.variant_base == "root"
        assert len(expanded.select("root*")) == 4


class TestExplicitLabels:
    """Test naming the variants directly rather than formatting their values.

    A format string is the natural way to name numbers, but it cannot render an
    object: a list of dust curves formatted with %s gives labels like
    "PowerLaw(slope=-1.0)", and those labels are the keys emissions are stored
    under and the names of groups in an HDF5 file.
    """

    def test_labels_name_the_variants(self):
        """Test the suffixes come from the labels given."""
        plist = ParameterList([0.1, 0.5], labels=["low", "high"])

        assert plist.suffixes == ["_low", "_high"]
        assert list(plist.items()) == [(0.1, "_low"), (0.5, "_high")]

    def test_a_format_string_still_works(self):
        """Test the suffixes still come from a format string when given one."""
        plist = ParameterList([0.1, 0.5], label_modifier="fesc_%.2f")

        assert plist.suffixes == ["_fesc_0.10", "_fesc_0.50"]

    def test_exactly_one_naming_is_required(self):
        """Test neither or both of the naming arguments is rejected."""
        with pytest.raises(exceptions.InconsistentArguments, match="exactly"):
            ParameterList([0.1, 0.5])

        with pytest.raises(exceptions.InconsistentArguments, match="exactly"):
            ParameterList(
                [0.1, 0.5],
                label_modifier="f_%.1f",
                labels=["low", "high"],
            )

    def test_a_label_per_value_is_required(self):
        """Test a mismatched number of labels is rejected."""
        with pytest.raises(
            exceptions.InconsistentArguments, match="one label"
        ):
            ParameterList([0.1, 0.5, 0.9], labels=["low", "high"])

    def test_labels_must_be_usable_names(self):
        """Test labels have to be non-empty strings."""
        with pytest.raises(exceptions.InconsistentArguments):
            ParameterList([0.1, 0.5], labels=["low", ""])

        with pytest.raises(exceptions.InconsistentArguments):
            ParameterList([0.1, 0.5], labels=["low", 7])

    def test_duplicate_labels_are_rejected(self):
        """Test two values cannot end up with the same label."""
        with pytest.raises(exceptions.InconsistentArguments, match="distinct"):
            ParameterList([0.1, 0.5], labels=["same", "same"])

    def test_unrenderable_values_say_to_use_labels(self):
        """Test a format string which cannot render the values explains itself.

        Formatting an object with a numeric format string fails, and the error
        should point at the argument which does work rather than at printf.
        """
        with pytest.raises(exceptions.InconsistentArguments, match="labels"):
            ParameterList([object(), object()], label_modifier="thing_%.2f")

    def test_a_distribution_can_be_named(self):
        """Test a sampled variation can use explicit labels too."""
        plist = ParameterUniformDist(
            0.0,
            1.0,
            n=3,
            seed=42,
            labels=["low", "middle", "high"],
        ).realise()

        assert plist.suffixes == ["_low", "_middle", "_high"]
        assert len(plist.values) == 3

    def test_a_distribution_needs_exactly_one_naming(self):
        """Test the same rule applies to a distribution."""
        with pytest.raises(exceptions.InconsistentArguments, match="exactly"):
            ParameterUniformDist(0.0, 1.0, n=3)

    def test_labelled_functions_get_readable_labels(self, test_grid):
        """Test a list of ParameterFunctions produces usable labels.

        Without explicit labels these come out as the repr of the function
        wrapper, which is unusable as an emission key.
        """
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.parameters import ParameterFunction
        from synthesizer.emission_models.transformers import EscapingFraction

        def low(ages):
            """Return a low escape fraction."""
            return 0.1 + 0.0 * ages

        def high(ages):
            """Return a high escape fraction."""
            return 0.5 + 0.0 * ages

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = StellarEmissionModel(
            label="escaped",
            apply_to=incident,
            transformer=EscapingFraction(("fesc",)),
            fesc=ParameterList(
                [
                    ParameterFunction(low, "fesc", ["ages"]),
                    ParameterFunction(high, "fesc", ["ages"]),
                ],
                labels=["low_fesc", "high_fesc"],
            ),
        )

        expanded = model.expand_models()

        assert sorted(x.label for x in expanded.select("escaped*")) == [
            "escaped_high_fesc",
            "escaped_low_fesc",
        ]

        # And each variant holds one of the functions, ready to be called
        for model in expanded.select("escaped*"):
            assert isinstance(
                model.fixed_parameters["fesc"], ParameterFunction
            )


class TestVaryingTransformersAndGenerators:
    """Test varying the objects a model uses, not only its parameters.

    A transformer and a generator are held on the model itself rather than
    among its fixed parameters, so they need finding and setting separately,
    but varying one is as reasonable as varying a number.
    """

    def test_a_transformer_can_be_varied(self, test_grid):
        """Test a model can be varied over several dust curves."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import (
            Calzetti2000,
            PowerLaw,
        )

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.3,
            dust_curve=ParameterList(
                [PowerLaw(slope=-1), Calzetti2000()],
                labels=["powerlaw", "calzetti"],
            ),
        )

        expanded = model.expand_models()

        curves = {
            variant.label: type(variant.transformer).__name__
            for variant in expanded.select("attenuated*")
        }
        assert curves == {
            "attenuated_powerlaw": "PowerLaw",
            "attenuated_calzetti": "Calzetti2000",
        }

        # The extraction underneath is still shared
        assert len(expanded.select("incident*")) == 1

    def test_a_generator_can_be_varied(self, test_grid):
        """Test a model can be varied over several dust emission models."""
        from unyt import K

        from synthesizer.emission_models import (
            AttenuatedEmission,
            DustEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw
        from synthesizer.emission_models.generators.dust.blackbody import (
            Blackbody,
        )
        from synthesizer.emission_models.generators.dust.greybody import (
            Greybody,
        )

        intrinsic = StellarEmissionModel(
            label="intrinsic",
            grid=test_grid,
            extract="incident",
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            apply_to=intrinsic,
            emitter="stellar",
            tau_v=0.3,
            dust_curve=PowerLaw(slope=-1),
        )
        model = DustEmission(
            dust_emission_model=ParameterList(
                [
                    Blackbody(temperature=30 * K),
                    Greybody(temperature=60 * K, emissivity=1.5),
                ],
                labels=["blackbody", "greybody"],
            ),
            emitter="stellar",
            label="dust_emission",
            dust_lum_intrinsic=intrinsic,
            dust_lum_attenuated=attenuated,
        )

        expanded = model.expand_models()

        generators = {
            variant.label: type(variant.generator).__name__
            for variant in expanded.select("dust_emission*")
        }
        assert generators == {
            "dust_emission_blackbody": "Blackbody",
            "dust_emission_greybody": "Greybody",
        }

        # Every candidate generator got the energy balance wiring, not just the
        # first one
        for variant in expanded.select("dust_emission*"):
            assert variant.generator.is_energy_balance

    def test_generating_before_expanding_says_so(self, test_grid):
        """Test an unexpanded declaration is reported before any work is done.

        A declaration on the transformer would otherwise fail deep inside the
        calculation, where the message is about a missing method rather than
        about the model needing expanding.
        """
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import (
            Calzetti2000,
            PowerLaw,
        )

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.3,
            dust_curve=ParameterList(
                [PowerLaw(slope=-1), Calzetti2000()],
                labels=["powerlaw", "calzetti"],
            ),
        )

        with pytest.raises(
            exceptions.InconsistentArguments, match="expand_models"
        ) as excinfo:
            model._check_expanded()

        # And it names the model and what is waiting on it
        assert "attenuated" in str(excinfo.value)
        assert "transformer" in str(excinfo.value)

    def test_an_expanded_model_passes_the_check(self, test_grid):
        """Test the check is happy once the declarations are gone."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        model = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.3,
            dust_curve=ParameterList(
                [PowerLaw(slope=-1), PowerLaw(slope=-0.7)],
                labels=["steep", "shallow"],
            ),
        )

        # No exception
        model.expand_models()._check_expanded()


class TestLabelClashErrors:
    """Test the error raised when two models share a label.

    Mixing an expanded tree with the models it was built from puts two
    different models with the same label in one tree. That has to say so
    clearly, which means summarising a model which has never been unpacked.
    """

    def test_a_clash_between_trees_is_reported(self, test_grid):
        """Test stitching an expanded tree onto its originals raises."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        expanded = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=ParameterList([0.1, 0.3], label_modifier="tau_v_%.1f"),
            dust_curve=PowerLaw(slope=-1),
        ).expand_models()

        # The expansion works on a copy, so this combines two models labelled
        # "incident": the original and the copy inside the expanded tree
        with pytest.raises(
            exceptions.InconsistentArguments, match="already in use"
        ):
            StellarEmissionModel(
                label="total",
                combine=(incident, expanded),
            )

    def test_a_model_which_was_never_unpacked_can_be_summarised(
        self, test_grid
    ):
        """Test a detached model summarises itself rather than raising."""
        from synthesizer.emission_models import StellarEmissionModel

        model = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )

        # A copied node carries no tree until it is unpacked
        summary = str(model._copy_node())

        assert "incident" in summary.lower()


class TestVaryingTransformerAndGeneratorArguments:
    """Test declaring a variation on an argument of a transformer/generator.

    The arguments of a dust curve or a dust emission model belong to the
    curve rather than to the model, so a declaration passed to one of them is
    found by looking inside the object. Each variant gets its own copy of the
    object with the value substituted, so the copies cannot affect each other
    or the object the user built.
    """

    def _attenuated(self, test_grid, curve, **kwargs):
        """Return an attenuated model using a given dust curve."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        return AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=kwargs.pop("tau_v", 0.5),
            dust_curve=curve,
            **kwargs,
        )

    def test_a_transformer_argument_expands(self, test_grid):
        """Test a declaration on a dust curve argument makes models."""
        from synthesizer.emission_models.attenuation import PowerLaw

        curve = PowerLaw(
            slope=ParameterList([-1.0, -0.5], label_modifier="slope%.1f")
        )
        expanded = self._attenuated(test_grid, curve).expand_models()

        variants = sorted(
            expanded.select("attenuated*"), key=lambda m: m.label
        )

        assert [model.label for model in variants] == [
            "attenuated_slope-0.5",
            "attenuated_slope-1.0",
        ]

        # Each variant has its own curve holding its own value
        assert [model.transformer.slope for model in variants] == [-0.5, -1.0]

        # And the curve the user built is untouched, still holding the
        # declaration
        assert isinstance(curve.slope, ParameterList)
        for model in variants:
            assert model.transformer is not curve

    def test_the_extraction_is_shared(self, test_grid):
        """Test only the varied model and its cone are duplicated."""
        from synthesizer.emission_models.attenuation import PowerLaw

        curve = PowerLaw(
            slope=ParameterList([-1.0, -0.5], label_modifier="slope%.1f")
        )
        expanded = self._attenuated(test_grid, curve).expand_models()

        assert len(expanded.select("incident")) == 1

    def test_the_variant_spectra_differ(self, test_grid, particle_stars_A):
        """Test the substituted value really reaches the calculation."""
        from synthesizer.emission_models.attenuation import PowerLaw

        curve = PowerLaw(
            slope=ParameterList([-1.0, -0.5], label_modifier="slope%.1f")
        )
        expanded = self._attenuated(test_grid, curve).expand_models()

        particle_stars_A.get_spectra(expanded)

        steep = particle_stars_A.spectra[
            "attenuated_slope-1.0"
        ].bolometric_luminosity
        shallow = particle_stars_A.spectra[
            "attenuated_slope-0.5"
        ].bolometric_luminosity

        assert not np.isclose(steep, shallow)

    def test_a_generator_argument_expands(self, test_grid):
        """Test a declaration on a dust emission model argument."""
        from unyt import kelvin

        from synthesizer.emission_models import (
            AttenuatedEmission,
            DustEmission,
            Greybody,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.5,
            dust_curve=PowerLaw(slope=-1),
        )
        expanded = DustEmission(
            dust_emission_model=Greybody(
                temperature=ParameterList(
                    [20 * kelvin, 60 * kelvin],
                    label_modifier="T%dK",
                ),
                emissivity=1.5,
            ),
            emitter="stellar",
            label="dust_emission",
            dust_lum_intrinsic=incident,
            dust_lum_attenuated=attenuated,
        ).expand_models()

        variants = sorted(
            expanded.select("dust_emission*"), key=lambda m: m.label
        )

        assert [model.label for model in variants] == [
            "dust_emission_T20K",
            "dust_emission_T60K",
        ]
        assert [model.generator.temperature for model in variants] == [
            20 * kelvin,
            60 * kelvin,
        ]

        # The energy balance wiring survived the substitution
        for model in variants:
            assert model.generator.is_energy_balance

    def test_a_unit_carrying_argument_expands(self, test_grid):
        """Test a declaration passes the unit checks on a constructor.

        A constructor argument which must carry units is checked when the
        object is built, which a declaration cannot satisfy since it holds
        the values rather than being one.
        """
        from unyt import angstrom

        from synthesizer.emission_models.attenuation import Calzetti2000

        curve = Calzetti2000(
            cent_lam=ParameterList(
                [2000 * angstrom, 2175 * angstrom],
                labels=["lam2000", "lam2175"],
            )
        )
        expanded = self._attenuated(test_grid, curve).expand_models()

        variants = sorted(
            expanded.select("attenuated*"), key=lambda m: m.label
        )

        assert [model.label for model in variants] == [
            "attenuated_lam2000",
            "attenuated_lam2175",
        ]
        assert [model.transformer.cent_lam for model in variants] == [
            2000 * angstrom,
            2175 * angstrom,
        ]

    def test_two_arguments_multiply(self, test_grid):
        """Test two declarations on one object are independent axes."""
        from synthesizer.emission_models.attenuation import Calzetti2000

        curve = Calzetti2000(
            slope=ParameterList([0.0, -0.2], label_modifier="slope%.1f"),
            ampl=ParameterList([0.0, 1.0], label_modifier="ampl%.1f"),
        )
        expanded = self._attenuated(test_grid, curve).expand_models()

        assert len(expanded.select("attenuated*")) == 4

    def test_an_argument_and_a_model_parameter_multiply(self, test_grid):
        """Test a curve argument and a model parameter are independent."""
        from synthesizer.emission_models.attenuation import PowerLaw

        curve = PowerLaw(
            slope=ParameterList([-1.0, -0.5], label_modifier="slope%.1f")
        )
        expanded = self._attenuated(
            test_grid,
            curve,
            tau_v=ParameterList([0.1, 1.0], label_modifier="tauv%.1f"),
        ).expand_models()

        assert sorted(
            model.label for model in expanded.select("attenuated*")
        ) == [
            "attenuated_tauv0.1_slope-0.5",
            "attenuated_tauv0.1_slope-1.0",
            "attenuated_tauv1.0_slope-0.5",
            "attenuated_tauv1.0_slope-1.0",
        ]

    def test_a_shared_declaration_couples_two_curves(self, test_grid):
        """Test one declaration in two curves varies them in step."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw

        shared = ParameterList([-1.0, -0.5], label_modifier="slope%.1f")

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        first = AttenuatedEmission(
            label="att1",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.5,
            dust_curve=PowerLaw(slope=shared),
        )
        second = AttenuatedEmission(
            label="att2",
            apply_to=first,
            emitter="stellar",
            tau_v=0.5,
            dust_curve=PowerLaw(slope=shared),
        )
        expanded = StellarEmissionModel(
            label="total",
            combine=(first, second),
        ).expand_models()

        # Two variants, not four, and the two curves in each agree
        assert len(expanded.select("total*")) == 2
        for suffix in ("slope-1.0", "slope-0.5"):
            assert (
                expanded[f"att1_{suffix}"].transformer.slope
                == expanded[f"att2_{suffix}"].transformer.slope
            )

    def test_a_distribution_inside_a_transformer(self, test_grid):
        """Test a distribution on a curve argument is realised and expanded."""
        from synthesizer.emission_models.attenuation import PowerLaw
        from synthesizer.emission_models.parameters import ParameterNormalDist

        curve = PowerLaw(
            slope=ParameterNormalDist(
                -1.0,
                0.2,
                3,
                seed=42,
                label_modifier="slope%.2f",
            )
        )
        expanded = self._attenuated(test_grid, curve).expand_models()

        variants = expanded.select("attenuated*")

        assert len(variants) == 3
        assert len({model.transformer.slope for model in variants}) == 3

    def test_variant_params_use_the_bare_name(self, test_grid):
        """Test the recorded parameter is named as the user named it."""
        from synthesizer.emission_models.attenuation import PowerLaw

        curve = PowerLaw(
            slope=ParameterList([-1.0, -0.5], label_modifier="slope%.1f")
        )
        expanded = self._attenuated(test_grid, curve).expand_models()

        for model in expanded.select("attenuated*"):
            assert set(model.variant_params) == {"slope"}
            assert model.variant_params["slope"] == model.transformer.slope

    def test_the_guard_names_the_argument(self, test_grid):
        """Test generating from an unexpanded model says what is waiting."""
        from synthesizer.emission_models.attenuation import PowerLaw

        curve = PowerLaw(
            slope=ParameterList([-1.0, -0.5], label_modifier="slope%.1f")
        )
        model = self._attenuated(test_grid, curve)

        with pytest.raises(
            exceptions.InconsistentArguments, match="transformer.slope"
        ):
            model._check_expanded()


class TestDeclarationUnitChecks:
    """Test unit checked arguments check the values a declaration holds.

    A constructor argument which must carry units is checked when the object
    is built, which a declaration cannot satisfy by itself since it holds the
    values rather than being one. The check is applied to what it holds, so
    declaring a variation does not buy a way past the unit checks.
    """

    def _curve(self, cent_lam):
        """Return a Calzetti curve with a given central wavelength."""
        from synthesizer.emission_models.attenuation import Calzetti2000

        return Calzetti2000(cent_lam=cent_lam)

    def test_values_are_converted(self, test_grid):
        """Test values given in other units end up in the expected ones."""
        from unyt import angstrom, um

        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )

        curve = self._curve(
            ParameterList(
                [2000 * angstrom, 0.2175 * um],
                labels=["lam2000", "lam2175"],
            )
        )
        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        expanded = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.5,
            dust_curve=curve,
        ).expand_models()

        variants = sorted(
            expanded.select("attenuated*"), key=lambda m: m.label
        )

        assert [model.transformer.cent_lam for model in variants] == [
            2000 * angstrom,
            2175 * angstrom,
        ]

        # The labels still name what the user wrote, not what it converted to
        assert [model.label for model in variants] == [
            "attenuated_lam2000",
            "attenuated_lam2175",
        ]

    def test_a_value_without_units_is_rejected(self):
        """Test a bare number in a list is caught, not carried through."""
        from unyt import angstrom

        with pytest.raises(exceptions.MissingUnits, match="cent_lam"):
            self._curve(
                ParameterList([2000 * angstrom, 2175], labels=["a", "b"])
            )

    def test_incompatible_units_are_rejected(self):
        """Test values in the wrong units are caught."""
        from unyt import K

        with pytest.raises(exceptions.IncorrectUnits, match="cent_lam"):
            self._curve(ParameterList([2000 * K, 2175 * K], labels=["a", "b"]))

    def test_a_distribution_is_checked_and_converted(self):
        """Test a distribution's bounds are checked, and its samples carry."""
        from unyt import angstrom, um

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        curve = self._curve(
            ParameterUniformDist(
                0.2 * um,
                0.25 * um,
                3,
                seed=1,
                label_modifier="lam%.0f",
            )
        )

        # The bounds were converted when the curve was built
        assert curve.cent_lam.low == 2000 * angstrom
        assert curve.cent_lam.high == 2500 * angstrom

        # And the samples land between them, carrying the units
        for value in curve.cent_lam.realise().values:
            assert value.units == angstrom.units
            assert 2000 * angstrom <= value <= 2500 * angstrom

    def test_a_distribution_without_units_is_rejected(self):
        """Test a distribution described by bare numbers is caught."""
        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        with pytest.raises(exceptions.MissingUnits, match="cent_lam"):
            self._curve(
                ParameterUniformDist(2000, 2500, 3, label_modifier="lam%.0f")
            )

    def test_a_log_space_distribution_needs_its_units_stating(self):
        """Test a distribution in log10 space is caught without units.

        Its mean and sigma are in log10 space, so they cannot say what units
        the samples carry. Nothing is guessed: the units have to be stated.
        """
        from synthesizer.emission_models.parameters import (
            ParameterLogNormalDist,
        )

        with pytest.raises(exceptions.MissingUnits, match="cent_lam"):
            self._curve(
                ParameterLogNormalDist(3.3, 0.1, 3, label_modifier="lam%.0f")
            )

    def test_a_log_space_distribution_works_with_units(self):
        """Test stating the units is all a log space distribution needs."""
        from unyt import angstrom

        from synthesizer.emission_models.parameters import (
            ParameterLogNormalDist,
        )

        curve = self._curve(
            ParameterLogNormalDist(
                3.3,
                0.05,
                3,
                seed=1,
                label_modifier="lam%.0f",
                units=angstrom,
            )
        )

        for value in curve.cent_lam.realise().values:
            assert value.units == angstrom.units
            assert 1000 * angstrom < value < 4000 * angstrom

    def test_magnitudes_with_units_given_are_converted(self):
        """Test magnitudes with units given are checked and converted."""
        from unyt import angstrom, um

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        curve = self._curve(
            ParameterUniformDist(
                0.2,
                0.25,
                3,
                seed=1,
                label_modifier="lam%.0f",
                units=um,
            )
        )

        # The check converted the units, and the bounds followed
        assert curve.cent_lam.units == angstrom.units
        assert curve.cent_lam.low == 2000.0
        assert curve.cent_lam.high == 2500.0

    def test_units_given_take_the_quantities_with_them(self):
        """Test stated units win, with any quantities converted into them."""
        from unyt import angstrom, um

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        distribution = ParameterUniformDist(
            0.2 * um,
            0.25 * um,
            3,
            seed=1,
            label_modifier="lam%.0f",
            units=angstrom,
        )

        assert distribution.units == angstrom.units
        assert distribution.low == 2000.0
        assert distribution.high == 2500.0

    def test_units_disagreeing_with_the_values_are_rejected(self):
        """Test units which cannot describe the values are caught."""
        from unyt import K, angstrom

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        with pytest.raises(
            exceptions.InconsistentArguments, match="does not agree"
        ):
            ParameterUniformDist(
                2000 * angstrom,
                2500 * angstrom,
                3,
                label_modifier="lam%.0f",
                units=K,
            )

    def test_mixed_units_in_a_distribution_are_reconciled(self):
        """Test a distribution can be described in compatible units."""
        from unyt import angstrom, um

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        distribution = ParameterUniformDist(
            2000 * angstrom,
            0.3 * um,
            3,
            seed=1,
            label_modifier="lam%.0f",
        )

        for value in distribution.realise().values:
            assert value.units == angstrom.units
            assert 2000 * angstrom <= value <= 3000 * angstrom

    def test_partial_units_in_a_distribution_are_rejected(self):
        """Test a distribution half stated in units is caught.

        Reading the bare value as being in the other one's units would be a
        guess, so it is refused instead.
        """
        from unyt import angstrom

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        with pytest.raises(
            exceptions.InconsistentArguments, match="all carry units"
        ):
            ParameterUniformDist(
                2000 * angstrom,
                3000,
                3,
                label_modifier="lam%.0f",
            )

    def test_a_dimensionless_argument_is_untouched(self):
        """Test an argument with no expected units keeps its declaration."""
        from synthesizer.emission_models.attenuation import PowerLaw

        declaration = ParameterList([-1.0, -0.5], label_modifier="slope%.1f")
        curve = PowerLaw(slope=declaration)

        assert curve.slope is declaration


class TestVariantProvenanceInHDF5:
    """Test a written model records what made it a variant.

    A variant's label is generated, so on its own it is a poor record of the
    parameters behind it: a label can be anything the user chose. The values
    are written alongside it so a file can be read without parsing labels or
    re-running the script which produced it.
    """

    def _write(self, model, tmp_path):
        """Write a model to HDF5 and return its group's contents."""
        import h5py

        with h5py.File(tmp_path / "model.hdf5", "w") as hdf:
            model.to_hdf5(hdf.create_group(model.label))

        with h5py.File(tmp_path / "model.hdf5", "r") as hdf:
            group = hdf[model.label]
            return (
                dict(group.attrs),
                dict(group["VariantParameters"].attrs)
                if "VariantParameters" in group
                else None,
            )

    def test_values_are_written(self, test_grid, tmp_path):
        """Test the varied values are written with the base label."""
        from synthesizer.emission_models import PacmanEmission

        expanded = PacmanEmission(
            test_grid,
            tau_v=ParameterList([0.1, 0.5], label_modifier="tauv%.1f"),
            fesc=ParameterList([0.0, 0.5], label_modifier="fesc%.1f"),
        ).expand_models()

        attrs, variant = self._write(
            expanded["emergent_fesc0.5_tauv0.1"], tmp_path
        )

        assert attrs["variant_base"] == "emergent"
        assert variant == {"fesc": 0.5, "tau_v": 0.1}

    def test_units_are_kept(self, test_grid, tmp_path):
        """Test a varied quantity keeps its units, which HDF5 cannot hold."""
        from unyt import kelvin

        from synthesizer.emission_models import (
            AttenuatedEmission,
            DustEmission,
            Greybody,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import PowerLaw

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        attenuated = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.5,
            dust_curve=PowerLaw(slope=-1),
        )
        expanded = DustEmission(
            dust_emission_model=Greybody(
                temperature=ParameterList(
                    [20 * kelvin, 60 * kelvin],
                    label_modifier="T%dK",
                ),
                emissivity=1.5,
            ),
            emitter="stellar",
            label="dust_emission",
            dust_lum_intrinsic=incident,
            dust_lum_attenuated=attenuated,
        ).expand_models()

        _attrs, variant = self._write(expanded["dust_emission_T60K"], tmp_path)

        assert variant["temperature"] == 60.0
        assert variant["temperature_units"] == "K"

    def test_an_object_variation_is_described(self, test_grid, tmp_path):
        """Test a varied dust curve says which curve it was."""
        from synthesizer.emission_models import (
            AttenuatedEmission,
            StellarEmissionModel,
        )
        from synthesizer.emission_models.attenuation import (
            Calzetti2000,
            PowerLaw,
        )

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        expanded = AttenuatedEmission(
            label="attenuated",
            apply_to=incident,
            emitter="stellar",
            tau_v=0.5,
            dust_curve=ParameterList(
                [PowerLaw(slope=-1), Calzetti2000()],
                labels=["powerlaw", "calzetti"],
            ),
        ).expand_models()

        _attrs, variant = self._write(
            expanded["attenuated_calzetti"], tmp_path
        )

        # The label says "calzetti"; the file says what that was
        assert "Calzetti2000" in variant["transformer"]

    def test_a_model_which_is_not_a_variant_writes_nothing(
        self, test_grid, tmp_path
    ):
        """Test an ordinary model gains no variant bookkeeping."""
        from synthesizer.emission_models import StellarEmissionModel

        model = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )

        attrs, variant = self._write(model, tmp_path)

        assert "variant_base" not in attrs
        assert variant is None


class TestPipelineWithVariants:
    """Test a Pipeline handles an expanded model's labels.

    A Pipeline drives its work by model label and writes its output keyed by
    the same, so an expansion should be extra keys and nothing more. The
    example script demonstrates this end to end during the docs build; this
    keeps it in the test suite as well.
    """

    def test_photometry_is_written_for_every_variant(
        self,
        test_grid,
        list_of_random_particle_galaxies,
        uvj_instrument,
        tmp_path,
    ):
        """Test every variant appears in the output, with its parameters."""
        import h5py

        from synthesizer.emission_models import PacmanEmission
        from synthesizer.pipeline import Pipeline

        model = PacmanEmission(
            test_grid,
            tau_v=ParameterList([0.1, 1.0], label_modifier="tauv%.1f"),
            fesc=ParameterList([0.0, 0.5], label_modifier="fesc%.1f"),
        ).expand_models()

        # Only the emergent variants are wanted out the other side
        model.save_spectra("emergent*")
        variants = sorted(model.select("emergent*"), key=lambda m: m.label)
        n_galaxies = len(list_of_random_particle_galaxies)

        pipeline = Pipeline(model, nthreads=1, verbose=0)
        pipeline.add_galaxies(list_of_random_particle_galaxies)
        pipeline.get_photometry_luminosities(uvj_instrument)
        pipeline.run()
        pipeline.write(str(tmp_path / "out.hdf5"), verbose=0)

        with h5py.File(tmp_path / "out.hdf5", "r") as hdf:
            photometry = hdf["Galaxies/Stars/Photometry/Luminosities"]

            # Four variants of the emergent emission, and nothing else
            assert set(photometry.keys()) == {
                model.label for model in variants
            }

            for variant in variants:
                for filter_code in uvj_instrument.filters.filter_codes:
                    lums = photometry[f"{variant.label}/{filter_code}"][:]
                    assert lums.shape == (n_galaxies,)

                # And the file says what each variant was made of
                written = hdf[
                    f"EmissionModel/{variant.label}/VariantParameters"
                ].attrs
                assert dict(written) == variant.variant_params


class _SlopedTransformer:
    """A stand in transformer holding a parameter of its own.

    Defined here rather than reusing a dust curve because the point is a
    transformer the machinery has never seen, holding an argument whose name
    clashes with a model parameter.
    """

    def __init__(self, slope):
        """Store the slope, which may be a declaration."""
        self._required_params = ("slope",)
        self.slope = slope

    def _transform(self, emission, emitter, model, *args, **kwargs):
        """Return the emission unchanged."""
        return emission


class TestVariationsOnUnknownObjects:
    """Test declarations are found on any transformer, not just known ones.

    find_variations looks at every attribute of a transformer or generator
    rather than at the parameters it declares as required. That is deliberate:
    it means an argument which is never looked up on the model can still be
    varied, and it means a transformer written by a user works the same way as
    one that ships with synthesizer. The cost is that any attribute holding a
    declaration is treated as a variation, which these tests pin down.
    """

    def _varied(self, test_grid, **kwargs):
        """Return an expanded model using the stand in transformer."""
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        return StellarEmissionModel(
            label="scaled",
            apply_to=incident,
            emitter="stellar",
            **kwargs,
        ).expand_models()

    def test_a_declaration_on_an_unknown_transformer_is_found(self, test_grid):
        """Test a transformer synthesizer knows nothing about still expands."""
        expanded = self._varied(
            test_grid,
            transformer=_SlopedTransformer(
                slope=ParameterList([1.0, 2.0], label_modifier="curve%.0f")
            ),
        )

        variants = sorted(expanded.select("scaled*"), key=lambda m: m.label)

        assert [model.label for model in variants] == [
            "scaled_curve1",
            "scaled_curve2",
        ]
        assert [model.transformer.slope for model in variants] == [1.0, 2.0]

    def test_a_clashing_name_keeps_both_parameters(self, test_grid):
        """Test a model parameter and an argument of the same name coexist.

        Both are recorded, the transformer's under its qualified name, so
        neither is silently lost.
        """
        expanded = self._varied(
            test_grid,
            transformer=_SlopedTransformer(
                slope=ParameterList([1.0, 2.0], label_modifier="curve%.0f")
            ),
            slope=ParameterList([-1.0, -0.5], label_modifier="model%.1f"),
        )

        variants = sorted(expanded.select("scaled*"), key=lambda m: m.label)

        # Two independent axes, so four variants
        assert len(variants) == 4

        recorded = variants[0].variant_params
        assert recorded == {"slope": -0.5, "transformer.slope": 1.0}

    def test_no_clash_keeps_the_bare_name(self, test_grid):
        """Test the qualified name is only used when it is needed."""
        expanded = self._varied(
            test_grid,
            transformer=_SlopedTransformer(
                slope=ParameterList([1.0, 2.0], label_modifier="curve%.0f")
            ),
        )

        for model in expanded.select("scaled*"):
            assert set(model.variant_params) == {"slope"}


class TestUnseededDistributions:
    """Test an unseeded distribution says its labels will not stay put.

    The labels naming a distribution's variants are rendered from the sampled
    values, so without a seed both the values and the labels change every run,
    and with them the keys the emissions are stored under.
    """

    def test_no_seed_warns(self):
        """Test a distribution with a format string and no seed warns."""
        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        with pytest.warns(RuntimeWarning, match="different values every run"):
            ParameterUniformDist(0.1, 0.5, 3, label_modifier="tauv%.2f")

    def test_a_seed_is_silent(self):
        """Test passing a seed means no warning."""
        import warnings

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ParameterUniformDist(
                0.1, 0.5, 3, seed=42, label_modifier="tauv%.2f"
            )

    def test_explicit_labels_are_silent(self):
        """Test named variants don't need a seed to keep their labels.

        The values still change, but the labels do not depend on them, so
        nothing about the stored emissions moves.
        """
        import warnings

        from synthesizer.emission_models.parameters import (
            ParameterUniformDist,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ParameterUniformDist(0.1, 0.5, 3, labels=["low", "mid", "high"])


class TestCopiesShareNothingMutable:
    """Test a copied model shares no mutable container with its original.

    _copy_node is a shallow copy with an explicit list of containers to copy
    independently. Anything mutable added to EmissionModel later would be
    shared by default, and two variants quietly writing to one container is a
    hard bug to find. This fails if that ever happens.
    """

    def test_no_container_is_shared(self, test_grid):
        """Test every dict, list or set on a copy is the copy's own."""
        from synthesizer.emission_models import BimodalPacmanEmission
        from synthesizer.emission_models.attenuation import PowerLaw

        model = BimodalPacmanEmission(
            grid=test_grid,
            dust_curve_ism=PowerLaw(slope=-0.7),
            dust_curve_birth=PowerLaw(slope=-1.3),
            age_pivot=7.0 * dimensionless,
        )

        for original in model._models.values():
            copied = original._copy_node(label=original.label + "_copy")

            for name, value in vars(original).items():
                if not isinstance(value, (dict, list, set)):
                    continue

                # An empty container is a fresh one either way, and the tree
                # wiring is rebuilt by unpack_model rather than copied
                if name in ("_children", "_parents", "_models"):
                    continue

                assert getattr(copied, name) is not value, (
                    f"{original.label} shares its {name} with its copy, so "
                    "one variant writing to it would change the other. Copy "
                    "it in _copy_node."
                )
