"""Tests for the parameter variation machinery.

This covers the ParameterList/ParameterDistribution markers, the guards which
stop an unexpanded marker being used as a value, and (in later steps) the
expansion of a model tree into its variant models.
"""

import numpy as np
import pytest
import unyt

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


@pytest.fixture
def energy_balance_model(test_grid):
    """Return a tree whose generator depends on two other models."""
    from synthesizer.emission_models import DustEmission, StellarEmissionModel
    from synthesizer.emission_models.generators.dust.blackbody import (
        Blackbody,
    )

    intrinsic = StellarEmissionModel(
        label="intrinsic",
        grid=test_grid,
        extract="incident",
    )
    attenuated = StellarEmissionModel(
        label="attenuated",
        apply_to=intrinsic,
        transformer=EscapingFraction(("fesc",)),
        fesc=0.1,
    )
    return DustEmission(
        dust_emission_model=Blackbody(temperature=50 * unyt.K),
        emitter="stellar",
        label="dust_emission",
        dust_lum_intrinsic=intrinsic,
        dust_lum_attenuated=attenuated,
    )


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
