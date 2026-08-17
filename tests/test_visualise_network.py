"""Tests for the emission model network visualisation.

These cover the plotting entry point and the layout helpers behind it, which
were moved out of base_model into their own submodule.
"""

import matplotlib
import pytest

from synthesizer.emission_models.expansion import _dependency_labels
from synthesizer.emission_models.visualise_network import (
    _get_model_positions,
    _get_tree_levels,
)

# Render without a display
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures a test leaves behind."""
    yield

    import matplotlib.pyplot as plt

    plt.close("all")


class TestTreeLevels:
    """Test assigning models to levels of the tree."""

    def test_root_is_level_zero(self, transmitted_emission_model):
        """Test the root sits at the top of the tree."""
        levels, _, _, _ = _get_tree_levels(
            transmitted_emission_model,
            transmitted_emission_model.label,
        )

        assert levels[0] == [transmitted_emission_model.label]

    def test_dependency_is_deeper(self, transmitted_emission_model):
        """Test a dependency is placed below the model using it."""
        levels, _, _, _ = _get_tree_levels(
            transmitted_emission_model,
            transmitted_emission_model.label,
        )

        assert "full_transmitted" in levels[1]

    def test_dependency_cone_is_placed(self, pacman_emission_model):
        """Test exactly the models the root depends on get a level."""
        levels, _, _, _ = _get_tree_levels(
            pacman_emission_model,
            pacman_emission_model.label,
        )

        placed = {label for labels in levels.values() for label in labels}
        expected = {pacman_emission_model.label} | _dependency_labels(
            pacman_emission_model
        )

        assert placed == expected

    def test_related_models_are_not_placed(self, pacman_emission_model):
        """Test related models are left out of the tree.

        The walk only follows dependencies from the root, so a related model,
        which is a root in its own right, never appears. This is a real
        limitation rather than a desirable behaviour: an expanded parameter
        variation attaches all but one of its variants as related models, so
        only one variant is currently drawn. Pinned here so the replacement
        layout has something to change.
        """
        levels, _, _, _ = _get_tree_levels(
            pacman_emission_model,
            pacman_emission_model.label,
        )

        placed = {label for labels in levels.values() for label in labels}
        assert placed != set(pacman_emission_model._models)
        assert "intrinsic" in pacman_emission_model._models
        assert "intrinsic" not in placed

    def test_extract_labels_found(self, transmitted_emission_model):
        """Test extraction models are flagged for drawing."""
        _, _, extract_labels, _ = _get_tree_levels(
            transmitted_emission_model,
            transmitted_emission_model.label,
        )

        assert "full_transmitted" in extract_labels

    def test_links_are_typed(self, transmitted_emission_model):
        """Test the links record how each model depends on another."""
        _, links, _, _ = _get_tree_levels(
            transmitted_emission_model,
            transmitted_emission_model.label,
        )

        # The transmitted model is a transformation of the incident model, so
        # the link is drawn dashed
        assert ("full_transmitted", "--") in links[
            transmitted_emission_model.label
        ]

    def test_masked_labels_found(self, transmitted_emission_model, test_grid):
        """Test masked models are flagged for drawing."""
        import unyt

        transmitted_emission_model["full_transmitted"].add_mask(
            "ages", "<", 10 * unyt.Myr
        )

        _, _, _, masked_labels = _get_tree_levels(
            transmitted_emission_model,
            transmitted_emission_model.label,
        )

        assert "full_transmitted" in masked_labels


class TestModelPositions:
    """Test laying the tree out on a plane."""

    def test_root_at_origin(self, transmitted_emission_model):
        """Test the root is placed at the origin."""
        root = transmitted_emission_model.label
        levels, _, _, _ = _get_tree_levels(transmitted_emission_model, root)
        pos = _get_model_positions(transmitted_emission_model, levels, root)

        assert pos[root] == (0.0, 0.0)

    def test_every_placed_model_positioned(self, pacman_emission_model):
        """Test every model given a level also gets a position."""
        root = pacman_emission_model.label
        levels, _, _, _ = _get_tree_levels(pacman_emission_model, root)
        pos = _get_model_positions(pacman_emission_model, levels, root)

        placed = {label for labels in levels.values() for label in labels}
        assert set(pos) == placed

    def test_levels_are_separated_vertically(self, pacman_emission_model):
        """Test models on different levels get different y positions."""
        root = pacman_emission_model.label
        levels, _, _, _ = _get_tree_levels(pacman_emission_model, root)
        pos = _get_model_positions(pacman_emission_model, levels, root)

        for level, labels in levels.items():
            for label in labels:
                assert pos[label][1] == pytest.approx(level * 10.0)


class TestPlotEmissionTree:
    """Test the plotting entry point."""

    def test_returns_figure_and_axis(self, transmitted_emission_model):
        """Test a figure and axis come back from the plot."""
        fig, ax = transmitted_emission_model.plot_emission_tree(show=False)

        assert fig is not None
        assert ax is not None

    def test_a_node_is_drawn_per_placed_model(self, pacman_emission_model):
        """Test every model reached from the root is drawn."""
        root = pacman_emission_model.label
        levels, _, _, _ = _get_tree_levels(pacman_emission_model, root)
        placed = {label for labels in levels.values() for label in labels}

        fig, ax = pacman_emission_model.plot_emission_tree(show=False)

        drawn = {text.get_text() for text in ax.texts}
        assert placed <= drawn

    def test_subtree_can_be_plotted(self, transmitted_emission_model):
        """Test passing a root plots only that subtree."""
        fig, ax = transmitted_emission_model.plot_emission_tree(
            root="incident",
            show=False,
        )

        drawn = {text.get_text() for text in ax.texts}
        assert drawn == {"incident"}

    def test_figsize_is_applied(self, transmitted_emission_model):
        """Test the requested figure size is used."""
        fig, _ = transmitted_emission_model.plot_emission_tree(
            show=False,
            figsize=(8, 4),
        )

        assert tuple(fig.get_size_inches()) == (8.0, 4.0)

    def test_expanded_tree_can_be_plotted(self, test_grid):
        """Test a tree of parameter variants can be drawn.

        This is the case the visualisation needs to get better at, so at
        minimum it must not fall over.
        """
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.parameters import ParameterList
        from synthesizer.emission_models.transformers import EscapingFraction

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

        fig, ax = expanded.plot_emission_tree(show=False)

        drawn = {text.get_text() for text in ax.texts}
        assert "escaped_fesc_0.10" in drawn

        # But the other variant is a related model, so it is missing. This is
        # the headline problem for the replacement layout to solve.
        assert "escaped_fesc_0.50" in expanded._models
        assert "escaped_fesc_0.50" not in drawn
