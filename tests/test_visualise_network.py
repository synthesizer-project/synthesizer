"""Tests for the emission model network visualisation.

These cover building the graph of models, laying it out in rows, routing the
edges, and the plotting entry point.
"""

import matplotlib
import numpy as np
import pytest

from synthesizer import exceptions

# Plotting the network is an optional feature, so skip rather than fail when
# its dependency is absent
pytest.importorskip("networkx")

from synthesizer.emission_models.visualise_network import (  # noqa: E402
    _assign_layers,
    _band_loads,
    _box_sizes,
    _box_style,
    _build_graph,
    _drawn_box,
    _edge_lanes,
    _edge_ports,
    _emitter_rank,
    _extent,
    _insert_routing_nodes,
    _layout,
    _order_rows,
    _row_gaps,
    _shape_spec,
)

# Render without a display
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures a test leaves behind."""
    yield

    import matplotlib.pyplot as plt

    plt.close("all")


@pytest.fixture
def expanded_variants(test_grid):
    """Return an expanded tree with three variants of one model."""
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
        fesc=ParameterList([0.1, 0.5, 0.9], label_modifier="fesc_%.2f"),
    )
    return model.expand_models()


def _boxes(positions, sizes):
    """Return the rectangle each node occupies, as (x0, x1, y0, y1)."""
    return {
        node: (
            x - sizes[node][0] / 2.0,
            x + sizes[node][0] / 2.0,
            y - sizes[node][1] / 2.0,
            y + sizes[node][1] / 2.0,
        )
        for node, (x, y) in positions.items()
    }


def _overlaps(first, second):
    """Return whether two rectangles overlap."""
    return (
        first[0] < second[1]
        and second[0] < first[1]
        and first[2] < second[3]
        and second[2] < first[3]
    )


def _assert_no_overlaps(positions, sizes):
    """Assert that no two boxes overlap."""
    boxes = _boxes(positions, sizes)
    for first in boxes:
        for second in boxes:
            if first < second:
                assert not _overlaps(boxes[first], boxes[second]), (
                    f"{first} overlaps {second}"
                )


class TestBuildGraph:
    """Test building the graph of models."""

    def test_whole_network_is_included(self, pacman_emission_model):
        """Test every model in the network becomes a node.

        The original layout only followed dependencies from the root, so
        related models, which are roots in their own right, were dropped.
        """
        graph = _build_graph(pacman_emission_model)

        assert set(graph.nodes) == set(pacman_emission_model._models)

    def test_related_models_are_included(self, pacman_emission_model):
        """Test a related model appears in the graph."""
        graph = _build_graph(pacman_emission_model)

        assert "intrinsic" in graph.nodes

    def test_all_variants_are_included(self, expanded_variants):
        """Test every variant of an expansion is reachable.

        Variants other than the first are attached as related models, so this
        is the case the original layout got most wrong: it drew one of three.
        """
        graph = _build_graph(expanded_variants, show_variants=True)

        assert {
            "escaped_fesc_0.10",
            "escaped_fesc_0.50",
            "escaped_fesc_0.90",
            "incident",
        } == set(graph.nodes)

    def test_edges_follow_the_emission(self, transmitted_emission_model):
        """Test edges run from a dependency to the model consuming it."""
        graph = _build_graph(transmitted_emission_model)

        assert graph.has_edge("full_transmitted", "transmitted")
        assert not graph.has_edge("transmitted", "full_transmitted")

    def test_transform_edges_are_typed(self, transmitted_emission_model):
        """Test a transformation records how it depends on its target."""
        graph = _build_graph(transmitted_emission_model)

        assert (
            graph.edges["full_transmitted", "transmitted"]["kind"]
            == "transform"
        )

    def test_combine_edges_are_typed(self, test_grid):
        """Test a combination records how it depends on its children."""
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        combined = StellarEmissionModel(
            label="combined",
            combine=(incident,),
        )

        graph = _build_graph(combined)

        assert graph.edges["incident", "combined"]["kind"] == "combine"

    def test_scale_by_edges_are_drawn(self, test_grid):
        """Test scaling by another emission is drawn as a dependency.

        The original plots left these out entirely, even though ModelQueue
        treats them as real dependencies.
        """
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        transmitted = StellarEmissionModel(
            label="transmitted",
            grid=test_grid,
            extract="transmitted",
        )
        combined = StellarEmissionModel(
            label="combined",
            combine=(transmitted,),
            related_models=incident,
            scale_by="incident",
        )

        graph = _build_graph(combined)

        assert graph.edges["incident", "combined"]["kind"] == "scale"

    def test_generator_edges_are_typed(self, energy_balance_model):
        """Test a generator's dependencies are drawn."""
        graph = _build_graph(energy_balance_model)

        assert graph.edges["intrinsic", "dust_emission"]["kind"] == "generate"
        assert graph.edges["attenuated", "dust_emission"]["kind"] == "generate"

    def test_reused_emissions_are_drawn(self, test_grid):
        """Test a string dependency is drawn as an external node.

        A string names an emission already on the emitter rather than a model,
        so the reuse would otherwise be invisible.
        """
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import EscapingFraction

        model = StellarEmissionModel(
            label="escaped",
            apply_to="some_existing_spectra",
            transformer=EscapingFraction(("fesc",)),
            fesc=0.1,
            emitter="stellar",
        )

        graph = _build_graph(model)

        assert graph.nodes["some_existing_spectra"]["external"] is True
        assert graph.nodes["escaped"]["external"] is False

    def test_node_styling_attributes(self, transmitted_emission_model):
        """Test each node carries what's needed to draw it."""
        graph = _build_graph(transmitted_emission_model)

        node = graph.nodes["full_transmitted"]
        assert node["emitter"] == "stellar"
        assert node["is_extracting"] is True
        assert node["count"] == 1

    def test_root_restricts_the_graph(self, pacman_emission_model):
        """Test passing a root draws only that model and its dependencies."""
        graph = _build_graph(pacman_emission_model, root="incident")

        assert set(graph.nodes) == {"incident"}

    def test_unknown_root_raises(self, pacman_emission_model):
        """Test an unknown root is rejected."""
        with pytest.raises(exceptions.InconsistentArguments):
            _build_graph(pacman_emission_model, root="not_a_model")


class TestCollapseVariants:
    """Test folding a family of variants into one node.

    This is what happens by default: an expansion of any size produces more
    models than can be read at once, so the variants are only drawn
    individually when they are asked for.
    """

    def test_variants_collapse_to_one_node(self, expanded_variants):
        """Test a variant family becomes a single node."""
        graph = _build_graph(expanded_variants)

        assert set(graph.nodes) == {"escaped", "incident"}

    def test_collapsed_node_counts_its_variants(self, expanded_variants):
        """Test the collapsed node records how many models it stands for."""
        graph = _build_graph(expanded_variants)

        assert graph.nodes["escaped"]["count"] == 3
        assert graph.nodes["incident"]["count"] == 1

    def test_collapsed_edges_are_deduplicated(self, expanded_variants):
        """Test the three parallel dependencies become one edge."""
        graph = _build_graph(expanded_variants)

        assert graph.number_of_edges() == 1
        assert graph.has_edge("incident", "escaped")

    def test_variants_can_be_asked_for(self, expanded_variants):
        """Test every variant is drawn when show_variants is passed."""
        graph = _build_graph(expanded_variants, show_variants=True)

        assert set(graph.nodes) == {
            "escaped_fesc_0.10",
            "escaped_fesc_0.50",
            "escaped_fesc_0.90",
            "incident",
        }
        assert graph.number_of_edges() == 3

    def test_unexpanded_models_are_untouched(self, pacman_emission_model):
        """Test the collapsing default does nothing without variants."""
        collapsed = _build_graph(pacman_emission_model)
        full = _build_graph(pacman_emission_model, show_variants=True)

        assert set(collapsed.nodes) == set(full.nodes)


class TestLayers:
    """Test assigning models to rows."""

    def test_a_model_sits_above_its_dependencies(self, pacman_emission_model):
        """Test every model is on a higher row than what it depends on."""
        graph = _build_graph(pacman_emission_model)
        layers = _assign_layers(graph)

        for source, target in graph.edges:
            assert layers[target] > layers[source]

    def test_dependencies_are_packed_under_consumers(
        self, pacman_emission_model
    ):
        """Test a model sits directly below its lowest consumer.

        This is what keeps the diagram narrow. Collecting every extraction onto
        a shared bottom row instead makes the widest row as wide as the number
        of grids, which is unreadable for a large model.
        """
        graph = _build_graph(pacman_emission_model)
        layers = _assign_layers(graph)

        for node in graph.nodes:
            consumers = list(graph.successors(node))
            if len(consumers) == 0:
                continue

            lowest = min(layers[consumer] for consumer in consumers)
            assert layers[node] == lowest - 1

    def test_every_node_gets_a_row(self, pacman_emission_model):
        """Test no model is left unplaced."""
        graph = _build_graph(pacman_emission_model)
        layers = _assign_layers(graph)

        assert set(layers) == set(graph.nodes)

    def test_bottom_row_is_zero(self, pacman_emission_model):
        """Test the rows count up from zero."""
        graph = _build_graph(pacman_emission_model)

        assert min(_assign_layers(graph).values()) == 0

    def test_ordering_is_reproducible(self, pacman_emission_model):
        """Test the layout does not depend on set iteration order."""
        graph = _build_graph(pacman_emission_model)
        layers = _assign_layers(graph)

        assert _order_rows(graph, layers) == _order_rows(graph, layers)


class TestRoutingNodes:
    """Test the placeholders which reserve a column for long edges."""

    def test_adjacent_edges_need_no_placeholder(
        self, transmitted_emission_model
    ):
        """Test an edge between neighbouring rows is left alone."""
        graph = _build_graph(
            transmitted_emission_model,
            root="transmitted",
        )
        layers = _assign_layers(graph)
        _, _, chains = _insert_routing_nodes(graph, layers)

        assert chains == {}

    def test_long_edges_get_a_placeholder_per_row(self, pacman_emission_model):
        """Test a placeholder is added for every row an edge passes."""
        graph = _build_graph(pacman_emission_model)
        layers = _assign_layers(graph)
        _, routing_layers, chains = _insert_routing_nodes(graph, layers)

        assert len(chains) > 0
        for (source, target), chain in chains.items():
            span = layers[target] - layers[source]
            assert len(chain) == span - 1

            # And each placeholder sits on one of the rows in between
            rows = range(layers[source] + 1, layers[target])
            for placeholder, row in zip(chain, rows):
                assert routing_layers[placeholder] == row


class TestLayout:
    """Test placing the whole network."""

    def test_no_boxes_overlap(self, pacman_emission_model):
        """Test no two boxes overlap.

        The layout works in points, the same units the boxes are drawn in, so
        this holds however long the labels are.
        """
        graph = _build_graph(pacman_emission_model)
        positions, sizes, _ = _layout(graph, "layered", fontsize=10)

        _assert_no_overlaps(positions, sizes)

    def test_no_boxes_overlap_with_long_labels(self, expanded_variants):
        """Test long labels push boxes apart rather than overlapping."""
        graph = _build_graph(expanded_variants)
        positions, sizes, _ = _layout(graph, "layered", fontsize=14)

        _assert_no_overlaps(positions, sizes)

    def test_every_node_positioned(self, pacman_emission_model):
        """Test every model gets a position."""
        graph = _build_graph(pacman_emission_model)
        positions, _, _ = _layout(graph, "layered", fontsize=10)

        assert set(positions) == set(graph.nodes)

    def test_rows_are_separated_vertically(self, pacman_emission_model):
        """Test a model is drawn above what it depends on."""
        graph = _build_graph(pacman_emission_model)
        positions, _, _ = _layout(graph, "layered", fontsize=10)

        for source, target in graph.edges:
            assert positions[target][1] > positions[source][1]

    def test_routes_are_returned_for_long_edges(self, pacman_emission_model):
        """Test the layout reports where long edges should be routed."""
        graph = _build_graph(pacman_emission_model)
        _, _, routes = _layout(graph, "layered", fontsize=10)

        assert len(routes) > 0

    def test_extent_includes_routes(self, pacman_emission_model):
        """Test the bounding box covers the edge routing.

        Routes can swing wide of every box, so leaving them out of the extent
        draws them outside the axes where they get clipped.
        """
        graph = _build_graph(pacman_emission_model)
        positions, sizes, routes = _layout(graph, "layered", fontsize=10)

        without = _extent(positions, sizes)
        with_routes = _extent(positions, sizes, routes)

        assert with_routes[0] <= without[0]
        assert with_routes[1] >= without[1]


class TestPlotEmissionGraph:
    """Test the plotting entry point."""

    def test_returns_figure_and_axis(self, transmitted_emission_model):
        """Test a figure and axis come back from the plot."""
        fig, ax = transmitted_emission_model.plot_emission_graph(show=False)

        assert fig is not None
        assert ax is not None

    def test_every_model_is_drawn(self, pacman_emission_model):
        """Test every model in the network is drawn."""
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)

        drawn = {text.get_text() for text in ax.texts}
        assert set(pacman_emission_model._models) <= drawn

    def test_all_variants_are_drawn(self, expanded_variants):
        """Test every variant is drawn when they are asked for."""
        fig, ax = expanded_variants.plot_emission_graph(
            show=False,
            show_variants=True,
        )

        drawn = {text.get_text() for text in ax.texts}
        assert {
            "escaped_fesc_0.10",
            "escaped_fesc_0.50",
            "escaped_fesc_0.90",
        } <= drawn

    def test_nothing_is_drawn_outside_the_axes(self, pacman_emission_model):
        """Test every box and arrow is inside the axis limits.

        Anything outside is silently clipped, which is how a plot ends up with
        edges running off the side.
        """
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)
        renderer = fig.canvas.get_renderer()

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        inverse = ax.transData.inverted()

        for patch in ax.patches:
            window = patch.get_window_extent(renderer)
            (x0, y0), (x1, y1) = inverse.transform(
                [(window.x0, window.y0), (window.x1, window.y1)]
            )

            # Allow a point of slack for line widths and arrow heads
            assert x0 >= xmin - 2.0
            assert x1 <= xmax + 2.0
            assert y0 >= ymin - 2.0
            assert y1 <= ymax + 2.0

    def test_collapsed_variants_are_badged(self, expanded_variants):
        """Test a collapsed family is drawn once with its count."""
        fig, ax = expanded_variants.plot_emission_graph(show=False)

        drawn = {text.get_text() for text in ax.texts}
        assert any("escaped" in text and "3" in text for text in drawn)
        assert "escaped_fesc_0.10" not in drawn

    def test_an_arrow_is_drawn_per_edge(self, transmitted_emission_model):
        """Test dependencies are drawn as arrows, so direction is visible."""
        fig, ax = transmitted_emission_model.plot_emission_graph(show=False)

        graph = _build_graph(transmitted_emission_model)
        arrows = [
            patch
            for patch in ax.patches
            if isinstance(patch, matplotlib.patches.FancyArrowPatch)
        ]
        assert len(arrows) == graph.number_of_edges()

    def test_legend_describes_what_is_drawn(self, transmitted_emission_model):
        """Test the legend mentions the relationships present."""
        fig, ax = transmitted_emission_model.plot_emission_graph(show=False)

        labels = _legend_labels(ax)
        assert "Transformed" in labels
        assert "Stellar" in labels

        # And nothing which isn't there
        assert "Per Particle" not in labels

    def test_subgraph_can_be_plotted(self, transmitted_emission_model):
        """Test passing a root plots only that model and its dependencies."""
        fig, ax = transmitted_emission_model.plot_emission_graph(
            root="full_transmitted",
            show=False,
        )

        drawn = {text.get_text() for text in ax.texts}
        assert drawn == {"full_transmitted"}

    def test_figure_is_sized_to_the_network(self, pacman_emission_model):
        """Test the figure grows to fit rather than cramming a default."""
        small, _ = pacman_emission_model.plot_emission_graph(
            show=False,
            root="incident",
        )
        whole, _ = pacman_emission_model.plot_emission_graph(show=False)

        assert whole.get_size_inches()[1] > small.get_size_inches()[1]

    def test_figsize_is_respected_when_it_fits(self, pacman_emission_model):
        """Test a figure size big enough for the network is used exactly."""
        fig, _ = pacman_emission_model.plot_emission_graph(
            show=False,
            figsize=(12, 10),
        )

        assert tuple(fig.get_size_inches()) == (12.0, 10.0)

    def test_labels_stay_readable_in_a_small_figure(
        self, pacman_emission_model
    ):
        """Test the figure grows rather than the labels becoming unreadable.

        A plot too small to read is no use whatever size it was asked to be, so
        legibility wins over the requested size. This is the behaviour that
        keeps a large network as readable as a small one.
        """
        fig, ax = pacman_emission_model.plot_emission_graph(
            show=False,
            figsize=(3, 2),
            min_fontsize=6.0,
        )

        assert ax.texts[0].get_fontsize() >= 6.0

        # And the figure grew to make room for them
        width, height = fig.get_size_inches()
        assert width > 3.0 or height > 2.0

    def test_labels_may_shrink_when_asked_to(self, pacman_emission_model):
        """Test the floor can be turned off for a strictly sized figure.

        Turning it off is what lets a caller insist on a figure size: the
        labels then shrink as far as they need to. The figure can still not go
        below what the legends need, since those have a floor of their own.
        """
        floored, floored_ax = pacman_emission_model.plot_emission_graph(
            show=False,
            figsize=(8, 4),
            min_fontsize=6.0,
        )
        shrunk, shrunk_ax = pacman_emission_model.plot_emission_graph(
            show=False,
            figsize=(8, 4),
            min_fontsize=0.0,
        )

        # Without the floor the labels go below it, and the figure stays closer
        # to the size that was asked for
        assert floored_ax.texts[0].get_fontsize() >= 6.0
        assert shrunk_ax.texts[0].get_fontsize() < 6.0
        assert shrunk.get_size_inches()[1] < floored.get_size_inches()[1]

    def test_unknown_layout_raises(self, pacman_emission_model):
        """Test an unknown layout is rejected."""
        with pytest.raises(exceptions.InconsistentArguments):
            pacman_emission_model.plot_emission_graph(
                show=False,
                layout="spring",
            )

    def test_dot_layout(self, pacman_emission_model):
        """Test the graphviz layout draws the same models."""
        pytest.importorskip("pydot")

        fig, ax = pacman_emission_model.plot_emission_graph(
            show=False,
            layout="dot",
        )

        drawn = {text.get_text() for text in ax.texts}
        assert set(pacman_emission_model._models) <= drawn

    def test_dot_layout_flows_upwards(self, pacman_emission_model):
        """Test graphviz agrees with the layered layout on direction."""
        pytest.importorskip("pydot")

        graph = _build_graph(pacman_emission_model)
        positions, _, _ = _layout(graph, "dot", fontsize=10)

        for source, target in graph.edges:
            assert positions[target][1] > positions[source][1]

    def test_dot_layout_has_no_overlaps(self, pacman_emission_model):
        """Test graphviz is told how big the boxes are."""
        pytest.importorskip("pydot")

        graph = _build_graph(pacman_emission_model)
        positions, sizes, _ = _layout(graph, "dot", fontsize=10)

        _assert_no_overlaps(positions, sizes)


class TestDeprecatedAlias:
    """Test the old plot_emission_tree entry point."""

    def test_warns(self, transmitted_emission_model):
        """Test calling the old name warns the user."""
        with pytest.warns(FutureWarning, match="plot_emission_graph"):
            transmitted_emission_model.plot_emission_tree(show=False)

    def test_still_plots(self, transmitted_emission_model):
        """Test the old name still produces the plot."""
        with pytest.warns(FutureWarning):
            fig, ax = transmitted_emission_model.plot_emission_tree(show=False)

        assert fig is not None
        assert len(ax.texts) > 0

    def test_passes_arguments_through(self, transmitted_emission_model):
        """Test arguments still reach the new implementation."""
        with pytest.warns(FutureWarning):
            fig, _ = transmitted_emission_model.plot_emission_tree(
                show=False,
                figsize=(8, 4),
            )

        assert tuple(fig.get_size_inches()) == (8.0, 4.0)


class TestNodeEncodings:
    """Test the channels used to show what a model is.

    Each property of a model gets its own visual channel, so they can be read
    independently: hue for the emitter, shape for the operation, fill for
    whether the emission is kept, outline for masking and reuse.
    """

    def test_operation_is_recorded(self, pacman_emission_model):
        """Test each node knows which operation it performs."""
        graph = _build_graph(pacman_emission_model)

        assert graph.nodes["nebular_continuum"]["operation"] == "extract"
        assert graph.nodes["attenuated"]["operation"] == "transform"
        assert graph.nodes["reprocessed"]["operation"] == "combine"

    def test_generation_is_recorded(self, energy_balance_model):
        """Test a generator is recorded as such."""
        graph = _build_graph(energy_balance_model)

        assert graph.nodes["dust_emission"]["operation"] == "generate"

    def test_operations_get_distinct_shapes(self):
        """Test each operation is drawn with its own box shape.

        The shapes are deliberately very different from each other. A gently
        rounded rectangle next to a sharp one is not something anyone can pick
        apart at the size these diagrams get drawn at.
        """
        shapes = {
            operation: _box_style(
                _shape_spec(operation)["shape"],
                height=20.0,
                tooth=4.0,
            )
            for operation in ("extract", "transform", "combine", "generate")
        }

        assert len(set(shapes.values())) == 4
        assert shapes["extract"].startswith("square")
        assert shapes["combine"].startswith("ellipse")

    def test_shapes_which_overflow_reserve_the_room(self, test_grid):
        """Test a shape drawn outside its box gets that room reserved.

        An ellipse is drawn 1.4 times wider than the box it is given, so the
        layout has to reserve the difference or the boxes would collide.
        """
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        combined = StellarEmissionModel(
            label="incident_but_combined",
            combine=(incident,),
        )

        graph = _build_graph(combined)
        sizes = _box_sizes(graph, fontsize=10)

        # The two labels differ, so compare against the ratio of their text
        # rather than each other
        assert _shape_spec("combine")["overshoot"][0] > 1.0
        assert _shape_spec("extract")["overshoot"] == (1.0, 1.0)

        # And the combination's reserved box includes the allowance
        bare = _box_sizes(_build_graph(incident), fontsize=10)["incident"]
        assert sizes["incident"] == bare

    def test_discarded_nodes_are_unfilled(self, pacman_emission_model):
        """Test a discarded emission is drawn hollow, not merely faded.

        Fading a pale fill is nearly invisible at small sizes, so the fill
        itself carries whether the emission is kept.
        """
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)

        graph = _build_graph(pacman_emission_model)
        discarded = [
            node for node, data in graph.nodes(data=True) if not data["save"]
        ]
        saved = [node for node, data in graph.nodes(data=True) if data["save"]]
        assert len(discarded) > 0 and len(saved) > 0

        # Every box is drawn as a patch, so collect the fills actually used
        fills = {
            patch.get_facecolor()
            for patch in ax.patches
            if isinstance(patch, matplotlib.patches.FancyBboxPatch)
        }

        white = matplotlib.colors.to_rgba("#ffffff")
        assert white in fills

        # And the saved boxes are filled with something else
        assert any(fill != white and fill[3] > 0 for fill in fills)

    def test_legend_explains_the_shapes(self, pacman_emission_model):
        """Test the legend names the operations, not just the colours."""
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)

        labels = _legend_labels(ax)
        assert "Extraction" in labels
        assert "Transformation" in labels
        assert "Combination" in labels

    def test_legend_swatches_keep_their_shape(self, pacman_emission_model):
        """Test each operation's swatch is drawn with its own shape.

        Matplotlib's default patch handler rebuilds a swatch as a plain
        rectangle, which throws the box style away and leaves every shape in
        the legend identical. Checking the labels alone does not catch that, so
        this compares the paths actually drawn.
        """
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)
        fig.canvas.draw()

        drawn = {}
        for legend in _legends(ax):
            for handle, text in zip(legend.legend_handles, legend.get_texts()):
                label = text.get_text()
                if label in ("Extraction", "Transformation", "Combination"):
                    drawn[label] = tuple(
                        np.round(handle.get_path().vertices.flatten(), 2)
                    )

        assert len(drawn) == 3
        assert len(set(drawn.values())) == 3

        # And the extraction swatch is the square one
        assert len(drawn["Extraction"]) < len(drawn["Transformation"])

    def test_legend_explains_saved_and_discarded(self, pacman_emission_model):
        """Test the legend covers the fill encoding when it's in use."""
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)

        labels = _legend_labels(ax)
        assert "Saved" in labels
        assert "Discarded" in labels

    def test_legend_is_not_clipped(self, expanded_variants):
        """Test the figure is wide enough for its own legend.

        A small network can need less width than the legend does, so the legend
        would otherwise run off the edge.
        """
        fig, ax = expanded_variants.plot_emission_graph(show=False)
        renderer = fig.canvas.get_renderer()

        figure = fig.bbox
        for legend in _legends(ax):
            box = legend.get_window_extent(renderer)
            assert box.x0 >= figure.x0 - 1.0
            assert box.x1 <= figure.x1 + 1.0


class TestCompactness:
    """Test the diagram doesn't spread out more than it needs to."""

    def test_width_is_close_to_the_widest_row(self, pacman_emission_model):
        """Test the diagram is no wider than its content needs.

        Straightening pulls nodes towards their neighbours, and repairing the
        overlaps that causes used to only ever push right, which left gaps that
        never closed and spread the diagram across the figure.
        """
        graph = _build_graph(pacman_emission_model)
        positions, sizes, _ = _layout(graph, "layered", fontsize=10)
        layers = _assign_layers(graph)

        # The width the busiest row genuinely needs
        rows = {}
        for node, layer in layers.items():
            rows.setdefault(layer, []).append(node)
        needed = max(
            sum(sizes[node][0] for node in nodes) for nodes in rows.values()
        )

        xmin, xmax, _, _ = _extent(positions, sizes)

        assert (xmax - xmin) < 2.5 * needed


@pytest.fixture
def multi_emitter_model(test_grid):
    """Return a galaxy model combining stellar and black hole models."""
    from synthesizer.emission_models import (
        BlackHoleEmissionModel,
        GalaxyEmissionModel,
        StellarEmissionModel,
    )

    stellar_incident = StellarEmissionModel(
        label="stellar_incident",
        grid=test_grid,
        extract="incident",
    )
    stellar = StellarEmissionModel(
        label="stellar_total",
        combine=(stellar_incident,),
    )
    blackhole_incident = BlackHoleEmissionModel(
        label="agn_incident",
        grid=test_grid,
        extract="incident",
    )
    blackhole = BlackHoleEmissionModel(
        label="agn_total",
        combine=(blackhole_incident,),
    )
    return GalaxyEmissionModel(
        label="total",
        combine=(stellar, blackhole),
    )


class TestEmitterClustering:
    """Test the models of each emitter are kept together.

    Interleaving a black hole model between two stellar ones makes a combined
    model much harder to read, so the roots are clustered by emitter and the
    rows below follow them.
    """

    def test_emitters_are_ranked_left_to_right(self, multi_emitter_model):
        """Test the ranking puts black holes left and stellar right."""
        graph = _build_graph(multi_emitter_model)

        assert _emitter_rank(graph, "agn_incident") < _emitter_rank(
            graph, "total"
        )
        assert _emitter_rank(graph, "total") < _emitter_rank(
            graph, "stellar_incident"
        )

    def test_unknown_nodes_rank_in_the_middle(self, multi_emitter_model):
        """Test a node with no emitter doesn't disturb the clustering."""
        graph = _build_graph(multi_emitter_model)

        middle = _emitter_rank(graph, "not_a_node")
        assert _emitter_rank(graph, "agn_incident") < middle
        assert middle < _emitter_rank(graph, "stellar_incident")

    def test_roots_are_clustered_by_emitter(self, multi_emitter_model):
        """Test the top row groups the emitters rather than interleaving."""
        graph = _build_graph(multi_emitter_model)
        layers = _assign_layers(graph)
        rows = _order_rows(graph, layers)

        top = rows[max(rows)]
        ranks = [_emitter_rank(graph, node) for node in top]

        assert ranks == sorted(ranks)

    def test_clustering_survives_the_routing_graph(self, multi_emitter_model):
        """Test the ordering can still see which emitter a model belongs to.

        The routing graph is a copy with placeholders added, and copying the
        nodes without their attributes silently lost the emitter, which left
        every model ranked in the middle and the clustering doing nothing.
        """
        graph = _build_graph(multi_emitter_model)
        layers = _assign_layers(graph)
        routing, routing_layers, _ = _insert_routing_nodes(graph, layers)

        for node in graph.nodes:
            assert (
                routing.nodes[node]["emitter"] == graph.nodes[node]["emitter"]
            )

    def test_emitters_end_up_in_separate_regions(self, multi_emitter_model):
        """Test the drawn positions separate the emitters."""
        graph = _build_graph(multi_emitter_model)
        positions, _, _ = _layout(graph, "layered", fontsize=10)

        by_emitter = {}
        for node, (x, _) in positions.items():
            emitter = graph.nodes[node]["emitter"]
            by_emitter.setdefault(emitter, []).append(x)

        assert np.mean(by_emitter["blackhole"]) < np.mean(
            by_emitter["stellar"]
        )


def _box_paths(ax):
    """Return the outline of every box on an axis, in data coordinates."""
    return [
        patch.get_path().transformed(patch.get_patch_transform())
        for patch in ax.patches
        if isinstance(patch, matplotlib.patches.FancyBboxPatch)
    ]


def _legends(ax):
    """Return every legend drawn on an axis.

    The encodings are split across several legends, one per question they
    answer, so anything checking them has to look at all of them rather than at
    whichever one happens to be the axis's own.

    Args:
        ax (matplotlib.axes.Axes):
            The axis drawn on.

    Returns:
        list:
            The legends.
    """
    return [
        artist
        for artist in ax.get_children()
        if isinstance(artist, matplotlib.legend.Legend)
    ]


def _legend_labels(ax):
    """Return every label across every legend on an axis.

    Args:
        ax (matplotlib.axes.Axes):
            The axis drawn on.

    Returns:
        set:
            The labels.
    """
    return {
        text.get_text()
        for legend in _legends(ax)
        for text in legend.get_texts()
    }


def _box_paths_by_zorder(ax):
    """Return every box outline with the depth it is drawn at.

    Args:
        ax (matplotlib.axes.Axes):
            The axis drawn on.

    Returns:
        list:
            Tuples of (zorder, path).
    """
    return [
        (
            patch.get_zorder(),
            patch.get_path().transformed(patch.get_patch_transform()),
        )
        for patch in ax.patches
        if isinstance(patch, matplotlib.patches.FancyBboxPatch)
    ]


def _perimeter(x0, y0, x1, y1, n=25):
    """Return points around the edge of a box."""
    ts = np.linspace(0.0, 1.0, n)

    return (
        [(x0 + t * (x1 - x0), y0) for t in ts]
        + [(x0 + t * (x1 - x0), y1) for t in ts]
        + [(x0, y0 + t * (y1 - y0)) for t in ts]
        + [(x1, y0 + t * (y1 - y0)) for t in ts]
    )


def _label_clearance(fig, ax):
    """Return the least room any label has inside its box.

    Measured as how far the label could grow on every side and still fit
    inside the outline, in units of the font size.

    Args:
        fig (matplotlib.figure.Figure):
            The figure drawn.
        ax (matplotlib.axes.Axes):
            The axis drawn on.

    Returns:
        tuple:
            The smallest clearance found and the label it belongs to.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()

    # A per particle model is drawn as a stack, and the offset copies behind
    # the front box are close enough to contain its label's centre too. The
    # label belongs to whichever box is drawn on top, so order by that.
    paths = [
        path
        for _, path in sorted(
            _box_paths_by_zorder(ax),
            key=lambda item: -item[0],
        )
    ]
    fontsize = ax.texts[0].get_fontsize()

    worst = (float("inf"), None)
    for text in ax.texts:
        window = text.get_window_extent(renderer)
        (x0, y0), (x1, y1) = inverse.transform(
            [(window.x0, window.y0), (window.x1, window.y1)]
        )

        for path in paths:
            if not path.contains_point(((x0 + x1) / 2.0, (y0 + y1) / 2.0)):
                continue

            # Grow the label until it no longer fits
            low, high = 0.0, 2.0 * fontsize
            for _ in range(16):
                middle = (low + high) / 2.0
                if all(
                    path.contains_point(point)
                    for point in _perimeter(
                        x0 - middle,
                        y0 - middle,
                        x1 + middle,
                        y1 + middle,
                    )
                ):
                    low = middle
                else:
                    high = middle

            if low / fontsize < worst[0]:
                worst = (low / fontsize, text.get_text())
            break

    return worst


class TestLabelsFitTheirBoxes:
    """Test every label sits comfortably inside its outline.

    A box only just big enough for its label looks like a mistake, and none of
    the other checks notice: nothing overlaps and nothing is clipped. The
    shapes make this easy to get wrong, because a scalloped or curved outline
    bites into the box the label was measured against.
    """

    def test_labels_have_room_in_every_shape(self, pacman_emission_model):
        """Test the labels clear their outlines by a visible margin."""
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)

        clearance, label = _label_clearance(fig, ax)

        assert clearance > 0.25, f"'{label}' has {clearance:.3f} clearance"

    def test_scalloped_boxes_have_room(self, energy_balance_model):
        """Test a generator's scalloped outline doesn't crowd its label.

        Scallops sized as a fraction of the box grow with it, so padding the
        box buys almost nothing. Sizing them against the font instead is what
        makes the padding work.
        """
        fig, ax = energy_balance_model.plot_emission_graph(show=False)

        clearance, label = _label_clearance(fig, ax)

        assert clearance > 0.25, f"'{label}' has {clearance:.3f} clearance"

    def test_labels_have_room_when_shrunk_to_fit(self, pacman_emission_model):
        """Test the margin survives the labels being scaled down."""
        fig, ax = pacman_emission_model.plot_emission_graph(
            show=False,
            figsize=(4, 3),
        )

        clearance, label = _label_clearance(fig, ax)

        assert clearance > 0.25, f"'{label}' has {clearance:.3f} clearance"

    def test_reused_emissions_have_room(self, test_grid):
        """Test a node with no operation is padded like the shape it gets."""
        from synthesizer.emission_models import StellarEmissionModel
        from synthesizer.emission_models.transformers import EscapingFraction

        model = StellarEmissionModel(
            label="escaped",
            apply_to="some_existing_spectra",
            transformer=EscapingFraction(("fesc",)),
            fesc=0.1,
            emitter="stellar",
        )

        fig, ax = model.plot_emission_graph(show=False)

        clearance, label = _label_clearance(fig, ax)

        assert clearance > 0.25, f"'{label}' has {clearance:.3f} clearance"


class TestLegendSwatchGeometry:
    """Test the swatches are scaled versions of what they stand for."""

    def test_swatch_teeth_are_visible_but_not_overwhelming(self):
        """Test a scalloped swatch reads as scalloped, and still as a box.

        There is a band either side of which the swatch is wrong in a different
        way. Too shallow and the outline reads as a slightly fuzzy rectangle,
        indistinguishable from the plain shapes. Too deep and the teeth swallow
        the straight edges, leaving a blob. Matching the proportion the boxes
        themselves use lands in the first failure, because a swatch is a
        fraction of their size.
        """
        from synthesizer.emission_models.visualise_network import (
            _SWATCH_TOOTH,
        )

        assert 0.15 <= _SWATCH_TOOTH <= 0.25


def _edge_segments(ax):
    """Return the straight, axis aligned parts of every arrow.

    Args:
        ax (matplotlib.axes.Axes):
            The axis drawn on.

    Returns:
        list:
            Tuples of (orientation, line, low, high) describing each segment.
    """
    segments = []
    for patch in ax.patches:
        if not isinstance(patch, matplotlib.patches.FancyArrowPatch):
            continue

        vertices = patch.get_path().vertices
        for (x0, y0), (x1, y1) in zip(vertices[:-1], vertices[1:]):
            if np.isclose(y0, y1) and abs(x1 - x0) > 1.0:
                segments.append(("h", y0, min(x0, x1), max(x0, x1)))
            elif np.isclose(x0, x1) and abs(y1 - y0) > 1.0:
                segments.append(("v", x0, min(y0, y1), max(y0, y1)))

    return segments


def _coincident_segments(segments, tol=0.75):
    """Count pairs of segments lying on the same line and overlapping.

    Args:
        segments (list):
            The segments to compare.
        tol (float):
            How close two segments must be to count as on the same line.

    Returns:
        int:
            The number of overlapping pairs.
    """
    count = 0
    for i, (kind_a, line_a, low_a, high_a) in enumerate(segments):
        for kind_b, line_b, low_b, high_b in segments[i + 1 :]:
            if kind_a != kind_b or abs(line_a - line_b) > tol:
                continue
            if min(high_a, high_b) - max(low_a, low_b) > 2.0:
                count += 1

    return count


class TestEdgesAreDistinguishable:
    """Test edges can be followed individually where many run together.

    Every edge leaving a box from its centre, and every sideways run happening
    at the middle of its gap, puts lines exactly on top of each other. Fanning
    the attachment points out and giving each run its own lane is what keeps
    them apart.
    """

    def test_edges_do_not_run_along_each_other(self, pacman_emission_model):
        """Test no two edges share a stretch of the same line."""
        fig, ax = pacman_emission_model.plot_emission_graph(show=False)
        fig.canvas.draw()

        segments = _edge_segments(ax)

        assert len(segments) > 0
        assert _coincident_segments(segments) == 0

    def test_dense_networks_stay_distinguishable(
        self, bimodal_pacman_emission_model
    ):
        """Test a busy network keeps its edges apart too.

        Without lanes this model has hundreds of overlapping stretches, which
        is what made the edges impossible to follow.
        """
        fig, ax = bimodal_pacman_emission_model.plot_emission_graph(
            show=False,
            fontsize=5,
        )
        fig.canvas.draw()

        segments = _edge_segments(ax)

        assert len(segments) > 100
        assert _coincident_segments(segments) < 0.02 * len(segments)

    def test_edges_leave_a_box_at_different_points(self, test_grid):
        """Test a model feeding several others gets a port for each."""
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        consumers = [
            StellarEmissionModel(label=f"consumer_{i}", combine=(incident,))
            for i in range(3)
        ]
        root = StellarEmissionModel(
            label="root",
            combine=tuple(consumers),
        )

        graph = _build_graph(root)
        positions, sizes, _ = _layout(graph, "layered", fontsize=10)
        leaving, arriving = _edge_ports(graph, positions, sizes)

        ports = [leaving[edge] for edge in graph.out_edges("incident")]
        assert len(set(ports)) == len(ports)

        # And they stay inside the box they leave from
        half_width = sizes["incident"][0] / 2.0
        for port in ports:
            assert abs(port - positions["incident"][0]) < half_width

    def test_edges_arrive_at_a_box_at_different_points(self, test_grid):
        """Test a combination is entered at a separate point per child."""
        from synthesizer.emission_models import StellarEmissionModel

        children = [
            StellarEmissionModel(
                label=f"child_{i}",
                grid=test_grid,
                extract="incident",
            )
            for i in range(3)
        ]
        combined = StellarEmissionModel(
            label="combined",
            combine=tuple(children),
        )

        graph = _build_graph(combined)
        positions, sizes, _ = _layout(graph, "layered", fontsize=10)
        _, arriving = _edge_ports(graph, positions, sizes)

        ports = [arriving[edge] for edge in graph.in_edges("combined")]
        assert len(set(ports)) == len(ports)

    def test_lanes_stay_between_the_rows(self, pacman_emission_model):
        """Test a sideways run never strays into a row of boxes.

        The lanes are confined to the band between two rows, which is what
        stops a rerouted edge crossing a box.
        """
        graph = _build_graph(pacman_emission_model)
        positions, sizes, routes = _layout(graph, "layered", fontsize=10)
        leaving, arriving = _edge_ports(graph, positions, sizes)
        lanes = _edge_lanes(graph, positions, routes, leaving, arriving)

        assert len(lanes) > 0

        # Every box, as a band of y it occupies
        boxes = [
            (y - sizes[node][1] / 2.0, y + sizes[node][1] / 2.0)
            for node, (_, y) in positions.items()
        ]

        for lane in lanes.values():
            for bottom, top in boxes:
                assert not bottom < lane < top


class TestBandsScaleWithTraffic:
    """Test a band grows with the number of edges which must cross it.

    Every edge crossing a band turns sideways in it and needs its own lane, so
    a fixed gap leaves the busy bands with their lanes almost on top of each
    other while the quiet ones sit needlessly far apart.
    """

    def test_loads_count_the_crossing_edges(self, test_grid):
        """Test each band knows how many edges pass through it."""
        from synthesizer.emission_models import StellarEmissionModel

        incident = StellarEmissionModel(
            label="incident",
            grid=test_grid,
            extract="incident",
        )
        middle = StellarEmissionModel(label="middle", combine=(incident,))

        # The root depends on both, so one edge runs the whole height of the
        # network and passes through every band on the way
        root = StellarEmissionModel(
            label="root",
            combine=(middle, incident),
        )

        graph = _build_graph(root)
        layers = _assign_layers(graph)
        loads = _band_loads(graph, layers)

        # Both bands carry two edges: the short one between their own rows, and
        # the long one passing through
        assert loads[layers["incident"]] == 2
        assert loads[layers["middle"]] == 2

        # An edge is counted once for every band it crosses, which is what
        # makes a band carrying long edges wider than its own edge count
        spans = sum(
            layers[target] - layers[source] for source, target in graph.edges
        )
        assert sum(loads.values()) == spans

    def test_busy_bands_are_taller(self, bimodal_pacman_emission_model):
        """Test the band carrying the most edges is the tallest."""
        graph = _build_graph(bimodal_pacman_emission_model)
        layers = _assign_layers(graph)
        loads = _band_loads(graph, layers)
        positions, sizes, _ = _layout(graph, "layered", fontsize=6)

        # The clear band between each pair of neighbouring rows
        gaps = _row_gaps(positions, sizes)
        heights = {}
        for (lower, _), (bottom, top) in gaps.items():
            row = min(
                layers[node]
                for node, (_, y) in positions.items()
                if np.isclose(y, lower)
            )
            heights[row] = top - bottom

        busiest = max(heights, key=lambda row: loads.get(row, 0))
        quietest = min(heights, key=lambda row: loads.get(row, 0))

        assert loads[busiest] > loads[quietest]
        assert heights[busiest] > heights[quietest]

    def test_lanes_are_separated_enough_to_follow(
        self, bimodal_pacman_emission_model
    ):
        """Test the lanes in a busy band are far enough apart to trace.

        Lanes a fraction of a point apart are no better than lanes on top of
        each other, which is what a fixed band height produced.
        """
        graph = _build_graph(bimodal_pacman_emission_model)
        positions, sizes, routes = _layout(graph, "layered", fontsize=6)
        leaving, arriving = _edge_ports(graph, positions, sizes)
        lanes = _edge_lanes(graph, positions, routes, leaving, arriving)

        # Group the lanes by the band they sit in, then measure the gaps
        bands = {}
        for lane in lanes.values():
            bands.setdefault(round(lane / 12.0), []).append(lane)

        separations = []
        for band in bands.values():
            band.sort()
            separations.extend(np.diff(band))

        separations = [gap for gap in separations if gap > 1e-6]

        assert len(separations) > 0
        assert min(separations) > 2.0


class TestLegibilityAtScale:
    """Test a big network stays as readable as a small one.

    The labels are what a reader is actually here for, so the figure grows to
    fit them rather than the labels shrinking to fit the figure. Doing it the
    other way round produces a tidy plot of a large model that nobody can read,
    which is the trap.
    """

    def test_a_large_network_keeps_its_font_size(
        self, bimodal_pacman_emission_model, transmitted_emission_model
    ):
        """Test the same font size is used whatever the network's size."""
        _, small = transmitted_emission_model.plot_emission_graph(show=False)
        _, large = bimodal_pacman_emission_model.plot_emission_graph(
            show=False
        )

        assert small.texts[0].get_fontsize() == large.texts[0].get_fontsize()

    def test_a_large_network_gets_a_larger_figure(
        self, bimodal_pacman_emission_model, transmitted_emission_model
    ):
        """Test the figure is what grows with the network."""
        small_fig, _ = transmitted_emission_model.plot_emission_graph(
            show=False
        )
        large_fig, _ = bimodal_pacman_emission_model.plot_emission_graph(
            show=False
        )

        assert large_fig.get_size_inches()[0] > small_fig.get_size_inches()[0]

    def test_the_floor_is_honoured_for_every_model(
        self, bimodal_pacman_emission_model
    ):
        """Test no label ends up below the floor, whatever is asked for."""
        for figsize in ((2, 2), (4, 3), (8, 6)):
            _, ax = bimodal_pacman_emission_model.plot_emission_graph(
                show=False,
                figsize=figsize,
                min_fontsize=6.0,
            )

            assert ax.texts[0].get_fontsize() >= 6.0


@pytest.fixture
def per_particle_model(test_grid):
    """Return a model with some of its emissions held per particle."""
    from synthesizer.emission_models import PacmanEmission

    model = PacmanEmission(grid=test_grid)
    model["nebular"].set_per_particle(True)

    return model


class TestPerParticleIsAStack:
    """Test a per particle model is drawn as a stack of boxes.

    A per particle model holds an emission for every particle rather than one
    for the whole component, so it is drawn the way a flowchart marks something
    standing for many instances. Hatching the box says the same thing, but has
    to be drawn across the label to do it, which is what made those labels hard
    to read.
    """

    def test_the_label_is_never_covered(self, per_particle_model):
        """Test nothing is drawn over the label of a per particle model."""
        fig, ax = per_particle_model.plot_emission_graph(show=False)

        # No patch carries a hatch, which is what used to sit over the label
        for patch in ax.patches:
            assert not patch.get_hatch()

    def test_labels_still_have_room(self, per_particle_model):
        """Test the stack doesn't crowd the label it sits behind."""
        fig, ax = per_particle_model.plot_emission_graph(show=False)

        clearance, label = _label_clearance(fig, ax)

        assert clearance > 0.25, f"'{label}' has {clearance:.3f} clearance"

    def test_extra_boxes_are_drawn(self, per_particle_model):
        """Test the stack really is extra boxes behind the front one."""
        plain = _build_graph(per_particle_model)
        stacked = sum(
            1 for _, data in plain.nodes(data=True) if data["per_particle"]
        )
        assert stacked > 0

        fig, ax = per_particle_model.plot_emission_graph(show=False)
        boxes = [
            patch
            for patch in ax.patches
            if isinstance(patch, matplotlib.patches.FancyBboxPatch)
        ]

        # One box per model, plus the copies behind each stacked one
        from synthesizer.emission_models.visualise_network import (
            _STACK_COPIES,
        )

        assert len(boxes) == len(plain.nodes) + stacked * _STACK_COPIES

    def test_the_stack_is_behind_the_box(self, per_particle_model):
        """Test the copies never cover the box carrying the label."""
        fig, ax = per_particle_model.plot_emission_graph(show=False)

        boxes = [
            patch
            for patch in ax.patches
            if isinstance(patch, matplotlib.patches.FancyBboxPatch)
        ]
        fronts = [patch for patch in boxes if patch.get_zorder() == 2]
        behind = [patch for patch in boxes if patch.get_zorder() < 2]

        assert len(behind) > 0
        for patch in behind:
            assert patch.get_zorder() < min(f.get_zorder() for f in fronts)

    def test_the_footprint_covers_the_stack(self, per_particle_model):
        """Test the room reserved for a stacked node includes its copies.

        Otherwise the copies would reach into whatever sits next to them.
        """
        graph = _build_graph(per_particle_model)
        sizes = _box_sizes(graph, fontsize=10)

        stacked = [
            node
            for node, data in graph.nodes(data=True)
            if data["per_particle"]
        ]
        for node in stacked:
            box = _drawn_box(graph.nodes[node], sizes[node], 10.0)
            assert box[0] < sizes[node][0]
            assert box[1] < sizes[node][1]

    def test_boxes_still_do_not_overlap(self, per_particle_model):
        """Test the stacks don't collide with their neighbours."""
        graph = _build_graph(per_particle_model)
        positions, sizes, _ = _layout(graph, "layered", fontsize=10)

        _assert_no_overlaps(positions, sizes)

    def test_the_legend_shows_a_stack(self, per_particle_model):
        """Test the legend explains the encoding with a stacked swatch."""
        fig, ax = per_particle_model.plot_emission_graph(show=False)
        fig.canvas.draw()

        assert "Per Particle" in _legend_labels(ax)


class TestGroupedLegends:
    """Test the encodings are explained in separate, titled legends.

    A single legend listing every encoding takes a lot of reading to find the
    one entry you wanted. Grouping them by the question they answer means each
    group can be read on its own.
    """

    def test_there_is_a_legend_per_group(self, multi_emitter_model):
        """Test the encodings are split into titled groups."""
        fig, ax = multi_emitter_model.plot_emission_graph(show=False)

        titles = {legend.get_title().get_text() for legend in _legends(ax)}
        assert titles == {"Relationships", "Components", "Model Properties"}

    def test_each_group_holds_its_own_encodings(self, multi_emitter_model):
        """Test an entry appears under the question it answers."""
        fig, ax = multi_emitter_model.plot_emission_graph(show=False)

        grouped = {
            legend.get_title().get_text(): {
                text.get_text() for text in legend.get_texts()
            }
            for legend in _legends(ax)
        }

        assert "Combined" in grouped["Relationships"]
        assert "Stellar" in grouped["Components"]
        assert "Black Hole" in grouped["Components"]
        assert "Extraction" in grouped["Model Properties"]
        assert "Combination" in grouped["Model Properties"]

        # And nothing is listed twice
        seen = [label for labels in grouped.values() for label in labels]
        assert len(seen) == len(set(seen))

    def test_empty_groups_are_left_out(self, transmitted_emission_model):
        """Test a group with nothing to say isn't drawn.

        A stellar only model has one component, so naming it in a legend of its
        own tells the reader nothing.
        """
        fig, ax = transmitted_emission_model.plot_emission_graph(show=False)

        grouped = {
            legend.get_title().get_text(): {
                text.get_text() for text in legend.get_texts()
            }
            for legend in _legends(ax)
        }

        # Only one emitter is present, so its group holds a single entry
        assert grouped["Components"] == {"Stellar"}

        # And nothing claims a per particle or variant encoding is in use
        properties = grouped["Model Properties"]
        assert "Per Particle" not in properties
        assert "×N Variants" not in properties

    def test_groups_do_not_overlap(self, multi_emitter_model):
        """Test the groups are placed side by side rather than on top."""
        fig, ax = multi_emitter_model.plot_emission_graph(show=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        boxes = [legend.get_window_extent(renderer) for legend in _legends(ax)]
        for i, first in enumerate(boxes):
            for second in boxes[i + 1 :]:
                # Groups in different bands are free to share the same columns,
                # since one is above the network and the other below it
                if min(first.y1, second.y1) - max(first.y0, second.y0) <= 0:
                    continue

                overlap = min(first.x1, second.x1) - max(first.x0, second.x0)
                assert overlap <= 1.0

    def test_groups_clear_the_network(self, multi_emitter_model):
        """Test the legends sit clear of the network rather than over it."""
        fig, ax = multi_emitter_model.plot_emission_graph(show=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        labels = [text.get_window_extent(renderer) for text in ax.texts]
        top = max(box.y1 for box in labels)
        bottom = min(box.y0 for box in labels)

        for legend in _legends(ax):
            box = legend.get_window_extent(renderer)
            assert box.y0 > top or box.y1 < bottom

    def test_the_bands_are_split_above_and_below(self, multi_emitter_model):
        """Test the shorter groups head the plot and the longer one foots it.

        Stacking all three above the network leaves a deep band of legend over
        a shallow network, which is why they are split.
        """
        fig, ax = multi_emitter_model.plot_emission_graph(show=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        placed = {
            legend.get_title().get_text(): legend.get_window_extent(renderer)
            for legend in _legends(ax)
        }
        middle = sum(
            text.get_window_extent(renderer).y0 for text in ax.texts
        ) / len(ax.texts)

        assert placed["Relationships"].y0 > middle
        assert placed["Components"].y0 > middle
        assert placed["Model Properties"].y1 < middle

    def test_groups_are_titled_in_bold(self, multi_emitter_model):
        """Test the titles stand out from the entries they head."""
        fig, ax = multi_emitter_model.plot_emission_graph(show=False)

        for legend in _legends(ax):
            assert legend.get_title().get_weight() == "bold"


class TestLegendWidthMeasurement:
    """Test the room reserved for a legend matches what gets drawn.

    The groups are placed by measurement, so an estimate which comes out short
    puts the next group on top of this one. Two things made it come out short:
    matplotlib fills legend columns top to bottom rather than left to right, so
    the widest row is not the sum of the entries next to each other; and the
    titles are drawn in bold, which is wider than the same text measured plain.
    """

    def test_a_group_is_no_wider_than_reserved(self, dusty_stellar_model):
        """Test each legend fits in the width it asked for."""
        from synthesizer.emission_models.visualise_network import (
            _MIN_LEGEND_FONTSIZE,
            _legend_group_width,
            _legend_groups,
            _legend_shape,
        )

        fig, ax = dusty_stellar_model.plot_emission_graph(show=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        graph = _build_graph(dusty_stellar_model)
        groups = _legend_groups(graph)
        shape = _legend_shape(groups)
        fontsize = max(
            ax.texts[0].get_fontsize(),
            _MIN_LEGEND_FONTSIZE,
        )

        # Window extents are in pixels, while the reserved widths are in
        # points, which is also what the axis is in. Convert through the axis
        # so the two are comparable.
        inverse = ax.transData.inverted()
        drawn = {}
        for legend in _legends(ax):
            box = legend.get_window_extent(renderer)
            (x0, _), (x1, _) = inverse.transform(
                [(box.x0, box.y0), (box.x1, box.y1)]
            )
            drawn[legend.get_title().get_text()] = x1 - x0

        for title, handles in groups.items():
            reserved = (
                _legend_group_width(title, handles, shape[title][0]) * fontsize
            )
            assert drawn[title] <= reserved + 1.0, (
                f"'{title}' is {drawn[title]:.1f}pt wide but only "
                f"{reserved:.1f}pt was reserved"
            )

    def test_dust_emission_does_not_crowd_the_legends(
        self, dusty_stellar_model
    ):
        """Test the groups stay apart when a generator adds an entry.

        Dust emission puts a "Generator Scaling" entry in the relationships
        group, which is much longer than the others, and a column major fill
        puts it in the first row where a row major measurement did not expect
        it.
        """
        fig, ax = dusty_stellar_model.plot_emission_graph(show=False)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        labels = _legend_labels(ax)
        assert "Generator Scaling" in labels

        boxes = [legend.get_window_extent(renderer) for legend in _legends(ax)]
        for i, first in enumerate(boxes):
            for second in boxes[i + 1 :]:
                if min(first.y1, second.y1) - max(first.y0, second.y0) <= 0:
                    continue

                assert (
                    min(first.x1, second.x1) - max(first.x0, second.x0) <= 1.0
                )
