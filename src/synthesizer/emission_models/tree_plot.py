"""A submodule for plotting the tree structure of emission models."""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np


def get_tree_levels(models_dict, root):
    """Get the levels of the tree.

    Args:
        models_dict (dict): Dictionary mapping model labels to model objects
        root (str): The root of the tree to get the levels for

    Returns:
        tuple: (levels, links, extract_labels, masked_labels)
            - levels (dict): Dictionary mapping levels to lists of model labels
            - links (dict): Dictionary mapping source labels to lists of
                (target, style) tuples
            - extract_labels (set): Set of labels for models that are
                extracting
            - masked_labels (list): List of labels for models that are masked
    """

    def _assign_levels(
        levels,
        links,
        extract_labels,
        masked_labels,
        model,
        level,
    ):
        """Recursively assign levels to the models."""
        # Get the model label
        label = model.label

        # Assign the level
        levels[model.label] = max(levels.get(model.label, level), level)

        # Define the links
        if model._is_transforming:
            links.setdefault(label, []).append((model.apply_to.label, "--"))
        if model._is_combining:
            links.setdefault(label, []).extend(
                [(child.label, "-") for child in model._combine]
            )
        if model._is_dust_emitting or model._is_generating:
            links.setdefault(label, []).extend(
                [
                    (
                        model._lum_intrinsic_model.label
                        if model._lum_intrinsic_model is not None
                        else None,
                        "dotted",
                    ),
                    (
                        model._lum_attenuated_model.label
                        if model._lum_attenuated_model is not None
                        else None,
                        "dotted",
                    ),
                ]
            )

        if model._is_masked:
            masked_labels.append(label)
        if model._is_extracting:
            extract_labels.add(label)

        # Recurse
        for child in model._children:
            (
                levels,
                links,
                extract_labels,
                masked_labels,
            ) = _assign_levels(
                levels,
                links,
                extract_labels,
                masked_labels,
                child,
                level + 1,
            )

        return levels, links, extract_labels, masked_labels

    # Get the root model
    root_model = models_dict[root]

    # Recursively assign levels
    (
        model_levels,
        links,
        extract_labels,
        masked_labels,
    ) = _assign_levels({}, {}, set(), [], root_model, 0)

    # Unpack the levels
    levels = {}
    for label, level in model_levels.items():
        levels.setdefault(level, []).append(label)

    return levels, links, extract_labels, masked_labels


def _get_adaptive_spacing(levels, links):
    """Calculate adaptive spacing based on tree structure.

    Args:
        levels (dict): Dictionary mapping levels to lists of model labels
        links (dict): Dictionary of links between models

    Returns:
        tuple: (xchunk, ychunk) horizontal and vertical spacing values
    """
    max_level_width = max(len(models) for models in levels.values())
    max_label_length = max(
        len(label)
        for level_models in levels.values()
        for label in level_models
    )

    # Base spacing on content and tree width
    xchunk = max(15.0, max_label_length * 1.2 + 5.0)
    ychunk = max(8.0, 10.0 + len(levels) * 1.5)

    # Adjust for very wide trees
    if max_level_width > 5:
        xchunk *= 1.2

    return xchunk, ychunk


def _resolve_overlaps(pos, min_distance=None):
    """Resolve overlapping nodes by shifting them apart.

    Uses an iterative approach to detect and resolve overlaps between nodes
    at the same level and ensures minimum vertical spacing between levels.

    Args:
        pos (dict): Dictionary mapping node labels to (x, y) positions
        min_distance (float, optional): Minimum distance between nodes.
            If None, calculated based on average label length

    Returns:
        dict: Updated positions with overlaps resolved
    """
    if min_distance is None:
        # Calculate minimum distance based on average label length
        avg_label_length = np.mean([len(label) for label in pos.keys()])
        min_distance = max(20.0, avg_label_length * 1.5)

    positions = pos.copy()
    max_iterations = 50

    for iteration in range(max_iterations):
        changed = False

        # Group nodes by level for more efficient checking
        levels = {}
        for node, (x, y) in positions.items():
            if y not in levels:
                levels[y] = []
            levels[y].append((node, x))

        # Check overlaps within each level and between adjacent levels
        for level_y, nodes in levels.items():
            # Sort by x position
            nodes.sort(key=lambda x: x[1])

            # Resolve horizontal overlaps within level
            for i in range(len(nodes) - 1):
                node1, x1 = nodes[i]
                node2, x2 = nodes[i + 1]

                distance = abs(x2 - x1)
                if distance < min_distance:
                    # Move nodes apart
                    shift = (min_distance - distance) / 2 + 1.0
                    positions[node1] = (x1 - shift, level_y)
                    positions[node2] = (x2 + shift, level_y)
                    changed = True

        # Check vertical overlaps between levels
        level_ys = sorted(levels.keys())
        for i in range(len(level_ys) - 1):
            curr_level = level_ys[i]
            next_level = level_ys[i + 1]

            if abs(next_level - curr_level) < min_distance * 0.6:
                # Levels too close vertically - this shouldn't happen with
                # proper ychunk but handle edge cases
                for node, (x, y) in positions.items():
                    if y == next_level:
                        positions[node] = (x, curr_level + min_distance * 0.8)
                changed = True

        if not changed:
            break

    return positions


def _apply_force_layout(pos, links, iterations=30):
    """Apply light force-directed adjustments to improve layout.

    Uses a simplified force-directed algorithm with repulsion forces to prevent
    overlaps and light attraction forces to maintain hierarchical structure.
    Vertical forces are reduced to preserve the tree hierarchy.

    Args:
        pos (dict): Dictionary mapping node labels to (x, y) positions
        links (dict): Dictionary of links between models for attraction forces
        iterations (int): Number of force simulation iterations to run

    Returns:
        dict: Updated positions after force-directed adjustment
    """
    positions = {node: list(coord) for node, coord in pos.items()}

    for iteration in range(iterations):
        forces = {node: [0.0, 0.0] for node in positions}

        # Light repulsion forces to prevent overlaps
        for node1, (x1, y1) in positions.items():
            for node2, (x2, y2) in positions.items():
                if node1 != node2:
                    dx = x1 - x2
                    dy = y1 - y2
                    dist = max(1.0, (dx**2 + dy**2) ** 0.5)

                    # Only apply repulsion if nodes are very close
                    if dist < 35:
                        force_magnitude = 200 / (dist**2 + 1)
                        forces[node1][0] += force_magnitude * dx / dist
                        # Reduce vertical forces to maintain hierarchy
                        forces[node1][1] += force_magnitude * dy / dist * 0.3

        # Light attraction forces to maintain structure
        for source, targets in links.items():
            if source in positions:
                for target, _ in targets:
                    if target and target in positions:
                        sx, sy = positions[source]
                        tx, ty = positions[target]
                        dx = tx - sx
                        dy = ty - sy
                        dist = max(1.0, (dx**2 + dy**2) ** 0.5)

                        # Very light attraction
                        force_magnitude = 0.05 * dist
                        forces[source][0] += force_magnitude * dx / dist * 0.5
                        forces[target][0] -= force_magnitude * dx / dist * 0.5

        # Apply forces with strong damping
        damping = 0.3 - (iteration / iterations) * 0.2  # Decrease over time
        for node in positions:
            positions[node][0] += forces[node][0] * damping
            positions[node][1] += forces[node][1] * damping

    # Convert back to tuples
    return {node: tuple(coord) for node, coord in positions.items()}


def _get_improved_level_pos(
    pos, level, levels, models_dict, links, xchunk, ychunk
):
    """Improved level positioning that centers children under parents.

    Groups models by their parent sets and positions children centered under
    their parents. Handles orphaned nodes and prevents collisions with existing
    nodes. Recursively processes all levels in the tree.

    Args:
        pos (dict): Dictionary mapping node labels to (x, y) positions
        level (int): Current level being processed
        levels (dict): Dictionary mapping levels to lists of model labels
        models_dict (dict): Dictionary mapping model labels to model objects
        links (dict): Dictionary of links between models
        xchunk (float): Horizontal spacing between models
        ychunk (float): Vertical spacing between levels

    Returns:
        dict: Updated positions for all nodes at current and subsequent levels
    """
    if level not in levels:
        return pos

    models = levels[level]

    # Group models by their parent sets
    parent_groups = {}
    for model in models:
        parents = tuple(
            sorted(
                [
                    parent.label
                    for parent in models_dict[model]._parents
                    if parent.label in pos
                ]
            )
        )
        if parents not in parent_groups:
            parent_groups[parents] = []
        parent_groups[parents].append(model)

    # Sort groups to process those with parents first
    sorted_groups = sorted(
        parent_groups.items(), key=lambda x: (len(x[0]) == 0, x[0])
    )

    occupied_positions = set()
    group_positions = {}

    for parents, children in sorted_groups:
        if parents:
            # Center children under their parents
            parent_positions = [pos[parent][0] for parent in parents]
            center_x = np.mean(parent_positions)

            # Calculate span needed for children
            total_width = (len(children) - 1) * xchunk
            start_x = center_x - total_width / 2.0
        else:
            # No parents - find free space
            if occupied_positions:
                start_x = max(occupied_positions) + xchunk * 1.5
            else:
                start_x = -(len(children) - 1) * xchunk / 2.0

        # Position children and avoid collisions
        child_positions = []
        for i, child in enumerate(children):
            x = start_x + i * xchunk

            # Ensure minimum spacing from existing nodes
            attempts = 0
            while (
                any(
                    abs(x - existing) < xchunk * 0.8
                    for existing in occupied_positions
                )
                and attempts < 20
            ):
                x += xchunk * 0.3
                attempts += 1

            pos[child] = (x, level * ychunk)
            occupied_positions.add(x)
            child_positions.append(x)

        group_positions[parents] = child_positions

    # Recurse to next level
    if level + 1 in levels:
        pos = _get_improved_level_pos(
            pos, level + 1, levels, models_dict, links, xchunk, ychunk
        )

    return pos


def get_model_positions(
    levels, models_dict, root, links, ychunk=10.0, xchunk=20.0
):
    """Get optimized positions for each model in the tree.

    Args:
        levels (dict): Dictionary mapping levels to lists of model labels
        models_dict (dict): Dictionary mapping model labels to model objects
        root (str): Root node label
        links (dict): Dictionary of links between models
        ychunk (float): Base vertical spacing between levels
        xchunk (float): Base horizontal spacing between models

    Returns:
        dict: Dictionary mapping model labels to (x, y) positions
    """
    # Calculate adaptive spacing
    adaptive_xchunk, adaptive_ychunk = _get_adaptive_spacing(levels, links)

    # Use adaptive spacing, but allow override if explicitly set
    final_xchunk = (
        max(xchunk, adaptive_xchunk) if xchunk != 20.0 else adaptive_xchunk
    )
    final_ychunk = (
        max(ychunk, adaptive_ychunk) if ychunk != 10.0 else adaptive_ychunk
    )

    # Get initial positions using improved algorithm
    pos = _get_improved_level_pos(
        {root: (0.0, 0.0)},
        1,
        levels,
        models_dict,
        links,
        final_xchunk,
        final_ychunk,
    )

    # Resolve overlaps
    pos = _resolve_overlaps(pos)

    # Apply light force-based refinement for larger or complex trees
    if len(pos) > 8 or len(levels) > 4:
        pos = _apply_force_layout(pos, links, iterations=25)
        # Final overlap resolution after force layout
        pos = _resolve_overlaps(pos)

    return pos


def _calculate_figure_size(figsize, pos):
    """Calculate optimal figure size based on tree dimensions.

    Auto-adjusts figure size when using default dimensions by analyzing
    the span of node positions and scaling appropriately.

    Args:
        figsize (tuple): Requested figure size (width, height)
        pos (dict): Dictionary mapping node labels to (x, y) positions

    Returns:
        tuple: Optimal figure size (width, height)
    """
    if figsize != (8, 6) or not pos:
        return figsize

    x_span = (
        max(pos.values(), key=lambda p: p[0])[0]
        - min(pos.values(), key=lambda p: p[0])[0]
    )
    y_span = (
        max(pos.values(), key=lambda p: p[1])[1]
        - min(pos.values(), key=lambda p: p[1])[1]
    )

    width = max(8, x_span / 25 + 4)
    height = max(6, y_span / 15 + 3)
    return (width, height)


def _get_edge_style(linestyle):
    """Map internal linestyle to matplotlib style and alpha.

    Converts internal linestyle codes to matplotlib-compatible styles
    with appropriate alpha values for visual hierarchy.

    Args:
        linestyle (str): Internal linestyle code ("--", "-", "dotted")

    Returns:
        tuple: (matplotlib_style, alpha_value)
    """
    style_map = {
        "--": ("dashed", 0.8),
        "-": ("solid", 0.9),
        "dotted": ("dotted", 0.7),
    }
    return style_map.get(linestyle, ("dotted", 0.7))


def _draw_edges(ax, links, pos, edge_width):
    """Draw all edges between nodes.

    Iterates through all links and draws edges with appropriate styling.
    Skips invalid links (missing nodes, None targets). Inverts y-coordinates
    for bottom-to-top tree display.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes to draw on
        links (dict): Dictionary mapping source nodes to lists of
            (target, style) tuples
        pos (dict): Dictionary mapping node labels to (x, y) positions
        edge_width (float): Width of edge lines

    Returns:
        set: Set of linestyles used (for legend creation)
    """
    linestyles = set()

    for source, targets in links.items():
        for target, linestyle in targets:
            if target is None or target not in pos or source not in pos:
                continue

            linestyles.add(linestyle)
            sx, sy = pos[source]
            tx, ty = pos[target]

            plot_style, alpha = _get_edge_style(linestyle)

            ax.plot(
                [sx, tx],
                [-sy, -ty],  # Invert y-axis for bottom-to-top
                linestyle=plot_style,
                color="black",
                lw=edge_width,
                alpha=alpha,
                zorder=0,
            )

    return linestyles


def _get_node_color(emitter_type):
    """Get color for node based on emitter type.

    Maps emitter types to consistent colors used throughout the visualization.

    Args:
        emitter_type (str): Type of emitter ("stellar", "blackhole", or other)

    Returns:
        str: Matplotlib color name for the emitter type
    """
    color_map = {"stellar": "gold", "blackhole": "royalblue"}
    return color_map.get(emitter_type, "forestgreen")


def _draw_single_node(
    ax,
    node,
    pos,
    model,
    extract_labels,
    masked_labels,
    fontsize,
    node_size_factor,
):
    """Draw a single node with all its styling.

    Handles the complete rendering of a single node including:
    - Base node with color and transparency based on model properties
    - Per-particle hatching overlay if applicable
    - Masked node styling (dashed outline)
    - Box style selection based on extract status

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes to draw on
        node (str): Node label
        pos (dict): Dictionary mapping node labels to (x, y) positions
        model: Model object with emitter, save, and per_particle attributes
        extract_labels (set): Set of labels for extracting models
        masked_labels (list): List of labels for masked models
        fontsize (int): Base font size for node text
        node_size_factor (float): Scale factor for node text size

    Returns:
        tuple: (emitter_type, is_discarded, is_per_particle) for tracking
            statistics
    """
    x, y = pos[node]
    color = _get_node_color(model.emitter)
    alpha = 0.7 if not model.save else 1.0
    box_style = (
        "round,pad=0.6" if node not in extract_labels else "square,pad=0.6"
    )

    # Main node
    text = ax.text(
        x,
        -y,  # Invert y-axis for bottom-to-top
        node,
        ha="center",
        va="center",
        bbox=dict(
            facecolor=color,
            edgecolor="black",
            boxstyle=box_style,
            alpha=alpha,
            linewidth=1.5,
        ),
        fontsize=fontsize * node_size_factor,
        zorder=2,
    )

    # Per particle hatching overlay
    if model.per_particle:
        ax.text(
            x,
            -y,
            node,
            ha="center",
            va="center",
            bbox=dict(
                facecolor="none",
                edgecolor="black",
                boxstyle=box_style,
                hatch="//",
                alpha=0.4,
                linewidth=1.5,
            ),
            fontsize=fontsize * node_size_factor,
            zorder=3,
        )

    # Masked node styling
    if node in masked_labels:
        bbox = text.get_bbox_patch()
        bbox.set_linestyle("dashed")
        bbox.set_linewidth(2.0)

    return model.emitter, not model.save, model.per_particle


def _draw_nodes(
    ax,
    pos,
    models_dict,
    extract_labels,
    masked_labels,
    fontsize,
    node_size_factor,
):
    """Draw all nodes and return component information.

    Iterates through all node positions and draws each node with appropriate
    styling. Collects statistics about components, discarded models, and
    per-particle models for legend creation.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes to draw on
        pos (dict): Dictionary mapping node labels to (x, y) positions
        models_dict (dict): Dictionary mapping model labels to model objects
        extract_labels (set): Set of labels for extracting models
        masked_labels (list): List of labels for masked models
        fontsize (int): Base font size for node text
        node_size_factor (float): Scale factor for node text size

    Returns:
        tuple: (components, some_discarded, some_per_particle) for legend
          creation:
            - components (set): Set of emitter types present in the tree
            - some_discarded (bool): Whether any models are not saved
            - some_per_particle (bool): Whether any models use per-particle
              mode
    """
    components = set()
    some_discarded = False
    some_per_particle = False

    for node, node_pos in pos.items():
        model = models_dict[node]
        emitter, is_discarded, is_per_particle = _draw_single_node(
            ax,
            node,
            pos,
            model,
            extract_labels,
            masked_labels,
            fontsize,
            node_size_factor,
        )

        components.add(emitter)
        if is_discarded:
            some_discarded = True
        if is_per_particle:
            some_per_particle = True

    return components, some_discarded, some_per_particle


def _create_edge_legend_handles(linestyles, edge_width):
    """Create legend handles for edge styles.

    Creates matplotlib legend handles for each edge style present in the plot.
    Maps internal linestyle codes to descriptive labels.

    Args:
        linestyles (set): Set of linestyle codes used in the plot
        edge_width (float): Width of edge lines for consistent legend styling

    Returns:
        list: List of matplotlib legend handles for edge styles
    """
    handles = []

    edge_labels = {
        "--": "Transformed",
        "-": "Combined",
        "dotted": "Dust Luminosity",
    }

    for style, label in edge_labels.items():
        if style in linestyles:
            plot_style = "dashed" if style == "--" else style
            handles.append(
                mlines.Line2D(
                    [],
                    [],
                    color="black",
                    linestyle=plot_style,
                    label=label,
                    linewidth=edge_width,
                )
            )

    return handles


def _create_component_legend_handles(components):
    """Create legend handles for component types.

    Creates colored box legend handles for each emitter component type
    present in the tree (stellar, blackhole, galaxy).

    Args:
        components (set): Set of emitter types present in the tree

    Returns:
        list: List of matplotlib legend handles for component types
    """
    handles = []

    component_info = {
        "stellar": ("gold", "Stellar"),
        "blackhole": ("royalblue", "Black Hole"),
        "galaxy": ("forestgreen", "Galaxy"),
    }

    for component, (color, label) in component_info.items():
        if component in components:
            handles.append(
                mpatches.FancyBboxPatch(
                    (0.1, 0.1),
                    width=0.5,
                    height=0.1,
                    facecolor=color,
                    edgecolor="black",
                    label=label,
                    boxstyle="round,pad=0.5",
                )
            )

    return handles


def _create_special_legend_handles(
    masked_labels, some_discarded, some_per_particle
):
    """Create legend handles for special node types.

    Creates legend handles for special node styling including:
    - Masked nodes (dashed outline)
    - Saved vs discarded models (transparency)
    - Per-particle models (hatched overlay)
    Only creates handles for features that are actually present in the plot.

    Args:
        masked_labels (list): List of labels for masked models
        some_discarded (bool): Whether any models are not saved
        some_per_particle (bool): Whether any models use per-particle mode

    Returns:
        list: List of matplotlib legend handles for special node types
    """
    handles = []

    if len(masked_labels) > 0:
        handles.append(
            mpatches.FancyBboxPatch(
                (0.1, 0.1),
                width=0.5,
                height=0.1,
                facecolor="none",
                edgecolor="black",
                label="Masked",
                linestyle="dashed",
                boxstyle="round,pad=0.5",
            )
        )

    if some_discarded:
        handles.extend(
            [
                mpatches.FancyBboxPatch(
                    (0.1, 0.1),
                    width=0.5,
                    height=0.1,
                    facecolor="grey",
                    edgecolor="black",
                    label="Saved",
                    boxstyle="round,pad=0.5",
                ),
                mpatches.FancyBboxPatch(
                    (0.1, 0.1),
                    width=0.5,
                    height=0.1,
                    facecolor="grey",
                    edgecolor="black",
                    label="Discarded",
                    alpha=0.6,
                    boxstyle="round,pad=0.5",
                ),
            ]
        )

    if some_per_particle:
        handles.append(
            mpatches.FancyBboxPatch(
                (0.1, 0.1),
                width=0.5,
                height=0.1,
                facecolor="none",
                edgecolor="black",
                label="Per Particle",
                hatch="//",
                alpha=0.3,
                boxstyle="round,pad=0.5",
            )
        )

    return handles


def _create_legend(
    ax,
    linestyles,
    components,
    masked_labels,
    some_discarded,
    some_per_particle,
    edge_width,
    fontsize,
):
    """Create and add legend to the plot.

    Coordinates creation of all legend elements and adds them to the plot
    with appropriate positioning and formatting. Combines edge styles,
    component types, and special node types into a single legend.

    Args:
        ax (matplotlib.axes.Axes): Matplotlib axes to add legend to
        linestyles (set): Set of linestyle codes used in the plot
        components (set): Set of emitter types present in the tree
        masked_labels (list): List of labels for masked models
        some_discarded (bool): Whether any models are not saved
        some_per_particle (bool): Whether any models use per-particle mode
        edge_width (float): Width of edge lines for consistent styling
        fontsize (int): Base font size for legend text
    """
    handles = []

    # Add all legend components
    handles.extend(_create_edge_legend_handles(linestyles, edge_width))
    handles.extend(_create_component_legend_handles(components))
    handles.extend(
        _create_special_legend_handles(
            masked_labels, some_discarded, some_per_particle
        )
    )

    if handles:
        ncols = min(len(handles), 4)  # Limit columns for readability
        ax.legend(
            handles=handles,
            loc="upper center",
            frameon=False,
            bbox_to_anchor=(0.5, 1.05),
            ncol=ncols,
            fontsize=max(8, fontsize - 1),
        )


def _plot_emission_tree(
    models_dict,
    root,
    show=True,
    fontsize=10,
    figsize=(8, 6),
    node_size_factor=1.0,
    edge_width=1.2,
):
    """Plot the tree defining the spectra with improved layout and styling.

    Args:
        models_dict (dict): Dictionary mapping model labels to model objects
        root (str): Root node label
        show (bool): Whether to show the plot
        fontsize (int): Font size for node labels
        figsize (tuple): Figure size (width, height)
        node_size_factor (float): Scale factor for node text size
        edge_width (float): Width of edges

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects
    """
    # Get the tree structure
    levels, links, extract_labels, masked_labels = get_tree_levels(
        models_dict, root
    )

    # Get optimized positions
    pos = get_model_positions(levels, models_dict, root, links)

    # Calculate optimal figure size
    figsize = _calculate_figure_size(figsize, pos)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Draw edges first (behind nodes)
    linestyles = _draw_edges(ax, links, pos, edge_width)

    # Draw nodes and collect component information
    components, some_discarded, some_per_particle = _draw_nodes(
        ax,
        pos,
        models_dict,
        extract_labels,
        masked_labels,
        fontsize,
        node_size_factor,
    )

    # Create and add legend
    _create_legend(
        ax,
        linestyles,
        components,
        masked_labels,
        some_discarded,
        some_per_particle,
        edge_width,
        fontsize,
    )

    # Final plot styling
    ax.axis("off")
    ax.set_aspect("equal", adjustable="box")

    if show:
        plt.tight_layout()
        plt.show()

    return fig, ax
