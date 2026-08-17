"""A submodule for visualising emission model networks.

An EmissionModel describes a network of models, connected by the operations
which turn one emission into another. This submodule contains the machinery for
drawing that network, which is the only practical way to understand a model of
any size.

Example usage::

    model.plot_emission_tree()

Note that EmissionModel.plot_emission_tree is a thin wrapper around the
function of the same name here, so this module does not usually need to be
imported directly.
"""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from synthesizer.emission_models.base_model import EmissionModel


def _get_tree_levels(model_tree, root):
    """Get the levels of the tree.

    Args:
        model_tree (EmissionModel):
            The model whose tree is being drawn.
        root (str):
            The root of the tree to get the levels for.

    Returns:
        levels (dict):
            The levels of the models in the tree.
        links (dict):
            The links between models.
        extract_labels (set):
            The labels of models that are extracting.
        masked_labels (list):
            The labels of models that are masked.
    """

    def _assign_levels(
        levels,
        links,
        extract_labels,
        masked_labels,
        model,
        level,
    ):
        """Recursively assign levels to the models.

        Args:
            levels (dict):
                The levels of the models.
            links (dict):
                The links between models.
            extract_labels (set):
                The labels of models that are extracting.
            masked_labels (list):
                The labels of models that are masked.
            components (dict):
                The component each model acts on.
            model (EmissionModel):
                The model to assign the level to.
            level (int):
                The level to assign to the model.

        Returns:
            levels (dict):
                The levels of the models.
            links (dict):
                The links between models.
            extract_labels (set):
                The labels of models that are extracting.
            masked_labels (list):
                The labels of models that are masked.
            components (dict):
                The component each model acts on.
        """
        # Get the model label
        label = model.label

        # Assign the level
        levels[model.label] = max(levels.get(model.label, level), level)

        # Define the links
        if model._is_transforming:
            links.setdefault(label, []).append(
                (
                    model.apply_to.label
                    if isinstance(model.apply_to, EmissionModel)
                    else model.apply_to,
                    "--",
                )
            )
        if model._is_combining:
            links.setdefault(label, []).extend(
                [
                    (child.label, "-")
                    if isinstance(child, EmissionModel)
                    else (child, "-")
                    for child in model._combine
                ]
            )
        if model._is_generating:
            if model._scaler_model is not None:
                links.setdefault(label, []).append(
                    (
                        model._scaler_model.label
                        if isinstance(model._scaler_model, EmissionModel)
                        else model._scaler_model,
                        "dotted",
                    )
                )
            if model._energy_balance_models is not None:
                intrinsic, attenuated = model._energy_balance_models
                if intrinsic is not None:
                    links.setdefault(label, []).append(
                        (
                            intrinsic.label
                            if isinstance(intrinsic, EmissionModel)
                            else intrinsic,
                            "dotted",
                        )
                    )
                if attenuated is not None:
                    links.setdefault(label, []).append(
                        (
                            attenuated.label
                            if isinstance(attenuated, EmissionModel)
                            else attenuated,
                            "dotted",
                        )
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
    root_model = model_tree._models[root]

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


def _get_model_positions(model_tree, levels, root, ychunk=10.0, xchunk=20.0):
    """Get the position of each model in the tree.

    Args:
        model_tree (EmissionModel):
            The model whose tree is being drawn.
        levels (dict):
            The levels of the models in the tree.
        root (str):
            The root of the tree to get the positions for.
        ychunk (float):
            The vertical spacing between levels.
        xchunk (float):
            The horizontal spacing between models.

    Returns:
        pos (dict):
            The position of each model in the tree.
    """

    def _get_parent_pos(pos, model):
        """Get the position of the parent/s of a model.

        Args:
            pos (dict):
                The position of each model in the tree.
            model (EmissionModel):
                The model to get the parent position for.

        Returns:
            x (float):
                The x position of the parent.
        """
        # Get the parents
        parents = [
            parent.label for parent in model._parents if parent.label in pos
        ]
        if len(set(parents)) == 0:
            return 0.0
        elif len(set(parents)) == 1:
            return pos[parents[0]][0]

        return np.mean([pos[parent][0] for parent in set(parents)])

    def _get_child_pos(x, pos, children, level, xchunk):
        """Get the position of the children of a model.

        Args:
            x (float):
                The x position of the parent.
            pos (dict):
                The position of each model in the tree.
            children (list):
                The children of the model.
            level (int):
                The level of the children.
            xchunk (float):
                The horizontal spacing between models.

        Returns:
            pos (dict):
                The position of each model in the tree.
        """
        # Get the start x
        start_x = x - (xchunk * (len(children) - 1) / 2.0)
        for child in children:
            pos[child] = (start_x, level * ychunk)
            start_x += xchunk
        return pos

    def _get_level_pos(pos, level, levels, xchunk, ychunk):
        """Get the position of the models in a level.

        Args:
            pos (dict):
                The position of each model in the tree.
            level (int):
                The level to get the position for.
            levels (dict):
                The levels of the models in the tree.
            xchunk (float):
                The horizontal spacing between models.
            ychunk (float):
                The vertical spacing between levels.

        Returns:
            pos (dict):
                The position of each model in the tree.
        """
        # Get the models in this level
        models = levels.get(level, [])

        # Get the position of the parents
        parent_pos = [
            _get_parent_pos(pos, model_tree._models[model]) for model in models
        ]

        # Sort models by parent_pos
        models = [
            model
            for _, model in sorted(zip(parent_pos, models), key=lambda x: x[0])
        ]

        # Get the parents
        parents = []
        for model in models:
            parents.extend(
                [
                    parent.label
                    for parent in model_tree._models[model]._parents
                    if parent.label in pos
                ]
            )

        # If we only have one parent for this level then we can assign
        # the position based on the parent
        if len(set(parents)) == 1:
            x = _get_parent_pos(pos, model_tree._models[models[0]])
            pos = _get_child_pos(x, pos, models, level, xchunk)
        else:
            # Get the position of the first model
            x = -xchunk * (len(models) - 1) / 2.0

            # Assign the positions
            xs = []
            for model in models:
                pos[model] = (x, level * ychunk)
                xs.append(x)
                x += xchunk

        # Recurse
        if level + 1 in levels:
            pos = _get_level_pos(pos, level + 1, levels, xchunk, ychunk)

        return pos

    return _get_level_pos({root: (0.0, 0.0)}, 1, levels, xchunk, ychunk)


def plot_emission_tree(
    model_tree,
    root=None,
    show=True,
    fontsize=10,
    figsize=(6, 6),
):
    """Plot the tree defining the spectra.

    Args:
        model_tree (EmissionModel):
            The model whose tree is being drawn.
        root (str):
            If not None this defines the root of a sub tree to plot.
        show (bool):
            Whether to show the plot.
        fontsize (int):
            The fontsize to use for the labels.
        figsize (tuple):
            The size of the figure to plot (width, height).

    Returns:
        fig (matplotlib.figure.Figure):
            The figure containing the plot.
        ax (matplotlib.axes.Axes):
            The axis containing the plot.
    """
    # Get the tree levels
    levels, links, extract_labels, masked_labels = _get_tree_levels(
        model_tree, root if root is not None else model_tree.label
    )

    # Get the postion of each node
    pos = _get_model_positions(
        model_tree, levels, root=root if root is not None else model_tree.label
    )

    # Keep track of which components are included
    components = set()

    # We need a flag for the for whether any models are discarded so we
    # know whether to include it in the legend
    some_discarded = False

    # Define a flag for whether there are per_particle models
    some_per_particle = False

    # Plot the tree using Matplotlib
    fig, ax = plt.subplots(figsize=figsize)

    # Draw nodes with different styles if they are masked
    for node, (x, y) in pos.items():
        components.add(model_tree[node].emitter)
        if model_tree[node].emitter == "stellar":
            color = "gold"
        elif model_tree[node].emitter == "blackhole":
            color = "royalblue"
        else:
            color = "forestgreen"

        # If the model isn't saved apply some transparency
        if not model_tree[node].save:
            alpha = 0.7
            some_discarded = True
        else:
            alpha = 1.0
        text = ax.text(
            x,
            -y,  # Invert y-axis for bottom-to-top
            node,
            ha="center",
            va="center",
            bbox=dict(
                facecolor=color,
                edgecolor="black",
                boxstyle="round,pad=0.5"
                if node not in extract_labels
                else "square,pad=0.5",
                alpha=alpha,
            ),
            fontsize=fontsize,
            zorder=1,
        )

        # If we have a per particle model overlay a hatched box. To make]
        # this readable we need to overlay a box with a transparent face
        if model_tree[node].per_particle:
            some_per_particle = True
            ax.text(
                x,
                -y,  # Invert y-axis for bottom-to-top
                node,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor=color,
                    edgecolor="black",
                    boxstyle="round,pad=0.5"
                    if node not in extract_labels
                    else "square,pad=0.5",
                    hatch="//",
                    alpha=0.3,
                ),
                fontsize=fontsize,
                alpha=alpha,
                zorder=2,
            )

        # Used a dashed outline for masked nodes
        bbox = text.get_bbox_patch()
        if node in masked_labels:
            bbox.set_linestyle("dashed")

    # Draw edges with different styles based on link type
    linestyles = set()
    for source, targets in links.items():
        for target, linestyle in targets:
            if target is None:
                continue
            if target not in pos or source not in pos:
                continue
            linestyles.add(linestyle)
            sx, sy = pos[source]
            tx, ty = pos[target]
            ax.plot(
                [sx, tx],
                [-sy, -ty],  # Invert y-axis for bottom-to-top
                linestyle=linestyle,
                color="black",
                lw=1,
                zorder=0,
            )

    # Create legend elements
    handles = []
    if "--" in linestyles:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="dashed",
                label="Transformed",
            )
        )
    if "-" in linestyles:
        handles.append(
            mlines.Line2D(
                [], [], color="black", linestyle="solid", label="Combined"
            )
        )
    if "dotted" in linestyles:
        handles.append(
            mlines.Line2D(
                [],
                [],
                color="black",
                linestyle="dotted",
                label="Generator Scaling",
            )
        )

    # Include a component legend element
    if "stellar" in components:
        handles.append(
            mpatches.FancyBboxPatch(
                (0.1, 0.1),
                width=0.5,
                height=0.1,
                facecolor="gold",
                edgecolor="black",
                label="Stellar",
                boxstyle="round,pad=0.5",
            )
        )
    if "blackhole" in components:
        handles.append(
            mpatches.FancyBboxPatch(
                (0.1, 0.1),
                width=0.5,
                height=0.1,
                facecolor="royalblue",
                edgecolor="black",
                label="Black Hole",
                boxstyle="round,pad=0.5",
            )
        )
    if "galaxy" in components:
        handles.append(
            mpatches.FancyBboxPatch(
                (0.1, 0.1),
                width=0.5,
                height=0.1,
                facecolor="forestgreen",
                edgecolor="black",
                label="Galaxy",
                boxstyle="round,pad=0.5",
            )
        )

    # Include a masked legend element if we have masked nodes
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

    # Include a transparent legend element for non-saved nodes if needed
    if some_discarded:
        handles.append(
            mpatches.FancyBboxPatch(
                (0.1, 0.1),
                width=0.5,
                height=0.1,
                facecolor="grey",
                edgecolor="black",
                label="Saved",
                boxstyle="round,pad=0.5",
            )
        )
        handles.append(
            mpatches.FancyBboxPatch(
                (0.1, 0.1),
                width=0.5,
                height=0.1,
                facecolor="grey",
                edgecolor="black",
                label="Discarded",
                alpha=0.6,
                boxstyle="round,pad=0.5",
            )
        )

    # If we have per particle models include them in the legend
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

    # Add legend to the bottom of the plot
    ax.legend(
        handles=handles,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 1.1),
        ncol=6,
    )

    ax.axis("off")
    if show:
        plt.show()

    return fig, ax
