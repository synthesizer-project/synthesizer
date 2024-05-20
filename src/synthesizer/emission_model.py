"""A module defining the emission model from which spectra are constructed.

Generating spectra involves the following steps:
1. Extraction from a Grid.
2. Attenuation due to dust in the ISM/nebular.
3. Masking for different types of emission.
4. Combination of different types of emission.

An emission model defines the parameters necessary to perform these steps and
gives an interface for simply defining the construction of complex spectra.
"""

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from unyt import unyt_quantity

from synthesizer import exceptions


class EmissionModel:
    """A class to define the construction of a spectra from a grid."""

    def __init__(
        self,
        grid,
        label,
        extract=None,
        combine=None,
        apply_dust_to=None,
        dust_curve=None,
        tau_v=None,
        dust_emission_model=None,
        mask_attr=None,
        mask_thresh=None,
        mask_op=None,
        fesc=None,
    ):
        """
        Initialise the emission model.

        Each instance of an emission model describes a single step in the
        process of constructing a spectra. These different steps can be:
        - Extracting a spectra from a grid. (extract must be set)
        - Combining multiple spectra. (combine must be set with a list/tuple
          of child emission models).
        - Applying a dust curve to the spectra. (dust_curve and apply_dust_to
          must be set)

        Within any of these steps a mask can be applied to the spectra to
        remove certain elements (e.g. if you want to eliminate particles).

        """
        # Attach the grid
        self.grid = grid

        # What is the key for the spectra that will be produced?
        self.label = label

        # What base key will we be extracting? (can be None if we're applying
        # a dust curve to an existing spectra)
        self.extract = extract

        # Attach the dust curve
        self.dust_curve = dust_curve

        # Attach the optical depth
        self.tau_v = tau_v

        # Attach the dust emission model
        self.dust_emission_model = dust_emission_model

        # Initialise the container for combined spectra we'll add together
        self.combine = list(combine) if combine is not None else None

        # If we have a dust_curve, we need to know where to apply it
        self.apply_dust_to = apply_dust_to

        # Attach the escape fraction
        self.fesc = fesc

        # Define the container which will hold mask information
        self.masks = []

        # If we have been given a mask, add it
        if (
            mask_attr is not None
            and mask_thresh is not None
            and mask_op is not None
        ):
            self.masks.append(
                {"attr": mask_attr, "thresh": mask_thresh, "op": mask_op}
            )
        else:
            # Ensure we haven't been given incomplete mask arguments
            if any(
                arg is not None for arg in [mask_attr, mask_thresh, mask_op]
            ):
                raise exceptions.InconsistentArguments(
                    "For a mask mask_attr, mask_thresh, and mask_op "
                    "must be passed."
                )

        # Check everything makes sense
        self._check_args()

        # Intialise and populate a dictionary to hold all child models for
        # traversing the tree
        self._children = []
        for child in self.combine:
            self._children.append(child)
        if self.apply_dust_to is not None:
            self._children.append(self.apply_dust_to)

        # Initilaise a container for parent models
        self.parents = []

        # Attach the parent model to the children
        for child in self._children:
            child.parents.append(self)

        # Define the private containers we'll unpack everything into. These
        # are dictionaries of the form {<result_label>: <operation props>}
        self._extract_keys = {}
        self._dust_attenuation = {}
        self._dust_emission = {}
        self._combine_keys = {}
        self._mask_keys = {}
        self._bottom_to_top = []
        self._models = {}
        self._unpack_model_recursively(self)

    def _check_args(self):
        """Ensure we have a valid model."""
        if self.extract is not None and any(
            arg is not None
            for arg in [self.dust_curve, self.combine, self.apply_dust_to]
        ):
            raise exceptions.InconsistentArguments(
                "Cannot extract and apply dust or combine spectra. "
                "Can only do one operation at a time."
            )
        if self.dust_curve is not None and self.apply_dust_to is None:
            raise exceptions.InconsistentArguments(
                "Must specify where to apply dust curve."
            )
        if self.apply_dust_to is not None and self.dust_curve is None:
            raise exceptions.InconsistentArguments(
                "Must specify a dust curve to apply."
            )
        if self.dust_curve is not None and any(
            arg is not None for arg in [self.extract, self.combine]
        ):
            raise exceptions.InconsistentArguments(
                "Cannot apply dust and extract or combine spectra. "
                "Can only do one operation at a time."
            )
        if self.combine is not None and any(
            arg is not None
            for arg in [self.extract, self.dust_curve, self.apply_dust_to]
        ):
            raise exceptions.InconsistentArguments(
                "Cannot combine and extract or apply dust to spectra. "
                "Can only do one operation at a time."
            )

        # Ensure the grid contains any keys we want to extract
        if (
            self.extract is not None
            and self.extract not in self.grid.spectra.keys()
        ):
            raise exceptions.InconsistentArguments(
                f"Grid does not contain key: {self.extract}"
            )

        # Now that we've checked everything is consistent we can make an empty
        # list for combine if it is None
        if self.combine is None:
            self.combine = []

    def _unpack_model_recursively(self, model):
        """
        Traverse the model tree and collect what we will need to do.

        Args:
            model (EmissionModel): The model to unpack.
        """
        # Store the model
        self._models[model.label] = model

        # If we are extracting a spectra, store the key
        if model.extract is not None:
            self._extract_keys[model.label] = model.extract

        # If we are applying a dust curve, store the key
        if model.dust_curve is not None:
            self._dust_attenuation[model.label] = (
                model.apply_dust_to,
                model.dust_curve,
            )

        # If we are applying a dust emission model, store the key
        if model.dust_emission_model is not None:
            self._dust_emission[model.label] = model.dust_emission_model

        # If we are masking, store the key
        if len(model.masks):
            self._mask_keys[model.label] = model.masks

        # If we are combining spectra, store the key
        if len(model.combine) > 0:
            self._combine_keys[model.label] = model.combine

        # Recurse over children
        for child in model._children:
            self._unpack_model_recursively(child)

        # Populate the top to bottom list but ignoring extraction since
        # we do the all at once
        if (
            model.label not in self._extract_keys
            and model.label not in self._bottom_to_top
        ):
            self._bottom_to_top.append(model.label)

    def _change_attr_recursively(self, model, attr, value, label, set_all):
        """Recurse through the emission model tree to modifty dust curve."""
        # Are we changing this attribute on everything?
        if set_all:
            # Onyl set it if the model already has the attribute set
            if getattr(model, attr, None) is not None:
                setattr(model, attr, value)

            # Recurse
            for child in model._children:
                self._recursive_change_dust_curve(
                    child, attr, value, label, set_all
                )

            # We're done here
            return 0

        # Are we looking for a specific label?
        if model.label == label:
            setattr(model, attr, value)

            # We're done here
            return 0

        # Recurse
        fail = 1
        for child in model._children:
            fail = self._change_attr_recursively(
                child, attr, value, label, set_all
            )
            if not fail:
                return fail
        return fail

    def change_dust_curve(self, dust_curve, label=None, set_all=False):
        """Swap the dust curve on this model or a child."""
        # Are we modifying everything?
        if set_all:
            if self.dust_curve is not None:
                self.dust_curve = dust_curve

            # Recurse
            for child in self._children:
                self._change_attr_recursively(
                    child,
                    "dust_curve",
                    dust_curve,
                    label,
                    set_all,
                )

            # And unpack everything to update this change
            self._unpack_model_recursively(self)
            return

        # Are we just modifying the top level dust curve?
        if label is None or label == self.label:
            self.dust_curve = dust_curve
            return

        # Ok, in that case recurse through the children
        if self._change_attr_recursively(
            self, "dust_curve", dust_curve, label, set_all
        ):
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # And unpack everything to update this change
        self._unpack_model_recursively(self)

    def change_tau_v(self, tau_v, label=None, set_all=False):
        """Swap the optical depth on this model or a child."""
        # Are we modifying everything?
        if set_all:
            if self.tau_v is not None:
                self.tau_v = tau_v

            # Recurse
            for child in self._children:
                self._change_attr_recursively(
                    child,
                    "tau_v",
                    tau_v,
                    label,
                    set_all,
                )

            # And unpack everything to update this change
            self._unpack_model_recursively(self)
            return

        # Are we just modifying the top level dust curve?
        if label is None or label == self.label:
            self.tau_v = tau_v
            return

        # Ok, in that case recurse through the children
        if self._change_attr_recursively(self, "tau_v", tau_v, label, set_all):
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # And unpack everything to update this change
        self._unpack_model_recursively(self)

    def change_fesc(self, fesc, label=None, set_all=False):
        """Swap the escape fraction on this model or a child."""
        # Are we modifying everything?
        if set_all:
            if self.fesc is not None:
                self.fesc = fesc

            # Recurse
            for child in self._children:
                self._change_attr_recursively(
                    child,
                    "fesc",
                    fesc,
                    label,
                    set_all,
                )

            # And unpack everything to update this change
            self._unpack_model_recursively(self)
            return

        # Are we just modifying the top level dust curve?
        if label is None or label == self.label:
            self.fesc = fesc
            return

        # Ok, in that case recurse through the children
        if self._change_attr_recursively(self, "fesc", fesc, label, set_all):
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # And unpack everything to update this change
        self._unpack_model_recursively(self)

    def _add_mask_recursively(self, model, attr, thresh, op, label, set_all):
        """Recursively add a mask to a model."""
        # Are we adding a mask to everything?
        if set_all:
            model.add_mask(attr, thresh, op)
            for child in model._children:
                self._add_mask_recursive(
                    child,
                    attr,
                    thresh,
                    op,
                    label,
                    set_all,
                )
            return 0

        # Are we adding a mask to this model?
        if model.label == label:
            model.add_mask(attr, thresh, op)
            return 0

        # Recurse
        fail = 1
        for child in model._children:
            fail = self._add_mask_recursively(
                child, attr, thresh, op, label, set_all
            )
            if not fail:
                return fail
        return fail

    def add_mask(self, attr, thresh, op, label=None, set_all=False):
        """
        Add a mask.

        Args:
            attr (str): The component attribute to mask on.
            thresh (unyt_quantity): The threshold for the mask.
            op (str): The operation to apply. Can be "<", ">", "<=", or ">=".
        """
        # Ensure we have units
        if not isinstance(thresh, unyt_quantity):
            raise exceptions.MissingUnits(
                "Mask threshold must be given with units."
            )

        # Ensure operation is valid
        if op not in ["<", ">", "<=", ">=", "=="]:
            raise exceptions.InconsistentArguments(
                "Invalid mask operation. Must be one of: <, >, <=, >="
            )

        # If no label has been specified we're adding a mask on this object
        if label is None:
            self.masks.append({"attr": attr, "thresh": thresh, "op": op})
        else:
            # Otherwise, we need to recurse to find the requested label
            if self._add_mask_recursively(
                self,
                attr,
                thresh,
                op,
                label,
                set_all,
            ):
                raise exceptions.InconsistentArguments(
                    f"Could not find a model with the label: {label}"
                )

        # And unpack everything to update this change
        self._unpack_model_recursively(self)

    def add_model(self, model):
        pass

    def remove_model(self, label):
        pass

    def relabel_model(self, old_label, new_label):
        pass

    def plot_emission_tree(self):
        """Plot the tree defining the spectra."""
        # Define a dictionary to hold the links (of the form
        # {<label>: [(<linked label>, <link_type>)]})
        links = {}

        # Get the attenuation links
        for label, (model, _) in self._dust_attenuation.items():
            links.setdefault(label, []).append((model.label, "attenuate"))

        # Get the combination links
        for label, models in self._combine_keys.items():
            links.setdefault(label, []).extend(
                [(model.label, "combine") for model in models]
            )

        # Get a list of which labels were masked
        masked_labels = []
        for label in self._mask_keys:
            masked_labels.append(label)

        # Extract all unique nodes
        all_nodes = set(links.keys())
        for target_list in links.values():
            for target, _ in target_list:
                all_nodes.add(target)

        # Get all extraction keys
        extract_labels = set(self._extract_keys)

        # Initialise dictionary to hold node positions and start with the
        # root
        pos = {self.label: [0.0, 0.0]}
        levels = {0: [self.label]}

        # Recursively assign positions for nodes
        def assign_pos(pos, model, levels, level, xchunk):
            # Get current position
            x, y = pos[model.label]

            # Increment y for this new layer
            new_y = y + 10

            # Compute the x width
            width = (len(model._children) - 1) * xchunk
            start_x = x - width / 2

            # Loop over children setting their position
            for i, child in enumerate(model._children):
                if child.label not in pos:
                    pos[child.label] = [start_x + i * xchunk, new_y]
                    levels.setdefault(level + 1, []).append(child.label)
                else:
                    pos[child.label][0] += xchunk / 2

            for i, child in enumerate(model._children):
                pos = assign_pos(pos, child, levels, level + 1, xchunk)

            # Remove any duplicates we found along the way
            levels[level] = list(set(levels[level]))

            return pos

        # Recursively get the node positions
        pos = assign_pos(pos, self, levels, level=0, xchunk=20)

        # Plot the tree using Matplotlib
        fig, ax = plt.subplots(figsize=(6, 6))

        # Draw nodes with different styles if they are masked
        for node, (x, y) in pos.items():
            color = "lightgreen" if node in masked_labels else "lightblue"
            ax.text(
                x,
                -y,  # Invert y-axis for bottom-to-top
                node,
                ha="center",
                va="center",
                bbox=dict(
                    facecolor=color,
                    edgecolor="black",
                    boxstyle="round,pad=0.3"
                    if node not in extract_labels
                    else "square,pad=0.3",
                ),
            )

        # Draw edges with different styles based on link type
        for source, targets in links.items():
            for target, link_type in targets:
                sx, sy = pos[source]
                tx, ty = pos[target]
                linestyle = "dashed" if link_type == "attenuate" else "solid"
                ax.plot(
                    [sx, tx],
                    [-sy, -ty],  # Invert y-axis for bottom-to-top
                    linestyle=linestyle,
                    color="black",
                    lw=1,
                )

        # Create legend elements
        attenuate_line = mlines.Line2D(
            [], [], color="black", linestyle="dashed", label="Attenuated"
        )
        combine_line = mlines.Line2D(
            [], [], color="black", linestyle="solid", label="Combined"
        )
        masked_patch = mpatches.Patch(color="lightgreen", label="Masked")
        unmasked_patch = mpatches.Patch(color="lightblue", label="Unmasked")
        extraction_box = mpatches.Patch(
            facecolor="none", edgecolor="black", label="Extraction"
        )

        # Add legend to plot
        ax.legend(
            handles=[
                attenuate_line,
                combine_line,
                masked_patch,
                unmasked_patch,
                extraction_box,
            ],
            frameon=False,
        )

        ax.set_aspect("equal")
        ax.axis("off")
        plt.show()

    def iplot_emission_tree(self):
        pass


class IncidentEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="incident",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="incident",
            fesc=fesc,
        )


class LineContinuumEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="linecont",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="linecont",
            fesc=fesc,
        )


class TransmittedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="transmitted",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=fesc,
        )


class EscapedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="escaped",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="transmitted",
            fesc=(1 - fesc),
        )


class NebularContinuumEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="nebular_continuum",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            extract="nebular_continuum",
            fesc=fesc,
        )


class NebularEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="nebular",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                LineContinuumEmission(grid=grid, fesc=fesc),
                NebularContinuumEmission(grid=grid, fesc=fesc),
            ),
            fesc=fesc,
        )


class ReprocessedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        label="reprocessed",
        fesc=0.0,
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                NebularEmission(grid=grid, fesc=fesc),
                TransmittedEmission(grid=grid, fesc=fesc),
            ),
            fesc=fesc,
        )


class AttenuatedEmission(EmissionModel):
    def __init__(
        self,
        grid,
        dust_curve,
        apply_dust_to,
        tau_v,
        label="attenuated",
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            dust_curve=dust_curve,
            apply_dust_to=apply_dust_to,
            tau_v=tau_v,
        )


class EmergentEmission(EmissionModel):
    def __init__(
        self,
        grid,
        dust_curve,
        apply_dust_to,
        tau_v,
        fesc=0.0,
        label="emergent",
    ):
        EmissionModel.__init__(
            self,
            grid=grid,
            label=label,
            combine=(
                AttenuatedEmission(
                    grid=grid,
                    dust_curve=dust_curve,
                    apply_dust_to=apply_dust_to,
                    tau_v=tau_v,
                ),
                EscapedEmission(grid=grid, fesc=fesc),
            ),
            fesc=fesc,
        )


class DustEmission(EmissionModel):
    def __init__(
        self,
        dust_emission_model,
        label="dust_emission",
    ):
        EmissionModel.__init__(
            self,
            grid=None,
            label=label,
            dust_emission_model=dust_emission_model,
        )


class TotalEmission(EmissionModel):
    def __init__(
        self,
        grid,
        dust_curve,
        tau_v,
        dust_emission_model=None,
        label="total",
        fesc=0.0,
    ):
        # If a dust emission model has been passed then we need combine
        if dust_emission_model is not None:
            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(
                    EmergentEmission(
                        grid=grid,
                        dust_curve=dust_curve,
                        tau_v=tau_v,
                        fesc=fesc,
                        apply_dust_to=ReprocessedEmission(
                            grid=grid,
                            fesc=fesc,
                        ),
                    ),
                    DustEmission(dust_emission_model=dust_emission_model),
                ),
                fesc=fesc,
            )
        else:
            # Otherwise, total == emergent
            EmissionModel.__init__(
                self,
                grid=grid,
                label=label,
                combine=(
                    AttenuatedEmission(
                        grid=grid,
                        dust_curve=dust_curve,
                        apply_dust_to=ReprocessedEmission(
                            grid=grid,
                            fesc=fesc,
                        ),
                        tau_v=tau_v,
                    ),
                    EscapedEmission(grid=grid, fesc=fesc),
                ),
                fesc=fesc,
            )
