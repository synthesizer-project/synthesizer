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
    """
    A class to define the construction of a spectra from a grid.

    An emission model describes the steps necessary to construct a spectra
    from a grid. These steps can be:
    - Extracting a spectra from a grid.
    - Combining multiple spectra.
    - Applying a dust curve to a spectra.
    - Generating spectra from a dust emission model.

    All of these stages can also have masks applied to them to remove certain
    elements from the spectra (e.g. if you want to eliminate young stars).

    By chaining together multiple emission models, complex spectra can be
    constructed from a grid of particles.

    Note: Every time a property is changed anywhere in the tree the tree must
    be reconstructed. This is a cheap process but must happen to ensure
    the expected properties are used when the model is used. This means most
    attributes are accessed through properties and set via setters to
    ensure the tree remains consistent.

    Attributes:
        label (str):
            The key for the spectra that will be produced.
        grid (Grid):
            The grid to extract from.
        extract (str):
            The key for the spectra to extract.
        dust_curve (dust.attenuation.*):
            The dust curve to apply.
        apply_dust_to (EmissionModel):
            The model to apply the dust curve to.
        tau_v (float/ndarray/str/tuple):
            The optical depth to apply. Can be a float, ndarray, or a string
            to a component attribute. Can also be a tuple combining any of
            these.
        dust_emission_model (EmissionModel):
            The dust emission model to generate from.
        combine (list):
            A list of models to combine.
        fesc (float):
            The escape fraction.
        masks (list):
            A list of masks to apply.
        parents (list):
            A list of models which depend on this model.
    """

    def __init__(
        self,
        label,
        grid=None,
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
        - Applying a dust curve to the spectra. (dust_curve, apply_dust_to
          and tau_v must be set)
        - Generating spectra from a dust emission model. (dust_emission_model
          must be set)

        Within any of these steps a mask can be applied to the spectra to
        remove certain elements (e.g. if you want to eliminate particles).

        Args:
            label (str):
                The key for the spectra that will be produced.
            grid (Grid):
                The grid to extract from.
            extract (str):
                The key for the spectra to extract.
            combine (list):
                A list of models to combine.
            apply_dust_to (EmissionModel):
                The model to apply the dust curve to.
            dust_curve (dust.attenuation.*):
                The dust curve to apply.
            tau_v (float/ndarray/str/tuple):
                The optical depth to apply. Can be a float, ndarray, or a
                string to a component attribute. Can also be a tuple combining
                any of these.
            dust_emission_model (EmissionModel):
                The dust emission model to generate from.
            mask_attr (str):
                The component attribute to mask on.
            mask_thresh (unyt_quantity):
                The threshold for the mask.
            mask_op (str):
                The operation to apply. Can be "<", ">", "<=", ">=", "==",
                or "!=".
            fesc (float):
                The escape fraction.
        """
        # What is the key for the spectra that will be produced?
        self.label = label

        # Attach the grid
        self._grid = grid

        # What base key will we be extracting? (can be None if we're applying
        # a dust curve to an existing spectra)
        self._extract = extract

        # Attach the dust attenuation properties
        self._dust_curve = dust_curve
        self._apply_dust_to = apply_dust_to
        self._tau_v = tau_v

        # Attach the dust emission model
        self._dust_emission_model = dust_emission_model

        # Initialise the container for combined spectra we'll add together
        self._combine = list(combine) if combine is not None else combine

        # Attach the escape fraction
        self._fesc = fesc

        # Define flags for what we're doing
        self._is_extracting = True if extract is not None else False
        self._is_combining = (
            True
            if self._combine is not None and len(self._combine) > 0
            else False
        )
        self._is_dust_attenuating = (
            True
            if dust_curve is not None
            or apply_dust_to is not None
            or tau_v is not None
            else False
        )
        self._is_dust_emitting = (
            True if dust_emission_model is not None else False
        )

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
        for child in self._combine:
            self._children.append(child)
        if self._apply_dust_to is not None:
            self._children.append(self._apply_dust_to)

        # Initilaise a container for parent models
        self.parents = []

        # Attach the parent model to the children
        for child in self._children:
            child.parents.append(self)

        # We're done with setup, so unpack the model
        self.unpack_model()

    @property
    def grid(self):
        """Get the Grid object used for extraction."""
        return self._grid

    @property
    def extract(self):
        """Get the key for the spectra to extract."""
        return self._extract

    @property
    def dust_curve(self):
        """Get the dust curve to apply."""
        return self._dust_curve

    @property
    def apply_dust_to(self):
        """Get the spectra to apply the dust curve to."""
        return self._apply_dust_to

    @property
    def tau_v(self):
        """Get the optical depth to apply."""
        return self._tau_v

    @property
    def dust_emission_model(self):
        """Get the dust emission model to generate from."""
        return self._dust_emission_model

    @property
    def combine(self):
        """Get the models that will be combined."""
        return self._combine

    @property
    def fesc(self):
        """Get the escape fraction."""
        return self._fesc

    def _check_args(self):
        """Ensure we have a valid model."""
        # Ensure we have a valid operation
        if (
            sum(
                [
                    self._is_extracting,
                    self._is_combining,
                    self._is_dust_attenuating,
                    self._is_dust_emitting,
                ]
            )
            == 0
        ):
            raise exceptions.InconsistentArguments(
                "No valid operation found from the arguments given. "
                "Currently have:\n"
                "\tFor extraction: "
                f"(grid={self._grid} extract={self._extract})\n"
                "\tFor combination: "
                f"(combine={self._combine})\n"
                "\tFor dust attenuation: "
                f"(dust_curve={self._dust_curve} "
                f"apply_dust_to={self._apply_dust_to} tau_v={self._tau_v})\n"
                "\tFor dust emission: "
                f"(dust_emission_model={self._dust_emission_model})"
            )

        # Ensure only one type of operation is being done
        if (
            sum(
                [
                    self._is_extracting,
                    self._is_combining,
                    self._is_dust_attenuating,
                    self._is_dust_emitting,
                ]
            )
            > 1
        ):
            raise exceptions.InconsistentArguments(
                "Can only extract, combine, or apply dust to spectra in one "
                "model. (Attempting to: "
                f"extract: {self._is_extracting}, "
                f"combine: {self._is_combining}, "
                f"dust_attenuate: {self._is_dust_attenuating}, "
                f"dust_emission: {self._is_dust_emitting})\n"
                "Currently have:\n"
                "\tFor extraction: "
                f"(grid={self._grid} extract={self._extract})\n"
                "\tFor combination: "
                f"(combine={self._combine})\n"
                "\tFor dust attenuation: "
                f"(dust_curve={self._dust_curve} "
                f"apply_dust_to={self._apply_dust_to} tau_v={self._tau_v})\n"
                "\tFor dust emission: "
                f"(dust_emission_model={self._dust_emission_model})"
            )

        # Ensure we have what we need for all operations
        if self._is_extracting and self._grid is None:
            raise exceptions.InconsistentArguments(
                "Must specify a grid to extract from."
            )
        if self._is_dust_attenuating and self._dust_curve is None:
            raise exceptions.InconsistentArguments(
                "Must specify a dust curve to apply."
            )
        if self._is_dust_attenuating and self._apply_dust_to is None:
            raise exceptions.InconsistentArguments(
                "Must specify where to apply the dust curve."
            )
        if self._is_dust_attenuating and self._tau_v is None:
            raise exceptions.InconsistentArguments(
                "Must specify an optical depth to apply."
            )

        # Ensure the grid contains any keys we want to extract
        if (
            self._is_extracting
            and self._extract not in self._grid.spectra.keys()
        ):
            raise exceptions.InconsistentArguments(
                f"Grid does not contain key: {self._extract}"
            )

        # If we haven't been given an escape fraction and we're extracting
        # ensure it's 0.0
        if self._is_extracting:
            if self._fesc is None:
                self._fesc = 0.0

        # Now that we've checked everything is consistent we can make an empty
        # list for combine if it is None
        if self._combine is None:
            self._combine = []

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

    def unpack_model(self):
        """Unpack the model tree to get the order of operations."""
        # Define the private containers we'll unpack everything into. These
        # are dictionaries of the form {<result_label>: <operation props>}
        self._extract_keys = {}
        self._dust_attenuation = {}
        self._dust_emission = {}
        self._combine_keys = {}
        self._mask_keys = {}
        self._models = {}

        # Define the list to hold the model labels in order they need to be
        # generated
        self._bottom_to_top = []

        # Unpack...
        self._unpack_model_recursively(self)

    def set_grid(self, grid, label=None, set_all=False):
        """
        Set the grid to extract from.

        Args:
            grid (Grid):
                The grid to extract from.
            label (str):
                The label of the model to set the grid on. If None, sets the
                grid on this model.
            set_all (bool):
                Whether to set the grid on all models.
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Set the grid
        if label is not None:
            for model in self._models.values():
                # Only set the grid if we're setting all and the model is
                # extracting or the label matches
                if label == model.label or set_all and model._is_extracting:
                    model._grid = grid
        else:
            if self._is_extracting:
                self._grid = grid
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set a grid on a model that is not extracting."
                )

        # Unpack the model now we're done
        self.unpack_model()

    def set_extract(self, extract, label=None, set_all=False):
        """
        Set the spectra to extract from the grid.

        Either sets the spectra to extract from a specific model or from all
        models.

        Args:
            extract (str):
                The key of the spectra to extract.
            label (str):
                The label of the model to set the extraction key on. If None,
                sets the extraction key on this model.
            set_all (bool):
                Whether to set the extraction key on all models.
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Set the extraction key
        if label is not None:
            for model in self._models.values():
                # Only set the extraction key if we're setting all and the
                # model is extracting or the label matches
                if label == model.label or set_all and model._is_extracting:
                    model._extract = extract
        else:
            if self._is_extracting:
                self._extract = extract
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set an extraction key on a model that is "
                    "not extracting."
                )

        # Unpack the model now we're done
        self.unpack_model()

    def set_dust_props(
        self,
        dust_curve,
        apply_dust_to,
        tau_v,
        label=None,
        set_all=False,
    ):
        """
        Set the dust attenuation properties on this model.

        Either sets the properties on a specific model or on all models.

        Args:
            dust_curve (dust.attenuation.*):
                A dust curve instance to apply.
            apply_dust_to (EmissionModel):
                The model to apply the dust curve to.
            tau_v (float/ndarray/str/tuple):
                The optical depth to apply. Can be a float, ndarray, or a
                string to a component attribute. Can also be a tuple combining
                any of these. If a tuple then the eventual tau_v will be the
                product of all contributors.
            label (str):
                The label of the model to set the properties on. If
                None, sets the properties on this model.
            set_all (bool):
                Whether to set the properties on all models.
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Set the properties
        if label is not None:
            for model in self._models.values():
                # Only set the properties if we're setting all and the model is
                # an attenuation step or the label matches
                if (
                    label == model.label
                    or set_all
                    and model._is_dust_attenuating
                ):
                    model._dust_curve = dust_curve
                    model._apply_dust_to = apply_dust_to
                    model._tau_v = tau_v
        else:
            if self.is_dust_attenuating:
                self._dust_curve = dust_curve
                self._apply_dust_to = apply_dust_to
                self._tau_v = tau_v
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set dust attenuation properties on a model that "
                    "is not dust attenuating."
                )

        # Unpack the model now we're done
        self.unpack_model()

    def set_dust_emission_model(self, dust_emission_model, label=None):
        """
        Set the dust emission model on this model.

        Either sets the dust emission model on a specific model or on all
        models.

        Args:
            dust_emission_model (EmissionModel):
                The dust emission model to set.
            label (str):
                The label of the model to set the dust emission model on. If
                None, sets the dust emission model on this model.
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Ensure model is a dust emission model and change the model
        if label is not None and self._models[label]._is_dust_emitting:
            self._models[label]._dust_emission_model = dust_emission_model
        elif self._is_dust_emitting:
            self._dust_emission_model = dust_emission_model
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set a dust emission model on a model that is not "
                "dust emitting."
            )

        # Unpack the model now we're done
        self.unpack_model()

    def set_combine(self, combine, label=None):
        """
        Set the models to combine on this model.

        Either sets the models to combine on a specific model or on all models.

        Args:
            combine (list):
                A list of models to combine.
            label (str):
                The label of the model to set the models to combine on. If
                None, sets the models to combine on this model.
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Ensure all models are EmissionModels
        for model in combine:
            if not isinstance(model, EmissionModel):
                raise exceptions.InconsistentArguments(
                    "All models to combine must be EmissionModels."
                )

        # Set the models to combine ensure the model we are setting on is
        # a combination step
        if label is not None and self._models[label]._is_combining:
            self._models[label]._combine = combine
        elif self._is_combining:
            self._combine = combine
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set models to combine on a model that is not "
                "combining."
            )

        # Unpack the model now we're done
        self.unpack_model()

    def set_fesc(self, fesc, label=None, set_all=False):
        """
        Set the escape fraction on this model.

        Either sets the escape fraction on a specific model or on all models.

        Args:
            fesc (float):
                The escape fraction to set.
            label (str):
                The label of the model to set the escape fraction on. If None,
                sets the escape fraction on this model.
            set_all (bool):
                Whether to set the escape fraction on all models.
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Set the escape fraction
        if label is not None:
            for model in self._models.values():
                # Only set the escape fraction if setting all or the
                # label matches
                if label == model.label or set_all and model._is_extracting:
                    model._fesc = fesc
        else:
            if self._is_extracting:
                self._fesc = fesc
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set an escape fraction on a model that is not "
                    "extracting."
                )

        # Unpack the model now we're done
        self.unpack_model()

    def add_mask(self, attr, thresh, op, label=None, set_all=False):
        """
        Add a mask.

        Either adds a mask to a specific model or to all models.

        Args:
            attr (str):
                The component attribute to mask on.
            thresh (unyt_quantity):
                The threshold for the mask.
            op (str):
                The operation to apply. Can be "<", ">", "<=", ">=", "==",
                or "!=".
            label (str):
                The label of the model to add the mask to. If None, adds the
                mask to this model.
            set_all (bool):
                Whether to add the mask to all models.
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

        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Add the mask
        for model in self._models.values():
            # Only add the mask if we're setting all or the label matches
            if label == model.label or set_all:
                model.masks.append({"attr": attr, "thresh": thresh, "op": op})

        # Unpack now that we're done
        self.unpack_model()

    def add_model(self, model):
        pass

    def remove_model(self, label):
        pass

    def relabel_model(self, old_label, new_label):
        """
        Change the label associated to an existing spectra.

        Args:
            old_label (str): The current label of the spectra.
            new_label (str): The new label to assign to the spectra.
        """
        # Ensure the new label is not already in use
        if new_label in self._models:
            raise exceptions.InconsistentArguments(
                f"Label {new_label} is already in use."
            )

        # Ensure the old label is in use
        if old_label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Label {old_label} is not in use."
            )

        # Get the model
        model = self._models[old_label]

        # Update the label
        model.label = new_label

        # Update the models dictionary
        self._models[new_label] = model
        del self._models[old_label]

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
