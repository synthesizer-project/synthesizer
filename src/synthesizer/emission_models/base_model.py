"""A module defining the emission model from which spectra are constructed.

Generating spectra involves the following steps:
1. Extraction from a Grid.
2. Generation of spectra.
3. Attenuation due to dust in the ISM/nebular.
4. Masking for different types of emission.
5. Combination of different types of emission.

An emission model defines the parameters necessary to perform these steps and
gives an interface for simply defining the construction of complex spectra.

Example usage:
    # Define the grid
    grid = Grid(...)

    # Define the dust curve
    dust_curve = dust.attenuation.PowerLaw(...)

    # Define the emergent emission model
    emergent_emission_model = EmissionModel(
        label="emergent",
        grid=grid,
        dust_curve=dust_curve,
        apply_dust_to=dust_emission_model,
        tau_v=tau_v,
        fesc=fesc,
        emitter="stellar",
    )

    # Generate the spectra
    spectra = stars.get_spectra(emergent_emission_model)

    # Generate the lines
    lines = stars.get_lines(
                line_ids=("Ne 4 1601.45A, He 2 1640.41A", "O3 1660.81A"),
                emission_model=emergent_emission_model
            )
"""

import copy

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from unyt import unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.operations import (
    Combination,
    DustAttenuation,
    Extraction,
    Generation,
)
from synthesizer.line import LineCollection
from synthesizer.warnings import warn


class EmissionModel(Extraction, Generation, DustAttenuation, Combination):
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
        lam (unyt_array):
            The wavelength array.
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
        generator (EmissionModel):
            The emission generation model. This must define a get_spectra
            method.
        combine (list):
            A list of models to combine.
        fesc (float):
            The escape fraction.
        masks (list):
            A list of masks to apply.
        parents (list):
            A list of models which depend on this model.
        children (list):
            A list of models this model depends on.
        related_models (list):
            A list of related models to this model. A related model is a model
            that is connected somewhere within the model tree but is required
            in the construction of the "root" model encapulated by self.
        fixed_parameters (dict):
            A dictionary of component attributes/parameters which should be
            fixed and thus ignore the value of the component attribute. This
            should take the form {<parameter_name>: <value>}.
        scale_by (str/list/tuple):
            Either a component attribute to scale the resultant spectra by,
            a spectra key to scale by (based on the bolometric luminosity). or
            a tuple/list containing strings defining either of the former two
            options. Instead of a string, an EmissionModel can also be passed
            to scale by the luminosity of that model.
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
        generator=None,
        dust_lum_intrinsic=None,
        dust_lum_attenuated=None,
        mask_attr=None,
        mask_thresh=None,
        mask_op=None,
        fesc=None,
        related_models=None,
        emitter=None,
        fixed_parameters={},
        scale_by=None,
        **kwargs,
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
        - Generating spectra from a dust emission model. (generator
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
            generator (EmissionModel):
                The emission generation model. This must define a get_spectra
                method.
            dust_lum_intrinsic_key (EmissionModel):
                The intrinsic model to use deriving the dust
                luminosity when computing dust emission.
            dust_lum_attenuated_key (EmissionModel):
                The attenuated model to use deriving the dust
                luminosity when computing dust emission.
            mask_attr (str):
                The component attribute to mask on.
            mask_thresh (unyt_quantity):
                The threshold for the mask.
            mask_op (str):
                The operation to apply. Can be "<", ">", "<=", ">=", "==",
                or "!=".
            fesc (float):
                The escape fraction.
            related_models (list):
                A list of related models to this model. A related model is
                a model that is connected somwhere within the model tree but
                is required in the construction of the "root" model
                encapulated by self.
            emitter (str):
                The emitter this emission model acts on. Default is
                "stellar".
            fixed_parameters (dict):
                A dictionary of component attributes/parameters which should be
                fixed and thus ignore the value of the component attribute.
                This should take the form {<parameter_name>: <value>}.
            scale_by (str/list/tuple/EmissionModel):
                Either a component attribute to scale the resultant spectra by,
                a spectra key to scale by (based on the bolometric luminosity).
                or a tuple/list containing strings defining either of the
                former two options. Instead of a string, an EmissionModel can
                be passed to scale by the luminosity of that model.
            **kwargs:
                Any additional keyword arguments to store. These can be used
                to store additional information needed by the model.
        """
        # What is the key for the spectra that will be produced?
        self.label = label

        # Attach the wavelength array and store it on the model
        if grid is not None:
            self.lam = grid.lam
        else:
            self.lam = None

        # Store any extra kwargs this can either be used to store additional
        # information needed by the model
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Store any fixed parameters
        self.fixed_parameters = fixed_parameters

        # Attach which emitter we are working with
        self._emitter = emitter

        # Define the container which will hold mask information
        self.masks = []

        # If we have been given a mask, add it
        self._init_masks(mask_attr, mask_thresh, mask_op)

        # Get operation flags
        self._get_operation_flags(
            grid=grid,
            extract=extract,
            combine=combine,
            dust_curve=dust_curve,
            apply_dust_to=apply_dust_to,
            tau_v=tau_v,
            generator=generator,
            dust_lum_intrinsic=dust_lum_intrinsic,
            dust_lum_attenuated=dust_lum_attenuated,
        )

        # Initilaise the corresponding operation (also checks we have a
        # valid set of arguments and also have everything we need for the
        # operation
        self._init_operations(
            grid=grid,
            extract=extract,
            combine=combine,
            apply_dust_to=apply_dust_to,
            dust_curve=dust_curve,
            tau_v=tau_v,
            generator=generator,
            dust_lum_intrinsic=dust_lum_intrinsic,
            dust_lum_attenuated=dust_lum_attenuated,
            fesc=fesc,
        )

        # Containers for children and parents
        self._children = set()
        self._parents = set()

        # Store the arribute to scale the spectra by
        if isinstance(scale_by, (list, tuple)):
            self._scale_by = scale_by
            self._scale_by = [
                s if isinstance(s, str) else s.label for s in scale_by
            ]
        elif isinstance(scale_by, EmissionModel):
            self._scale_by = (scale_by.label,)
        else:
            self._scale_by = (scale_by,)

        # Attach the related models
        self.related_models = (
            related_models if related_models is not None else []
        )

        # We're done with setup, so unpack the model
        self.unpack_model()

    def _init_operations(
        self,
        grid,
        extract,
        combine,
        apply_dust_to,
        dust_curve,
        tau_v,
        generator,
        dust_lum_intrinsic,
        dust_lum_attenuated,
        fesc,
    ):
        """
        Initialise the correct parent operation.

        Args:
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
            generator (EmissionModel):
                The emission generation model. This must define a get_spectra
                method.
            dust_lum_intrinsic (EmissionModel):
                The intrinsic model to use deriving the dust
                luminosity when computing dust emission.
            dust_lum_attenuated (EmissionModel):
                The attenuated model to use deriving the dust
                luminosity when computing dust emission.
            fesc (float):
                The escape fraction.
        """
        # Which operation are we doing?
        if self._is_extracting:
            Extraction.__init__(self, grid, extract, fesc)
        elif self._is_combining:
            Combination.__init__(self, combine)
        elif self._is_dust_attenuating:
            DustAttenuation.__init__(self, dust_curve, apply_dust_to, tau_v)
        elif self._is_dust_emitting:
            Generation.__init__(
                self, generator, dust_lum_intrinsic, dust_lum_attenuated
            )
        elif self._is_generating:
            Generation.__init__(self, generator, None, None)
        else:
            raise exceptions.InconsistentArguments(
                "No valid operation found from the arguments given "
                f"(label={self.label}). "
                "Currently have:\n"
                "\tFor extraction: grid=("
                f"{grid.grid_name if grid is not None else None}"
                f" extract={extract})\n"
                "\tFor combination: "
                f"(combine={combine})\n"
                "\tFor dust attenuation: "
                f"(dust_curve={dust_curve} "
                f"apply_dust_to={apply_dust_to} tau_v={tau_v})\n"
                "\tFor generation "
                f"(generator={generator}, "
                f"dust_lum_intrinsic={dust_lum_intrinsic}, "
                f"dust_lum_attenuated={dust_lum_attenuated})"
            )

        # Double check we have been asked for only one operation
        if (
            sum(
                [
                    self._is_extracting,
                    self._is_combining,
                    self._is_dust_attenuating,
                    self._is_dust_emitting,
                    self._is_generating,
                ]
            )
            != 1
        ):
            raise exceptions.InconsistentArguments(
                "Can only extract, combine, generate or apply dust to "
                f"spectra in one model (label={self.label}). (Attempting to: "
                f"extract: {self._is_extracting}, "
                f"combine: {self._is_combining}, "
                f"dust_attenuate: {self._is_dust_attenuating}, "
                f"dust_emission: {self._is_dust_emitting})\n"
                "Currently have:\n"
                "\tFor extraction: grid=("
                f"{grid.grid_name if grid is not None else None}"
                "\tFor combination: "
                f"(combine={combine})\n"
                "\tFor dust attenuation: "
                f"(dust_curve={dust_curve} "
                f"apply_dust_to={apply_dust_to} tau_v={tau_v})\n"
                "\tFor generation "
                f"(generator={generator}, "
                f"dust_lum_intrinsic={dust_lum_intrinsic}, "
                f"dust_lum_attenuated={dust_lum_attenuated})"
            )

        # Ensure we have what we need for all operations
        if self._is_extracting and grid is None:
            raise exceptions.InconsistentArguments(
                "Must specify a grid to extract from."
            )
        if self._is_dust_attenuating and dust_curve is None:
            raise exceptions.InconsistentArguments(
                "Must specify a dust curve to apply."
            )
        if self._is_dust_attenuating and apply_dust_to is None:
            raise exceptions.InconsistentArguments(
                "Must specify where to apply the dust curve."
            )
        if self._is_dust_attenuating and tau_v is None:
            raise exceptions.InconsistentArguments(
                "Must specify an optical depth to apply."
            )

        # Ensure the grid contains any keys we want to extract
        if self._is_extracting and extract not in grid.spectra.keys():
            raise exceptions.InconsistentArguments(
                f"Grid does not contain key: {extract}"
            )

    def _init_masks(self, mask_attr, mask_thresh, mask_op):
        """
        Initialise the mask operation.

        Args:
            mask_attr (str):
                The component attribute to mask on.
            mask_thresh (unyt_quantity):
                The threshold for the mask.
            mask_op (str):
                The operation to apply. Can be "<", ">", "<=", ">=", "==",
                or "!=".
        """
        # If we have been given a mask, add it
        if (
            mask_attr is not None
            and mask_thresh is not None
            and mask_op is not None
        ):
            # Ensure mask_thresh comes with units
            if not isinstance(mask_thresh, unyt_quantity):
                raise exceptions.MissingUnits(
                    "Mask threshold must be given with units."
                )
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

    @property
    def emitter(self):
        """Get the emitter this emission model acts on."""
        return getattr(self, "_emitter", None)

    @property
    def grid(self):
        """Get the Grid object used for extraction."""
        return getattr(self, "_grid", None)

    @property
    def extract(self):
        """Get the key for the spectra to extract."""
        return getattr(self, "_extract", None)

    @property
    def fesc(self):
        """Get the escape fraction."""
        return getattr(self, "_fesc", 0.0)

    @property
    def generator(self):
        """Get the emission generation model."""
        return getattr(self, "_generator", None)

    @property
    def dust_lum_intrinsic(self):
        """Get the intrinsic model for computing dust luminosity."""
        return getattr(self, "_dust_lum_intrinsic", None)

    @property
    def dust_lum_attenuated(self):
        """Get the attenuated model for computing dust luminosity."""
        return getattr(self, "_dust_lum_attenuated", None)

    @property
    def dust_curve(self):
        """Get the dust curve to apply."""
        return getattr(self, "_dust_curve", None)

    @property
    def apply_dust_to(self):
        """Get the spectra to apply the dust curve to."""
        return getattr(self, "_apply_dust_to", None)

    @property
    def tau_v(self):
        """Get the optical depth to apply."""
        return getattr(self, "_tau_v", None)

    @property
    def combine(self):
        """Get the models to combine."""
        return getattr(self, "_combine", tuple())

    @property
    def scale_by(self):
        """Get the attribute to scale the spectra by."""
        return self._scale_by

    def _summary(self):
        """Call the correct summary method."""
        if self._is_extracting:
            return self._extract_summary()
        elif self._is_combining:
            return self._combine_summary()
        elif self._is_dust_attenuating:
            return self._attenuate_summary()
        elif self._is_dust_emitting:
            return self._generate_summary()
        elif self._is_generating:
            return self._generate_summary()
        else:
            return []

    def __str__(self):
        """Return a string summarising the model."""
        parts = []

        # Get the labels in reverse order
        labels = [*self._extract_keys, *self._bottom_to_top]

        for label in labels:
            # Get the model
            model = self._models[label]

            # Make the model title
            parts.append("-")
            parts.append(f"  {model.label}".upper())
            if model._emitter is not None:
                parts[-1] += f" ({model._emitter})"
            else:
                parts[-1] += " (galaxy)"
            parts.append("-")

            # Get this models summary
            parts.extend(model._summary())

            # Print any fixed parameters if there are any
            if len(model.fixed_parameters) > 0:
                parts.append("  Fixed parameters:")
                for key, value in model.fixed_parameters.items():
                    parts.append(f"    - {key}: {value}")

            if model._is_masked:
                parts.append("  Masks:")
                for mask in model.masks:
                    parts.append(
                        f"    - {mask['attr']} {mask['op']} {mask['thresh']} "
                    )

        # Get the length of the longest line
        longest = max(len(line) for line in parts) + 10

        # Place divisions between each model
        for ind, line in enumerate(parts):
            if line == "-":
                parts[ind] = "-" * longest

        # Pad all lines to the same length
        for ind, line in enumerate(parts):
            line += " " * (longest - len(line))
            parts[ind] = line

        # Attach a header and footer line
        parts.insert(0, f" EmissionModel: {self.label} ".center(longest, "="))
        parts.append("=" * longest)

        parts[0] = "|" + parts[0]
        parts[-1] += "|"

        return "|\n|".join(parts)

    def __getitem__(self, label):
        """
        Enable label look up.

        Lets us access the models in the tree as if an EmissionModel were a
        dictionary.

        Args:
            label (str): The label of the model to get.
        """
        if label not in self._models:
            raise KeyError(
                f"Could not find {label}! Model has: {self._models.keys()}"
            )
        return self._models[label]

    def _get_operation_flags(
        self,
        grid=None,
        extract=None,
        combine=None,
        dust_curve=None,
        apply_dust_to=None,
        tau_v=None,
        generator=None,
        dust_lum_attenuated=None,
        dust_lum_intrinsic=None,
    ):
        """Define the flags for what operation the model does."""
        # Define flags for what we're doing
        self._is_extracting = extract is not None and grid is not None
        self._is_combining = combine is not None and len(combine) > 0
        self._is_dust_attenuating = (
            dust_curve is not None
            or apply_dust_to is not None
            or tau_v is not None
        )
        self._is_dust_emitting = (
            generator is not None
            and dust_lum_attenuated is not None
            and dust_lum_intrinsic is not None
        )
        self._is_generating = (
            generator is not None and not self._is_dust_emitting
        )
        self._is_masked = len(self.masks) > 0

    def _unpack_model_recursively(self, model):
        """
        Traverse the model tree and collect what we will need to do.

        Args:
            model (EmissionModel): The model to unpack.
        """
        # Store the model
        self._models[model.label] = model

        # If we are extracting a spectra, store the key
        if model._is_extracting:
            self._extract_keys[model.label] = model.extract

        # If we are applying a dust curve, store the key
        if model._is_dust_attenuating:
            self._dust_attenuation[model.label] = (
                model.apply_dust_to,
                model.dust_curve,
            )
            model._children.add(model.apply_dust_to)
            model.apply_dust_to._parents.add(model)

        # If we are applying a dust emission model, store the key
        if model._is_generating or model._is_dust_emitting:
            self._generator_models[model.label] = model.generator
            if model._dust_lum_attenuated is not None:
                model._children.add(model._dust_lum_attenuated)
                model._dust_lum_attenuated._parents.add(model)
            if model._dust_lum_intrinsic is not None:
                model._children.add(model._dust_lum_intrinsic)
                model._dust_lum_intrinsic._parents.add(model)

        # If we are masking, store the key
        if model._is_masked:
            self._mask_keys[model.label] = model.masks

        # If we are combining spectra, store the key
        if model._is_combining:
            self._combine_keys[model.label] = model.combine
            for child in model.combine:
                model._children.add(child)
                child._parents.add(model)

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
        self._generator_models = {}
        self._combine_keys = {}
        self._mask_keys = {}
        self._models = {}

        # Define the list to hold the model labels in order they need to be
        # generated
        self._bottom_to_top = []

        # Unpack...
        self._unpack_model_recursively(self)

        # Also unpack any related models
        for model in self.related_models:
            if model.label not in self._models:
                self._unpack_model_recursively(model)

        # Now we've worked through the full tree we can set parent pointers
        for model in self._models.values():
            for child in model._children:
                child._parents.add(model)

        # If we don't have a wavelength array, find one in the models
        if self.lam is None:
            for model in self._models.values():
                if model.lam is not None:
                    self.lam = model.lam

                # Ensure the wavelength arrays agree for all models
                if (
                    self.lam is not None
                    and model.lam is not None
                    and not np.array_equal(self.lam, model.lam)
                ):
                    raise exceptions.InconsistentArguments(
                        "Wavelength arrays do not match somewhere in the tree."
                    )

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
        if label is not None or set_all:
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

    def set_extract(self, extract, label=None):
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
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Set the extraction key
        if label is not None:
            if self._models[label]._is_extracting:
                self._models[label]._extract = extract
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set an extraction key on a model that is not "
                    "extracting."
                )
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
        if label is not None or set_all:
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
                    model._tau_v = (
                        tau_v if isinstance(tau_v, (tuple, list)) else [tau_v]
                    )
        else:
            if self.is_dust_attenuating:
                self._dust_curve = dust_curve
                self._apply_dust_to = apply_dust_to
                self._tau_v = (
                    tau_v if isinstance(tau_v, (tuple, list)) else [tau_v]
                )
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set dust attenuation properties on a model that "
                    "is not dust attenuating."
                )

        # Unpack the model now we're done
        self.unpack_model()

    def set_generator(self, generator, label=None):
        """
        Set the dust emission model on this model.

        Either sets the dust emission model on a specific model or on all
        models.

        Args:
            generator (EmissionModel):
                The emission generation model to set.
            label (str):
                The label of the model to set the dust emission model on. If
                None, sets the dust emission model on this model.
        """
        # Ensure the label exists (if not None)
        if label is not None and label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Ensure model is a emission generation model and change the model
        if (
            label is not None
            and self._models[label]._is_dust_emitting
            or self._models[label]._is_generating
        ):
            self._models[label]._generator = generator
        elif self._is_dust_emitting or self._is_generating:
            self._generator = generator
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
        if label is not None or set_all:
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
        if label is not None or set_all:
            for model in self._models.values():
                # Only add the mask if we're setting all or the label matches
                if label == model.label or set_all:
                    model.masks.append(
                        {"attr": attr, "thresh": thresh, "op": op}
                    )
        else:
            self.masks.append({"attr": attr, "thresh": thresh, "op": op})

        # Unpack now that we're done
        self.unpack_model()

    def replace_model(self, replace_label, *replacements, new_label=None):
        """
        Remove a child model from this model.

        Args:
            replace_label (str):
                The label of the model to replace.
            replacements (EmissionModel):
                The models to replace the model with.
            new_label (str):
                The label for the new combination step if multiple replacements
                have been passed (ignored otherwise).
        """
        # Ensure the label exists
        if replace_label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {replace_label}"
            )

        # Ensure all replacements are EmissionModels
        for model in replacements:
            if not isinstance(model, EmissionModel):
                raise exceptions.InconsistentArguments(
                    "All replacements must be EmissionModels."
                )

        # Get the model we are replacing
        replace_model = self._models[replace_label]

        # Get the children and parents of this model
        parents = replace_model._parents
        children = replace_model._children

        # Define the relation to all parents and children
        relations = {}
        for parent in parents:
            if replace_model in parent.combine:
                relations[parent.label] = "combine"
            if parent.apply_dust_to == replace_model:
                relations[parent.label] = "dust_attenuate"
            if parent.dust_lum_intrinsic == replace_model:
                relations[parent.label] = "dust_intrinsic"
            if parent.dust_lum_attenuated == replace_model:
                relations[parent.label] = "dust_attenuated"
        for child in children:
            if child in replace_model.combine:
                relations[child.label] = "combine"
            if child.apply_dust_to == replace_model:
                relations[child.label] = "dust_attenuate"
            if child.dust_lum_intrinsic == replace_model:
                relations[child.label] = "dust_intrinsic"
            if child.dust_lum_attenuated == replace_model:
                relations[child.label] = "dust_attenuated"

        # Remove the model we are replacing
        self._models.pop(replace_label)
        for parent in parents:
            parent._children.remove(replace_model)
            if relations[parent.label] == "combine":
                parent._combine.remove(replace_model)
            if relations[parent.label] == "dust_attenuate":
                parent._apply_dust_to = None
            if relations[parent.label] == "dust_intrinsic":
                parent._dust_lum_intrinsic = None
            if relations[parent.label] == "dust_attenuated":
                parent._dust_lum_attenuated = None
        for child in children:
            child._parents.remove(replace_model)

        # Do we have more than 1 replacement?
        if len(replacements) > 1:
            # We'll have to include a combination model
            new_model = EmissionModel(
                label=replace_label if new_label is None else new_label,
                combine=replacements,
            )
        else:
            new_model = replacements[0]

            # Warn the user if they passed a new label, it won't be used
            if new_label is not None:
                warn(
                    "Warning: new_label is only used when multiple "
                    "replacements are passed."
                )

        # Attach the new model/s to the children
        for child in children:
            child._parents.update(set(replacements))

        # Attach the new model to the parents
        for parent in parents:
            parent._children.add(new_model)
            if relations[parent.label] == "combine":
                parent._combine.append(new_model)
            if relations[parent.label] == "dust_attenuate":
                parent._apply_dust_to = new_model
            if relations[parent.label] == "dust_intrinsic":
                parent._dust_lum_intrinsic = new_model
            if relations[parent.label] == "dust_attenuated":
                parent._dust_lum_attenuated = new_model

        # Unpack now we're done
        self.unpack_model()

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

    def _get_tree_levels(self, root):
        """
        Get the levels of the tree.

        Args:
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
            """
            Recursively assign levels to the models.

            Args:
                levels (dict):
                    The levels of the models.
                links (dict):
                    The links between models.
                extract_labels (set):
                    The labels of models that are extracting.
                masked_labels (list):
                    The labels of models that are masked.
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
            """
            # Get the model label
            label = model.label

            # Assign the level
            levels[model.label] = max(levels.get(model.label, level), level)

            # Define the links
            if model._is_dust_attenuating:
                links.setdefault(label, []).append(
                    (model.apply_dust_to.label, "--")
                )
            if model._is_combining:
                links.setdefault(label, []).extend(
                    [(child.label, "-") for child in model._combine]
                )
            if model._is_dust_emitting:
                links.setdefault(label, []).extend(
                    [
                        (
                            model._dust_lum_intrinsic.label
                            if model._dust_lum_intrinsic is not None
                            else None,
                            "dotted",
                        ),
                        (
                            model._dust_lum_attenuated.label
                            if model._dust_lum_attenuated is not None
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
                levels, links, extract_labels, masked_labels = _assign_levels(
                    levels,
                    links,
                    extract_labels,
                    masked_labels,
                    child,
                    level + 1,
                )

            return levels, links, extract_labels, masked_labels

        # Get the root model
        root_model = self._models[root]

        # Recursively assign levels
        model_levels, links, extract_labels, masked_labels = _assign_levels(
            {}, {}, set(), [], root_model, 0
        )

        # Unpack the levels
        levels = {}

        for label, level in model_levels.items():
            levels.setdefault(level, []).append(label)

        return levels, links, extract_labels, masked_labels

    def _get_model_positions(self, levels, root, ychunk=10.0, xchunk=20.0):
        """
        Get the position of each model in the tree.

        Args:
            levels (dict):
                The levels of the models in the tree.
            ychunk (float):
                The vertical spacing between levels.
            xchunk (float):
                The horizontal spacing between models.

        Returns:
            pos (dict):
                The position of each model in the tree.
        """

        def _get_parent_pos(pos, model):
            """
            Get the position of the parent/s of a model.

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
                parent.label
                for parent in model._parents
                if parent.label in pos
            ]
            if len(set(parents)) == 0:
                return 0.0
            elif len(set(parents)) == 1:
                return pos[parents[0]][0]

            return np.mean([pos[parent][0] for parent in set(parents)])

        def _get_child_pos(x, pos, children, level, xchunk):
            """
            Get the position of the children of a model.

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
            """
            Get the position of the models in a level.

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
                _get_parent_pos(pos, self._models[model]) for model in models
            ]

            # Sort models by parent_pos
            models = [
                model
                for _, model in sorted(
                    zip(parent_pos, models), key=lambda x: x[0]
                )
            ]

            # Get the parents
            parents = []
            for model in models:
                parents.extend(
                    [
                        parent.label
                        for parent in self._models[model]._parents
                        if parent.label in pos
                    ]
                )

            # If we only have one parent for this level then we can assign
            # the position based on the parent
            if len(set(parents)) == 1:
                x = _get_parent_pos(pos, self._models[models[0]])
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
        self,
        root=None,
        show=True,
        fontsize=10,
    ):
        """
        Plot the tree defining the spectra.

        Args:
            root (str):
                If not None this defines the root of a sub tree to plot.
            show (bool):
                Whether to show the plot.
            fontsize (int):
                The fontsize to use for the labels.
        """
        # Get the tree levels
        levels, links, extract_labels, masked_labels = self._get_tree_levels(
            root if root is not None else self.label
        )

        # Get the postion of each node
        pos = self._get_model_positions(
            levels, root=root if root is not None else self.label
        )

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
                fontsize=fontsize,
            )

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
                    label="Attenuated",
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
                    label="Dust Luminosity",
                )
            )
        if len(masked_labels) > 0:
            handles.append(mpatches.Patch(color="lightgreen", label="Masked"))
            handles.append(mpatches.Patch(color="lightblue", label="Unmasked"))
        handles.append(
            mpatches.Patch(
                facecolor="none", edgecolor="black", label="Extraction"
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

    def _apply_overrides(self, emission_model, dust_curves, tau_v, fesc, mask):
        """
        Apply overrides to an emission model copy.

        This function is used in _get_spectra to apply any emission model
        property overrides passed to that method before generating the
        spectra.

        _get_spectra will make a copy of the emission model and then pass it
        to this function to apply any overrides before generating the spectra.

        Args:
            emission_model (EmissionModel):
                The emission model copy to apply the overrides to.
            dust_curves (dict):
                An overide to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An overide to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                        should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form:
                          {<label>: float(<tau_v>)}
                      to use a specific optical depth with a particular
                      model or
                          {<label>: str(<attribute>)}
                      to use an attribute of the component as the optical
                      depth.
            fesc (dict):
                An overide to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form:
                          {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or
                          {<label>: str(<attribute>)}
                      to use an attribute of the component as the escape
                      fraction.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
        """
        # If we have dust curves to apply, apply them
        if dust_curves is not None:
            for label, dust_curve in dust_curves.items():
                emission_model._models[label]._dust_curve = dust_curve

        # If we have optical depths to apply, apply them
        if tau_v is not None:
            for label, value in tau_v.items():
                emission_model._models[label]._tau_v = value

        # If we have escape fractions to apply, apply them
        if fesc is not None:
            for label, value in fesc.items():
                emission_model._models[label]._fesc = value

        # If we have masks to apply, apply them
        if mask is not None:
            for label, mask in mask.items():
                emission_model.add_mask(**mask, label=label)

    def _get_spectra(
        self,
        emitters,
        per_particle,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        spectra={},
        **kwargs,
    ):
        """
        Generate stellar spectra as described by the emission model.

        If the emission model defines a dust_curve or fesc and no overide
        is provided in the arguments to this function.

        Args:
            emitters (Stars/BlackHoles):
                The emitters to generate the spectra for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
            per_particle (bool):
                Are we generating per particle?
            dust_curves (dict):
                An overide to the emisison model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An overide to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                        should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form:
                            {<label>: float(<tau_v>)}
                        to use a specific optical depth with a particular
                        model or
                            {<label>: str(<attribute>)}
                        to use an attribute of the component as the optical
                        depth.
            fesc (dict):
                An overide to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form:
                            {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or
                            {<label>: str(<attribute>)}
                      to use an attribute of the component as the escape
                      fraction.
            covering_fraction (dict):
                An overide to the emission model covering fraction. Either:
                    - None, indicating the covering fraction defined on the
                      emission model should be used.
                    - A float to use as the covering fraction for all models.
                    - A dictionary of the form:
                            {<label>: float(<covering_fraction>)}
                      to use a specific covering fraction with a particular
                      model or
                            {<label>: str(<attribute>)}
                      to use an attribute of the component as the covering
                      fraction.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            verbose (bool)
                Are we talking?
            spectra (dict)
                A dictionary of spectra to add to. This is used for recursive
                calls to this function.
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of spectra which can be attached to the
                appropriate spectra attribute of the component
                (spectra/particle_spectra)
        """
        # We don't want to modify the original emission model with any
        # modifications made here so we'll make a copy of it (this is a
        # shallow copy so very cheap and doesn't copy any pointed to objects
        # only their reference)
        emission_model = copy.copy(self)

        # Apply any overides we have
        self._apply_overrides(emission_model, dust_curves, tau_v, fesc, mask)

        # Define a dictionary to keep track of any spectra that need scaling
        # by other spectra
        scale_later = {}

        print("Spectra length:", len(spectra))

        # Perform all extractions
        spectra = self._extract_spectra(
            emission_model,
            emitters,
            per_particle,
            spectra,
            verbose,
            **kwargs,
        )

        for key in spectra:
            print("Spectra shape:", spectra[key]._lnu.shape)

        # With all base spectra extracted we can now loop from bottom to top
        # of the tree creating each spectra
        for label in emission_model._bottom_to_top:
            # Get this model
            this_model = emission_model._models[label]

            # Get the spectra for the related models that don't appear in the
            # tree
            for related_model in this_model.related_models:
                if related_model.label not in spectra:
                    spectra.update(
                        related_model._get_spectra(
                            emitters,
                            per_particle,
                            dust_curves=dust_curves,
                            tau_v=tau_v,
                            fesc=fesc,
                            mask=mask,
                            verbose=verbose,
                            spectra=spectra,
                            **kwargs,
                        )
                    )

            # Skip models for a different emitters
            if (
                this_model.emitter not in emitters
                and this_model.emitter != "galaxy"
            ):
                continue

            # Get the emitter (as long as we aren't doing a combination for a
            # galaxy spectra
            if this_model.emitter != "galaxy":
                emitter = emitters[this_model.emitter]
            else:
                emitter = None

            # Do we have to define a mask?
            this_mask = None
            for mask_dict in this_model.masks:
                this_mask = emitter.get_mask(**mask_dict, mask=this_mask)
                print(
                    label,
                    this_mask.shape,
                    mask_dict,
                    emitter.nparticles,
                )

            # Are we doing a combination?
            if this_model._is_combining:
                spectra = self._combine_spectra(
                    emission_model,
                    spectra,
                    this_model,
                )

            elif this_model._is_dust_attenuating:
                print(spectra[this_model.apply_dust_to]._lnu.shape)
                spectra = self._dust_attenuate_spectra(
                    this_model,
                    spectra,
                    emitter,
                    this_mask,
                )

            elif this_model._is_dust_emitting or this_model._is_generating:
                spectra = self._generate_spectra(
                    this_model,
                    emission_model,
                    spectra,
                    np.tile(self.lam, (emitter.nparticles, 1))
                    if per_particle
                    else self.lam,
                )

            # Are we scaling the spectra?
            for scaler in this_model.scale_by:
                if scaler is None:
                    continue
                elif hasattr(emitter, scaler):
                    # Scale the spectra by this attribute
                    spectra[label]._lnu *= getattr(emitter, scaler)
                elif scaler in spectra:
                    # Compute the scaling
                    scaling = (
                        spectra[scaler].measure_bolometric_luminosity()
                        / spectra[label].measure_bolometric_luminosity()
                    ).value

                    print(label, scaler, spectra[label]._lnu.ndim)

                    # Scale the spectra (handling 1D and 2D cases)
                    if spectra[label]._lnu.ndim > 1:
                        spectra[label]._lnu *= scaling[:, None]
                    else:
                        spectra[label]._lnu *= scaling
                else:
                    scale_later.setdefault(label, []).append(scaler)

        # Handle any final scalings we need to do
        for spec, scalers in scale_later.items():
            for scaler in scalers:
                if isinstance(scaler, EmissionModel):
                    scaler = scaler.label
                # Compute the scaling
                scaling = (
                    spectra[scaler].measure_bolometric_luminosity()
                    / spectra[spec].measure_bolometric_luminosity()
                )

                spectra[spec] *= scaling

        return spectra

    def _get_lines(
        self,
        line_ids,
        emitters,
        per_particle,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        mask=None,
        verbose=True,
        lines={},
        **kwargs,
    ):
        """
        Generate stellar lines as described by the emission model.

        If the emission model defines a dust_curve or fesc and no overide
        is provided in the arguments to this function.

        Args:
            line_ids (list):
                The line ids to extract.
            emitters (Stars/BlackHoles):
                The emitters to generate the lines for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
            per_particle (bool):
                Are we generating lines per particle?
            dust_curves (dict):
                An overide to the emisison model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An overide to the dust model optical depth. Either:
                    - None, indicating the tau_v defined on the emission model
                        should be used.
                    - A float to use as the optical depth for all models.
                    - A dictionary of the form:
                            {<label>: float(<tau_v>)}
                        to use a specific optical depth with a particular
                        model or
                            {<label>: str(<attribute>)}
                        to use an attribute of the component as the optical
                        depth.
            fesc (dict):
                An overide to the emission model escape fraction. Either:
                    - None, indicating the fesc defined on the emission model
                      should be used.
                    - A float to use as the escape fraction for all models.
                    - A dictionary of the form:
                            {<label>: float(<fesc>)}
                      to use a specific escape fraction with a particular
                      model or
                            {<label>: str(<attribute>)}
                      to use an attribute of the component as the escape
                      fraction.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            verbose (bool)
                Are we talking?
            lines (dict)
                A dictionary of lines to add to. This is used for recursive
                calls to this function.
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of lines which can be attached to the
                appropriate lines attribute of the component
                (lines/particle_lines)
        """
        # We don't want to modify the original emission model with any
        # modifications made here so we'll make a copy of it (this is a
        # shallow copy so very cheap and doesn't copy any pointed to objects
        # only their reference)
        emission_model = copy.copy(self)

        # Apply any overides we have
        self._apply_overrides(emission_model, dust_curves, tau_v, fesc, mask)

        # Define a dictionary to keep track of any spectra that need scaling
        # by other spectra
        scale_later = {}

        # Perform all extractions
        lines = self._extract_lines(
            line_ids,
            emission_model,
            emitters,
            per_particle,
            lines,
            verbose,
            **kwargs,
        )

        # With all base lines extracted we can now loop from bottom to top
        # of the tree creating each lines
        for label in emission_model._bottom_to_top:
            # Get this model
            this_model = emission_model._models[label]

            # Get the spectra for the related models that don't appear in the
            # tree
            for related_model in this_model.related_models:
                if related_model.label not in lines:
                    lines.update(
                        related_model._get_lines(
                            line_ids,
                            emitters,
                            per_particle,
                            dust_curves=dust_curves,
                            tau_v=tau_v,
                            fesc=fesc,
                            mask=mask,
                            verbose=verbose,
                            lines=lines,
                            **kwargs,
                        )
                    )

            # Skip models for a different emitters
            if (
                this_model.emitter not in emitters
                and this_model.emitter != "galaxy"
            ):
                continue

            # Get the emitter (as long as we aren't doing a combination for a
            # galaxy spectra
            if this_model.emitter != "galaxy":
                emitter = emitters[this_model.emitter]
            else:
                emitter = None

            # Do we have to define a mask?
            this_mask = None
            for mask_dict in this_model.masks:
                this_mask = emitter.get_mask(**mask_dict, mask=this_mask)

            # Are we doing a combination?
            if this_model._is_combining:
                lines = self._combine_lines(
                    line_ids,
                    emission_model,
                    lines,
                    this_model,
                )

            elif this_model._is_dust_attenuating:
                lines = self._dust_attenuate_lines(
                    line_ids,
                    this_model,
                    lines,
                    emitter,
                    this_mask,
                )

            elif this_model._is_dust_emitting or this_model._is_generating:
                lines = self._generate_lines(
                    line_ids,
                    this_model,
                    emission_model,
                    lines,
                    per_particle,
                    emitter,
                )

            # Are we scaling the spectra?
            for scaler in this_model.scale_by:
                if scaler is None:
                    continue
                elif hasattr(emitter, scaler):
                    for line_id in line_ids:
                        lines[label][line_id]._luminosity *= getattr(
                            emitter, scaler
                        )
                        lines[label][line_id]._continuum *= getattr(
                            emitter, scaler
                        )
                else:
                    scale_later.setdeafault(label, []).append(scaler)

        # Handle any final scalings we need to do
        for spec, scalers in scale_later.items():
            for scaler in scalers:
                for line_id in line_ids:
                    # Compute the scaling
                    scaling = (
                        lines[scaler]._luminosity / lines[spec]._luminosity
                    )

                    lines[spec] *= scaling

        # Finally, loop over everything we've created and convert the nested
        # dictionaries to LineCollections
        for label in lines:
            lines[label] = LineCollection(lines[label])

        return lines


class StellarEmissionModel(EmissionModel):
    """
    An emission model for stellar components.

    This is a simple wrapper to quickly apply that the emitter a model
    should act on is stellar.

    Attributes:
        emitter (str):
            The emitter this model is for.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a StellarEmissionModel instance."""
        EmissionModel.__init__(self, *args, **kwargs)
        self._emitter = "stellar"


class BlackHoleEmissionModel(EmissionModel):
    """
    An emission model for black hole components.

    This is a simple wrapper to quickly apply that the emitter a model
    should act on is a black hole.

    Attributes:
        emitter (str):
            The emitter this model is for.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a BlackHoleEmissionModel instance."""
        EmissionModel.__init__(self, *args, **kwargs)
        self._emitter = "blackhole"


class GalaxyEmissionModel(EmissionModel):
    """
    An emission model for whole galaxy.

    A galaxy model sets emitter to "galaxy" to flag to the get_spectra method
    that the model is for a galaxy. By definition a galaxy level spectra can
    only be a combination of component spectra.

    Attributes:
        emitter (str):
            The emitter this model is for.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a GalaxyEmissionModel instance."""
        EmissionModel.__init__(self, *args, **kwargs)
        self._emitter = "galaxy"

        # Ensure we are only combining
        if not self._is_combining:
            raise exceptions.InconsistentArguments(
                "A GalaxyEmissionModel must be either combining or dust "
                "attenuating."
            )
