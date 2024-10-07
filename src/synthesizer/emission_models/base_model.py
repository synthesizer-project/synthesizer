"""A module defining the emission model from which spectra are constructed.

Generating spectra involves the following steps:
1. Extraction from a Grid.
2. Generation of spectra.
3. Attenuation due to dust in the ISM/nebular.
4. Masking for different types of emission.
5. Combination of different types of emission.

An emission model defines the parameters necessary to perform these steps and
gives an interface for simply defining the construction of complex spectra.

Example usage::

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
from unyt import kpc, unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.operations import (
    Combination,
    DustAttenuation,
    Extraction,
    Generation,
)
from synthesizer.line import LineCollection
from synthesizer.units import Quantity, accepts
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

    A number of attributes are defined as properties to protect their values
    and ensure the tree is correctly reconstructed when they are changed.
    This also means the they are no included below in the Attributes section
    to avoid duplication.

    Attributes:
        label (str):
            The key for the spectra that will be produced.
        lam (unyt_array):
            The wavelength array.
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
        emitter (str):
            The emitter this emission model acts on. Default is "stellar".
        apply_dust_to (EmissionModel):
            The model to apply the dust curve to.
        dust_curve (emission_models.attenuation.*):
            The dust curve to apply.
        tau_v (float/ndarray/str/tuple):
            The optical depth to apply. Can be a float, ndarray, or a string
            to a component attribute. Can also be a tuple combining any of
            these.
        generator (EmissionModel):
            The emission generation model. This must define a get_spectra
            method.
        lum_intrinsic_model (EmissionModel):
            The intrinsic model to use deriving the dust luminosity when
            computing dust emission.
        lum_attenuated_model (EmissionModel):
            The attenuated model to use deriving the dust luminosity when
            computing dust emission.
        mask_attr (str):
            The component attribute to mask on.
        mask_thresh (unyt_quantity):
            The threshold for the mask.
        mask_op (str):
            The operation to apply. Can be "<", ">", "<=", ">=", "==", or "!=".
        fesc (float):
            The escape fraction.
        scale_by (list):
            A list of attributes to scale the spectra by.
        post_processing (list):
            A list of post processing functions to apply to the emission after
            it has been generated. Each function must take a dict containing
            the spectra/lines, the emitters, and the emission model, and return
            the same dict with the post processing applied.
        save (bool):
            A flag for whether the emission produced by this model should be
            "saved", i.e. attached to the emitter. If False, the emission will
            be discarded after it has been used. Default is True.
        per_particle (bool):
            A flag for whether the emission produced by this model should be
            "per particle". If True, the spectra and lines will be stored per
            particle. Integrated spectra are made automatically by summing the
            per particle spectra. Default is False.
    """

    # Define quantities
    lam = Quantity()

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
        lum_intrinsic_model=None,
        lum_attenuated_model=None,
        mask_attr=None,
        mask_thresh=None,
        mask_op=None,
        fesc=None,
        related_models=None,
        emitter=None,
        fixed_parameters={},
        scale_by=None,
        post_processing=(),
        save=True,
        per_particle=False,
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
            dust_curve (emission_models.attenuation.*):
                The dust curve to apply.
            tau_v (float/ndarray/str/tuple):
                The optical depth to apply. Can be a float, ndarray, or a
                string to a component attribute. Can also be a tuple combining
                any of these.
            generator (EmissionModel):
                The emission generation model. This must define a get_spectra
                method.
            lum_intrinsic_model_key (EmissionModel):
                The intrinsic model to use deriving the dust
                luminosity when computing dust emission.
            lum_attenuated_model_key (EmissionModel):
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
            related_models (set/list/EmissionModel):
                A set of related models to this model. A related model is a
                model that is connected somewhere within the model tree but is
                required in the construction of the "root" model encapulated by
                self.
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
            post_processing (list):
                A list of post processing functions to apply to the emission
                after it has been generated. Each function must take a dict
                containing the spectra/lines, the emitters, and the emission
                model, and return the same dict with the post processing
                applied.
            save (bool):
                A flag for whether the emission produced by this model should
                be "saved", i.e. attached to the emitter. If False, the
                emission will be discarded after it has been used. Default is
                True.
            per_particle (bool):
                A flag for whether the emission produced by this model should
                be "per particle". If True, the spectra and lines will be
                stored per particle. Integrated spectra are made automatically
                by summing the per particle spectra. Default is False.
            **kwargs:
                Any additional keyword arguments to store. These can be used
                to store additional information needed by the model.
        """
        # What is the key for the spectra that will be produced?
        self.label = label

        # Attach the wavelength array and store it on the model
        if grid is not None:
            self.lam = grid.lam
        elif generator is not None and hasattr(generator, "lam"):
            self.lam = generator.lam
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

        # Are we making per particle emission?
        self._per_particle = per_particle

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
            lum_intrinsic_model=lum_intrinsic_model,
            lum_attenuated_model=lum_attenuated_model,
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
            lum_intrinsic_model=lum_intrinsic_model,
            lum_attenuated_model=lum_attenuated_model,
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
        elif scale_by is None:
            self._scale_by = ()
        else:
            self._scale_by = (scale_by,)

        # Store the post processing functions
        self._post_processing = post_processing

        # Attach the related models
        if related_models is None:
            self.related_models = set()
        elif isinstance(related_models, set):
            self.related_models = related_models
        elif isinstance(related_models, list):
            self.related_models = set(related_models)
        elif isinstance(related_models, tuple):
            self.related_models = set(related_models)
        elif isinstance(related_models, EmissionModel):
            self.related_models = {
                related_models,
            }
        else:
            raise exceptions.InconsistentArguments(
                "related_models must be a set, list, tuple, or EmissionModel."
            )

        # Are we saving this emission?
        self._save = save

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
        lum_intrinsic_model,
        lum_attenuated_model,
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
            dust_curve (emission_models.attenuation.*):
                The dust curve to apply.
            tau_v (float/ndarray/str/tuple):
                The optical depth to apply. Can be a float, ndarray, or a
                string to a component attribute. Can also be a tuple combining
                any of these.
            generator (EmissionModel):
                The emission generation model. This must define a get_spectra
                method.
            lum_intrinsic_model (EmissionModel):
                The intrinsic model to use deriving the dust
                luminosity when computing dust emission.
            lum_attenuated_model (EmissionModel):
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
                self, generator, lum_intrinsic_model, lum_attenuated_model
            )
        elif self._is_generating:
            Generation.__init__(self, generator, lum_intrinsic_model, None)
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
                f"lum_intrinsic_model={lum_intrinsic_model}, "
                f"lum_attenuated_model={lum_attenuated_model})"
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
                f"lum_intrinsic_model={lum_intrinsic_model}, "
                f"lum_attenuated_model={lum_attenuated_model})"
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

            # Report if the resulting emission will be saved
            parts.append(f"  Save emission: {model._save}")

            # Report if the resulting emission will be per particle
            if model._per_particle:
                parts.append("  Per particle emission: True")

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

            # Report any attributes we are scaling by
            if len(model.scale_by) > 0:
                parts.append("  Scaling by:")
                for scale_by in model._scale_by:
                    parts.append(f"    - {scale_by}")

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
        lum_attenuated_model=None,
        lum_intrinsic_model=None,
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
            and lum_attenuated_model is not None
            and lum_intrinsic_model is not None
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
        # Store the model (ensuring the label isn't already used for a
        # different model)
        if model.label not in self._models:
            self._models[model.label] = model
        elif self._models[model.label] is model:
            # The model is already in the tree so nothing to do
            pass
        else:
            raise exceptions.InconsistentArguments(
                f"Label {model.label} is already in use."
            )

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
            if model._lum_attenuated_model is not None:
                model._children.add(model._lum_attenuated_model)
                model._lum_attenuated_model._parents.add(model)
            if model._lum_intrinsic_model is not None:
                model._children.add(model._lum_intrinsic_model)
                model._lum_intrinsic_model._parents.add(model)

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

        # Collect any related models
        self.related_models.update(model.related_models)

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

    def _set_attr(self, attr, value):
        """
        Set an attribute on the model.

        Args:
            attr (str): The attribute to set.
            value (Any): The value to set the attribute to.
        """
        if hasattr(self, attr):
            setattr(self, "_" + attr, value)
        else:
            raise exceptions.InconsistentArguments(
                f"Cannot set attribute {attr} on model {self.label}. The "
                "model is not compatible with this attribute."
            )

    @property
    def grid(self):
        """Get the Grid object used for extraction."""
        return getattr(self, "_grid", None)

    def set_grid(self, grid, set_all=False):
        """
        Set the grid to extract from.

        Args:
            grid (Grid):
                The grid to extract from.
            set_all (bool):
                Whether to set the grid on all models.
        """
        # Set the grid
        if not set_all:
            if self._is_extracting:
                self._set_attr("grid", grid)
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set a grid on a model that is not extracting."
                )
        else:
            for model in self._models.values():
                if self._is_extracting:
                    model.set_grid(grid)

        # Unpack the model now we're done
        self.unpack_model()

        # Set the wavelength array at the root
        self.lam = grid.lam

    @property
    def extract(self):
        """Get the key for the spectra to extract."""
        return getattr(self, "_extract", None)

    def set_extract(self, extract):
        """
        Set the spectra to extract from the grid.

        Args:
            extract (str):
                The key of the spectra to extract.
        """
        # Set the extraction key
        if self._is_extracting:
            self._extract = extract
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set an extraction key on a model that is "
                "not extracting."
            )

        # Unpack the model now we're done
        self.unpack_model()

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

    def set_dust_props(
        self,
        dust_curve=None,
        apply_dust_to=None,
        tau_v=None,
        set_all=False,
    ):
        """
        Set the dust attenuation properties on this model.

        Args:
            dust_curve (emission_models.attenuation.*):
                A dust curve instance to apply.
            apply_dust_to (EmissionModel):
                The model to apply the dust curve to.
            tau_v (float/ndarray/str/tuple):
                The optical depth to apply. Can be a float, ndarray, or a
                string to a component attribute. Can also be a tuple combining
                any of these. If a tuple then the eventual tau_v will be the
                product of all contributors.
            set_all (bool):
                Whether to set the properties on all models.
        """
        # Set the properties
        if not set_all:
            if self._is_dust_attenuating:
                if dust_curve is not None:
                    self._set_attr("dust_curve", dust_curve)
                if apply_dust_to is not None:
                    self._set_attr("apply_dust_to", apply_dust_to)
                if tau_v is not None:
                    self._set_attr(
                        "tau_v",
                        tau_v if isinstance(tau_v, (tuple, list)) else [tau_v],
                    )
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set dust attenuation properties on a model that "
                    "is not dust attenuating."
                )
        else:
            for model in self._models.values():
                if self._is_dust_attenuating:
                    model.set_dust_props(
                        dust_curve=dust_curve,
                        apply_dust_to=apply_dust_to,
                        tau_v=tau_v,
                    )

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def generator(self):
        """Get the emission generation model."""
        return getattr(self, "_generator", None)

    def set_generator(self, generator):
        """
        Set the dust emission model on this model.

        Args:
            generator (EmissionModel):
                The emission generation model to set.
            label (str):
                The label of the model to set the dust emission model on. If
                None, sets the dust emission model on this model.
        """
        # Ensure model is a emission generation model and change the model
        if self._models._is_dust_emitting or self._models._is_generating:
            self._set_attr("generator", generator)
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set a dust emission model on a model that is not "
                "dust emitting."
            )

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def emitter(self):
        """Get the emitter this emission model acts on."""
        return getattr(self, "_emitter", None)

    def set_emitter(self, emitter, set_all=False):
        """
        Set the emitter this emission model acts on.

        Args:
            emitter (str):
                The emitter this emission model acts on.
            set_all (bool):
                Whether to set the emitter on all models.
        """
        # Set the emitter
        if not set_all:
            self._set_attr("emitter", emitter)
        else:
            for model in self._models.values():
                model.set_emitter(emitter)

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def per_particle(self):
        """Get the per particle flag."""
        return self._per_particle

    def set_per_particle(self, per_particle):
        """
        Set the per particle flag.

        For per particle spectra we need all children to also be per particle.

        Args:
            per_particle (bool):
                Whether to set the per particle flag.
            set_all (bool):
                Whether to set the per particle flag on all models.
        """
        # Set the per particle flag (but we don't want to set it on a
        # galaxy model since they are never per particle by definition)
        if self.emitter != "galaxy":
            self._per_particle = per_particle

        # Set the per particle flag on all children
        for model in self._children:
            model.set_per_particle(per_particle)

        # If this model also has related spectra we'd better make those
        # per particle too or bad things will happen downstream
        for model in self.related_models:
            model.set_per_particle(per_particle)

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def lum_intrinsic_model(self):
        """Get the intrinsic model for computing dust luminosity."""
        return getattr(self, "_lum_intrinsic_model", None)

    def set_lum_intrinsic_model(self, lum_intrinsic_model):
        """
        Set the intrinsic model for computing dust luminosity.

        Args:
            lum_intrinsic_model (EmissionModel):
                The intrinsic model to set.
        """
        # Ensure model is a emission generation model and change the model
        if self._models._is_dust_emitting:
            self._set_attr("lum_intrinsic_model", lum_intrinsic_model)
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set an intrinsic model on a model that is not "
                "dust emitting."
            )

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def lum_attenuated_model(self):
        """Get the attenuated model for computing dust luminosity."""
        return getattr(self, "_lum_attenuated_model", None)

    def set_lum_attenuated_model(self, lum_attenuated_model):
        """
        Set the attenuated model for computing dust luminosity.

        Args:
            lum_attenuated_model (EmissionModel):
                The attenuated model to set.
        """
        # Ensure model is a emission generation model and change the model
        if self._models._is_dust_emitting:
            self._set_attr("lum_attenuated_model", lum_attenuated_model)
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set an attenuated model on a model that is not "
                "dust emitting."
            )

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def combine(self):
        """Get the models to combine."""
        return getattr(self, "_combine", tuple())

    def set_combine(self, combine):
        """
        Set the models to combine on this model.

        Args:
            combine (list):
                A list of models to combine.
        """
        # Ensure all models are EmissionModels
        for model in combine:
            if not isinstance(model, EmissionModel):
                raise exceptions.InconsistentArguments(
                    "All models to combine must be EmissionModels."
                )

        # Set the models to combine ensurign the model we are setting on is
        # a combination step
        if self._is_combining:
            self._combine = combine
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set models to combine on a model that is not "
                "combining."
            )

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def fesc(self):
        """Get the escape fraction."""
        return getattr(self, "_fesc", 0.0)

    def set_fesc(self, fesc, set_all=False):
        """
        Set the escape fraction on this model.

        Args:
            fesc (float):
                The escape fraction to set.
            set_all (bool):
                Whether to set the escape fraction on all models.
        """
        # Set the escape fraction
        if not set_all:
            if self._is_extracting:
                self._fesc = fesc
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set an escape fraction on a model that is not "
                    "extracting."
                )
        else:
            for model in self._models.values():
                if model._is_extracting or model._is_generating:
                    model.set_fesc(fesc)

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def scale_by(self):
        """Get the attribute to scale the spectra by."""
        return self._scale_by

    def set_scale_by(self, scale_by, set_all=False):
        """
        Set the attribute to scale the spectra by.

        Args:
            scale_by (str/list/tuple/EmissionModel):
                Either a component attribute to scale the resultant spectra by,
                a spectra key to scale by (based on the bolometric luminosity).
                or a tuple/list containing strings defining either of the
                former two options. Instead of a string, an EmissionModel can
                be passed to scale by the luminosity of that model.
            set_all (bool):
                Whether to set the scale by attribute on all models.
        """
        # Set the attribute to scale by
        if not set_all:
            if isinstance(scale_by, (list, tuple)):
                self._scale_by = scale_by
                self._scale_by = [
                    s if isinstance(s, str) else s.label for s in scale_by
                ]
            elif isinstance(scale_by, EmissionModel):
                self._scale_by = (scale_by.label,)
            elif scale_by is None:
                self._scale_by = ()
            else:
                self._scale_by = (scale_by,)
        else:
            for model in self._models.values():
                model.set_scale_by(scale_by)

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def post_processing(self):
        """Get the post processing functions."""
        return getattr(self, "_post_processing", [])

    def set_post_processing(self, post_processing, set_all=False):
        """
        Set the post processing functions on this model.

        Args:
            post_processing (list):
                A list of post processing functions to apply to the emission
                after it has been generated. Each function must take a dict
                containing the spectra/lines and return the same dict with the
                post processing applied.
            set_all (bool):
                Whether to set the post processing functions on all models.
        """
        # Set the post processing functions
        if not set_all:
            self._post_processing = list(self._post_processing)
            self._post_processing.extend(post_processing)
            self._post_processing = tuple(set(self._post_processing))
        else:
            for model in self._models.values():
                model.set_post_processing(post_processing)

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def save(self):
        """Get the flag for whether to save the emission."""
        return getattr(self, "_save", True)

    def set_save(self, save, set_all=False):
        """
        Set the flag for whether to save the emission.

        Args:
            save (bool):
                Whether to save the emission.
            set_all (bool):
                Whether to set the save flag on all models.
        """
        # Set the save flag
        if not set_all:
            self._save = save
        else:
            for model in self._models.values():
                model.set_save(save)

        # Unpack the model now we're done
        self.unpack_model()

    def save_spectra(self, *args):
        """
        Set the save flag to True for the given spectra.

        Args:
            args (str):
                The spectra to save.
        """
        # First set all models to not save
        self.set_save(False, set_all=True)

        # Now set the given spectra to save
        for arg in args:
            self[arg].set_save(True)

    def add_mask(self, attr, op, thresh, set_all=False):
        """
        Add a mask.

        Args:
            attr (str):
                The component attribute to mask on.
            op (str):
                The operation to apply. Can be "<", ">", "<=", ">=", "==",
                or "!=".
            thresh (unyt_quantity):
                The threshold for the mask.
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

        # Add the mask
        if not set_all:
            self.masks.append({"attr": attr, "thresh": thresh, "op": op})
            self._is_masked = True
        else:
            for model in self._models.values():
                model.masks.append({"attr": attr, "thresh": thresh, "op": op})
                model._is_masked = True

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
            if parent.lum_intrinsic_model == replace_model:
                relations[parent.label] = "dust_intrinsic"
            if parent.lum_attenuated_model == replace_model:
                relations[parent.label] = "dust_attenuated"
        for child in children:
            if child in replace_model.combine:
                relations[child.label] = "combine"
            if child.apply_dust_to == replace_model:
                relations[child.label] = "dust_attenuate"
            if child.lum_intrinsic_model == replace_model:
                relations[child.label] = "dust_intrinsic"
            if child.lum_attenuated_model == replace_model:
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
                parent._lum_intrinsic_model = None
            if relations[parent.label] == "dust_attenuated":
                parent._lum_attenuated_model = None
        for child in children:
            child._parents.remove(replace_model)

        # Do we have more than 1 replacement?
        if len(replacements) > 1:
            # We'll have to include a combination model
            new_model = EmissionModel(
                label=replace_label if new_label is None else new_label,
                combine=replacements,
                emitter=self.emitter,
            )
        else:
            new_model = replacements[0]

            # Warn the user if they passed a new label, it won't be used
            if new_label is not None:
                warn(
                    "new_label is only used when multiple "
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
                parent._lum_intrinsic_model = new_model
            if relations[parent.label] == "dust_attenuated":
                parent._lum_attenuated_model = new_model

        # Unpack now we're done
        self.unpack_model()

    def relabel(self, old_label, new_label):
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

    def fix_parameters(self, **kwargs):
        """
        Fix parameters of the model.

        Args:
            **kwargs:
                The parameters to fix.
        """
        self.fixed_parameters.update(kwargs)

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
            if model._is_dust_attenuating:
                links.setdefault(label, []).append(
                    (model.apply_dust_to.label, "--")
                )
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
        root_model = self._models[root]

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
        self, root=None, show=True, fontsize=10, figsize=(6, 6)
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
            components.add(self[node].emitter)
            if self[node].emitter == "stellar":
                color = "gold"
            elif self[node].emitter == "blackhole":
                color = "royalblue"
            else:
                color = "forestgreen"

            # If the model isn't saved apply some transparency
            if not self[node].save:
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
            if self[node].per_particle:
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

    def _apply_overrides(
        self, emission_model, dust_curves, tau_v, fesc, covering_fraction, mask
    ):
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
        """
        # If we have dust curves to apply, apply them
        if dust_curves is not None:
            if isinstance(dust_curves, dict):
                for label, dust_curve in dust_curves.items():
                    emission_model._models[label]._dust_curve = dust_curve
            else:
                for model in emission_model._models.values():
                    model._dust_curve = dust_curves

        # If we have optical depths to apply, apply them
        if tau_v is not None:
            if isinstance(tau_v, dict):
                for label, value in tau_v.items():
                    emission_model._models[label]._tau_v = (
                        (value,)
                        if isinstance(value, (float, "str"))
                        else value
                    )
            else:
                for model in emission_model._models.values():
                    model._tau_v = (tau_v,)

        # If we have escape fractions to apply, apply them
        if fesc is not None:
            if isinstance(fesc, dict):
                for label, value in fesc.items():
                    emission_model._models[label]._fesc = value
            else:
                for model in emission_model._models.values():
                    model._fesc = fesc

        # If we have covering fractions to apply, apply them
        if covering_fraction is not None:
            if isinstance(covering_fraction, dict):
                for label, value in covering_fraction.items():
                    emission_model._models[label]._covering_fraction = value
            else:
                for model in emission_model._models.values():
                    model._covering_fraction = covering_fraction

        # If we have masks to apply, apply them
        if mask is not None:
            for label, mask in mask.items():
                emission_model[label].add_mask(**mask)

    def _get_spectra(
        self,
        emitters,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        spectra=None,
        particle_spectra=None,
        _is_related=False,
        **kwargs,
    ):
        """
        Generate stellar spectra as described by the emission model.

        NOTE: post processing methods defined on the model will be called
        once all spectra are made (these models are preceeded by post_ and
        take the dictionary of lines/spectra as an argument).

        Args:
            emitters (Stars/BlackHoles):
                The emitters to generate the spectra for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
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
            particle_spectra (dict)
                A dictionary of particle spectra to add to. This is used for
                recursive calls to this function.
            _is_related (bool)
                Are we generating related model spectra? If so we don't want
                to apply any post processing functions or delete any spectra,
                this will be done outside the recursive call.
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

        # Before we do anything, check that we have the emitters we need
        for model in emission_model._models.values():
            # Galaxy is always missing
            if model.emitter == "galaxy":
                continue
            if emitters.get(model.emitter, None) is None:
                raise exceptions.InconsistentArguments(
                    f"Missing {model.emitter} in emitters."
                )

        # Apply any overides we have
        self._apply_overrides(
            emission_model, dust_curves, tau_v, fesc, covering_fraction, mask
        )

        # Make a spectra dictionary if we haven't got one yet
        if spectra is None:
            spectra = {}
        if particle_spectra is None:
            particle_spectra = {}

        # We need to make sure the root is being saved, otherwise this is a bit
        # nonsensical.
        if not _is_related and not self.save:
            raise exceptions.InconsistentArguments(
                f"{self.label} is not being saved. There's no point in "
                "generating at the root if they are not saved. Maybe you "
                "want to use a child model you are saving instead?"
            )

        # Perform all extractions
        spectra, particle_spectra = self._extract_spectra(
            emission_model,
            emitters,
            spectra,
            particle_spectra,
            verbose,
            **kwargs,
        )

        # With all base spectra extracted we can now loop from bottom to top
        # of the tree creating each spectra
        for label in emission_model._bottom_to_top:
            # Get this model
            this_model = emission_model._models[label]

            # Get the spectra for the related models that don't appear in the
            # tree
            for related_model in this_model.related_models:
                if related_model.label not in spectra:
                    (
                        rel_spectra,
                        rel_particle_spectra,
                    ) = related_model._get_spectra(
                        emitters,
                        dust_curves=dust_curves,
                        tau_v=tau_v,
                        fesc=fesc,
                        mask=mask,
                        verbose=verbose,
                        spectra=spectra,
                        particle_spectra=particle_spectra,
                        _is_related=True,
                        **kwargs,
                    )

                    spectra.update(rel_spectra)
                    particle_spectra.update(rel_particle_spectra)

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
                try:
                    spectra, particle_spectra = self._combine_spectra(
                        emission_model,
                        spectra,
                        particle_spectra,
                        this_model,
                    )
                except Exception as e:
                    print(f"Error in {this_model.label}!")
                    raise e

            elif this_model._is_dust_attenuating:
                try:
                    spectra, particle_spectra = self._dust_attenuate_spectra(
                        this_model,
                        spectra,
                        particle_spectra,
                        emitter,
                        this_mask,
                    )
                except Exception as e:
                    print(f"Error in {this_model.label}!")
                    raise e

            elif this_model._is_dust_emitting or this_model._is_generating:
                try:
                    spectra, particle_spectra = self._generate_spectra(
                        this_model,
                        emission_model,
                        spectra,
                        particle_spectra,
                        self.lam,
                        emitter,
                    )
                except Exception as e:
                    print(f"Error in {this_model.label}!")
                    raise e

            # Are we scaling the spectra?
            for scaler in this_model.scale_by:
                if scaler is None:
                    continue
                if hasattr(emitter, scaler):
                    # Ok, the emitter has this scaler, get it
                    scaler_arr = getattr(emitter, f"_{scaler}", None)
                    if scaler_arr is None:
                        scaler_arr = getattr(emitter, scaler)

                    # Ensure we can actually multiply the spectra by it
                    if (
                        scaler_arr.ndim == 1
                        and spectra[label].shape[0] != scaler_arr.shape[0]
                    ):
                        raise exceptions.InconsistentArguments(
                            f"Can't scale a spectra with shape "
                            f"{spectra[label].shape} by an array of "
                            f"shape {scaler_arr.shape}. Did you mean to "
                            "make particle spectra?"
                        )
                    elif (
                        scaler_arr.ndim == spectra[label].ndim
                        and scaler_arr.shape != spectra[label].shape
                    ):
                        raise exceptions.InconsistentArguments(
                            f"Can't scale a spectra with shape "
                            f"{spectra[label].shape} by an array of "
                            f"shape {scaler_arr.shape}. Did you mean to "
                            "make particle spectra?"
                        )

                    # Scale the spectra by this attribute
                    if this_model.per_particle:
                        particle_spectra[label] *= scaler_arr
                    spectra[label]._lnu *= scaler_arr

                elif scaler in spectra:
                    # Compute the scaling
                    if this_model.per_particle:
                        scaling = (
                            particle_spectra[scaler].bolometric_luminosity
                            / particle_spectra[label].bolometric_luminosity
                        ).value
                    else:
                        scaling = (
                            spectra[scaler].bolometric_luminosity
                            / spectra[label].bolometric_luminosity
                        ).value

                    # Scale the spectra (handling 1D and 2D cases)
                    if this_model.per_particle:
                        particle_spectra[label] *= scaling[:, None]
                    spectra[label]._lnu *= scaling

                else:
                    raise exceptions.InconsistentArguments(
                        f"Can't scale spectra by {scaler}."
                    )

        # Only apply post processing and deletion if we aren't in a recursive
        # related model call
        if not _is_related:
            # Apply any post processing functions
            for func in self._post_processing:
                spectra = func(spectra, emitters, self)
                if len(particle_spectra) > 0:
                    particle_spectra = func(particle_spectra, emitters, self)

            # Loop over all models and delete those spectra if we aren't saving
            # them (we have to this after post processing incase the deleted
            # spectra are needed during post processing)
            for model in emission_model._models.values():
                if not model.save and model.label in spectra:
                    del spectra[model.label]
                    if model.per_particle and model.label in particle_spectra:
                        del particle_spectra[model.label]

        return spectra, particle_spectra

    def _get_lines(
        self,
        line_ids,
        emitters,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        covering_fraction=None,
        mask=None,
        verbose=True,
        lines=None,
        particle_lines=None,
        _is_related=False,
        **kwargs,
    ):
        """
        Generate stellar lines as described by the emission model.

        NOTE: post processing methods defined on the model will be called
        once all spectra are made (these models are preceeded by post_ and
        take the dictionary of lines/spectra as an argument).

        Args:
            line_ids (list):
                The line ids to extract.
            emitters (Stars/BlackHoles):
                The emitters to generate the lines for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
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
            lines (dict)
                A dictionary of lines to add to. This is used for recursive
                calls to this function.
            particle_lines (dict)
                A dictionary of particle lines to add to. This is used for
                recursive calls to this function.
            _is_related (bool)
                Are we generating related model lines? If so we don't want
                to apply any post processing functions or delete any lines,
                this will be done outside the recursive call.
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
        self._apply_overrides(
            emission_model, dust_curves, tau_v, fesc, covering_fraction, mask
        )

        # If we haven't got a lines dictionary yet we'll make one
        if lines is None:
            lines = {}
        if particle_lines is None:
            particle_lines = {}

        # We need to make sure the root is being saved, otherwise this is a bit
        # nonsensical.
        if not _is_related and not self.save:
            raise exceptions.InconsistentArguments(
                f"{self.label} is not being saved. There's no point in "
                "generating at the root if they are not saved. Maybe you "
                "want to use a child model you are saving instead?"
            )

        # Perform all extractions
        lines, particle_lines = self._extract_lines(
            line_ids,
            emission_model,
            emitters,
            lines,
            particle_lines,
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
                    rel_lines, rel_particle_lines = related_model._get_lines(
                        line_ids,
                        emitters,
                        dust_curves=dust_curves,
                        tau_v=tau_v,
                        fesc=fesc,
                        mask=mask,
                        verbose=verbose,
                        lines=lines,
                        particle_lines=particle_lines,
                        _is_related=True,
                        **kwargs,
                    )

                    lines.update(rel_lines)
                    particle_lines.update(rel_particle_lines)

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
                try:
                    lines, particle_lines = self._combine_lines(
                        line_ids,
                        emission_model,
                        lines,
                        particle_lines,
                        this_model,
                    )
                except Exception as e:
                    print(f"Error in {this_model.label}!")
                    raise e

            elif this_model._is_dust_attenuating:
                try:
                    lines, particle_lines = self._dust_attenuate_lines(
                        line_ids,
                        this_model,
                        lines,
                        particle_lines,
                        emitter,
                        this_mask,
                    )
                except Exception as e:
                    print(f"Error in {this_model.label}!")
                    raise e

            elif this_model._is_dust_emitting or this_model._is_generating:
                try:
                    lines, particle_lines = self._generate_lines(
                        line_ids,
                        this_model,
                        emission_model,
                        lines,
                        particle_lines,
                        emitter,
                    )
                except Exception as e:
                    print(f"Error in {this_model.label}!")
                    raise e

            # Are we scaling the spectra?
            for scaler in this_model.scale_by:
                if scaler is None:
                    continue
                elif hasattr(emitter, scaler):
                    for line_id in line_ids:
                        if this_model.per_particle:
                            particle_lines[label][
                                line_id
                            ]._luminosity *= getattr(emitter, scaler)
                            particle_lines[label][
                                line_id
                            ]._continuum *= getattr(emitter, scaler)
                        lines[label][line_id]._luminosity *= getattr(
                            emitter, scaler
                        )
                        lines[label][line_id]._continuum *= getattr(
                            emitter, scaler
                        )
                else:
                    raise exceptions.InconsistentArguments(
                        f"Can't scale lines by {scaler}."
                    )

        # Only convert to LineCollections, apply post processing and deletion
        # if we aren't in a recursive related model call
        if not _is_related:
            # Finally, loop over everything we've created and convert the
            # nested dictionaries to LineCollections
            for label in lines:
                # If we are in a related model we might have already done this
                # conversion
                if isinstance(lines[label], dict):
                    lines[label] = LineCollection(lines[label])
                if self._models[label].per_particle and isinstance(
                    particle_lines[label], dict
                ):
                    particle_lines[label] = LineCollection(
                        particle_lines[label]
                    )

            # Apply any post processing functions
            for func in self._post_processing:
                lines = func(lines, emitters, self)
                if len(particle_lines) > 0:
                    particle_lines = func(particle_lines, emitters, self)

            # Loop over all models and delete those lines if we aren't saving
            # them (we have to this after post processing incase the deleted
            # lines are needed during post processing)
            for model in emission_model._models.values():
                if not model.save and model.label in lines:
                    del lines[model.label]
                    if model.per_particle and model.label in particle_lines:
                        del particle_lines[model.label]

        return lines, particle_lines

    @accepts(resolution=kpc, fov=kpc)
    def _get_images(
        self,
        resolution,
        fov,
        emitters,
        img_type="smoothed",
        mask=None,  # TODO: impelement the masking properly
        verbose=True,  # TODO: Don't think this is needed
        images=None,
        _is_related=False,
        limit_to=None,
        do_flux=False,
        kernel=None,
        kernel_threshold=1.0,
        nthreads=1,
        **kwargs,
    ):
        """
        Generate images as described by the emission model.

        This will create images for all models in the emission model which
        have been saved in the passed dictionary of photometry, unless
        limit_to is set to a specific model, in which case only that model
        will have images generated (including any models required for that
        passed model, e.g. a combination of AGN and Stellar spectra).

        Note that unlike spectra or line creation, images are always either
        generated from existing photometry or combined from existing images
        further down in the tree.

        Args:
            resolution (float):
                The pixel resolution of the image in spatial units (e.g. pc,
                kpc, Mpc).
            fov (float):
                The field of view of the image in angular units (e.g. arcsec,
                arcmin, deg).
            emitters (Stars/BlackHoles):
                The emitters to generate the lines for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
            mask (dict):
                An overide to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            verbose (bool)
                Are we talking?
            images (dict)
                A dictionary of images to add to. This is used for recursive
                calls to this function.
            _is_related (bool)
                Are we generating related model lines? If so we don't want
                to apply any post processing functions or delete any lines,
                this will be done outside the recursive call.
            limit_to (str)
                If not None, defines a specifical model to limit the image
                generation to. Otherwise, all models with saved spectra will
                have images generated.
            do_flux (bool)
                If True, the images will be generated from fluxes, if False
                they will be generated from luminosities.
            kwargs (dict)
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of ImageCollections which can be attached to the
                appropriate images attribute of the component.
        """
        # We don't want to modify the original emission model with any
        # modifications made here so we'll make a copy of it (this is a
        # shallow copy so very cheap and doesn't copy any pointed to objects
        # only their reference)
        emission_model = copy.copy(self)

        # Apply any overides we have
        self._apply_overrides(
            emission_model,
            None,
            None,
            None,
            None,
            mask,
        )

        # If we haven't got an images dictionary yet we'll make one
        if images is None:
            images = {}

        # Get all the images at the extraction leaves of the tree
        images.update(
            self._extract_images(
                resolution,
                fov,
                img_type,
                do_flux,
                emitters,
                images,
                kernel,
                kernel_threshold,
                nthreads,
                limit_to,
            )
        )

        # Loop through the models from bottom to top order creating the
        # images as we go
        for label in emission_model._bottom_to_top:
            # If we are limiting to a specific model, skip all others
            if limit_to is not None and label != limit_to:
                continue

            # Get this model
            this_model = emission_model._models[label]

            # Get the images for the related models that don't appear in the
            # main tree
            for related_model in this_model.related_models:
                if related_model.label not in images:
                    images.update(
                        related_model._get_images(
                            resolution,
                            fov,
                            emitters,
                            img_type,
                            mask,
                            verbose,
                            images,
                            _is_related=True,
                            limit_to=limit_to,
                            do_flux=do_flux,
                            kernel=kernel,
                            kernel_threshold=kernel_threshold,
                            nthreads=nthreads,
                            **kwargs,
                        )
                    )

            # Skip models for a different emitters
            if (
                this_model.emitter not in emitters
                and this_model.emitter != "galaxy"
            ):
                continue

            # Check we haven't already made this image
            if label in images:
                continue

            # Get the emitter
            emitter = (
                emitters[this_model.emitter]
                if this_model.emitter != "galaxy"
                else None
            )

            # Call the appropriate method to generate the image for this model
            if this_model._is_combining:
                images = self._combine_images(
                    images,
                    this_model,
                )
            elif this_model._is_dust_attenuating:
                images = self._attenuate_images(
                    resolution,
                    fov,
                    this_model,
                    img_type,
                    do_flux,
                    emitter,
                    images,
                    kernel,
                    kernel_threshold,
                    nthreads,
                )
            elif this_model._is_dust_emitting or this_model._is_generating:
                images = self._generate_images(
                    resolution,
                    fov,
                    this_model,
                    img_type,
                    do_flux,
                    emitter,
                    images,
                    kernel,
                    kernel_threshold,
                    nthreads,
                )

        return images


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

        # Ensure we aren't extracting, this cannot be done for a galaxy.
        if self._is_extracting:
            raise exceptions.InconsistentArguments(
                "A GalaxyEmissionModel cannot be an extraction model."
            )

        # Ensure we aren't trying to make a per particle galaxy emission
        if self.per_particle:
            raise exceptions.InconsistentArguments(
                "A GalaxyEmissionModel cannot be a per particle model."
            )
