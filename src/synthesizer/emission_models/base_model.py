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
        apply_to=dust_emission_model,
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
import fnmatch
import sys

import numpy as np
from unyt import unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.model_queue import ModelQueue
from synthesizer.emission_models.operations import (
    Combination,
    Extraction,
    Generation,
    Transformation,
)
from synthesizer.emission_models.parameters import VARIATION_TYPES
from synthesizer.synth_warnings import deprecated, warn
from synthesizer.units import Quantity
from synthesizer.utils.operation_timers import timed, timer


class EmissionModel(Extraction, Generation, Transformation, Combination):
    """A class to define the construction of an emission from a grid.

    An emission can either be a spectra (Sed) or a set of
    lines (LineCollection).

    An emission model describes the steps necessary to construct a emissions
    from an emitter (a galaxy or one of its components). These steps can be:
    - Extracting an emission from a grid.
    - Combining multiple emissions.
    - Transforming an emission (e.g. applying a dust curve).
    - Generating an emission (e.g. dust emission).

    All of these stages can also have masks applied to them to remove certain
    elements from the emission (e.g. if you want to eliminate young stars).
    Note that regardless of any masks all emissions will be the same shape
    (i.e. have the same number of elements but with some set to zero based
    on the mask).

    By chaining together multiple emission models, complex emissions can be
    constructed from parametric of particle based inputs. This chained
    together network of models we call the tree. The tree has a single
    model at it's root which is the model that will be used directly called
    from by the user. Each model in the tree is connected to at least one
    other model in the tree. Each node can also have related models which
    may not be directly connected to the root model but can still be used
    to generate an emission.

    Note: Every time a property is changed anywhere in the model tree the
    tree must be reconstructed. This is a cheap process but must happen to
    ensure the expected properties are used when the model is used. This means
    most attributes are accessed through properties and set via setters to
    ensure the tree remains consistent.

    A number of attributes are defined as properties to protect their values
    and ensure the tree is correctly reconstructed when they are changed.

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
            in the construction of the "root" model encapsulated by self.
        fixed_parameters (dict):
            A dictionary of component attributes/parameters which should be
            fixed and thus ignore the value of the component attribute. This
            should take the form {<parameter_name>: <value>}.
        emitter (str):
            The emitter this emission model acts on.
        apply_to (EmissionModel):
            The model to apply the dust curve to.
        dust_curve (emission_models.attenuation.*):
            The dust curve to apply.
        generator (EmissionModel):
            The emission generation model. This must inherit from a Generator
            base class and thus define _generate_spectra and _generate_lines.
        mask_attr (str):
            The component attribute to mask on.
        mask_thresh (unyt_quantity):
            The threshold for the mask.
        mask_op (str):
            The operation to apply. Can be "<", ">", "<=", ">=", "==", or "!=".
        lam_mask (ndarray):
            The mask to apply to the wavelength array.
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
        vel_shift (bool):
            A flag for whether the emission produced by this model should take
            into account the velocity shift due to peculiar velocities.
            Only applicable to particle based emitters. Default is False.
    """

    # Define quantities
    lam = Quantity("wavelength")

    @timed("EmissionModel.__init__")
    def __init__(
        self,
        label,
        grid=None,
        extract=None,
        combine=None,
        apply_to=None,
        dust_curve=None,
        igm=None,
        generator=None,
        transformer=None,
        mask_attr=None,
        mask_thresh=None,
        mask_op=None,
        lam_mask=None,
        related_models=None,
        emitter=None,
        scale_by=None,
        post_processing=(),
        save=True,
        per_particle=False,
        vel_shift=False,
        **fixed_parameters,
    ):
        """Initialise the emission model.

        Each instance of an emission model describes a single step in the
        process of constructing an emission. These different steps can be:
        - Extracting an emission from a grid. (extract must be set)
        - Combining multiple emissions. (combine must be set with a list/tuple
          of child emission models).
        - Transforming an emission. (dust_curve/igm/transformer and apply_to
          must be set)
        - Generating an emission. (generator must be set)

        Within any of these steps a mask can be applied to the emission to
        remove certain elements (e.g. if you want to eliminate particles).

        Args:
            label (str):
                The key for the emission that will be produced.
            grid (Grid):
                The grid to extract from.
            extract (str):
                The key for the emission to extract from the input Grid.
            combine (list):
                A list of models to combine.
            apply_to (EmissionModel):
                The model to apply the transformer to.
            dust_curve (emission_models.attenuation.*):
                The dust curve to apply (when doing a transformation). This
                is a friendly alias argument for the transformer argument.
                Setting both will raise an exception.
            igm (emission_models.transformers.igm.*):
                The IGM model to apply (when doing a transformation). This is
                a friendly alias argument for the transformer argument.
                Setting both will raise an exception.
            generator (Generator):
                The emission generation model. This must inherit from a
                Generator base class and thus define _generate_spectra and
                _generate_lines.
            transformer (Transformer):
                The transform to apply. This is also an alternative (but
                less obvious) argument for passing a dust curve or IGM model
                (both are transformers).
            mask_attr (str):
                The component attribute to mask on.
            mask_thresh (unyt_quantity):
                The threshold for the mask.
            mask_op (str):
                The operation to apply. Can be "<", ">", "<=", ">=", "==",
                or "!=".
            lam_mask (ndarray):
                The mask to apply to the wavelength array.
            related_models (set/list/EmissionModel):
                A set of related models to this model. A related model is a
                model that is connected somewhere within the model tree but is
                required in the construction of the "root" model encapsulated
                by self.
            emitter (str):
                The emitter this emission model acts on. Default is
                "galaxy". Can be "stellar", "gas", "blackhole", or "galaxy".
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
            vel_shift (bool):
                A flag for whether the emission produced by this model should
                take into account the velocity shift due to peculiar
                velocities. Default is False.
            fixed_parameters (dict):
                Any extra keyword arguments will be treated as fixed parameters
                and stored in a dictionary. When the model looks to extract
                parameters needed for an operation, it will first check this
                dictionary for a fixed override to an emitter parameter.
        """
        # What is the key for the emission that will be produced?
        self.label = label

        # Attach the wavelength array and store it on the model
        if grid is not None:
            self.lam = grid.lam
        elif generator is not None and hasattr(generator, "lam"):
            self.lam = generator.lam
        else:
            self.lam = None

        # Store any fixed parameters
        self.fixed_parameters = fixed_parameters

        # Ensure we have been given an emitter
        if emitter is None:
            raise exceptions.InconsistentArguments(
                "Must specify an emitter, either 'stellar', 'gas', "
                "'blackhole', or 'galaxy'."
            )

        # Attach which emitter we are working with
        self._emitter = emitter.lower()

        # Ensure emitter is an acceptable value
        if self._emitter not in ["stellar", "gas", "blackhole", "galaxy"]:
            raise exceptions.InconsistentArguments(
                "Emitter must be either 'stellar', 'gas', 'blackhole', or "
                f"'galaxy' (got {self._emitter})."
            )
        elif self._emitter == "gas":
            raise exceptions.UnimplementedFunctionality(
                "The 'gas' emitter is not yet implemented."
            )
        else:
            # All good, as you were
            pass

        # Are we making per particle emission?
        self._per_particle = per_particle

        # Make sure we aren't trying to be a per particle galaxy model
        if self._per_particle and self._emitter == "galaxy":
            raise exceptions.InconsistentArguments(
                "Cannot make a per particle galaxy emission model."
            )

        # Define the container which will hold mask information
        self.masks = []

        # If we have been given a mask, add it
        self._init_masks(mask_attr, mask_thresh, mask_op)

        # Initialise the lam mask
        self._lam_mask = lam_mask

        # Reject removed apply_dust_to compatibility argument
        if "apply_dust_to" in fixed_parameters:
            raise exceptions.InconsistentArguments(
                "The apply_dust_to argument has been removed. "
                "Please use apply_to instead."
            )

        # Get operation flags
        self._get_operation_flags(
            grid=grid,
            extract=extract,
            combine=combine,
            dust_curve=dust_curve,
            igm=igm,
            transformer=transformer,
            apply_to=apply_to,
            generator=generator,
        )

        # Initialise the corresponding operation (also checks we have a
        # valid set of arguments and also have everything we need for the
        # operation)
        self._init_operations(
            grid=grid,
            extract=extract,
            combine=combine,
            apply_to=apply_to,
            transformer=(
                transformer
                if transformer is not None
                else dust_curve
                if dust_curve is not None
                else igm
            ),
            generator=generator,
            vel_shift=vel_shift,
        )

        # Containers for children and parents
        self._children = set()
        self._parents = set()

        # Store the attribute to scale the emission by
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

        # Friendly private pointers for generator dependencies
        self._energy_balance_models = None
        self._scaler_model = None

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
        apply_to,
        transformer,
        generator,
        vel_shift,
    ):
        """Initialise the correct parent operation.

        Args:
            grid (Grid):
                The grid to extract from.
            extract (str):
                The key for the emission to extract.
            combine (list):
                A list of models to combine.
            apply_to (EmissionModel):
                The model to apply the dust curve to.
            transformer (Transformer):
                The transformer to apply (can be a dust curve or IGM model, or
                any other transformer as long as it inherits from Transformer).
            generator (EmissionModel):
                The emission generation model. This must inherit from a
                Generator base class and thus define _generate_spectra and
                _generate_lines.
            vel_shift (bool):
                A flag for whether the emission produced by this model should
                take into account the velocity shift due to peculiar
                velocities. Particle emitters only.
        """
        # Which operation are we doing?
        if self._is_extracting:
            Extraction.__init__(self, grid, extract, vel_shift)
        elif self._is_combining:
            Combination.__init__(self, combine)
        elif self._is_transforming:
            Transformation.__init__(self, transformer, apply_to)
        elif self._is_generating:
            Generation.__init__(self, generator)
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
                f"(transformer={transformer}, "
                f"apply_to={apply_to})\n"
                "\tFor generation "
                f"(generator={generator}, "
            )

        # Double check we have been asked for only one operation
        if (
            sum(
                [
                    self._is_extracting,
                    self._is_combining,
                    self._is_transforming,
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
                f"transform: {self._is_transforming}, "
                f"generate: {self._is_generating}). "
                "Currently have:\n"
                "\tFor extraction: grid=("
                f"{grid.grid_name if grid is not None else None}"
                "\tFor combination: "
                f"(combine={combine})\n"
                "\tFor dust attenuation: "
                f"(transformer={transformer} "
                f"apply_to={apply_to})\n"
                "\tFor generation "
                f"(generator={generator}, "
            )

        # Ensure we have what we need for all operations
        if self._is_extracting and grid is None:
            raise exceptions.InconsistentArguments(
                "Must specify a grid to extract from."
            )
        if self._is_transforming and transformer is None:
            raise exceptions.InconsistentArguments(
                "Must specify a dust curve to apply."
            )
        if self._is_transforming and apply_to is None:
            raise exceptions.InconsistentArguments(
                "Must specify where to apply the dust curve."
            )
        if self._is_generating and generator is None:
            raise exceptions.InconsistentArguments(
                "Must specify a generator to generate emission."
            )

        # Ensure the grid contains any keys we want to extract
        if self._is_extracting and extract not in grid.available_emissions:
            raise exceptions.InconsistentArguments(
                f"Grid does not contain key: {extract}"
            )

    def _init_masks(self, mask_attr, mask_thresh, mask_op):
        """Initialise the mask operation.

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
            # Ensure mask_thresh comes with units (or is a string alias
            # that will be resolved to a unyt_quantity at generation time)
            if not isinstance(mask_thresh, (unyt_quantity, str)):
                raise exceptions.MissingUnits(
                    "Mask threshold must be given with units (or as a string "
                    "attribute name to be resolved from the emitter)."
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
        elif self._is_transforming:
            return self._transform_summary()
        elif self._is_generating:
            return self._generate_summary()
        else:
            return []

    def __str__(self):
        """Return a string summarising the model."""
        parts = []

        # Summarise models in their discovery order. A model which hasn't been
        # unpacked has no tree to summarise, so it summarises itself; this is
        # the case when a model is printed before it has been used, which an
        # error message may well do.
        models = list(self._models.values()) or [self]

        for model in models:
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

    def items(self):
        """Return the items in the model."""
        return self._models.items()

    def __getitem__(self, label):
        """Enable label look up.

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
        igm=None,
        transformer=None,
        apply_to=None,
        generator=None,
    ):
        """Define the flags for what operation the model does."""
        # Define flags for what we're doing
        self._is_extracting = extract is not None and grid is not None
        self._is_combining = combine is not None and len(combine) > 0
        self._is_transforming = (
            dust_curve is not None
            or igm is not None
            or transformer is not None
        ) and apply_to is not None
        self._is_generating = generator is not None

    def _unpack_model_recursively(self, model):
        """Traverse the model tree and collect what we will need to do.

        Args:
            model (EmissionModel): The model to unpack.
        """
        # Store the model (ensuring the label isn't already used for a
        # different model)
        if model.label not in self._models:
            self._models[model.label] = model
        elif self._models[model.label] is model:
            # The model is already in the tree so nothing to do
            return
        else:
            # Ensure model has a unique name
            if len(model.masks) > 0:
                for _m in model.masks:
                    model.label += (
                        f"_{_m['attr']}{_m['op']}{_m['thresh']}".replace(
                            " ", "-"
                        )
                    )
            else:
                raise exceptions.InconsistentArguments(
                    f"Label {model.label} is already in use by another model. "
                    f"Existing model: \n{self._models[model.label]}, \n"
                    f"New model: \n{model})"
                )

            self._models[model.label] = model

        # If we are extracting an emission, store the key
        if model._is_extracting:
            self._extract_keys[model.label] = model.extract

        # If we are applying a transform, store the key
        if model._is_transforming:
            # Nothing to do if apply_to is a string, then the model is a leaf
            if not isinstance(model.apply_to, str):
                model._children.add(model.apply_to)
                model.apply_to._parents.add(model)

        # If we are generating an emission, store the key
        if model._is_generating:
            # Are there any models attached to the generator?
            if (
                hasattr(model.generator, "_intrinsic")
                and hasattr(model.generator, "_attenuated")
                and model.generator._intrinsic is not None
                and model.generator._attenuated is not None
            ):
                child = model.generator._intrinsic
                model._children.add(child)
                child._parents.add(model)
                child = model.generator._attenuated
                model._children.add(child)
                child._parents.add(model)
                self._energy_balance_models = (
                    model.generator._intrinsic,
                    model.generator._attenuated,
                )
            if (
                hasattr(model.generator, "_scaler")
                and model.generator._scaler is not None
            ):
                child = model.generator._scaler
                model._children.add(child)
                child._parents.add(model)
                self._scaler_model = model.generator._scaler

        # If we are combining spectra, store the key
        if model._is_combining:
            for child in model.combine:
                # Only add as a child if it's not a string (strings are leaves)
                if not isinstance(child, str):
                    model._children.add(child)
                    child._parents.add(model)

        # Recurse over children
        for child in model._children:
            self._unpack_model_recursively(child)

        # Collect any related models
        self.related_models.update(model.related_models)

    def unpack_model(self):
        """Unpack the model tree to get the order of operations."""
        # Define the private containers we'll unpack everything into. These
        # are dictionaries of the form {<result_label>: <operation props>}
        self._extract_keys = {}
        self._models = {}

        # Unpack...
        self._unpack_model_recursively(self)

        # Also unpack any related models
        related_models = list(self.related_models)
        for model in related_models:
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
        """Set an attribute on the model.

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

    def set_attribute_overload(self, attr_str, overload_str):
        """Redirect a model attribute to read a different attribute.

        This function is useful for redirecting get_param to read a different
        attribute from the one the model would expect. This is particularly,
        useful when redirecting a grid axis to read from a different attribute
        on the emitter.

        For example, if we have a grid that contains "ages" and "metallicities"
        axes but we want to instead use "special_ages" and
        "special_metallicities" attributes on the emitter, we can use this
        function which will cause get_param to first attempt to extract
        "ages" and "metallicities" from the model, see the overload_str
        string, recurse and attempt to extract "special_ages" and
        "special_metallicities" from the model, find they are not there, and
        then look for "special_ages" and "special_metallicities" on the
        emitter and return them.

        This same behaviour can be achieved by passing kwargs at init but
        this function enables retroactive redirection, especially useful when
        using premade models.

        Args:
            attr_str (str): The attribute to redirect.
            overload_str (str): The attribute to redirect to.
        """
        # Ensure we don't already have an override. Previously this checked
        # hasattr(self, attr_str) which is incorrect because that tests for a
        # property/attribute name on the object rather than whether an overload
        # has already been stored in fixed_parameters. Use membership in the
        # fixed_parameters dict so attempting to set the same overload twice
        # raises an error and reports the existing value.
        if attr_str in self.fixed_parameters:
            raise exceptions.InconsistentArguments(
                f"Cannot overload attribute {attr_str} on model {self.label}. "
                f"{attr_str} is already set on the model "
                f"{self.fixed_parameters[attr_str]}. "
            )

        # Store the overload in the fixed parameters dictionary so it will be
        # picked up by get_param when trying to access attr_str
        self.fixed_parameters[attr_str] = overload_str

    def set_fixed_parameter(self, param_name, value):
        """Set a fixed parameter.

        This method will set a fixed parameter on the model. This parameter
        will take precedence over any parameter of the same name on an
        emitter.

        When get_param is called to access a parameter, it will first check
        the fixed_parameters dictionary on the model, prior to looking for
        the parameter on the emitter.

        Args:
            param_name (str): The name of the parameter to fix.
            value (Any): The value to fix the parameter to.
        """
        # Ensure we don't already have an override for this parameter
        if param_name in self.fixed_parameters:
            raise exceptions.InconsistentArguments(
                f"Cannot set fixed parameter {param_name} on model "
                f"{self.label} to {value}. This parameter is already fixed "
                f"to {self.fixed_parameters[param_name]}."
            )

        self.fixed_parameters[param_name] = value

    def clear_fixed_parameter(self, param_name):
        """Clear a fixed parameter.

        This method will clear a fixed parameter on the model, allowing
        get_param to access the parameter on the emitter again.

        Args:
            param_name (str): The name of the parameter to clear.
        """
        if param_name in self.fixed_parameters:
            del self.fixed_parameters[param_name]
        else:
            raise exceptions.InconsistentArguments(
                f"Cannot clear fixed parameter {param_name} on model "
                f"{self.label}. This parameter is not currently fixed."
            )

    def clear_fixed_parameters(self):
        """Clear all fixed parameters on the model."""
        self.fixed_parameters = {}

    @property
    def grid(self):
        """Get the Grid object used for extraction."""
        return getattr(self, "_grid", None)

    def set_grid(self, grid, set_all=False):
        """Set the grid to extract from.

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
        """Set the spectra to extract from the grid.

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
    def transformer(self):
        """Get the transformer to apply."""
        return getattr(self, "_transformer", None)

    def set_transformer(self, transformer):
        """Set the transformer to apply."""
        self._set_attr("transformer", transformer)

    @property
    def dust_curve(self):
        """Get the dust curve to apply."""
        return getattr(self, "_transformer", None)

    def set_dust_curve(self, dust_curve):
        """Set the dust curve to apply."""
        self.set_transformer(dust_curve)

    @property
    def igm(self):
        """Get the IGM model to apply."""
        return getattr(self, "_transformer", None)

    def set_igm(self, igm):
        """Set the IGM model to apply."""
        self.set_transformer(igm)

    @property
    def apply_to(self):
        """Get the spectra to apply the dust curve to."""
        return getattr(self, "_apply_to", None)

    def set_apply_to(self, apply_to):
        """Set the spectra to apply the dust curve to.

        Args:
            apply_to (EmissionModel):
                The model to apply the dust curve to.
        """
        # Set the spectra to apply the dust curve to
        if self._is_transforming:
            self._set_attr("apply_to", apply_to)
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set a spectra to apply a dust curve to on a model "
                "that is not dust attenuating."
            )

        # Unpack the model now we're done
        self.unpack_model()

    def set_dust_props(
        self,
        dust_curve=None,
        apply_to=None,
        set_all=False,
    ):
        """Set the dust attenuation properties on this model.

        Args:
            dust_curve (emission_models.attenuation.*):
                A dust curve instance to apply.
            apply_to (EmissionModel):
                The model to apply the dust curve to.
            set_all (bool):
                Whether to set the properties on all models.
        """
        # Set the properties
        if not set_all:
            if self._is_transforming:
                if dust_curve is not None:
                    self._set_attr("transformer", dust_curve)
                if apply_to is not None:
                    self._set_attr("apply_to", apply_to)
            else:
                raise exceptions.InconsistentArguments(
                    "Cannot set dust attenuation properties on a model that "
                    "is not dust attenuating."
                )
        else:
            for model in self._models.values():
                if self._is_transforming:
                    model.set_dust_props(
                        dust_curve=dust_curve,
                        apply_to=apply_to,
                    )

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def generator(self):
        """Get the emission generation model."""
        return getattr(self, "_generator", None)

    def set_generator(self, generator):
        """Set the dust emission model on this model.

        Args:
            generator (EmissionModel):
                The emission generation model to set.
            label (str):
                The label of the model to set the dust emission model on. If
                None, sets the dust emission model on this model.
        """
        # Ensure model is a emission generation model and change the model
        if self._is_generating:
            self._set_attr("generator", generator)
        else:
            raise exceptions.InconsistentArguments(
                "Cannot set a generator on a model that is not generating."
            )

        # Unpack the model now we're done
        self.unpack_model()

    @property
    def emitter(self):
        """Get the emitter this emission model acts on."""
        return getattr(self, "_emitter", None)

    def set_emitter(self, emitter, set_all=False):
        """Set the emitter this emission model acts on.

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
        """Set the per particle flag.

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
    def vel_shift(self):
        """Get the velocity shift flag."""
        if hasattr(self, "_use_vel_shift"):
            return self._use_vel_shift
        else:
            return False

    def set_vel_shift(self, vel_shift, set_all=False):
        """Set whether we should apply velocity shifts to the spectra.

        Only applicable to particle emitters.

        Args:
            vel_shift (bool):
                Whether to set the velocity shift flag.
            set_all (bool):
                Whether to set the emitter on all models.
        """
        if not set_all:
            self._use_vel_shift = vel_shift
        else:
            for model in self._models.values():
                model.set_vel_shift(vel_shift)

    @property
    def combine(self):
        """Get the models to combine."""
        return getattr(self, "_combine", tuple())

    def set_combine(self, combine):
        """Set the models to combine on this model.

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

        # Set the models to combine ensuring the model we are setting on is
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
    def scale_by(self):
        """Get the attribute to scale the spectra by."""
        return self._scale_by

    def set_scale_by(self, scale_by, set_all=False):
        """Set the attribute to scale the spectra by.

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
        """Set the post processing functions on this model.

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
        """Set the flag for whether to save the emission.

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

    def _expand_label_patterns(self, *patterns):
        """Expand glob patterns into the labels they match.

        Args:
            patterns (str):
                Labels, or glob patterns matching labels (e.g. "*_fesc_0.10").
                A plain label with no wildcard matches only itself.

        Returns:
            list:
                The matching labels, sorted so the result is reproducible.

        Raises:
            InconsistentArguments
                If a pattern matches no models.
        """
        labels = set()
        for pattern in patterns:
            matches = fnmatch.filter(self._models, pattern)

            # A pattern matching nothing is almost always a mistake, and
            # silently ignoring it would leave the caller thinking it worked
            if len(matches) == 0:
                raise exceptions.InconsistentArguments(
                    f"No models match '{pattern}'. Model has: "
                    f"{sorted(self._models)}"
                )

            labels.update(matches)

        return sorted(labels)

    def select(self, *patterns):
        """Return the models whose labels match the given glob patterns.

        This is the bulk counterpart to indexing a model by label, useful for
        applying a change to a family of models at once. It is particularly
        handy after expanding parameter variations, where the variants of a
        model all share a label prefix or suffix::

            for model in expanded.select("*_fesc_0.10"):
                model.set_save(False)

        Args:
            patterns (str):
                Labels, or glob patterns matching labels. A plain label with no
                wildcard matches only itself.

        Returns:
            list:
                The matching models, ordered by label. Empty if nothing
                matches, since this is a query rather than a change.
        """
        matched = set()
        for pattern in patterns:
            matched.update(fnmatch.filter(self._models, pattern))

        return [self._models[label] for label in sorted(matched)]

    def save_emission(self, *args):
        """Set the save flag to True for the given emission.

        Args:
            args (str):
                The emission to save. Glob patterns are accepted, so
                "*_fesc_0.10" will save that variant of every model.
        """
        # Resolve any patterns before changing anything, so an unmatched
        # pattern doesn't leave every model with save switched off
        labels = self._expand_label_patterns(*args)

        # First set all models to not save
        self.set_save(False, set_all=True)

        # Now set the given spectra to save
        for label in labels:
            self[label].set_save(True)

    def save_spectra(self, *args):
        """Set the save flag to True for the given spectra.

        This is just a friendly alias for save_emission.

        Args:
            args (str):
                The spectra to save. Glob patterns are accepted.
        """
        self.save_emission(*args)

    def save_lines(self, *args):
        """Set the save flag to True for the given lines.

        This is just a friendly alias for save_emission.

        Args:
            args (str):
                The lines to save. Glob patterns are accepted.
        """
        self.save_emission(*args)

    @property
    def saved_labels(self):
        """Return a list of model labels that are set to be saved."""
        return [label for label, model in self._models.items() if model.save]

    @property
    def lam_mask(self):
        """Get the wavelength mask."""
        return getattr(self, "_lam_mask", None)

    def set_lam_mask(self, lam_mask, set_all=False):
        """Set the wavelength mask.

        Args:
            lam_mask (array_like):
                The wavelength mask to apply.
            set_all (bool):
                Whether to set the wavelength mask on all models.
        """
        # Set the wavelength mask
        if not set_all:
            self._lam_mask = lam_mask
        else:
            for model in self._models.values():
                model.set_lam_mask(lam_mask)

        # Unpack the model now we're done
        self.unpack_model()

    def add_mask(self, attr, op, thresh, set_all=False):
        """Add a mask.

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
        # Ensure we have units, or a string alias resolved at generation time
        if not isinstance(thresh, (unyt_quantity, str)):
            raise exceptions.MissingUnits(
                "Mask threshold must be given with units (or as a string "
                + "attribute name to be resolved from the emitter)."
            )

        # Ensure operation is valid
        if op not in ["<", ">", "<=", ">=", "=="]:
            raise exceptions.InconsistentArguments(
                "Invalid mask operation. Must be one of: <, >, <=, >="
            )

        # Add the mask
        if not set_all:
            self.masks.append({"attr": attr, "thresh": thresh, "op": op})
        else:
            for model in self._models.values():
                model.masks.append({"attr": attr, "thresh": thresh, "op": op})

        # Unpack now that we're done
        self.unpack_model()

    def clear_masks(self, set_all=False):
        """Clear all masks.

        Args:
            set_all (bool):
                Whether to clear the masks on all models.
        """
        # Clear the masks
        if not set_all:
            self.masks = []
        else:
            for model in self._models.values():
                model.masks = []

        # Unpack now that we're done
        self.unpack_model()

    @property
    def _is_masked(self):
        """Check if the model is masked."""
        return len(self.masks) > 0

    @property
    def variant_params(self):
        """Return the parameter values which distinguish this model variant.

        A model produced by expanding a parameter variation records the values
        which were varied to produce it, so which variant a model (and thus its
        emission) belongs to can be recovered without parsing its label.

        Returns:
            dict:
                A dictionary of the form {<param_name>: <value>}. Empty for any
                model which isn't the product of an expansion.
        """
        return getattr(self, "_variant_params", {})

    @property
    def variant_base(self):
        """Return the label this model's variant family was expanded from.

        Expanding a parameter variation appends a suffix to the label of every
        affected model. This is the label before any of those suffixes were
        added, so the variants of a model can be grouped back together.

        Returns:
            str or None:
                The original label, or None for any model which isn't the
                product of an expansion.
        """
        return getattr(self, "_variant_base", None)

    def _check_expanded(self):
        """Check no parameter variation is still waiting to be expanded.

        A variation is a declaration rather than a value, so a model still
        holding one cannot generate an emission. Checking the whole tree up
        front says so plainly, rather than failing somewhere inside the
        calculation once part of the work has already been done.

        Raises:
            InconsistentArguments
                If any model in the tree still holds a declaration.
        """
        # Imported here to avoid a circular import at module load time
        from synthesizer.emission_models.parameters import find_variations

        waiting = {}
        for label, model in self._models.items():
            variations = find_variations(model)
            if len(variations) > 0:
                waiting[label] = sorted(variations)

        if len(waiting) > 0:
            described = ", ".join(
                f"{label} ({', '.join(params)})"
                for label, params in sorted(waiting.items())
            )
            raise exceptions.InconsistentArguments(
                "Cannot generate an emission from a model which still has "
                f"parameter variations declared on it: {described}. Call "
                "expand_models() to turn those declarations into models "
                "first."
            )

    def expand_models(self):
        """Expand any parameter variations in this tree into new models.

        Passing a ParameterList or ParameterDistribution as a model parameter
        declares that the model should be varied over a set of values. This
        method turns those declarations into models: the varied model, and
        everything which depends on it, is duplicated once per value, with each
        copy labelled using the variation's label_modifier. Models which the
        varied model depends on are shared between the copies rather than
        duplicated, so the parts of the tree unaffected by the variation are
        only ever computed once.

        This model is not modified. The expansion is performed on a copy.

        The returned root has the other root variants attached as related
        models, so a single get_spectra call generates every variant. Each
        variant's emission is stored on the emitter under its own label.

        Returns:
            EmissionModel:
                The root of the expanded tree. If nothing in the tree declared
                a variation this is just a copy of this model.
        """
        # Imported here to avoid a circular import at module load time
        from synthesizer.emission_models.expansion import expand_models

        return expand_models(self)

    @property
    def _generator_model_attrs(self):
        """Return the generator attributes which point at other models.

        A generator can depend on other models, either for energy balance
        (an intrinsic and an attenuated model) or for scaling (a single scaler
        model). These pointers live on the generator rather than the model, so
        they need handling separately whenever a model is copied or replaced.

        Returns:
            tuple:
                The names of the attributes on this model's generator which
                currently hold an EmissionModel.
        """
        # Only generation models can have generator dependencies
        if not self._is_generating:
            return ()

        return tuple(
            attr
            for attr in ("_intrinsic", "_attenuated", "_scaler")
            if isinstance(getattr(self.generator, attr, None), EmissionModel)
        )

    def _replace_generator_dependency(self, old_model, new_model):
        """Point a generator dependency at a different model.

        Args:
            old_model (EmissionModel):
                The model currently referenced by the generator.
            new_model (EmissionModel):
                The model to reference instead.
        """
        # Only generation models can have generator dependencies
        if not self._is_generating:
            return

        generator = self.generator

        # Swap the scaler, going through the setter so the generator's
        # energy-balance/scaled flags stay consistent
        if getattr(generator, "_scaler", None) is old_model:
            if hasattr(generator, "set_scaler"):
                generator.set_scaler(new_model)
            else:
                generator._scaler = new_model

        # Swap either half of an energy balance pair
        if getattr(generator, "_intrinsic", None) is old_model:
            generator.set_energy_balance(new_model, generator._attenuated)
        if getattr(generator, "_attenuated", None) is old_model:
            generator.set_energy_balance(generator._intrinsic, new_model)

    def _copy_node(self, label=None):
        """Return a copy of this single model, detached from any tree.

        The copy shares immutable and expensive state with the original (the
        grid, the wavelength array, the transformer) but gets its own copies of
        everything mutable, so that modifying the copy cannot reach back and
        change the original. The copy has no children or parents; the caller is
        responsible for wiring it into a tree.

        Note that a generator holding references to other models is also
        copied. Without this a copy would share its generator with the
        original, and rewiring the copy's dependencies would silently rewire
        the original's too.

        Args:
            label (str):
                An optional label for the copy. Defaults to the original label,
                which the caller must then change before the copy shares a tree
                with the original.

        Returns:
            EmissionModel:
                The detached copy.
        """
        new_model = copy.copy(self)

        # Apply the new label if we've been given one
        if label is not None:
            new_model.label = label

        # Detach from any tree, these are rebuilt by unpack_model
        new_model._children = set()
        new_model._parents = set()
        new_model._models = {}
        new_model._extract_keys = {}
        new_model._energy_balance_models = None
        new_model._scaler_model = None

        # Take independent copies of the mutable state. Without this the copy
        # and the original would share these containers.
        new_model.masks = [dict(mask) for mask in self.masks]
        new_model.fixed_parameters = dict(self.fixed_parameters)
        new_model.related_models = set(self.related_models)
        new_model._scale_by = copy.copy(self._scale_by)
        new_model._post_processing = copy.copy(self._post_processing)
        if self._is_combining:
            new_model._combine = list(self._combine)

        # Copy a generator which points at other models so that rewiring this
        # copy's dependencies leaves the original's alone
        if len(self._generator_model_attrs) > 0:
            new_model._generator = copy.copy(self.generator)

        return new_model

    def replace_model(self, replace_label, *replacements, new_label=None):
        """Remove a child model from this model.

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
        replace_model = self._models.pop(replace_label)

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

        # Remove the old model from everywhere
        for model in self._models.values():
            if replace_model in model._children:
                model._children.remove(replace_model)
            if replace_model in model._parents:
                model._parents.remove(replace_model)
            if model._is_combining and replace_model in model.combine:
                model._combine = tuple(
                    m for m in model.combine if m.label != replace_model.label
                )
            if model._is_transforming and model.apply_to == replace_model:
                model._apply_to = new_model
            model._replace_generator_dependency(replace_model, new_model)
            if replace_label in model._models:
                model._models.pop(replace_label)

        # Unpack now we're done
        self.unpack_model()

    def relabel(self, old_label, new_label):
        """Change the label associated to an existing spectra.

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

        # Unpack now we're done
        self.unpack_model()

    def fix_parameters(self, **kwargs):
        """Fix parameters of the model.

        Args:
            **kwargs:
                The parameters to fix.
        """
        if "apply_dust_to" in kwargs:
            raise exceptions.InconsistentArguments(
                "The apply_dust_to argument has been removed. "
                "Please use apply_to instead."
            )
        self.fixed_parameters.update(kwargs)

    def to_hdf5(self, group):
        """Save the model to an HDF5 group.

        Args:
            group (h5py.Group):
                The group to save the model to.
        """
        # First off call the operation to save operation specific attributes
        # to the group
        if self._is_extracting:
            self.extract_to_hdf5(group)
        elif self._is_combining:
            self.combine_to_hdf5(group)
        elif self._is_transforming:
            self.transformation_to_hdf5(group)
        elif self._is_generating:
            self.generate_to_hdf5(group)

        # Save the model attributes
        group.attrs["label"] = self.label
        group.attrs["emitter"] = self.emitter
        group.attrs["per_particle"] = self.per_particle
        group.attrs["save"] = self.save
        group.attrs["scale_by"] = self.scale_by
        group.attrs["post_processing"] = (
            [func.__name__ for func in self.post_processing]
            if len(self._post_processing) > 0
            else "None"
        )

        # Save the masks
        if len(self.masks) > 0:
            masks = group.create_group("Masks")
            for ind, mask in enumerate(self.masks):
                mask_group = masks.create_group(f"mask_{ind}")
                mask_group.attrs["attr"] = mask["attr"]
                mask_group.attrs["op"] = mask["op"]
                mask_group.attrs["thresh"] = mask["thresh"]

        # Save the fixed parameters
        if len(self.fixed_parameters) > 0:
            fixed_parameters = group.create_group("FixedParameters")
            for key, value in self.fixed_parameters.items():
                if key == "apply_dust_to":
                    raise exceptions.InconsistentArguments(
                        "The apply_dust_to argument has been removed. "
                        "Please use apply_to instead."
                    )
                if isinstance(value, VARIATION_TYPES):
                    raise exceptions.InconsistentArguments(
                        f"Cannot write an unexpanded {type(value).__name__} "
                        f"for '{key}' on model '{self.label}' to HDF5. Call "
                        "expand_models() on the root model first."
                    )
                fixed_parameters.attrs[key] = value

        # Save the children
        if len(self._children) > 0:
            group.create_dataset(
                "Children",
                data=[child.label.encode("utf-8") for child in self._children],
            )

    def plot_emission_graph(
        self,
        root=None,
        show=True,
        fontsize=10,
        figsize=None,
        layout="layered",
        collapse_variants=False,
        min_fontsize=6.0,
    ):
        """Plot the network of models defining the emission.

        The whole network is drawn, including related models, which are roots
        in their own right. Grid extractions sit along the bottom row with each
        model above everything it depends on, and arrows point in the direction
        the emission flows.

        Args:
            root (str):
                If not None only this model and the models it depends on are
                drawn.
            show (bool):
                Whether to show the plot.
            fontsize (int):
                The fontsize to use for the labels.
            figsize (tuple):
                The size of the figure to plot (width, height). By default the
                figure is sized to fit the network.
            layout (str):
                Either "layered" (the default), which needs only networkx, or
                "dot", which lays the network out with graphviz and needs pydot
                and the graphviz binary.
            collapse_variants (bool):
                Whether to collapse each family of parameter variants into a
                single node badged with the number of variants it stands for.
            min_fontsize (float):
                The size below which labels stop being readable. A network
                which cannot fit the figure with labels this big grows the
                figure rather than shrinking them further, so a large network
                stays as readable as a small one. Pass 0 to let the labels
                shrink as far as they need to.

        Returns:
            fig (matplotlib.figure.Figure):
                The figure containing the plot.
            ax (matplotlib.axes.Axes):
                The axis containing the plot.
        """
        # Imported here to avoid a circular import at module load time
        from synthesizer.emission_models.visualise_network import (
            plot_emission_graph,
        )

        return plot_emission_graph(
            self,
            root=root,
            show=show,
            fontsize=fontsize,
            figsize=figsize,
            layout=layout,
            collapse_variants=collapse_variants,
            min_fontsize=min_fontsize,
        )

    @deprecated(
        "is deprecated in favour of plot_emission_graph (an emission model is "
        "a network rather than a tree) and will be removed in a future version"
    )
    def plot_emission_tree(self, *args, **kwargs):
        """Plot the network of models defining the emission.

        Deprecated alias for plot_emission_graph.

        Args:
            *args:
                Positional arguments passed to plot_emission_graph.
            **kwargs:
                Keyword arguments passed to plot_emission_graph.

        Returns:
            fig (matplotlib.figure.Figure):
                The figure containing the plot.
            ax (matplotlib.axes.Axes):
                The axis containing the plot.
        """
        return self.plot_emission_graph(*args, **kwargs)

    def _apply_overrides(
        self,
        emission_model,
        dust_curves,
        tau_v,
        fesc,
        covering_fraction,
        mask,
        vel_shift,
    ):
        """Apply overrides to an emission model copy.

        This function is used in _get_spectra to apply any emission model
        property overrides passed to that method before generating the
        spectra.

        _get_spectra will make a copy of the emission model and then pass it
        to this function to apply any overrides before generating the spectra.

        Args:
            emission_model (EmissionModel):
                The emission model copy to apply the overrides to.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
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
                An override to the emission model escape fraction. Either:
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
                An override to the emission model covering fraction. Either:
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
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            vel_shift (dict/bool):
                Override the models flag for using peculiar velocities to apply
                doppler shift to the generated spectra. Only applicable for
                particle spectra. Can be a boolean to apply to all models or a
                dictionary of the form:
                    {<label>: bool}
                to apply to specific models.
        """
        # If we have dust curves to apply, apply them
        if dust_curves is not None:
            if isinstance(dust_curves, dict):
                for label, dust_curve in dust_curves.items():
                    emission_model._models[label].fixed_parameters[
                        "dust_curve"
                    ] = dust_curve
            else:
                for model in emission_model._models.values():
                    model.fixed_parameters["dust_curve"] = dust_curves

        # If we have optical depths to apply, apply them
        if tau_v is not None:
            if isinstance(tau_v, dict):
                for label, value in tau_v.items():
                    emission_model._models[label].fixed_parameters["tau_v"] = (
                        value
                    )
            else:
                for model in emission_model._models.values():
                    model.fixed_parameters["tau_v"] = tau_v

        # If we have escape fractions to apply, apply them
        if fesc is not None:
            if isinstance(fesc, dict):
                for label, value in fesc.items():
                    emission_model._models[label].fixed_parameters["fesc"] = (
                        value
                    )
            else:
                for model in emission_model._models.values():
                    model.fixed_parameters["fesc"] = fesc

        # If we have covering fractions to apply, apply them
        if covering_fraction is not None:
            if isinstance(covering_fraction, dict):
                for label, value in covering_fraction.items():
                    emission_model._models[label].fixed_parameters["fesc"] = (
                        1 - value
                    )
            else:
                for model in emission_model._models.values():
                    model.fixed_parameters["fesc"] = 1 - covering_fraction

        # If we have masks to apply, apply them
        if mask is not None:
            for label, mask in mask.items():
                emission_model[label].add_mask(**mask)

        # If we have velocity shifts to apply, apply them
        if vel_shift is not None:
            if isinstance(vel_shift, dict):
                for label, value in vel_shift.items():
                    emission_model._models[label].set_vel_shift(value)
            else:
                for model in emission_model._models.values():
                    if model._is_extracting:
                        model.set_vel_shift(vel_shift)

    def _get_existing_emissions(
        self,
        emitters,
        emissions,
        particle_emissions,
        emission_type="spectra",
        models=None,
    ):
        """Unpack any existing emissions from the emitters.

        This method will populate the emissions and particle_emissions
        dictionaries (either spectra or lines) with any existing emissions
        that we are using as signalled on the emission model. Any of the
        arguments which can take a model can also take a string to denote
        reuse of existing emissions from the emitter.

        If the emission is missing an error is raised.

        Args:
            emitters (dict):
                The emitters we are generating emissions for.
            emissions (dict):
                The dictionary of emissions to populate.
            particle_emissions (dict):
                The dictionary of particle emissions to populate.
            emission_type (str):
                The type of emission to get. Either "spectra" or "lines".
            models (iterable):
                Optional iterable of models to inspect. If omitted, all models
                discovered on the emission tree are considered.

        Returns:
            dict, dict
                The updated emissions and particle_emissions dictionaries.
        """
        # Ensure we have a valid emission type
        if emission_type not in ["spectra", "lines"]:
            raise exceptions.InconsistentArguments(
                f"Invalid emission_type {emission_type}. Must be "
                "'spectra' or 'lines'."
            )

        # Restrict reuse checks to the supplied models when the caller has
        # already pruned the execution graph down to its active subset.
        if models is None:
            models = self._models.values()

        for this_model in models:
            # Skip extractions, these can't reuse existing lines by their
            # nature
            if this_model._is_extracting:
                continue

            # Skip models for a different emitters
            if this_model.emitter not in emitters:
                continue

            # Get the emitter
            emitter = emitters[this_model.emitter]

            # Get the emitter emissions
            emitter_emissions = getattr(emitter, emission_type)
            emitter_particle_emissions = getattr(
                emitter, f"particle_{emission_type}", None
            )

            # Check apply_to for strings (only applicable to Transformations)
            if (
                isinstance(this_model.apply_to, str)
                and this_model.apply_to in emitter_emissions
            ):
                if this_model.per_particle:
                    particle_emissions[this_model.apply_to] = (
                        emitter_particle_emissions[this_model.apply_to]
                    )
                emissions[this_model.apply_to] = emitter_emissions[
                    this_model.apply_to
                ]
            elif isinstance(this_model.apply_to, str):
                raise exceptions.InconsistentArguments(
                    f"Can't reuse existing emission for {this_model.apply_to} "
                    "since it could not be found in "
                    f"{emitter.__class__.__name__}.{emission_type}. "
                    f"Generate {this_model.apply_to} first or point "
                    "apply_to to a model not a string."
                )
            else:
                # Not reusing existing emission
                pass

            # Check combine for strings (only applicable to Combinations)
            for comp in this_model.combine:
                if isinstance(comp, str) and comp in emitter_emissions:
                    if this_model.per_particle:
                        particle_emissions[comp] = emitter_particle_emissions[
                            comp
                        ]
                    emissions[comp] = emitter_emissions[comp]
                elif isinstance(comp, str):
                    raise exceptions.InconsistentArguments(
                        f"Can't reuse existing emission for {comp} since it "
                        "could not be found in "
                        f"{emitter.__class__.__name__}.{emission_type}. "
                        f"Generate {comp} first or point combine to a model "
                        "not a string."
                    )
                else:
                    # Not reusing existing emission
                    pass

            # Check generator dependencies for strings
            intrinsic_model = (
                this_model.generator._intrinsic
                if this_model.generator is not None
                and hasattr(this_model.generator, "_intrinsic")
                else None
            )
            attenuated_model = (
                this_model.generator._attenuated
                if this_model.generator is not None
                and hasattr(this_model.generator, "_attenuated")
                else None
            )
            scaler_model = (
                this_model.generator._scaler
                if this_model.generator is not None
                and hasattr(this_model.generator, "_scaler")
                else None
            )
            if (
                isinstance(intrinsic_model, str)
                and intrinsic_model in emitter_emissions
            ):
                if this_model.per_particle:
                    particle_emissions[intrinsic_model] = (
                        emitter_particle_emissions[intrinsic_model]
                    )
                emissions[intrinsic_model] = emitter_emissions[intrinsic_model]
            elif isinstance(intrinsic_model, str):
                raise exceptions.InconsistentArguments(
                    f"Can't reuse existing emission for {intrinsic_model} "
                    "since it could not be found in "
                    f"{emitter.__class__.__name__}.{emission_type}. "
                    f"Generate {intrinsic_model} first or point the "
                    "Generator to a model not a string."
                )
            else:
                # Not reusing existing emission
                pass
            if (
                isinstance(attenuated_model, str)
                and attenuated_model in emitter_emissions
            ):
                if this_model.per_particle:
                    particle_emissions[attenuated_model] = (
                        emitter_particle_emissions[attenuated_model]
                    )
                emissions[attenuated_model] = emitter_emissions[
                    attenuated_model
                ]
            elif isinstance(attenuated_model, str):
                raise exceptions.InconsistentArguments(
                    "Can't reuse existing emission for "
                    f"{attenuated_model} since it could not be found in "
                    f"{emitter.__class__.__name__}.{emission_type}. "
                    f"Generate {attenuated_model} first or point the "
                    "Generator to a model not a string."
                )
            else:
                # Not reusing existing emission
                pass
            if (
                isinstance(scaler_model, str)
                and scaler_model in emitter_emissions
            ):
                if this_model.per_particle:
                    particle_emissions[scaler_model] = (
                        emitter_particle_emissions[scaler_model]
                    )
                emissions[scaler_model] = emitter_emissions[scaler_model]
            elif isinstance(scaler_model, str):
                raise exceptions.InconsistentArguments(
                    "Can't reuse existing emission for "
                    f"{scaler_model} since it could not be found in "
                    f"{emitter.__class__.__name__}.{emission_type}. "
                    f"Generate {scaler_model} first or point the "
                    "Generator to a model not a string."
                )
            else:
                # Not reusing existing emission
                pass

        return emissions, particle_emissions

    @timed("EmissionModel._get_spectra")
    def _get_spectra(
        self,
        emitters,
        dust_curves=None,
        tau_v=None,
        fesc=None,
        covering_fraction=None,
        mask=None,
        vel_shift=None,
        verbose=True,
        spectra=None,
        particle_spectra=None,
        nthreads=1,
        grid_assignment_method="cic",
        **fixed_parameters,
    ):
        """Generate stellar spectra as described by the emission model.

        NOTE: post processing methods defined on the model will be called
        once all spectra are made (these models are preceded by post_ and
        take the dictionary of lines/spectra as an argument).

        Args:
            emitters (Stars/BlackHoles):
                The emitters to generate the spectra for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
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
                An override to the emission model escape fraction. Either:
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
                An override to the emission model covering fraction. Either:
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
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            verbose (bool):
                Are we talking?
            spectra (dict):
                A dictionary of spectra to add to before executing the model.
            particle_spectra (dict):
                A dictionary of particle spectra to add to before executing the
                model.
            vel_shift (bool):
                override the models flag for using peculiar velocities to apply
                doppler shift to the generated spectra. Only applicable for
                particle spectra.
            nthreads (int):
                The number of threads to use when generating the spectra.
            grid_assignment_method (str):
                The method to use when assigning particles to the grid. Options
                are "cic" (cloud in cell) or "ngp" (nearest grid point).
            **fixed_parameters (dict):
                A dictionary of fixed parameters to apply to the model. Each
                of these will be applied to the model before generating the
                spectra.

        Returns:
            dict
                A dictionary of spectra which can be attached to the
                appropriate spectra attribute of the component
                (spectra/particle_spectra)
        """
        # A variation is a declaration, not a value, so nothing can be
        # generated while one is still waiting to be expanded
        self._check_expanded()

        # We don't want to modify the original emission model with any
        # modifications made here so we'll make a copy of it (this is a
        # shallow copy so very cheap and doesn't copy any pointed to objects
        # only their reference)
        emission_model = copy.copy(self)

        # Apply any overrides we have
        with timer("EmissionModel._get_spectra.apply_overrides"):
            self._apply_overrides(
                emission_model,
                dust_curves=dust_curves,
                tau_v=tau_v,
                fesc=fesc,
                covering_fraction=covering_fraction,
                mask=mask,
                vel_shift=vel_shift,
            )

        # Work with the overridden root instance stored in the model tree so
        # root-level overrides are reflected in any queue and reuse logic.
        root_model = emission_model._models[self.label]

        # Make a spectra dictionary if we haven't got one yet
        if spectra is None:
            spectra = {}
        if particle_spectra is None:
            particle_spectra = {}

        # We need to make sure the root is being saved, otherwise this is a bit
        # nonsensical.
        if not root_model.save:
            raise exceptions.InconsistentArguments(
                f"{root_model.label} is not being saved. There's no point in "
                "generating at the root if they are not saved. Maybe you "
                "want to use a child model you are saving instead?"
            )

        # Build the execution queue for the active model tree before doing any
        # existing-emission reuse checks so inactive branches are skipped.
        with timer("EmissionModel._get_spectra.build_model_queue"):
            queue = ModelQueue(root_model)

        # Before we do anything else, check that we have the emitters needed by
        # the active models in the queued execution graph.
        for model in queue.models.values():
            if emitters.get(model.emitter, None) is None:
                raise exceptions.InconsistentArguments(
                    f"Missing emitter '{model.emitter}' required by active "
                    f"EmissionModel '{model.label}'."
                )

        # Get any existing spectra we are reusing
        with timer("EmissionModel._get_spectra.get_existing_emissions"):
            spectra, particle_spectra = root_model._get_existing_emissions(
                emitters,
                spectra,
                particle_spectra,
                emission_type="spectra",
                models=queue.models.values(),
            )

        # Execute the full model closure by processing each ready model once.
        while len(queue) > 0:
            this_model = queue.pop()
            label = this_model.label

            # Reused or externally supplied emissions still need to unlock the
            # graph, but they do not need to be regenerated.
            if label in spectra:
                queue.done(this_model, spectra, particle_spectra)
                continue

            # Active queued models must always have a matching emitter.
            if this_model.emitter not in emitters:
                raise exceptions.InconsistentArguments(
                    f"Active EmissionModel '{this_model.label}' requires "
                    f"missing emitter '{this_model.emitter}'."
                )

            # Get the emitter for this model now that it is ready to execute.
            emitter = emitters[this_model.emitter]

            # Build the combined mask once for operations that use it.
            this_mask = None
            for mask_dict in this_model.masks:
                this_mask = emitter.get_mask(
                    **mask_dict,
                    mask=this_mask,
                    attr_override_obj=this_model,
                )

            # Dispatch to the appropriate operation for this model.
            if this_model._is_extracting:
                try:
                    spectra, particle_spectra = self._extract_spectra(
                        this_model,
                        emitters,
                        spectra,
                        particle_spectra,
                        verbose=verbose,
                        nthreads=nthreads,
                        grid_assignment_method=grid_assignment_method,
                    )
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {label}]"
                        ).with_traceback(e.__traceback__)
            elif this_model._is_combining:
                try:
                    spectra, particle_spectra = self._combine_spectra(
                        emission_model,
                        spectra,
                        particle_spectra,
                        this_model,
                        emitter,
                        nthreads,
                    )
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)
            elif this_model._is_transforming:
                try:
                    spectra, particle_spectra = this_model._transform_emission(
                        this_model,
                        spectra,
                        particle_spectra,
                        emitter,
                        this_mask,
                        self.lam,
                        nthreads,
                    )
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)
            elif this_model._is_generating:
                try:
                    spectra, particle_spectra = self._generate_spectra(
                        this_model,
                        emission_model,
                        spectra,
                        particle_spectra,
                        self.lam,
                        emitter,
                        nthreads,
                    )
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)

            # Apply any requested scaling after the model has been generated.
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
                        particle_spectra[label].scale(
                            scaler_arr,
                            inplace=True,
                            nthreads=nthreads,
                        )
                    spectra[label].scale(
                        scaler_arr,
                        inplace=True,
                        nthreads=nthreads,
                    )
                elif scaler in spectra:
                    # Compute the scaling against another generated spectrum.
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
                        particle_spectra[label].scale(
                            scaling,
                            inplace=True,
                            nthreads=nthreads,
                        )
                    spectra[label].scale(
                        scaling,
                        inplace=True,
                        nthreads=nthreads,
                    )
                else:
                    raise exceptions.InconsistentArguments(
                        f"Can't scale spectra by {scaler}."
                    )

            # Unlock downstream models and delete expired unsaved emissions.
            queue.done(this_model, spectra, particle_spectra)

        # Ensure the dependency graph was fully traversed before returning.
        queue.assert_finished()

        # Apply any post processing functions to the surviving emissions.
        for func in self._post_processing:
            spectra = func(spectra, emitters, self)
            if len(particle_spectra) > 0:
                particle_spectra = func(particle_spectra, emitters, self)

        return spectra, particle_spectra

    @timed("EmissionModel._get_lines")
    def _get_lines(
        self,
        line_ids,
        emitters,
        dust_curves=None,
        tau_v=None,
        fesc=0.0,
        covering_fraction=None,
        mask=None,
        verbose=True,
        lines=None,
        particle_lines=None,
        nthreads=1,
        grid_assignment_method="cic",
        **kwargs,
    ):
        """Generate stellar lines as described by the emission model.

        NOTE: post processing methods defined on the model will be called
        once all spectra are made (these models are preceded by post_ and
        take the dictionary of lines/spectra as an argument).

        Args:
            line_ids (list):
                The line ids to extract.
            emitters (Stars/BlackHoles):
                The emitters to generate the lines for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
            dust_curves (dict):
                An override to the emission model dust curves. Either:
                    - None, indicating the dust_curves defined on the emission
                      models should be used.
                    - A single dust curve to apply to all emission models.
                    - A dictionary of the form:
                          {<label>: <dust_curve instance>}
                      to use a specific dust curve instance with particular
                      properties.
            tau_v (dict):
                An override to the dust model optical depth. Either:
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
                An override to the emission model escape fraction. Either:
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
                An override to the emission model covering fraction. Either:
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
                An override to the emission model mask. Either:
                    - None, indicating the mask defined on the emission model
                      should be used.
                    - A dictionary of the form:
                      {<label>: {"attr": <attr>, "thresh": <thresh>, "op":<op>}
                      to add a specific mask to a particular model.
            verbose (bool):
                Are we talking?
            lines (dict):
                A dictionary of lines to add to before executing the model.
            particle_lines (dict):
                A dictionary of particle lines to add to before executing the
                model.
            nthreads (int):
                The number of threads to use when generating the lines.
            grid_assignment_method (str):
                The method to use when assigning particles to the grid. Options
                are "cic" (cloud in cell) or "ngp" (nearest grid point).
            **kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of lines which can be attached to the
                appropriate lines attribute of the component
                (lines/particle_lines)
        """
        # A variation is a declaration, not a value, so nothing can be
        # generated while one is still waiting to be expanded
        self._check_expanded()

        # We don't want to modify the original emission model with any
        # modifications made here so we'll make a copy of it (this is a
        # shallow copy so very cheap and doesn't copy any pointed to objects
        # only their reference)
        emission_model = copy.copy(self)

        # Apply any overrides we have
        self._apply_overrides(
            emission_model,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            covering_fraction=covering_fraction,
            mask=mask,
            vel_shift=None,
        )

        # Work with the overridden root instance stored in the model tree so
        # root-level overrides are reflected in any queue and reuse logic.
        root_model = emission_model._models[self.label]

        # If we haven't got a lines dictionary yet we'll make one
        if lines is None:
            lines = {}
        if particle_lines is None:
            particle_lines = {}

        # We need to make sure the root is being saved, otherwise this is a bit
        # nonsensical.
        if not root_model.save:
            raise exceptions.InconsistentArguments(
                f"{root_model.label} is not being saved. There's no point in "
                "generating at the root if they are not saved. Maybe you "
                "want to use a child model you are saving instead?"
            )

        # Build the execution queue for the active model tree before doing any
        # existing-emission reuse checks so inactive branches are skipped.
        with timer("EmissionModel._get_lines.build_model_queue"):
            queue = ModelQueue(root_model)

        # Before we do anything else, check that we have the emitters needed by
        # the active models in the queued execution graph.
        for model in queue.models.values():
            if emitters.get(model.emitter, None) is None:
                raise exceptions.InconsistentArguments(
                    f"Missing emitter '{model.emitter}' required by active "
                    f"EmissionModel '{model.label}'."
                )

        # Get any existing lines we are reusing
        lines, particle_lines = root_model._get_existing_emissions(
            emitters,
            lines,
            particle_lines,
            emission_type="lines",
            models=queue.models.values(),
        )

        # Collect existing spectra from all emitters for scaling purposes
        spectra = {}
        particle_spectra = {}
        for emitter in emitters.values():
            if hasattr(emitter, "spectra"):
                spectra.update(emitter.spectra)
            if hasattr(emitter, "particle_spectra"):
                particle_spectra.update(emitter.particle_spectra)

        # Cache a representative wavelength array before any eager deletion.
        line_lams = None
        if len(lines) > 0:
            line_lams = lines[list(lines.keys())[0]].lam

        # Execute the full model closure by processing each ready model once.
        while len(queue) > 0:
            this_model = queue.pop()
            label = this_model.label

            # Reused or externally supplied emissions still need to unlock the
            # graph, but they do not need to be regenerated.
            if label in lines:
                if line_lams is None:
                    line_lams = lines[label].lam
                queue.done(this_model, lines, particle_lines)
                continue

            # Active queued models must always have a matching emitter.
            if this_model.emitter not in emitters:
                raise exceptions.InconsistentArguments(
                    f"Active EmissionModel '{this_model.label}' requires "
                    f"missing emitter '{this_model.emitter}'."
                )

            # Get the emitter for this model now that it is ready to execute.
            emitter = emitters[this_model.emitter]

            # Build the combined mask once for operations that use it.
            this_mask = None
            for mask_dict in this_model.masks:
                this_mask = emitter.get_mask(
                    **mask_dict,
                    mask=this_mask,
                    attr_override_obj=this_model,
                )

            # Dispatch to the appropriate operation for this model.
            if this_model._is_extracting:
                try:
                    lines, particle_lines = self._extract_lines(
                        line_ids,
                        this_model,
                        emitters,
                        lines,
                        particle_lines,
                        verbose=verbose,
                        nthreads=nthreads,
                        grid_assignment_method=grid_assignment_method,
                    )
                    if line_lams is None and label in lines:
                        line_lams = lines[label].lam
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {label}]"
                        ).with_traceback(e.__traceback__)
            elif this_model._is_combining:
                try:
                    lines, particle_lines = self._combine_lines(
                        emission_model,
                        lines,
                        particle_lines,
                        this_model,
                        emitter,
                    )
                    if line_lams is None and label in lines:
                        line_lams = lines[label].lam
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)
            elif this_model._is_transforming:
                try:
                    lines, particle_lines = this_model._transform_emission(
                        this_model,
                        lines,
                        particle_lines,
                        emitter,
                        this_mask,
                        self.lam,
                    )
                    if line_lams is None and label in lines:
                        line_lams = lines[label].lam
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)
            elif this_model._is_generating:
                try:
                    lines, particle_lines = self._generate_lines(
                        this_model,
                        emission_model,
                        lines,
                        particle_lines,
                        emitter,
                        line_lams,
                        line_ids,
                        spectra,
                        particle_spectra,
                    )
                    if line_lams is None and label in lines:
                        line_lams = lines[label].lam
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)

            # Apply any requested scaling after the model has been generated.
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

            # Unlock downstream models and delete expired unsaved emissions.
            queue.done(this_model, lines, particle_lines)

        # Ensure the dependency graph was fully traversed before returning.
        queue.assert_finished()

        # Apply any post processing functions to the surviving emissions.
        for func in self._post_processing:
            lines = func(lines, emitters, self)
            if len(particle_lines) > 0:
                particle_lines = func(particle_lines, emitters, self)

        return lines, particle_lines

    def add_label_prefix(self, prefix):
        """Re-labels spectra by adding a prefix.

        This will relabel all spectra in the model by adding a prefix.

        Args:
            prefix (str):
                The prefix to use when relabelling.
        """
        self._relabel_models(
            {label: f"{prefix}_{label}" for label in self._models}
        )

    def _get_downstream_labels(self, label):
        """Get the label of a model and the labels of all its dependents.

        A model's dependents are the models which consume its emission, i.e.
        its parents in the tree, and their parents in turn. This is the set of
        models which must be duplicated alongside a model when that model is
        varied.

        Args:
            label (str):
                The label of the model to walk from.

        Returns:
            set:
                The label itself along with the labels of every model which
                depends on it, directly or indirectly.
        """
        # Ensure the label exists
        if label not in self._models:
            raise exceptions.InconsistentArguments(
                f"Could not find a model with the label: {label}"
            )

        # Walk up through the parents collecting everything we touch. Iterate
        # in label order since _parents is a set of objects and would
        # otherwise be traversed in a memory address dependent order.
        downstream = set()
        pending = [self._models[label]]
        while len(pending) > 0:
            model = pending.pop()

            # Skip models we've already collected (a diamond in the tree will
            # reach the same dependent by more than one route)
            if model.label in downstream:
                continue

            downstream.add(model.label)

            # Only follow parents which are still part of this tree. Replacing
            # a model, as expanding a parameter variation does, leaves the
            # models it depended on still pointing back at it, and following
            # those would collect models which are no longer here at all.
            pending.extend(
                parent
                for parent in sorted(model._parents, key=lambda m: m.label)
                if self._models.get(parent.label) is parent
            )

        return downstream

    def _relabel_models(self, label_map):
        """Relabel a set of models, rewriting any references to them.

        Labels can be referenced by other models as strings (in apply_to,
        combine, and scale_by) where the string denotes an emission rather than
        a model object. Any such reference to a model we are relabelling must
        be rewritten, otherwise the reference either silently detaches from the
        tree or resolves against a stale emission on the emitter.

        Note that strings in fixed_parameters are deliberately left alone.
        Those point at attributes of the emitter, not at models.

        Args:
            label_map (dict):
                A dictionary of the form {<old_label>: <new_label>} defining
                the models to relabel.

        Raises:
            InconsistentArguments
                If an old label doesn't exist, or a new label is already in
                use by a model which isn't being relabelled.
        """
        # Nothing to do without a mapping
        if len(label_map) == 0:
            return

        # Ensure every model we've been asked to relabel exists
        missing = set(label_map) - set(self._models)
        if len(missing) > 0:
            raise exceptions.InconsistentArguments(
                f"Could not find models with the labels: {sorted(missing)}"
            )

        # Ensure we aren't about to collide with a label which is staying put.
        # Note that we don't check against the labels being relabelled since
        # those are all moving at once.
        staying = set(self._models) - set(label_map)
        collisions = staying & set(label_map.values())
        if len(collisions) > 0:
            raise exceptions.InconsistentArguments(
                f"Labels {sorted(collisions)} are already in use."
            )

        # Ensure the new labels don't collide with each other
        new_labels = list(label_map.values())
        if len(set(new_labels)) != len(new_labels):
            raise exceptions.InconsistentArguments(
                f"Relabelling would produce duplicate labels: {new_labels}"
            )

        # Apply the new labels and rewrite any string references. We do this
        # in a single pass over every model rather than one relabel at a time
        # to avoid transient label collisions part way through.
        for model in self._models.values():
            # Relabel the model itself
            if model.label in label_map:
                model.label = label_map[model.label]

            # Rewrite a string apply_to reference
            if model._is_transforming and isinstance(model.apply_to, str):
                if model.apply_to in label_map:
                    model._apply_to = label_map[model.apply_to]

            # Rewrite any string entries in combine
            if model._is_combining:
                model._combine = [
                    label_map.get(child, child)
                    if isinstance(child, str)
                    else child
                    for child in model._combine
                ]

            # Rewrite any string entries in scale_by. Note that a string here
            # can also be an emitter attribute, so we only rewrite it when it
            # matches a model we are relabelling.
            model._scale_by = tuple(
                label_map.get(scaler, scaler)
                if isinstance(scaler, str)
                else scaler
                for scaler in model._scale_by
            )

        # Unpack now we're done to rebuild the label keyed containers
        self.unpack_model()

    def add_label_suffix(self, suffix):
        """Re-label all models in the tree by adding a suffix.

        This is the counterpart to add_label_prefix. Where a prefix is
        typically used to namespace a whole model (e.g. "stellar" vs "agn"), a
        suffix is typically used to distinguish variations of a model (e.g. a
        particular escape fraction).

        Args:
            suffix (str):
                The suffix to append to every label. An underscore is inserted
                between the label and the suffix.
        """
        self._relabel_models(
            {label: f"{label}_{suffix}" for label in self._models}
        )

    def add_label_suffix_downstream(self, label, suffix):
        """Re-label a model and everything depending on it by adding a suffix.

        Only the given model and the models which consume its emission
        (directly or indirectly) are relabelled. Anything the model depends on
        keeps its label, and is therefore still shared with the unmodified
        tree rather than duplicated.

        This is the relabelling behaviour needed when varying a parameter on a
        single model: the varied model produces a different emission, so
        everything downstream of it does too, but everything upstream is
        unaffected.

        Args:
            label (str):
                The label of the model to start from.
            suffix (str):
                The suffix to append to the affected labels. An underscore is
                inserted between the label and the suffix.
        """
        self._relabel_models(
            {
                downstream_label: f"{downstream_label}_{suffix}"
                for downstream_label in self._get_downstream_labels(label)
            }
        )


class StellarEmissionModel(EmissionModel):
    """An emission model for stellar components.

    This is a simple wrapper to quickly apply that the emitter a model
    should act on is stellar.

    Attributes:
        emitter (str):
            The emitter this model is for.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a StellarEmissionModel instance."""
        kwargs.setdefault("emitter", "stellar")
        EmissionModel.__init__(self, *args, **kwargs)


class BlackHoleEmissionModel(EmissionModel):
    """An emission model for black hole components.

    This is a simple wrapper to quickly apply that the emitter a model
    should act on is a black hole.

    Attributes:
        emitter (str):
            The emitter this model is for.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a BlackHoleEmissionModel instance."""
        kwargs.setdefault("emitter", "blackhole")
        EmissionModel.__init__(self, *args, **kwargs)


class GalaxyEmissionModel(EmissionModel):
    """An emission model for whole galaxy.

    A galaxy model sets emitter to "galaxy" to flag to the get_spectra method
    that the model is for a galaxy. By definition a galaxy level spectra can
    only be a combination of component spectra.

    Attributes:
        emitter (str):
            The emitter this model is for.
    """

    def __init__(self, *args, **kwargs):
        """Instantiate a GalaxyEmissionModel instance."""
        kwargs.setdefault("emitter", "galaxy")
        EmissionModel.__init__(self, *args, **kwargs)

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
