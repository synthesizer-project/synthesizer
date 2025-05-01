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
import sys

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from unyt import arcsecond, kpc, unyt_quantity

from synthesizer import exceptions
from synthesizer.emission_models.operations import (
    Combination,
    Extraction,
    Generation,
    Transformation,
)
from synthesizer.extensions.timers import tic, toc
from synthesizer.imaging import ImageCollection
from synthesizer.imaging.image_generators import (
    _generate_image_collection_generic,
)
from synthesizer.photometry import PhotometryCollection
from synthesizer.synth_warnings import deprecation, warn
from synthesizer.units import Quantity, accepts


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
    toegther network of models we call the tree. The tree has a single
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
            in the construction of the "root" model encapulated by self.
        fixed_parameters (dict):
            A dictionary of component attributes/parameters which should be
            fixed and thus ignore the value of the component attribute. This
            should take the form {<parameter_name>: <value>}.
        emitter (str):
            The emitter this emission model acts on. Default is "stellar".
        apply_to (EmissionModel):
            The model to apply the dust curve to.
        dust_curve (emission_models.attenuation.*):
            The dust curve to apply.
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
        lum_intrinsic_model=None,
        lum_attenuated_model=None,
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
                is a friendly alias arguement for the transformer argument.
                Setting both will raise an exception.
            igm (emission_models.transformers.igm.*):
                The IGM model to apply (when doing a transformation). This is
                a friendly alias arguement for the transformer argument.
                Setting both will raise an exception.
            generator (DustEmission/...):
                The emission generation model. This must define a get_spectra
                method.
            transformer (Transformer):
                The transform to apply. This is also an alternative (but
                less obvious) argument for passing a dust curve or IGM model
                (both are transformers).
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
                The operation to apply. Can be "<", ">", "<=", ">=", "==",
                or "!=".
            lam_mask (ndarray):
                The mask to apply to the wavelength array.
            related_models (set/list/EmissionModel):
                A set of related models to this model. A related model is a
                model that is connected somewhere within the model tree but is
                required in the construction of the "root" model encapulated by
                self.
            emitter (str):
                The emitter this emission model acts on. Default is
                "stellar".
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

        # Attach which emitter we are working with
        self._emitter = emitter

        # Are we making per particle emission?
        self._per_particle = per_particle

        # Define the container which will hold mask information
        self.masks = []

        # If we have been given a mask, add it
        self._init_masks(mask_attr, mask_thresh, mask_op)

        # Initialise the lam mask
        self._lam_mask = lam_mask

        # Temporarily handle old apply_dust_to argument
        if "apply_dust_to" in fixed_parameters:
            deprecation(
                "The apply_dust_to argument has been deprecated. "
                "Please use the apply_to argument instead."
            )
            apply_to = fixed_parameters.pop("apply_dust_to")

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
            lum_intrinsic_model=lum_intrinsic_model,
            lum_attenuated_model=lum_attenuated_model,
        )

        # Initilaise the corresponding operation (also checks we have a
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
            lum_intrinsic_model=lum_intrinsic_model,
            lum_attenuated_model=lum_attenuated_model,
            vel_shift=vel_shift,
        )

        # Containers for children and parents
        self._children = set()
        self._parents = set()

        # Store the arribute to scale the emission by
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
        apply_to,
        transformer,
        generator,
        lum_intrinsic_model,
        lum_attenuated_model,
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
                The emission generation model. This must define a get_spectra
                method.
            lum_intrinsic_model (EmissionModel):
                The intrinsic model to use deriving the dust
                luminosity when computing dust emission.
            lum_attenuated_model (EmissionModel):
                The attenuated model to use deriving the dust
                luminosity when computing dust emission.
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
                f"(transformer={transformer}, "
                f"apply_to={apply_to})\n"
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
                    self._is_transforming,
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
                f"transform: {self._is_transforming}, "
                f"dust_emission: {self._is_dust_emitting})\n"
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
                f"lum_intrinsic_model={lum_intrinsic_model}, "
                f"lum_attenuated_model={lum_attenuated_model})"
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

        # Ensure the grid contains any keys we want to extract
        if self._is_extracting and extract not in grid.spectra.keys():
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
        elif self._is_transforming:
            return self._transform_summary()
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
        lum_attenuated_model=None,
        lum_intrinsic_model=None,
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
        self._is_dust_emitting = (
            generator is not None
            and lum_attenuated_model is not None
            and lum_intrinsic_model is not None
        )
        self._is_generating = (
            generator is not None and not self._is_dust_emitting
        )

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
            pass
        else:
            raise exceptions.InconsistentArguments(
                f"Label {model.label} is already in use."
            )

        # If we are extracting an emission, store the key
        if model._is_extracting:
            self._extract_keys[model.label] = model.extract

        # If we are applying a transform, store the key
        if model._is_transforming:
            self._transformation[model.label] = (
                model.apply_to,
                model.dust_curve,
            )
            model._children.add(model.apply_to)
            model.apply_to._parents.add(model)

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
        self._transformation = {}
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
    def lum_intrinsic_model(self):
        """Get the intrinsic model for computing dust luminosity."""
        return getattr(self, "_lum_intrinsic_model", None)

    def set_lum_intrinsic_model(self, lum_intrinsic_model):
        """Set the intrinsic model for computing dust luminosity.

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
        """Set the attenuated model for computing dust luminosity.

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

    def save_spectra(self, *args):
        """Set the save flag to True for the given spectra.

        Args:
            args (str):
                The spectra to save.
        """
        # First set all models to not save
        self.set_save(False, set_all=True)

        # Now set the given spectra to save
        for arg in args:
            self[arg].set_save(True)

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
        replace_model = self._models[replace_label]

        # Get the children and parents of this model
        parents = replace_model._parents
        children = replace_model._children

        # Define the relation to all parents and children
        relations = {}
        for parent in parents:
            if replace_model in parent.combine:
                relations[parent.label] = "combine"
            if parent.apply_to == replace_model:
                relations[parent.label] = "transform"
            if parent.lum_intrinsic_model == replace_model:
                relations[parent.label] = "dust_intrinsic"
            if parent.lum_attenuated_model == replace_model:
                relations[parent.label] = "dust_attenuated"
        for child in children:
            if child in replace_model.combine:
                relations[child.label] = "combine"
            if child.apply_to == replace_model:
                relations[child.label] = "transform"
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
            if relations[parent.label] == "transform":
                parent._apply_to = None
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
            if relations[parent.label] == "transform":
                parent._apply_to = new_model
            if relations[parent.label] == "dust_intrinsic":
                parent._lum_intrinsic_model = new_model
            if relations[parent.label] == "dust_attenuated":
                parent._lum_attenuated_model = new_model

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

    def fix_parameters(self, **kwargs):
        """Fix parameters of the model.

        Args:
            **kwargs:
                The parameters to fix.
        """
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
        elif self._is_dust_emitting:
            self.generate_to_hdf5(group)
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
                fixed_parameters.attrs[key] = value

        # Save the children
        if len(self._children) > 0:
            group.create_dataset(
                "Children",
                data=[child.label.encode("utf-8") for child in self._children],
            )

    def _get_tree_levels(self, root):
        """Get the levels of the tree.

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
                    (model.apply_to.label, "--")
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
        """Get the position of each model in the tree.

        Args:
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
        figsize=(6, 6),
    ):
        """Plot the tree defining the spectra.

        Args:
            root (str):
                If not None this defines the root of a sub tree to plot.
            show (bool):
                Whether to show the plot.
            fontsize (int):
                The fontsize to use for the labels.
            figsize (tuple):
                The size of the figure to plot (width, height).
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
            vel_shift (dict/bool):
                Overide the models flag for using peculiar velocities to apply
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
        _is_related=False,
        nthreads=1,
        grid_assignment_method="cic",
        **fixed_parameters,
    ):
        """Generate stellar spectra as described by the emission model.

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
            verbose (bool):
                Are we talking?
            spectra (dict):
                A dictionary of spectra to add to. This is used for recursive
                calls to this function.
            particle_spectra (dict):
                A dictionary of particle spectra to add to. This is used for
                recursive calls to this function.
            vel_shift (bool):
                Overide the models flag for using peculiar velocities to apply
                doppler shift to the generated spectra. Only applicable for
                particle spectra.
            _is_related (bool):
                Are we generating related model spectra? If so we don't want
                to apply any post processing functions or delete any spectra,
                this will be done outside the recursive call.
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
        start = tic()

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
        start_overrides = tic()
        if not _is_related:
            self._apply_overrides(
                emission_model,
                dust_curves=dust_curves,
                tau_v=tau_v,
                fesc=fesc,
                covering_fraction=covering_fraction,
                mask=mask,
                vel_shift=vel_shift,
            )
        toc("Applying model overrides", start_overrides)

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
        for label, spectra_key in emission_model._extract_keys.items():
            # Skip if we don't need to extract this spectra
            if label in spectra:
                continue
            try:
                spectra, particle_spectra = self._extract_spectra(
                    emission_model[label],
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
                        vel_shift=vel_shift,
                        verbose=verbose,
                        spectra=spectra,
                        particle_spectra=particle_spectra,
                        _is_related=True,
                        nthreads=nthreads,
                        **fixed_parameters,
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
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)

            # Are we doing a transformation?
            elif this_model._is_transforming:
                try:
                    spectra, particle_spectra = this_model._transform_emission(
                        this_model,
                        spectra,
                        particle_spectra,
                        emitter,
                        this_mask,
                    )
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)

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
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)

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

        toc("Generating all spectra", start)

        return spectra, particle_spectra

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
        _is_related=False,
        nthreads=1,
        grid_assignment_method="cic",
        **kwargs,
    ):
        """Generate stellar lines as described by the emission model.

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
            verbose (bool):
                Are we talking?
            lines (dict):
                A dictionary of lines to add to. This is used for recursive
                calls to this function.
            particle_lines (dict):
                A dictionary of particle lines to add to. This is used for
                recursive calls to this function.
            _is_related (bool):
                Are we generating related model lines? If so we don't want
                to apply any post processing functions or delete any lines,
                this will be done outside the recursive call.
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
        start = tic()

        # We don't want to modify the original emission model with any
        # modifications made here so we'll make a copy of it (this is a
        # shallow copy so very cheap and doesn't copy any pointed to objects
        # only their reference)
        emission_model = copy.copy(self)

        # Apply any overides we have
        self._apply_overrides(
            emission_model,
            dust_curves=dust_curves,
            tau_v=tau_v,
            fesc=fesc,
            covering_fraction=covering_fraction,
            mask=mask,
            vel_shift=None,
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

        # Perform all extractions first
        for label in emission_model._extract_keys.keys():
            # Skip it if we happen to already have the lines
            if label in lines:
                continue

            try:
                lines, particle_lines = self._extract_lines(
                    line_ids,
                    emission_model[label],
                    emitters,
                    lines,
                    particle_lines,
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
                        emission_model,
                        lines,
                        particle_lines,
                        this_model,
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
                    lines, particle_lines = this_model._transform_emission(
                        this_model,
                        lines,
                        particle_lines,
                        emitter,
                        this_mask,
                    )
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)

            elif this_model._is_dust_emitting or this_model._is_generating:
                try:
                    lines, particle_lines = self._generate_lines(
                        this_model,
                        emission_model,
                        lines,
                        particle_lines,
                        emitter,
                    )
                except Exception as e:
                    if sys.version_info >= (3, 11):
                        e.add_note(f"EmissionModel.label: {this_model.label}")
                        raise
                    else:
                        raise type(e)(
                            f"{e} [EmissionModel.label: {this_model.label}]"
                        ).with_traceback(e.__traceback__)

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

        toc("Generating all lines", start)

        return lines, particle_lines

    @accepts(fov=(kpc, arcsecond))
    def _get_images(
        self,
        instrument,
        fov,
        emitters,
        img_type="smoothed",
        images=None,
        _is_related=False,
        limit_to=None,
        do_flux=False,
        kernel=None,
        kernel_threshold=1.0,
        cosmo=None,
        nthreads=1,
        **kwargs,
    ):
        """Generate images as described by the emission model.

        This will create images for all models in the emission model which
        have been saved in the passed dictionary of photometry, unless
        limit_to is set to a specific model, in which case only that model
        will have images generated (including any models required for that
        passed model, e.g. a combination of AGN and Stellar spectra).

        Note that unlike spectra or line creation, images are always either
        generated from existing photometry or combined from existing images
        further down in the tree.

        Args:
            instrument (Instrument):
                The instrument to use for the image generation.
            fov (float):
                The field of view of the image in angular units (e.g. arcsec,
                arcmin, deg).
            emitters (Stars/BlackHoles):
                The emitters to generate the lines for in the form of a
                dictionary, {"stellar": <emitter>, "blackhole": <emitter>}.
            img_type (str):
                The type of image to generate. Options are "smoothed" or
                "unsmoothed". If "smoothed" is selected, the image will be
                convolved with a kernel. If "unsmoothed" is selected, the
                image will not be convolved with a kernel.
            images (dict):
                A dictionary of images to add to. This is used for recursive
                calls to this function.
            _is_related (bool):
                Are we generating related model lines? If so we don't want
                to apply any post processing functions or delete any lines,
                this will be done outside the recursive call.
            limit_to (str, list):
                If not None, defines a specific model (or list of models) to
                limit the image generation to. Otherwise, all models with saved
                spectra will have images generated.
            do_flux (bool):
                If True, the images will be generated from fluxes, if False
                they will be generated from luminosities.
            kernel (str):
                The convolution kernel to use for the image generation. If
                None, no convolution will be applied.
            kernel_threshold (float):
                The threshold for the convolution kernel.
            cosmo (Cosmology):
                The cosmology to use for the image generation. If None,
                the default cosmology will be used.
            nthreads (int):
                The number of threads to use for the image generation.
            **kwargs (dict):
                Any additional keyword arguments to pass to the generator
                function.

        Returns:
            dict
                A dictionary of ImageCollections which can be attached to the
                appropriate images attribute of the component.
        """
        start = tic()

        # We don't want to modify the original emission model with any
        # modifications made here so we'll make a copy of it (this is a
        # shallow copy so very cheap and doesn't copy any pointed to objects
        # only their reference)
        emission_model = copy.copy(self)

        # If we haven't got an images dictionary yet we'll make one
        if images is None:
            images = {}

        # Convert `limit_to` to a list if it is a string
        if limit_to is not None:
            limit_to = (
                [limit_to] if isinstance(limit_to, str) else limit_to.copy()
            )

        # If we are limiting to a specific model/s and these are a combination
        # model, we need to make sure we include the models they are
        # combining.
        _orig_limit_to = limit_to
        if limit_to is not None:
            _orig_limit_to = limit_to.copy()
            for label in limit_to:
                # Get this model
                this_model = emission_model._models[label]

                # If this is a combination model, add the models it is
                # combining to the list
                if this_model._is_combining:
                    limit_to.extend([m.label for m in this_model.combine])

            # Remove duplicates
            limit_to = list(set(limit_to))

        # Set up the list to collect all the photometry into so we can generate
        # images for all models at once
        photometry = {e: {} for e in emitters.keys()}

        # Loop through all models and collect their photometry
        for label in emission_model._models.keys():
            # Get this model
            this_model = emission_model._models[label]

            # If we are limiting to a specific model, skip all others
            if limit_to is not None and label not in limit_to:
                continue

            # Skip if we didn't save this model
            if not this_model.save:
                continue

            # Get the emitter
            emitter = (
                emitters[this_model.emitter]
                if this_model.emitter != "galaxy"
                else None
            )

            # If we have no emitter, we can't generate an image. This is
            # relevant when the emitter is a galaxy. In this case the images
            # must be combined in the loop below.
            if emitter is None:
                continue

            # Get the appropriate photometry (particle/integrated and
            # flux/luminosity)
            try:
                if do_flux:
                    this_phot = (
                        emitter.particle_photo_fnu[label]
                        if this_model.per_particle
                        else emitter.photo_fnu[label]
                    )
                else:
                    this_phot = (
                        emitter.particle_photo_lnu[label]
                        if this_model.per_particle
                        else emitter.photo_lnu[label]
                    )
            except KeyError:
                # Ok we are missing the photometry
                raise exceptions.MissingSpectraType(
                    f"Can't make an image for {label} without the photometry. "
                    "Did you not save the spectra or produce the photometry?"
                )

            # Get only the filters we want for this instrument
            this_phot = this_phot.select(*instrument.filters.filter_codes)

            # Include this photometry in the list
            for key, phot in this_phot.items():
                photometry[this_model.emitter][f"{label}--{key}"] = phot

        # With everything collected, we can now generate the images for each
        # emitter in one go
        for emitter in emitters.keys():
            # Do we have anything to do?
            if len(photometry[emitter]) == 0:
                continue

            # Create the combined photometry object that we'll pass to the
            # C extension for generating the images
            phot = PhotometryCollection(
                instrument.filters,
                **photometry[emitter],
            )

            # Generate the images for all the models. These will be labelled
            # incorrectly though
            _imgs = _generate_image_collection_generic(
                instrument,
                phot,
                fov,
                img_type=img_type,
                kernel=kernel,
                kernel_threshold=kernel_threshold,
                nthreads=nthreads,
                emitter=emitters[emitter],
                cosmo=cosmo,
            )

            # Now we need to loop over the imgs we've create and split them
            # into the correct models and populate the images dictionary
            individual_imgs = {}
            for key in _imgs.keys():
                # Get the model label
                model_label, fcode = key.split("--", 1)

                # Create the entry for this model if needed
                individual_imgs.setdefault(model_label, {})

                # Get the image for this model
                individual_imgs[model_label][fcode] = _imgs[key]

            # Finally, populate the images dictionary
            for key in individual_imgs.keys():
                images[key] = ImageCollection(
                    resolution=instrument.resolution,
                    fov=fov,
                    imgs=individual_imgs[key],
                )

        # Loop over combination models and create any images we haven't already
        for label in emission_model._bottom_to_top:
            # If we are limiting to a specific model, skip all others
            if limit_to is not None and label not in limit_to:
                continue

            # Get this model
            this_model = emission_model._models[label]

            # Skip if we didn't save this model
            if not this_model.save:
                continue

            # Check we haven't already made this image
            if label in images:
                continue

            # Call the appropriate method to generate the image for this model
            if this_model._is_combining:
                try:
                    images = self._combine_images(
                        images,
                        this_model,
                        instrument,
                        fov,
                        img_type,
                        do_flux,
                        emitters,
                        kernel,
                        kernel_threshold,
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

        # If we are limiting to a specific model, we might might have generated
        # images for models we don't want to hold on to. Throw them away
        # if we are limiting to a specific model (but only if not a related
        # call otherwise we might delete images we need for the combination
        # models)
        if limit_to is not None and not _is_related:
            for key in set(images) - set(_orig_limit_to):
                del images[key]

        toc("Generating all images", start)

        return images


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
        EmissionModel.__init__(self, *args, **kwargs)
        self._emitter = "stellar"


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
        EmissionModel.__init__(self, *args, **kwargs)
        self._emitter = "blackhole"


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
