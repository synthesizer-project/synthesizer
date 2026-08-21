"""A submodule containing the expansion of parameter variations into models.

An EmissionModel can declare that one of its parameters should be varied over
a set of values, by passing a ParameterList or ParameterDistribution instead of
a single value. Nothing happens until EmissionModel.expand_models is called,
at which point the declarations in the tree are turned into concrete models by
the machinery in this module.

The expansion works by repeatedly duplicating the "cone" of a varied model,
i.e. the model itself plus every model which depends on it:

    1. Every ParameterDistribution is realised into a ParameterList. This
       happens once, up front, so that models downstream of a varied model
       share one set of samples rather than resampling in each branch.
    2. The deepest model still carrying a variation is found, and its cone is
       duplicated once per value, with each copy labelled using the
       variation's label_modifier. Models the cone depends on are not
       duplicated, they are shared by every copy.
    3. Step 2 repeats until no variations remain. Duplicating a cone leaves
       several copies of any downstream variation, each of which is then
       expanded in turn, so independent variations multiply out.

Working cone by cone means a model which is depended on by two different
models is only duplicated once, since the cone contains both routes. It also
means the labels accumulate their suffixes in dependency order without any
explicit sorting.

The result is a single root model with the other root variants attached as
related models. ModelQueue treats related models as extra roots and prunes to
whatever feeds a saved output, so one get_spectra call generates every variant
while computing the shared, unvaried models exactly once.

Each pass around the loop re-derives the label keyed view of every model (see
_gather), so the work is proportional to the number of variations times the
number of models. That is cheap next to generating an emission: a
BimodalPacmanEmission with three axes expanding into 100 variants and 522
models takes ~8 ms.
"""

from synthesizer import exceptions
from synthesizer.emission_models.parameters import (
    ParameterDistribution,
    find_variations,
    set_variation_value,
)


def _realise_distributions(root):
    """Replace every distribution in a tree with a realised ParameterList.

    A premade model often hands the same declaration to several of its own
    models, so each distribution is sampled once and the result shared.
    Sampling per model instead would give each of them different values and
    turn one variation into several.

    Args:
        root (EmissionModel):
            The root of the tree to realise distributions in. Modified in
            place.
    """
    realised = {}
    for model in root._models.values():
        for param, variation in find_variations(model).items():
            if not isinstance(variation, ParameterDistribution):
                continue

            if id(variation) not in realised:
                realised[id(variation)] = variation.realise()

            set_variation_value(model, param, realised[id(variation)])


def _clear_related_models(root):
    """Drop every related model reference in a tree.

    Related models are extra roots, which is how a model tree keeps hold of
    something not reachable from its own root. During an expansion they would
    also keep hold of models which have been replaced, so the expansion carries
    every model in one set instead and puts the references back at the end.

    Args:
        root (EmissionModel): The root of the tree. Modified in place.
    """
    for model in root._models.values():
        model.related_models = set()


def _gather(designated, models):
    """Gather a set of models into one tree, rooted at a given model.

    Args:
        designated (EmissionModel): The model to root the tree at.
        models (set): Every model which must be reachable from it.

    Returns:
        EmissionModel:
            The designated model, with the label keyed containers rebuilt over
            every model in the set.
    """
    # Anything already reachable through the tree is reached twice here, which
    # unpacking is happy with: it keys models by label and stops when it meets
    # one it has already seen.
    designated.related_models = models - {designated}
    designated.unpack_model()

    return designated


def _attach_roots(designated, models):
    """Finish an expansion by attaching the true roots of the result.

    Args:
        designated (EmissionModel): The model the result is rooted at.
        models (set): Every model in the result.

    Returns:
        EmissionModel:
            The designated model, holding the other roots as related models.
    """
    # Every model is reachable while this is gathered, which is what lets the
    # parent pointers be rebuilt, and a root is a model nothing depends on
    _gather(designated, models)

    roots = {model for model in models if len(model._parents) == 0} - {
        designated
    }

    designated.related_models = roots
    designated.unpack_model()

    return designated


def _copy_subgraph(root, labels, suffix):
    """Copy a set of models, optionally relabelling them.

    References from a copied model to a model outside the set are left
    pointing at the original, so the copies share it rather than duplicating
    it. References between copied models are rewired to the copies, and string
    references to copied models are rewritten to follow the relabelling.

    Args:
        root (EmissionModel): The root of the tree the models belong to.
        labels (set): The labels of the models to copy.
        suffix (str):
            A suffix to append to the label of every copy. Pass an empty
            string to keep the labels as they are.

    Returns:
        dict:
            A dictionary of the form {<original_label>: <copied model>}.
    """
    # Copy each model in isolation first so the wiring below can look up any
    # copy it needs
    copies = {
        label: root[label]._copy_node(label=label + suffix) for label in labels
    }

    def _copy_of(model_or_label):
        """Return the copy of a model or label, or None if it wasn't copied."""
        label = (
            model_or_label
            if isinstance(model_or_label, str)
            else model_or_label.label
        )
        return copies.get(label, None)

    # Now rewire the copies to each other
    for label, new_model in copies.items():
        old_model = root[label]

        # Point a transformation at the copy of its target
        if old_model._is_transforming:
            target = _copy_of(old_model.apply_to)
            if target is not None:
                # A string target names an emission rather than a model, so it
                # follows the relabelling as a string
                new_model._apply_to = (
                    target.label
                    if isinstance(old_model.apply_to, str)
                    else target
                )

        # Point a combination at the copies of its children
        if old_model._is_combining:
            new_combine = []
            for child in old_model.combine:
                child_copy = _copy_of(child)
                if child_copy is None:
                    new_combine.append(child)
                elif isinstance(child, str):
                    new_combine.append(child_copy.label)
                else:
                    new_combine.append(child_copy)
            new_model._combine = new_combine

        # Point a generator at the copies of its dependencies
        for attr in old_model._generator_model_attrs:
            dependency = getattr(old_model.generator, attr)
            dependency_copy = _copy_of(dependency)
            if dependency_copy is not None:
                new_model._replace_generator_dependency(
                    dependency,
                    dependency_copy,
                )

        # Rewrite scale_by, which holds labels rather than models. A label here
        # can also name an emitter attribute, so only rewrite it when it names
        # a model we copied.
        new_model._scale_by = tuple(
            _copy_of(scaler).label if _copy_of(scaler) is not None else scaler
            for scaler in old_model.scale_by
        )

        # Point at the copies of any related models
        new_model.related_models = {
            _copy_of(related) or related
            for related in old_model.related_models
        }

    return copies


def _dependency_labels(model):
    """Return the labels of everything a model depends on.

    Args:
        model (EmissionModel): The model to walk down from.

    Returns:
        set:
            The labels of every model this model depends on, directly or
            indirectly. The model's own label is not included.
    """
    # Walk down through the children. Iterate in label order since _children
    # is a set of objects and would otherwise be traversed in a memory address
    # dependent order.
    dependencies = set()
    pending = sorted(model._children, key=lambda m: m.label)
    while len(pending) > 0:
        child = pending.pop()

        # Skip children we've already collected (a diamond in the tree reaches
        # the same dependency by more than one route)
        if child.label in dependencies:
            continue

        dependencies.add(child.label)
        pending.extend(sorted(child._children, key=lambda m: m.label))

    return dependencies


def _find_variation_target(root):
    """Find the next variation to expand.

    A single declaration can be attached to several models at once, which is
    what a premade model does when it hands one escape fraction down to each of
    the models that needs it. Those are one variation, not several, so they are
    expanded together: they are recognised by being the same declaration rather
    than by holding equal values.

    Variations are expanded from the bottom of the tree up, so a model is only
    expanded once everything it depends on has been. This is what makes
    independent variations multiply out: expanding a dependency first leaves
    several copies of this model, each of which is then expanded in turn.

    Args:
        root (EmissionModel): The root of the tree to search.

    Returns:
        tuple or None:
            The (label, parameter) pairs sharing one declaration, and the
            ParameterList they share, or None when no variations remain.
    """
    # Gather where each declaration has ended up, keyed by the declaration
    # itself rather than by its values
    groups = {}
    for label, model in sorted(root._models.items()):
        for param, variation in sorted(find_variations(model).items()):
            group, _ = groups.setdefault(id(variation), ([], variation))
            group.append((label, param))

    if len(groups) == 0:
        return None

    # Which declaration each model is waiting on
    waiting = {}
    for key, (group, _) in groups.items():
        for label, _param in group:
            waiting[label] = key

    # A declaration can only be expanded once every declaration its models
    # depend on has been. Members of the same group don't count: they are one
    # variation and are expanded in one go.
    ready = []
    for key, (group, _) in groups.items():
        dependencies = set()
        for label, _param in group:
            dependencies |= _dependency_labels(root[label])

        blocking = {
            waiting[label]
            for label in dependencies
            if label in waiting and waiting[label] != key
        }
        if len(blocking) == 0:
            ready.append(key)

    # A cycle in the tree would leave nothing ready, which unpack_model should
    # already have made impossible
    if len(ready) == 0:
        raise exceptions.InconsistentArguments(
            "Could not find a variation to expand which has no varied "
            f"dependencies (varied models: {sorted(waiting)}). This suggests "
            "a cyclic dependency in the model tree."
        )

    # Sort so the expansion order, and therefore the order suffixes are
    # applied in, is reproducible
    key = min(ready, key=lambda key: sorted(groups[key][0]))
    group, variation = groups[key]

    return sorted(group), variation


def _recorded_params(group, value, already_recorded):
    """Return the varied parameters to record on a variant, and their names.

    An argument of a transformer or generator is found under a dotted name
    ("transformer.slope"), but the bare name is what the user wrote and is what
    they will look for, so that is what is recorded. The exception is a clash:
    if a model parameter and a transformer argument in the same model share a
    name, recording both under the bare name would lose one of them, so the
    dotted name is kept for the one which needs qualifying.

    Args:
        group (list): The (label, parameter) pairs sharing this declaration.
        value:
            The value this variant uses.
        already_recorded (dict):
            What earlier expansions recorded on this model.

    Returns:
        dict:
            The parameters to record, keyed by the name to record them under.
    """
    recorded = {}
    for _label, param in group:
        bare = param.split(".")[-1]

        # Keep the qualified name when the bare one is taken by a different
        # parameter, so nothing is silently dropped
        clashes = bare in already_recorded and already_recorded[bare] != value
        recorded[param if clashes else bare] = value

    return recorded


def _expand_variation(root, designated, models, group, parameter_list):
    """Expand a single variation into one model per value.

    The varied model and everything depending on it (its cone) is duplicated
    once per value. Everything the cone depends on is shared between the
    copies rather than duplicated, which is the entire point of the exercise:
    the models which don't change are only computed once.

    Args:
        root (EmissionModel):
            The gathered view of every model, i.e. the designated model with
            the rest attached. Models outside the cone are shared with the
            result rather than copied.
        designated (EmissionModel):
            The model the result is rooted at, which stays the root through the
            expansion so the model the user calls is the model they built.
        models (set): Every model going into this expansion.
        group (list):
            The (label, parameter) pairs sharing this declaration. They are one
            variation and are expanded together, so a parameter handed to
            several models varies them in step rather than independently.
        parameter_list (ParameterList): The values to vary the parameters over.

    Returns:
        designated (EmissionModel): The model the result is rooted at.
        models (set): Every model coming out of this expansion.
    """
    # Find everything affected by this variation. A declaration attached to
    # several models affects everything downstream of any of them.
    cone = set()
    for label, _param in group:
        cone |= root._get_downstream_labels(label)

    # The models in the cone are replaced by their copies; everything else
    # carries straight through and is shared by every variant
    expanded = {model for model in models if model.label not in cone}

    # The model the result is rooted at is replaced by its copy from the first
    # value, if it is in the cone at all, so the label the user asked for still
    # exists on the model they get back
    designated_label = designated.label
    first_copies = None

    for value, suffix in parameter_list.items():
        copies = _copy_subgraph(root, cone, suffix=suffix)

        # Every model is carried in the set, so the copies keep no references
        # of their own to models this expansion has replaced
        for new_model in copies.values():
            new_model.related_models = set()

        # Fix the parameters on this variant, replacing the declaration. A
        # declaration on the transformer or the generator is set on the model
        # itself rather than among its parameters.
        for label, param in group:
            set_variation_value(copies[label], param, value)

        # Record what distinguishes every model in this variant's cone, so the
        # variant a model belongs to is recoverable without parsing labels.
        # The base label is the label before any variation was expanded, which
        # is what groups a family of variants back together.
        for old_label, new_model in copies.items():
            old_model = root[old_label]
            new_model._variant_params = {
                **old_model.variant_params,
                **_recorded_params(group, value, old_model.variant_params),
            }
            new_model._variant_base = (
                old_model.variant_base
                if old_model.variant_base is not None
                else old_label
            )

        expanded |= set(copies.values())

        if first_copies is None:
            first_copies = copies

    if designated_label in cone:
        designated = first_copies[designated_label]

    return designated, expanded


def expand_models(root):
    """Expand every parameter variation in a model tree into new models.

    Args:
        root (EmissionModel):
            The root of the model tree to expand. This model is not modified,
            the expansion is performed on a copy.

    Returns:
        EmissionModel:
            The root of the expanded tree. If the root itself was varied (or
            depends on something which was) this is the first of the root's
            variants, with the rest attached as related models.
    """
    # Work on a copy so the user's model is left exactly as they built it
    working = _copy_subgraph(root, set(root._models), suffix="")[root.label]
    working.unpack_model()

    # Turn every distribution into a concrete list of values before anything
    # is duplicated. Sampling per copy instead would give each branch a
    # different set of values.
    _realise_distributions(working)

    # Expanding a variation replaces models with copies of them, and a related
    # model reference left pointing at a replaced model would resurrect it and
    # the expansion would never end. Rather than track which references are
    # still live, drop them all and carry every model in one set instead: they
    # are re-derived at the end.
    _clear_related_models(working)
    models = set(working._models.values())
    designated = working

    # Expand one variation at a time, deepest first, until none are left
    while True:
        combined = _gather(designated, models)

        target = _find_variation_target(combined)
        if target is None:
            break

        designated, models = _expand_variation(
            combined,
            designated,
            models,
            *target,
        )

    return _attach_roots(designated, models)
