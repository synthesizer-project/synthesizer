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
"""

from synthesizer import exceptions
from synthesizer.emission_models.parameters import (
    ParameterDistribution,
    find_variations,
)


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

    # Expanding a variation turns one root into several, so from here on we
    # track the roots explicitly rather than leaving them scattered through
    # related_models. A stale related model reference would resurrect a model
    # we have already expanded and the expansion would never terminate.
    roots = _hoist_related_models(working)

    # Expand one variation at a time, deepest first, until none are left
    while True:
        combined = _combine_roots(roots)

        target = _find_variation_target(combined)
        if target is None:
            break

        roots = _expand_variation(combined, roots, *target)

    return combined


def _hoist_related_models(root):
    """Collect the related models in a tree as roots in their own right.

    Related models are extra roots: ModelQueue walks them alongside the root
    model. Gathering them into a single list of roots, and clearing the
    references, gives the expansion one place to track them.

    Args:
        root (EmissionModel):
            The root of the tree. Modified in place.

    Returns:
        list:
            The root, followed by every related model found in the tree.
    """
    roots = [root]
    seen = {root.label}

    # Walk in label order so the root list is reproducible. Note that
    # root._models already contains the related models, since unpack_model
    # walks them.
    for model in sorted(root._models.values(), key=lambda m: m.label):
        for related in sorted(model.related_models, key=lambda m: m.label):
            if related.label not in seen:
                seen.add(related.label)
                roots.append(related)

        model.related_models = set()

    return roots


def _combine_roots(roots):
    """Gather a list of roots back into a single tree.

    Args:
        roots (list):
            The roots to combine. The first is the root of the resulting tree.

    Returns:
        EmissionModel:
            The first root, with the rest attached as related models and the
            label keyed containers rebuilt.
    """
    designated_root = roots[0]
    designated_root.related_models = set(roots[1:])
    designated_root.unpack_model()

    return designated_root


def _realise_distributions(root):
    """Replace every distribution in a tree with a realised ParameterList.

    Args:
        root (EmissionModel):
            The root of the tree to realise distributions in. Modified in
            place.
    """
    for model in root._models.values():
        for param, variation in find_variations(model).items():
            if isinstance(variation, ParameterDistribution):
                model.fixed_parameters[param] = variation.realise()


def _find_variation_target(root):
    """Find the next variation to expand.

    Variations are expanded from the bottom of the tree up, so a model is only
    expanded once everything it depends on has been. This is what makes
    independent variations multiply out: expanding a dependency first leaves
    several copies of this model, each of which is then expanded in turn.

    Args:
        root (EmissionModel):
            The root of the tree to search.

    Returns:
        tuple or None:
            The label, parameter name and ParameterList of the variation to
            expand next, or None when no variations remain.
    """
    # Collect every model still carrying a variation
    varied = {
        label: find_variations(model)
        for label, model in root._models.items()
        if len(find_variations(model)) > 0
    }
    if len(varied) == 0:
        return None

    # A model can only be expanded once its dependencies have been, so keep
    # only those with no varied model among their dependencies
    ready = [
        label
        for label in varied
        if len(_dependency_labels(root[label]) & set(varied)) == 0
    ]

    # A cycle in the tree would leave nothing ready, which unpack_model should
    # already have made impossible
    if len(ready) == 0:
        raise exceptions.InconsistentArguments(
            "Could not find a variation to expand which has no varied "
            f"dependencies (varied models: {sorted(varied)}). This suggests a "
            "cyclic dependency in the model tree."
        )

    # Sort so the expansion order, and therefore the order suffixes are
    # applied in, is reproducible
    label = min(ready)
    param = min(varied[label])

    return label, param, varied[label][param]


def _dependency_labels(model):
    """Return the labels of everything a model depends on.

    Args:
        model (EmissionModel):
            The model to walk down from.

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


def _expand_variation(root, roots, label, param, parameter_list):
    """Expand a single variation into one model per value.

    The varied model and everything depending on it (its cone) is duplicated
    once per value. Everything the cone depends on is shared between the
    copies rather than duplicated, which is the entire point of the exercise:
    the models which don't change are only computed once.

    Args:
        root (EmissionModel):
            The combined view of the forest being expanded, i.e. roots[0] with
            the other roots attached. Its models are not modified, but they are
            shared with the returned roots wherever they fall outside the cone.
        roots (list):
            The current roots of the forest. The first is the root the user
            called, and stays the root of the result.
        label (str):
            The label of the varied model.
        param (str):
            The name of the varied parameter.
        parameter_list (ParameterList):
            The values to vary the parameter over.

    Returns:
        list:
            The roots of the expanded forest, with the user's root first.
    """
    # Find everything affected by this variation
    cone = root._get_downstream_labels(label)

    # Duplicate the cone once per value, collecting the new roots as we go.
    # Every root in the cone gains a copy per value, while roots outside it are
    # untouched and shared between the variants.
    designated_root = None
    other_roots = []
    for value in parameter_list.values:
        copies = _copy_subgraph(
            root,
            cone,
            suffix=parameter_list.suffix(value),
        )

        # The forest is tracked through the root list, so drop the related
        # model references the copies inherited from the combined view. Leaving
        # them would resurrect the models this expansion replaces.
        for new_model in copies.values():
            new_model.related_models = set()

        # Fix the parameter on this variant, replacing the declaration
        copies[label].fixed_parameters[param] = value

        # Record what distinguishes every model in this variant's cone, so the
        # variant a model belongs to is recoverable without parsing labels.
        # The base label is the label before any variation was expanded, which
        # is what groups a family of variants back together.
        for old_label, new_model in copies.items():
            old_model = root[old_label]
            new_model._variant_params = {
                **old_model.variant_params,
                param: value,
            }
            new_model._variant_base = (
                old_model.variant_base
                if old_model.variant_base is not None
                else old_label
            )

        for old_root in roots:
            if old_root.label not in cone:
                continue

            # Keep the first copy of the user's root as the root of the result,
            # so the model they call is still the model they built
            if old_root is roots[0] and designated_root is None:
                designated_root = copies[old_root.label]
            else:
                other_roots.append(copies[old_root.label])

    # Roots outside the cone survive untouched, shared by every variant
    for old_root in roots:
        if old_root.label in cone:
            continue

        if old_root is roots[0]:
            designated_root = old_root
        else:
            other_roots.append(old_root)

    return [designated_root, *other_roots]


def _copy_subgraph(root, labels, suffix):
    """Copy a set of models, optionally relabelling them.

    References from a copied model to a model outside the set are left
    pointing at the original, so the copies share it rather than duplicating
    it. References between copied models are rewired to the copies, and string
    references to copied models are rewritten to follow the relabelling.

    Args:
        root (EmissionModel):
            The root of the tree the models belong to.
        labels (set):
            The labels of the models to copy.
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
