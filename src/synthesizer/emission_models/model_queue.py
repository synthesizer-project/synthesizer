"""Queue machinery for executing emission model dependency graphs.

This module defines the runtime queue used by
``EmissionModel._get_spectra`` and ``EmissionModel._get_lines``. The queue is
responsible for constructing the executable closure of models for a single
call, compiling the direct dependency graph between those models, and then
tracking when each model becomes ready to execute.

The queue also manages emission lifetimes. Once a model's emission has been
consumed by all downstream dependents, and the model is not marked to be
saved, the queue deletes that emission from the working output dictionaries.
Keeping this logic in a dedicated module keeps execution-specific state out of
``EmissionModel`` and makes the scheduling logic easier to reason about.
"""

from collections import deque

from synthesizer import exceptions
from synthesizer.extensions.timers import tic, toc


class ModelQueue:
    """Runtime queue for executing an emission model dependency graph.

    The queue performs three related tasks for a single execution:

    - walk the model tree and collect the full executable closure
    - compile the dependency graph between models in that closure
    - manage ready-to-run models and eagerly delete expired unsaved emissions

    Args:
        root_model (EmissionModel):
            The root emission model being executed.

    Attributes:
        models (dict):
            Mapping from model label to the ``EmissionModel`` instance in the
            executable closure for this run.
        dependencies (dict):
            Mapping from model label to the labels of its direct upstream
            dependencies.
        dependents (dict):
            Mapping from model label to the labels of its direct downstream
            dependents.
        execution_rank (dict):
            Mapping from model label to its stable discovery order in the
            queue. This is used to keep execution deterministic when multiple
            models become ready at the same time.
        pending_dependencies (dict):
            Mapping from model label to the number of upstream dependencies
            that are still unresolved.
        lifetime (dict):
            Mapping from model label to the number of downstream consumers that
            still need the model's emission.
    """

    def __init__(self, root_model):
        """Initialise the queue for an emission model execution.

        Args:
            root_model (EmissionModel):
                The root model for the emission calculation being executed.

        Returns:
            None
        """
        # Keep a reference to the root model for error messages and checks.
        self._root_model = root_model

        # Build the executable model closure before compiling dependencies.
        tic("Collecting model queue tree")
        self.models = {}
        self._related_models = set()
        self._collect_model_tree(root_model)

        related_models = list(self._related_models)
        for model in related_models:
            if model.label not in self.models:
                self._collect_model_tree(model)
        toc("Collecting model queue tree")

        # Compile the dependency graph and runtime counters for this closure.
        tic("Compiling model queue")
        self.dependencies = {}
        self.dependents = {label: [] for label in self.models}
        self.execution_rank = {}

        for rank, (label, model) in enumerate(self.models.items()):
            self.execution_rank[label] = rank
            model_dependencies = self._get_model_dependencies(model)
            self.dependencies[label] = tuple(
                dependency.label for dependency in model_dependencies
            )

            for dependency in model_dependencies:
                self.dependents[dependency.label].append(label)

        # Initialise the pending dependency counts and model lifetimes.
        self.pending_dependencies = {
            label: len(dep_labels)
            for label, dep_labels in self.dependencies.items()
        }
        self.lifetime = {
            label: len(dep_labels)
            for label, dep_labels in self.dependents.items()
        }

        # Seed the ready queue using the compiled execution ordering.
        ready_labels = sorted(
            [
                label
                for label, count in self.pending_dependencies.items()
                if count == 0
            ],
            key=self.execution_rank.get,
        )
        self._queue = deque(ready_labels)
        self._processed = set()
        toc("Compiling model queue")

    def __len__(self):
        """Return the number of models currently ready to execute.

        Args:
            None

        Returns:
            int:
                The number of models currently waiting in the ready queue.
        """
        return len(self._queue)

    def pop(self):
        """Pop and return the next ready model from the queue.

        Args:
            None

        Returns:
            EmissionModel:
                The next model that is ready to be executed.
        """
        # Pop the next ready label and resolve it back to the model object.
        return self.models[self._queue.popleft()]

    def done(self, model, emissions, particle_emissions):
        """Mark a model as processed and update queue state.

        Args:
            model (EmissionModel):
                The model that has just finished execution.
            emissions (dict):
                The integrated emission dictionary to clean up.
            particle_emissions (dict):
                The particle emission dictionary to clean up.

        Returns:
            None
        """
        # Record completion so we can verify the whole graph ran later.
        label = model.label
        self._processed.add(label)

        # Unlock any direct dependents whose upstream work is now complete.
        newly_ready = []
        for dependent_label in self.dependents[label]:
            self.pending_dependencies[dependent_label] -= 1
            if self.pending_dependencies[dependent_label] == 0:
                newly_ready.append(dependent_label)

        # Keep the queue order deterministic for independent branches.
        for dependent_label in sorted(
            newly_ready, key=self.execution_rank.get
        ):
            self._queue.append(dependent_label)

        # Decrement dependency lifetimes because this model has consumed them.
        for dependency_label in self.dependencies[label]:
            self.lifetime[dependency_label] -= 1
            if self.lifetime[dependency_label] == 0:
                self._delete_expired_emission(
                    dependency_label,
                    emissions,
                    particle_emissions,
                )

        # Delete the model itself once nobody downstream needs it anymore.
        if self.lifetime[label] == 0:
            self._delete_expired_emission(
                label,
                emissions,
                particle_emissions,
            )

    def assert_finished(self):
        """Ensure the dependency graph was fully traversed.

        Args:
            None

        Returns:
            None

        Raises:
            exceptions.InconsistentArguments:
                Raised if some models were never processed, which indicates the
                dependency graph could not be fully resolved.
        """
        # Raise a clear error if some models could never be unlocked.
        if len(self._processed) != len(self.models):
            remaining = sorted(set(self.models) - self._processed)
            raise exceptions.InconsistentArguments(
                "Emission model dependency graph could not be fully "
                f"resolved. Remaining models: {remaining}"
            )

    def _collect_model_tree(self, model):
        """Walk the model tree reachable through true dependencies.

        Args:
            model (EmissionModel):
                The model currently being visited.

        Returns:
            None
        """
        # Store this model while enforcing unique labels within the closure.
        if model.label not in self.models:
            self.models[model.label] = model
        elif self.models[model.label] is model:
            # Stop immediately when this exact model object was already seen.
            return
        else:
            # Reuse an existing node when another model points at the same
            # logical label. This mirrors the existing related-model handling,
            # where label identity defines the node in the execution graph.
            if len(model.masks) == 0:
                return

            # Mirror the existing masked-model behaviour by extending the
            # label when the collision is caused by a masked variant.
            for mask_dict in model.masks:
                model.label += (
                    f"_{mask_dict['attr']}"
                    f"{mask_dict['op']}"
                    f"{mask_dict['thresh']}"
                ).replace(" ", "-")

            self.models[model.label] = model

        # Walk all direct model dependencies for this model.
        for dependency in self._get_model_dependencies(model):
            self._collect_model_tree(dependency)

        # Record related models so they can be added as extra roots later.
        self._related_models.update(model.related_models)

    def _get_model_dependencies(self, model):
        """Return the direct in-graph model dependencies for a model.

        Args:
            model (EmissionModel):
                The model whose dependencies should be inspected.

        Returns:
            list[EmissionModel]:
                The direct dependencies represented by ``EmissionModel``
                instances in the execution graph.
        """
        # Define a local container for model dependencies.
        model_dependencies = []

        # Transformations depend on the model they are applied to.
        if model._is_transforming and self._is_model_instance(model.apply_to):
            model_dependencies.append(model.apply_to)

        # Combinations depend on every contributing child model.
        if model._is_combining:
            for child in model.combine:
                if self._is_model_instance(child):
                    model_dependencies.append(child)

        # Generators can depend on intrinsic, attenuated, and scaler models.
        if model._is_generating:
            generator_dependencies = ()

            if hasattr(model.generator, "_intrinsic"):
                generator_dependencies += (model.generator._intrinsic,)
            if hasattr(model.generator, "_attenuated"):
                generator_dependencies += (model.generator._attenuated,)
            if hasattr(model.generator, "_scaler"):
                generator_dependencies += (model.generator._scaler,)

            for dependency in generator_dependencies:
                if self._is_model_instance(dependency):
                    model_dependencies.append(dependency)

        # Scaling by another model object is also a true dependency because the
        # downstream model reads that emission during execution.
        for scaler in model.scale_by:
            if self._is_model_instance(scaler):
                model_dependencies.append(scaler)

        # Scaling by another model label is also a true dependency if that
        # label resolves within the compiled closure.
        for scaler in model.scale_by:
            if isinstance(scaler, str) and scaler in self.models:
                model_dependencies.append(self.models[scaler])

        # Deduplicate while preserving discovery order.
        ordered_model_dependencies = []
        seen_model_labels = set()

        for dependency in model_dependencies:
            if dependency.label in seen_model_labels:
                continue
            seen_model_labels.add(dependency.label)
            ordered_model_dependencies.append(dependency)

        return ordered_model_dependencies

    def _delete_expired_emission(
        self,
        label,
        emissions,
        particle_emissions,
    ):
        """Delete an unsaved emission once its lifetime has expired.

        Args:
            label (str):
                The label to delete.
            emissions (dict):
                The integrated emissions dictionary.
            particle_emissions (dict):
                The particle emissions dictionary.

        Returns:
            None
        """
        # Skip labels that are explicitly requested to survive execution.
        if self.models[label].save:
            return

        # Remove the integrated emission if it is still present.
        if label in emissions:
            del emissions[label]

        # Remove the particle emission when that representation exists.
        if label in particle_emissions:
            del particle_emissions[label]

    @staticmethod
    def _is_model_instance(obj):
        """Return whether an object is an ``EmissionModel`` instance.

        Args:
            obj (Any):
                The object to test.

        Returns:
            bool:
                ``True`` if ``obj`` is an ``EmissionModel`` instance,
                otherwise ``False``.
        """
        # Import lazily so this module does not create a circular import.
        from synthesizer.emission_models.base_model import EmissionModel

        # Return whether this object participates in the model graph.
        return isinstance(obj, EmissionModel)
