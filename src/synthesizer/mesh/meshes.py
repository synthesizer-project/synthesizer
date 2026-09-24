"""Meshes of component attributes.

This module provides the ``Meshes`` container, which holds any number of
particle attributes deposited onto a single shared mesh geometry, and the
``MeshableComponent`` mixin that gives particle components (``Gas`` and
``Stars``) a ``get_meshes`` method storing their meshes as ``meshes``.

The mesh is either a uniform grid of cubic cells or an adaptively refined
octree whose root level is that uniform grid. In both cases the domain is set
by the particles themselves: it spans the full extent of the particles'
support (their kernels, or their clouds for cloud-in-cell deposition), so no
particle ever loses any of its value off the edge of the mesh.

Deposition is conservative: every particle's weights are normalised to sum to
one, so the total of an extensive field over all cells equals the total over
the particles to floating-point precision. Any number of attributes is
deposited in a single pass over the particles by the C++ backend.

Two kinds of field are supported:

- Extensive fields (e.g. masses) are stored as cell totals.
- Intensive fields (e.g. metallicities, ages) are stored as weighted means,
  computed from the deposited ``weight * value`` and ``weight`` totals. The
  weight defaults to the component's mass attribute.

Example usage:

    gas.get_meshes(
        resolution=0.5 * kpc,
        extensive=("masses", "dust_masses"),
        intensive=("metallicities",),
    )
    gas.meshes["masses"]      # (nx, ny, nz) cell totals in Msun
    gas.meshes.metallicities  # mass weighted mean metallicity per cell
"""

import numpy as np
from unyt import unyt_array, unyt_quantity

from synthesizer import exceptions
from synthesizer.extensions.mesh import (
    build_refined_mesh,
    cell_index,
    deposit_to_mesh,
)
from synthesizer.utils.precision import resolve_out_dtype


def _raw(arr, units=None):
    """Return the raw values of a (possibly unyt) array.

    Args:
        arr (array-like):
            The array to strip.
        units (unyt.Unit, optional):
            Units to convert to before stripping. Only used if ``arr`` carries
            units.

    Returns:
        np.ndarray:
            The raw values.
    """
    if isinstance(arr, unyt_array):
        return arr.to_value(units) if units is not None else arr.ndview
    return np.asarray(arr)


def _float_dtype(*arrays):
    """Return the common float dtype (float32 or float64) of some arrays.

    Args:
        *arrays (np.ndarray):
            The arrays to consider.

    Returns:
        np.dtype:
            float32 if every array is float32, otherwise float64.
    """
    if all(arr.dtype == np.float32 for arr in arrays):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _as_names(names, argument):
    """Normalise a string or sequence of strings to a tuple of strings.

    Args:
        names (str or sequence of str):
            The attribute name(s).
        argument (str):
            The argument name, used in error messages.

    Returns:
        tuple of str:
            The names.
    """
    if isinstance(names, str):
        return (names,)
    names = tuple(names)
    if not all(isinstance(name, str) for name in names):
        raise exceptions.InconsistentArguments(
            f"{argument} must be a string or a sequence of strings."
        )
    return names


class Meshes:
    """A set of attribute meshes sharing one geometry.

    Fields are accessed like a dictionary (``meshes["masses"]``) or as
    attributes (``meshes.masses``). Each field is a contiguous ``unyt_array``
    with one value per cell: shape ``(nx, ny, nz)`` for a uniform mesh, or
    ``(nleaf,)`` for a refined mesh (leaves in Morton order within each root
    cell, root cells in C order).

    Code that works on any mesh should use the flat cell ordering: ``flat``
    returns a field as a 1D array in the same order as ``cell_index`` and the
    flattened ``cell_centres``, ``cell_widths`` and ``cell_volumes`` (a free
    view for a uniform mesh, the field itself for a refined one).

    Meshes should be created with ``Meshes.from_particles`` (or a component's
    ``get_meshes`` method) rather than directly.

    Attributes:
        origin (unyt_array):
            The lower corner of the mesh domain.
        resolution (unyt_quantity):
            The width of the root (coarsest) cells.
        dims (tuple of int):
            The number of root cells along each axis.
        max_depth (int):
            The maximum refinement depth allowed when the mesh was built.
        refine_attr (str):
            The attribute used to drive refinement, or None.
        refine_threshold (unyt_quantity or float):
            The per-cell total of ``refine_attr`` above which cells were
            refined, or None.
        field_info (dict):
            Metadata for each field: ``kind`` ("extensive" or "intensive"),
            ``weight`` (the weighting attribute for intensive fields),
            ``as_points`` and ``kernel``.
    """

    def __init__(
        self,
        origin,
        resolution,
        dims,
        tree=None,
        max_depth=0,
        refine_attr=None,
        refine_threshold=None,
    ):
        """Initialise an empty set of meshes on a geometry.

        Args:
            origin (unyt_array):
                The lower corner of the mesh domain.
            resolution (unyt_quantity):
                The width of the root cells.
            dims (tuple of int):
                The number of root cells along each axis.
            tree (tuple, optional):
                The refinement tree ``(first_child, node_depth, node_ijk,
                node_leaf, nleaf)`` returned by the backend, or None for a
                uniform mesh.
            max_depth (int):
                The maximum refinement depth used to build the mesh.
            refine_attr (str, optional):
                The attribute used to drive refinement.
            refine_threshold (unyt_quantity or float, optional):
                The refinement threshold.
        """
        self.origin = origin
        self.resolution = resolution
        self.dims = tuple(int(d) for d in dims)
        self._tree = tree
        self.max_depth = max_depth
        self.refine_attr = refine_attr
        self.refine_threshold = refine_threshold

        # The fields themselves and their metadata
        self._fields = {}
        self.field_info = {}

    @classmethod
    def from_particles(
        cls,
        particles,
        resolution,
        extensive=(),
        intensive=(),
        weights=None,
        kernel=None,
        as_points=False,
        mask=None,
        like=None,
        refine_attr=None,
        refine_threshold=None,
        max_depth=10,
        nthreads=1,
        out_dtype=None,
    ):
        """Deposit particle attributes onto a new set of meshes.

        The mesh domain spans the support of the (masked) particles: each
        particle's kernel for smoothed deposition, or a half cell either side
        of each particle for cloud-in-cell deposition. The number of root
        cells along each axis is the smallest that covers that extent at the
        requested resolution, with the domain centred on the particles.

        Args:
            particles (Particles):
                The particle component to mesh.
            resolution (unyt_quantity):
                The width of the (root) cells. Ignored when ``like`` is given.
            extensive (str or sequence of str):
                Attributes deposited as per-cell totals (e.g. "masses").
            intensive (str or sequence of str):
                Attributes deposited as per-cell weighted means (e.g.
                "metallicities").
            weights (str or dict, optional):
                The weighting attribute for intensive fields. A string applies
                to every intensive field; a dict maps field names to weighting
                attributes (fields missing from the dict use the default).
                Defaults to the component's mass attribute (``masses`` for
                gas, ``initial_masses`` for stars).
            kernel (Kernel or str, optional):
                The SPH kernel used for smoothed deposition. Defaults to
                "sph_anarchy".
            as_points (bool):
                Deposit particles with cloud-in-cell (each particle a cube the
                width of the cell containing it) instead of their kernels.
            mask (array-like of bool, optional):
                Only mesh the particles where the mask is True. The domain is
                then set by the masked particles.
            like (Meshes, optional):
                Deposit onto the geometry (including any refinement) of an
                existing set of meshes instead of building a new one. The
                particles' support must lie inside that domain.
            refine_attr (str, optional):
                Attribute driving adaptive refinement. If given, cells are
                split while their total of this attribute exceeds
                ``refine_threshold``.
            refine_threshold (unyt_quantity or float, optional):
                The per-cell total of ``refine_attr`` above which cells are
                refined. Required with ``refine_attr``.
            max_depth (int):
                The maximum refinement depth.
            nthreads (int):
                The number of threads to use.
            out_dtype (dtype-like, optional):
                Floating-point dtype of the fields. Defaults to the global
                Synthesizer output dtype.

        Returns:
            Meshes:
                The new meshes.
        """
        extensive = _as_names(extensive, "extensive")
        intensive = _as_names(intensive, "intensive")
        if len(extensive) + len(intensive) == 0:
            raise exceptions.InconsistentArguments(
                "At least one extensive or intensive attribute is required."
            )
        clash = set(extensive) & set(intensive)
        if clash:
            raise exceptions.InconsistentArguments(
                f"Attributes cannot be both extensive and intensive: {clash}"
            )
        if refine_attr is not None and refine_threshold is None:
            raise exceptions.InconsistentArguments(
                "refine_threshold is required when refine_attr is given."
            )
        if like is not None and refine_attr is not None:
            raise exceptions.InconsistentArguments(
                "Cannot refine when depositing onto an existing geometry "
                "(like=...); the existing refinement is reused."
            )

        # Resolve the kernel name used by the backend.
        if kernel is None:
            kernel_name = "sph_anarchy"
        elif isinstance(kernel, str):
            kernel_name = kernel
        else:
            kernel_name = kernel.name

        # Select the particles to mesh.
        if particles.coordinates is None:
            raise exceptions.InconsistentArguments(
                f"{particles.name} object is missing coordinates!"
            )
        if mask is None:
            mask = np.ones(particles.nparticles, dtype=bool)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (particles.nparticles,):
            raise exceptions.InconsistentArguments(
                "The mask must have one entry per particle."
            )
        if not mask.any():
            raise exceptions.InconsistentArguments(
                "Cannot build meshes from zero particles."
            )

        # Work in the particles' coordinate units throughout.
        spatial_units = particles.coordinates.units
        pos = _raw(particles.coordinates)[mask]
        if as_points:
            sml = None
            part_dtype = _float_dtype(pos)
        else:
            if particles.smoothing_lengths is None:
                raise exceptions.InconsistentArguments(
                    f"{particles.name} object is missing smoothing lengths! "
                    "Use as_points=True to deposit with cloud-in-cell."
                )
            sml = _raw(particles.smoothing_lengths, spatial_units)[mask]
            part_dtype = _float_dtype(pos, sml)
            sml = np.ascontiguousarray(sml, dtype=part_dtype)
        pos = np.ascontiguousarray(pos, dtype=part_dtype)

        # Build the list of arrays to deposit. Extensive fields are deposited
        # directly; intensive fields as weight * value alongside each distinct
        # weight so the means can be formed afterwards.
        # Each component class declares its natural weight (see
        # MeshableComponent); fall back to masses for anything else.
        default_weight = getattr(particles, "_mesh_default_weight", "masses")
        if weights is None:
            weights = {}
        elif isinstance(weights, str):
            weights = dict.fromkeys(intensive, weights)
        field_weights = {
            name: weights.get(name, default_weight) for name in intensive
        }

        def get_values(name):
            values = getattr(particles, name, None)
            if values is None:
                raise exceptions.InconsistentArguments(
                    f"{particles.name} object is missing {name}!"
                )
            if np.shape(values) != (particles.nparticles,):
                raise exceptions.InconsistentArguments(
                    f"{particles.name} attribute {name} must have one value "
                    "per particle."
                )
            return values

        deposits = []
        slots = {}
        for name in extensive:
            slots[name] = len(deposits)
            deposits.append(_raw(get_values(name))[mask])
        for name in intensive:
            slots[("wx", name)] = len(deposits)
            weight = _raw(get_values(field_weights[name]))[mask]
            deposits.append(weight * _raw(get_values(name))[mask])
        for weight_name in sorted(set(field_weights.values())):
            slots[("w", weight_name)] = len(deposits)
            deposits.append(_raw(get_values(weight_name))[mask])
        value_dtype = _float_dtype(*deposits)
        deposits = tuple(
            np.ascontiguousarray(arr, dtype=value_dtype) for arr in deposits
        )

        # Set up the geometry, either from the particles or from like.
        lo, hi = cls._support_extent(pos, sml, resolution, spatial_units, like)
        if like is None:
            res = float(resolution.to_value(spatial_units))
            if not res > 0:
                raise exceptions.InconsistentArguments(
                    "resolution must be positive."
                )
            dims = np.maximum(np.ceil((hi - lo) / res).astype(np.int64), 1)
            origin = lo - 0.5 * (dims * res - (hi - lo))
            # Report the geometry in the units the resolution was given in.
            meshes = cls(
                origin=unyt_array(origin, spatial_units).to(resolution.units),
                resolution=unyt_quantity(res, spatial_units).to(
                    resolution.units
                ),
                dims=dims,
                max_depth=max_depth,
                refine_attr=refine_attr,
                refine_threshold=refine_threshold,
            )
        else:
            meshes = cls(
                origin=like.origin,
                resolution=like.resolution,
                dims=like.dims,
                tree=like._tree,
                max_depth=like.max_depth,
                refine_attr=like.refine_attr,
                refine_threshold=like.refine_threshold,
            )
            origin = like.origin.to_value(spatial_units)
            res = float(like.resolution.to_value(spatial_units))
            dims = np.array(like.dims)
            extent_hi = origin + dims * res
            tol = 1e-9 * res
            if np.any(lo < origin - tol) or np.any(hi > extent_hi + tol):
                raise exceptions.InconsistentArguments(
                    "The particles' support extends outside the domain of "
                    "the meshes passed as like."
                )

        origin = tuple(float(x) for x in origin)
        dims = tuple(int(d) for d in dims)

        # Build the refinement tree if requested.
        if refine_attr is not None:
            refine_values = get_values(refine_attr)
            if isinstance(refine_threshold, unyt_quantity):
                if not isinstance(refine_values, unyt_array):
                    raise exceptions.InconsistentArguments(
                        f"refine_threshold has units but {refine_attr} "
                        "does not."
                    )
                threshold = float(
                    refine_threshold.to_value(refine_values.units)
                )
            else:
                threshold = float(refine_threshold)
            refine_values = _raw(refine_values)[mask]
            meshes._tree = build_refined_mesh(
                pos,
                sml,
                np.ascontiguousarray(
                    refine_values, dtype=_float_dtype(refine_values)
                ),
                origin,
                res,
                dims,
                kernel_name,
                int(as_points),
                threshold,
                int(max_depth),
                int(nthreads),
            )

        # Deposit everything in one pass over the particles.
        raw = deposit_to_mesh(
            pos,
            sml,
            deposits,
            origin,
            res,
            dims,
            kernel_name,
            int(as_points),
            meshes._tree,
            int(nthreads),
            resolve_out_dtype(out_dtype),
        )
        raw = raw.reshape((len(deposits),) + meshes.shape)

        # Attach units and form the intensive means.
        for name in extensive:
            meshes._set_field(
                name,
                raw[slots[name]],
                getattr(get_values(name), "units", None),
                {
                    "kind": "extensive",
                    "weight": None,
                    "as_points": bool(as_points),
                    "kernel": None if as_points else kernel_name,
                },
            )
        for name in intensive:
            weight_sum = raw[slots[("w", field_weights[name])]]
            mean = np.zeros_like(weight_sum)
            np.divide(
                raw[slots[("wx", name)]],
                weight_sum,
                out=mean,
                where=weight_sum > 0,
            )
            meshes._set_field(
                name,
                mean,
                getattr(get_values(name), "units", None),
                {
                    "kind": "intensive",
                    "weight": field_weights[name],
                    "as_points": bool(as_points),
                    "kernel": None if as_points else kernel_name,
                },
            )

        return meshes

    @classmethod
    def from_parametric(cls, *args, **kwargs):
        """Create meshes from a parametric component.

        TODO: Parametric morphologies are currently two dimensional, so a
        three dimensional density profile is needed before parametric
        components can be meshed.

        Raises:
            UnimplementedFunctionality:
                Always, until parametric meshing is implemented.
        """
        raise exceptions.UnimplementedFunctionality(
            "Meshes cannot yet be created from parametric components."
        )

    @staticmethod
    def _support_extent(pos, sml, resolution, spatial_units, like):
        """Compute the extent of the particles' support.

        Args:
            pos (np.ndarray):
                The particle positions.
            sml (np.ndarray or None):
                The smoothing lengths, or None for cloud-in-cell.
            resolution (unyt_quantity):
                The requested root cell width (used for cloud-in-cell when
                ``like`` is None).
            spatial_units (unyt.Unit):
                The units of ``pos``.
            like (Meshes or None):
                The meshes supplying the geometry, if any.

        Returns:
            tuple of np.ndarray:
                The lower and upper corners of the support.
        """
        if sml is not None:
            lo = (pos - sml[:, None]).min(axis=0)
            hi = (pos + sml[:, None]).max(axis=0)
        else:
            # A cloud is at most one root cell wide.
            res = like.resolution if like is not None else resolution
            half = 0.5 * float(res.to_value(spatial_units))
            lo = pos.min(axis=0) - half
            hi = pos.max(axis=0) + half
        return lo.astype(np.float64), hi.astype(np.float64)

    def add(self, particles, extensive=(), intensive=(), **kwargs):
        """Deposit more attributes onto this geometry.

        Any existing field with the same name is replaced.

        Args:
            particles (Particles):
                The particle component to deposit.
            extensive (str or sequence of str):
                Attributes deposited as per-cell totals.
            intensive (str or sequence of str):
                Attributes deposited as per-cell weighted means.
            **kwargs:
                Any other argument accepted by ``from_particles`` except
                ``resolution``, ``like`` and the refinement arguments.

        Returns:
            Meshes:
                This object, for chaining.
        """
        other = Meshes.from_particles(
            particles,
            self.resolution,
            extensive=extensive,
            intensive=intensive,
            like=self,
            **kwargs,
        )
        self._fields.update(other._fields)
        self.field_info.update(other.field_info)
        return self

    def _set_field(self, name, values, units, info):
        """Store a field and its metadata.

        Args:
            name (str):
                The field name.
            values (np.ndarray):
                The per-cell values.
            units (unyt.Unit or None):
                The field units.
            info (dict):
                The field metadata.
        """
        if units is not None:
            values = unyt_array(values, units, bypass_validation=True)
        self._fields[name] = values
        self.field_info[name] = info

    @property
    def refined(self):
        """Whether the mesh is adaptively refined.

        Returns:
            bool:
                True if the mesh has a refinement tree.
        """
        return self._tree is not None

    @property
    def ncells(self):
        """The number of (leaf) cells.

        Returns:
            int:
                The number of cells holding field values.
        """
        if self.refined:
            return int(self._tree[4])
        return int(np.prod(self.dims))

    @property
    def shape(self):
        """The shape of each field.

        Returns:
            tuple of int:
                ``(nx, ny, nz)`` for a uniform mesh, ``(nleaf,)`` if refined.
        """
        return (self.ncells,) if self.refined else self.dims

    @property
    def extent(self):
        """The upper corner of the mesh domain.

        Returns:
            unyt_array:
                The upper corner of the domain.
        """
        return self.origin + np.array(self.dims) * self.resolution

    def _leaf_nodes(self):
        """Get the tree node of each leaf, in leaf order.

        Returns:
            np.ndarray:
                The node index of each leaf.
        """
        node_leaf = self._tree[3]
        nodes = np.flatnonzero(node_leaf >= 0)
        return nodes[np.argsort(node_leaf[nodes])]

    @property
    def cell_depths(self):
        """The refinement depth of each cell.

        Returns:
            np.ndarray:
                The depth of each cell (all zero for a uniform mesh).
        """
        if not self.refined:
            return np.zeros(self.shape, dtype=np.int32)
        return self._tree[1][self._leaf_nodes()]

    @property
    def cell_widths(self):
        """The width of each cell.

        Returns:
            unyt_array:
                The width of each cell with the shape of a field.
        """
        if not self.refined:
            return np.full(self.shape, self.resolution.value) * (
                self.resolution.units
            )
        return self.resolution / 2.0**self.cell_depths

    @property
    def cell_volumes(self):
        """The volume of each cell.

        Returns:
            unyt_array:
                The volume of each cell with the shape of a field.
        """
        return self.cell_widths**3

    @property
    def cell_centres(self):
        """The centre of each cell.

        Returns:
            unyt_array:
                Cell centres with shape ``(nx, ny, nz, 3)`` for a uniform mesh
                or ``(nleaf, 3)`` if refined.
        """
        res = self.resolution.value
        origin = self.origin.to_value(self.resolution.units)
        if not self.refined:
            axes = [
                origin[a] + (np.arange(n) + 0.5) * res
                for a, n in enumerate(self.dims)
            ]
            centres = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
        else:
            nodes = self._leaf_nodes()
            widths = res / 2.0 ** self._tree[1][nodes]
            centres = origin + (self._tree[2][nodes] + 0.5) * widths[:, None]
        return unyt_array(centres, self.resolution.units)

    def flat(self, name):
        """Get a field as a flat array with one value per cell.

        The ordering matches ``cell_index`` and the flattened cell geometry
        properties, so code looping over cells works identically for uniform
        and refined meshes.

        Args:
            name (str):
                The field name.

        Returns:
            unyt_array:
                The field with shape ``(ncells,)``. For a uniform mesh this is
                a view of the ``(nx, ny, nz)`` field, not a copy.
        """
        return self._fields[name].reshape(-1)

    def cell_index(self, points, nthreads=1):
        """Find the cell containing each of a set of points.

        For a uniform mesh this is index arithmetic; for a refined mesh the
        refinement tree is descended from the root cell, so no separate
        search structure is needed.

        Args:
            points (unyt_array):
                Positions with shape ``(N, 3)``.
            nthreads (int):
                The number of threads to use.

        Returns:
            np.ndarray:
                The flat cell index of each point (see ``flat``), or -1 for
                points outside the mesh domain.
        """
        if not isinstance(points, unyt_array):
            raise exceptions.MissingUnits("points must have units.")
        units = self.resolution.units
        raw = np.asarray(points.to_value(units))
        if raw.ndim != 2 or raw.shape[1] != 3:
            raise exceptions.InconsistentArguments(
                "points must have shape (N, 3)."
            )
        return cell_index(
            np.ascontiguousarray(raw, dtype=_float_dtype(raw)),
            tuple(float(x) for x in self.origin.to_value(units)),
            float(self.resolution.value),
            self.dims,
            self._tree,
            int(nthreads),
        )

    def sample(self, points, names=None, nthreads=1):
        """Get field values in the cells containing a set of points.

        Fields are piecewise constant over cells, so each point takes the
        value of the cell containing it. Points outside the domain take a
        value of zero: the domain covers the support of every deposited
        particle, so nothing was deposited there. The containing cells are
        found once and reused for every requested field.

        Note that extensive fields are cell totals; for a density at the
        points use ``meshes.density(name).reshape(-1)[meshes.cell_index(
        points)]``.

        Args:
            points (unyt_array):
                Positions with shape ``(N, 3)``.
            names (str or sequence of str, optional):
                The field(s) to sample. Defaults to every field.
            nthreads (int):
                The number of threads to use for the cell lookup.

        Returns:
            unyt_array or dict:
                The sampled values for a single field name, otherwise a dict
                mapping each field name to its sampled values.
        """
        single = isinstance(names, str)
        names = (
            tuple(self._fields) if names is None else _as_names(names, "names")
        )
        idx = self.cell_index(points, nthreads=nthreads)
        outside = idx < 0
        samples = {}
        for name in names:
            values = self.flat(name)[np.where(outside, 0, idx)]
            values[outside] = 0
            samples[name] = values
        return samples[names[0]] if single else samples

    def density(self, name):
        """Get an extensive field divided by the cell volumes.

        Args:
            name (str):
                The name of an extensive field.

        Returns:
            unyt_array:
                The field per unit volume.
        """
        if self.field_info[name]["kind"] != "extensive":
            raise exceptions.InconsistentArguments(
                f"{name} is intensive; only extensive fields have densities."
            )
        return self[name] / self.cell_volumes

    def __getitem__(self, name):
        """Get a field by name.

        Args:
            name (str):
                The field name.

        Returns:
            unyt_array:
                The field.
        """
        return self._fields[name]

    def __getattr__(self, name):
        """Get a field as an attribute.

        This is only called when normal attribute lookup fails, so real
        attributes always take precedence over fields of the same name.

        Args:
            name (str):
                The field name.

        Returns:
            unyt_array:
                The field.
        """
        # Avoid recursion before _fields exists (e.g. during copying).
        fields = self.__dict__.get("_fields", {})
        if name in fields:
            return fields[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute or field "
            f"'{name}'"
        )

    def __contains__(self, name):
        """Check whether a field exists."""
        return name in self._fields

    def __iter__(self):
        """Iterate over the field names."""
        return iter(self._fields)

    def __len__(self):
        """Get the number of fields."""
        return len(self._fields)

    def keys(self):
        """Get the field names."""
        return self._fields.keys()

    def values(self):
        """Get the fields."""
        return self._fields.values()

    def items(self):
        """Get the (name, field) pairs."""
        return self._fields.items()

    def __repr__(self):
        """Get a short description of the meshes.

        Returns:
            str:
                The description.
        """
        kind = f"refined, {self.ncells} leaves" if self.refined else "uniform"
        return (
            f"Meshes({kind}, dims={self.dims}, resolution={self.resolution}, "
            f"fields={list(self._fields)})"
        )


class MeshableComponent:
    """Mixin giving a particle component the ability to build meshes.

    This is not a component in its own right; it is inherited alongside
    ``Particles`` and a component base class (e.g. ``Gas(Particles,
    Component, MeshableComponent)``) to add ``get_meshes``. Classes using it
    must set ``self.meshes = None`` in their ``__init__``.

    Attributes:
        _mesh_default_weight (str):
            The attribute used to weight intensive mesh fields when no
            weight is given. Subclasses override this (e.g. stars use
            ``"initial_masses"``).
    """

    # The attribute used to weight intensive fields by default.
    _mesh_default_weight = "masses"

    def get_meshes(self, resolution, extensive=(), intensive=(), **kwargs):
        """Deposit attributes onto meshes and store them on the component.

        This is a thin wrapper around ``Meshes.from_particles`` (see there
        for the full set of arguments) that also attaches the result as
        ``self.meshes``.

        Args:
            resolution (unyt_quantity):
                The width of the (root) mesh cells.
            extensive (str or sequence of str):
                Attributes deposited as per-cell totals (e.g. "masses").
            intensive (str or sequence of str):
                Attributes deposited as per-cell weighted means, weighted by
                ``_mesh_default_weight`` unless ``weights`` is given.
            **kwargs:
                Any other argument accepted by ``Meshes.from_particles``.

        Returns:
            Meshes:
                The new meshes (also stored as ``self.meshes``).
        """
        self.meshes = Meshes.from_particles(
            self,
            resolution,
            extensive=extensive,
            intensive=intensive,
            **kwargs,
        )
        return self.meshes
