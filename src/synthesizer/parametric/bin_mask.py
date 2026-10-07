"""A module defining masks for binned parametric populations.

A parametric population is a set of bins (e.g. in age and metallicity), each
holding a mass spread uniformly over its extent. A mask such as "age < 10 Myr"
can therefore cut through a bin. Rather than including or excluding whole
bins, a BinMask records the allowed interval along each axis and converts it
into the fraction of each bin that satisfies the conditions. Since the mass is
uniform within a bin, scaling each bin's mass by this fraction is exact.

Conditions on the same axis intersect (e.g. 1 Myr < age < 10 Myr), so they
are combined as intervals rather than as products of fractions. Conditions on
different axes multiply, since a bin's mass is uniform over its N dimensional
extent.

Zero width bins (points) have every condition evaluated exactly. For a bin of
finite width none of its (uniformly spread) mass sits exactly at a single
value, so "==" passes none of it and "!=" passes all of it.

Example usage:

    mask = stars.get_mask("log10ages", 7, "<")
    mask = stars.get_mask("metallicities", 0.01, ">", mask=mask)
    fractions = mask.get_fractions(stars)
"""

import operator

import numpy as np
from unyt import unyt_quantity

from synthesizer import exceptions

# The comparison operators a mask can apply
OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "!=": operator.ne,
}


class BinMask:
    """A mask for binned parametric populations.

    Attributes:
        axis_conditions (dict):
            The conditions applied along each bin axis, as a list of
            (operator string, threshold) pairs keyed by axis name. Thresholds
            are in the same (linear) units as the bin edges.
        include (bool/np.ndarray of bool):
            Whether each population is included at all. This is set by
            conditions on population level quantities (e.g. a fixed model
            parameter, or a per population parameter with one value per
            population) rather than on a bin axis.
    """

    def __init__(self):
        """Initialise an empty mask, which includes everything."""
        self.axis_conditions = {}
        self.include = True

    def copy(self):
        """Return a copy of this mask.

        Returns:
            BinMask:
                The copied mask.
        """
        new = BinMask()
        new.axis_conditions = {
            axis: list(conds) for axis, conds in self.axis_conditions.items()
        }
        new.include = np.copy(self.include)
        return new

    def get_key(self):
        """Get a hashable key identifying this mask's conditions.

        Two masks with the same key select exactly the same mass, so their
        grid weights can be shared.

        Returns:
            tuple:
                The key.
        """
        return (
            tuple(np.atleast_1d(self.include).tolist()),
            tuple(
                (axis, tuple(sorted(conds)))
                for axis, conds in sorted(self.axis_conditions.items())
            ),
        )

    def get_population_mask(self, npop):
        """Get which populations the mask includes.

        Conditions on the bin axes act on the bins when the emission is
        extracted, so they don't exclude whole populations.

        Args:
            npop (int):
                The number of populations.

        Returns:
            np.ndarray of bool:
                Whether each population is included, or None if they all are.
        """
        include = np.broadcast_to(self.include, (npop,))
        if np.all(include):
            return None
        return np.array(include)

    def add_axis_condition(self, axis, op, thresh):
        """Add a condition along a bin axis.

        Args:
            axis (str):
                The name of the bin axis (e.g. "ages").
            op (str):
                The comparison operator.
            thresh (float):
                The threshold in the units of the bin edges.
        """
        if op not in OPERATORS:
            raise exceptions.InconsistentArguments(
                "Masking operation must be '<', '>', '<=', '>=', '==', or "
                f"'!=', not {op}"
            )
        self.axis_conditions.setdefault(axis, []).append((op, thresh))

    def add_population_condition(self, value, op, thresh):
        """Add a condition on a population level quantity.

        Args:
            value (float/np.ndarray of float):
                The population's value, or one value per population.
            op (str):
                The comparison operator.
            thresh (float):
                The threshold.
        """
        if op not in OPERATORS:
            raise exceptions.InconsistentArguments(
                "Masking operation must be '<', '>', '<=', '>=', '==', or "
                f"'!=', not {op}"
            )
        self.include = np.logical_and(
            self.include, OPERATORS[op](np.asarray(value), thresh)
        )

    def _get_allowed_interval(self, axis):
        """Get the interval a finite bin's mass must lie in along an axis.

        This is the intersection of the half lines defined by the conditions.
        Strict and non-strict inequalities only differ on a boundary of zero
        width so they are treated the same. For the same reason none of a
        finite bin's mass is exactly equal to a value, so "==" allows nothing
        and "!=" allows everything.

        Args:
            axis (str):
                The name of the bin axis.

        Returns:
            tuple of float:
                The lower and upper bounds of the allowed interval.
        """
        allowed_lo = -np.inf
        allowed_hi = np.inf
        for op, thresh in self.axis_conditions.get(axis, []):
            if op in ("<", "<="):
                allowed_hi = min(allowed_hi, thresh)
            elif op in (">", ">="):
                allowed_lo = max(allowed_lo, thresh)
            elif op == "==":
                allowed_hi = -np.inf
        return allowed_lo, allowed_hi

    def _get_axis_fractions(self, axis, edges):
        """Get the fraction of each bin along an axis passing the conditions.

        Args:
            axis (str):
                The name of the bin axis.
            edges (np.ndarray):
                The bin edges along this axis (linear units).

        Returns:
            np.ndarray:
                The fraction of each bin satisfying every condition.
        """
        conditions = self.axis_conditions.get(axis, [])
        lo = edges[:-1]
        hi = edges[1:]
        if len(conditions) == 0:
            return np.ones(lo.size)

        # Zero width bins (points) are evaluated exactly
        points = hi == lo
        point_pass = np.ones(lo.size, dtype=bool)
        for op, thresh in conditions:
            point_pass &= OPERATORS[op](lo, thresh)

        # Finite bins pass the part of them inside the allowed interval
        allowed_lo, allowed_hi = self._get_allowed_interval(axis)
        widths = np.where(points, 1.0, hi - lo)
        overlap = np.clip(
            np.minimum(hi, allowed_hi) - np.maximum(lo, allowed_lo),
            0.0,
            None,
        )
        return np.where(points, point_pass.astype(float), overlap / widths)

    def get_fractions(self, stars, edges=None, masses=None):
        """Get the fraction of each bin's mass passing the mask.

        Args:
            stars (parametric.Stars):
                The binned population the mask applies to.
            edges (dict):
                The bin edges keyed by axis name, defaults to the
                population's.
            masses (np.ndarray):
                The bin masses, defaults to the population's.

        Returns:
            np.ndarray:
                The fractions, broadcastable to the shape of the masses.
        """
        if edges is None:
            edges, masses = stars._bin_edges, stars.bin_masses

        # The axis conditions are the same for every population so the
        # fractions broadcast over the population axis, only the population
        # level conditions can differ between populations
        include = np.atleast_1d(self.include).astype(float)
        fractions = include.reshape((include.size,) + (1,) * (masses.ndim - 1))
        for iaxis, axis in enumerate(stars.bin_axes):
            axis_fracs = self._get_axis_fractions(axis, edges[axis])
            shape = [1] * fractions.ndim
            shape[iaxis + 1] = axis_fracs.size
            fractions = fractions * axis_fracs.reshape(shape)
        return fractions

    def get_masked_bins(self, stars, edges=None, masses=None):
        """Get the bins of a population with this mask applied.

        Scaling each bin's mass by its passing fraction gets the passing mass
        right, but that mass must also sit only in the passing part of the
        bin. The allowed interval along each axis is the same for every bin,
        so the shared edges are clipped to it, which keeps them valid shared
        edges. Each clipped bin then holds its passing mass spread uniformly
        over exactly the part of the bin that passes.

        Args:
            stars (parametric.Stars):
                The binned population the mask applies to.
            edges (dict):
                The bin edges keyed by axis name, defaults to the
                population's.
            masses (np.ndarray):
                The bin masses, defaults to the population's.

        Returns:
            tuple:
                The clipped bin edges keyed by axis name and the masked
                masses.
        """
        if edges is None:
            edges, masses = stars._bin_edges, stars.bin_masses
        clipped = {}
        for axis in stars.bin_axes:
            axis_edges = edges[axis]
            allowed_lo, allowed_hi = self._get_allowed_interval(axis)
            if allowed_lo <= allowed_hi:
                axis_edges = np.clip(axis_edges, allowed_lo, allowed_hi)
            clipped[axis] = axis_edges
        return clipped, masses * self.get_fractions(stars, edges, masses)


def to_axis_threshold(thresh, attr, axis_units):
    """Convert a mask threshold into the linear units of a bin axis.

    Args:
        thresh (float/unyt_quantity):
            The threshold as given to the mask.
        attr (str):
            The masked attribute (e.g. "log10ages" or "ages").
        axis_units (unyt.Unit or None):
            The units of the bin axis' edges (None if dimensionless).

    Returns:
        float:
            The threshold in the axis' linear units.
    """
    # A log10 attribute's threshold is the log10 of the linear value
    if attr.startswith("log10"):
        if isinstance(thresh, unyt_quantity):
            thresh = thresh.ndview
        return 10 ** float(thresh)

    # Otherwise convert any units to the edge units
    if isinstance(thresh, unyt_quantity):
        if axis_units is None:
            return float(thresh.ndview)
        return float(thresh.to(axis_units).ndview)
    if axis_units is not None:
        raise exceptions.InconsistentArguments(
            f"Masking attribute ({attr}) has units but threshold does not "
            f"({thresh})."
        )
    return float(thresh)


def union_edges(edges, other_edges):
    """Get the union of two sets of bin edges.

    A value repeated in either set (i.e. a zero width bin) is repeated in the
    union as many times as in the set repeating it most, so every bin of
    both sets can be rebuilt from bins of the union.

    Args:
        edges (np.ndarray of float):
            The first (non-decreasing) bin edges.
        other_edges (np.ndarray of float):
            The second (non-decreasing) bin edges.

    Returns:
        np.ndarray of float:
            The union of the edges.
    """
    values = np.union1d(edges, other_edges)

    def count(arr):
        return np.searchsorted(arr, values, "right") - np.searchsorted(
            arr, values, "left"
        )

    return np.repeat(
        values, np.maximum(np.maximum(count(edges), count(other_edges)), 1)
    )


def rebin_axis(masses, axis, edges, new_edges):
    """Rebin masses along one axis onto finer edges.

    The new edges must contain every one of the old edges (e.g. the union
    of two sets of edges), so every new bin lies within a single old bin. A
    finite bin's mass is uniform within it, so each new finite bin takes the
    fraction of its old bin's mass given by their widths. A zero width bin's
    mass moves to the zero width bin at the same value.

    Args:
        masses (np.ndarray):
            The masses.
        axis (int):
            The axis of masses to rebin.
        edges (np.ndarray of float):
            The current bin edges along that axis.
        new_edges (np.ndarray of float):
            The new bin edges along that axis.

    Returns:
        np.ndarray:
            The rebinned masses.
    """
    lo, hi = edges[:-1], edges[1:]
    new_lo, new_hi = new_edges[:-1], new_edges[1:]
    nbins = lo.size

    # Find the old bin holding each new finite bin (searching to the right
    # of any repeated edge lands in the finite bin after a zero width one)
    source = np.searchsorted(edges, new_lo, side="right") - 1
    inside = (source >= 0) & (source < nbins) & (new_hi > new_lo)
    source = np.clip(source, 0, nbins - 1)
    inside &= (hi[source] > lo[source]) & (new_hi <= hi[source])
    fraction = np.where(
        inside,
        (new_hi - new_lo) / np.where(inside, hi[source] - lo[source], 1.0),
        0.0,
    )

    # Zero width bins take the mass of the zero width bin at the same value
    points = np.flatnonzero(lo == hi)
    new_points = np.flatnonzero(new_lo == new_hi)
    matches = np.searchsorted(lo[points], new_lo[new_points])
    matches = np.clip(matches, 0, max(points.size - 1, 0))
    if points.size > 0:
        found = lo[points][matches] == new_lo[new_points]
        source[new_points[found]] = points[matches[found]]
        fraction[new_points[found]] = 1.0

    shape = [1] * masses.ndim
    shape[axis] = fraction.size
    return np.take(masses, source, axis=axis) * fraction.reshape(shape)
