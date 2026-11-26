"""Utilities for data loading methods.

These utilities are used through the load_data module as helpers for loading
data from different simulations sources.

Examples usage:

    lengths = np.array([10, 20, 30])
    begin, end = get_begin_end_pointers(lengths)
    print(begin)  # Output: [ 0 10 30]

    table = age_lookup_table(ages, redshift=0.5, delta_a=0.1)
    print(table)  # Output: (array([0.1, 0.2, 0.3]), array([10., 20., 30.]))

    ages = lookup_age(0.2, table[0], table[1])
"""

import math

import numpy as np

from synthesizer import exceptions


def cast_component_dtype(component, dtype):
    """Cast floating-point arrays in component input data to one dtype.

    Args:
        component (mapping or None):
            Component constructor arguments from any data source.
        dtype (numpy dtype):
            Target floating-point dtype.

    Returns:
        dict or None:
            Shallow copy with floating-point arrays cast, preserving units.
    """
    if component is None:
        return None

    return {
        key: value.astype(dtype, copy=False)
        if isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.floating)
        else value
        for key, value in component.items()
    }


def get_begin_end_pointers(length):
    """Find the beginning and ending indices from a length array.

    Args:
        length (np.ndarray of int):
            The number of particles in each galaxy.

    Returns:
        begin (np.ndarray of int): Beginning indices.
        end (np.ndarray of int): Ending indices.
    """
    begin = np.zeros(len(length), dtype=np.int64)
    end = np.zeros(len(length), dtype=np.int64)
    begin[1:] = np.cumsum(length)[:-1]
    end = np.cumsum(length)
    return begin, end


def get_subfind_particle_indices(
    lentype,
    grp_lentype,
    grnr,
    grp_firstsub,
    ptype,
    ids=None,
    part_ids=None,
):
    """Find the snapshot indices of the particles bound to each subhalo.

    A Subfind catalogue orders particles by FOF group, then by subhalo
    within the group (each subhalo ordered by particle type), with the
    unbound particles of the group last.

    If the catalogue stores the member IDs they are matched against the
    snapshot IDs, which does not require the snapshot to be group ordered.
    Otherwise the snapshot is assumed to be group ordered.

    The member IDs are assumed to follow the catalogue order above, with
    each subhalo in type blocks, as in the CAMELS catalogues. IDs need not
    be unique across types (e.g. Simba), so the type of a member cannot be
    recovered from its ID alone; a member that is not a snapshot particle
    of this type raises an error.

    Args:
        lentype (np.ndarray of int):
            Number of particles of each type in each subhalo
            (Subhalo/SubhaloLenType).
        grp_lentype (np.ndarray of int):
            Number of particles of each type in each group
            (Group/GroupLenType).
        grnr (np.ndarray of int):
            Group index of each subhalo (Subhalo/SubhaloGrNr).
        grp_firstsub (np.ndarray of int):
            Index of the first subhalo in each group (Group/GroupFirstSub).
        ptype (int):
            The particle type.
        ids (np.ndarray of int):
            Member IDs of all groups, for all types (IDs/ID), if stored.
        part_ids (np.ndarray of int):
            IDs of the snapshot particles of this type. Required if `ids`
            is given.

    Returns:
        indices (np.ndarray of int):
            Snapshot indices of the bound particles of this type,
            concatenated in subhalo order. Slice with
            `get_begin_end_pointers(lentype[:, ptype])`.
    """
    lentype = np.asarray(lentype, dtype=np.int64)
    grp_lentype = np.asarray(grp_lentype, dtype=np.int64)
    grnr = np.asarray(grnr, dtype=np.int64)
    grp_firstsub = np.asarray(grp_firstsub, dtype=np.int64)

    def offsets(sub, grp):
        # Preceding groups, plus preceding subhaloes in the same group
        c = np.append(0, np.cumsum(sub))
        g = np.append(0, np.cumsum(grp))
        return g[grnr] + c[:-1] - c[grp_firstsub[grnr]]

    n = lentype[:, ptype]
    within = np.arange(n.sum()) - np.repeat(get_begin_end_pointers(n)[0], n)

    if ids is None:
        return np.repeat(offsets(n, grp_lentype[:, ptype]), n) + within

    if part_ids is None:
        raise exceptions.InconsistentArguments(
            "part_ids are required to match the Subfind member IDs."
        )

    start = offsets(lentype.sum(axis=1), grp_lentype.sum(axis=1))
    start += lentype[:, :ptype].sum(axis=1)
    members = np.asarray(ids)[np.repeat(start, n) + within]

    # Match on ID. IDs can repeat (e.g. Simba), so pair the k-th occurrence
    # in the catalogue with the k-th occurrence in the snapshot.
    part_ids = np.asarray(part_ids)
    p_order = np.argsort(part_ids, kind="stable")
    m_order = np.argsort(members, kind="stable")
    p_sorted, m_sorted = part_ids[p_order], members[m_order]
    rank = np.arange(len(m_sorted)) - np.searchsorted(m_sorted, m_sorted)
    loc = np.searchsorted(p_sorted, m_sorted) + rank
    found = loc < len(p_sorted)
    found[found] = p_sorted[loc[found]] == m_sorted[found]
    if not np.all(found):
        raise exceptions.InconsistentArguments(
            f"{np.sum(~found)} Subfind member IDs of type {ptype} are "
            "not in the snapshot."
        )

    indices = np.empty(len(members), dtype=np.int64)
    indices[m_order] = p_order[loc]
    return indices


def age_lookup_table(cosmo, redshift=0.0, delta_a=1e-3, low_lim=1e-4):
    """Create a look-up table for age as a function of scale factor.

    Defaults to start at the lower resolution limit (`delta_a`),
    and proceeds in steps of `delta-a` until the scale factor given
    by the input `redshift` minus the `low_lim`.

    Args:
        cosmo (astropy.cosmology):
            Astropy cosmology object.
        redshift (float):
            Redshift of the snapshot.
        delta_a (int):
            Scale factor resolution to approximate.
        low_lim (float):
            Lower limit of scale factor.

    Returns:
        scale_factor (np.ndarray of float):
            Array of scale factors.
        age (unyt_array of float):
            Array of ages (Gyr).
    """
    # Find the scale factor for the input snapshot
    root_scale_factor = 1.0 / (1.0 + redshift)

    # Find the (integer) resolution of the grid
    resolution = (root_scale_factor - low_lim) / delta_a
    resolution = math.ceil(resolution)

    # Create the (linear) scale factor array
    scale_factor = np.linspace(
        delta_a, root_scale_factor - low_lim, resolution
    )

    # Find the ages at these scale factors
    ages = cosmo.age(1.0 / scale_factor - 1)

    return scale_factor, ages


def split_age_bins(lo, hi, grid_ages):
    """Split top-hat age bins at every grid age they contain.

    A bin narrower than the grid spacing is returned unchanged (one segment
    at its midpoint). Wider bins become one segment per grid age they
    straddle, so a constant SFR across the bin is resolved by the grid.
    A zero-width bin is an instantaneous burst: one segment at its age.

    Args:
        lo (np.ndarray of float):
            Lower (younger) edge of each bin, in years.
        hi (np.ndarray of float):
            Upper (older) edge of each bin, in years.
        grid_ages (np.ndarray of float):
            Ascending grid ages, in years.

    Returns:
        tuple: (bin index, midpoint age, width fraction) per segment.
    """
    bin_index, mid, frac = [], [], []
    for b, (a, c) in enumerate(zip(lo, hi)):
        cuts = np.concatenate(
            ([a], grid_ages[(grid_ages > a) & (grid_ages < c)], [c])
        )
        bin_index.append(np.full(cuts.size - 1, b))
        mid.append(0.5 * (cuts[1:] + cuts[:-1]))
        frac.append(np.diff(cuts) / (c - a) if c > a else [1.0])
    return np.concatenate(bin_index), np.concatenate(mid), np.concatenate(frac)


def bin_overlap_matrix(lo, hi, grid_ages):
    """Fraction of each top-hat age bin falling in each grid age cell.

    Cell edges follow ``parametric.Stars``: zero, the linear midpoints
    between adjacent grid ages, then infinity so mass older than the grid
    is kept in the last cell. A zero-width bin is an instantaneous burst
    placed wholly in the cell containing its age.

    Args:
        lo (np.ndarray of float):
            Lower (younger) edge of each bin, in years.
        hi (np.ndarray of float):
            Upper (older) edge of each bin, in years.
        grid_ages (np.ndarray of float):
            Ascending grid ages, in years.

    Returns:
        np.ndarray of float: (nbin, ngrid) overlap fractions.
    """
    lo, hi = lo[:, None], hi[:, None]
    edges = np.concatenate(
        ([0.0], 0.5 * (grid_ages[:-1] + grid_ages[1:]), [np.inf])
    )
    overlap = np.minimum(hi, edges[1:]) - np.maximum(lo, edges[:-1])
    width = hi - lo
    burst = (edges[:-1] <= lo) & (lo < edges[1:])
    return np.where(
        width > 0,
        np.clip(overlap, 0.0, None) / np.where(width > 0, width, 1.0),
        burst,
    )


def cic_matrix(values, grid):
    """Cloud-in-cell weights placing values on an ascending grid.

    Values beyond the grid are clamped to the end points.

    Args:
        values (np.ndarray of float):
            The values to place (e.g. log10 metallicities).
        grid (np.ndarray of float):
            Ascending grid coordinates.

    Returns:
        np.ndarray of float: (nvalues, ngrid) weights, each row summing to 1.
    """
    x = np.clip(values, grid[0], grid[-1])
    j = np.clip(np.searchsorted(grid, x, side="right") - 1, 0, grid.size - 2)
    frac = (x - grid[j]) / (grid[j + 1] - grid[j])
    weights = np.zeros((x.size, grid.size))
    rows = np.arange(x.size)
    weights[rows, j] = 1.0 - frac
    weights[rows, j + 1] += frac
    return weights


def lookup_age(scale_factor, scale_factors, ages):
    """Look up the age given a scale factor.

    Args:
        scale_factor (np.ndarray of float):
            Scale factors to convert to ages.
        scale_factors (np.ndarray of float):
            Array of lookup scale factors.
        ages (unyt_array of float):
            Array of lookup ages.

    Returns:
        age (float): Age based on the input scale factor/s.
    """
    return np.interp(scale_factor, scale_factors, ages)
