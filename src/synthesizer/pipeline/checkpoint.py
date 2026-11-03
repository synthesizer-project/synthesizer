"""Helpers for checkpointing a Pipeline run so it can be resumed.

While checkpointing, each rank appends its galaxy outputs to a series of
segment files rather than its output file. A checkpoint closes the open
segment, so a crash can only damage galaxies processed since the last
checkpoint. The checkpoint itself records how many galaxies are done, which
segments hold them, and the running state (e.g. cosmic SED sums) needed to
carry on. It is written atomically, and the previous one is kept as a
fallback.

Once the run is complete the segments are merged into the normal output
file and deleted, along with the checkpoints.
"""

import hashlib
import os
from contextlib import ExitStack

import h5py
import numpy as np
from unyt import unyt_array

# The number of closed segments at which a checkpoint merges them into one
# to keep the number of files bounded
_MAX_SEGMENTS = 8

# The most bytes copied at once when merging segments
_MERGE_BLOCK_NBYTES = 2**26


def segment_path(filepath, index):
    """Return the path of a segment of an output file.

    Args:
        filepath (str): The output file (per rank when using MPI).
        index (int): The index of the segment.

    Returns:
        str: The path of the segment.
    """
    stem, ext = os.path.splitext(filepath)
    return f"{stem}.seg{index}{ext}"


def segment_index(path):
    """Return the index of a segment from its path.

    Args:
        path (str): The path of the segment.

    Returns:
        int: The index of the segment.
    """
    return int(os.path.basename(path).rsplit(".seg", 1)[1].split(".")[0])


def checkpoint_path(filepath):
    """Return the path of the checkpoint of an output file.

    Args:
        filepath (str): The output file (per rank when using MPI).

    Returns:
        str: The path of the checkpoint.
    """
    stem, ext = os.path.splitext(filepath)
    return f"{stem}.ckpt{ext}"


def describe(value):
    """Return a deterministic description of a value for fingerprinting.

    Unlike repr, this never includes memory addresses, so the description
    of the same inputs is the same in every run.

    Args:
        value: The value to describe.

    Returns:
        str/tuple: The description.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)
    if isinstance(value, unyt_array):
        return (describe(value.ndview), str(value.units))
    if isinstance(value, np.ndarray):
        return (
            str(value.dtype),
            value.shape,
            hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        )
    if isinstance(value, np.generic):
        return repr(value.item())
    if isinstance(value, dict):
        return tuple(
            (describe(k), describe(v))
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [describe(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items = sorted(items, key=str)
        return tuple(items)
    if hasattr(value, "label"):
        return (type(value).__name__, describe(value.label))
    if callable(value):
        return getattr(value, "__qualname__", type(value).__name__)
    return type(value).__name__


def fingerprint(value):
    """Return a hash of a value's description.

    Args:
        value: The value to hash.

    Returns:
        str: The hex digest of the hash.
    """
    return hashlib.sha256(str(describe(value)).encode("utf-8")).hexdigest()


def _write_nested(group, data):
    """Write a nested dictionary of arrays and counts to a group.

    Args:
        group (h5py.Group): The group to write into.
        data (dict): The nested dictionary.
    """
    for key, value in data.items():
        if isinstance(value, dict):
            _write_nested(group.create_group(key), value)
        else:
            dset = group.create_dataset(key, data=np.asarray(value))
            if isinstance(value, unyt_array):
                dset.attrs["Units"] = str(value.units)


def read_nested(group):
    """Read a nested dictionary written by _write_nested.

    Args:
        group (h5py.Group): The group to read.

    Returns:
        dict: The nested dictionary.
    """
    data = {}
    for key, value in group.items():
        if isinstance(value, h5py.Group):
            data[key] = read_nested(value)
        elif "Units" in value.attrs:
            data[key] = unyt_array(value[()], value.attrs["Units"])
        else:
            data[key] = value[()].item()
    return data


def write_checkpoint(path, attrs, fingerprints, segments, nested, op_counts):
    """Write a checkpoint atomically, keeping the previous one as path.prev.

    Args:
        path (str): The path of the checkpoint.
        attrs (dict): The scalar state of the run (written as root attrs).
        fingerprints (dict): Hashes of the run's configuration and inputs.
        segments (list of tuple): The (path, number of rows) of each closed
            segment holding the galaxies done so far.
        nested (dict): Nested dictionaries of running state (e.g. cosmic
            SED sums) keyed by the group to write them to.
        op_counts (dict): The number of each operation done so far.
    """
    tmp = f"{path}.tmp"
    with h5py.File(tmp, "w") as hdf:
        hdf.attrs.update(attrs)
        hdf.create_group("Fingerprint").attrs.update(fingerprints)

        # Segments are referenced by name relative to the checkpoint
        seg_group = hdf.create_group("Segments")
        seg_group.create_dataset(
            "Files",
            data=np.array(
                [os.path.basename(seg) for seg, _ in segments], dtype="S"
            ),
        )
        seg_group.create_dataset(
            "Rows", data=np.array([n for _, n in segments], dtype=np.int64)
        )

        for name, data in nested.items():
            _write_nested(hdf.create_group(name), data)
        hdf.create_group("OpCounts").attrs.update(op_counts)

    # Make sure the new checkpoint is on disk before it replaces the old
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    if os.path.exists(path):
        os.replace(path, f"{path}.prev")
    os.replace(tmp, path)


def read_checkpoint(path):
    """Read a checkpoint written by write_checkpoint.

    Args:
        path (str): The path of the checkpoint.

    Returns:
        dict: The root attrs ("attrs"), "fingerprints", "segments" (the
            (path, number of rows) of each), "nested" running state and
            "op_counts".
    """
    with h5py.File(path, "r") as hdf:
        dirname = os.path.dirname(path)
        files = hdf["Segments/Files"][()]
        rows = hdf["Segments/Rows"][()]
        return {
            "attrs": dict(hdf.attrs),
            "fingerprints": dict(hdf["Fingerprint"].attrs),
            "segments": [
                (os.path.join(dirname, name.decode()), int(nrows))
                for name, nrows in zip(files, rows)
            ],
            "nested": {
                key: read_nested(group)
                for key, group in hdf.items()
                if key not in ("Fingerprint", "Segments", "OpCounts")
            },
            "op_counts": dict(hdf["OpCounts"].attrs),
        }


def merge_segments(segments, dest_path):
    """Append the rows of a set of segments to a file, in order.

    A dataset missing from a segment, or holding no rows there (an empty
    placeholder), gets zero rows for that segment's galaxies, as it would
    had the rows been appended to one file.

    Args:
        segments (list of tuple): The (path, number of rows) of each segment.
        dest_path (str): The file to append to (created if missing).
    """
    with ExitStack() as stack:
        srcs = [
            (stack.enter_context(h5py.File(path, "r")), nrows)
            for path, nrows in segments
        ]

        # Find every dataset, taking its shape from a segment with rows
        datasets = {}

        def _gather(name, obj):
            if isinstance(obj, h5py.Dataset) and (
                name not in datasets or datasets[name].shape[0] == 0
            ):
                datasets[name] = obj

        for src, _ in srcs:
            src.visititems(_gather)

        total = sum(nrows for _, nrows in srcs)
        with h5py.File(dest_path, "a") as dest:
            for name, template in datasets.items():
                # The common case: everything is in one segment, so copy the
                # dataset as it is
                if len(srcs) == 1:
                    dest.require_group(os.path.dirname(name) or "/")
                    srcs[0][0].copy(template, dest, name=name)
                    continue

                # A dataset that never got any rows stays an empty
                # placeholder
                trailing = template.shape[1:]
                nrows = total if template.shape[0] > 0 else 0
                dset = dest.create_dataset(
                    name,
                    shape=(nrows, *trailing),
                    dtype=template.dtype,
                    maxshape=(None, *trailing),
                    compression="gzip",
                )
                dset.attrs.update(template.attrs)
                if nrows == 0:
                    continue

                # Copy each segment's rows in blocks to bound the memory
                row_nbytes = max(
                    int(np.prod(trailing)) * template.dtype.itemsize, 1
                )
                block = max(_MERGE_BLOCK_NBYTES // row_nbytes, 1)
                start = 0
                for src, seg_rows in srcs:
                    src_dset = src.get(name)
                    if src_dset is not None and src_dset.shape[0] > 0:
                        if src_dset.shape[0] != seg_rows:
                            raise ValueError(
                                f"Cannot merge {name}: {src.filename} has "
                                f"{src_dset.shape[0]} rows for {seg_rows} "
                                "galaxies"
                            )
                        for i in range(0, seg_rows, block):
                            j = min(i + block, seg_rows)
                            dset[start + i : start + j] = src_dset[i:j]
                    start += seg_rows
