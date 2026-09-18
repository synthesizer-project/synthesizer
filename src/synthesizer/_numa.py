"""NUMA memory placement helpers.

Large arrays here are allocated and filled by a single Python thread (h5py
reading a grid, NumPy allocating an output buffer), so every page of them
lands on the NUMA domain of whichever core happened to be running at the
time. Threads on the other domains then read all of that data remotely, and
the aggregate bandwidth is capped at one memory controller no matter how many
threads are used.

On a COSMA8 node (two EPYC 7H12, eight NUMA domains of sixteen cores) that cap
is around 37 GB/s. Interleaving pages across all domains instead lifts the
streaming kernels to roughly 118 GB/s at 32 threads and 195 GB/s at 64.

Interleaving is not free: below about eight threads everything a thread touches
would otherwise have been local, and spreading it costs 5-20%. It is therefore
opt-in, via the ``SYNTHESIZER_NUMA_INTERLEAVE`` environment variable, and only
worth setting for runs using more cores than one NUMA domain holds.

``numactl --interleave=all <command>`` does the same thing from outside the
process and needs no cooperation from Synthesizer. This module exists for the
cases where the command line is not the caller's to change.
"""

import ctypes
import os

__all__ = ["maybe_interleave_memory"]


def maybe_interleave_memory():
    """Interleave this process's memory across all NUMA domains, if asked.

    Only acts when SYNTHESIZER_NUMA_INTERLEAVE is set to a truthy value. The
    policy applies to allocations made after this call, so it has to run before
    any grid or output array is created, i.e. at import time.

    Returns:
        bool:
            True if the interleave policy was applied, False otherwise.
    """
    flag = os.environ.get("SYNTHESIZER_NUMA_INTERLEAVE", "")
    if flag.lower() in ("", "0", "false", "no"):
        return False

    try:
        libnuma = ctypes.CDLL("libnuma.so.1")
    except OSError:
        # No libnuma (not Linux, or numactl not installed).
        return False

    try:
        if libnuma.numa_available() < 0:
            return False

        # numa_all_nodes_ptr is a struct bitmask * that libnuma fills in its
        # own constructor when the library is loaded.
        all_nodes = ctypes.c_void_p.in_dll(libnuma, "numa_all_nodes_ptr")
        if not all_nodes.value:
            return False

        libnuma.numa_set_interleave_mask.argtypes = [ctypes.c_void_p]
        libnuma.numa_set_interleave_mask.restype = None
        libnuma.numa_set_interleave_mask(all_nodes)
    except (AttributeError, ValueError):
        return False

    return True
