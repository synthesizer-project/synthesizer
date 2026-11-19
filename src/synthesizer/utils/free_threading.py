"""Utilities for running Python level work on threads.

Python level work (e.g. executing emission models or processing galaxies) can
only run in parallel on a free-threaded (no-GIL) build of Python with the GIL
disabled. This module provides the runtime check for that and resolves a
requested number of threads, falling back to serial execution when the GIL is
enabled.

Every extension module declares itself safe to run without the GIL, so
importing synthesizer doesn't re-enable it on a free-threaded build.
"""

import os
import sys
import sysconfig

import numpy as np

from synthesizer import exceptions
from synthesizer.synth_warnings import warn


def gil_enabled():
    """Return whether the GIL is enabled in the running interpreter.

    This is a runtime check rather than a check of how Python was built. A
    free-threaded interpreter re-enables the GIL when it imports an extension
    module that has not declared itself safe to run without it, or when the
    ``PYTHON_GIL=1`` environment variable is set.

    Returns:
        bool:
            ``True`` if the GIL is enabled, otherwise ``False``.
    """
    # Interpreters before 3.13 have no way to disable the GIL at all.
    return getattr(sys, "_is_gil_enabled", lambda: True)()


def resolve_threads(nr_threads, name, nthreads=1):
    """Return the number of threads to run Python level work with.

    Args:
        nr_threads (int):
            The requested number of threads.
        name (str):
            The name of the argument the threads were requested with, used in
            error and warning messages.
        nthreads (int):
            The number of OpenMP threads each thread will use, used to warn
            about oversubscribing the available cores (-1 means one per
            core).

    Returns:
        int:
            The number of threads to use. This is 1 when the GIL is enabled,
            since the Python work cannot then run in parallel.

    Raises:
        exceptions.InconsistentArguments:
            If nr_threads is not a positive integer.
    """
    # Reject anything that isn't a positive integer (bools are ints, but
    # passing True here is almost certainly a mistake).
    if (
        isinstance(nr_threads, bool)
        or not isinstance(nr_threads, (int, np.integer))
        or nr_threads < 1
    ):
        raise exceptions.InconsistentArguments(
            f"{name} must be a positive integer (got {nr_threads!r})."
        )
    nr_threads = int(nr_threads)

    # Nothing to resolve for serial execution.
    if nr_threads == 1:
        return 1

    # With the GIL the threads would contend for it rather than run in
    # parallel, which is slower than running serially.
    if gil_enabled():
        # On a free-threaded build the GIL has been turned back on, either
        # explicitly or by importing an extension module that hasn't declared
        # it can run without it.
        if sysconfig.get_config_var("Py_GIL_DISABLED"):
            reason = (
                "This is a free-threaded build of Python but the GIL has been "
                "re-enabled, usually by importing an extension module that "
                "has not declared it is safe without the GIL (at the time of "
                "writing astropy.table, imported via astropy.cosmology, does "
                "this). Run with PYTHON_GIL=0 (or -Xgil=0) to keep the GIL "
                "disabled."
            )
        else:
            reason = (
                "This requires a free-threaded (no-GIL) build of Python, "
                "e.g. python3.14t."
            )
        warn(
            f"{name}={nr_threads} was requested but the GIL is enabled. "
            f"{reason} Running serially instead."
        )
        return 1

    # Warn if the threads and the OpenMP threads they each spawn together
    # exceed the cores available to this process.
    ncores = getattr(os, "process_cpu_count", os.cpu_count)()
    if ncores is not None:
        omp_threads = ncores if nthreads == -1 else max(nthreads, 1)
        if nr_threads * omp_threads > ncores:
            warn(
                f"{name}={nr_threads} with nthreads={nthreads} uses "
                f"{nr_threads * omp_threads} threads but only {ncores} cores "
                "are available."
            )

    return nr_threads
