Internal Timers
===============

Synthesizer includes a lightweight timing system that instruments both Python and C++ code paths.
When enabled, it accumulates wall-clock times and call counts for named operations, making it straightforward to identify bottlenecks and verify performance.

This page is aimed at developers who want to add timing to new code or understand how the existing profiling infrastructure works.

How It Works
^^^^^^^^^^^^

The timing system is a simple ``tic``/``toc`` pattern:

1. Call ``tic()`` to record a start time.
2. Perform the work you want to measure.
3. Call ``toc("Operation name", start)`` to record the elapsed time.

When ``ATOMIC_TIMING`` is enabled at compile time, ``toc()`` prints the elapsed time to stdout and accumulates the measurement into a global C++ map keyed by the operation name.
Each entry in the map stores the cumulative time, the number of calls, and whether the timing originated from C++ or Python.
When ``ATOMIC_TIMING`` is **not** enabled, both ``tic()`` and ``toc()`` are complete no-ops with zero runtime cost.

The same ``tic``/``toc`` interface is available in both Python and C++.
C++ extensions call the accumulation function via a cached function pointer (retrieved at module-init time through a PyCapsule), so **no GIL is acquired during timing**.

Enabling Timing
^^^^^^^^^^^^^^^

The timing accumulation is controlled by the ``ATOMIC_TIMING`` compile-time flag.
When this flag is **not** set, ``tic()`` and ``toc()`` compile down to no-ops with zero runtime cost.

To enable timing, install with:

.. code-block:: bash

    ATOMIC_TIMING=1 pip install -e .

You can check at runtime whether timing is active:

.. code-block:: python

    from synthesizer import check_atomic_timing

    if check_atomic_timing():
        print("Timing is enabled")

.. note::

    The profiling scripts under ``profiling/`` require ``ATOMIC_TIMING`` to be enabled.
    Without it, no timing data will be collected and the profiling utilities will have nothing to report.

Adding Timings to Python Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Import ``tic`` and ``toc`` from the timers extension and wrap the code you want to measure:

.. code-block:: python

    from synthesizer.extensions.timers import tic, toc

    def my_expensive_function(data):
        start = tic()

        # ... do the work ...
        result = compute(data)

        toc("My expensive operation", start)
        return result

When ``ATOMIC_TIMING`` is off, both calls are no-ops and the operation name string is never evaluated at the C level, so there is no overhead in production builds.

Adding Timings to C++ Code
^^^^^^^^^^^^^^^^^^^^^^^^^^

In C++ extension source files, include ``timers.h`` and use the same pattern:

.. code-block:: c++

    #include "timers.h"

    void my_function(...) {
        double start = tic();

        // ... do the work ...

        toc("My C++ operation", start);
    }

If your ``.cpp`` file is the **main file of an extension module** (i.e. it contains ``PyInit_*``), you also need to initialise the timing function pointer at module-init time.
Add the following to the includes section, guarded by the preprocessor flag:

.. code-block:: c++

    #include "timers.h"
    #ifdef ATOMIC_TIMING
    #include "timers_init.h"
    #endif

And in the ``PyInit_*`` function, after ``PyModule_Create``:

.. code-block:: c++

    PyMODINIT_FUNC PyInit_my_extension(void) {
        PyObject *m = PyModule_Create(&moduledef);
        if (m == NULL)
            return NULL;

        // ... numpy import, etc. ...

    #ifdef ATOMIC_TIMING
        if (import_toc_capsule() < 0) {
            Py_DECREF(m);
            return NULL;
        }
    #endif

        return m;
    }

.. note::

    ``toc()`` should be called **outside** ``#pragma omp parallel`` regions.
    It is intended to measure the overall wall-clock time of an operation, including its parallel portion.
    The accumulation function is protected by a mutex and atomic operations as a defensive measure, but current call sites are all intended to be single-threaded.

Python-Side API
^^^^^^^^^^^^^^^

The timing data can be accessed through two interfaces.

OperationTimers Class
~~~~~~~~~~~~~~~~~~~~~

For convenience, ``synthesizer.utils.operation_timers.OperationTimers`` provides a dictionary-like wrapper around the low-level functions:

.. code-block:: python

    from synthesizer.utils.operation_timers import OperationTimers

    timers = OperationTimers()
    timers.reset()

    # ... run some synthesizer operations ...

    for name, (cumtime, count, source) in timers.items():
        print(f"{name}: {cumtime:.4f}s ({count} calls, from {source})")

Key methods:

- ``timers.reset()`` — Clear all accumulated data.
- ``timers.keys()`` — List of operation names.
- ``timers[name]`` — Returns ``(cumulative_time, call_count, source)``.
- ``timers.get_source(name)`` — Returns ``"C"`` or ``"Python"``.
- ``timers.items()`` — Yields ``(name, (cumtime, count, source))`` pairs.
- ``len(timers)`` — Number of recorded operations.

The ``source`` field is particularly useful for profiling plots: it distinguishes operations timed in C++ (typically the compute-heavy loops) from those timed in Python (typically higher-level orchestration).

Low-Level C Extension Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These are exposed directly from ``synthesizer.extensions.timers``:

- ``tic()`` — Returns the current wall-clock time as a ``float``.
- ``toc(msg, start_time)`` — Records elapsed time for operation ``msg`` (a string) since ``start_time``.
- ``reset_timings()`` — Clears all accumulated timing data. Call this before each profiling run.
- ``get_operation_names()`` — Returns a ``list`` of all recorded operation names.
- ``get_operation_timings(name)`` — Returns ``(cumulative_time, call_count, source)`` for the given operation. Raises ``KeyError`` if the operation does not exist.
- ``get_operation_source(name)`` — Returns ``"C"`` or ``"Python"`` for the given operation.

