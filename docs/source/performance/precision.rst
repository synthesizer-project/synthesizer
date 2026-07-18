Controlling Precision
=====================

Synthesizer lets you control the floating-point precision of both its inputs
(particle data and grids) and its outputs (spectra, lines, photometry, etc.).
Running at reduced (float32) precision halves the memory footprint of the
largest arrays in a calculation, which can be the difference between fitting
a run in memory or not.

Two principles govern how precision works throughout the code:

1. **Outputs are controlled by you.** Every function that produces a
   floating-point output takes an ``out_dtype`` argument, and a global
   default controls what happens when you don't pass one.
2. **Inputs are never copied behind the scenes.** Arrays you provide are
   used at the precision (and memory) you provide them at. If a combination
   of precisions can't be used directly, Synthesizer raises an error telling
   you how to fix it rather than silently casting (and therefore copying)
   your data.

Output precision
~~~~~~~~~~~~~~~~

By default all outputs are float64. To change this globally, set the default
output dtype once at the top of your script:

.. code-block:: python

    import numpy as np
    from synthesizer import set_default_out_dtype

    set_default_out_dtype(np.float32)

Every operation that generates output arrays (spectra extraction, photometry,
line luminosities, SFZH grids, LOS optical depths, integration helpers, and
so on) will now allocate float32 results, halving their memory footprint.

Any individual call can override the global default by passing ``out_dtype``
explicitly:

.. code-block:: python

    # Everything else float32, but this spectra extraction in float64
    galaxy.stars.get_spectra(model, out_dtype=np.float64)

The ``Pipeline`` operation methods (``get_spectra``, ``get_photometry_luminosities``,
etc.) also accept ``out_dtype``; the first dtype passed for each operation wins
for the whole run.

Derived products inherit their source dtype
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Products derived from an existing emission — observed spectra and line
fluxes, spectroscopy, images, data cubes, and cosmic SEDs — behave slightly
differently: when ``out_dtype`` is not given they inherit the dtype of the
emission they are derived from (float32 spectra produce float32 fluxes and
images), rather than resolving to the global default. Passing ``out_dtype``
explicitly always wins:

.. code-block:: python

    # Spectra in float64, but store the (much larger) data cubes in float32
    pipeline.get_spectra()
    pipeline.get_data_cubes_lnu(ifu, fov=30 * kpc, out_dtype=np.float32)

Accuracy of reduced precision outputs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Requesting float32 outputs does **not** mean sums are accumulated in float32.
All integrated quantities (integrated spectra, reductions over particles,
grid weight accumulation, and numerical integration) accumulate internally in
double precision and only cast to the requested output dtype at the end, so a
float32 result is the correctly-rounded float32 representation of the float64
answer rather than a value degraded by millions of low-precision additions.

Input precision
~~~~~~~~~~~~~~~

Particle data
^^^^^^^^^^^^^

The particle loaders (``load_data``) accept a ``dtype`` argument so simulation
data can be loaded directly at float32. Arrays you construct yourself are used
as-is: load or build them at the precision you want to pay for.

Grids
^^^^^

Grids can be loaded at reduced precision with ``use_precision``:

.. code-block:: python

    from synthesizer import Grid

    grid = Grid("bc03", use_precision=np.float32)

The conversion happens *during* the HDF5 read, so the float64 version of the
grid is never materialised in memory — peak memory during loading is the
float32 grid itself. An already-loaded grid can also be converted after the
fact with :meth:`~synthesizer.grid.Grid.convert_precision` (this one does
copy, since the float64 data already exists).

Mixing precisions
~~~~~~~~~~~~~~~~~

Within one logical group of arrays (e.g. the arrays making up a grid, or the
property arrays describing a particle distribution) all floating-point arrays
must share a single dtype — float32 or float64. If they don't, Synthesizer
raises a ``TypeError`` naming the offending array.

*Between* groups, precisions can be mixed freely: float32 particle data can
be combined with a float64 grid (and vice versa), and the output dtype is
independent of both. Each array is read at its own precision inside the C++
kernels; nothing is cast or copied.

Understanding the errors
~~~~~~~~~~~~~~~~~~~~~~~~

Because Synthesizer refuses to cast inputs behind your back, you may see
errors like:

.. code-block:: text

    TypeError: ages must share the same floating-point dtype as masses
    (got float64 and float32). Cast the offending array (e.g. with
    arr.astype(np.float32)) or, for grid arrays, load the grid at the
    matching precision with Grid(..., use_precision=...).

This means one array in a group doesn't match its siblings. Fix it at the
source — load the data at a consistent precision, or cast the named array
once yourself — rather than working around it per call.

Similarly, arrays passed to the C++ extensions must be C-contiguous. If you
slice or transpose an array in a way that breaks contiguity you will get a
``ValueError`` asking for a contiguous array; use ``np.ascontiguousarray``
at the point where you create the slice.

Finally, some attenuation models cannot be evaluated at float32 without
overflowing. If that happens you will get an error asking you to use float64
outputs for that operation, rather than spectra silently full of ``inf``.
