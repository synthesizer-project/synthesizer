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
emission they are derived from (float32 photometry produces float32 images,
float32 spectra produce float32 fluxes, data cubes, and spectroscopy),
rather than resolving to the global default. Passing ``out_dtype``
explicitly always wins.

The imaging and data cube backends read particle geometry and signals at
their native precision (per-pixel accumulation still happens in double
internally for accuracy), so a float32 spectra array is smoothed into a
float32 cube without any hidden float64 copy being made:

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

Luminosity units
^^^^^^^^^^^^^^^^

float32 can only represent values up to about 3.4e38, but luminosities in
cgs units exceed that for realistic sources: line luminosities reach
1e40–1e45 erg/s and black hole bolometric luminosities ~1e45 erg/s.
Synthesizer therefore stores luminosities in solar luminosities by default
(the ``luminosity`` unit category is ``Lsun``), which keeps them comfortably
within the float32 range. Spectral densities (e.g. ``lnu`` in erg/s/Hz) fit
in float32 and keep their cgs units.

Grids are converted to these internal units when they are loaded, so this
makes no difference to the results: only the units of the returned
luminosities change, and they can be converted with ``.to("erg/s")`` as
usual.

Note: If your units file predates this change and still uses the old
``erg / s`` default, it is updated automatically (and the change is printed);
any units you customised are left alone. Changes to the default units are
listed in ``units_changelog.yml``. You can switch back to erg/s in the units
file if you prefer, but float32 luminosities may then overflow.

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

Kernels
^^^^^^^

The SPH kernel lookup tables used for line-of-sight column densities (and
smoothed imaging) take a ``dtype`` argument:

.. code-block:: python

    from synthesizer.kernel_functions import Kernel

    kernel = Kernel("sph_anarchy", dtype=np.float32)

The tables are always built in float64 for accuracy and then stored at the
requested dtype, so a float32 kernel halves their memory. The kernel is its own
precision group, so a float64 kernel can still be used with float32 particles
(and vice versa). A saved kernel reloads at the dtype it was saved at, or at
the dtype passed to ``Kernel.load(path, dtype=...)``.

Mixing precisions
~~~~~~~~~~~~~~~~~

Within one logical group of arrays (e.g. the arrays making up a grid, or the
property arrays describing a particle distribution) all floating-point arrays
must share a single dtype — float32 or float64. If they don't, Synthesizer
raises a ``TypeError`` naming the offending array.

*Between* groups, precisions can be mixed freely: float32 particle data can
be combined with a float64 grid (and vice versa), float32 stars can have
line-of-sight column densities computed through float64 gas, and the output
dtype is independent of all of them. Each array is read at its own precision inside the C++
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
