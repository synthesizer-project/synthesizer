FAQ
===

A collection of frequently asked questions about Synthesizer.

Why must particle arrays share a floating-point dtype?
-------------------------------------------------------

All floating-point arrays describing one particle collection must use the
same precision, either ``float32`` or ``float64``. This includes masses, ages,
metallicities, coordinates, velocities, smoothing lengths, and other particle
properties. Synthesizer does not silently cast these arrays because doing so
would create potentially large hidden copies. A mismatch therefore raises an
error such as:

.. code-block:: text

    TypeError: ages must share the same floating-point dtype as masses

Choose one dtype when loading or constructing the collection. When attaching
units, use ``unyt_array`` to preserve that dtype; multiplying by some physical
constant-backed units, including ``Msun``, can promote ``float32`` values to
``float64``:

.. code-block:: python

    import numpy as np
    from unyt import unyt_array

    dtype = np.float32
    masses = unyt_array(np.asarray(mass_values, dtype=dtype), "Msun")
    ages = unyt_array(np.asarray(age_values, dtype=dtype), "Myr")
    metallicities = np.asarray(metallicity_values, dtype=dtype)

Particle and grid precision may differ from each other, and ``out_dtype`` only
controls generated outputs; neither fixes inconsistent particle inputs. See
:doc:`Controlling Precision <../performance/precision>` for full details.

Why do I get an SVO inaccessible warning on a HPC but it works locally?
-----------------------------------------------------------------------

Your compute node doesn't have internet access. Use the ``write`` method to
save your instruments or filters locally, then use the ``load`` method to read
them back in on the HPC.
