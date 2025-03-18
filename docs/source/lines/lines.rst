Emission Lines
==============

In addition to creating and manipulating spectral energy distributions, Synthesizer can also create collections of emission lines contained within ``LineCollection`` objects.

Like spectral energy distributions, lines can be extracted directly from ``Grid`` objects or generated from a ``Galaxy`` or its components (i.e. ``Stars``, ``Gas``, or ``BlackHoles``).
In the following examples we show how to interact with a ``LineCollection``, how to compute line ratios, and demonstrate how to generate lines from both a ``Grid`` and a ``Galaxy``

Contents
^^^^^^^^

.. toctree::
   :maxdepth: 1

   lines_example
   line_ratios
   grid_lines
   galaxy_lines
