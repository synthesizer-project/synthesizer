Getting Photometry
==================

With `spectra <../spectra/spectra.rst>`_ in hand we can produce photometry by combining the ``Sed`` containing the spectra with a ``FilterCollection`` defining the transmission curve of a set of filters.
Doing so produces a ``PhotometryCollection`` containing the photometry, it's units and methods for manipulating and visuallising the photometry.
In the sections below we demonstrate the production and use of photometry.

.. toctree::
   :maxdepth: 2

   photometry_example
   galaxy_phot