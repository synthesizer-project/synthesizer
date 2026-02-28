Photometry
==========

With `spectra <../spectra/spectra.rst>`_ in hand we can produce photometry by combining the ``Sed`` object containing the (multidimensional) spectra with a ``FilterCollection`` defining the transmission curve of a set of filters.
Doing so produces a ``PhotometryCollection`` containing the photometry, its units, and methods for manipulating and visualising the photometry.
Once photometry has been generated it can be used to produce images; for more details see the [imaging docs](../imaging/imaging.rst).

.. note::

   For performance, filter-specific integration "artifacts" (i.e. constant values related to the integration of the transmission curve) are cached and reused when repeated photometry is evaluated on the same prepared wavelength/frequency grids.

In the pages below we demonstrate how to produce and use photometry.



.. toctree::
   :maxdepth: 2

   photometry_example
   galaxy_phot
