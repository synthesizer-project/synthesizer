Spectral Data Cubes
####################

Overview
--------

Synthesizer can be used to make spectral data cubes, similar to the output of
integral field unit instruments on telescopes.

For standard resolved-spectroscopy workflows, the main public interface is the
high-level galaxy and component cube API used together with an
``IntegratedFieldUnit``. In this workflow you:

- generate spectra
- construct an ``IntegratedFieldUnit``
- call ``galaxy.get_data_cube(...)`` or ``component.get_data_cube(...)``

These methods will return a ``SpectralCube``, which is the main data container for spectral data cubes in Synthesizer, and provides a consistent interface for working with cubes regardless of how they were generated.

The documentation below starts with the high-level particle and parametric
galaxy workflows, then shows how the returned cube can be analysed or how a
``SpectralCube`` can be created directly when needed.

.. toctree::
   :maxdepth: 2

   particle_data_cube
   parametric_data_cube
