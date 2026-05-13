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

The lower-level ``SpectralCube`` container is still available for custom
workflows that need direct control over the cube object and its generation
inputs.

The documentation below starts with the high-level particle and parametric
galaxy workflows, then shows how the returned cube can be analysed or how a
``SpectralCube`` can be created directly when needed.

.. toctree::
   :maxdepth: 2

   particle_data_cube
   parametric_data_cube
