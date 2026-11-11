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

How the ``IntegratedFieldUnit`` wavelengths are interpreted depends on the
quantity in the cube. For observer frame quantities (``"fnu"``, ``"flux"`` and
``"flam"``) they are observer frame wavelengths, so the fluxes are resampled
directly onto them; these cubes need the observed spectra (e.g. from
``get_observed_spectra``) to have been generated first. For all other
quantities (e.g. ``"lnu"``) they are rest frame wavelengths.

Spectral cubes can be spatially convolved with a PSF through an
``IntegratedFieldUnit``. Both wavelength-independent 2D kernels and
wavelength-dependent 3D kernels are supported, with optional temporary spatial
supersampling during convolution. See the :doc:`PSF-convolution section of the
particle spectral-cube example <particle_data_cube>` for how to use this.

The documentation below starts with the high-level particle and parametric
galaxy workflows, then shows how the returned cube can be analysed or how a
``SpectralCube`` can be created directly when needed.

.. toctree::
   :maxdepth: 2

   particle_data_cube
   parametric_data_cube
