Imaging
========

Synthesizer can be used to generate photometric images or property maps.

For standard synthetic observations, the main public interface is the
high-level galaxy and component imaging API used together with a
``PhotometricImager``. In this workflow you:

- generate spectra and photometry
- construct a ``PhotometricImager``
- call ``galaxy.get_images_luminosity(...)`` or ``galaxy.get_images_flux(...)``
- optionally apply instrument-owned PSF and noise configuration through that
  same workflow

The lower-level imaging containers are still available for custom workflows:

- ``Image`` for individual images
- ``ImageCollection`` for related multi-band images

Histogram and Smoothed Imaging 
------------------------------

For particle distributions, Synthesizer implements both histogram based
imaging, where pixel values are sorted into individual pixels, and smoothed
imaging, where pixel values are smoothed over kernels to produce a continuous
image. In contrast, for parametric components and galaxies there is only
smoothed imaging, where the pixel values are calculated from the parametric
model and smoothed over a ``Morphology`` object.

Histogram images are simple and fast, only requiring positions and pixel
values as inputs. However, in the vast majority of cases this method is not
suitable for realistic, scientifically accurate imaging. When working with
particles that are representative of a smooth distribution, for example SPH
particles, the smoothing of the particles must be taken into account.

In Synthesizer, this is done by combining a ``Kernel`` with the smoothing
lengths of the particles. This means ``smoothing_lengths`` must be provided for
each particle in the input data.

Generating Images 
-----------------

The documentation below starts with the high-level galaxy workflows for
particle and parametric imaging. ``property_maps`` then demonstrates the more
custom, low-level container-oriented interface.

.. toctree::
   :maxdepth: 2

   particle_imaging
   parametric_imaging
   property_maps
