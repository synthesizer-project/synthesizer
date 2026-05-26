Imaging
========

Synthesizer can be used to generate photometric images or property maps.

The main public interface to this functionality is the high-level galaxy and
component imaging API used together with a ``PhotometricImager``. In this
workflow you:

- generate spectra and photometry
- construct a ``PhotometricImager``
- call ``galaxy.get_images_luminosity(...)`` or ``galaxy.get_images_flux(...)`` 
- optionally apply instrument-owned PSF and noise configuration through that
  same workflow

(Or simply call the appropriate property map method on a component or galaxy, 
which will use the same underlying workflow without the emission specific 
steps.)

These methods will return either an ``Image`` or an ``ImageCollection`` depending on the number of bands requested. The ``Image`` and ``ImageCollection`` classes are the main data containers for imaging data in Synthesizer, and they provide a consistent interface for working with images regardless of how they were generated.

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
particles that are representative of a smooth distribution the smoothing of the
particles must be taken into account.

In Synthesizer, this is done by combining a ``Kernel`` with the smoothing
lengths of the particles. This means ``smoothing_lengths`` must be provided for
each particle in the input data.

Generating Images 
-----------------

The pages linked below provide detailed walkthroughs of how to generate images and property maps using Synthesizer, including examples of how to use the various imaging methods and classes.

.. toctree::
   :maxdepth: 2

   particle_imaging
   parametric_imaging
   property_maps
