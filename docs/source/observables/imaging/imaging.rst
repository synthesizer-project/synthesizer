Imaging
=======

Synthesizer can be used to generate photometric images or property maps. The underlying engines for these are 

- `Image` objects, containers for individual images providing functionality for generating, modifying and plotting individual images
- `ImageCollection` objects, containers for multiple `Images` which collect together and provide interfaces to work with related images.

Histogram and Smoothed Imaging 
------------------------------

For particle distributions, Synthesizer implements both histogram based imaging, where pixel values are sorted into individual pixels, and smoothed imaging, where pixel values are smoothed over kernels to produce a continuous image.
In contrast, for parametric components and galaxies there is only smoothed imaging, where the pixel values are calculated from the parametric model and smoothed over a ``Morphology`` object.

Histogram images are simple and fast, only requiring positions and pixel values as inputs. 
However, in the vast majority of cases this method is not suitable for realistic, scientifically accurate imaging. 
When working with particles that are representative of a smooth distribution (e.g. SPH particles), the smoothing of the particles must be taken into account. 
In Synthesizer, this is done by combining a ``Kernel`` (effectively a look up table for a spline kernel integrated over the z-axis) with the smoothing lengths of the particles. 
This means ``smoothing_lengths`` must be provided for each particle in the input data, and we need to first extract a ``Kernel`` from the ``kernel_functions`` module. In the examples below demonstrate this in detail.

Generating Images 
-----------------

In the documentation below we demonstrate producing photometric images from parametric and particle `Galaxy` objects, and producing property maps from particle distributions.

.. toctree::
   :maxdepth: 2

   particle_imaging
   parametric_imaging
   property_maps
