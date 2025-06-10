Imaging
=======

Synthesizer can be used to photometric images or property maps. The underlying engines for these are 

- `Image` objects, containers for individual images providing functionality for generating, modifying and plotting individual images
- `ImageCollection` objects, containers for multiple `Images` which collect together and provide interfaces to work with related images.

In the documentation below we demonstrate producing photometric images from parametric and particle `Galaxy` objects, and producing property maps from particle distributions.

.. toctree::
   :maxdepth: 2

   particle_imaging
   parametric_imaging
   property_maps
