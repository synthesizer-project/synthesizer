Imaging
=======

Synthesizer can be used to generate photometric images and property maps. The underlying engine for these are `Image` objects, containers for individual images providing functionality for modifying and plotting individual images, and `ImageCollection` objects, containers for multiple `Images` which collect together and provide interfaces to work with related images.

In the documentation below we demonstrate producing photometric images from parametric and particle `Galaxy` objects and producing property maps from particle distributions.

In the following we show some examples using particles sampled from parametric star formation histories and distributions.

.. toctree::
   :maxdepth: 2

   parametric_imaging
   particle_imaging
   property_maps
