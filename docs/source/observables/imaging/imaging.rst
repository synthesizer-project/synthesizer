Imaging
=======

Synthesizer can be used to generate spectral data cubes, photometric images and property maps. The underlying engines for these are 

- `SpectralCube` objects, a container for a spaxel data cube. These provide functionality to create, manipulate and visualise spectral data cubes
- `Image` objects, containers for individual images providing functionality for generating, modifying and plotting individual images
- `ImageCollection` objects, containers for multiple `Images` which collect together and provide interfaces to work with related images.

In the documentation below we demonstrate producing spectral data cubes and photometric images from parametric and particle `Galaxy` objects, and producing property maps from particle distributions.

.. toctree::
   :maxdepth: 2

   particle_imaging
   parametric_imaging
   particle_data_cube
   parametric_data_cube
   property_maps
