Observables
================ 

Observables are theoretical emissions that have been translated into the observer space accounting for observational effects such as pixel scales, transmission cuves, point spread functions, resolving powers, and noise.

In this section, we will cover the different types of observable that can be generated with Synthesizer, including photometry, spectroscopy, imaging, and spectral data cubes. 

These observables are typically generated through the specialised instrument
classes introduced in the observatories section: ``PhotometricInstrument`` for
photometry, ``PhotometricImager`` for imaging, ``SpectroscopicInstrument`` for
one-dimensional spectroscopy, and ``IntegratedFieldUnit`` for spectral cubes.


.. toctree::
   :maxdepth: 1

   photometry/photometry
   spectroscopy/spectroscopy
   imaging/imaging
   spectral_data_cubes/spectral_data_cubes

   
