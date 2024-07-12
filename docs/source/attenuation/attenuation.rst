Attenuation
===========

Synthesizer provides a number of different dust attenuation laws that can be used to attenuate the emission from a galaxy or galaxy component. 
These models can be attached to an ``EmissionModel`` (see the `emisison model docs <../emission_models/emission_models.rst>`_) or used directly with an ``Sed`` object (see `the SED docs <../emission_models/emission_models.rst>`_).

In addition to these attenuation laws, we also provide two Intergalactic Medium (IGM) absorption models, which can be used to attenuate the flux from a galaxy or galaxy component.

In the linked sections below we detail the basic functionality and usage of both attenuation laws and IGM absorption models.

.. toctree::
   :maxdepth: 1

   dust_attenuation
   igm