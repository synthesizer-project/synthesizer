Emission models
###############

Overview
--------

To simplify the calculation of complex emission, with many contributing components and different operations involved in their construction, Synthesizer provides ``EmissionModels``.
At their simplest, ``EmissionModels`` define a set of inputs and produce an emission (spectra or emission lines), e.g. the incident emission from a stellar component based on an SPS ``Grid``.
However, ``EmissionModels`` can be arbitrarily complex, defining multiple different types of spectra and lines from different components, and defining how they interact.
The possible operations that ``EmissionModels`` can define are:

- **Extraction** of an emission from a ``Grid`` (see the `Emission model basics <model_usage.ipynb>`_).
- **Generation** of SED or line from stars or AGN (see `Stellar and AGN Models in <premade_models/premade_models.rst>`_), including the addition of dust emission (`dust emission docs <dust_emission.ipynb>`_) or AGN torus emission (`AGN models docs <premade_models/agn_models.ipynb>`_)
- **Transformation** of an emission, e.g. applying a dust curve (see the `dust attenuation docs <attenuation/dust_attenuation.ipynb>`_) or IGM attenuation (see the `IGM attenuation docs <attenuation/igm.ipynb>`_).
- **Combination** of spectra.


Any of these operations can be done in the presence of a property mask, to apply the operation to a subset of the components contents (e.g. applying dust attenuation only to young stars), or a wavelength mask to apply the operation only to a subset of the wavelength range.
These masks can be applied identically to particle or parametric models.

Once an ``EmissionModel`` is constructed it can be used to generate spectra.
This is done by passing the ``EmissionMmodel`` to the ``get_spectra`` or ``get_lines`` method on a ``Galaxy`` or galaxy component.
This will then generate the spectra defined within the ``EmissionModel``, given the properties of the Galaxy or component.
For more details for manipulating these, see `Galaxy spectra <../emissions/spectra/galaxy.ipynb>`_, `Stellar spectra <../emissions/spectra/stars.ipynb>`_ or `Blackhole spectra <../emissions/spectra/blackholes.ipynb>`_.

Working with ``EmissionModels``
-------------------------------

In the sections linked below we detail the basic functionality of an ``EmissionModel``, the premade stellar and black hole emission models, dust emission generators, and how to customise a model or construct your own.

.. toctree::
   :maxdepth: 2

   model_usage
   premade_models/premade_models
   modify_models
   custom_models
   attenuation/attenuation
   dust_emission
   combined_models
   emission_names
