Emission models
###############

Overview
--------

To simplify the calculation of complex emission, with many contributing components and different operations involved in their construction, synthesizer provides ``EmissionModels``.
At their simplest, ``EmissionModels`` define a set of inputs and produce an emission (spectra or emission lines), e.g. the incident emission from a stellar component based on an SPS ``Grid``.
However, ``EmissionModels`` can be arbitrarily complex, defining multiple different types of spectra and lines from different components, and defining how they interact.
The possible operations that ``EmissionModels`` can define are:

- Extraction of an emission from a ``Grid`` (see the `grid docs <../grids/grids_example.ipynb>`_).
- Generation of spectra, i.e. dust emission (see the `dust emission docs <.../dust/dust_emission.ipynb>`_) or AGN torus emission (see the `AGN models docs <agn_models.ipynb>`_).
- Combination of spectra.
- Transformation of an emission, e.g. applying a dust curve (see the `dust attenuation docs <../dust/dust_attenuation.ipynb>`_).

Any of these operations can be done in the presence of a property mask, to apply the operation to a subset of the components contents (e.g. applying dust attenuation only to young stars), or a wavelength mask to apply the operation only to a subset of the wavelength range.
These masks can be applied identically to particle or parametric models.

Once an ``EmissionModel`` is constructed it can be used to generate spectra.
This is done by passing the ``EmissionMmodel`` to the ``get_spectra`` or ``get_lines`` method on a ``Galaxy`` or galaxy component.
This will then generate the spectra defined within the ``EmissionModel``, given the properties of the Galaxy or component.
For more details see `Generating spectra <../spectra/spectra.rst>`_.

Working with ``EmissionModels``
-------------------------------

In the sections linked below we detail the basic functionality of an ``EmissionModel``, the premade stellar and black hole emission models, dust emission generators, and how to customise a model or construct your own.

.. toctree::
   :maxdepth: 2

   model_usage
   premade_models/premade_models
   dust_emission
   modify_models
   custom_models
   attenuation/attenuation
   combined_models
   emission_names
