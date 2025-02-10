Emission models
###############

Overview
--------

To simplify the calculation of complex emission, with many contributing components and different operations involved in their construction, synthesizer provides ``EmissionModels``.
At their simplest, ``EmissionModels`` define a set of inputs and produce an emission (spectra or emission lines), e.g. the incident emission from a stellar component based on an SPS ``Grid``.
Hwoever, ``EmissionModels`` can be arbitrarily complex, defining multiple different types of spectra and lines from different components, and defining how they interact.
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

Named spectra
-------------

Synthesizer enables the generation of many different spectra which are associated with ``Galaxy`` objects or their components through ``EmissionModels``.
These spectra are given standard labels that reflect their origin and the masks that have been applied (though custom labels can be provided).
The flowchart also shows how these different spectra are typically generated and related by an emission model.

.. image:: ../img/synthesizer_flowchart.png
  :alt: Flowchart showing the different emission types in synthesizer
  :target: ../img/synthesizer_flowchart.png

Our standard naming system, which is used in the premade ``EmissionModels``, is listed below.

* ``incident`` spectra are the spectra that serve as an input to the photoionisation modelling. In the context of stellar population synthesis these are the spectra that are produced by these codes and equivalent to the "pure stellar" spectra.

* ``transmitted`` spectra is the incident spectra that is transmitted through the gas in the photoionisation modelling. Functionally the main difference between ``transmitted`` and ``incident`` is that the ``transmitted`` has little flux below the Lyman-limit, since this has been absorbed by the gas. This depends on ``fesc``.

* ``nebular`` is the nebular continuum and line emission predicted by the photoionisation model. This depends on ``fesc``.

* ``reprocessed`` is the emission which has been reprocessed by the gas. This is the sum of ``nebular`` and ``transmitted`` emission. 

* ``escaped`` is the incident emission that escapes reprocessing by gas. This is ``fesc * incident``. This is not subsequently affected by dust.

* ``intrinsic`` is the sum of the ``escaped`` and ``reprocessed`` emission, essentially the emission before dust attenuated.

* ``attenuated`` is the ``reprocessed`` emission with attenuation by dust.

* ``emergent`` is the combined emission including dust attenuation and is the sum of ``reprocessed_attenuated`` and ``escaped``. NOTE: this does not include thermal dust emission, so is only valid from the UV to near-IR.

* ``dust_emission`` is the thermal dust emission calculated using an energy balance approach, and assuming a dust emission model.

* ``total`` is the sum of ``attenuated`` and ``dust``, i.e. it includes both the effect of dust attenuation and dust emission.

* For two component dust models (e.g. Charlot & Fall 2000 or ``BimodalPacmanEmission``) we also generate the individual spectra of the young and old component. This includes ``young_incident``, ``young_nebular``, ``young_attenuated`` etc. ``young`` and ``old`` are equivalent to ``total`` for the young and old components.

All premade models follow these conventions and we encourage the user to employ the same system.

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
   combined_models
