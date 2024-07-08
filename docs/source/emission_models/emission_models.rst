Emission models
###############

Overview
--------

To simplify the construction of complex emission (spectra and lines) with many contributing components and different operations involved in their construction Synthesizer uses ``EmissionModels``. These objects define the generation of a single emission in the construction of a more complex whole. The possible operations that an ``EmissionModel`` can define are:

- Extraction of a spectra/line from a ``Grid`` (see the `grid docs <../grids/grids_example.ipynb>`_).
- Generation of spectra, i.e. dust emission (see the `dust emission docs <.../dust/dust_emission.ipynb>`_) or AGN torus emission (see the `AGN models docs <agn_models.ipynb>`_).
- Combination of different emissions.
- Attenuation of an emission with a dust curve (see the `dust attenuation docs <../dust/dust_attenuation.ipynb>`_).

In addition any one of these operations can also be done in the presence of a mask to apply the operation to a subset of the components contents (either particles or a parametric model).

Once an ``EmissionModel`` is constructed it can be passed to any ``get_spectra`` (or ``get_lines``) method on an emitter (a ``Galaxy`` or galaxy component) to get the spectra (lines) defined in the passed ``EmissionModel`` for that emitter. For more details see [generating a parametric spectra](parametric/generate_sed.ipynb) or generating a particle spectra [WIP]().

Named spectra
-------------

Synthesizer enables the generation of many different spectra which are associated with ``Galaxy`` objects or their components. An ``EmissionModel`` is labeled with a label that represents the spectra it creates. Although these labels can be chosen freely, we provide a standard naming system for these different spectra to ensure consistency, which is used in the premade ``EmissionModels`` and can be found listed below.

The flowchart below shows how these different spectra are typically generated and related by an emission model.

.. image:: ../img/synthesizer_flowchart.png
  :alt: Flowchart showing the different emission types in synthesizer
  :target: ../img/synthesizer_flowchart.png


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

In the sections linked below we detail the basic functionality of an ``EmissionModel``, dust emission generators, the premade stellar and black hole emission models, and how to customise a model or construct your own.

.. toctree::
   :maxdepth: 2

   model_usage
   dust_emission
   attenuation/attenuation
   premade_models
   modify_models
   custom_models
   combined_models
