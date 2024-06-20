Emission models
###############

Overview
--------

To simplify the construction of complex spectra with many contributing components and different operations involved in their construction Synthesizer uses ``EmissionModels``. These objects define the generation of a single spectra in the construction of a more complex whole. The possible operations that an ``EmissionModel`` can define are:

- Extraction of a spectra from a ``Grid`` (see the ``Grid`` `docs <grids/grids_example.ipynb>`_).
- Generation of spectra, i.e. dust emission (see the dust emission `docs <dust_emission.ipynb>`_) or AGN torus emission (see the AGN models `docs <agn_models.ipynb>`_).
- Combination of different spectra.
- Attenuation of a spectra with a dust curve (see the dust attenuation docs `[WIP] <>`_).

In addition any one of these operations can also be done in the presence of a mask to apply the operation to a subset of the components contents (either particles or a parametric model).

Once an ``EmissionModel`` is constructed it can be passed to any ``get_spectra`` (or ``get_lines``) method on an emitter (a ``Galaxy`` or galaxy component) to get the spectra (lines) defined in the passed ``EmissionModel`` for that emitter. For more details see [generating a parametric spectra](parametric/generate_sed.ipynb) or generating a particle spectra [WIP]().

Dependencies
------------

Each operation a model can define depends on different properties being set.

For extraction operations you need:

- A grid to extract from (``grid``).
- A spectra key to extract (``extract``).
- Optionally, an escape fraction to apply which defaults to 0.0 (``fesc`` or ``covering_fraction`` for AGN).

For combination operations you need:

- The a list of models which will be combined into the resultant spectra (``combine``).

For generation you need:

- A generator class (e.g. a `dust emission model <dust_emission.ipynb>`_) from which to generate spectra (``generator``).

For attenuation operations you need:

- The `dust curve <>`_ to apply (``dust_curve``).
- The model to apply the attenuation to (``apply_dust_to``).
- The optical depth to use with the dust curve (``tau_v``).

As mentioned, masking can be applied alongside any of these operations. Additionally, any number of masks can be combined on the same operation. Each mask is defined by:

- The attribute of the component to mask on (``mask_attr``).
- The threshold of the mask (``mask_thresh``).
- The operator to use when generating the mask, i.e. ``"<"``, ``">"``, ``"<="``, ``">="``, ``"=="``, or ``"!="`` (``mask_op``).


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

* ``dust`` is the thermal dust emission calculated using an energy balance approach, and assuming a dust emission model.

* ``total`` is the sum of ``attenuated`` and ``dust``, i.e. it includes both the effect of dust attenuation and dust emission.

* For two component dust models (e.g. Charlot & Fall 2000 or ``BimodalPacmanEmission``) we also generate the individual spectra of the young and old component. This includes ``young_incident``, ``young_nebular``, ``young_attenuated`` etc. ``young`` and ``old`` are equivalent to ``total`` for the young and old components.

All premade models follow these conventions and we encourage the user to employ the same system.

Working with ``EmissionModels``
-------------------------------

In the sections linked below we detail the premade stellar and black hole emission models and how to customise a model or construct your own.

.. toctree::
   :maxdepth: 1

   stellar_models
   agn_models
   combined_models
   modify_models
   custom_models
