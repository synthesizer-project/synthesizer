Premade Models
==============

Although it is possible to define your own custom model (see `custom models <../custom_models.ipynb>`_), most use cases will require a common set of emission models with a common set of properties.
To avoid users having to redefine common models every time, we provide some pre--made models which can be used "out of the box", or as a foundation for constructing more complex models.

These pre-made models can be imported directly from the ``emission_models`` submodule, which also defines several lists detailing all available models.


.. code-block:: python

    from synthesizer.emission_models import (
        AGN_MODELS,
        COMMON_MODELS,
        PREMADE_MODELS,
        STELLAR_MODELS,
    )

    print(STELLAR_MODELS)
    print(AGN_MODELS)
    print(COMMON_MODELS)
    print(PREMADE_MODELS)

As you can see in the code above, there are ``STELLAR_MODELS``, ``AGN_MODELS``, and ``COMMON_MODELS``. In the following sections we'll detail each of these.

.. toctree::
   :maxdepth: 1

   stellar_models
   agn_models
   common_models
