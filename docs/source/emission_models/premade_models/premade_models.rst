Premade Models
==============

Although it is possible to define your own custom model (see `custom models <../custom_models.ipynb>`_), most use cases will require a common set of emission models with a common set of properties.
To avoid users having to redefine commonly used models every time, we provide some pre--made models, which can be used "out of the box", or as a foundation for constructing more complex models.

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

Automatic Emission Model Creation 
---------------------------------

Many premade emission models require that other "child" models exist. These can be passed explicitly by the user, allowing the user to customise their parameters and label. 
However, if they are not passed, Synthesizer will automatically create them for you, based on the input to the parent model. 
When this happens, you'll be presented with a warning message, which will look like this:

.. code-block:: text

    /path/to/script/script.py:11: RuntimeWarning: 
    ReprocessedEmission requires a transmitted model. We'll create one for you
    with the label '_reprocessed_transmitted'. If you want to use a different
    transmitted model, please pass your own to the transmitted argument.
        total = ReprocessedEmission(grid=grid, dust_curve=PowerLaw(slope=-1), nebular=nebular_model)

As you can see from the warning above, when a model is created automagically, it will be given a label preceeded by an underscore, to indicate that it was created automatically.
This label will also include the name of the parent model, as a prefix, so that you can easily identify which parent model created it. 
In this case, a ``ReprocessedEmission`` model was not passed a ``TransmittedEmission`` model, so Synthesizer created one for you with the label ``_reprocessed_transmitted``.