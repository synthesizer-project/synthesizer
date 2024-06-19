Galaxy
======

One of the core objects in Synthesizer is the `Galaxy` object. 
A `Galaxy` is essentially a container object for different `components <../components/components.rst>`_, such as its constituent stars, gas, and black holes, while providing methods for interacting with this data.
Importantly, this include methods for predicting the emission, from the galaxy as a whole and from the individual components.

Particle vs Parametric
^^^^^^^^^^^^^^^^^^^^^^

As described in the `overview <../getting_started/overview.rst>`_, galaxy objects can take two different forms depending on the data representation: *particle* or *parametric*.
In the `example page <particle_parametric>`_` below, we demonstrate how to initialise galaxy objects of these two different types, and how the galaxy factory function handles this for you in the majority of cases.

Global galaxy properties
^^^^^^^^^^^^^^^^^^^^^^^^

A galaxy can also have a number of attributes deinfed at the global galaxy level. These include, but are not limited to, the redshift.
Redshift in particular is of value when calculating the the emission of a galaxy in the observer frame.

.. toctree::
   :maxdepth: 1

   particle_parametric
