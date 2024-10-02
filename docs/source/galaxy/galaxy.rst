Galaxy
======

One of the core objects in synthesizer is the ``Galaxy`` object.
A ``Galaxy`` is essentially a container object for different `components <../components/components.rst>`_ (stars, gas, and black holes), and provides methods for interacting with and combining these components.
Importantly, this includes methods for computing the emission, from the galaxy as a whole and from the individual components.

Particle vs Parametric
^^^^^^^^^^^^^^^^^^^^^^

As described in the `overview <../getting_started/overview.rst>`_, galaxy objects can take two different forms depending on the data representation: *particle* or *parametric*.
In the Particle vs Parametric docs linked below, we demonstrate how to initialise galaxy objects of these two different fundamental types, and how the galaxy factory function can handle this for you.

Global galaxy properties
^^^^^^^^^^^^^^^^^^^^^^^^

In addition to the component attributes, a galaxy can also hold galaxy level attributes. 
These include a ``name`` for the galaxy, and more importantly the redshift of the galaxy, an attribute required to calculate the observer frame emission of the galaxy.
Beyond the redshift, and like any other container object in synthesizer, the user can provide additional kwargs to the galaxy object, which will be stored as galaxy level attributes.
This enables the storing of arbitrary data needed later in a pipeline (e.g. predefined optical depths).

.. toctree::
   :maxdepth: 2

   particle_parametric
   line_of_sight
